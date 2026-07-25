"""
Broker Research Ingest Pipeline:
Extracts research signals from text using LLM orchestration and evaluates them for auto-follow or escalation.
"""
import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from src.utils import pdf_parser
from src.agent.escalation_manager import EscalationManager

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """
You are an expert financial analyst extraction system.
Extract all target asset recommendations, ratings, price targets, and full detailed investment rationales from the provided broker research text.
For each asset/company found, output a JSON object containing complete details about the recommendation.

CRITICAL INSTRUCTIONS FOR RATIONALE & CONTEXT PRESERVATION:
- Do NOT summarize loosely or omit key analytical details.
- In "rationale", extract the FULL detailed investment reasoning ("reason why") from the report text, including key financial growth drivers, profit margins, loan/revenue growth, dividend yield per share, and strategic catalysts.
- In "risk_factors", extract an array of specific risk factors and headwinds mentioned in the analysis.
- In "document_links", extract any embedded URLs or document links (e.g. audited financials, AGM results, public announcements) associated with this company.

Your output must be a single valid JSON list of objects, where each object matches this schema:
{
  "symbol": "TICKER_SYMBOL",
  "market": "kenyan" or "international",
  "current_price": float_value or null,
  "target_price": float_value or null,
  "upside_pct": float_value or null,
  "recommendation": "BUY" or "SELL" or "HOLD" or "ACCUMULATE" or "REDUCE",
  "rationale": "complete detailed investment reasoning and earnings growth catalysts from report text",
  "risk_factors": ["risk factor 1", "risk factor 2"],
  "document_links": ["https://link1.com", "https://link2.com"],
  "time_horizon": "short_term" or "medium_term" or "long_term",
  "confidence": float_value_between_0_and_1
}

Ensure the output is strictly valid raw JSON without markdown formatting or trailing text.
"""


class BrokerResearchIngest:
    def __init__(self, llm_orchestrator, escalation_manager: EscalationManager, config: Dict[str, Any]):
        self.llm = llm_orchestrator
        self.escalation_manager = escalation_manager
        self.config = config
        
        # Load rules from config or use safe defaults
        ingest_cfg = config.get("research_ingest", {})
        rules_cfg = ingest_cfg.get("auto_follow_rules", {})
        self.require_existing_symbol = rules_cfg.get("require_existing_symbol", True)
        self.allowed_recommendations = [r.upper() for r in rules_cfg.get("allowed_recommendations", ["BUY", "HOLD", "ACCUMULATE"])]
        self.min_upside_pct = rules_cfg.get("min_upside_pct", 5.0)

        # Get list of configured active symbols
        dm_cfg = config.get("data_manager", {})
        self.configured_symbols = set([s.upper() for s in dm_cfg.get("symbols", []) + dm_cfg.get("nse_symbols", [])])

    def process_pdf(self, file_path: str, source: str = 'aib_axys') -> Dict[str, Any]:
        """
        Ingest a PDF research paper, extract signals, and apply follow/escalation logic.
        """
        logger.info(f"Starting ingest of research PDF: {file_path}")
        
        # Record upload in DB
        upload_id = self.escalation_manager.record_upload(os.path.basename(file_path), source)
        
        try:
            # 1. Parse PDF with full table & page structure
            extracted = pdf_parser.extract_all(file_path)
            text = extracted.get("text", "")
            
            if not text:
                raise ValueError("No text could be extracted from the PDF.")
                
            # 2. Extract signals via LLM across all page chunks
            signals = self._extract_signals_via_llm(extracted)
            logger.info(f"Extracted {len(signals)} total signals from research PDF across all pages.")
            
            processed_count = 0
            auto_followed = []
            escalated = []
            
            # 3. Process each signal
            for signal in signals:
                symbol = signal.get("symbol", "").upper()
                if not symbol:
                    continue
                
                # Record signal in DB
                signal_id = self.escalation_manager.record_signal(upload_id, signal)
                signal["id"] = signal_id
                
                # 4. Evaluate signal against auto-follow/escalation rules
                action, reason, risk_level = self._evaluate_signal(signal)
                
                if action == "auto_follow":
                    # Add to watchlist database directly
                    self.escalation_manager.add_to_watchlist(
                        symbol=symbol,
                        market=signal.get("market", "kenyan"),
                        source=f"upload_{upload_id}",
                        recommendation=signal.get("recommendation", "HOLD"),
                        target_price=signal.get("target_price", 0.0),
                        rationale=signal.get("rationale", "")
                    )
                    auto_followed.append(symbol)
                    logger.info(f"Auto-followed symbol: {symbol}. Reason: {reason}")
                else:
                    # Create escalation in DB for operator approval
                    self.escalation_manager.create_escalation(
                        signal_id=signal_id,
                        symbol=symbol,
                        action="follow",
                        reason=reason,
                        risk_level=risk_level
                    )
                    escalated.append((symbol, reason))
                    logger.info(f"Escalated symbol {symbol} for operator approval. Reason: {reason}")
                    
                processed_count += 1
                
            # Update upload record status
            self.escalation_manager.update_upload_status(upload_id, "completed", processed_count)
            
            return {
                "upload_id": upload_id,
                "status": "completed",
                "signals_processed": processed_count,
                "auto_followed": auto_followed,
                "escalated": escalated
            }
            
        except Exception as e:
            logger.error(f"Error processing research PDF {file_path}: {e}")
            self.escalation_manager.update_upload_status(upload_id, "failed", 0)
            return {
                "upload_id": upload_id,
                "status": "failed",
                "error": str(e)
            }

    def _extract_signals_fallback(self, text: str) -> List[Dict[str, Any]]:
        """Fallback heuristic matcher to extract signals across all known company names and symbols."""
        logger.info("Running enhanced heuristic signal extractor across all NSE company profiles...")
        
        COMPANY_MAP = {
            "ABSA": ["ABSA", "ABSA BANK"],
            "COOP": ["COOP", "CO-OPERATIVE BANK", "COOPERATIVE BANK"],
            "DTB": ["DTB", "DIAMOND TRUST", "DTB-K"],
            "EABL": ["EABL", "EAST AFRICAN BREWERIES"],
            "PORTLAND": ["PORTLAND", "E.A PORTLAND", "EAST AFRICAN PORTLAND"],
            "EQTY": ["EQTY", "EQUITY GROUP", "EQUITY BANK"],
            "I&M": ["I&M", "I&M GROUP", "I&M BANK"],
            "KAPCHORUA": ["KAPCHORUA", "KAPCHORUA TEA"],
            "KCB": ["KCB", "KCB GROUP", "KCB BANK"],
            "KPLC": ["KPLC", "KENYA POWER"],
            "NCBA": ["NCBA", "NCBA GROUP"],
            "SCOM": ["SCOM", "SAFARICOM"],
            "SBIC": ["STANBIC", "STANBIC BANK"],
            "SCBK": ["STANDARD CHARTERED", "STANCHART"],
            "WILLIAMSON": ["WILLIAMSON", "WILLIAMSON TEA"],
            "LIBERTY": ["LIBERTY", "LIBERTY KENYA"],
            "CIC": ["CIC", "CIC INSURANCE"],
            "JUBILEE": ["JUBILEE", "JUBILEE HOLDINGS", "JUBILEE INSURANCE"],
            "KEGN": ["KENGEN", "KENYA ELECTRICITY"],
            "SASINI": ["SASINI", "SASINI PLC"],
            "BOC": ["BOC", "B.O.C KENYA"],
            "BAT": ["BAT", "BAT KENYA", "BRITISH AMERICAN TOBACCO"],
            "BRIT": ["BRITAM", "BRITAM HOLDINGS"]
        }
        
        extracted = []
        import re
        seen_symbols = set()

        for canonical_sym, aliases in COMPANY_MAP.items():
            for alias in aliases:
                matches = list(re.finditer(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE))
                if not matches or canonical_sym in seen_symbols:
                    continue
                    
                for match in matches:
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].replace('\n', ' ').strip()
                    context_lower = context.lower()

                    sentiment_map = {
                        "buy": "BUY", "accumulate": "BUY", "overweight": "BUY", "outperform": "BUY",
                        "sell": "SELL", "reduce": "SELL", "underweight": "SELL", "underperform": "SELL",
                        "hold": "HOLD", "neutral": "HOLD", "maintain": "HOLD"
                    }
                    recommendation = "HOLD"
                    min_dist = float('inf')
                    sym_pos = context_lower.find(alias.lower())
                    if sym_pos != -1:
                        for kw, category in sentiment_map.items():
                            for kw_match in re.finditer(rf"\b{kw}\b", context_lower):
                                dist = abs(kw_match.start() - sym_pos)
                                if dist < min_dist:
                                    min_dist = dist
                                    recommendation = category

                    # Target price / Current price parsing - exclude 4-digit years like 2024-2027 and percentage values
                    price_matches = re.findall(r"(?:price|target|kes|closing|@|\$)\s*:?\s*(\d+(?:\.\d+)?)", context, re.IGNORECASE)
                    if not price_matches:
                        raw_nums = re.findall(r"\b(?:\d{1,3}(?:\.\d+)?|\d+\.\d+)\b", context)
                        price_matches = [n for n in raw_nums if not (len(n) == 4 and n.startswith("202"))]

                    current_price = None
                    target_price = None
                    if len(price_matches) >= 2:
                        try:
                            p1 = float(price_matches[0])
                            p2 = float(price_matches[1])
                            if recommendation == "BUY":
                                current_price = min(p1, p2)
                                target_price = max(p1, p2)
                            else:
                                current_price = p1
                                target_price = p2
                        except ValueError:
                            pass
                    elif len(price_matches) == 1:
                        try:
                            current_price = float(price_matches[0])
                        except ValueError:
                            pass

                    # Extract any URLs near this context
                    doc_links = re.findall(r"https?://[^\s\)\>]+", context)

                    # Clean rationale text
                    clean_rationale = re.sub(r"https?://[^\s\)\>]+", "", context).strip()

                    extracted.append({
                        "symbol": canonical_sym,
                        "market": "kenyan",
                        "current_price": current_price,
                        "target_price": target_price,
                        "upside_pct": round(((target_price - current_price) / current_price * 100), 2) if (target_price and current_price and current_price > 0) else None,
                        "recommendation": recommendation,
                        "rationale": f"[Extracted Analyst Rationale]: {clean_rationale[:350]}",
                        "document_links": doc_links
                    })
                    seen_symbols.add(canonical_sym)
                    break
            
        logger.info(f"Enhanced heuristic extractor recovered {len(extracted)} distinct recommendations.")
        return extracted

    def _extract_signals_via_llm(self, input_data: Any) -> List[Dict[str, Any]]:
        """Extract structured signal JSON by processing PDF content in page chunks."""
        if not self.llm or not getattr(self.llm, "enabled", False):
            logger.warning("LLM Orchestrator is not active. Using fallback extractor.")
            text = input_data.get("text", "") if isinstance(input_data, dict) else str(input_data)
            return self._extract_signals_fallback(text)

        pdf_extractor_model = self.config.get("swarm", {}).get("agents", {}).get("pdf_extractor", "anthropic/claude-sonnet-4.6")
        all_signals = []
        seen_symbols = set()

        if isinstance(input_data, dict) and input_data.get("pages"):
            pages = input_data["pages"]
            # Process in 2-page chunks so output token limits are never hit
            chunk_size = 2
            for i in range(0, len(pages), chunk_size):
                chunk_pages = pages[i:i+chunk_size]
                chunk_text = "\n\n".join([p["combined"] for p in chunk_pages])
                
                proposal = self.llm.propose_json(_EXTRACTION_SYSTEM_PROMPT, chunk_text, model_override=pdf_extractor_model)
                results = []
                if isinstance(proposal, list):
                    results = proposal
                elif isinstance(proposal, dict) and "recommendations" in proposal:
                    results = proposal["recommendations"]
                elif isinstance(proposal, dict):
                    results = [proposal]
                
                for item in results:
                    sym = item.get("symbol", "").upper()
                    if sym and sym not in seen_symbols:
                        seen_symbols.add(sym)
                        all_signals.append(item)
        else:
            text = input_data.get("text", "") if isinstance(input_data, dict) else str(input_data)
            proposal = self.llm.propose_json(_EXTRACTION_SYSTEM_PROMPT, text, model_override=pdf_extractor_model)
            if isinstance(proposal, list):
                all_signals = proposal
            elif isinstance(proposal, dict) and "recommendations" in proposal:
                all_signals = proposal["recommendations"]
            elif isinstance(proposal, dict):
                all_signals = [proposal]

        if not all_signals:
            logger.warning("LLM returned no recommendation items. Triggering enhanced fallback.")
            text = input_data.get("text", "") if isinstance(input_data, dict) else str(input_data)
            return self._extract_signals_fallback(text)
            
        return all_signals

    def _evaluate_signal(self, signal: Dict[str, Any]) -> Tuple[str, str, str]:
        """
        Evaluate if a signal is safe to follow automatically or needs escalation.
        Returns: (action, reason, risk_level)
            action: 'auto_follow' | 'escalate'
            reason: str text explaining decision
            risk_level: 'low' | 'medium' | 'high'
        """
        symbol = signal.get("symbol", "").upper()
        recommendation = signal.get("recommendation", "HOLD").upper()
        
        # Calculate upside percentage if current/target price are available
        current_price = signal.get("current_price")
        target_price = signal.get("target_price")
        upside_pct = signal.get("upside_pct")
        
        if upside_pct is None and current_price and target_price and current_price > 0:
            upside_pct = ((target_price - current_price) / current_price) * 100
            signal["upside_pct"] = upside_pct

        # Check 1: Symbol config validation
        is_known = symbol in self.configured_symbols
        if self.require_existing_symbol and not is_known:
            return "escalate", f"Symbol '{symbol}' is not in active trading configuration.", "high"

        # Check 2: Recommendation validation
        if recommendation not in self.allowed_recommendations:
            return "escalate", f"Recommendation '{recommendation}' is not in pre-approved list ({self.allowed_recommendations}).", "medium"

        # Check 3: Upside validation
        if recommendation in ("BUY", "ACCUMULATE"):
            val_upside = upside_pct or 0.0
            if val_upside < self.min_upside_pct:
                return "escalate", f"Extracted upside ({val_upside:.1f}%) is below minimum target threshold ({self.min_upside_pct}%).", "medium"
                
        # Check 4: Sentiment confidence validation
        confidence = signal.get("confidence", 1.0)
        if confidence < 0.6:
            return "escalate", f"LLM extraction confidence ({confidence:.2f}) is too low (requires >= 0.6).", "low"

        # Multi-Model Consensus Voting for high-stakes BUY signals
        rec_upper = signal.get('recommendation', '').upper()
        if rec_upper in ('BUY', 'ACCUMULATE') and getattr(self, 'llm', None):
            try:
                consensus_prompt = (
                    f"You are an independent financial analyst. A broker report recommends "
                    f"{rec_upper} on {signal.get('symbol')} at {signal.get('current_price')} "
                    f"with target {signal.get('target_price')}. "
                    f"Reason: {signal.get('reason', 'N/A')}. "
                    f"Do you agree? Reply with JSON: {{\"agree\": true/false, \"recommendation\": \"BUY\"/\"HOLD\"/\"SELL\", \"reasoning\": \"...\"}}"
                )
                secondary_opinion = self.llm.propose_json_secondary(
                    "You are a risk-aware equity analyst providing independent signal verification.",
                    consensus_prompt
                )
                if secondary_opinion and isinstance(secondary_opinion, dict):
                    sec_rec = secondary_opinion.get('recommendation', '').upper()
                    agrees = secondary_opinion.get('agree', False)
                    signal['consensus_verified'] = agrees and sec_rec in ('BUY', 'ACCUMULATE')
                    signal['secondary_recommendation'] = sec_rec
                    signal['secondary_reasoning'] = secondary_opinion.get('reasoning', '')
                    
                    if signal['consensus_verified']:
                        signal['confidence'] = min(0.95, signal.get('confidence', 0.7) + 0.15)
                        logger.info(f"✅ Multi-model consensus VERIFIED for {signal.get('symbol')}: "
                                   f"Primary={rec_upper}, Secondary={sec_rec}")
                        return ("auto_follow", "Dual-model consensus verified. Both LLMs agree on recommendation.", "low")
                    else:
                        logger.warning(f"⚠️ Multi-model consensus DISAGREEMENT for {signal.get('symbol')}: "
                                      f"Primary={rec_upper}, Secondary={sec_rec}")
                        return ("escalate", f"Multi-model disagreement: Primary says {rec_upper} but secondary says {sec_rec}. Requires operator review.", "medium")
                else:
                    signal['consensus_verified'] = None
                    logger.info(f"Multi-model consensus unavailable for {signal.get('symbol')}, proceeding with single-model auto-follow.")
            except Exception as e:
                logger.warning(f"Consensus voting error for {signal.get('symbol')}: {e}")
                signal['consensus_verified'] = None
        
        return ("auto_follow", "Asset matches all auto-follow criteria.", "low")
