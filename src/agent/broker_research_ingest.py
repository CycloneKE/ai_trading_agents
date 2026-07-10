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
Extract all target asset recommendations, ratings, and price targets from the provided broker research text.
For each asset/company found, output a JSON object containing details about the recommendation.
Extract the symbol (e.g. SCOM, EQTY, RIVN, AAPL, BABA), market type, current price, target price, recommendations, and rationale.

Your output must be a single valid JSON list of objects, where each object matches this schema:
{
  "symbol": "TICKER_SYMBOL",
  "market": "kenyan" or "international",
  "current_price": float_value or null,
  "target_price": float_value or null,
  "upside_pct": float_value or null,
  "recommendation": "BUY" or "SELL" or "HOLD" or "ACCUMULATE" or "REDUCE",
  "rationale": "detailed reason or investment rationale from text",
  "risk_factors": ["risk factor 1", "risk factor 2"],
  "time_horizon": "short_term" or "medium_term" or "long_term",
  "confidence": float_value_between_0_and_1
}

Ensure the output is strictly valid raw JSON. Do not wrap in markdown blocks, do not add trailing text or comments.
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
            # 1. Parse PDF
            extracted = pdf_parser.extract_all(file_path)
            text = extracted.get("text", "")
            
            if not text:
                raise ValueError("No text could be extracted from the PDF.")
                
            # 2. Extract signals via LLM
            signals = self._extract_signals_via_llm(text)
            logger.info(f"Extracted {len(signals)} signals from research PDF.")
            
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
        """Fallback heuristic regex matcher to extract signals when LLMs fail or are rate-limited."""
        logger.info("Running regex-based fallback signal extractor...")
        symbols = ["SCOM", "EQTY", "KCB", "COOP", "SCBK", "SBIC", "ABSA", "BAT", "EABL", "KEGN", 
                   "KNRE", "BAMB", "TOTL", "CTUM", "NMG", "NCBA", "BRIT", "CIC"]
        
        extracted = []
        import re
        
        for sym in symbols:
            # Find symbol as a separate word
            matches = list(re.finditer(rf"\b{sym}\b", text, re.IGNORECASE))
            if not matches:
                continue
                
            # If found, extract surrounding context (20 characters before and 120 after to prevent overlap)
            first_match = matches[0]
            start = max(0, first_match.start() - 20)
            end = min(len(text), first_match.end() + 120)
            context = text[start:end].replace('\n', ' ').strip()
            
            # Determine recommendation based on closest keyword to the symbol match to prevent cross-talk
            context_lower = context.lower()
            sentiment_map = {
                "buy": "buy", "accumulate": "buy", "overweight": "buy", "outperform": "buy",
                "sell": "sell", "reduce": "sell", "underweight": "sell", "underperform": "sell",
                "hold": "hold", "neutral": "hold", "maintain": "hold"
            }
            recommendation = "hold"
            min_dist = float('inf')
            sym_pos_in_context = context_lower.find(sym.lower())
            if sym_pos_in_context != -1:
                for kw, category in sentiment_map.items():
                    for kw_match in re.finditer(rf"\b{kw}\b", context_lower):
                        dist = abs(kw_match.start() - sym_pos_in_context)
                        if dist < min_dist:
                            min_dist = dist
                            recommendation = category
                
            # Try to find a target price or current price (numbers)
            numbers = re.findall(r"\b\d+(?:\.\d+)?\b", context)
            current_price = None
            target_price = None
            if len(numbers) >= 2:
                try:
                    num1 = float(numbers[0])
                    num2 = float(numbers[1])
                    if recommendation == "buy":
                        current_price = min(num1, num2)
                        target_price = max(num1, num2)
                    else:
                        current_price = num1
                        target_price = num2
                except ValueError:
                    pass
            
            extracted.append({
                "symbol": sym,
                "market": "kenyan",
                "current_price": current_price,
                "target_price": target_price,
                "upside_pct": round(((target_price - current_price) / current_price * 100), 2) if (target_price and current_price) else None,
                "recommendation": recommendation,
                "rationale": f"[Heuristic Extraction] Found symbol {sym} in context: '... {context[:120]} ...'"
            })
            
        logger.info(f"Fallback regex extractor recovered {len(extracted)} potential recommendations.")
        return extracted

    def _extract_signals_via_llm(self, text: str) -> List[Dict[str, Any]]:
        """Extract structured signal JSON from unstructured text via LLM."""
        if not self.llm or not getattr(self.llm, "enabled", False):
            logger.warning("LLM Orchestrator is not active. Cannot extract research signals.")
            return []
            
        # Call LLM to propose parsed JSON with the dynamic pdf_extractor model from config
        pdf_extractor_model = self.config.get("swarm", {}).get("agents", {}).get("pdf_extractor", "anthropic/claude-sonnet-4.6")
        proposal = self.llm.propose_json(_EXTRACTION_SYSTEM_PROMPT, text, model_override=pdf_extractor_model)

        # Parse output
        results = []
        if not proposal:
            logger.warning("LLM proposal returned empty or invalid JSON. Triggering heuristic fallback.")
            return self._extract_signals_fallback(text)
            
        if isinstance(proposal, list):
            results = proposal
        elif isinstance(proposal, dict) and "recommendations" in proposal:
            results = proposal["recommendations"]
        elif isinstance(proposal, dict):
            # Sometimes LLM outputs dictionary instead of list
            results = [proposal]
            
        if not results:
            logger.warning("LLM returned no recommendation items. Triggering heuristic fallback.")
            return self._extract_signals_fallback(text)
            
        return results

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

        # If it passes all checks, it's safe to follow automatically
        return "auto_follow", f"Asset matches all auto-follow criteria.", "low"
