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
            # 0. AIB-AXYS's daily Market Pulse is market data, not a note
            #    with ratings: read its tables exactly (market_pulse.py)
            #    rather than asking for recommendations it does not contain.
            from src.agent import market_pulse
            texts = market_pulse.page_texts(file_path)
            if market_pulse.is_market_pulse(texts):
                summary = market_pulse.ingest(file_path, texts)
                self.escalation_manager.update_upload_status(
                    upload_id, "completed", summary['stocks_mapped'])
                return self._summarised(upload_id, {
                    "status": "completed", "signals_processed": 0, "auto_followed": [],
                    "escalated": [], **summary})

            # 1. Parse PDF with full table & page structure
            extracted = pdf_parser.extract_all(file_path)
            text = extracted.get("text", "")
            
            if not text:
                raise ValueError("No text could be extracted from the PDF.")
                
            # 2. Extract signals via LLM across all page chunks
            signals = self._extract_signals_via_llm(extracted)
            logger.info(f"Extracted {len(signals)} total signals from research PDF across all pages.")
            return self._summarised(upload_id, {"document_type": "analyst_note",
                                                **self._handle_signals(upload_id, signals)})

        except Exception as e:
            logger.error(f"Error processing research PDF {file_path}: {e}")
            self.escalation_manager.update_upload_status(upload_id, "failed", 0)
            return self._summarised(upload_id, {"status": "failed", "error": str(e)})

    def process_image(self, file_path: str, source: str = 'aib_axys') -> Dict[str, Any]:
        """A recommendation sheet sent as a picture (AIB-AXYS's Daily
        Whispers): read by an AI that can see images, then every row checked
        by rules before it counts (daily_whispers.py)."""
        from src.agent import daily_whispers
        upload_id = self.escalation_manager.record_upload(os.path.basename(file_path), source)
        try:
            sheet = daily_whispers.read(file_path, self.llm)
            daily_whispers.remember(sheet['accepted'], os.path.basename(file_path))
            signals = [{'symbol': r['symbol'], 'market': 'kenyan',
                        'current_price': r['current_price'], 'target_price': r['target_price'],
                        'upside_pct': r['upside_pct'], 'recommendation': r['recommendation'],
                        'rationale': r['rationale'], 'risk_factors': [],
                        'time_horizon': 'medium_term', 'confidence': 1.0}
                       for r in sheet['accepted']]
            return self._summarised(upload_id, {**sheet, **self._handle_signals(upload_id, signals)})
        except Exception as e:
            logger.error(f"Error processing research image {file_path}: {e}")
            self.escalation_manager.update_upload_status(upload_id, "failed", 0)
            return self._summarised(upload_id, {"document_type": "recommendation_sheet",
                                                "status": "failed", "error": str(e)})

    def _summarised(self, upload_id: int, result: Dict[str, Any]) -> Dict[str, Any]:
        """The result with `actions`, what the agent did in plain words, kept
        with the upload so the Research page can show it later too."""
        actions = describe(result)
        result = {"upload_id": upload_id, **result, "actions": actions}
        try:
            self.escalation_manager.set_upload_summary(
                upload_id, result.get("document_type") or "unknown", actions)
        except Exception as e:
            logger.warning(f"Could not store the upload summary: {e}")
        return result

    def _handle_signals(self, upload_id: int, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Record each signal, then follow it or ask the operator."""
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
            "status": "completed",
            "signals_processed": processed_count,
            "auto_followed": auto_followed,
            "escalated": escalated
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
        alias_owner = {a.upper(): sym for sym, aliases in COMPANY_MAP.items() for a in aliases}
        other_names = re.compile(r"\b(" + "|".join(re.escape(a) for a in sorted(alias_owner, key=len, reverse=True))
                                 + r")\b", re.IGNORECASE)
        owner_of = lambda name: alias_owner.get(name.upper())

        for canonical_sym, aliases in COMPANY_MAP.items():
            for alias in aliases:
                matches = list(re.finditer(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE))
                if not matches or canonical_sym in seen_symbols:
                    continue
                    
                for match in matches:
                    # From the start of the company's sentence (or the last
                    # other company named in it) to the next company named,
                    # at most ~160 characters on: a rating that follows
                    # another company's name belongs to that company.
                    bounds = [m.end() for m in re.finditer(r"[.!?](?=\s+[A-Z])|\n", text[:match.start()])]
                    start = bounds[-1] if bounds else 0
                    others_before = [m.end() for m in other_names.finditer(text, start, match.start())
                                     if owner_of(m.group(0)) != canonical_sym]
                    start = max([start] + others_before)
                    end = min(len(text), match.end() + 160)
                    nxt = next((m.start() for m in other_names.finditer(text, match.end(), end)
                                if owner_of(m.group(0)) != canonical_sym), None)
                    end = nxt if nxt is not None else end
                    context = text[start:end].replace('\n', ' ').strip()
                    context_lower = context.lower()

                    sentiment_map = {
                        "buy": "BUY", "accumulate": "BUY", "overweight": "BUY", "outperform": "BUY",
                        "sell": "SELL", "reduce": "SELL", "underweight": "SELL", "underperform": "SELL",
                        "hold": "HOLD", "neutral": "HOLD", "maintain": "HOLD"
                    }
                    recommendation = None
                    min_dist = float('inf')
                    sym_pos = context_lower.find(alias.lower())
                    if sym_pos != -1:
                        for kw, category in sentiment_map.items():
                            for kw_match in re.finditer(rf"\b{kw}\b", context_lower):
                                dist = abs(kw_match.start() - sym_pos)
                                if dist < min_dist:
                                    min_dist = dist
                                    recommendation = category
                    # A company named without a rating beside it is market
                    # commentary, not a recommendation. This used to default
                    # to HOLD, so a market report produced a "HOLD" for every
                    # company it mentioned, with prices read from percentages.
                    if recommendation is None:
                        continue

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


def _pct(v) -> str:
    return f"{v * 100:.2f}%"


def describe(result: Dict[str, Any]) -> List[str]:
    """What the agent did with an upload, one plain sentence per line."""
    kind = result.get("document_type")
    if result.get("status") != "completed":
        return [f"Nothing was changed: {result.get('error') or 'the document could not be read'}."]
    followed = result.get("auto_followed") or []
    queued = [e[0] if isinstance(e, (list, tuple)) else e for e in result.get("escalated") or []]
    follow_lines = []
    if followed:
        follow_lines.append(f"Added to the research watchlist (stocks the agent trades): {', '.join(followed)}.")
    if queued:
        follow_lines.append(f"Sent to the approval queue for your decision, because the agent does not "
                            f"trade them yet or the rating needs a look: {', '.join(queued)}.")
    if kind == "market_pulse":
        lines = [f"Read as the AIB-AXYS Market Pulse of {result.get('as_of')}."]
        lines.append(f"Stored price, earnings, dividend, P/E and yield for {result.get('stocks_mapped', 0)} of "
                     f"{result.get('stocks_read', 0)} stocks. The dividend sleeve ranks from these, the "
                     f"Market Scan shows them and the AI sees them when it reviews a trade.")
        added, kept = result.get("price_bars_added", 0), result.get("price_bars_already_recorded", 0)
        lines.append(f"Added {added} closing prices to the price history"
                     + (f"; {kept} days were already recorded from the live feed and were left as they were."
                        if kept else "."))
        rates = result.get("rates") or {}
        if rates.get("tbill_91"):
            lines.append(f"91-day T-bill rate {_pct(rates['tbill_91'])}: now used as the NSE benchmark, for "
                         f"interest on idle paper cash and for the dividend sleeve's T-bill test.")
        news = result.get("announcements") or []
        if news:
            lines.append(f"Kept {len(news)} company announcements; the AI sees each company's in its trade "
                         f"reviews for 30 days.")
        if result.get("unmapped"):
            lines.append(f"Not yet matched to a stock code: {', '.join(result['unmapped'])}. They are matched "
                         f"automatically once the live feed has recorded them.")
        lines.append("This report has no buy or sell ratings, so nothing was sent to the approval queue.")
        return lines
    if kind == "recommendation_sheet":
        accepted, rejected = result.get("accepted") or [], result.get("rejected") or []
        lines = [f"Read as an AIB-AXYS rating sheet{' (' + result['title'] + ')' if result.get('title') else ''} "
                 f"of {result.get('as_of')}, from the picture, and checked row by row."]
        if not result.get("date_read", True):
            lines.append("The report date could not be read, so today's date was used.")
        if accepted:
            lines.append("Accepted: " + "; ".join(
                f"{r['symbol']} {r['recommendation']} at {r['current_price']}, target {r['target_price']} "
                f"({r['upside_pct']:+.1f}%)" for r in accepted) + ".")
            lines.append("For the next 30 days the AI sees each rating, target and rationale when it "
                         "reviews a trade in that stock. A rating never places a trade by itself.")
        for r in rejected:
            lines.append(f"Not used: {r['name']}, {r['reason']}.")
        return lines + follow_lines
    n = result.get("signals_processed", 0)
    lines = [f"Read as an analyst note: {n} rating{'s' if n != 1 else ''} found."]
    if not n:
        lines.append("Nothing was changed.")
    return lines + follow_lines
