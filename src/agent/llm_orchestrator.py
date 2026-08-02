"""
LLM Orchestrator Layer
Handles the reasoning layer for trading decisions.
"""
import math
import os
import json
import logging
import time
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

VALID_ACTIONS = ('buy', 'sell', 'hold')


def _as_unit_float(value: Any, default: float) -> float:
    """Coerce an LLM-supplied number into [0.0, 1.0].

    Anything unusable (missing, non-numeric, NaN, inf) falls back to the
    base signal's own value rather than propagating into sizing arithmetic.
    """
    if isinstance(value, bool) or value is None:
        return default
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(num) or math.isinf(num):
        return 0.0 if math.isnan(num) else max(0.0, min(1.0, num))
    return max(0.0, min(1.0, num))


def validate_verdict(verdict: Any, base_signal: Dict[str, Any]) -> Dict[str, Any]:
    """Constrain an LLM verdict to what a *reviewer* is allowed to decide.

    The reviewer's remit is to approve, resize, or reject a signal the
    ensemble already produced — never to originate a trade in the opposite
    direction. `validate_trade` used to return the model's parsed JSON
    verbatim and main.py executed it, so a model answering 'sell' to a
    proposed 'buy' would have opened a short nothing in the system proposed.

    Guarantees for callers:
      * 'action' is one of buy/sell/hold, and is either the base signal's
        direction or 'hold'.
      * 'confidence' and 'position_size' are always present floats in [0, 1].
    """
    base_action = str(base_signal.get('action', 'hold')).strip().lower()
    base_conf = _as_unit_float(base_signal.get('confidence'), 0.0)
    base_size = _as_unit_float(base_signal.get('position_size'), 0.0)

    if not isinstance(verdict, dict):
        logger.warning(f"LLM returned {type(verdict).__name__}, not an object; "
                       f"using the base signal")
        return dict(base_signal)

    out = dict(verdict)
    action = str(out.get('action', '')).strip().lower()

    if action not in VALID_ACTIONS:
        logger.warning(f"LLM returned unrecognised action {out.get('action')!r}; holding")
        action = 'hold'
    elif action != base_action and action != 'hold':
        # A direction flip is out of scope for a reviewer. Treat it as a
        # rejection of the proposed trade, which is the conservative reading.
        logger.warning(
            f"LLM tried to flip {base_action} -> {action}; downgrading to hold "
            f"(a reviewer may reject a signal, not reverse it)")
        action = 'hold'

    out['action'] = action
    out['confidence'] = _as_unit_float(out.get('confidence'), base_conf)
    out['position_size'] = _as_unit_float(out.get('position_size'), base_size)
    if 'symbol' not in out and base_signal.get('symbol'):
        out['symbol'] = base_signal['symbol']
    return out

class LLMOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        # Determine primary model strategy
        self.primary_provider = config.get("primary_llm_provider", "openrouter")
        self.enabled = config.get("llm_enabled", True)

        # Valid free Gemini model (the old default gemini-2.5-flash-lite 404s
        # on the v1beta endpoint). Override with config 'gemini_model'.
        self.gemini_model = config.get("gemini_model", "gemini-2.0-flash")

        self.cache_ttl = config.get('llm_cache_ttl', 900)  # 15 min default
        self._verdict_cache: Dict[str, tuple] = {}  # key -> (expires_at, verdict)

        # Per-provider 429 circuit-breaker: when a provider rate-limits us, skip
        # it for a cooldown window (and use the other provider) instead of
        # hammering it every cycle and flooding the log.
        self.cooldown_seconds = config.get('llm_cooldown_seconds', 60)
        self._cooldown_until: Dict[str, float] = {}  # provider -> epoch

        if not self.openrouter_api_key and not self.gemini_api_key:
            logger.warning("No LLM API keys found. LLM Orchestrator will be disabled.")
            self.enabled = False

    def _provider_order(self):
        """Primary provider first, then the other — each included only if its
        key is set. Enables bidirectional fallback regardless of which is
        primary (the old code only fell back openrouter->gemini)."""
        order = ([self.primary_provider] +
                 [p for p in ('gemini', 'openrouter') if p != self.primary_provider])
        return [p for p in order
                if (p == 'gemini' and self.gemini_api_key)
                or (p == 'openrouter' and self.openrouter_api_key)]

    def _status_code(self, err) -> Optional[int]:
        resp = getattr(err, 'response', None)
        return getattr(resp, 'status_code', None) if resp is not None else None

    def _complete(self, system_prompt: str, user_prompt: str,
                  fallback, model_override: Optional[str] = None):
        """Try each usable provider in order; on 429 put that provider on
        cooldown and try the next; on any other error try the next. Returns
        the first success, else `fallback`. Providers on cooldown are skipped
        without a call."""
        import time as _time
        now = _time.time()
        callers = {'gemini': self._call_gemini, 'openrouter': self._call_openrouter}
        for provider in self._provider_order():
            if self._cooldown_until.get(provider, 0) > now:
                continue
            try:
                return callers[provider](system_prompt, user_prompt, fallback,
                                         model_override=model_override)
            except Exception as e:
                if self._status_code(e) == 429:
                    self._cooldown_until[provider] = now + self.cooldown_seconds
                    logger.warning(f"LLM provider '{provider}' rate-limited (429); "
                                   f"cooling down {self.cooldown_seconds}s, using fallback provider.")
                else:
                    logger.debug(f"LLM provider '{provider}' failed: {e}")
        return fallback

    def validate_trade(self, symbol: str, strategy_signal: Dict[str, Any], market_data: Dict[str, Any], news_data: list = None, research_context: Dict[str, Any] = None, sector_outlook: Dict[str, Any] = None, track_record: str = None) -> Dict[str, Any]:
        """
        Takes the base strategy signal and validates it against current market context using an LLM.
        """
        if not self.enabled:
            return strategy_signal
            
        action = strategy_signal.get("action", "hold")
        if action == "hold" and strategy_signal.get("confidence", 0) < 0.5:
            return strategy_signal  # Don't ask LLM about weak holds to save costs

        conf_bucket = round(strategy_signal.get('confidence', 0) * 10)  # 0.71/0.73 share a verdict
        # Price belongs in the key: with a 900s TTL and no price component a
        # verdict formed at one price was reapplied for fifteen minutes at a
        # materially different one. Buckets are log-scaled so the band is a
        # constant *percentage* move regardless of the symbol's price level.
        cache_key = f"{symbol}:{action}:{conf_bucket}:{self._price_bucket(market_data)}"
        cached = self._verdict_cache.get(cache_key)
        if cached and cached[0] > time.time():
            return dict(cached[1])

        system_prompt = (
            "You are a quantitative trading risk reviewer.\n"
            "A proposed trade signal from a technical/statistical ensemble has already cleared its "
            "own confidence threshold before reaching you — treat it as a real signal worth taking "
            "seriously, not a default you're expected to talk yourself out of.\n"
            "Review it against the available fundamental data, market news, and volatility context, "
            "and decide whether to approve it, resize it, or reject it.\n"
            "Only downgrade confidence or switch the action to 'hold' when you have a SPECIFIC, "
            "concrete reason: fresh news that directly contradicts the signal's direction, volatility "
            "that is genuinely elevated (not just unreported), or a clear fundamental red flag. "
            "Incomplete information or general uncertainty is normal and is NOT by itself a reason "
            "to veto — approve the signal as given unless something concrete argues against it.\n"
            "Return your response EXACTLY as a valid JSON object with the following schema:\n"
            '{"action": "buy|sell|hold", "confidence": <float 0.0-1.0>, "position_size": <float 0.0-1.0>, "reasoning": "<your rationale>"}\n'
            "Ensure the JSON output is raw JSON without markdown codeblocks or trailing characters."
        )

        # Extract volatility explicitly for the prompt. market_data here is the
        # already per-symbol-extracted dict from main.py's _extract_symbol_data
        # (flat: close/open/price/... at the top level, not nested under a
        # 'data'/symbol wrapper) — read the fields directly off it.
        volatility = market_data.get('volatility', 'Unknown')
        if isinstance(market_data, dict) and market_data.get('close') and market_data.get('open'):
            try:
                volatility = f"{abs((market_data['close'] - market_data['open']) / market_data['open']):.4f} (Intraday proxy)"
            except (TypeError, ZeroDivisionError):
                pass

        user_prompt = f"Target Asset: {symbol}\n" \
                      f"Proposed Ensemble Signal: {json.dumps(strategy_signal)}\n" \
                      f"Market Data Context (Close Price): {json.dumps(market_data.get('close', 'Unknown'))}\n" \
                      f"Current Asset Volatility: {volatility}\n"
        
        if research_context:
            user_prompt += (
                f"Broker Research Context (AIB AXYS): "
                f"Recommendation={research_context.get('recommendation')}, "
                f"Target Price={research_context.get('target_price')}, "
                f"Rationale: {research_context.get('rationale')}\n"
            )

        if sector_outlook:
            user_prompt += (
                f"Sector Outlook Context: "
                f"Outlook Score={sector_outlook.get('outlook_score')}\n"
                f"Sector Analysis: {sector_outlook.get('updated_profile_text')}\n"
                f"Sector Risks: {json.dumps(sector_outlook.get('risk_factors', []))}\n"
            )

        if news_data:
            user_prompt += f"Recent News Context: {json.dumps(news_data[:3])}\n"

        if track_record:
            user_prompt += f"Agent Track Record: {track_record}\n"

        user_prompt += "\nEvaluate this signal critically. Do you agree with the ensemble? Provide your validated JSON output now."
        
        # Resolve dynamic model config
        model = self.config.get("swarm", {}).get("agents", {}).get("synthesizer", "meta-llama/llama-3.1-8b-instruct")
        
        def _remember(verdict, _outage_sentinel=strategy_signal):
            # Only cache genuine LLM verdicts. _complete returns the base
            # strategy_signal object unchanged on an all-providers-down outage
            # (or a JSON-decode failure); caching that would suppress real LLM
            # validation for this (symbol, action, confidence) bucket for the
            # full TTL even after the provider recovers from a brief 429.
            if not isinstance(verdict, dict) or verdict is _outage_sentinel:
                # An outage returns the base signal object itself; caching it
                # would suppress real validation for the whole TTL.
                return dict(strategy_signal)

            validated = validate_verdict(verdict, strategy_signal)
            # A rejected direction-flip is our decision, not the model's —
            # caching it would keep vetoing the symbol for the full TTL on the
            # strength of one bad response.
            flipped = str(verdict.get('action', '')).strip().lower() not in (
                validated['action'], '')
            if not flipped:
                self._verdict_cache[cache_key] = (time.time() + self.cache_ttl,
                                                  dict(validated))
            return validated

        # All-providers-down returns the base signal: a trade is never
        # force-held by an LLM outage.
        return _remember(self._complete(system_prompt, user_prompt,
                                        strategy_signal, model_override=model))

    # Each bucket spans this fractional price move; a verdict is reused only
    # while the price stays inside the band it was formed in.
    PRICE_BUCKET_BAND = 0.005      # 0.5%

    def _price_bucket(self, market_data: Any) -> int:
        """Log-scaled price bucket, so the band is a constant percentage."""
        try:
            price = None
            if isinstance(market_data, dict):
                price = market_data.get('price') or market_data.get('close')
            price = float(price)
        except (TypeError, ValueError):
            return 0
        if price <= 0:
            return 0
        band = self.config.get('llm_cache_price_band', self.PRICE_BUCKET_BAND)
        return int(math.log(price) / math.log(1.0 + band))

    def propose_json(self, system_prompt: str, user_prompt: str, model_override: Optional[str] = None):
        """Generic JSON completion (used by the weight allocator and sector specialists).

        Returns the parsed dict, or None when no provider is configured or
        the call/parse fails.
        """
        if not self.enabled:
            return None
        result = self._complete(system_prompt, user_prompt, None,
                                model_override=model_override)
        return result if isinstance(result, dict) else None

    def propose_json_secondary(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """Query the secondary LLM provider for cross-validation consensus voting.
        
        This forces a call to the non-primary provider to get an independent
        opinion for multi-model consensus verification.
        """
        if not self.enabled:
            return None
        
        providers = self._provider_order()
        if len(providers) < 2:
            logger.warning("Multi-model consensus unavailable: only one LLM provider configured.")
            return None
        
        secondary = providers[1]
        try:
            dispatch = {'openrouter': self._call_openrouter, 'gemini': self._call_gemini}
            fn = dispatch.get(secondary)
            if fn:
                result = fn(system_prompt, user_prompt, None)
                if result:
                    try:
                        import json as _json
                        return _json.loads(result) if isinstance(result, str) else result
                    except (ValueError, TypeError):
                        return None
        except Exception as e:
            logger.warning(f"Secondary LLM provider ({secondary}) consensus call failed: {e}")
        return None

    def _call_openrouter(self, system_prompt: str, user_prompt: str, fallback_signal: Optional[Dict[str, Any]], model_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "Content-Type": "application/json"
        }
        
        model = model_override or self.config.get("swarm", {}).get("agents", {}).get("synthesizer", "meta-llama/llama-3.1-8b-instruct")
        
        # Allow up to 45 seconds for DeepSeek-R1 or o1 models to produce thinking tokens
        timeout = 45 if ("r1" in model.lower() or "o1" in model.lower()) else 15
        
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        
        result_text = response.json()['choices'][0]['message']['content']
        try:
            result = json.loads(result_text)
            if isinstance(result, dict) and fallback_signal is not None:
                result['strategy'] = 'llm_orchestrated'
            return result
        except json.JSONDecodeError:
            logger.error(f"Failed to decode OpenRouter JSON response. Model: {model}. Text: {result_text}")
            return fallback_signal

    def _call_gemini(self, system_prompt: str, user_prompt: str, fallback_signal: Optional[Dict[str, Any]], model_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        # Map dynamic model to gemini endpoints if applicable, otherwise default
        model = model_override or self.gemini_model
        if "/" in model:
            # An OpenRouter-style id (vendor/model) reached the Gemini path —
            # use the configured valid Gemini model instead.
            model = self.gemini_model
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": [{"text": f"{system_prompt}\n\nUser Data:\n{user_prompt}"}]
            }],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        
        result_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        try:
            result = json.loads(result_text)
            if isinstance(result, dict) and fallback_signal is not None:
                result['strategy'] = 'llm_orchestrated'
            return result
        except (json.JSONDecodeError, KeyError):
            logger.error(f"Failed to decode Gemini JSON response. Text: {result_text}")
            return fallback_signal

