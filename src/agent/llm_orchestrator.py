"""
LLM Orchestrator Layer
Handles the reasoning layer for trading decisions.
"""
import os
import json
import logging
import time
import re
import requests
from typing import Dict, Any, List, Optional

from src.agent.guardrails import bound_verdict

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# Tried in turn when the configured Gemini model answers 404, which is what
# Google returns once a model is retired. The first that answers is kept.
# Flash-Lite first: on the free tier Google allows it about 500 requests a
# day, against about 20 for the full Flash models (September 2026).
GEMINI_FALLBACK_MODELS = ("gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
                          "gemini-flash-latest", "gemini-2.5-flash")
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
# Free OpenRouter models come and go. When the configured one is withdrawn,
# a free model that returns JSON is picked from OpenRouter's own list, these
# families first, larger context windows first within a family.
FREE_MODEL_FAMILIES = ("meta-llama/", "deepseek/", "qwen/", "mistralai/", "google/", "openai/", "nvidia/")
QUOTA_COOLDOWN_SECONDS = 1800   # a spent daily allowance: stop asking for a while
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
# Used when Groq retires the configured model: the first of these families
# that Groq still lists, larger context windows and then newer versions
# first. Vision reads the rating-sheet pictures, so only models that accept
# images qualify there: Qwen 3.x (qwen/qwen3.8-27b in September 2026; the
# older text-only qwen/qwen3-32b does not match "qwen/qwen3.") and Llama 4.
GROQ_TEXT_FAMILIES = ("openai/gpt-oss-120b", "qwen", "openai/gpt-oss", "kimi", "llama")
GROQ_VISION_FAMILIES = ("qwen/qwen3.", "llama-4-maverick", "llama-4-scout", "vision", "-vl")
GROQ_SUBSTITUTE_TRIES = 3
GROQ_NOT_CHAT = ("whisper", "guard", "tts", "orpheus", "playai", "distil")


KEY_NAMES = {'anthropic': 'ANTHROPIC_API_KEY', 'gemini': 'GEMINI_API_KEY',
             'openrouter': 'OPENROUTER_API_KEY', 'groq': 'GROQ_API_KEY'}


def advice(provider: str, error: Optional[str]) -> str:
    """What the operator can do about a provider's last failure."""
    e = (error or '').lower()
    if 'http 401' in e or 'http 403' in e or 'api key' in e or 'permission' in e:
        return f"The service refused the key: check {KEY_NAMES.get(provider, 'the API key')} in Coolify."
    if 'http 429' in e or 'quota' in e or 'rate' in e:
        if provider == 'anthropic':
            return "Claude is rate-limited for now; it recovers on its own."
        if provider == 'groq':
            return "Groq's free limit is used up for now; it resets within the day and recovers on its own."
        return "The free allowance is used up for now; Gemini's resets daily. It recovers on its own."
    if 'budget' in e:
        return "This month's Claude budget is used up; the free models take over until next month."
    if 'http 404' in e:
        if provider == 'gemini':
            return ("The Gemini model named in config.json is no longer offered. The agent now tries "
                    "newer ones on its own; if this stays, set gemini_model to gemini-flash-lite-latest.")
        if provider == 'groq':
            return ("Groq no longer offers this model on the free plan. The agent now picks a current "
                    "one from Groq's list on its own; if this stays, set groq_model (text) or "
                    "groq_vision_model (pictures) in config.json.")
        if provider == 'openrouter':
            return ("The OpenRouter model is no longer offered. The agent now picks a current free "
                    "model on its own; if this stays, set swarm.agents.synthesizer in config.json.")
    return "The service did not answer; this usually clears on its own."


def cooldown_for(err: Exception, default: float) -> float:
    """How long to leave a rate-limited provider alone: the provider's own
    Retry-After or retryDelay when it gives one; half an hour when the
    message says a quota is spent (a daily allowance does not come back in
    a minute, and Groq's daily token limit says "per day"); otherwise the
    default."""
    resp = getattr(err, 'response', None)
    try:
        after = (getattr(resp, 'headers', None) or {}).get('Retry-After')
        if after:
            return max(float(after), default)
    except (TypeError, ValueError):
        pass
    try:
        body = resp.json() if resp is not None else {}
        for d in (body.get('error') or {}).get('details') or []:
            delay = str(d.get('retryDelay') or '')
            if delay.endswith('s'):
                return max(float(delay[:-1]), default)
    except Exception:
        pass
    said = describe_error(err).lower()
    return QUOTA_COOLDOWN_SECONDS if 'quota' in said or 'per day' in said else default


def describe_error(err: Exception) -> str:
    """A short, safe account of a failed AI call for the dashboard: the
    HTTP status and the provider's own message, never the request URL (the
    Gemini key used to travel in it)."""
    resp = getattr(err, 'response', None)
    code = getattr(resp, 'status_code', None) or getattr(err, 'status_code', None)
    msg = ''
    if resp is not None:
        try:
            body = resp.json()
            e = body.get('error') if isinstance(body, dict) else None
            msg = (e.get('message') if isinstance(e, dict) else e) or ''
        except Exception:
            msg = ''
    if not msg:
        msg = getattr(err, 'message', None) or (str(err) if resp is None else '') or type(err).__name__
    # Keep the provider's own sentence; drop its links and "for more
    # information" tails, which only cut off mid-address on the dashboard.
    limit = re.search(r'limit:\s*(\d+),\s*model:\s*([\w.\-]+)', str(msg))
    msg = re.split(r'\s(?:For more information|To monitor|Learn more)', str(msg))[0]
    msg = re.sub(r'https?://\S+', '', msg).strip(' .') + '.'
    if limit:
        msg += f" (limit {limit.group(1)}, model {limit.group(2)})"
    text = f"HTTP {code}: {msg}" if code else str(msg)
    return re.sub(r'key=[^&\s]+', 'key=...', ' '.join(str(text).split()))[:220]

logger = logging.getLogger(__name__)

class LLMOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        # Groq: free tier, fast, OpenAI-style API; one of its models reads
        # pictures too. These are the free plan's models as of September 2026
        # (Llama 3.3 and Llama 4 left it); change them in config.json.
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = config.get("groq_model", "openai/gpt-oss-120b")
        self.groq_vision_model = config.get("groq_vision_model", "qwen/qwen3.8-27b")
        self._groq_models_cache: tuple = (0.0, [])

        # Determine primary model strategy
        self.primary_provider = config.get("primary_llm_provider", "openrouter")
        self.enabled = config.get("llm_enabled", True)

        # Google's alias for its current Flash-Lite model, the free model
        # with the largest daily allowance. Override with config 'gemini_model'.
        self.gemini_model = config.get("gemini_model", "gemini-flash-lite-latest")

        self.cache_ttl = config.get('llm_cache_ttl', 900)  # 15 min default
        self._verdict_cache: Dict[str, tuple] = {}  # key -> (expires_at, verdict)

        # Per-provider 429 circuit-breaker: when a provider rate-limits us, skip
        # it for a cooldown window (and use the other provider) instead of
        # hammering it every cycle and flooding the log.
        self.cooldown_seconds = config.get('llm_cooldown_seconds', 60)
        self._cooldown_until: Dict[str, float] = {}  # provider -> epoch
        # Each provider's recent record, for the dashboard: when it last
        # answered, and what it said when it last failed.
        self._health: Dict[str, Dict[str, Any]] = {}
        self.last_image_errors: List[tuple] = []
        # OpenRouter models found withdrawn, and the free model used instead.
        self._withdrawn_models: set = set()
        self._openrouter_substitute: Optional[str] = None

        # Paid tier: Claude, off unless ANTHROPIC_API_KEY is set, and then
        # held to a monthly budget (claude_provider.py, ai_budget.py). The
        # free providers above stay as the fallback.
        self.claude = None
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            from src.agent.ai_budget import AiBudget
            from src.agent.claude_provider import ClaudeProvider
            from src.utils.paths import DATA_DIR
            ccfg = config.get('claude', {}) or {}
            budget = AiBudget(DATA_DIR / 'ai_spend.json', ccfg.get('monthly_budget_usd', 20),
                              ccfg.get('prices'))
            self.claude = ClaudeProvider(anthropic_key, budget, ccfg)
            logger.info(f"Claude paid tier on: {self.claude.review_model} reviews trades, "
                        f"{self.claude.volume_model} does volume work, "
                        f"${budget.cap:.2f}/month cap")

        if not (self.openrouter_api_key or self.gemini_api_key or self.groq_api_key) and self.claude is None:
            logger.warning("No LLM API keys found. LLM Orchestrator will be disabled.")
            self.enabled = False

    def _provider_order(self):
        """Claude first while it has budget left this month; then the primary
        free provider, then the other, each only if its key is set. Enables
        bidirectional fallback regardless of which is primary (the old code
        only fell back openrouter->gemini)."""
        order = ([self.primary_provider] +
                 [p for p in ('groq', 'gemini', 'openrouter') if p != self.primary_provider])
        keys = {'groq': self.groq_api_key, 'gemini': self.gemini_api_key,
                'openrouter': self.openrouter_api_key}
        free = [p for p in order if keys.get(p)]
        paid = ['anthropic'] if self.claude is not None and self.claude.available() else []
        return paid + free

    def _status_code(self, err) -> Optional[int]:
        resp = getattr(err, 'response', None)
        return getattr(resp, 'status_code', None) if resp is not None else None

    def _complete(self, system_prompt: str, user_prompt: str,
                  fallback, model_override: Optional[str] = None, purpose: str = 'volume'):
        """Try each usable provider in order; on 429 put that provider on
        cooldown and try the next; on any other error try the next. Returns
        the first success, else `fallback`. Providers on cooldown are skipped
        without a call. `purpose` ('review' or 'volume') picks Claude's model."""
        import time as _time
        now = _time.time()
        callers = {'gemini': self._call_gemini, 'openrouter': self._call_openrouter,
                   'groq': self._call_groq,
                   'anthropic': lambda s, u, fb, model_override=None:
                       self._call_claude(s, u, fb, model_override, purpose)}
        for provider in self._provider_order():
            if self._cooldown_until.get(provider, 0) > now:
                continue
            try:
                result = callers[provider](system_prompt, user_prompt, fallback,
                                           model_override=model_override)
                self._record(provider)
                return result
            except Exception as e:
                self._record(provider, e)
                if self._status_code(e) == 429:
                    wait = cooldown_for(e, self.cooldown_seconds)
                    self._cooldown_until[provider] = now + wait
                    logger.warning(f"LLM provider '{provider}' rate-limited (429); "
                                   f"cooling down {wait:.0f}s, using fallback provider.")
        return fallback

    def _record(self, provider: str, err: Optional[Exception] = None) -> None:
        """Note a provider's answer or failure. The first failure after a
        success is logged as a warning (it used to be debug only, so an
        outage never showed); repeats are logged every 20th time."""
        h = self._health.setdefault(provider, {'last_ok': None, 'last_error': None,
                                               'last_error_at': None, 'failures': 0})
        if err is None:
            h['last_ok'], h['failures'] = time.time(), 0
            return
        h['last_error'], h['last_error_at'] = describe_error(err), time.time()
        h['failures'] += 1
        if h['failures'] == 1 or h['failures'] % 20 == 0:
            logger.warning(f"AI provider '{provider}' failed ({h['failures']} in a row): {h['last_error']}")

    def provider_health(self) -> List[Dict[str, Any]]:
        """Each configured AI provider: its model, whether it can read
        pictures, and whether it is answering."""
        out = []
        configured = [('anthropic', self.claude is not None,
                       self.claude.review_model if self.claude else None, True),
                      ('groq', bool(self.groq_api_key), self.groq_model, True),
                      ('gemini', bool(self.gemini_api_key), self.gemini_model, True),
                      ('openrouter', bool(self.openrouter_api_key),
                       self._openrouter_substitute
                       or self.config.get("swarm", {}).get("agents", {}).get("synthesizer"), False)]
        for name, on, model, sees in configured:
            if not on:
                continue
            h = self._health.get(name, {})
            failing = bool(h.get('failures')) and (h.get('last_ok') is None
                                                   or h['last_error_at'] > h['last_ok'])
            out.append({'provider': name, 'model': model, 'reads_images': sees,
                        'advice': advice(name, h.get('last_error')) if failing else None,
                        'status': 'failing' if failing else ('ok' if h.get('last_ok') else 'unused'),
                        'last_ok': h.get('last_ok'), 'last_error': h.get('last_error'),
                        'last_error_at': h.get('last_error_at'), 'failures': h.get('failures', 0)})
        return out

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
        cache_key = f"{symbol}:{action}:{conf_bucket}"
        cached = self._verdict_cache.get(cache_key)
        if cached and cached[0] > time.time():
            return bound_verdict(strategy_signal, dict(cached[1]))

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
        
        def _remember(verdict):
            # Only cache genuine LLM verdicts. _complete returns the base
            # strategy_signal object unchanged on an all-providers-down outage
            # (or a JSON-decode failure); caching that would suppress real LLM
            # validation for this (symbol, action, confidence) bucket for the
            # full TTL even after the provider recovers from a brief 429.
            if isinstance(verdict, dict) and verdict is not strategy_signal:
                self._verdict_cache[cache_key] = (time.time() + self.cache_ttl, dict(verdict))
            # The AI may confirm, weaken or veto; never reverse or strengthen
            # (guardrails.bound_verdict).
            return bound_verdict(strategy_signal, verdict)

        # All-providers-down returns the base signal: a trade is never
        # force-held by an LLM outage.
        return _remember(self._complete(system_prompt, user_prompt,
                                        strategy_signal, model_override=model,
                                        purpose='review'))

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
            dispatch = {'openrouter': self._call_openrouter, 'gemini': self._call_gemini,
                        'groq': self._call_groq, 'anthropic': self._call_claude}
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

    def _call_claude(self, system_prompt: str, user_prompt: str,
                     fallback_signal: Optional[Dict[str, Any]],
                     model_override: Optional[str] = None,
                     purpose: str = 'volume') -> Optional[Dict[str, Any]]:
        """Claude's JSON answer. Raises when Claude gives none (budget spent,
        a refusal, an API error), so _complete moves on to the free models."""
        result = self.claude.complete_json(system_prompt, user_prompt, purpose=purpose,
                                           model_override=model_override)
        if fallback_signal is not None:
            result['strategy'] = 'llm_orchestrated'
        return result

    def ai_budget(self) -> Dict[str, Any]:
        """The paid tier's state for the dashboard."""
        if self.claude is None:
            return {'paid_tier': False,
                    'note': 'Claude is off (no ANTHROPIC_API_KEY); trade reviews use the free models.'}
        return {'paid_tier': True, 'review_model': self.claude.review_model,
                'volume_model': self.claude.volume_model, **self.claude.budget.summary()}

    def read_image_json(self, system_prompt: str, user_prompt: str, image_b64: str,
                        media_type: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """A JSON object read from an image by the first model that can see
        one: Claude while it has budget, then Groq (its picture model has an
        allowance of its own), then Gemini, whose small free allowance is
        better kept for trade reviews. OpenRouter's free model is text only.
        None when none answers."""
        import time as _time
        self.last_image_errors = []
        callers = []
        if self.claude is not None and self.claude.available():
            callers.append(('anthropic', lambda: self.claude.read_image_json(
                system_prompt, user_prompt, image_b64, media_type, schema)))
        elif self.claude is not None:
            self.last_image_errors.append(('Claude', "this month's AI budget is used up"))
        if self.groq_api_key:
            callers.append(('groq', lambda: self._groq_image(
                system_prompt, user_prompt, image_b64, media_type)))
        if self.gemini_api_key:
            callers.append(('gemini', lambda: self._gemini_image(
                system_prompt, user_prompt, image_b64, media_type)))
        if not callers and not self.last_image_errors:
            self.last_image_errors.append((
                'setup', 'no AI service that can read pictures is set up: the OpenRouter model '
                         'reads text only. Add GROQ_API_KEY or GEMINI_API_KEY (both free), or '
                         'ANTHROPIC_API_KEY, in Coolify'))
        names = {'anthropic': 'Claude', 'gemini': 'Gemini', 'groq': 'Groq'}
        for provider, call in callers:
            if self._cooldown_until.get(provider, 0) > _time.time():
                self.last_image_errors.append((names[provider], 'rate-limited a moment ago; try again in a minute'))
                continue
            try:
                result = call()
                self._record(provider)
                if isinstance(result, dict):
                    return result
                self.last_image_errors.append((names[provider], 'answered, but not with the table'))
            except Exception as e:
                self._record(provider, e)
                if self._status_code(e) == 429:
                    self._cooldown_until[provider] = _time.time() + cooldown_for(e, self.cooldown_seconds)
                self.last_image_errors.append((names[provider], describe_error(e)))
        return None

    def _gemini_image(self, system_prompt: str, user_prompt: str, image_b64: str,
                      media_type: str) -> Optional[Dict[str, Any]]:
        data = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": media_type, "data": image_b64}},
                {"text": f"{system_prompt}\n\n{user_prompt}"}]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0},
        }
        response = self._gemini_post(data, timeout=60)
        text = response.json()['candidates'][0]['content']['parts'][0]['text']
        result = json.loads(text)
        return result if isinstance(result, dict) else None

    def _gemini_post(self, data: Dict[str, Any], timeout: float, model: Optional[str] = None):
        """POST to Gemini with the key in a header, not the URL, so it never
        appears in an error message or a log. When the model answers 404
        (retired), newer models are tried and the first that answers kept."""
        first = model or self.gemini_model
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.gemini_api_key or ''}
        tried = [first] + [m for m in GEMINI_FALLBACK_MODELS if m != first]
        response = None
        for candidate in tried:
            response = requests.post(GEMINI_ENDPOINT.format(model=candidate), headers=headers,
                                     json=data, timeout=timeout)
            if response.status_code == 404 and candidate != tried[-1]:
                continue
            response.raise_for_status()
            if candidate != first:
                logger.warning(f"Gemini model '{first}' is not available (404); using '{candidate}' instead. "
                               f"Set gemini_model in config.json to make it permanent.")
                if first == self.gemini_model:
                    self.gemini_model = candidate
            return response
        return response

    def _call_openrouter(self, system_prompt: str, user_prompt: str, fallback_signal: Optional[Dict[str, Any]], model_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "Content-Type": "application/json"
        }
        
        model = model_override or self.config.get("swarm", {}).get("agents", {}).get("synthesizer", "meta-llama/llama-3.1-8b-instruct")
        if model in self._withdrawn_models and self._openrouter_substitute:
            model = self._openrouter_substitute

        def post(model_id):
            # Allow up to 45 seconds for DeepSeek-R1 or o1 models to produce thinking tokens
            timeout = 45 if ("r1" in model_id.lower() or "o1" in model_id.lower()) else 15
            data = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "response_format": {"type": "json_object"}
            }
            return requests.post(url, headers=headers, json=data, timeout=timeout)

        response = post(model)
        if response.status_code == 404:
            # The model is withdrawn (free versions are, from time to time):
            # use a free model that is offered now, and keep using it.
            self._withdrawn_models.add(model)
            substitute = self._free_openrouter_model()
            if substitute:
                logger.warning(f"OpenRouter model '{model}' is no longer offered; using the free "
                               f"model '{substitute}' instead. Set swarm.agents.synthesizer in "
                               f"config.json to choose another.")
                self._openrouter_substitute = model = substitute
                response = post(model)
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

    # ------------------------------------------------------------------ Groq

    def _groq_post(self, messages: List[Dict[str, Any]], vision: bool = False,
                   json_mode: bool = True, timeout: float = 20, max_tokens: Optional[int] = None):
        """POST a chat to Groq. When Groq says the model is gone (retired, or
        no longer on the free plan), current ones are picked from Groq's own
        list and tried in turn; the first that answers is kept."""
        attr = 'groq_vision_model' if vision else 'groq_model'
        headers = {"Authorization": f"Bearer {self.groq_api_key}", "Content-Type": "application/json"}
        model, gone = getattr(self, attr), set()
        for _ in range(GROQ_SUBSTITUTE_TRIES + 1):
            response = self._groq_send(model, messages, json_mode, max_tokens, headers, timeout)
            if not model_gone(response):
                break
            gone.add(model)
            substitute = self._current_groq_model(vision, exclude=gone)
            if not substitute:
                break
            logger.warning(f"Groq model '{model}' is not offered; trying '{substitute}'. "
                           f"Set {attr} in config.json to choose another.")
            model = substitute
        response.raise_for_status()
        setattr(self, attr, model)
        return response

    @staticmethod
    def _groq_send(model: str, messages: List[Dict[str, Any]], json_mode: bool,
                   max_tokens: Optional[int], headers: Dict[str, str], timeout: float):
        body: Dict[str, Any] = {"model": model, "messages": messages, "temperature": 0.2}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if max_tokens:
            body["max_completion_tokens"] = max_tokens
        if model.startswith("openai/gpt-oss"):
            # A trade review needs little reasoning, and less keeps each call
            # inside the free plan's per-minute token limit.
            body["reasoning_effort"] = "low"
        response = requests.post(GROQ_CHAT_URL, headers=headers, json=body, timeout=timeout)
        if (getattr(response, 'status_code', None) == 400 and "reasoning_effort" in body
                and 'reasoning' in error_text(response)):
            body.pop("reasoning_effort")
            response = requests.post(GROQ_CHAT_URL, headers=headers, json=body, timeout=timeout)
        return response

    def _current_groq_model(self, vision: bool, exclude=()) -> Optional[str]:
        """A model Groq lists as active now, chosen by family (see
        GROQ_TEXT_FAMILIES / GROQ_VISION_FAMILIES). The list is fetched at
        most once an hour."""
        at, models = self._groq_models_cache
        if time.time() - at > 3600 or not models:
            try:
                resp = requests.get(GROQ_MODELS_URL, timeout=15,
                                    headers={"Authorization": f"Bearer {self.groq_api_key}"})
                resp.raise_for_status()
                models = resp.json().get('data') or []
                self._groq_models_cache = (time.time(), models)
            except Exception as e:
                logger.warning(f"Could not list Groq's models: {describe_error(e)}")
                return None
        return pick_groq_model(models, vision, exclude)

    def _call_groq(self, system_prompt: str, user_prompt: str,
                   fallback_signal: Optional[Dict[str, Any]],
                   model_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        # model_override names an OpenRouter or Gemini model; Groq uses its own.
        response = self._groq_post([{"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}])
        text = response.json()['choices'][0]['message']['content']
        try:
            result = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            logger.error(f"Groq did not answer in JSON (model {self.groq_model}): {str(text)[:200]}")
            return fallback_signal
        if isinstance(result, dict) and fallback_signal is not None:
            result['strategy'] = 'llm_orchestrated'
        return result

    def _groq_image(self, system_prompt: str, user_prompt: str, image_b64: str,
                    media_type: str) -> Optional[Dict[str, Any]]:
        """A JSON object read from a picture by Groq's vision model."""
        response = self._groq_post([{"role": "user", "content": [
            {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}}]}],
            vision=True, timeout=60, max_tokens=4096)
        text = response.json()['choices'][0]['message']['content']
        result = json_from_text(text)
        return result if isinstance(result, dict) else None

    def _free_openrouter_model(self) -> Optional[str]:
        """A free OpenRouter model that returns JSON, from OpenRouter's own
        list of models, or None. Checked at most once an hour."""
        now = time.time()
        if now - getattr(self, '_free_model_checked_at', 0) < 3600:
            return self._openrouter_substitute
        self._free_model_checked_at = now
        try:
            resp = requests.get(OPENROUTER_MODELS_URL, timeout=15,
                                headers={"Authorization": f"Bearer {self.openrouter_api_key}"})
            resp.raise_for_status()
            models = resp.json().get('data') or []
        except Exception as e:
            logger.warning(f"Could not list OpenRouter's models: {describe_error(e)}")
            return None
        return pick_free_model(models, self._withdrawn_models)

    def _call_gemini(self, system_prompt: str, user_prompt: str, fallback_signal: Optional[Dict[str, Any]], model_override: Optional[str] = None) -> Optional[Dict[str, Any]]:
        # Map dynamic model to gemini endpoints if applicable, otherwise default
        model = model_override or self.gemini_model
        if "/" in model:
            # An OpenRouter-style id (vendor/model) reached the Gemini path —
            # use the configured valid Gemini model instead.
            model = self.gemini_model
            
        data = {
            "contents": [{
                "parts": [{"text": f"{system_prompt}\n\nUser Data:\n{user_prompt}"}]
            }],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }
        
        response = self._gemini_post(data, timeout=15, model=model)
        result_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        try:
            result = json.loads(result_text)
            if isinstance(result, dict) and fallback_signal is not None:
                result['strategy'] = 'llm_orchestrated'
            return result
        except (json.JSONDecodeError, KeyError):
            logger.error(f"Failed to decode Gemini JSON response. Text: {result_text}")
            return fallback_signal



def pick_free_model(models: List[Dict[str, Any]], exclude=()) -> Optional[str]:
    """The best free model in OpenRouter's model list that can answer in
    JSON: FREE_MODEL_FAMILIES in order, then the largest context window."""
    def free(m):
        p = m.get('pricing') or {}
        try:
            return float(p.get('prompt', 1)) == 0 and float(p.get('completion', 1)) == 0
        except (TypeError, ValueError):
            return False

    def json_capable(m):
        params = m.get('supported_parameters')
        return not params or 'response_format' in params or 'structured_outputs' in params

    def family(mid):
        return next((i for i, f in enumerate(FREE_MODEL_FAMILIES) if mid.startswith(f)), len(FREE_MODEL_FAMILIES))

    candidates = [m for m in models if isinstance(m, dict) and m.get('id')
                  and m['id'] not in exclude and free(m) and json_capable(m)]
    if not candidates:
        return None
    candidates.sort(key=lambda m: (family(m['id']), -int(m.get('context_length') or 0), m['id']))
    return candidates[0]['id']


def model_gone(response) -> bool:
    """Whether a provider refused the request because the model no longer
    exists or was retired (404, or a 400 that says so)."""
    code = getattr(response, 'status_code', None)
    if code == 404:
        return True
    if code != 400:
        return False
    text = error_text(response)
    if 'decommissioned' in text or 'model_not_found' in text:
        return True
    return 'model' in text and ('does not exist' in text or 'no longer supported' in text)


def error_text(response) -> str:
    """A failed reply's body, lower case, for matching words in it."""
    try:
        return json.dumps(response.json()).lower()
    except Exception:
        return str(getattr(response, 'text', '')).lower()


def pick_groq_model(models: List[Dict[str, Any]], vision: bool, exclude=()) -> Optional[str]:
    """The model to use from Groq's list: an active chat model, by family
    preference, then the largest context window. For pictures, only the
    families known to read images qualify."""
    families = GROQ_VISION_FAMILIES if vision else GROQ_TEXT_FAMILIES

    def family(mid):
        return next((i for i, f in enumerate(families) if f in mid), None)

    usable = [m for m in models if isinstance(m, dict) and m.get('id')
              and m['id'] not in exclude and m.get('active', True) is not False
              and not any(w in m['id'].lower() for w in GROQ_NOT_CHAT)
              and family(m['id']) is not None]
    if not usable:
        return None
    usable.sort(key=lambda m: m['id'], reverse=True)                  # newer versions first
    usable.sort(key=lambda m: (family(m['id']), -int(m.get('context_window') or 0)))
    return usable[0]['id']


def json_from_text(text: Any) -> Optional[Dict[str, Any]]:
    """The JSON object in a model's reply, allowing a code fence or words
    around it, or a model's own reasoning in <think> tags before it; None
    when there is none."""
    body = re.sub(r'<think>.*?</think>', '', str(text or ''), flags=re.S).strip()
    if body.startswith('```'):
        body = body.split('\n', 1)[1] if '\n' in body else ''
        body = body.rsplit('```', 1)[0]
    try:
        data = json.loads(body)
        return data if isinstance(data, dict) else None
    except ValueError:
        start, end = body.find('{'), body.rfind('}')
        if 0 <= start < end:
            try:
                data = json.loads(body[start:end + 1])
                return data if isinstance(data, dict) else None
            except ValueError:
                return None
    return None
