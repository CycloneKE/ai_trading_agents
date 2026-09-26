"""Claude as the paid AI tier, under a monthly budget cap.

Switched off unless ANTHROPIC_API_KEY is set. Then Claude Sonnet 5 reviews
trade signals and Claude Haiku 4.5 does the high-volume JSON work (sector
outlooks, weight proposals, the dividend sleeve's check), as the gap analysis
recommends. Every call is priced and counted against `AiBudget`; once the
month's cap would be passed, the orchestrator skips Claude and the free
models (Gemini, OpenRouter) carry on.

Uses the official `anthropic` SDK. It is imported only when a key is set,
so the agent runs without the package installed.
"""
import json
import logging
from typing import Any, Dict, Optional

from src.agent.ai_budget import AiBudget

logger = logging.getLogger(__name__)

REVIEW_MODEL_DEFAULT = 'claude-sonnet-5'
VOLUME_MODEL_DEFAULT = 'claude-haiku-4-5'

# The shape validate_trade expects back. Structured outputs guarantee the
# reply parses and matches it, so a review never fails on a stray sentence.
TRADE_VERDICT_SCHEMA = {
    'type': 'object',
    'properties': {
        'action': {'type': 'string', 'enum': ['buy', 'sell', 'hold']},
        'confidence': {'type': 'number'},
        'position_size': {'type': 'number'},
        'reasoning': {'type': 'string'},
    },
    'required': ['action', 'confidence', 'position_size', 'reasoning'],
    'additionalProperties': False,
}


class ClaudeUnavailable(Exception):
    """Claude was not asked (no budget left) or gave no usable answer."""


class ClaudeProvider:
    def __init__(self, api_key: Optional[str], budget: AiBudget,
                 config: Optional[Dict[str, Any]] = None, client: Any = None):
        cfg = config or {}
        self.api_key = api_key
        self.budget = budget
        self.review_model = cfg.get('trade_review_model', REVIEW_MODEL_DEFAULT)
        self.volume_model = cfg.get('volume_model', VOLUME_MODEL_DEFAULT)
        self.vision_model = cfg.get('vision_model', self.review_model)
        # Effort trades thoroughness for tokens. A trade review is a short
        # judgement, so medium; Haiku 4.5 does not take an effort setting.
        self.review_effort = cfg.get('review_effort', 'medium')
        self.review_max_tokens = int(cfg.get('review_max_tokens', 8000))
        self.volume_max_tokens = int(cfg.get('volume_max_tokens', 2000))
        self._client = client
        self._timeout = float(cfg.get('timeout_seconds', 30))

    @property
    def configured(self) -> bool:
        return bool(self.api_key) or self._client is not None

    def available(self) -> bool:
        """Configured and with budget left this month."""
        return self.configured and self.budget.remaining() > 0

    def _get_client(self):
        if self._client is None:
            import anthropic
            # Short timeout and one retry: the trading loop is waiting, and
            # a slow review falls back to the free models instead.
            self._client = anthropic.Anthropic(api_key=self.api_key, timeout=self._timeout,
                                               max_retries=1)
        return self._client

    def complete_json(self, system_prompt: str, user_prompt: str,
                      purpose: str = 'volume',
                      model_override: Optional[str] = None) -> Dict[str, Any]:
        """A JSON object from Claude, or ClaudeUnavailable.

        `purpose` 'review' uses the trade-review model with the verdict
        schema; anything else uses the volume model and asks for JSON in the
        prompt. SDK errors (rate limits included) propagate so the
        orchestrator can cool the provider down.
        """
        review = purpose == 'review'
        model = self.review_model if review else self.volume_model
        if model_override and str(model_override).startswith('claude-'):
            model = model_override
        max_tokens = self.review_max_tokens if review else self.volume_max_tokens
        if not self.budget.allows(self.budget.worst_case(
                model, len(system_prompt) + len(user_prompt), max_tokens)):
            raise ClaudeUnavailable('monthly AI budget reached')

        kwargs: Dict[str, Any] = {
            'model': model, 'max_tokens': max_tokens,
            'system': system_prompt if review else
            system_prompt + '\n\nReply with a single JSON object and nothing else.',
            'messages': [{'role': 'user', 'content': user_prompt}],
        }
        output_config: Dict[str, Any] = {}
        if review:
            output_config['format'] = {'type': 'json_schema', 'schema': TRADE_VERDICT_SCHEMA}
        if review and not model.startswith('claude-haiku'):
            output_config['effort'] = self.review_effort
        if output_config:
            kwargs['output_config'] = output_config

        response = self._get_client().messages.create(**kwargs)
        # Billed whatever the outcome, so counted before anything else.
        self.budget.record(model, response.usage, purpose)

        if response.stop_reason == 'refusal':
            raise ClaudeUnavailable('Claude declined the request')
        if response.stop_reason == 'max_tokens':
            raise ClaudeUnavailable(f'reply cut off at {max_tokens} tokens')
        text = next((b.text for b in response.content if b.type == 'text'), '')
        return _parse_json(text)


    def read_image_json(self, system_prompt: str, user_prompt: str, image_b64: str,
                        media_type: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """A JSON object read from an image, shaped by `schema`, or
        ClaudeUnavailable. Uses the review model: transcribing figures is
        where a misread digit costs most."""
        model = self.vision_model
        max_tokens = 4000
        # An image costs up to about 1,600 input tokens; counted as text.
        if not self.budget.allows(self.budget.worst_case(
                model, len(system_prompt) + len(user_prompt) + 4800, max_tokens)):
            raise ClaudeUnavailable('monthly AI budget reached')
        response = self._get_client().messages.create(
            model=model, max_tokens=max_tokens, system=system_prompt,
            messages=[{'role': 'user', 'content': [
                {'type': 'image', 'source': {'type': 'base64', 'media_type': media_type,
                                             'data': image_b64}},
                {'type': 'text', 'text': user_prompt}]}],
            output_config={'format': {'type': 'json_schema', 'schema': schema}})
        self.budget.record(model, response.usage, 'vision')
        if response.stop_reason == 'refusal':
            raise ClaudeUnavailable('Claude declined the request')
        if response.stop_reason == 'max_tokens':
            raise ClaudeUnavailable(f'reply cut off at {max_tokens} tokens')
        text = next((b.text for b in response.content if b.type == 'text'), '')
        return _parse_json(text)


def _parse_json(text: str) -> Dict[str, Any]:
    """The JSON object in a reply, tolerating a markdown code fence."""
    body = text.strip()
    if body.startswith('```'):
        body = body.split('\n', 1)[1] if '\n' in body else ''
        body = body.rsplit('```', 1)[0]
    try:
        data = json.loads(body)
    except ValueError:
        start, end = body.find('{'), body.rfind('}')
        if start < 0 or end <= start:
            raise ClaudeUnavailable('reply was not JSON')
        try:
            data = json.loads(body[start:end + 1])
        except ValueError:
            raise ClaudeUnavailable('reply was not JSON')
    if not isinstance(data, dict):
        raise ClaudeUnavailable('reply was not a JSON object')
    return data
