"""LLM-based red-flag check for sleeve candidates. Structurally cannot pick
stocks, score them, or size positions — the deterministic scorer already
made that decision. This layer's only job is to surface a concrete, recent
reason NOT to buy, for the operator to see before approving the ticket.
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a red-team reviewer for a long-term dividend investment sleeve. "
    "You do not pick stocks, score them, or size positions — another system "
    "already decided this company is a numeric candidate for accumulation. "
    "Your ONLY job is to check for a recent, concrete reason NOT to buy more: "
    "a dividend cut or suspension, a profit warning, a rights issue, delisting "
    "risk, or a governance/fraud concern. General uncertainty or lack of news "
    "is NOT a reason to flag.\n"
    "Return your response EXACTLY as a valid JSON object with this schema:\n"
    '{"flag": <true|false>, "reason": "<short explanation, empty string if not flagged>"}\n'
    "Ensure the JSON output is raw JSON without markdown codeblocks."
)


@dataclass
class VetoResult:
    flag: bool
    reason: str
    available: bool  # False when the LLM check could not be completed


def check_candidate(llm_orchestrator, symbol: str, company_name: str = '') -> VetoResult:
    if llm_orchestrator is None or not getattr(llm_orchestrator, 'enabled', False):
        return VetoResult(flag=False, reason='', available=False)

    user_prompt = f"Symbol: {symbol}\nCompany: {company_name or symbol}\n"
    result = llm_orchestrator.propose_json(_SYSTEM_PROMPT, user_prompt)
    if not isinstance(result, dict) or 'flag' not in result:
        logger.debug(f"Sleeve veto unavailable for {symbol}: no usable LLM response")
        return VetoResult(flag=False, reason='', available=False)
    return VetoResult(flag=bool(result.get('flag')),
                      reason=str(result.get('reason', '')), available=True)
