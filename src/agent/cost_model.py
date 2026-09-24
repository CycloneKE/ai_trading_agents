"""Per-market transaction costs.

A single global `trading.commission` cannot describe this portfolio. Alpaca
charges no commission on US equities, while a Nairobi Securities Exchange
trade carries brokerage plus the CMA levy, the NSE levy and CDSC fees. The
shipped 0.1% is about right for the former and roughly 15 to 20 times too
low for the latter.

That gap is not cosmetic. At 1.7% a side the round trip is 3.4%, so an NSE
position must gain 3.4% before it breaks even. A strategy rebalancing
monthly needs better than 40% a year gross just to stand still. Costing NSE
trades at 0.1% makes any backtest of them fiction, and makes short-horizon
NSE trading look viable when it is not.

The NSE figures below are PLACEHOLDERS. Confirm the real schedule with your
broker and update `config.json`; see the note on `verified` in each entry.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Fallbacks used only when config carries no `costs` section at all.
DEFAULT_COSTS: Dict[str, Dict[str, Any]] = {
    'us_equity': {
        'commission_pct': 0.0,
        'min_commission': 0.0,
        'slippage_pct': 0.0005,
        'verified': True,
        'note': 'Alpaca charges no commission on US equities.',
    },
    'nse': {
        'commission_pct': 0.017,
        'min_commission': 0.0,
        'slippage_pct': 0.0030,
        'verified': False,
        'note': 'PLACEHOLDER. Brokerage plus CMA levy, NSE levy and CDSC fees. '
                'Confirm the current schedule with your broker and set verified=true.',
    },
    'crypto': {
        'commission_pct': 0.006,
        'min_commission': 0.0,
        'slippage_pct': 0.0010,
        'verified': False,
        'note': 'PLACEHOLDER. Coinbase taker fees vary by volume tier.',
    },
}

DEFAULT_MARKET = 'us_equity'


def _configured_markets(config: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Market entries from config, skipping documentation keys.

    The `costs` block carries a `_comment` string alongside the market
    dictionaries; treating that as a market raises on `dict.update`.
    """
    if not isinstance(config, dict):
        return {}
    section = config.get('costs') or {}
    if not isinstance(section, dict):
        return {}
    return {k: v for k, v in section.items()
            if isinstance(v, dict) and not k.startswith('_')}


def classify(symbol: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Map a symbol to a market key.

    Driven by the configured symbol lists rather than guesswork, so adding a
    symbol to `data_manager.nse_symbols` is enough to have it costed as one.
    """
    if not symbol:
        return DEFAULT_MARKET
    sym = str(symbol).upper()
    dm = (config or {}).get('data_manager', {}) if isinstance(config, dict) else {}

    if sym in {s.upper() for s in dm.get('nse_symbols', []) or []}:
        return 'nse'
    if sym in {s.upper() for s in dm.get('crypto_symbols', []) or []}:
        return 'crypto'
    # A dash-suffixed fiat pair is the crypto convention used throughout the
    # config (BTC-USD, ETH-USD); plain tickers are US equities.
    if '-' in sym and sym.rsplit('-', 1)[-1] in {'USD', 'USDT', 'USDC', 'EUR'}:
        return 'crypto'
    return DEFAULT_MARKET


def market_costs(market: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve commission and slippage for a market key, as fractions per side."""
    base = dict(DEFAULT_COSTS.get(market, DEFAULT_COSTS[DEFAULT_MARKET]))
    override = _configured_markets(config).get(market)
    if override:
        base.update(override)
    base['market'] = market
    return base


def costs_for(symbol: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve commission and slippage for one symbol, as fractions per side."""
    return market_costs(classify(symbol, config), config)


def round_trip_pct(symbol: str, config: Optional[Dict[str, Any]] = None) -> float:
    """Total cost of entering and exiting, as a fraction of notional.

    This is the number a strategy has to clear before it earns anything, so
    it is the right figure to compare an expected edge against.
    """
    c = costs_for(symbol, config)
    return 2.0 * (float(c.get('commission_pct', 0.0)) + float(c.get('slippage_pct', 0.0)))


def unverified_markets(config: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Markets whose cost assumptions are still placeholders.

    Callers that report performance should surface these, so a backtest is
    never read as authoritative while it rests on a guessed fee schedule.
    """
    configured = _configured_markets(config)
    out = {}
    for market in set(DEFAULT_COSTS) | set(configured):
        merged = dict(DEFAULT_COSTS.get(market, {}))
        merged.update(configured.get(market, {}) or {})
        if not merged.get('verified', False):
            out[market] = merged.get('note', 'Cost assumptions not verified.')
    return out
