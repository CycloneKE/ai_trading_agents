"""Whether a paper run will produce evidence, or ninety days of nothing.

The existing scripts/preflight.py checks that a demo session can be shown to
someone: env vars set, config parses, dashboard built. That is a different
question from whether the agent, left alone for three months, will actually
trade and leave a record worth reading.

Every check here corresponds to a way this system has silently produced
nothing:

- Strategies falling back to MockStrategy, which returns 'hold' forever. Two
  independent import paths caused it; the agent started, reported healthy,
  and never traded.
- api_server failing to import, taking the dashboard and the operator's only
  view of the run with it, reported as an unrelated Flask problem.
- The data layer serving synthetic fallback prices, which the agent
  correctly refuses to trade on. Vendors down means the run quietly stops
  trading rather than erroring.
- LLM validation demand exceeding the quota, after which the orchestrator's
  cooldown skips validation and trades execute unchecked.

A run that trips any of these is not a failed experiment, it is an absence
of one, and the absence is invisible until someone looks.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

BLOCK = 'block'
WARN = 'warn'


@dataclass
class Check:
    level: str
    name: str
    ok: bool
    detail: str = ''


@dataclass
class Readiness:
    checks: List[Check] = field(default_factory=list)

    def add(self, level: str, name: str, ok: bool, detail: str = '') -> None:
        self.checks.append(Check(level, name, ok, detail))

    @property
    def blockers(self) -> List[Check]:
        return [c for c in self.checks if c.level == BLOCK and not c.ok]

    @property
    def warnings(self) -> List[Check]:
        return [c for c in self.checks if c.level == WARN and not c.ok]

    @property
    def ready(self) -> bool:
        return not self.blockers


def check_strategies(cfg: Dict[str, Any], r: Readiness) -> None:
    """The strategies must be real. This is the failure that cost the most."""
    try:
        from src.agent.strategy_manager import MockStrategy, StrategyManager
        mgr = StrategyManager(cfg)
        if not mgr.strategies:
            r.add(BLOCK, 'strategies loaded', False, 'no strategies configured')
            return
        mocks = [n for n, s in mgr.strategies.items() if isinstance(s, MockStrategy)]
        r.add(BLOCK, 'strategies are real, not mocks', not mocks,
              f"{mocks} loaded as MockStrategy, which always returns 'hold'. "
              f"The run would place no trades at all." if mocks else
              f"{len(mgr.strategies)} live: {', '.join(mgr.strategies)}")
    except Exception as e:
        r.add(BLOCK, 'strategies loaded', False, f'StrategyManager raised: {e}')


def check_api(r: Readiness) -> None:
    """Without the API there is no dashboard, and no way to watch the run.

    Two separate failures, because the first hides the second. The module can
    import cleanly while its auth layer did not, in which case the API starts,
    reports healthy, and answers every protected route with 503. The login
    page simply never lets anyone in, and the only clue is one CRITICAL line
    in a log nobody reads three weeks later.
    """
    try:
        import src.api.api_server as api
        r.add(BLOCK, 'API server importable', True)
    except Exception as e:
        r.add(BLOCK, 'API server importable', False,
              f'{e}. The dashboard will not start, leaving no view of the run.')
        return

    auth_ok = bool(getattr(api, 'AUTH_AVAILABLE', False))
    r.add(WARN, 'dashboard login works', auth_ok,
          '' if auth_ok else
          'the auth module did not load, usually because SECRET_KEY is unset. '
          'The API starts and looks healthy but answers every protected route '
          'with 503, so nobody can log in to watch the run. The journals still '
          'record everything; read them with scripts/paper_run_report.py.')


def check_market_data(cfg: Dict[str, Any], r: Readiness,
                      sample: Optional[Dict[str, Any]] = None) -> None:
    """Real prices, not the synthetic fallback the agent refuses to trade on."""
    if cfg.get('data_manager', {}).get('use_fallback_only'):
        r.add(BLOCK, 'real market data', False,
              'data_manager.use_fallback_only is set. Prices are synthetic and '
              'the agent will refuse to trade on every one of them.')
        return
    if sample is None:
        r.add(WARN, 'real market data', False,
              'not probed; run with --probe-data to confirm vendors respond')
        return
    fallback = [s for s, d in sample.items()
                if isinstance(d, dict) and d.get('source') == 'fallback']
    r.add(BLOCK, 'real market data', len(fallback) < len(sample),
          f"{len(fallback)}/{len(sample)} symbols returned synthetic fallback "
          f"prices ({', '.join(sorted(fallback)[:5])}). The agent will not "
          f"trade those." if fallback else f"{len(sample)} symbols priced")


def history_needed(cfg: Dict[str, Any]) -> int:
    """Daily bars the enabled technical strategies need before they can signal."""
    needs = [int(s.get('lookback_period', 50) or 50)
             for s in (cfg.get('strategies') or {}).values()
             if isinstance(s, dict) and s.get('enabled') and s.get('type', 'technical') == 'technical']
    return max(needs) if needs else 50


def check_history(cfg: Dict[str, Any], r: Readiness,
                  history_depth: Optional[Dict[str, int]] = None) -> None:
    """Can the strategies start with real daily history?

    The live agent needs this many completed daily bars per symbol before it
    can signal at all. The warm-start that supplied them had never run in
    production, and this gate did not notice: its signal-path check feeds
    daily bars from disk, which proves the strategies work on daily data but
    says nothing about whether production would have any.
    """
    need = history_needed(cfg)
    if history_depth is None:
        r.add(WARN, 'daily history for warm-start', False,
              f'not probed; run with --probe-data to confirm each symbol gets {need} daily bars')
        return
    dm = cfg.get('data_manager', {})
    live = list(dm.get('symbols', []) or [])
    nse = list(dm.get('nse_symbols', []) or [])
    short_live = {s: history_depth.get(s, 0) for s in live if history_depth.get(s, 0) < need}
    ready = len(live) - len(short_live)
    if live and ready == 0:
        r.add(BLOCK, 'daily history for warm-start', False,
              f'no US/crypto symbol got {need} daily bars from the history source, so no '
              f'strategy can signal for {need} trading days: {short_live}')
    else:
        r.add(WARN, 'daily history for warm-start', not short_live,
              f'{ready}/{len(live)} US/crypto symbols have {need}+ daily bars'
              + (f'; short: {short_live}' if short_live else ''))
    short_nse = {s: history_depth.get(s, 0) for s in nse if history_depth.get(s, 0) < need}
    if nse:
        r.add(WARN, 'NSE real daily history', not short_nse,
              f'{len(nse) - len(short_nse)}/{len(nse)} NSE symbols have {need}+ real daily bars'
              + (f'; the rest cannot signal until real history accumulates '
                 f'(about one bar per trading day): {short_nse}' if short_nse else ''))


def check_broker(cfg: Dict[str, Any], r: Readiness, broker=None) -> None:
    """Paper mode, actually creatable, actually the configured one, connected.

    This used to stop at the config's stated intent, which is how it passed a
    run whose execution venue was not the configured one: config/config.json
    marks Alpaca primary, the Alpaca connector could not import, BrokerManager
    dropped it with a warning, and the internal simulator took every order.
    The gate certified what the file said rather than what the process built.
    """
    try:
        from src.agent.broker_manager import available_broker_types, broker_is_enabled
    except Exception as e:
        r.add(BLOCK, 'broker layer importable', False,
              f'src.agent.broker_manager could not be imported: {e}')
        return

    configured = cfg.get('brokers') or {}
    enabled = {n: b for n, b in configured.items() if broker_is_enabled(b)}

    live = [n for n, b in enabled.items()
            if b.get('paper') is False or b.get('is_paper') is False]
    r.add(BLOCK, 'all enabled brokers are paper mode', not live,
          f"{live} would trade real money" if live else '')

    available = available_broker_types()
    unavailable = {n: (b.get('type') or '') for n, b in enabled.items()
                   if (b.get('type') or '').lower() not in available}
    r.add(BLOCK, 'every enabled broker can be created', not unavailable,
          f"{unavailable} name broker types this process cannot build "
          f"(available: {sorted(available)}); the connector import failed"
          if unavailable else f"{len(enabled)} enabled broker(s)")

    primary = sorted(n for n, b in configured.items() if b.get('primary', False))
    broken_primary = [n for n in primary if n not in enabled or n in unavailable]
    r.add(BLOCK, 'primary broker is the configured one', not broken_primary,
          f"{broken_primary} marked primary but disabled or unavailable, so "
          f"orders would be routed to a different venue"
          if broken_primary else (f"{primary[0]}" if primary else 'none marked primary'))

    if broker is None:
        r.add(WARN, 'broker connected', False,
              'not probed; the run cannot fill orders without a live broker')
        return
    connected = bool(getattr(broker, 'is_connected', False))
    r.add(BLOCK, 'broker connected', connected,
          '' if connected else
          f'orders cannot be submitted (probe returned {type(broker).__name__})')


def check_capacity(cfg: Dict[str, Any], r: Readiness,
                   signal_rate: float = 0.439,
                   per_symbol_seconds: float = 0.0054) -> None:
    """Validation demand inside the quota, including the slow-pass burst."""
    from src.agent.capacity import DEFAULT_PROVIDER_LIMITS, assess
    dm = cfg.get('data_manager', {})
    loop = len({s.upper() for s in dm.get('symbols', []) or []})
    burst = len({s.upper() for s in dm.get('nse_symbols', []) or []})
    provider = cfg.get('primary_llm_provider', 'gemini')
    limits = {**DEFAULT_PROVIDER_LIMITS,
              **(cfg.get('capacity', {}).get('provider_limits', {}))}

    report = assess(loop, per_symbol_seconds,
                    cycle_budget_seconds=float(cfg.get('trading_loop_interval', 60)),
                    signal_rate=signal_rate,
                    llm_limit_per_minute=limits.get(provider, 10),
                    llm_enabled=cfg.get('llm_enabled', True),
                    burst_symbols=burst,
                    burst_interval_seconds=float(dm.get('nse_eval_interval', 1800)))
    r.add(WARN, 'LLM validation within quota', not report.findings,
          ' '.join(report.findings))
    r.add(BLOCK, 'cycle fits its budget', report.within_budget,
          f'{report.cycle_seconds:.1f}s against a '
          f'{report.budget_seconds:.0f}s budget')


def check_journals(r: Readiness, data_dir: Optional[str] = None) -> None:
    """The run's only durable record. Unwritable journals mean no evidence."""
    from src.utils.paths import DATA_DIR
    path = data_dir or str(DATA_DIR)
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, '.write_probe')
        with open(probe, 'w', encoding='utf-8') as f:
            f.write('ok')
        os.remove(probe)
        r.add(BLOCK, 'journal directory writable', True, path)
    except Exception as e:
        r.add(BLOCK, 'journal directory writable', False, f'{path}: {e}')


def check_costs(cfg: Dict[str, Any], r: Readiness) -> None:
    """Unverified fees make the run's P&L provisional, not wrong."""
    from src.agent.cost_model import classify, unverified_markets
    dm = cfg.get('data_manager', {})
    traded = {s.upper() for s in
              (dm.get('symbols', []) or []) + (dm.get('nse_symbols', []) or [])}
    active = {classify(s, cfg) for s in traded}
    unverified = {m: n for m, n in unverified_markets(cfg).items() if m in active}
    r.add(WARN, 'transaction costs verified', not unverified,
          f"placeholder fees for {', '.join(sorted(unverified))}; P&L from "
          f"those markets is provisional" if unverified else '')


def check_risk_settings(cfg: Dict[str, Any], r: Readiness) -> None:
    """Stops wide enough to survive ordinary noise."""
    limits = cfg.get('risk_limits', {})
    has_atr = bool(limits.get('trailing_stop_atr_mult'))
    r.add(WARN, 'stops scale with volatility', has_atr,
          '' if has_atr else
          'risk_limits has no trailing_stop_atr_mult, so a fixed percentage '
          'applies to every instrument regardless of its own volatility')



def check_signal_path(cfg: Dict[str, Any], r: Readiness, bars=None,
                      min_confidence: float = 0.1) -> None:
    """The only check that answers the question directly: will it trade?

    Everything else here is a precondition. This one drives the real
    StrategyManager over real bars and counts signals that would clear the
    live loop's gate, which is `action != 'hold'` AND `confidence > 0.1` AND
    `position_size > 0` (src/agent/main.py). All three, because a
    configuration can satisfy two and still place no orders for ninety days.

    Needs historical bars, so it is a warning when none are supplied rather
    than a silent pass: an unanswered question is not a good answer.
    """
    if not bars:
        r.add(WARN, 'strategies produce actionable signals', False,
              'not tested; pass --csv-dir with historical bars to confirm '
              'this configuration would place any trade at all')
        return
    try:
        from src.agent.strategy_manager import StrategyManager
        mgr = StrategyManager(cfg)
    except Exception as e:
        r.add(BLOCK, 'strategies produce actionable signals', False,
              f'StrategyManager raised: {e}')
        return

    evaluated = 0
    actionable = 0
    per_symbol: Dict[str, int] = {}
    for symbol, df in bars.items():
        for ts, row in df.iterrows():
            price = float(row['close'])
            if price <= 0:
                continue
            sig = mgr.generate_signals({'symbol': symbol, 'price': price,
                                        'close': price,
                                        'timestamp': str(ts)}) or {}
            evaluated += 1
            if (sig.get('action', 'hold') != 'hold'
                    and float(sig.get('confidence', 0.0) or 0.0) > min_confidence
                    and float(sig.get('position_size', 0.0) or 0.0) > 0):
                actionable += 1
                per_symbol[symbol] = per_symbol.get(symbol, 0) + 1

    if not evaluated:
        r.add(WARN, 'strategies produce actionable signals', False,
              'no usable bars in the sample')
        return
    rate = actionable / evaluated
    r.add(BLOCK, 'strategies produce actionable signals', actionable > 0,
          f'{actionable} of {evaluated} bars cleared the live gate '
          f'({rate:.1%}), across {len(per_symbol)}/{len(bars)} symbols'
          if actionable else
          f'0 of {evaluated} bars produced a signal that clears '
          f"action!=hold, confidence>{min_confidence}, position_size>0. "
          f'This configuration would hold for the entire run.')



def check_health_endpoint(cfg: Dict[str, Any], r: Readiness) -> None:
    """Something must be able to say the agent is still alive.

    Over ninety days the process will be restarted, the machine will sleep,
    and a dependency will break. None of that matters if /health answers;
    all of it is invisible if it does not.
    """
    mon = cfg.get('monitoring', {}) or {}
    enabled = mon.get('enabled', True)
    r.add(WARN, 'liveness endpoint enabled', bool(enabled),
          f"/health will serve on port {mon.get('port', 8080)}" if enabled else
          'monitoring.enabled is false, so nothing will report whether the '
          'agent is still running. Over a long run that is the difference '
          'between a quiet week and a process that died on day three.')


def assess_readiness(cfg: Dict[str, Any], *, broker=None,
                     data_sample: Optional[Dict[str, Any]] = None,
                     data_dir: Optional[str] = None,
                     bars=None,
                     check_api_import: bool = True,
                     history_depth: Optional[Dict[str, int]] = None) -> Readiness:
    """Every check, in the order a reader would want them."""
    r = Readiness()
    check_strategies(cfg, r)
    check_signal_path(cfg, r, bars)
    check_history(cfg, r, history_depth)
    if check_api_import:
        check_api(r)
    check_market_data(cfg, r, data_sample)
    check_broker(cfg, r, broker)
    check_capacity(cfg, r)
    check_journals(r, data_dir)
    check_health_endpoint(cfg, r)
    check_costs(cfg, r)
    check_risk_settings(cfg, r)
    return r
