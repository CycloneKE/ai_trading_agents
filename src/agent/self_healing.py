"""Self-healing: notice what broke, fix what is safe to fix, tell a person the rest.

What the agent already did on its own: reconnect a dropped broker, fail over
to a backup, refuse to trade on fake prices, and halt (the "kill switch")
when its main loop froze. What it did not do:

- restart a background worker that died (the price feed, the NSE scraper, the
  dashboard's server): the loop would carry on with nothing feeding it;
- take a halt back once the freeze that caused it was over: every halt waited
  for a person, even a three-minute stall at 03:00;
- notice that the loop was failing again and again, or that a price feed had
  been down for an hour, or that the whole process had been killed and
  restarted by Docker;
- tell anyone.

What this adds, and what it will never do:

| Problem | What happens |
|---|---|
| A watched background thread dies | It is restarted, at most `max_restarts_per_hour` times; then the operator is told it could not be fixed |
| A halt caused by a frozen loop (HEARTBEAT_TIMEOUT) | Trading resumes by itself once the loop and the broker have been healthy for `healthy_seconds`, at most `max_per_day` times a day |
| Any other halt (a loss limit, the operator's own button, an unknown reason) | Never cleared by the agent. The operator is emailed |
| The trading loop raises an error again and again | Emailed at 3 in a row, again at 10 (critical) and every 30 after |
| A symbol has no real price for `degraded_symbol_minutes` | Emailed, with the symbols |
| The process was killed rather than stopped cleanly | Emailed on the next start |

It only ever un-halts a halt the agent itself caused by freezing, and never
places or changes an order. Everything it does is recorded in the alert log.
"""
import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional

from src.utils.paths import DATA_DIR

logger = logging.getLogger(__name__)

# Skip reasons that mean the agent could not get the facts it needs, not that
# it looked and found nothing. The same reasons the weekly paper-run report
# labels DEGRADED or "check".
DEGRADED_REASONS = frozenset({
    'fallback_price', 'no_price', 'stale_price', 'no_account_info',
    'position_unknown', 'no_broker'})

# The reason the watchdog gives when the loop froze; the only halt the agent
# will ever clear by itself.
HEARTBEAT_HALT = 'HEARTBEAT_TIMEOUT'

DEFAULTS: Dict[str, Any] = {
    'enabled': True,
    'check_interval_seconds': 30,
    'max_restarts_per_hour': 6,
    'auto_resume': {'enabled': True, 'healthy_seconds': 600, 'max_per_day': 3},
    'loop_error_alert_at': [3, 10],
    'loop_error_repeat_every': 30,
    'degraded_symbol_minutes': 30,
}


def settings(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = (config or {}).get('self_healing') or {}
    out = {**DEFAULTS, **{k: v for k, v in cfg.items() if not k.startswith('_')}}
    out['auto_resume'] = {**DEFAULTS['auto_resume'], **(cfg.get('auto_resume') or {})}
    return out


def _now_iso(clock: Callable[[], float]) -> str:
    return datetime.fromtimestamp(clock(), tz=timezone.utc).isoformat()


# ---------------------------------------------------------------- supervisor

class _Watched:
    def __init__(self, name: str, is_alive: Callable[[], bool], restart: Callable[[], Any]):
        self.name, self.is_alive, self.restart = name, is_alive, restart
        self.restarts: List[float] = []
        self.gave_up = False
        self.dead_since: Optional[float] = None


class Supervisor:
    """Restarts background workers that have died."""

    def __init__(self, alerter=None, max_restarts_per_hour: int = 6,
                 clock: Callable[[], float] = time.time):
        self.alerter = alerter
        self.max_restarts = int(max_restarts_per_hour)
        self._clock = clock
        self._watched: Dict[str, _Watched] = {}
        self.events: Deque[Dict[str, Any]] = deque(maxlen=100)

    def watch(self, name: str, is_alive: Callable[[], bool], restart: Callable[[], Any]) -> None:
        self._watched[name] = _Watched(name, is_alive, restart)

    def _event(self, name: str, what: str, detail: str = '') -> None:
        self.events.append({'ts': _now_iso(self._clock), 'worker': name, 'event': what, 'detail': detail})

    def _alert(self, key: str, subject: str, body: str, severity: str) -> None:
        if self.alerter is not None:
            self.alerter.send(key, subject, body, severity)

    def check_once(self) -> List[Dict[str, Any]]:
        """Look at every watched worker once; restart the dead. Returns the
        events this pass produced."""
        before = len(self.events)
        now = self._clock()
        for w in self._watched.values():
            try:
                alive = bool(w.is_alive())
            except Exception as e:                       # cannot tell: leave it alone
                logger.debug(f"Could not check {w.name}: {e}")
                continue
            if alive:
                w.dead_since, w.gave_up = None, False
                continue
            w.dead_since = w.dead_since or now
            w.restarts = [t for t in w.restarts if now - t < 3600]
            if len(w.restarts) >= self.max_restarts:
                if not w.gave_up:
                    w.gave_up = True
                    self._event(w.name, 'gave_up', f'{len(w.restarts)} restarts in the last hour')
                    self._alert(f'heal:{w.name}:gave_up', f'{w.name} keeps dying and could not be kept running',
                                f"The background worker '{w.name}' stopped {len(w.restarts)} times in the last hour "
                                f"and was restarted each time. The agent will not try again until an hour has "
                                f"passed. Look at the logs for what is killing it.", 'critical')
                continue
            w.restarts.append(now)
            try:
                w.restart()
            except Exception as e:
                self._event(w.name, 'restart_failed', str(e)[:200])
                logger.error(f"Restarting {w.name} failed: {e}")
                self._alert(f'heal:{w.name}:restart_failed', f'{w.name} died and could not be restarted',
                            f"The background worker '{w.name}' stopped and restarting it failed: {e}", 'critical')
                continue
            self._event(w.name, 'restarted')
            logger.warning(f"Self-healing: restarted {w.name}")
            self._alert(f'heal:{w.name}:restarted', f'{w.name} died and was restarted',
                        f"The background worker '{w.name}' had stopped. It was restarted automatically. "
                        f"No action is needed unless this repeats.", 'warning')
        return list(self.events)[before:]

    def status(self) -> List[Dict[str, Any]]:
        now = self._clock()
        out = []
        for w in self._watched.values():
            try:
                alive = bool(w.is_alive())
            except Exception:
                alive = None
            out.append({'worker': w.name, 'alive': alive,
                        'restarts_last_hour': sum(1 for t in w.restarts if now - t < 3600),
                        'gave_up': w.gave_up})
        return out


# --------------------------------------------------------------- auto resume

class AutoResume:
    """When may the agent lift a halt itself? Only a halt it caused by
    freezing, and only after a stretch of proof that the freeze is over."""

    def __init__(self, enabled: bool = True, healthy_seconds: float = 600, max_per_day: int = 3,
                 clock: Callable[[], float] = time.time):
        self.enabled = enabled
        self.healthy_seconds = float(healthy_seconds)
        self.max_per_day = int(max_per_day)
        self._clock = clock
        self._healthy_since: Optional[float] = None
        self._resumes: List[float] = []

    @staticmethod
    def eligible(reason: Optional[str]) -> bool:
        return str(reason or '').startswith(HEARTBEAT_HALT)

    def evaluate(self, halted: bool, reason: Optional[str], loop_healthy: bool,
                 brokers_connected: bool) -> str:
        """One of: 'idle' (not halted), 'not_eligible' (a halt only a person may
        lift), 'wait' (eligible, not yet proven healthy), 'limit' (already
        resumed the most times allowed today) or 'resume'."""
        if not halted:
            self._healthy_since = None
            return 'idle'
        if not self.enabled or not self.eligible(reason):
            return 'not_eligible'
        now = self._clock()
        if not (loop_healthy and brokers_connected):
            self._healthy_since = None
            return 'wait'
        self._healthy_since = self._healthy_since or now
        if now - self._healthy_since < self.healthy_seconds:
            return 'wait'
        self._resumes = [t for t in self._resumes if now - t < 86400]
        if len(self._resumes) >= self.max_per_day:
            return 'limit'
        self._resumes.append(now)
        self._healthy_since = None
        return 'resume'


# ------------------------------------------------------------- loop errors

class LoopErrorTracker:
    """Counts consecutive failures of the trading loop and says when to tell someone."""

    def __init__(self, alerter=None, alert_at=(3, 10), repeat_every: int = 30):
        self.alerter = alerter
        self.alert_at = tuple(alert_at)
        self.repeat_every = int(repeat_every)
        self.consecutive = 0
        self.last_error: Optional[str] = None

    def failed(self, error: BaseException) -> None:
        self.consecutive += 1
        self.last_error = f"{type(error).__name__}: {error}"[:300]
        n = self.consecutive
        if n in self.alert_at or (self.repeat_every and n > max(self.alert_at, default=0)
                                  and n % self.repeat_every == 0):
            if self.alerter is not None:
                severity = 'critical' if n >= 10 else 'warning'
                self.alerter.send(
                    f'heal:loop_errors:{n}', f'the trading loop has failed {n} times in a row',
                    f"Each pass of the trading loop is raising an error, so nothing is being decided. "
                    f"Latest error: {self.last_error}\n\nThe loop keeps retrying every cycle. "
                    f"Look at the logs; a restart may clear it.", severity, cooldown_s=0)

    def ok(self) -> None:
        if self.consecutive >= min(self.alert_at, default=3) and self.alerter is not None:
            self.alerter.send('heal:loop_recovered', 'the trading loop is running normally again',
                              f"It had failed {self.consecutive} times in a row and has now completed a full pass.",
                              'info', cooldown_s=0)
        self.consecutive = 0


# ------------------------------------------------------------- degraded data

class DegradationWatch:
    """Tells the operator when symbols have had no real price for a while."""

    def __init__(self, alerter=None, minutes: float = 30, clock: Callable[[], float] = time.time):
        self.alerter = alerter
        self.after = float(minutes) * 60
        self._clock = clock
        self._since: Dict[str, float] = {}

    def stuck(self) -> Dict[str, float]:
        """Symbols degraded for at least the alert threshold, with minutes."""
        now = self._clock()
        return {s: round((now - t) / 60, 1) for s, t in sorted(self._since.items())
                if now - t >= self.after}

    def observe(self, decisions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Feed one cycle's decisions (symbol -> record with a skip_reason)."""
        now = self._clock()
        for sym, d in (decisions or {}).items():
            if (d or {}).get('skip_reason') in DEGRADED_REASONS:
                self._since.setdefault(sym, now)
            else:
                self._since.pop(sym, None)
        stuck = self.stuck()
        if stuck and self.alerter is not None:
            listing = ', '.join(f"{s} ({m:.0f} min)" for s, m in list(stuck.items())[:12])
            self.alerter.send(
                'heal:degraded_prices', f'{len(stuck)} symbol(s) have had no real price for over '
                f'{self.after / 60:.0f} minutes',
                f"The agent refuses to trade on made-up prices, so it is not trading these: {listing}.\n\n"
                f"This usually means the price vendor (Yahoo Finance) is throttling or down, or the symbol "
                f"is wrong. It clears by itself when prices return.", 'warning', cooldown_s=4 * 3600)
        return stuck


# ----------------------------------------------------------------- run state

class RunState:
    """Remembers whether the last run ended cleanly, so a crash is noticed."""

    def __init__(self, path: Optional[str] = None, clock: Callable[[], float] = time.time):
        self.path = str(path or DATA_DIR / 'run_state.json')
        self._clock = clock

    def _read(self) -> Dict[str, Any]:
        try:
            with open(self.path, encoding='utf-8') as f:
                return json.load(f) or {}
        except (OSError, ValueError):
            return {}

    def _write(self, data: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
            tmp = self.path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except OSError as e:
            logger.warning(f"Could not save the run state: {e}")

    def begin(self, version: str = '') -> Optional[Dict[str, Any]]:
        """Mark a run started. Returns what the previous run left behind if it
        did not end cleanly (killed, out of memory, power loss), else None."""
        previous = self._read()
        self._write({'state': 'running', 'started_at': _now_iso(self._clock),
                     'last_seen': _now_iso(self._clock), 'version': version})
        return previous if previous.get('state') == 'running' else None

    def touch(self) -> None:
        data = self._read()
        if data.get('state') == 'running':
            data['last_seen'] = _now_iso(self._clock)
            self._write(data)

    def end_clean(self) -> None:
        data = self._read()
        data.update({'state': 'stopped', 'stopped_at': _now_iso(self._clock)})
        self._write(data)


# -------------------------------------------------------------- the whole

class SelfHealing:
    """Wires the pieces to a running TradingAgent."""

    def __init__(self, agent, alerter, config: Optional[Dict[str, Any]] = None,
                 clock: Callable[[], float] = time.time, run_state: Optional[RunState] = None):
        self.agent = agent
        self.alerter = alerter
        self.cfg = settings(config)
        self._clock = clock
        self.supervisor = Supervisor(alerter, self.cfg['max_restarts_per_hour'], clock)
        ar = self.cfg['auto_resume']
        self.auto_resume = AutoResume(ar['enabled'], ar['healthy_seconds'], ar['max_per_day'], clock)
        self.loop_errors = LoopErrorTracker(alerter, self.cfg['loop_error_alert_at'],
                                            self.cfg['loop_error_repeat_every'])
        self.degradation = DegradationWatch(alerter, self.cfg['degraded_symbol_minutes'], clock)
        self.run_state = run_state or RunState(clock=clock)
        self.resumes: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle
    def start(self, version: str = '') -> None:
        if not self.cfg['enabled']:
            return
        crashed = self.run_state.begin(version)
        if crashed:
            self.alerter.send(
                'heal:unclean_restart', 'the agent was restarted after it was killed, not stopped',
                f"The previous run did not shut down cleanly. It last reported in at "
                f"{crashed.get('last_seen', 'an unknown time')} (version {crashed.get('version') or 'unknown'}). "
                f"That usually means the container was killed (out of memory, a crash, or the host restarting). "
                f"The agent has started again by itself; check the logs from around that time.",
                'warning', cooldown_s=0)
        self._thread = threading.Thread(target=self._run, daemon=True, name='self-healing')
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.run_state.end_clean()

    def _run(self) -> None:
        while not self._stop.wait(float(self.cfg['check_interval_seconds'])):
            try:
                self.tick()
            except Exception as e:
                logger.error(f"Self-healing pass failed: {e}")

    # -- one pass
    def tick(self) -> None:
        self.supervisor.check_once()
        self._maybe_resume()
        self.run_state.touch()

    def _maybe_resume(self) -> None:
        agent = self.agent
        hb = getattr(agent, 'heartbeat_monitor', None)
        loop_healthy = bool(hb.is_healthy()) if hb is not None and hasattr(hb, 'is_healthy') else False
        bm = (getattr(agent, 'components', None) or {}).get('broker_manager')
        primary = bm.get_broker() if bm is not None else None
        connected = bool(primary is not None and getattr(primary, 'is_connected', False))
        verdict = self.auto_resume.evaluate(bool(getattr(agent, 'trading_halted', False)),
                                            getattr(agent, 'halt_reason', None), loop_healthy, connected)
        if verdict == 'resume':
            reason = getattr(agent, 'halt_reason', '')
            agent.resume_trading(by='the agent itself (the freeze that halted it is over)')
            self.resumes.append({'ts': _now_iso(self._clock), 'reason': reason})
            del self.resumes[:-20]
            self.alerter.send(
                'heal:auto_resumed', 'trading was halted by a freeze and has resumed by itself',
                f"The agent halted itself because its loop froze ({reason}). The loop and the brokers have "
                f"now been healthy for {self.auto_resume.healthy_seconds / 60:.0f} minutes, so it lifted the halt. "
                f"Orders that were open when it halted were cancelled and will be decided afresh. "
                f"If this repeats, something is repeatedly stalling the agent.", 'warning', cooldown_s=0)
        elif verdict == 'limit':
            self.alerter.send(
                'heal:resume_limit', 'trading is halted and the agent will not restart it again today',
                f"The agent has already lifted a freeze-caused halt {self.auto_resume.max_per_day} times in the last "
                f"day, so it is leaving this one for you. Press Resume on the dashboard once you have looked "
                f"at why the loop keeps freezing.", 'critical', cooldown_s=6 * 3600)

    # -- hooks the agent calls
    def note_halt(self, reason: str, flatten: bool = False) -> None:
        """The agent was halted. Tell the operator unless they did it themselves."""
        if not self.cfg['enabled'] or 'from the dashboard' in str(reason) or 'operator request' in str(reason):
            return
        if self.auto_resume.enabled and AutoResume.eligible(reason):
            after = (f"This kind of halt lifts by itself once the loop and brokers have been healthy for "
                     f"{self.auto_resume.healthy_seconds / 60:.0f} minutes, and you will get another email when it does.")
        else:
            after = ("The agent will NOT lift this halt by itself. When you are satisfied it is safe, open the "
                     "dashboard and press Resume.")
        self.alerter.send('heal:halt', 'trading has been halted', f"Reason: {reason}\n\n{after}",
                          'critical', cooldown_s=0)

    def loop_ok(self) -> None:
        self.loop_errors.ok()

    def loop_failed(self, error: BaseException) -> None:
        self.loop_errors.failed(error)

    def observe_cycle(self, decisions: Dict[str, Dict[str, Any]]) -> None:
        self.degradation.observe(decisions)

    def status(self) -> Dict[str, Any]:
        agent = self.agent
        return {
            'enabled': self.cfg['enabled'],
            'halted': bool(getattr(agent, 'trading_halted', False)),
            'halt_reason': getattr(agent, 'halt_reason', None),
            'halt_clears_itself': bool(getattr(agent, 'trading_halted', False))
                                  and self.auto_resume.enabled and AutoResume.eligible(getattr(agent, 'halt_reason', None)),
            'workers': self.supervisor.status(),
            'recent_events': list(self.supervisor.events)[-10:],
            'auto_resumes': self.resumes[-5:],
            'consecutive_loop_errors': self.loop_errors.consecutive,
            'last_loop_error': self.loop_errors.last_error,
            'symbols_without_price': self.degradation.stuck(),
        }
