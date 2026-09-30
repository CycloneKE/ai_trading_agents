"""Email alerts and self-healing (src/agent/alerts.py, src/agent/self_healing.py).

The rules that matter most, each tested: an alert never raises and never
blocks; the same alert does not flood; a dead worker is restarted but not
forever; and the agent lifts only a halt its own freeze caused, only after a
stretch of proof, and never one a person or a loss limit made.
"""
import threading
from types import SimpleNamespace

import pytest

from src.agent.alerts import AlertLog, EmailAlerter
from src.agent.self_healing import (AutoResume, DegradationWatch, LoopErrorTracker,
                                    RunState, SelfHealing, Supervisor, settings)

ENV = {'ALERT_EMAIL_TO': 'boss@example.com, second@example.com', 'SMTP_HOST': 'smtp.example.com',
       'SMTP_USER': 'agent@example.com', 'SMTP_PASSWORD': 'secret'}


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def alerter(tmp_path, env=ENV, transport=None, clock=None, **cfg):
    sent = []
    a = EmailAlerter({'alerts': cfg} if cfg else {}, env=env, log=AlertLog(tmp_path / 'alerts.jsonl'),
                     transport=transport or (lambda s, m: sent.append(m)), background=False,
                     clock=clock or Clock())
    a.sent = sent
    return a


# ------------------------------------------------------------------ email

def test_an_alert_is_emailed_to_every_recipient_with_the_severity_in_the_subject(tmp_path):
    a = alerter(tmp_path)
    assert a.configured
    assert a.send('halt', 'trading has been halted', 'Reason: test', 'critical') == 'sent'
    msg = a.sent[0]
    assert msg['Subject'] == '[Trading Agent] CRITICAL: trading has been halted'
    assert msg['To'] == 'boss@example.com, second@example.com' and msg['From'] == 'agent@example.com'
    assert 'Reason: test' in msg.get_content()
    assert 'secret' not in msg.as_string()                      # the password never leaves the settings


def test_without_email_settings_the_alert_is_recorded_and_says_so(tmp_path):
    a = alerter(tmp_path, env={})
    assert not a.configured
    assert a.send('halt', 'x', severity='critical') == 'not_configured'
    assert a.sent == []
    assert a.log.recent()[0]['status'] == 'not_configured'
    assert a.status()['configured'] is False


def test_the_same_alert_does_not_repeat_within_its_cooldown_but_a_different_one_does(tmp_path):
    clock = Clock()
    a = alerter(tmp_path, clock=clock)
    assert a.send('feed', 'feed down', severity='warning') == 'sent'
    clock.advance(600)
    assert a.send('feed', 'feed down', severity='warning') == 'suppressed'
    assert a.send('other', 'something else', severity='warning') == 'sent'
    clock.advance(3600)
    assert a.send('feed', 'feed down', severity='warning') == 'sent'       # an hour on: worth repeating
    assert len(a.sent) == 3
    assert [e['status'] for e in a.log.recent()] == ['sent', 'sent', 'sent']     # a repeat is not logged again


def test_non_critical_mail_is_capped_per_hour_but_critical_always_goes(tmp_path):
    a = alerter(tmp_path, max_per_hour=3)
    for i in range(6):
        a.send(f'k{i}', f'warning {i}', severity='warning')
    assert len(a.sent) == 3
    assert a.send('boom', 'critical', severity='critical') == 'sent'
    assert len(a.sent) == 4


def test_a_broken_mail_server_is_recorded_and_never_raises(tmp_path):
    def boom(settings_, message):
        raise ConnectionRefusedError('no route to host')
    a = alerter(tmp_path, transport=boom)
    assert a.send('halt', 'halted', severity='critical') == 'sent'          # it tried
    assert a.last_error and 'no route to host' in a.last_error
    assert a.log.recent()[0]['status'] == 'failed'
    assert a.status()['failed_24h'] == 1


def test_sending_does_not_block_the_caller(tmp_path):
    release = threading.Event()
    a = EmailAlerter({}, env=ENV, log=AlertLog(tmp_path / 'a.jsonl'), background=True,
                     transport=lambda s, m: release.wait(5))
    try:
        assert a.send('slow', 'slow server', severity='critical') == 'sent'   # returns at once
    finally:
        release.set()


def test_port_465_means_ssl_and_the_status_hides_the_address(tmp_path):
    a = alerter(tmp_path, env={**ENV, 'SMTP_PORT': '465'})
    assert a.settings['security'] == 'ssl' and a.settings['port'] == 465
    assert alerter(tmp_path, env={**ENV, 'SMTP_PORT': 'oops'}).settings['port'] == 587
    assert a.status()['recipients'] == ['b***@example.com', 's***@example.com']


# -------------------------------------------------------------- supervisor

def test_a_dead_worker_is_restarted_and_the_operator_is_told(tmp_path):
    alive = {'v': False}
    restarts = []
    a = alerter(tmp_path)
    sup = Supervisor(a, max_restarts_per_hour=3, clock=Clock())
    sup.watch('price feed', lambda: alive['v'], lambda: restarts.append(1) or alive.update(v=True))
    events = sup.check_once()
    assert restarts == [1] and [e['event'] for e in events] == ['restarted']
    assert 'price feed died and was restarted' in a.sent[0]['Subject']
    assert sup.check_once() == []                                # alive now: nothing to do


def test_a_worker_that_keeps_dying_is_given_up_on_and_that_is_critical(tmp_path):
    clock = Clock()
    a = alerter(tmp_path, clock=clock)
    starts = []
    sup = Supervisor(a, max_restarts_per_hour=2, clock=clock)
    sup.watch('scraper', lambda: False, lambda: starts.append(1))
    for _ in range(5):
        sup.check_once()
        clock.advance(60)
    assert len(starts) == 2                                       # not forever
    assert any('CRITICAL' in m['Subject'] and 'keeps dying' in m['Subject'] for m in a.sent)
    clock.advance(3600)
    sup.check_once()
    assert len(starts) == 3                                       # an hour later it may try again


def test_a_restart_that_raises_is_critical_and_does_not_stop_the_others(tmp_path):
    a = alerter(tmp_path)
    ran = []
    sup = Supervisor(a, clock=Clock())
    sup.watch('api', lambda: False, lambda: (_ for _ in ()).throw(OSError('port in use')))
    sup.watch('feed', lambda: False, lambda: ran.append('feed'))
    sup.check_once()
    assert ran == ['feed']
    assert any('could not be restarted' in m['Subject'] for m in a.sent)


def test_a_check_that_cannot_tell_leaves_the_worker_alone(tmp_path):
    restarts = []
    sup = Supervisor(alerter(tmp_path), clock=Clock())
    sup.watch('x', lambda: 1 / 0, lambda: restarts.append(1))
    sup.check_once()
    assert restarts == []


# ------------------------------------------------------------- auto resume

def resume_policy(**kw):
    clock = Clock()
    return AutoResume(healthy_seconds=600, max_per_day=2, clock=clock, **kw), clock


FROZE = 'HEARTBEAT_TIMEOUT: Agent loop unresponsive for >180s'


def test_a_freeze_halt_is_lifted_only_after_ten_healthy_minutes():
    p, clock = resume_policy()
    assert p.evaluate(True, FROZE, True, True) == 'wait'
    clock.advance(599)
    assert p.evaluate(True, FROZE, True, True) == 'wait'
    clock.advance(2)
    assert p.evaluate(True, FROZE, True, True) == 'resume'


def test_any_unhealthy_moment_restarts_the_ten_minutes():
    p, clock = resume_policy()
    p.evaluate(True, FROZE, True, True)
    clock.advance(500)
    assert p.evaluate(True, FROZE, False, True) == 'wait'         # the loop stalled again
    clock.advance(500)
    assert p.evaluate(True, FROZE, True, True) == 'wait'          # the clock began again just now
    assert p.evaluate(True, FROZE, True, False) == 'wait'         # or the broker is down


@pytest.mark.parametrize('reason', [
    'halted from the dashboard', 'halted from the dashboard and positions closed', 'operator request',
    'daily loss limit breached', 'engaged before the last restart (reason not recorded)', None, ''])
def test_a_halt_a_person_or_a_loss_limit_made_is_never_lifted_by_the_agent(reason):
    p, clock = resume_policy()
    for _ in range(5):
        assert p.evaluate(True, reason, True, True) == 'not_eligible'
        clock.advance(10_000)


def test_it_stops_after_the_daily_limit_and_leaves_the_rest_to_a_person():
    p, clock = resume_policy()
    outcomes = []
    for _ in range(4):
        p.evaluate(True, FROZE, True, True)
        clock.advance(700)
        outcomes.append(p.evaluate(True, FROZE, True, True))
        p.evaluate(False, None, True, True)                       # resumed, then froze again
    assert outcomes == ['resume', 'resume', 'limit', 'limit']
    clock.advance(86400)
    p.evaluate(True, FROZE, True, True)
    clock.advance(700)
    assert p.evaluate(True, FROZE, True, True) == 'resume'        # a day later it may again


def test_it_can_be_switched_off():
    p = AutoResume(enabled=False, clock=Clock())
    assert p.evaluate(True, FROZE, True, True) == 'not_eligible'


# ------------------------------------------------------------ loop errors

def test_repeated_loop_errors_are_reported_at_three_ten_and_every_thirty(tmp_path):
    a = alerter(tmp_path)
    t = LoopErrorTracker(a)
    for _ in range(31):
        t.failed(ValueError('bad data'))
    counts = [m['Subject'].split(' has failed ')[1].split(' times')[0] for m in a.sent]
    assert counts == ['3', '10', '30']
    assert 'CRITICAL' in a.sent[1]['Subject'] and 'WARNING' in a.sent[0]['Subject']
    assert 'ValueError: bad data' in a.sent[0].get_content()


def test_a_recovered_loop_says_so_only_if_it_had_been_reported(tmp_path):
    a = alerter(tmp_path)
    t = LoopErrorTracker(a)
    t.failed(RuntimeError('x'))
    t.ok()
    assert a.sent == []                                            # one blip is not news
    for _ in range(3):
        t.failed(RuntimeError('x'))
    a.sent.clear()
    t.ok()
    assert 'running normally again' in a.sent[0]['Subject'] and t.consecutive == 0


# ------------------------------------------------------------ degraded data

def test_symbols_without_a_real_price_for_thirty_minutes_are_reported_once(tmp_path):
    clock = Clock()
    a = alerter(tmp_path, clock=clock)
    w = DegradationWatch(a, minutes=30, clock=clock)
    bad = {'EUR_USD': {'skip_reason': 'fallback_price'}, 'AAPL': {'skip_reason': 'hold'},
           'SCOM': {'skip_reason': 'no_price'}}
    assert w.observe(bad) == {}
    clock.advance(29 * 60)
    assert w.observe(bad) == {} and a.sent == []
    clock.advance(2 * 60)
    stuck = w.observe(bad)
    assert set(stuck) == {'EUR_USD', 'SCOM'}
    assert 'EUR_USD (31 min)' in a.sent[0].get_content()
    clock.advance(60)
    w.observe(bad)
    assert len(a.sent) == 1                                        # not again for hours
    w.observe({'EUR_USD': {'skip_reason': 'hold'}, 'SCOM': {'skip_reason': 'hold'}})
    assert w.stuck() == {}


def test_a_market_being_closed_is_not_a_degraded_feed(tmp_path):
    clock = Clock()
    w = DegradationWatch(alerter(tmp_path, clock=clock), minutes=1, clock=clock)
    for _ in range(3):
        w.observe({'AAPL': {'skip_reason': 'market_closed'}})
        clock.advance(120)
    assert w.stuck() == {}


# --------------------------------------------------------------- run state

def test_a_run_that_did_not_end_cleanly_is_noticed_on_the_next_start(tmp_path):
    state = RunState(tmp_path / 'run_state.json', clock=Clock())
    assert state.begin('v1') is None                               # the very first run
    state.touch()
    crashed = state.begin('v2')                                    # never called end_clean
    assert crashed and crashed['version'] == 'v1'
    state.end_clean()
    assert state.begin('v3') is None                               # a clean stop is not a crash


# ------------------------------------------------------- the whole, on an agent

class FakeAgent:
    def __init__(self, halted=False, reason=None, primary_up=True, loop_healthy=True):
        self.trading_halted, self.halt_reason = halted, reason
        self.resumed_by = []
        self.components = {'broker_manager': SimpleNamespace(
            get_broker=lambda: SimpleNamespace(is_connected=primary_up))}
        self.heartbeat_monitor = SimpleNamespace(is_healthy=lambda: loop_healthy)

    def resume_trading(self, by='an operator'):
        self.trading_halted, self.halt_reason = False, None
        self.resumed_by.append(by)


def healing(tmp_path, agent, clock=None):
    clock = clock or Clock()
    a = alerter(tmp_path, clock=clock)
    sh = SelfHealing(agent, a, {'self_healing': {'auto_resume': {'healthy_seconds': 600}}}, clock=clock,
                     run_state=RunState(tmp_path / 'rs.json', clock=clock))
    return sh, a, clock


def test_the_agent_resumes_itself_after_a_freeze_and_says_so(tmp_path):
    agent = FakeAgent(True, FROZE)
    sh, a, clock = healing(tmp_path, agent)
    sh.tick()
    clock.advance(601)
    sh.tick()
    assert agent.resumed_by and 'the agent itself' in agent.resumed_by[0] and not agent.trading_halted
    assert 'resumed by itself' in a.sent[0]['Subject']
    assert sh.status()['auto_resumes'][0]['reason'] == FROZE


def test_a_dashboard_halt_is_left_alone_and_not_emailed_to_the_person_who_pressed_it(tmp_path):
    agent = FakeAgent(True, 'halted from the dashboard')
    sh, a, clock = healing(tmp_path, agent)
    sh.note_halt(agent.halt_reason)
    for _ in range(3):
        clock.advance(10_000)
        sh.tick()
    assert agent.trading_halted and a.sent == []


def test_any_other_halt_emails_the_operator_and_says_it_will_not_clear_itself(tmp_path):
    sh, a, _ = healing(tmp_path, FakeAgent(True, 'daily loss limit'))
    sh.note_halt('daily loss limit')
    assert 'CRITICAL' in a.sent[0]['Subject']
    assert 'will NOT lift this halt' in a.sent[0].get_content()
    sh2, a2, _ = healing(tmp_path, FakeAgent(True, FROZE))
    sh2.note_halt(FROZE)
    assert 'lifts by itself' in a2.sent[0].get_content()


def test_no_resume_while_the_primary_broker_is_down(tmp_path):
    agent = FakeAgent(True, FROZE, primary_up=False)
    sh, a, clock = healing(tmp_path, agent)
    sh.tick()
    clock.advance(5000)
    sh.tick()
    assert agent.trading_halted


def test_a_crash_is_reported_when_the_next_run_starts(tmp_path):
    clock = Clock()
    RunState(tmp_path / 'rs.json', clock=clock).begin('old')          # a run that never stopped
    sh, a, _ = healing(tmp_path, FakeAgent(), clock)
    sh.start('new')
    try:
        assert 'killed, not stopped' in a.sent[0]['Subject']
    finally:
        sh.stop()


def test_the_config_can_tune_and_switch_it_off():
    assert settings({})['auto_resume']['healthy_seconds'] == 600
    cfg = settings({'self_healing': {'enabled': False, 'auto_resume': {'max_per_day': 1}, '_comment': 'x'}})
    assert cfg['enabled'] is False and cfg['auto_resume'] == {'enabled': True, 'healthy_seconds': 600, 'max_per_day': 1}


# ------------------------------------------------- wiring inside the agent

def test_halting_the_agent_raises_the_alert_and_resuming_records_who_did_it():
    from src.agent.main import TradingAgent
    noted, stored = [], []
    agent = SimpleNamespace(
        trading_halted=False, halt_reason=None, halted_at=None,
        self_healing=SimpleNamespace(note_halt=lambda reason, flatten=False: noted.append((reason, flatten))),
        risk_manager=SimpleNamespace(set_persistent_kill_switch=lambda on, why: stored.append((on, why))),
        heartbeat_monitor=SimpleNamespace(reset=lambda: None), components={})
    TradingAgent.halt_trading(agent, reason=FROZE)
    assert agent.trading_halted and noted == [(FROZE, False)]
    TradingAgent.resume_trading(agent, by='the agent itself')
    assert not agent.trading_halted and stored[-1] == (False, 'resumed by the agent itself')
    TradingAgent.resume_trading(agent)
    assert stored[-1] == (False, 'resumed by an operator')            # the dashboard button is unchanged


def test_a_watched_thread_that_dies_is_started_again():
    from src.agent.main import TradingAgent
    runs = []
    agent = SimpleNamespace(running=True, self_healing=None)
    agent.self_healing = SimpleNamespace(supervisor=Supervisor(None, clock=Clock()))
    TradingAgent._spawn_watched(agent, 'broker_health', lambda: runs.append(1))   # returns at once: the thread ends
    agent._bg_threads['broker_health'].join(2)
    assert agent.self_healing.supervisor.status()[0]['alive'] is False
    agent.self_healing.supervisor.check_once()
    agent._bg_threads['broker_health'].join(2)
    assert len(runs) == 2
    agent.running = False                                                         # a stopped agent is not "dead"
    assert agent.self_healing.supervisor.status()[0]['alive'] is True


def test_a_lasting_problem_is_one_log_line_with_or_without_email(tmp_path):
    clock = Clock()
    for env in (ENV, {}):
        a = alerter(tmp_path / ('on' if env else 'off'), env=env, clock=clock)
        for _ in range(50):
            a.send('degraded', 'no prices', severity='warning', cooldown_s=4 * 3600)
            clock.advance(60)
        assert len(a.log.recent()) == 1
    off = alerter(tmp_path / 'off', env={}, clock=clock)
    assert off.log.recent()[0]['status'] == 'not_configured'


# ------------------------------------------------------- the test-email command

def test_the_test_email_command_says_what_is_missing_and_what_the_server_said(tmp_path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location('send_test_alert', 'scripts/send_test_alert.py')
    cmd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmd)
    assert cmd.main(env={'SMTP_HOST': 'smtp.example.com'}, log_path=tmp_path / 'a.jsonl') == 1
    out = capsys.readouterr().out
    assert 'ALERT_EMAIL_TO' in out and 'SMTP_USER or SMTP_FROM' in out and 'SMTP_HOST' not in out.split('Missing:')[1]

    def refuse(s, m):
        raise PermissionError('535 authentication failed')
    assert cmd.main(env=ENV, transport=refuse, log_path=tmp_path / 'a.jsonl') == 1
    assert '535 authentication failed' in capsys.readouterr().out

    sent = []
    assert cmd.main(env=ENV, transport=lambda s, m: sent.append(m), log_path=tmp_path / 'a.jsonl') == 0
    assert sent[0]['Subject'].endswith('test alert: email is working') and 'b***@example.com' in capsys.readouterr().out
