"""The herding check (bias_detector.detect_bias). The check it replaced
flagged nearly every buy of a book that only buys, counted one stock's
repeated signal as many votes, and could hold back exits in a sell-off: 216
signals downgraded in the first week of the paper run against 11 orders."""
from datetime import datetime, timedelta

from src.agent.bias_detector import BiasDetector
from src.utils.config_validator import load_config

T0 = datetime(2026, 9, 29, 14, 0)


def detector(**cfg):
    return BiasDetector({**load_config('config/config.json'), **cfg})


def check(bd, symbol, action='buy', at=T0):
    return bd.detect_bias({'action': action, 'confidence': 0.7}, {'symbol': symbol}, {}, symbol=symbol, now=at)


def test_a_steady_stream_of_buys_is_not_bias():
    bd = detector()
    # The old failure: one stock's buy signal re-checked every cycle.
    assert not any(check(bd, 'NVDA', at=T0 + timedelta(minutes=i)) for i in range(30))


def test_a_fourth_new_position_in_one_market_waits_a_day():
    bd = detector()
    assert [check(bd, s) for s in ('NVDA', 'TSLA', 'XLV')] == [False, False, False]
    assert check(bd, 'QQQ') is True                                       # the fourth US buy today
    assert check(bd, 'TSLA') is False                                     # an approved one is not new
    assert check(bd, 'QQQ', at=T0 + timedelta(hours=24, minutes=1)) is False   # a day later


def test_each_market_keeps_its_own_count():
    bd = detector()
    for s in ('NVDA', 'TSLA', 'XLV'):
        check(bd, s)
    assert check(bd, 'BTC-USD') is False and check(bd, 'EUR_USD') is False


def test_sells_are_never_held_back():
    bd = detector()
    assert not any(check(bd, s, action='sell') for s in ('NVDA', 'TSLA', 'XLV', 'QQQ', 'SPY', 'MSFT'))


def test_the_pace_is_set_in_config():
    bd = detector(bias_check={'max_new_buys_per_day': 1})
    assert check(bd, 'NVDA') is False and check(bd, 'TSLA') is True


def test_a_restart_does_not_reset_the_pace(tmp_path):
    from src.agent.order_journal import OrderJournal
    journal = OrderJournal(db_path=str(tmp_path / 'orders.db'))
    for i, sym in enumerate(('NVDA', 'TSLA', 'XLV')):
        journal.record_intent(f'b{i}', sym, 'buy', 1, 'market', strategy='momentum')
    journal.record_intent('c1', 'VOO', 'buy', 1, 'market', strategy='core')     # the core, not a new position
    journal.record_intent('b9', 'BTC-USD', 'buy', 1, 'market', strategy='crypto_trend')
    fresh = detector()                                                     # a new process after a redeploy
    now = datetime.utcnow()
    assert fresh.detect_bias({'action': 'buy'}, {}, {}, symbol='QQQ', now=now, journal=journal) is True
    assert fresh.detect_bias({'action': 'buy'}, {}, {}, symbol='NVDA', now=now, journal=journal) is False
    assert fresh.detect_bias({'action': 'buy'}, {}, {}, symbol='ETH-USD', now=now, journal=journal) is False
    journal.close()
