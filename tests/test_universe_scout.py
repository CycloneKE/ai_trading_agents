from src.agent.universe_scout import propose_candidates, scout_us_candidates


def test_untracked_big_mover_is_proposed():
    movers = [{'symbol': 'BAMB', 'change_pct': 4.2}, {'symbol': 'SCOM', 'change_pct': 5.0}]
    out = propose_candidates(tracked={'SCOM'}, movers=movers, news_texts=[])
    assert out == [{'symbol': 'BAMB', 'reason': 'Moved +4.2% today while untracked'}]


def test_untracked_big_mover_preserves_price_and_volume():
    movers = [{'symbol': 'BAMB', 'change_pct': 4.2, 'price': 35.0, 'volume': 50000}]
    out = propose_candidates(tracked={'SCOM'}, movers=movers, news_texts=[])
    assert len(out) == 1
    assert out[0]['symbol'] == 'BAMB'
    assert out[0]['price'] == 35.0
    assert out[0]['avg_dollar_volume'] == 35.0 * 50000


def test_small_moves_ignored():
    assert propose_candidates({'SCOM'}, [{'symbol': 'BAMB', 'change_pct': 1.0}], []) == []


def test_news_mention_of_untracked_symbol():
    out = propose_candidates({'SCOM'}, [], ['BAMB Cement announces record dividend'],
                             known_universe={'BAMB': 'BAMB Cement'})
    assert out and out[0]['symbol'] == 'BAMB'


def test_scout_us_candidates_identifies_movers():
    quotes = {
        'NVDA': {'close': 130.0, 'open': 120.0, 'volume': 1000000},  # +8.33%
        'AMD': {'close': 100.0, 'open': 99.0, 'volume': 500000},     # +1.01%
    }
    tracked = {'MSFT', 'AAPL'}
    cands = scout_us_candidates(tracked, quotes, threshold_pct=3.0)
    symbols = [c['symbol'] for c in cands]
    assert 'NVDA' in symbols
    assert 'AMD' not in symbols


def test_scout_us_candidates_skips_tracked_symbols():
    quotes = {
        'NVDA': {'close': 130.0, 'open': 120.0, 'volume': 1000000},
    }
    tracked = {'NVDA'}
    cands = scout_us_candidates(tracked, quotes, threshold_pct=3.0)
    assert cands == []
