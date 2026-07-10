from src.agent.universe_scout import propose_candidates


def test_untracked_big_mover_is_proposed():
    movers = [{'symbol': 'BAMB', 'change_pct': 4.2}, {'symbol': 'SCOM', 'change_pct': 5.0}]
    out = propose_candidates(tracked={'SCOM'}, movers=movers, news_texts=[])
    assert out == [{'symbol': 'BAMB', 'reason': 'Moved +4.2% today while untracked'}]


def test_small_moves_ignored():
    assert propose_candidates({'SCOM'}, [{'symbol': 'BAMB', 'change_pct': 1.0}], []) == []


def test_news_mention_of_untracked_symbol():
    out = propose_candidates({'SCOM'}, [], ['BAMB Cement announces record dividend'],
                             known_universe={'BAMB': 'BAMB Cement'})
    assert out and out[0]['symbol'] == 'BAMB'
