from poc_portfolio_optimizer import inverse_vol_weights, cap_weights, rebalance_with_turnover


def test_inverse_vol_weights_simple():
    data = {
        'A': [100, 101, 102, 103, 104],
        'B': [50, 49, 48, 47, 46],
        'C': [200, 201, 199, 198, 202]
    }
    w = inverse_vol_weights(data, window=3)
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_cap_weights_and_renormalize():
    weights = {'A': 0.6, 'B': 0.3, 'C': 0.1}
    capped = cap_weights(weights, 0.4)
    assert max(capped.values()) <= 0.4
    assert abs(sum(capped.values()) - 1.0) < 1e-9


def test_rebalance_with_turnover():
    current = {'A': 0.5, 'B': 0.5}
    target = {'A': 0.2, 'B': 0.8}
    new = rebalance_with_turnover(current, target, turnover_limit=0.2)
    # Changes should be scaled and still sum to 1
    assert abs(sum(new.values()) - 1.0) < 1e-9