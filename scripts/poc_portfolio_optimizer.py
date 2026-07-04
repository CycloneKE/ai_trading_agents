"""
PoC portfolio optimizer (dependency-light).
Provides inverse-volatility weights, capping, and turnover-limited rebalance blending.
"""
from typing import Dict, List
import math


def returns_from_prices(prices: List[float]) -> List[float]:
    if len(prices) < 2:
        return []
    rets = []
    for i in range(1, len(prices)):
        prev = prices[i-1]
        rets.append(0.0 if prev == 0 else (prices[i] - prev) / prev)
    return rets


def sample_std(x: List[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / (n - 1)
    return math.sqrt(var)


def inverse_vol_weights(price_series: Dict[str, List[float]], window: int = 20, min_vol: float = 1e-6) -> Dict[str, float]:
    vols = {}
    for s, prices in price_series.items():
        seg = prices[-(window+1):] if len(prices) >= (window+1) else prices
        rets = returns_from_prices(seg)
        vol = sample_std(rets)
        vols[s] = max(vol, min_vol) if vol is not None else float('inf')
    inv = {s: (1.0 / v if v and v != float('inf') else 0.0) for s, v in vols.items()}
    total = sum(inv.values())
    if total == 0:
        n = len(price_series)
        return {s: 1.0/n for s in price_series}
    return {s: v/total for s, v in inv.items()}


def cap_weights(weights: Dict[str, float], max_weight: float) -> Dict[str, float]:
    if max_weight <= 0 or max_weight >= 1:
        return weights.copy()
    # Start by capping initial values
    capped = {s: min(w, max_weight) for s, w in weights.items()}
    excess = sum(weights[s] - capped[s] for s in weights)

    # Iteratively redistribute excess equally among assets that are below cap
    # (waterfilling). This ensures nobody exceeds max_weight.
    below = {s: capped[s] for s in capped if capped[s] < max_weight}
    # Use a small epsilon to avoid infinite loops on floating point
    eps = 1e-12
    while excess > eps and below:
        per_asset = excess / len(below)
        removed = []
        for s in list(below.keys()):
            available = max_weight - capped[s]
            add = min(per_asset, available)
            capped[s] += add
            excess -= add
            # If this asset reached cap, remove from below in next iter
            if abs(max_weight - capped[s]) <= eps or capped[s] >= max_weight - eps:
                capped[s] = min(capped[s], max_weight)
                removed.append(s)
        for s in removed:
            below.pop(s, None)
        # Recompute below in case numerical issues
        below = {s: capped[s] for s in capped if capped[s] < max_weight - eps}

    total = sum(capped.values())
    if total == 0:
        n = len(weights)
        return {s: 1.0 / n for s in weights}
    # Renormalize to sum to 1
    return {s: w / total for s, w in capped.items()}


def rebalance_with_turnover(current: Dict[str, float], target: Dict[str, float], turnover_limit: float) -> Dict[str, float]:
    symbols = set(current) | set(target)
    required_change = sum(abs(target.get(s,0.0) - current.get(s,0.0)) for s in symbols)
    if required_change <= turnover_limit or turnover_limit >= 1.0:
        return target.copy()
    scale = turnover_limit / required_change if required_change > 0 else 0.0
    new = {}
    for s in symbols:
        cur = current.get(s, 0.0)
        tgt = target.get(s, 0.0)
        new[s] = cur + (tgt - cur) * scale
    total = sum(new.values())
    if total == 0:
        n = len(new)
        return {s: 1.0/n for s in new}
    return {s: w/total for s, w in new.items()}
