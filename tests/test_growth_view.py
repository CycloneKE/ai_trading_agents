"""Unit tests for the growth view builder."""

from src.api.growth_view import build_growth_view


def test_growth_view_empty_inputs_returns_defaults():
    view = build_growth_view(None, None, None)
    assert view is not None
    assert "champion" in view
    assert "challengers" in view
    assert "learning_cycle" in view
    assert view["champion"]["ensemble_method"] == "adaptive_confidence"
    assert view["challengers"] == []


def test_growth_view_populates_champion_and_challengers():
    config = {
        "ensemble_method": "adaptive_confidence",
        "strategies": {
            "momentum": {"weight": 1.0},
            "mean_reversion": {"weight": 1.0}
        }
    }
    report = {
        "timestamp": "2026-10-20T12:00:00Z",
        "attribution": {
            "momentum": {"realized_pnl": 500.0, "win_rate": 0.6, "closed_trades": 10}
        },
        "removals": [{"symbol": "TSLA", "reason": "3 consecutive losses"}],
        "fills_analyzed": 25
    }
    backtest = {
        "results": [
            {
                "window": "OOS regime_routed",
                "sharpe": 0.36,
                "total_return": 0.0822,
                "max_drawdown": 0.0391,
                "trades": 531,
                "correlation_to_spy": 0.15
            }
        ],
        "acceptance": {
            "sharpe > 0.5": False,
            "|SPY correlation| < 0.3": True,
            "max drawdown < 15%": True,
            "trades > 100": True,
            "beats baseline sharpe": True
        },
        "passed": False
    }

    view = build_growth_view(config, report, backtest)

    assert view["champion"]["ensemble_method"] == "adaptive_confidence"
    assert view["champion"]["weights"]["momentum"] == 1.0
    assert len(view["challengers"]) == 1

    challenger = view["challengers"][0]
    assert challenger["ensemble_method"] == "regime_routed"
    assert challenger["metrics"]["sharpe"] == 0.36
    assert challenger["metrics"]["total_return"] == 0.0822
    assert challenger["acceptance_gate"]["passed"] is False
    assert challenger["acceptance_gate"]["checks"]["sharpe > 0.5"] is False
    assert challenger["acceptance_gate"]["checks"]["trades > 100"] is True

    assert view["learning_cycle"]["last_run"] == "2026-10-20T12:00:00Z"
    assert len(view["learning_cycle"]["removal_proposals"]) == 1
    assert view["learning_cycle"]["fills_analyzed"] == 25
