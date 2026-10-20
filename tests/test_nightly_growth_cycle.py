"""Unit tests for the nightly growth cycle runner."""

import json
import os
import pytest

from scripts.run_nightly_growth_cycle import run_growth_cycle
from src.agent.order_journal import OrderJournal


@pytest.fixture
def growth_env(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    cfg = {
        "data_manager": {
            "symbols": ["AAPL", "MSFT", "TSLA"],
            "nse_symbols": ["SCOM", "EQTY"]
        },
        "strategies": {
            "momentum": {"type": "technical", "enabled": True, "weight": 1.0},
            "mean_reversion": {"type": "technical", "enabled": True, "weight": 1.0},
            "rsi_strategy": {"type": "technical", "enabled": True, "weight": 0.8}
        },
        "risk_limits": {
            "stop_loss_pct": 0.05,
            "trailing_stop_pct": 0.03
        },
        "universe": {
            "max_consecutive_losses": 3,
            "min_avg_dollar_volume": 100.0
        }
    }

    config_path = str(tmp_path / "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    return {
        "tmp_path": tmp_path,
        "data_dir": str(data_dir),
        "config_path": config_path,
        "config": cfg,
    }


def test_growth_cycle_runs_on_empty_journals(growth_env):
    report = run_growth_cycle(
        config=growth_env["config"],
        data_dir=growth_env["data_dir"],
        dry_run=False
    )
    assert report is not None
    assert "timestamp" in report
    assert "strategies" in report
    assert "removals" in report
    assert report["removals"] == []
    report_file = os.path.join(growth_env["data_dir"], "nightly_growth_report.json")
    assert os.path.exists(report_file)


def test_growth_cycle_attributes_and_adjusts_weights(growth_env):
    data_dir = growth_env["data_dir"]
    oj = OrderJournal(db_path=os.path.join(data_dir, "order_journal.db"))

    # Seed a profitable round-trip for momentum
    oj.record_intent("client-1", "AAPL", "buy", 10, 150.0, strategy="momentum")
    oj.mark_final("client-1", "filled", 10, 150.0)

    oj.record_intent("client-2", "AAPL", "sell", 10, 160.0, strategy="momentum")
    oj.mark_final("client-2", "filled", 10, 160.0)

    # Seed losing round-trips for mean_reversion on TSLA
    for i in range(4):
        cid_b = f"tsla-b-{i}"
        cid_s = f"tsla-s-{i}"
        oj.record_intent(cid_b, "TSLA", "buy", 5, 200.0, strategy="mean_reversion")
        oj.mark_final(cid_b, "filled", 5, 200.0)
        oj.record_intent(cid_s, "TSLA", "sell", 5, 180.0, strategy="mean_reversion")
        oj.mark_final(cid_s, "filled", 5, 180.0)

    report = run_growth_cycle(
        config=growth_env["config"],
        data_dir=data_dir,
        dry_run=False
    )

    assert "momentum" in report["attribution"]
    assert report["attribution"]["momentum"]["realized_pnl"] > 0
    assert report["attribution"]["mean_reversion"]["realized_pnl"] < 0

    # Overlay should be created and reflect updated weights
    overlay_path = os.path.join(data_dir, "config_overlay.json")
    assert os.path.exists(overlay_path)
    with open(overlay_path, "r", encoding="utf-8") as f:
        overlay = json.load(f)
    assert "strategies.momentum.weight" in overlay
    assert overlay["strategies.momentum.weight"] > 1.0


def test_growth_cycle_proposes_removals_for_losing_symbols(growth_env):
    data_dir = growth_env["data_dir"]
    oj = OrderJournal(db_path=os.path.join(data_dir, "order_journal.db"))

    # 4 consecutive losses on TSLA
    for i in range(4):
        cid_b = f"tsla-b-{i}"
        cid_s = f"tsla-s-{i}"
        oj.record_intent(cid_b, "TSLA", "buy", 5, 200.0, strategy="mean_reversion")
        oj.mark_final(cid_b, "filled", 5, 200.0)
        oj.record_intent(cid_s, "TSLA", "sell", 5, 180.0, strategy="mean_reversion")
        oj.mark_final(cid_s, "filled", 5, 180.0)

    report = run_growth_cycle(
        config=growth_env["config"],
        data_dir=data_dir,
        dry_run=False
    )

    removed_symbols = [r["symbol"] for r in report["removals"]]
    assert "TSLA" in removed_symbols
