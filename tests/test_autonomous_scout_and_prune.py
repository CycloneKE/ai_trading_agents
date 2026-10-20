"""Unit tests for the autonomous universe scout and pruning pipeline."""

import json
import os
import tempfile
from unittest.mock import MagicMock
import pytest

from src.agent.main import TradingAgent
from src.agent.escalation_manager import EscalationManager
from src.agent.order_journal import OrderJournal
from src.agent.universe_manager import effective_universe


@pytest.fixture
def agent_fixture(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    cfg = {
        "trading": {"initial_capital": 100000, "commission": 0.001, "slippage": 0.0005},
        "data_manager": {
            "symbols": ["AAPL", "MSFT"],
            "nse_symbols": ["SCOM", "EQTY"]
        },
        "strategies": {
            "momentum": {"type": "technical", "enabled": True, "weight": 1.0},
            "mean_reversion": {"type": "technical", "enabled": True, "weight": 1.0}
        },
        "risk_limits": {"stop_loss_pct": 0.05, "trailing_stop_pct": 0.03},
        "universe": {
            "min_avg_dollar_volume": 500000.0,
            "min_price": 5.0,
            "max_correlation": 0.85,
            "max_consecutive_losses": 3
        }
    }

    config_path = str(tmp_path / "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    agent = TradingAgent.__new__(TradingAgent)
    agent.config = cfg
    agent.running = False
    agent.components = {}

    em = EscalationManager(db_path=os.path.join(str(data_dir), "escalations.db"))
    oj = OrderJournal(db_path=os.path.join(str(data_dir), "order_journal.db"))
    agent.components["escalation_manager"] = em
    agent.order_journal = oj

    strat_mgr = MagicMock()
    # 40-bar flat history for AAPL and MSFT
    strat_mgr._regime_history = {
        "AAPL": [150.0 + (i % 2) for i in range(40)],
        "MSFT": [200.0 + (i % 3) for i in range(40)]
    }
    strat_mgr._regime_state = {"AAPL": "trending", "MSFT": "ranging"}
    agent.components["strategy_manager"] = strat_mgr

    return agent, em, oj, cfg, str(data_dir)


def test_scout_screens_and_escalates_qualified_us_mover(agent_fixture, monkeypatch):
    agent, em, oj, cfg, data_dir = agent_fixture

    # NVDA is moving +5% with high volume and uncorrelated price history
    market_data = {
        "NVDA": {
            "close": 120.0,
            "open": 114.0,
            "volume": 50000,
            "avg_dollar_volume": 6000000.0
        }
    }

    # Mock yahoo_history so NVDA returns 40 bars of valid prices
    import src.connectors.yahoo_history as yh
    nvda_bars = [100.0 + i * 0.5 for i in range(40)]
    monkeypatch.setattr(yh, "fetch_many", lambda syms, range_="6mo": {"NVDA": nvda_bars})

    agent._run_universe_scout_and_prune(market_data)

    pending = em.get_pending_escalations()
    nvda_esc = [e for e in pending if e["symbol"] == "NVDA"]
    assert len(nvda_esc) == 1
    assert nvda_esc[0]["action"] == "add_symbol"
    assert "US Screened" in nvda_esc[0]["reason"]


def test_scout_rejects_correlated_us_mover(agent_fixture, monkeypatch):
    agent, em, oj, cfg, data_dir = agent_fixture

    # TSLA moves +5%, but its price movement is 100% correlated with AAPL
    market_data = {
        "TSLA": {
            "close": 200.0,
            "open": 190.0,
            "volume": 50000,
            "avg_dollar_volume": 10000000.0
        }
    }

    import src.connectors.yahoo_history as yh
    # Identical return pattern as AAPL
    strat_mgr = agent.components["strategy_manager"]
    aapl_bars = strat_mgr._regime_history["AAPL"]
    monkeypatch.setattr(yh, "fetch_many", lambda syms, range_="6mo": {"TSLA": list(aapl_bars)})

    agent._run_universe_scout_and_prune(market_data)

    pending = em.get_pending_escalations()
    tsla_esc = [e for e in pending if e["symbol"] == "TSLA"]
    assert len(tsla_esc) == 0  # Screened out due to high correlation!


def test_prune_escalates_losing_symbol(agent_fixture):
    agent, em, oj, cfg, data_dir = agent_fixture

    # Seed 4 consecutive losses for AAPL
    for i in range(4):
        cid_b = f"aapl-b-{i}"
        cid_s = f"aapl-s-{i}"
        oj.record_intent(cid_b, "AAPL", "buy", 10, 150.0, strategy="momentum")
        oj.mark_final(cid_b, "filled", 10, 150.0)
        oj.record_intent(cid_s, "AAPL", "sell", 10, 140.0, strategy="momentum")
        oj.mark_final(cid_s, "filled", 10, 140.0)

    agent._run_universe_scout_and_prune({})

    pending = em.get_pending_escalations()
    removals = [e for e in pending if e["action"] == "remove_symbol" and e["symbol"] == "AAPL"]
    assert len(removals) == 1
    assert "consecutive losing" in removals[0]["reason"]
