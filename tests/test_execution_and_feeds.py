"""
Test suite for Pillar 2: Execution Engine, Iceberg Slicing, and Real-Time Data Feeds.
"""
import pytest
import time
from src.agent.order_execution_engine import OrderExecutionEngine, OrderRequest, OrderType
from src.agent.realtime_data_feed import RealTimeDataFeed

class DummyBroker:
    def place_order(self, order_request):
        class DummyResponse:
            order_id = "dummy_ord_123"
            status = "submitted"
        return DummyResponse()

    def get_account_info(self):
        class DummyAccount:
            buying_power = 50000.0
            equity = 100000.0
        return DummyAccount()

    def get_market_data(self, symbol):
        return {"symbol": symbol, "last_price": 150.0}

class DummyBrokerManager:
    def get_broker(self):
        return DummyBroker()

    def place_order(self, order_request):
        return DummyBroker().place_order(order_request)

    def get_account_info(self):
        return DummyBroker().get_account_info()

    def get_market_data(self, symbol):
        return DummyBroker().get_market_data(symbol)

def test_iceberg_order_execution():
    """Test Iceberg order plan creation and initial slice execution."""
    engine = OrderExecutionEngine({"slice_size": 0.2}, DummyBrokerManager())
    
    order = OrderRequest(
        symbol="AAPL",
        side="buy",
        quantity=20.0,
        order_type=OrderType.LIMIT,
        price=150.0
    )
    
    # Test execution plan creation
    plan = engine._create_execution_plan(order)
    assert plan.get("strategy") == "iceberg"
    assert plan.get("slice_size") == 0.2
    
    # Test iceberg execution
    order_id = engine._execute_iceberg(order, plan)
    assert order_id == "dummy_ord_123"
    assert "dummy_ord_123" in engine.pending_orders
    assert engine.pending_orders["dummy_ord_123"]["remaining_quantity"] == 16.0

def test_realtime_data_feed_subscribers():
    """Test RealTimeDataFeed tick dispatching and data buffer storage."""
    feed = RealTimeDataFeed({})
    
    received_ticks = []
    def sample_callback(data):
        received_ticks.append(data)
        
    feed.subscribe("AAPL", sample_callback)
    
    # Notify subscribers manually
    tick_data = {"symbol": "AAPL", "price": 185.50, "size": 100, "type": "trade"}
    feed._notify_subscribers("AAPL", tick_data)
    
    assert len(received_ticks) == 1
    assert received_ticks[0]["price"] == 185.50
    
    latest = feed.get_latest_data("AAPL")
    assert latest["price"] == 185.50
