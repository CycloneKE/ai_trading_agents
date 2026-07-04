import os
import sys
import logging
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.connectors.alpaca_broker import AlpacaBroker
from src.connectors.base_broker import OrderRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_alpaca_trade():
    """
    Test script to place a live trade on Alpaca Paper.
    """
    load_dotenv()
    
    print("Initializing Alpaca Paper Broker...")
    # Get config from environment or default
    config = {
        'api_key': os.getenv('TRADING_ALPACA_API_KEY'),
        'api_secret': os.getenv('TRADING_ALPACA_API_SECRET'),
        'paper': True
    }
    
    broker = AlpacaBroker(config)
    
    if not broker.connect():
        print("ERROR: Failed to connect to Alpaca Paper.")
        return
    
    print("Connected to Alpaca Paper.")
    
    # Get account info
    account = broker.get_account_info()
    if account:
        print(f"Account Balance: ${account.cash:,.2f}")
    
    # Place a test order: Buy 1 share of AAPL
    symbol = "AAPL"
    quantity = 1
    
    print(f"Placing MARKET BUY order for {quantity} share of {symbol}...")
    
    order_req = OrderRequest(
        symbol=symbol,
        quantity=quantity,
        side='buy',
        order_type='market',
        time_in_force='gtc'
    )
    
    response = broker.place_order(order_req)
    
    if response:
        print(f"SUCCESS: Order placed!")
        print(f"Order ID: {response.order_id}")
        print(f"Status: {response.status}")
    else:
        print("FAILED: Order could not be placed.")

if __name__ == "__main__":
    test_alpaca_trade()
