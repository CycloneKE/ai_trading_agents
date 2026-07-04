import sys
import os
import json

# Add root to pythonpath
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.connectors.real_data_connector import RealDataConnector
from src.connectors.market_data_connector import YahooFinanceConnector
from src.agent.main import TradingAgent

def run_check():
    print("----- Data Connectors Check -----")
    # 1. Test RealDataConnector
    rc = RealDataConnector({})
    try:
        data = rc.get_real_time_data("AAPL")
        print("RealDataConnector AAPL:", data)
    except Exception as e:
        print("RealDataConnector AAPL failed:", str(e))
        
    try:
        sectors = rc.get_sector_performance()
        print("RealDataConnector Sectors:", len(sectors))
    except Exception as e:
        print("RealDataConnector Sectors failed:", str(e))
        
    # 2. Test YahooFinanceConnector
    yf = YahooFinanceConnector({})
    try:
        yf_data = yf.get_real_time_data(["MSFT"])
        print("YahooFinance MSFT:", yf_data)
    except Exception as e:
        print("YahooFinance MSFT failed:", str(e))
        
    print("\n----- Trading Agent Initialization Check -----")
    try:
        agent = TradingAgent('config/config.json')
        print("Agent initialized successfully!")
    except Exception as e:
        print("Agent initialization failed:", str(e))

if __name__ == "__main__":
    run_check()
