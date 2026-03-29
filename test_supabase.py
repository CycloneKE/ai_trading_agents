import sys
import os

# Add parent directory to path to import DatabaseManager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from secure_config import SecureConfigManager
from database_manager import DatabaseManager

def test_supabase():
    print("Testing Supabase integration...")
    try:
        secure_config = SecureConfigManager()
        db_config = secure_config.get_database_config()
        
        if not db_config.get('supabase_url') or not db_config.get('supabase_key'):
            print("ERROR: SUPABASE_URL or SUPABASE_KEY not found in the environment variables.")
            sys.exit(1)
            
        print(f"Connecting to: {db_config['supabase_url']}")
        
        # Initialize DatabaseManager
        db = DatabaseManager(db_config)
        
        if db.supabase_client is None:
            print("ERROR: DatabaseManager failed to initialize the Supabase client.")
            sys.exit(1)
            
        print("Success! Supabase client initialized.")
        
        # We won't insert a trade just in case the tables aren't made, 
        # but we'll try a basic GET request to verify connectivity
        test_symbol = "TEST_API"
        print(f"Executing test query for {test_symbol}...")
        
        res = db.get_recent_market_data(test_symbol, limit=1)
        print(f"Query executed successfully! Found {len(res)} records.")

        # Test trade insertion
        print("Testing trade insertion...")
        trade_payload = {
            'symbol': 'BTC-USD',
            'action': 'buy',
            'quantity': 0.5,
            'price': 65000.0,
            'strategy': 'TEST_REALTIME_SUPABASE',
            'confidence': 0.95,
            'metadata': {'test': True}
        }
        db.store_trade(trade_payload)
        print("Trade insertion completed without error.")
        
    except Exception as e:
        print(f"Exception caught during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_supabase()
