import sys
import os
import logging

# Add the project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from src.utils.secure_config import SecureConfigManager
from src.utils.database_manager import DatabaseManager

# Setup logging to see what DatabaseManager is doing
logging.basicConfig(level=logging.INFO)

def debug_database():
    print("Debugging Database Connection...")
    try:
        secure_config = SecureConfigManager()
        db_config = secure_config.get_database_config()
        
        print(f"Config Host: {db_config.get('host')}")
        print(f"Config Port: {db_config.get('port')}")
        print(f"Config User: {db_config.get('username')}")
        print(f"Supabase URL: {db_config.get('supabase_url')}")
        
        db = DatabaseManager(db_config)
        
        print(f"Database Manager Enabled: {db.enabled}")
        print(f"Supabase Client Initialized: {db.supabase_client is not None}")
        print(f"Postgres Pool Initialized: {db.connection_pool is not None}")
        
        if db.connection_pool:
            print("Successfully connected to Postgres pool.")
            # Try to manually trigger table creation if it didn't run
            print("Triggering manual table creation check...")
            db._create_tables()
            print("Table creation check finished.")
        else:
            print("Postgres pool was NOT initialized.")
            
    except Exception as e:
        print(f"Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_database()
