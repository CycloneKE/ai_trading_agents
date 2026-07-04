import os
import json
import bcrypt
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from auth if possible to maintain hashing consistency
try:
    from src.api.auth import hash_password
except ImportError:
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

USERS_FILE = 'users.json'

def init_auth(username='admin', password='secure_trading_password_2024'):
    """
    Initializes the users.json file with a default admin user.
    """
    print(f"Initializing authentication database: {USERS_FILE}...")
    
    users = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            try:
                users = json.load(f)
            except json.JSONDecodeError:
                print(f"WARNING: {USERS_FILE} is corrupted. Overwriting.")
    
    if username in users:
        print(f"User '{username}' already exists. Updating password...")
    else:
        print(f"Adding default user: '{username}'")
        
    users[username] = hash_password(password)
    
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)
        
    print(f"SUCCESS: Authentication initialized. '{username}' is ready to login.")
    print(f"Please delete this script after running it for security.")

if __name__ == "__main__":
    # Allow command line overrides
    user = sys.argv[1] if len(sys.argv) > 1 else 'admin'
    pwd = sys.argv[2] if len(sys.argv) > 2 else 'secure_trading_password_2024'
    init_auth(user, pwd)
