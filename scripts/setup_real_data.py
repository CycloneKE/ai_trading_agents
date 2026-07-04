#!/usr/bin/env python3
"""
Setup script to configure real data integration
"""

import os
import json
from dotenv import load_dotenv

def setup_env_file():
    """Setup .env file with API keys"""
    env_content = """# Real Trading Data API Keys
TRADING_FMP_API_KEY=your_fmp_api_key_here
TRADING_ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
TRADING_FINNHUB_API_KEY=your_finnhub_key_here

# Broker API Keys (for live trading)
COINBASE_API_KEY=your_coinbase_key_here
COINBASE_API_SECRET=your_coinbase_secret_here
COINBASE_PASSPHRASE=your_coinbase_passphrase_here
OANDA_API_KEY=your_oanda_key_here
OANDA_ACCOUNT_ID=your_oanda_account_here

# Security
TRADING_MASTER_KEY=your_secure_master_key_here
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file - please add your API keys")
    else:
        print("⚠️  .env file already exists")

def update_startup_script():
    """Update startup script to use real API server"""
    startup_content = '''"""
Real data startup script for AI Trading System
"""

import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

def start_real_api_server():
    """Start the real data API server"""
    print("🚀 Starting Real Data API Server...")
    subprocess.run([sys.executable, "real_api_server.py"])

def start_frontend():
    """Start the frontend dashboard"""
    print("🎨 Starting frontend dashboard...")
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    if os.path.exists(frontend_dir):
        subprocess.run(["npm", "run", "dev"], shell=True, cwd=frontend_dir)

def main():
    """Main startup function"""
    print("=" * 60)
    print("🚀 AI TRADING SYSTEM - REAL DATA MODE")
    print("=" * 60)
    
    # Start real API server in background
    api_thread = Thread(target=start_real_api_server, daemon=True)
    api_thread.start()
    
    time.sleep(3)
    
    # Start frontend
    frontend_thread = Thread(target=start_frontend, daemon=True)
    frontend_thread.start()
    
    time.sleep(5)
    
    print("🌐 Opening dashboard...")
    webbrowser.open("http://localhost:3001")
    
    print("\\n✅ System Status:")
    print("   📊 Dashboard: http://localhost:3001")
    print("   🔌 Real Data API: http://localhost:5001")
    print("   📈 Data Source: Live Market Data")
    print("\\n⚠️  Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\n🛑 Shutting down...")

if __name__ == "__main__":
    main()
'''
    
    with open('start_real_system.py', 'w') as f:
        f.write(startup_content)
    
    print("✅ Created start_real_system.py")

def create_api_key_tester():
    """Create a script to test API keys"""
    tester_content = '''#!/usr/bin/env python3
"""
Test API keys for real data sources
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_fmp_api():
    """Test FMP API key"""
    api_key = os.getenv('TRADING_FMP_API_KEY')
    if not api_key:
        print("❌ FMP API key not found")
        return False
    
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                print(f"✅ FMP API working - AAPL price: ${data[0]['price']}")
                return True
        print(f"❌ FMP API error: {response.status_code}")
        return False
    except Exception as e:
        print(f"❌ FMP API error: {e}")
        return False

def test_alpha_vantage_api():
    """Test Alpha Vantage API key"""
    api_key = os.getenv('TRADING_ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("❌ Alpha Vantage API key not found")
        return False
    
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'Global Quote' in data:
                price = data['Global Quote']['05. price']
                print(f"✅ Alpha Vantage API working - AAPL price: ${price}")
                return True
        print(f"❌ Alpha Vantage API error: {response.status_code}")
        return False
    except Exception as e:
        print(f"❌ Alpha Vantage API error: {e}")
        return False

def main():
    """Test all API keys"""
    print("🔍 Testing API Keys...")
    print("=" * 40)
    
    fmp_ok = test_fmp_api()
    av_ok = test_alpha_vantage_api()
    
    print("=" * 40)
    if fmp_ok or av_ok:
        print("✅ At least one API is working - system ready!")
    else:
        print("❌ No working APIs found - please check your keys")
        print("\\n📝 Get free API keys:")
        print("   FMP: https://financialmodelingprep.com/developer/docs")
        print("   Alpha Vantage: https://www.alphavantage.co/support/#api-key")

if __name__ == "__main__":
    main()
'''
    
    with open('test_api_keys.py', 'w') as f:
        f.write(tester_content)
    
    print("✅ Created test_api_keys.py")

def main():
    """Main setup function"""
    print("🔧 Setting up real data integration...")
    print("=" * 50)
    
    setup_env_file()
    update_startup_script()
    create_api_key_tester()
    
    print("\n✅ Setup complete!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your API keys")
    print("2. Run: python test_api_keys.py")
    print("3. Run: python start_real_system.py")
    print("\n🔑 Get free API keys:")
    print("   FMP: https://financialmodelingprep.com/developer/docs")
    print("   Alpha Vantage: https://www.alphavantage.co/support/#api-key")

if __name__ == "__main__":
    main()