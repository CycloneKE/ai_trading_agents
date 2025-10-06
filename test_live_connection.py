#!/usr/bin/env python3
"""
Test live broker connections
"""

import os
from dotenv import load_dotenv
from live_coinbase_broker import LiveCoinbaseBroker

load_dotenv()

def test_coinbase():
    """Test Coinbase Pro connection"""
    print("🔍 Testing Coinbase Pro Connection...")
    print("=" * 40)
    
    # Check API keys
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    passphrase = os.getenv('COINBASE_PASSPHRASE')
    
    if not all([api_key, api_secret, passphrase]):
        print("❌ Missing Coinbase API credentials")
        print("Required in .env file:")
        print("   COINBASE_API_KEY=your_key")
        print("   COINBASE_API_SECRET=your_secret")
        print("   COINBASE_PASSPHRASE=your_passphrase")
        return False
    
    print(f"✅ API Key: {api_key[:8]}...")
    print(f"✅ Secret: {api_secret[:8]}...")
    print(f"✅ Passphrase: {passphrase[:4]}...")
    
    # Test connection
    broker = LiveCoinbaseBroker()
    
    if broker.connect():
        print("\n✅ Successfully connected to Coinbase Pro!")
        
        # Get account info
        accounts = broker.get_accounts()
        if accounts:
            print(f"\n📊 Account Summary ({len(accounts)} currencies):")
            for account in accounts:
                balance = float(account.get('balance', 0))
                if balance > 0:
                    currency = account['currency']
                    available = float(account.get('available', 0))
                    print(f"   {currency}: {balance:.8f} (Available: {available:.8f})")
        
        # Get current prices
        print("\n💰 Current Crypto Prices:")
        for symbol in ['BTC-USD', 'ETH-USD', 'ADA-USD']:
            ticker = broker.get_ticker(symbol)
            if ticker:
                price = float(ticker.get('price', 0))
                print(f"   {symbol}: ${price:,.2f}")
        
        # Get open orders
        orders = broker.get_orders()
        if orders is not None:
            print(f"\n📋 Open Orders: {len(orders)}")
            if orders:
                for order in orders[:3]:  # Show first 3
                    print(f"   {order.get('side')} {order.get('size')} {order.get('product_id')} @ ${order.get('price', 'market')}")
        
        return True
    else:
        print("❌ Failed to connect to Coinbase Pro")
        print("Check your API credentials and permissions")
        return False

def main():
    """Test all broker connections"""
    print("🔌 LIVE BROKER CONNECTION TEST")
    print("=" * 50)
    
    coinbase_ok = test_coinbase()
    
    print("\n" + "=" * 50)
    if coinbase_ok:
        print("✅ Ready for live trading!")
        print("\n🚨 IMPORTANT SAFETY NOTES:")
        print("   • Start with small amounts")
        print("   • Test in sandbox mode first")
        print("   • Always use stop losses")
        print("   • Monitor positions closely")
        print("\n🚀 To start live trading:")
        print("   python live_trading_agent.py")
    else:
        print("❌ Broker connection issues - fix before trading")

if __name__ == "__main__":
    main()