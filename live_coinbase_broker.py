#!/usr/bin/env python3
"""
Live Coinbase Broker Implementation
"""

import os
import hmac
import hashlib
import base64
import time
import json
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class LiveCoinbaseBroker:
    def __init__(self):
        self.api_key = os.getenv('COINBASE_API_KEY')
        self.api_secret = os.getenv('COINBASE_API_SECRET')
        self.passphrase = os.getenv('COINBASE_PASSPHRASE')
        self.base_url = "https://api.exchange.coinbase.com"
        self.is_connected = False
        
    def _create_signature(self, timestamp, method, path, body=''):
        """Create Coinbase Pro API signature"""
        message = timestamp + method + path + body
        signature = hmac.new(
            base64.b64decode(self.api_secret),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode()
    
    def _make_request(self, method, endpoint, data=None):
        """Make authenticated request to Coinbase Pro API"""
        timestamp = str(time.time())
        path = f"/{endpoint}"
        body = json.dumps(data) if data else ''
        
        headers = {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': self._create_signature(timestamp, method, path, body),
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
        
        url = f"{self.base_url}{path}"
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, headers=headers, data=body, timeout=10)
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Coinbase API error: {e}")
            return None
    
    def connect(self):
        """Test connection to Coinbase Pro"""
        try:
            accounts = self._make_request('GET', 'accounts')
            if accounts:
                self.is_connected = True
                print("✅ Connected to Coinbase Pro")
                return True
        except Exception as e:
            print(f"❌ Coinbase connection failed: {e}")
        
        self.is_connected = False
        return False
    
    def get_accounts(self):
        """Get account balances"""
        if not self.is_connected:
            return None
        return self._make_request('GET', 'accounts')
    
    def get_products(self):
        """Get available trading pairs"""
        return self._make_request('GET', 'products')
    
    def place_market_order(self, symbol, side, size):
        """Place market order"""
        if not self.is_connected:
            return None
            
        order_data = {
            'type': 'market',
            'side': side,  # 'buy' or 'sell'
            'product_id': symbol,  # e.g., 'BTC-USD'
            'size': str(size)
        }
        
        result = self._make_request('POST', 'orders', order_data)
        if result:
            print(f"✅ Order placed: {side} {size} {symbol}")
        return result
    
    def place_limit_order(self, symbol, side, size, price):
        """Place limit order"""
        if not self.is_connected:
            return None
            
        order_data = {
            'type': 'limit',
            'side': side,
            'product_id': symbol,
            'size': str(size),
            'price': str(price)
        }
        
        result = self._make_request('POST', 'orders', order_data)
        if result:
            print(f"✅ Limit order placed: {side} {size} {symbol} @ ${price}")
        return result
    
    def get_orders(self):
        """Get open orders"""
        if not self.is_connected:
            return None
        return self._make_request('GET', 'orders')
    
    def cancel_order(self, order_id):
        """Cancel order"""
        if not self.is_connected:
            return None
        return self._make_request('DELETE', f'orders/{order_id}')
    
    def get_ticker(self, symbol):
        """Get current price for symbol"""
        return self._make_request('GET', f'products/{symbol}/ticker')

def test_coinbase_connection():
    """Test Coinbase connection and basic operations"""
    print("🔍 Testing Coinbase Pro Connection...")
    
    broker = LiveCoinbaseBroker()
    
    # Test connection
    if not broker.connect():
        print("❌ Failed to connect to Coinbase Pro")
        print("Check your API keys in .env file")
        return False
    
    # Get account info
    accounts = broker.get_accounts()
    if accounts:
        print(f"📊 Found {len(accounts)} accounts")
        for account in accounts[:3]:  # Show first 3
            if float(account.get('balance', 0)) > 0:
                print(f"   {account['currency']}: {account['balance']}")
    
    # Get BTC price
    ticker = broker.get_ticker('BTC-USD')
    if ticker:
        print(f"💰 BTC-USD Price: ${ticker.get('price', 'N/A')}")
    
    # Get open orders
    orders = broker.get_orders()
    if orders is not None:
        print(f"📋 Open orders: {len(orders)}")
    
    return True

if __name__ == "__main__":
    test_coinbase_connection()