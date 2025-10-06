#!/usr/bin/env python3
"""
Live Trading Agent with Real Broker Connections
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from live_coinbase_broker import LiveCoinbaseBroker

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveTradingAgent:
    def __init__(self):
        self.coinbase = LiveCoinbaseBroker()
        self.is_running = False
        self.positions = {}
        self.trading_enabled = False
        
    def connect_brokers(self):
        """Connect to all brokers"""
        logger.info("🔌 Connecting to brokers...")
        
        # Connect to Coinbase
        if self.coinbase.connect():
            logger.info("✅ Coinbase Pro connected")
            return True
        else:
            logger.error("❌ Failed to connect to Coinbase Pro")
            return False
    
    def get_portfolio_status(self):
        """Get current portfolio status"""
        if not self.coinbase.is_connected:
            return {"error": "Not connected to brokers"}
        
        accounts = self.coinbase.get_accounts()
        portfolio = {}
        
        if accounts:
            for account in accounts:
                currency = account['currency']
                balance = float(account['balance'])
                if balance > 0:
                    portfolio[currency] = {
                        'balance': balance,
                        'available': float(account.get('available', 0)),
                        'hold': float(account.get('hold', 0))
                    }
        
        return portfolio
    
    def place_crypto_order(self, symbol, side, amount, order_type='market', price=None):
        """Place cryptocurrency order"""
        if not self.coinbase.is_connected:
            logger.error("Coinbase not connected")
            return None
        
        if not self.trading_enabled:
            logger.warning("Trading is disabled - use enable_trading() first")
            return None
        
        try:
            if order_type == 'market':
                result = self.coinbase.place_market_order(symbol, side, amount)
            elif order_type == 'limit' and price:
                result = self.coinbase.place_limit_order(symbol, side, amount, price)
            else:
                logger.error("Invalid order type or missing price for limit order")
                return None
            
            if result:
                logger.info(f"✅ Order executed: {side} {amount} {symbol}")
                return result
            else:
                logger.error("❌ Order failed")
                return None
                
        except Exception as e:
            logger.error(f"Order error: {e}")
            return None
    
    def get_market_price(self, symbol):
        """Get current market price"""
        if not self.coinbase.is_connected:
            return None
        
        ticker = self.coinbase.get_ticker(symbol)
        if ticker:
            return float(ticker.get('price', 0))
        return None
    
    def enable_trading(self):
        """Enable live trading (safety feature)"""
        self.trading_enabled = True
        logger.warning("🚨 LIVE TRADING ENABLED - Real money at risk!")
    
    def disable_trading(self):
        """Disable live trading"""
        self.trading_enabled = False
        logger.info("🛡️ Live trading disabled")
    
    def emergency_stop(self):
        """Emergency stop - cancel all orders"""
        logger.warning("🛑 EMERGENCY STOP - Cancelling all orders")
        
        if self.coinbase.is_connected:
            orders = self.coinbase.get_orders()
            if orders:
                for order in orders:
                    self.coinbase.cancel_order(order['id'])
                    logger.info(f"Cancelled order: {order['id']}")
        
        self.disable_trading()
    
    def run_simple_strategy(self):
        """Run a simple trading strategy"""
        logger.info("🤖 Starting simple trading strategy...")
        
        if not self.connect_brokers():
            logger.error("Cannot start - broker connection failed")
            return
        
        self.is_running = True
        
        while self.is_running:
            try:
                # Get BTC price
                btc_price = self.get_market_price('BTC-USD')
                if btc_price:
                    logger.info(f"💰 BTC Price: ${btc_price:,.2f}")
                
                # Get portfolio
                portfolio = self.get_portfolio_status()
                if 'USD' in portfolio:
                    usd_balance = portfolio['USD']['available']
                    logger.info(f"💵 USD Balance: ${usd_balance:,.2f}")
                
                # Simple strategy: Buy if price drops 2%, sell if up 3%
                # (This is just an example - implement your own strategy)
                
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                logger.info("🛑 Strategy stopped by user")
                break
            except Exception as e:
                logger.error(f"Strategy error: {e}")
                time.sleep(10)
        
        self.is_running = False

def main():
    """Main function for live trading"""
    print("🚨 LIVE TRADING AGENT")
    print("=" * 50)
    print("⚠️  WARNING: This connects to real brokers with real money!")
    print("⚠️  Only use with small amounts for testing!")
    print("=" * 50)
    
    agent = LiveTradingAgent()
    
    # Test connection
    if not agent.connect_brokers():
        print("❌ Failed to connect to brokers")
        return
    
    # Show portfolio
    portfolio = agent.get_portfolio_status()
    print("\n📊 Current Portfolio:")
    for currency, info in portfolio.items():
        if info['balance'] > 0:
            print(f"   {currency}: {info['balance']:.8f} (Available: {info['available']:.8f})")
    
    # Get current prices
    btc_price = agent.get_market_price('BTC-USD')
    eth_price = agent.get_market_price('ETH-USD')
    
    print(f"\n💰 Current Prices:")
    if btc_price:
        print(f"   BTC-USD: ${btc_price:,.2f}")
    if eth_price:
        print(f"   ETH-USD: ${eth_price:,.2f}")
    
    print("\n🔧 Available Commands:")
    print("   agent.enable_trading()  - Enable live trading")
    print("   agent.place_crypto_order('BTC-USD', 'buy', 0.001)  - Place order")
    print("   agent.emergency_stop()  - Cancel all orders")
    print("   agent.run_simple_strategy()  - Run automated strategy")
    
    # Interactive mode
    print("\n💡 Agent ready - use Python commands to trade")
    print("   Example: agent.enable_trading()")
    
    return agent

if __name__ == "__main__":
    agent = main()