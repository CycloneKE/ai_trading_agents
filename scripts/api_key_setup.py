"""
API Key Setup and Validation
Helps users configure and validate their API keys
"""

import os
import requests
import logging
from typing import Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class APIKeyValidator:
    """Validates API keys for various data providers"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_all_keys(self) -> Dict[str, bool]:
        """Validate all configured API keys"""
        results = {}
        
        # Alpha Vantage
        alpha_key = os.getenv('TRADING_ALPHA_VANTAGE_API_KEY')
        if alpha_key and alpha_key != 'your_alpha_vantage_api_key_here':
            results['alpha_vantage'] = self._validate_alpha_vantage(alpha_key)
        else:
            results['alpha_vantage'] = False
        
        # Financial Modeling Prep
        fmp_key = os.getenv('TRADING_FMP_API_KEY')
        if fmp_key and fmp_key != 'your_fmp_api_key_here':
            results['fmp'] = self._validate_fmp(fmp_key)
        else:
            results['fmp'] = False
        
        # Finnhub
        finnhub_key = os.getenv('TRADING_FINNHUB_API_KEY')
        if finnhub_key and finnhub_key != 'your_finnhub_api_key_here':
            results['finnhub'] = self._validate_finnhub(finnhub_key)
        else:
            results['finnhub'] = False
        
        # Alpaca
        alpaca_key = os.getenv('TRADING_ALPACA_API_KEY')
        alpaca_secret = os.getenv('TRADING_ALPACA_API_SECRET')
        if (alpaca_key and alpaca_key != 'your_alpaca_api_key_here' and 
            alpaca_secret and alpaca_secret != 'your_alpaca_api_secret_here'):
            results['alpaca'] = self._validate_alpaca(alpaca_key, alpaca_secret)
        else:
            results['alpaca'] = False
        
        self.validation_results = results
        return results
    
    def _validate_alpha_vantage(self, api_key: str) -> bool:
        """Validate Alpha Vantage API key"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': 'AAPL',
                'apikey': api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'Global Quote' in data:
                logger.info("Alpha Vantage API key validated successfully")
                return True
            elif 'Error Message' in data or 'Note' in data:
                logger.warning(f"Alpha Vantage API issue: {data}")
                return False
            else:
                logger.warning("Alpha Vantage API key validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Alpha Vantage validation error: {e}")
            return False
    
    def _validate_fmp(self, api_key: str) -> bool:
        """Validate Financial Modeling Prep API key"""
        try:
            url = "https://financialmodelingprep.com/api/v3/quote/AAPL"
            params = {'apikey': api_key}
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0 and 'symbol' in data[0]:
                logger.info("FMP API key validated successfully")
                return True
            else:
                logger.warning("FMP API key validation failed")
                return False
                
        except Exception as e:
            logger.error(f"FMP validation error: {e}")
            return False
    
    def _validate_finnhub(self, api_key: str) -> bool:
        """Validate Finnhub API key"""
        try:
            url = "https://finnhub.io/api/v1/quote"
            params = {'symbol': 'AAPL', 'token': api_key}
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'c' in data and isinstance(data['c'], (int, float)):
                logger.info("Finnhub API key validated successfully")
                return True
            else:
                logger.warning("Finnhub API key validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Finnhub validation error: {e}")
            return False
    
    def _validate_alpaca(self, api_key: str, api_secret: str) -> bool:
        """Validate Alpaca API credentials"""
        try:
            import base64
            
            # Create basic auth header
            credentials = f"{api_key}:{api_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                'Authorization': f'Basic {encoded_credentials}',
                'Content-Type': 'application/json'
            }
            
            # Use paper trading URL
            base_url = os.getenv('TRADING_ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
            url = f"{base_url}/v2/account"
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                logger.info("Alpaca API credentials validated successfully")
                return True
            else:
                logger.warning(f"Alpaca API validation failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Alpaca validation error: {e}")
            return False
    
    def get_setup_instructions(self) -> str:
        """Get setup instructions for API keys"""
        instructions = """
API Key Setup Instructions:

1. Alpha Vantage (Free tier available):
   - Visit: https://www.alphavantage.co/support/#api-key
   - Sign up for free API key
   - Set TRADING_ALPHA_VANTAGE_API_KEY in .env file

2. Financial Modeling Prep (Free tier available):
   - Visit: https://financialmodelingprep.com/developer/docs
   - Sign up for free API key
   - Set TRADING_FMP_API_KEY in .env file

3. Finnhub (Free tier available):
   - Visit: https://finnhub.io/register
   - Sign up for free API key
   - Set TRADING_FINNHUB_API_KEY in .env file

4. Alpaca (Paper trading free):
   - Visit: https://alpaca.markets/
   - Sign up for paper trading account
   - Get API key and secret from dashboard
   - Set TRADING_ALPACA_API_KEY and TRADING_ALPACA_API_SECRET in .env file

5. Copy .env.example to .env and fill in your API keys:
   cp .env.example .env

Note: Start with free tiers to test the system before upgrading to paid plans.
"""
        return instructions
    
    def print_validation_report(self):
        """Print validation report"""
        if not self.validation_results:
            self.validate_all_keys()
        
        print("\n" + "="*50)
        print("API KEY VALIDATION REPORT")
        print("="*50)
        
        for service, is_valid in self.validation_results.items():
            status = "✓ VALID" if is_valid else "✗ INVALID/MISSING"
            print(f"{service.upper():15} : {status}")
        
        valid_count = sum(self.validation_results.values())
        total_count = len(self.validation_results)
        
        print(f"\nSummary: {valid_count}/{total_count} API keys are valid")
        
        if valid_count == 0:
            print("\nNo valid API keys found. The system will use mock data.")
            print("Run 'python api_key_setup.py --help' for setup instructions.")
        elif valid_count < total_count:
            print(f"\n{total_count - valid_count} API key(s) need attention.")
            print("The system will use available data sources and fallback to mock data when needed.")
        else:
            print("\nAll API keys validated! Ready for live trading.")

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='API Key Setup and Validation')
    parser.add_argument('--validate', action='store_true', help='Validate all API keys')
    parser.add_argument('--help-setup', action='store_true', help='Show setup instructions')
    
    args = parser.parse_args()
    
    validator = APIKeyValidator()
    
    if args.help_setup:
        print(validator.get_setup_instructions())
    elif args.validate:
        validator.validate_all_keys()
        validator.print_validation_report()
    else:
        print("Use --validate to check API keys or --help-setup for instructions")

if __name__ == "__main__":
    main()