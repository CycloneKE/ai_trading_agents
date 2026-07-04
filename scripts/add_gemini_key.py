"""
Add Gemini API key to environment
"""

import os

def add_gemini_key():
    api_key = input("Enter your Gemini API key: ")
    
    # Add to .env file
    with open('.env', 'a') as f:
        f.write(f'\nGEMINI_API_KEY={api_key}\n')
    
    # Set for current session
    os.environ['GEMINI_API_KEY'] = api_key
    
    print("Gemini API key added successfully!")
    
    # Test the key
    from gemini_llm_integration import GeminiTradingLLM
    
    try:
        gemini = GeminiTradingLLM()
        result = gemini.analyze_market_sentiment("Apple stock rises on strong earnings", "AAPL")
        print("Test successful!")
        print(f"Sentiment: {result['recommendation']}")
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == '__main__':
    add_gemini_key()