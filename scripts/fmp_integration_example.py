from src.fmp_provider import FMPProvider

def integrate_fmp_data(symbol: str):
    """Example integration of FMP data for ML features"""
    fmp = FMPProvider()
    
    # Get fundamental data
    ratios = fmp.get_financial_ratios(symbol)
    quote = fmp.get_quote(symbol)
    
    # Extract key features for ML
    features = {
        'price': quote.get('price', 0),
        'pe_ratio': ratios.get('priceEarningsRatio', 0),
        'debt_ratio': ratios.get('debtRatio', 0),
        'current_ratio': ratios.get('currentRatio', 0),
        'market_cap': quote.get('marketCap', 0)
    }
    
    return features

if __name__ == "__main__":
    # Test with a sample symbol
    data = integrate_fmp_data("AAPL")
    print(f"FMP Data: {data}")