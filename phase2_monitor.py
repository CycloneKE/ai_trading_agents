"""
Phase 2 Performance Monitor
"""

import json
import os
from datetime import datetime
# import matplotlib.pyplot as plt  # Optional for plotting

def analyze_phase2_performance():
    """Analyze Phase 2 trading performance."""
    
    print("PHASE 2 PERFORMANCE ANALYSIS")
    print("=" * 40)
    
    # Load portfolio
    if os.path.exists('phase2_portfolio.json'):
        with open('phase2_portfolio.json', 'r') as f:
            portfolio = json.load(f)
        
        print(f"\nCURRENT PORTFOLIO:")
        print(f"Cash: ${portfolio['cash']:,.2f}")
        print(f"Positions: {len(portfolio['positions'])}")
        
        for symbol, pos in portfolio['positions'].items():
            print(f"  {symbol}: {pos['shares']} shares @ ${pos['avg_price']:.2f}")
        
        print(f"\nTRADE HISTORY: {len(portfolio['trade_history'])} trades")
        
        # Analyze trades by symbol
        trades_by_symbol = {}
        for trade in portfolio['trade_history']:
            symbol = trade['symbol']
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)
        
        print(f"\nTRADES BY SYMBOL:")
        for symbol, trades in trades_by_symbol.items():
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            print(f"  {symbol}: {len(buy_trades)} buys")
    
    # Load performance history
    if os.path.exists('phase2_performance.json'):
        with open('phase2_performance.json', 'r') as f:
            performance = json.load(f)
        
        print(f"\nPERFORMANCE HISTORY: {len(performance)} sessions")
        
        if performance:
            latest = performance[-1]
            print(f"Latest Portfolio Value: ${latest['portfolio_value']:,.2f}")
            print(f"Latest Return: {latest['total_return_pct']:+.2f}%")
            print(f"Win Rate: {latest['win_rate']:.1%}")
    
    # AI Decision Analysis
    print(f"\nAI DECISION ANALYSIS:")
    if os.path.exists('phase2_portfolio.json'):
        high_conviction_trades = [t for t in portfolio['trade_history'] if t['conviction'] in ['HIGH', 'VERY HIGH']]
        medium_conviction_trades = [t for t in portfolio['trade_history'] if t['conviction'] == 'MEDIUM']
        
        print(f"High Conviction Trades: {len(high_conviction_trades)}")
        print(f"Medium Conviction Trades: {len(medium_conviction_trades)}")
        
        # Average scores
        if portfolio['trade_history']:
            avg_genius_score = sum([t['genius_score'] for t in portfolio['trade_history']]) / len(portfolio['trade_history'])
            avg_enhanced_score = sum([t['enhanced_score'] for t in portfolio['trade_history']]) / len(portfolio['trade_history'])
            
            print(f"Average Genius Score: {avg_genius_score:.1f}/10")
            print(f"Average Enhanced Score: {avg_enhanced_score:.1f}/10")
    
    print(f"\nRECOMMENDations:")
    print("1. Continue running daily for 2 weeks")
    print("2. Monitor which stocks perform best")
    print("3. Adjust scoring thresholds based on results")
    print("4. Track AI accuracy vs actual market moves")

def create_performance_report():
    """Create detailed performance report."""
    
    if not os.path.exists('phase2_performance.json'):
        print("No performance data found. Run phase2_daily_trading.py first.")
        return
    
    with open('phase2_performance.json', 'r') as f:
        performance = json.load(f)
    
    report = {
        'summary': {
            'total_sessions': len(performance),
            'start_date': performance[0]['date'] if performance else None,
            'end_date': performance[-1]['date'] if performance else None,
            'final_portfolio_value': performance[-1]['portfolio_value'] if performance else 100000,
            'total_return': performance[-1]['total_return_pct'] if performance else 0,
            'best_day': max(performance, key=lambda x: x['total_return_pct']) if performance else None,
            'worst_day': min(performance, key=lambda x: x['total_return_pct']) if performance else None
        },
        'daily_performance': performance
    }
    
    with open('phase2_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Performance report saved to phase2_report.json")
    
    return report

if __name__ == '__main__':
    analyze_phase2_performance()
    create_performance_report()