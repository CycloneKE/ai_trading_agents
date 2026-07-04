import json
import logging
from datetime import datetime
import os
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Tracks and logs the performance of the trading agent over time.
    Calculates win rates, average profits, and tracks individual strategy efficacy.
    """
    
    def __init__(self, log_dir: str = "data"):
        self.log_dir = log_dir
        self.log_file = os.path.join(self.log_dir, "performance_log.json")
        self.trades = []
        self.daily_performance = {}
        
        # Ensure directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        self._load_history()

    def _load_history(self):
        """Loads historical performance data from the log file."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.trades = data.get('trades', [])
                    self.daily_performance = data.get('daily_performance', {})
            except Exception as e:
                logger.error(f"Failed to load performance history: {e}")
                self.trades = []
                self.daily_performance = {}

    def _save_history(self):
        """Saves current state to the log file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump({
                    "trades": self.trades,
                    "daily_performance": self.daily_performance,
                    "last_updated": datetime.utcnow().isoformat()
                }, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save performance history: {e}")

    def log_trade(self, symbol: str, action: str, price: float, quantity: float, strategy: str, result: float = None):
        """Logs an individual trade and its realized profit/loss."""
        trade = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "action": action,
            "price": price,
            "quantity": quantity,
            "strategy": strategy,
            "result": result
        }
        self.trades.append(trade)
        
        # Maintain a manageable list of recent trades
        if len(self.trades) > 1000:
            self.trades = self.trades[-1000:]
            
        self._save_history()

    def update_daily_summary(self, date_str: str, portfolio_value: float, day_pnl: float):
        """Updates the daily summary snapshot with portfolio PnL."""
        self.daily_performance[date_str] = {
            "portfolio_value": portfolio_value,
            "day_pnl": day_pnl,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._save_history()

    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Calculates win rates and average profits for each strategy."""
        strategy_stats = {}
        
        for trade in self.trades:
            if trade.get("result") is not None:
                strat = trade.get("strategy", "unknown")
                if strat not in strategy_stats:
                    strategy_stats[strat] = {"wins": 0, "losses": 0, "total_pnl": 0.0}
                
                res = trade["result"]
                if res > 0:
                    strategy_stats[strat]["wins"] += 1
                else:
                    strategy_stats[strat]["losses"] += 1
                
                strategy_stats[strat]["total_pnl"] += res
                
        # Calculate win rates
        for strat, stats in strategy_stats.items():
            total_trades = stats["wins"] + stats["losses"]
            stats["win_rate"] = stats["wins"] / total_trades if total_trades > 0 else 0.0
            
        return strategy_stats
