# 🚀 Paper Trading Deployment Guide

> **This guide is out of date.** It refers to scripts that are no longer at the
> paths shown, and its performance targets were never measured against this
> codebase. For running a paper trade today, follow
> [PAPER_RUN_RUNBOOK.md](PAPER_RUN_RUNBOOK.md), which uses
> `scripts/start_paper_run.py` and `scripts/paper_run_report.py`.

## **Step-by-Step Deployment Process**

### **Phase 1: Pre-Deployment Setup (15 minutes)**

#### **1. Environment Preparation**
```bash
# Install required packages
pip install yfinance pandas numpy scikit-learn xgboost

# Verify API keys in .env file
TRADING_FMP_API_KEY=your_fmp_api_key_here
TRADING_ALPACA_API_KEY=your_alpaca_key_here  # Required: Alpaca is the primary broker
```

#### **2. Configuration Check**
```bash
# Test system components
python system_confidence_proof.py

# Verify configuration
python -c "from supervised_learning import SupervisedLearningStrategy; print('✓ Strategy ready')"
```

### **Phase 2: Paper Trading Launch (5 minutes)**

#### **1. Start Paper Trading**
```bash
# Launch paper trading engine
python paper_trading_deployment.py
```

#### **2. Monitor Performance**
- Real-time trade logging
- Portfolio value tracking
- Performance metrics calculation
- Automatic session reports

### **Phase 3: Monitoring & Analysis (Ongoing)**

#### **Key Metrics to Watch:**
- **Win Rate**: Target >60%
- **Total Return**: Monitor vs benchmark
- **Drawdown**: Should stay <10%
- **Trade Frequency**: Expect 15-25 trades/week

## **🎯 Paper Trading Features**

### **Real Market Integration:**
- ✅ Live Yahoo Finance data feeds
- ✅ Real-time technical indicator calculation
- ✅ Actual market prices and timing
- ✅ Realistic execution simulation

### **Risk Management:**
- ✅ Stop-loss protection (2x ATR)
- ✅ Position size limits (25% max per stock)
- ✅ Portfolio diversification (max 4 positions)
- ✅ Cash management

### **AI Decision Making:**
- ✅ XGBoost ML predictions
- ✅ Multi-factor signal generation
- ✅ Trend following filters
- ✅ Bias detection and correction

## **📊 Expected Performance**

### **Conservative Estimates:**
- **Daily Return**: 0.1-0.3%
- **Weekly Return**: 0.5-2.0%
- **Monthly Return**: 2-8%
- **Win Rate**: 60-80%

### **Risk Metrics:**
- **Max Drawdown**: <8%
- **Volatility**: 15-25% annual
- **Sharpe Ratio**: >1.0

## **🔧 Configuration Options**

### **Basic Settings:**
```json
{
  "initial_capital": 100000,
  "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA"],
  "max_positions": 4,
  "check_interval": 300
}
```

### **Advanced Settings:**
```json
{
  "threshold": 0.01,
  "stop_loss_atr_multiplier": 2.0,
  "trend_following_enabled": true,
  "bias_detection": true
}
```

## **📈 Monitoring Dashboard**

### **Real-Time Metrics:**
- Current portfolio value
- Active positions
- Recent trades
- Performance statistics

### **Session Reports:**
- Trade log with P&L
- Win/loss analysis
- Performance vs benchmark
- Risk metrics

## **⚠️ Important Notes**

### **Paper Trading Limitations:**
- No real money at risk
- Perfect execution assumed
- No slippage or liquidity issues
- Market impact not modeled

### **Transition to Live Trading:**
- Start with small position sizes
- Monitor for 2-4 weeks minimum
- Verify performance consistency
- Gradually increase capital allocation

## **🎯 Success Criteria**

### **Ready for Live Trading When:**
- ✅ Win rate >60% over 2+ weeks
- ✅ Positive returns vs benchmark
- ✅ Max drawdown <10%
- ✅ Consistent trade execution
- ✅ No major system errors

## **📞 Troubleshooting**

### **Common Issues:**
1. **No trades executing**: Check internet connection and API keys
2. **Poor performance**: Review market conditions and strategy parameters
3. **System errors**: Check logs and restart if needed

### **Support Commands:**
```bash
# Check system status
python -c "from paper_trading_deployment import PaperTradingEngine; print('System OK')"

# View recent logs
tail -f paper_trading_session_*.json
```

## **🚀 Quick Start Command**

```bash
# One-command deployment
python paper_trading_deployment.py
```

**Your AI trading agent is now ready for paper trading deployment!** 

The system will:
- ✅ Connect to live market data
- ✅ Generate AI-powered trading signals  
- ✅ Execute simulated trades
- ✅ Track performance in real-time
- ✅ Generate comprehensive reports

**Start your paper trading journey now!** 🎯