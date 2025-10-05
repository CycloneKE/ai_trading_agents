# 🚀 AI Trading System - Quick Start Guide

## One-Click Startup Options

### Option 1: Windows Batch File (Easiest)
```bash
# Double-click this file or run in terminal:
start.bat
```

### Option 2: Python Script
```bash
python start_trading_system.py
```

### Option 3: Manual Commands (if needed)
```bash
# Terminal 1: API Server
python simple_api.py

# Terminal 2: Frontend (in new terminal)
cd frontend
npm run dev

# Terminal 3: Trading Bot (in new terminal)
python main.py --config config/config.json --mode production
```

## What Happens When You Start

1. **API Server** starts on port 5001
2. **Frontend Dashboard** starts on port 3001
3. **Browser opens** automatically to dashboard
4. **Trading Bot** starts with all strategies
5. **Dashboard shows** live trading data

## Access Points

- **📊 Dashboard**: http://localhost:3001
- **🔌 API**: http://localhost:5001
- **📈 Trading**: Runs automatically in background

## Stopping the System

Press `Ctrl+C` in the terminal to stop all services.

## Troubleshooting

If ports are busy:
- Close other applications using ports 3001 or 5001
- Or restart your computer and try again

**Your complete AI trading system is now ready to run with a single command!**