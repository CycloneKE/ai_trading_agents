## AI Trading Agent

### Overview
This project is a modular AI-powered trading agent supporting stocks, crypto, and forex. It features multi-broker support, robust error handling, monitoring, and automated trading strategies.

### Quick Start
1. Clone the repository.
2. Install Python 3.13 and required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys.
4. Edit `config/config.json` as needed.
5. Run the agent:
   ```bash
   python main.py --config config/config.json --mode production
   ```

### Configuration
All settings are in `config/config.json`. Key sections:
- `data_manager`: Symbols, update interval, connectors, Redis settings
- `strategies`: Momentum, mean reversion, sentiment, reinforcement
- `risk_management` & `risk_limits`: Portfolio and position risk controls
- `brokers`: Paper, Coinbase, OANDA (use environment variables for secrets)
- `trading`: Initial capital, commission, slippage
- `monitoring`: Enable/disable, port, metrics interval
- `security`: Encryption, API key rotation
- `logging`: Level, format, file, rotation

### Environment Variables
Set these in your `.env` file:
- `COINBASE_API_KEY`, `COINBASE_API_SECRET`, `COINBASE_PASSPHRASE`
- `OANDA_API_KEY`, `OANDA_ACCOUNT_ID`
- `TRADING_ALPHA_VANTAGE_API_KEY`, `TRADING_FMP_API_KEY`, `TRADING_FINNHUB_API_KEY`

### Resource Management
The agent automatically cleans up threads and closes database connections on shutdown. Use `agent.stop()` for graceful termination.

### Monitoring & Logging
Prometheus metrics and log rotation are enabled by default. See `config/config.json` for options.

### Troubleshooting
- Check logs in `logs/trading_agent.log` for errors.
- Ensure all required secrets are set in `.env`.
- Validate your config with `--debug` mode.

### License
MIT# AI Trading Agent with Adaptive Capabilities

A self-adaptive, goal-oriented AI trading system that learns and evolves based on market conditions and performance feedback.

## Key Features

- **Self-Adaptive Agent**: Automatically adjusts strategies based on performance
- **Goal-Oriented Behavior**: Sets and pursues dynamic trading goals
- **Supervised Learning**: ML-based trading strategies with continuous learning
- **Risk Management**: Comprehensive risk assessment and position sizing
- **Real-time Monitoring**: Performance tracking and health checks

## Core Components

- `main.py` - Main application entry point
- `supervised_learning.py` - Enhanced ML strategy with adaptive capabilities
- `src/adaptive_agent.py` - Self-adaptive agent core
- `src/goal_manager.py` - Dynamic goal management
- `src/adaptive_integration.py` - Integration layer

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create configuration:
   ```bash
   python main.py --create-config
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Run the agent:
   ```bash
   python main.py
   ```

## Configuration

Edit `config/config.json` to customize:
- Trading strategies and parameters
- Risk management settings
- Data sources and symbols
- Adaptive agent behavior

## Adaptive Features

The agent automatically:
- Adjusts position sizes based on market volatility
- Modifies strategy parameters based on performance
- Sets and pursues dynamic goals
- Adapts to changing market conditions

## Monitoring

Access the monitoring dashboard at http://localhost:8080 when running.

## License

MIT License
