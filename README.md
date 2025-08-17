# AI Trading Agent with Adaptive Capabilities

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
