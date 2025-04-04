# Cryptocurrency Trading Bot for Delta Exchange

## Overview

This cryptocurrency trading bot is designed for automated trading on Delta Exchange. It analyzes market trends, sentiment, news, and performs technical analysis to execute trades while implementing proper risk management. The bot includes backtesting capabilities and machine learning components that enhance its decision-making over time.

## Features

- **Delta Exchange API Integration**: Complete market data retrieval and trading capabilities
- **Technical Analysis**: Multiple indicators (SMA, EMA, MACD, RSI, Bollinger Bands, etc.) and trading strategies
- **Sentiment Analysis**: News monitoring, crypto-specific lexicon, and Fear & Greed Index integration
- **Risk Management**: Position sizing, stop-loss calculation, drawdown protection, and portfolio risk management
- **Backtesting Framework**: Comprehensive system for testing strategies with historical data
- **Machine Learning**: Various models to enhance trading decisions:
  - Linear models (LinearRegression, Ridge, Lasso, ElasticNet)
  - Tree-based models (RandomForest, XGBoost, GradientBoosting)
  - Deep learning models (LSTM, GRU, CNN) for time series prediction
  - Reinforcement learning models for strategy optimization

## Project Structure

```
crypto_trading_bot/
├── data/                  # Data storage directory
├── docs/                  # Documentation
├── models/                # Saved machine learning models
├── research/              # Research and analysis
├── src/                   # Source code
│   ├── config.py          # Configuration settings
│   ├── delta_client.py    # Delta Exchange API client
│   ├── data_collector.py  # Market data collection
│   ├── data_storage.py    # Data persistence
│   ├── technical_analysis.py # Technical indicators and analysis
│   ├── sentiment_collector.py # Sentiment data collection
│   ├── news_monitor.py    # News monitoring
│   ├── sentiment_integration.py # Sentiment analysis integration
│   ├── trading_strategy.py # Trading strategies
│   ├── risk_manager.py    # Risk management
│   ├── position_manager.py # Position management
│   ├── risk_management_system.py # Integrated risk system
│   ├── ml_base.py         # Machine learning base classes
│   ├── ml_models/         # Machine learning models
│   │   ├── linear_models.py # Linear regression models
│   │   ├── tree_models.py # Tree-based models
│   │   ├── deep_learning_models.py # Deep learning models
│   │   └── reinforcement_learning_models.py # RL models
│   └── trading_bot.py     # Main trading bot
├── tests/                 # Test scripts
│   ├── test_trading_bot.py # Unit and integration tests
│   └── optimize_trading_bot.py # Optimization scripts
├── main.py                # Main entry point
└── requirements.txt       # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto_trading_bot.git
cd crypto_trading_bot
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your API keys:
Edit `src/config.py` and add your Delta Exchange API keys.

## Usage

### Running the Bot

```bash
python main.py --mode run --product_id 1 --timeframe 1h
```

### Backtesting

```bash
python main.py --mode backtest --product_id 1 --timeframe 1h --days 30
```

### Training Machine Learning Models

```bash
python main.py --mode train --product_id 1 --timeframe 1h --days 60
```

### Optimizing Strategy Parameters

```bash
python tests/optimize_trading_bot.py --product_id 1 --timeframe 1h --days 30 --optimize strategy
```

### Command Line Arguments

- `--mode`: Operation mode (`run`, `backtest`, `train`, `optimize`)
- `--product_id`: Delta Exchange product ID to trade
- `--timeframe`: Timeframe for analysis (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`)
- `--trading_enabled`: Enable live trading
- `--risk_per_trade`: Risk percentage per trade
- `--stop_loss_pct`: Stop loss percentage
- `--take_profit_pct`: Take profit percentage
- `--max_open_positions`: Maximum number of open positions
- `--strategy`: Trading strategy name
- `--ml_models_enabled`: Enable machine learning models
- `--sentiment_enabled`: Enable sentiment analysis
- `--news_enabled`: Enable news monitoring
- `--days`: Number of days of historical data to use

## Configuration

Edit `src/config.py` to configure the bot:

```python
# API Configuration
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"
BASE_URL = "https://api.delta.exchange"

# Trading Configuration
DEFAULT_PRODUCT_ID = 1  # BTC-PERP
DEFAULT_TIMEFRAME = "1h"
TRADING_ENABLED = False
RISK_PER_TRADE = 1.0  # 1% of account balance
STOP_LOSS_PCT = 2.0   # 2% stop loss
TAKE_PROFIT_PCT = 4.0 # 4% take profit
MAX_OPEN_POSITIONS = 3

# Strategy Configuration
STRATEGY_NAME = "trend_following"
STRATEGY_PARAMS = {
    "sma_fast": 10,
    "sma_slow": 20,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30
}

# Machine Learning Configuration
ML_MODELS_ENABLED = True
ML_MODEL_TYPES = ["linear", "ridge", "random_forest", "xgboost", "lstm"]

# Sentiment Analysis Configuration
SENTIMENT_ENABLED = True
NEWS_ENABLED = True

# System Configuration
LOG_LEVEL = "INFO"
DATA_DIR = "/path/to/data/directory"
```

## Architecture

The trading bot follows a modular architecture with the following components:

1. **Data Collection**: Retrieves market data from Delta Exchange API
2. **Technical Analysis**: Calculates technical indicators and identifies patterns
3. **Sentiment Analysis**: Analyzes news and social media for market sentiment
4. **Signal Generation**: Combines technical and sentiment analysis to generate trading signals
5. **Risk Management**: Manages position sizing, stop-loss, and take-profit levels
6. **Execution**: Executes trades based on signals and risk parameters
7. **Machine Learning**: Enhances decision-making through predictive models
8. **Backtesting**: Tests strategies on historical data

## Trading Strategies

The bot supports multiple trading strategies:

1. **Trend Following**: Uses moving averages and trend indicators to follow market trends
2. **Mean Reversion**: Identifies overbought/oversold conditions for counter-trend trades
3. **Breakout**: Detects price breakouts from consolidation patterns
4. **Machine Learning**: Uses ML models to predict price movements

## Risk Management

The bot implements comprehensive risk management:

1. **Position Sizing**: Calculates position size based on account balance and risk percentage
2. **Stop Loss**: Sets stop-loss orders to limit potential losses
3. **Take Profit**: Sets take-profit orders to secure gains
4. **Maximum Drawdown**: Monitors and limits portfolio drawdown
5. **Correlation Analysis**: Manages risk across correlated assets

## Machine Learning Models

The bot uses various machine learning models:

1. **Linear Models**: Simple regression models for price prediction
2. **Tree-based Models**: Random Forest and XGBoost for complex pattern recognition
3. **Deep Learning**: LSTM and GRU networks for time series prediction
4. **Reinforcement Learning**: Q-learning and policy gradient methods for strategy optimization

## Testing and Optimization

The bot includes comprehensive testing and optimization tools:

1. **Unit Tests**: Tests for individual components
2. **Integration Tests**: Tests for component interactions
3. **Strategy Optimization**: Grid search for optimal strategy parameters
4. **Risk Parameter Optimization**: Finds optimal risk management settings
5. **Machine Learning Optimization**: Compares and optimizes ML models

## Extending the Bot

### Adding New Strategies

Create a new strategy in `trading_strategy.py`:

```python
def my_custom_strategy(data, params):
    # Implement your strategy logic
    return signals
```

### Adding New Technical Indicators

Add new indicators in `technical_analysis.py`:

```python
def calculate_custom_indicator(data, period):
    # Calculate your indicator
    return indicator_values
```

### Adding New Machine Learning Models

Create a new model class in the appropriate file in `ml_models/`:

```python
class MyCustomModel(BaseMLModel):
    def __init__(self, model_name, model_dir):
        super().__init__(model_name, "regression", model_dir)
        # Initialize your model
    
    def train(self, X, y):
        # Train your model
        
    def predict(self, X):
        # Make predictions
```

## Disclaimer

This trading bot is provided for educational and research purposes only. Trading cryptocurrencies involves significant risk of loss and is not suitable for all investors. Past performance is not indicative of future results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
