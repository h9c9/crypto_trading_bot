"""
Main script to run the cryptocurrency trading bot.
"""
import os
import sys
import logging
import argparse
import json
from datetime import datetime

from src.trading_bot import TradingBot
from src.config import (
    API_KEY, API_SECRET, BASE_URL, DEFAULT_PRODUCT_ID, 
    DEFAULT_TIMEFRAME, TRADING_ENABLED, RISK_PER_TRADE,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, MAX_OPEN_POSITIONS,
    STRATEGY_NAME, STRATEGY_PARAMS, ML_MODELS_ENABLED,
    ML_MODEL_TYPES, SENTIMENT_ENABLED, NEWS_ENABLED,
    LOG_LEVEL, DATA_DIR
)

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, f"main_{datetime.now().strftime('%Y%m%d')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Bot')
    
    parser.add_argument('--mode', type=str, default='run',
                        choices=['run', 'backtest', 'train', 'optimize'],
                        help='Mode to run the bot in')
    
    parser.add_argument('--product_id', type=int, default=DEFAULT_PRODUCT_ID,
                        help='Product ID to trade')
    
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME,
                        help='Timeframe for analysis')
    
    parser.add_argument('--trading_enabled', action='store_true',
                        default=TRADING_ENABLED,
                        help='Enable live trading')
    
    parser.add_argument('--risk_per_trade', type=float, default=RISK_PER_TRADE,
                        help='Risk percentage per trade')
    
    parser.add_argument('--stop_loss_pct', type=float, default=STOP_LOSS_PCT,
                        help='Stop loss percentage')
    
    parser.add_argument('--take_profit_pct', type=float, default=TAKE_PROFIT_PCT,
                        help='Take profit percentage')
    
    parser.add_argument('--max_open_positions', type=int, default=MAX_OPEN_POSITIONS,
                        help='Maximum number of open positions')
    
    parser.add_argument('--strategy', type=str, default=STRATEGY_NAME,
                        help='Trading strategy name')
    
    parser.add_argument('--strategy_params', type=str, default=json.dumps(STRATEGY_PARAMS),
                        help='Trading strategy parameters as JSON string')
    
    parser.add_argument('--ml_models_enabled', action='store_true',
                        default=ML_MODELS_ENABLED,
                        help='Enable machine learning models')
    
    parser.add_argument('--ml_model_types', type=str, default=','.join(ML_MODEL_TYPES),
                        help='Types of machine learning models to use (comma-separated)')
    
    parser.add_argument('--sentiment_enabled', action='store_true',
                        default=SENTIMENT_ENABLED,
                        help='Enable sentiment analysis')
    
    parser.add_argument('--news_enabled', action='store_true',
                        default=NEWS_ENABLED,
                        help='Enable news monitoring')
    
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days of historical data to use for backtest/train/optimize')
    
    return parser.parse_args()

def main():
    """
    Main function to run the trading bot.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Parse strategy parameters
    try:
        strategy_params = json.loads(args.strategy_params)
    except json.JSONDecodeError:
        logger.error(f"Invalid strategy parameters: {args.strategy_params}")
        strategy_params = STRATEGY_PARAMS
    
    # Parse ML model types
    ml_model_types = args.ml_model_types.split(',')
    
    # Create trading bot
    bot = TradingBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        base_url=BASE_URL,
        product_id=args.product_id,
        timeframe=args.timeframe,
        trading_enabled=args.trading_enabled,
        risk_per_trade=args.risk_per_trade,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        max_open_positions=args.max_open_positions,
        strategy_name=args.strategy,
        strategy_params=strategy_params,
        ml_models_enabled=args.ml_models_enabled,
        ml_model_types=ml_model_types,
        sentiment_enabled=args.sentiment_enabled,
        news_enabled=args.news_enabled,
        data_dir=DATA_DIR
    )
    
    # Run in specified mode
    if args.mode == 'run':
        # Train models if enabled
        if args.ml_models_enabled:
            bot.train_models(days=args.days)
        
        # Start bot
        bot.start()
        
        try:
            # Keep main thread alive
            import time
            while True:
                time.sleep(60)
                
                # Print status
                status = bot.get_status()
                logger.info(f"Bot status: Running={status['running']}, Last update={status['last_update']}")
                
                # Check if bot is still running
                if not status['running']:
                    logger.warning("Bot is not running, restarting...")
                    bot.start()
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, stopping bot")
            bot.stop()
        
        except Exception as e:
            logger.error(f"Error in main: {e}")
            import traceback
            logger.error(traceback.format_exc())
            bot.stop()
    
    elif args.mode == 'backtest':
        # Run backtest
        results = bot.backtest(days=args.days)
        
        # Print results
        if results.get('success', False):
            logger.info(f"Backtest results:")
            logger.info(f"Return: {results['return_pct']:.2f}%")
            logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            logger.info(f"Max Drawdown: {results['max_drawdown']:.2f}%")
            logger.info(f"Win Rate: {results['win_rate']:.2f}%")
            logger.info(f"Total Trades: {len(results['trades'])}")
        else:
            logger.error(f"Backtest failed: {results.get('error', 'Unknown error')}")
    
    elif args.mode == 'train':
        # Train models
        bot.train_models(days=args.days)
    
    elif args.mode == 'optimize':
        # Optimize strategy
        results = bot.optimize_strategy(days=args.days)
        
        # Print results
        if results.get('success', False):
            logger.info(f"Optimization results:")
            logger.info(f"Best parameters: {results['best_params']}")
            logger.info(f"Best Return: {results['best_return']:.2f}%")
            logger.info(f"Best Sharpe Ratio: {results['best_sharpe']:.2f}")
        else:
            logger.error(f"Optimization failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
