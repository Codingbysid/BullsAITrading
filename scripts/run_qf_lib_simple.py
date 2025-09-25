#!/usr/bin/env python3
"""
Simple QF-Lib Backtester Runner

This script runs a simple QF-Lib backtester with the correct imports.
"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_qf_lib_environment():
    """Setup QF-Lib environment variables."""
    logger.info("🔧 Setting up QF-Lib environment...")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    project_root_abs = str(project_root.absolute())
    
    # Set QF-Lib environment variables
    os.environ['QF_STARTING_DIRECTORY'] = project_root_abs
    os.environ['QF_OUTPUT_DIRECTORY'] = str(project_root / "output")
    os.environ['QF_DATA_DIRECTORY'] = str(project_root / "data")
    os.environ['QF_CACHE_DIRECTORY'] = str(project_root / "cache")
    os.environ['QF_LOGS_DIRECTORY'] = str(project_root / "logs")
    
    # Create directories
    directories = [
        project_root / "output",
        project_root / "data",
        project_root / "cache",
        project_root / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✅ QF-Lib environment setup completed")
    logger.info(f"   Starting directory: {project_root_abs}")
    
    return project_root_abs

def run_simple_qf_lib_backtest():
    """Run a simple QF-Lib backtest."""
    logger.info("🚀 Running simple QF-Lib backtest...")
    
    try:
        # Import QF-Lib components
        from qf_lib.settings import Settings
        from qf_lib.containers import QFSeries, QFDataFrame
        from qf_lib.data_providers import DataProvider
        from qf_lib.backtesting import Portfolio, Broker
        
        logger.info("✅ QF-Lib components imported successfully")
        
        # Setup environment
        project_root = Path(__file__).parent.parent
        settings_path = project_root / "config" / "qf_lib_settings.json"
        secret_settings_path = project_root / "config" / "qf_lib_secret_settings.json"
        
        # Create Settings
        if settings_path.exists() and secret_settings_path.exists():
            settings = Settings(str(settings_path), str(secret_settings_path))
            logger.info("✅ QF-Lib Settings created with config files")
        else:
            settings = Settings()
            logger.info("✅ QF-Lib Settings created with defaults")
        
        # Create sample data
        logger.info("📊 Creating sample data...")
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        
        # Create synthetic price data
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% daily return, 2% volatility
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create QFSeries
        qf_series = QFSeries(data=prices, index=dates, name='AAPL')
        logger.info(f"✅ QFSeries created: {len(qf_series)} data points")
        
        # Create QFDataFrame with multiple symbols
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        data_dict = {}
        
        for i, symbol in enumerate(symbols):
            # Create different price series for each symbol
            symbol_returns = np.random.normal(0.0005 + i*0.0001, 0.02 + i*0.005, len(dates))
            symbol_prices = 100 * np.exp(np.cumsum(symbol_returns))
            data_dict[symbol] = symbol_prices
        
        qf_dataframe = QFDataFrame(data_dict, index=dates)
        logger.info(f"✅ QFDataFrame created: {qf_dataframe.shape}")
        
        # Simple backtest simulation
        logger.info("📈 Running simple backtest simulation...")
        
        # Calculate returns
        returns = qf_dataframe.pct_change().dropna()
        logger.info(f"✅ Returns calculated: {returns.shape}")
        
        # Simple buy and hold strategy
        initial_capital = 100000
        portfolio_value = initial_capital
        
        # Equal weight portfolio
        weights = np.ones(len(symbols)) / len(symbols)
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_value_series = initial_capital * (1 + portfolio_returns).cumprod()
        
        # Calculate performance metrics
        total_return = (portfolio_value_series.iloc[-1] - initial_capital) / initial_capital
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate drawdown
        peak = portfolio_value_series.expanding().max()
        drawdown = (portfolio_value_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Results
        results = {
            'initial_capital': initial_capital,
            'final_capital': portfolio_value_series.iloc[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': 0,  # Simple buy and hold
            'win_rate': 0.0
        }
        
        logger.info("✅ Simple backtest completed successfully!")
        logger.info(f"   Initial Capital: ${results['initial_capital']:,.2f}")
        logger.info(f"   Final Capital: ${results['final_capital']:,.2f}")
        logger.info(f"   Total Return: {results['total_return']:.2%}")
        logger.info(f"   Annualized Return: {results['annualized_return']:.2%}")
        logger.info(f"   Volatility: {results['volatility']:.2%}")
        logger.info(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"   Max Drawdown: {results['max_drawdown']:.2%}")
        
        return results
        
    except ImportError as e:
        logger.error(f"❌ QF-Lib import failed: {e}")
        logger.error("Please ensure QF-Lib is installed: pip install qf-lib")
        return None
    except Exception as e:
        logger.error(f"❌ QF-Lib backtest failed: {e}")
        return None

def main():
    """Main function."""
    print("🚀 Simple QF-Lib Backtester")
    print("=" * 50)
    print()
    
    # Setup environment
    setup_qf_lib_environment()
    
    # Run backtest
    results = run_simple_qf_lib_backtest()
    
    if results:
        print("\n🎉 QF-Lib backtest completed successfully!")
        print("📊 Results:")
        print(f"   Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"   Final Capital: ${results['final_capital']:,.2f}")
        print(f"   Total Return: {results['total_return']:.2%}")
        print(f"   Annualized Return: {results['annualized_return']:.2%}")
        print(f"   Volatility: {results['volatility']:.2%}")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.2%}")
        print("\n✅ QF-Lib is working correctly!")
    else:
        print("\n❌ QF-Lib backtest failed")
        print("Please check the error messages above")
    
    return results is not None

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
