"""
Backtesting Systems for QuantAI Trading Platform.

This package contains all backtesting systems:
- Simple Backtester: Basic backtesting with minimal dependencies
- Standalone Backtester: Advanced backtesting without external dependencies
- Advanced Quantitative Backtester: Cutting-edge quantitative backtesting
- Focused 5-Ticker Backtester: Specialized for AMZN, META, NVDA, GOOGL, AAPL
"""

__version__ = "1.0.0"
__author__ = "QuantAI Trading Platform"

# Import all backtesters
from .simple_backtest import SimpleBacktester
from .standalone_backtest import StandaloneBacktester
from .advanced_quantitative_backtester import AdvancedQuantitativeBacktester
from .focused_5_ticker_backtester import Focused5TickerBacktester

__all__ = [
    'SimpleBacktester',
    'StandaloneBacktester', 
    'AdvancedQuantitativeBacktester',
    'Focused5TickerBacktester'
]
