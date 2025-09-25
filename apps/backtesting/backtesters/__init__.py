"""
Backtesting Systems for QuantAI Trading Platform.

This package contains all backtesting systems:
- Simple Backtester: Basic backtesting with minimal dependencies
- Standalone Backtester: Advanced backtesting without external dependencies
- QF-Lib Backtester: Event-driven backtesting with QF-Lib
- Advanced Quantitative Backtester: Cutting-edge quantitative backtesting
- Focused 5-Ticker Backtester: Specialized for AMZN, META, NVDA, GOOGL, AAPL
"""

__version__ = "1.0.0"
__author__ = "QuantAI Trading Platform"

# Import all backtesters
from .simple_backtest import *
from .standalone_backtest import *
from .qf_lib_backtester import *
from .advanced_quantitative_backtester import *
from .focused_5_ticker_backtester import *
