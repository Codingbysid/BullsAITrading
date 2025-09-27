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

# Import unified backtesters
from .unified_backtester import (
    FourModelBacktester,
    AdvancedTechnicalBacktester,
    MomentumBacktester,
    MeanReversionBacktester
)

__all__ = [
    'FourModelBacktester',
    'AdvancedTechnicalBacktester',
    'MomentumBacktester',
    'MeanReversionBacktester'
]
