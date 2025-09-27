"""
Common imports and utilities for the QuantAI Trading Platform.

This module provides standardized imports and common utilities to eliminate
code duplication across the codebase.
"""

# Standard library imports
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
import warnings

# Third-party imports
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
def setup_logger(name: str) -> logging.Logger:
    """Setup standardized logger for a module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# Common type aliases
PerformanceMetrics = Dict[str, float]
TradingSignal = Dict[str, Any]
PortfolioState = Dict[str, Any]
RiskMetrics = Dict[str, float]
ModelOutput = Dict[str, Any]

# Common constants
DEFAULT_LOOKBACK_PERIOD = 252
DEFAULT_RISK_FREE_RATE = 0.02
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_MAX_POSITION_SIZE = 0.2
DEFAULT_MAX_DRAWDOWN = 0.1

# Common utility functions
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate percentage returns from price series."""
    return prices.pct_change().dropna()

def annualize_metric(metric: float, periods: int, frequency: int = 252) -> float:
    """Annualize a metric based on the number of periods."""
    return metric * (frequency / periods) if periods > 0 else 0.0

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal as a percentage string."""
    return f"{value:.{decimals}%}"

def format_currency(value: float, decimals: int = 2) -> str:
    """Format a number as currency string."""
    return f"${value:,.{decimals}f}"

def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save data to JSON file with error handling."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logging.error(f"Failed to save JSON to {filepath}: {e}")

def load_json(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load data from JSON file with error handling."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON from {filepath}: {e}")
        return None

# Common data validation
def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame has required columns."""
    return all(col in df.columns for col in required_columns)

def validate_series(series: pd.Series, min_length: int = 1) -> bool:
    """Validate that Series meets minimum length requirement."""
    return len(series) >= min_length and not series.empty

# Common mathematical operations
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> float:
    """Calculate Sharpe ratio from returns series."""
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns.mean() * 252 - risk_free_rate
    volatility = returns.std() * np.sqrt(252)
    return safe_divide(excess_returns, volatility)

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> float:
    """Calculate Sortino ratio from returns series."""
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns.mean() * 252 - risk_free_rate
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
    return safe_divide(excess_returns, downside_deviation)

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns series."""
    if len(returns) == 0:
        return 0.0
    
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def calculate_calmar_ratio(returns: pd.Series) -> float:
    """Calculate Calmar ratio from returns series."""
    if len(returns) == 0:
        return 0.0
    
    annual_return = returns.mean() * 252
    max_dd = abs(calculate_max_drawdown(returns))
    return safe_divide(annual_return, max_dd)

def calculate_win_rate(returns: pd.Series) -> float:
    """Calculate win rate from returns series."""
    if len(returns) == 0:
        return 0.0
    
    winning_trades = (returns > 0).sum()
    total_trades = len(returns)
    return safe_divide(winning_trades, total_trades)

def calculate_profit_factor(returns: pd.Series) -> float:
    """Calculate profit factor from returns series."""
    if len(returns) == 0:
        return 0.0
    
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    return safe_divide(gross_profit, gross_loss)

# Common performance metrics calculation
def calculate_performance_metrics(returns: pd.Series, initial_value: float = 1.0) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics from returns series."""
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    
    # Risk-adjusted metrics
    sharpe_ratio = calculate_sharpe_ratio(returns)
    sortino_ratio = calculate_sortino_ratio(returns)
    calmar_ratio = calculate_calmar_ratio(returns)
    max_drawdown = calculate_max_drawdown(returns)
    
    # Trade statistics
    win_rate = calculate_win_rate(returns)
    profit_factor = calculate_profit_factor(returns)
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }

# Common data processing utilities
def resample_data(data: pd.DataFrame, frequency: str = 'D') -> pd.DataFrame:
    """Resample data to specified frequency."""
    if 'Date' in data.columns:
        data = data.set_index('Date')
    
    resampled = data.resample(frequency).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return resampled.reset_index()

def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to price data."""
    df = data.copy()
    
    # Moving averages
    df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
    bb_std = df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price momentum
    df['Price_Change_1D'] = df['Close'].pct_change(1)
    df['Price_Change_5D'] = df['Close'].pct_change(5)
    df['Price_Change_20D'] = df['Close'].pct_change(20)
    
    # Volatility
    df['Volatility_20D'] = df['Close'].pct_change().rolling(window=20, min_periods=1).std()
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(0)
    
    return df

# Common validation utilities
def validate_trading_signal(signal: Dict[str, Any]) -> bool:
    """Validate trading signal structure."""
    required_keys = ['signal', 'strength', 'reasoning']
    return all(key in signal for key in required_keys)

def validate_portfolio_state(state: Dict[str, Any]) -> bool:
    """Validate portfolio state structure."""
    required_keys = ['cash', 'positions', 'total_value']
    return all(key in state for key in required_keys)

# Common error handling
class QuantAIError(Exception):
    """Base exception for QuantAI Trading Platform."""
    pass

class DataError(QuantAIError):
    """Exception for data-related errors."""
    pass

class ModelError(QuantAIError):
    """Exception for model-related errors."""
    pass

class RiskError(QuantAIError):
    """Exception for risk management errors."""
    pass

# Common decorators
def handle_errors(default_return=None):
    """Decorator to handle common errors with default return value."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator

def validate_inputs(*validators):
    """Decorator to validate function inputs."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for validator in validators:
                if not validator(*args, **kwargs):
                    raise ValueError(f"Input validation failed for {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
