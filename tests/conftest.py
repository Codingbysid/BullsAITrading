"""
Pytest configuration and fixtures for QuantAI Trading Platform.

This module provides common fixtures and configuration for all tests.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tempfile
import os
from pathlib import Path

# Test data fixtures
@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)  # For reproducible tests
    
    data = {
        'date': dates,
        'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01) + np.random.rand(len(dates)) * 2,
        'low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01) - np.random.rand(len(dates)) * 2,
        'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01),
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df

@pytest.fixture
def sample_portfolio_data():
    """Generate sample portfolio data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    returns = np.random.normal(0.0005, 0.02, len(dates))
    portfolio_values = 100000 * np.exp(np.cumsum(returns))
    
    data = {
        'date': dates,
        'portfolio_value': portfolio_values,
        'cash': 10000 + np.random.randn(len(dates)) * 1000,
        'positions': np.random.randint(0, 5, len(dates))
    }
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df

@pytest.fixture
def sample_trading_signals():
    """Generate sample trading signals for testing."""
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    signals = np.random.choice([-1, 0, 1], len(dates), p=[0.2, 0.6, 0.2])
    signal_strength = np.random.rand(len(dates))
    confidence = np.random.rand(len(dates))
    
    data = {
        'date': dates,
        'signal': signals,
        'signal_strength': signal_strength,
        'confidence': confidence,
        'rsi': np.random.uniform(20, 80, len(dates)),
        'bb_position': np.random.uniform(0, 1, len(dates)),
        'momentum': np.random.uniform(-0.1, 0.1, len(dates))
    }
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df

@pytest.fixture
def sample_market_data():
    """Generate sample market data for multiple symbols."""
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    market_data = {}
    for symbol in symbols:
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = {
            'date': dates,
            'open': prices + np.random.randn(len(dates)) * 0.5,
            'high': prices + np.random.rand(len(dates)) * 2,
            'low': prices - np.random.rand(len(dates)) * 2,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        market_data[symbol] = df
    
    return market_data

@pytest.fixture
def temp_db_path():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)

@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'trading': {
            'initial_capital': 100000,
            'commission': 0.001,
            'slippage': 0.0005,
            'max_position_size': 0.1
        },
        'risk': {
            'max_portfolio_drawdown': 0.15,
            'max_ticker_drawdown': 0.20,
            'var_confidence': 0.05,
            'kelly_max_fraction': 0.25
        },
        'data': {
            'start_date': '2020-01-01',
            'end_date': '2024-12-31',
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        }
    }

@pytest.fixture
def mock_api_responses():
    """Mock API responses for testing."""
    return {
        'alpha_vantage': {
            'AAPL': {
                'Meta Data': {
                    '1. Information': 'Daily Prices (open, high, low, close) and Volumes',
                    '2. Symbol': 'AAPL',
                    '3. Last Refreshed': '2024-12-31',
                    '4. Output Size': 'Full size',
                    '5. Time Zone': 'US/Eastern'
                },
                'Time Series (Daily)': {
                    '2024-12-31': {
                        '1. open': '150.00',
                        '2. high': '155.00',
                        '3. low': '148.00',
                        '4. close': '152.00',
                        '5. volume': '50000000'
                    }
                }
            }
        },
        'news_api': {
            'articles': [
                {
                    'title': 'Apple stock rises on strong earnings',
                    'description': 'Apple reported better than expected earnings...',
                    'url': 'https://example.com/apple-earnings',
                    'publishedAt': '2024-12-31T10:00:00Z',
                    'source': {'name': 'Financial News'}
                }
            ]
        }
    }

# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")

# Test utilities
class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_returns(n_days: int, mean: float = 0.0005, std: float = 0.02) -> pd.Series:
        """Generate random returns for testing."""
        np.random.seed(42)
        returns = np.random.normal(mean, std, n_days)
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        return pd.Series(returns, index=dates)
    
    @staticmethod
    def generate_prices(initial_price: float = 100, n_days: int = 252) -> pd.Series:
        """Generate price series from returns."""
        returns = TestDataGenerator.generate_returns(n_days)
        prices = initial_price * np.exp(np.cumsum(returns))
        return prices
    
    @staticmethod
    def generate_portfolio(initial_capital: float = 100000, n_days: int = 252) -> pd.DataFrame:
        """Generate portfolio data for testing."""
        returns = TestDataGenerator.generate_returns(n_days)
        portfolio_values = initial_capital * np.exp(np.cumsum(returns))
        
        data = {
            'portfolio_value': portfolio_values,
            'cash': initial_capital * 0.1 + np.random.randn(n_days) * 1000,
            'positions': np.random.randint(0, 5, n_days)
        }
        
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        df = pd.DataFrame(data, index=dates)
        return df

# Test assertions
class TestAssertions:
    """Custom assertions for trading tests."""
    
    @staticmethod
    def assert_valid_returns(returns: pd.Series):
        """Assert that returns are valid."""
        assert not returns.isna().any(), "Returns contain NaN values"
        assert returns.index.is_monotonic_increasing, "Returns index is not monotonic"
        assert returns.dtype in [np.float64, np.float32], "Returns are not numeric"
    
    @staticmethod
    def assert_valid_portfolio(portfolio: pd.DataFrame):
        """Assert that portfolio data is valid."""
        required_columns = ['portfolio_value', 'cash', 'positions']
        for col in required_columns:
            assert col in portfolio.columns, f"Missing required column: {col}"
        
        assert portfolio['portfolio_value'].min() > 0, "Portfolio value must be positive"
        assert portfolio['cash'].min() >= 0, "Cash cannot be negative"
        assert portfolio['positions'].min() >= 0, "Positions cannot be negative"
    
    @staticmethod
    def assert_valid_signals(signals: pd.DataFrame):
        """Assert that trading signals are valid."""
        assert 'signal' in signals.columns, "Missing signal column"
        assert signals['signal'].isin([-1, 0, 1]).all(), "Signals must be -1, 0, or 1"
        
        if 'signal_strength' in signals.columns:
            assert (signals['signal_strength'] >= 0).all(), "Signal strength must be non-negative"
            assert (signals['signal_strength'] <= 1).all(), "Signal strength must be <= 1"
        
        if 'confidence' in signals.columns:
            assert (signals['confidence'] >= 0).all(), "Confidence must be non-negative"
            assert (signals['confidence'] <= 1).all(), "Confidence must be <= 1"