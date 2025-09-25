"""
Pytest configuration and fixtures for the QuantAI Trading Platform.

This module provides shared fixtures and configuration for all test modules.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config.settings import Settings
from src.data.data_sources import DataManager, FinazonDataSource, AlphaVantageDataSource, YFinanceDataSource
from src.data.sentiment_analysis import SentimentAggregator, NewsSentimentAnalyzer
from src.risk.risk_management import RiskManager, KellyCriterion
from src.models.trading_models import RandomForestModel, XGBoostModel, LSTMModel, EnsembleModel
from src.backtesting.backtesting_engine import BacktestingEngine


@pytest.fixture
def test_settings():
    """Test settings with mock API keys."""
    return Settings(
        alpha_vantage_api_key="test_alpha_vantage_key",
        finazon_api_key="test_finazon_key",
        news_api_key="test_news_api_key",
        gemini_api_key="test_gemini_key",
        max_position_size=0.2,
        max_drawdown=0.1,
        target_sharpe_ratio=1.5
    )


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
    volumes = np.random.randint(1000000, 10000000, len(dates))
    
    data = pd.DataFrame({
        'symbol': 'AAPL',
        'open': prices * (1 + np.random.randn(len(dates)) * 0.01),
        'high': prices * (1 + np.abs(np.random.randn(len(dates))) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(len(dates))) * 0.02),
        'close': prices,
        'volume': volumes,
        'source': 'test'
    }, index=dates)
    
    return data


@pytest.fixture
def sample_features():
    """Sample feature data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    features = pd.DataFrame({
        'rsi': np.random.uniform(20, 80, len(dates)),
        'sma_20': np.random.uniform(90, 110, len(dates)),
        'sma_50': np.random.uniform(95, 105, len(dates)),
        'bb_upper': np.random.uniform(105, 115, len(dates)),
        'bb_lower': np.random.uniform(85, 95, len(dates)),
        'macd': np.random.uniform(-2, 2, len(dates)),
        'macd_signal': np.random.uniform(-1, 1, len(dates)),
        'volume_ratio': np.random.uniform(0.5, 2.0, len(dates)),
        'sentiment_score': np.random.uniform(-1, 1, len(dates)),
        'pe_ratio': np.random.uniform(10, 30, len(dates)),
        'pb_ratio': np.random.uniform(1, 5, len(dates))
    }, index=dates)
    
    return features


@pytest.fixture
def sample_returns():
    """Sample returns data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    returns = pd.Series(
        np.random.randn(len(dates)) * 0.02,
        index=dates,
        name='returns'
    )
    
    return returns


@pytest.fixture
def sample_sentiment_data():
    """Sample sentiment data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    sentiment_data = []
    for i, date in enumerate(dates):
        sentiment_data.append({
            'symbol': 'AAPL',
            'timestamp': date,
            'sentiment_score': np.random.uniform(-1, 1),
            'confidence': np.random.uniform(0.5, 1.0),
            'source': 'news_api',
            'text': f'Test news article {i}'
        })
    
    return sentiment_data


@pytest.fixture
def data_manager():
    """Data manager instance for testing."""
    return DataManager()


@pytest.fixture
def sentiment_aggregator():
    """Sentiment aggregator instance for testing."""
    return SentimentAggregator()


@pytest.fixture
def risk_manager():
    """Risk manager instance for testing."""
    return RiskManager()


@pytest.fixture
def kelly_criterion():
    """Kelly criterion instance for testing."""
    return KellyCriterion()


@pytest.fixture
def backtesting_engine():
    """Backtesting engine instance for testing."""
    return BacktestingEngine()


@pytest.fixture
def trading_models():
    """Trading models for testing."""
    return {
        'random_forest': RandomForestModel(),
        'xgboost': XGBoostModel(),
        'lstm': LSTMModel(),
        'ensemble': EnsembleModel()
    }


@pytest.fixture
def event_loop():
    """Event loop for async testing."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Set test environment variables
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'WARNING'
    
    yield
    
    # Cleanup after test
    if 'TESTING' in os.environ:
        del os.environ['TESTING']


class MockAPIClient:
    """Mock API client for testing."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.call_count = 0
    
    async def get_historical_data(self, symbol: str, start_date, end_date):
        """Mock historical data response."""
        self.call_count += 1
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'symbol': symbol,
            'open': np.random.uniform(90, 110, len(dates)),
            'high': np.random.uniform(100, 120, len(dates)),
            'low': np.random.uniform(80, 100, len(dates)),
            'close': np.random.uniform(90, 110, len(dates)),
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'source': 'mock'
        }, index=dates)
        
        return data


@pytest.fixture
def mock_finazon_client():
    """Mock Finazon API client."""
    return MockAPIClient("test_finazon_key")


@pytest.fixture
def mock_alpha_vantage_client():
    """Mock Alpha Vantage API client."""
    return MockAPIClient("test_alpha_vantage_key")


@pytest.fixture
def mock_news_client():
    """Mock News API client."""
    class MockNewsClient:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.call_count = 0
        
        def get_everything(self, **kwargs):
            """Mock news response."""
            self.call_count += 1
            return {
                'status': 'ok',
                'articles': [
                    {
                        'title': f'Test news article {i}',
                        'description': f'Test description {i}',
                        'publishedAt': (datetime.now() - timedelta(days=i)).isoformat(),
                        'source': {'name': f'Test Source {i}'}
                    }
                    for i in range(10)
                ]
            }
    
    return MockNewsClient("test_news_key")


# Test data generators
def generate_test_portfolio_data(n_days: int = 252, n_assets: int = 5) -> pd.DataFrame:
    """Generate test portfolio data."""
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA'][:n_assets]
    
    data = []
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)
        prices = 100 + np.cumsum(np.random.randn(n_days) * 0.02)
        
        for i, date in enumerate(dates):
            data.append({
                'symbol': symbol,
                'date': date,
                'open': prices[i] * (1 + np.random.randn() * 0.01),
                'high': prices[i] * (1 + np.abs(np.random.randn()) * 0.02),
                'low': prices[i] * (1 - np.abs(np.random.randn()) * 0.02),
                'close': prices[i],
                'volume': np.random.randint(1000000, 10000000)
            })
    
    return pd.DataFrame(data)


def generate_test_signals(n_days: int = 252) -> pd.DataFrame:
    """Generate test trading signals."""
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    np.random.seed(42)
    
    signals = pd.DataFrame({
        'date': dates,
        'signal': np.random.choice([-1, 0, 1], n_days, p=[0.2, 0.6, 0.2]),
        'confidence': np.random.uniform(0.5, 1.0, n_days),
        'model': np.random.choice(['rf', 'xgb', 'lstm'], n_days)
    })
    
    return signals


# Performance testing utilities
class PerformanceTracker:
    """Track performance metrics during testing."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = None
    
    def start_timer(self, name: str):
        """Start timing a process."""
        self.start_time = datetime.now()
        self.metrics[name] = {'start': self.start_time}
    
    def end_timer(self, name: str):
        """End timing a process."""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            self.metrics[name]['duration'] = duration
            self.metrics[name]['end'] = datetime.now()
    
    def get_metrics(self) -> Dict:
        """Get all tracked metrics."""
        return self.metrics


@pytest.fixture
def performance_tracker():
    """Performance tracker for testing."""
    return PerformanceTracker()


# Integration test helpers
class IntegrationTestHelper:
    """Helper class for integration tests."""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.sentiment_aggregator = SentimentAggregator()
        self.risk_manager = RiskManager()
        self.backtesting_engine = BacktestingEngine()
    
    async def setup_test_environment(self):
        """Setup test environment for integration tests."""
        # Mock API responses
        pass
    
    async def run_full_pipeline(self, symbol: str = 'AAPL', days: int = 30):
        """Run the full trading pipeline."""
        # 1. Fetch market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        market_data = await self.data_manager.get_market_data(
            symbol=symbol,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        # 2. Get sentiment data
        sentiment = await self.sentiment_aggregator.get_comprehensive_sentiment(
            symbol=symbol,
            lookback_days=days
        )
        
        # 3. Calculate risk metrics
        if not market_data.empty:
            returns = market_data['close'].pct_change().dropna()
            risk_metrics = self.risk_manager.calculate_risk_metrics(returns)
            
            return {
                'market_data': market_data,
                'sentiment': sentiment,
                'risk_metrics': risk_metrics
            }
        
        return None


@pytest.fixture
def integration_helper():
    """Integration test helper."""
    return IntegrationTestHelper()
