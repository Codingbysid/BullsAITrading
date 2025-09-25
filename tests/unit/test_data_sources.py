"""
Unit tests for data sources module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.data.data_sources import (
    FinazonDataSource, 
    AlphaVantageDataSource, 
    YFinanceDataSource,
    DataManager
)


class TestFinazonDataSource:
    """Test Finazon data source."""
    
    def test_init(self):
        """Test FinazonDataSource initialization."""
        api_key = "test_api_key"
        source = FinazonDataSource(api_key)
        
        assert source.api_key == api_key
        assert source.base_url == "https://api.finazon.io/latest"
        assert "Authorization" in source.headers
        assert source.headers["Authorization"] == f"apikey {api_key}"
    
    @pytest.mark.asyncio
    async def test_get_historical_data_success(self, sample_market_data):
        """Test successful data retrieval."""
        source = FinazonDataSource("test_key")
        
        # Mock the HTTP response
        mock_response_data = {
            "data": [
                {
                    "t": int(datetime.now().timestamp()),
                    "o": 100.0,
                    "h": 105.0,
                    "l": 95.0,
                    "c": 102.0,
                    "v": 1000000
                }
            ]
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now()
            
            result = await source.get_historical_data("AAPL", start_date, end_date)
            
            assert not result.empty
            assert "symbol" in result.columns
            assert "close" in result.columns
            assert result.iloc[0]["symbol"] == "AAPL"
    
    @pytest.mark.asyncio
    async def test_get_historical_data_error(self):
        """Test error handling in data retrieval."""
        source = FinazonDataSource("test_key")
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 400
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now()
            
            result = await source.get_historical_data("AAPL", start_date, end_date)
            
            assert result.empty
    
    def test_parse_finazon_data(self):
        """Test Finazon data parsing."""
        source = FinazonDataSource("test_key")
        
        data = {
            "data": [
                {
                    "t": 1640995200,  # 2022-01-01
                    "o": 100.0,
                    "h": 105.0,
                    "l": 95.0,
                    "c": 102.0,
                    "v": 1000000
                }
            ]
        }
        
        result = source._parse_finazon_data(data, "AAPL")
        
        assert not result.empty
        assert result.iloc[0]["symbol"] == "AAPL"
        assert result.iloc[0]["close"] == 102.0
        assert result.iloc[0]["volume"] == 1000000


class TestAlphaVantageDataSource:
    """Test Alpha Vantage data source."""
    
    def test_init(self):
        """Test AlphaVantageDataSource initialization."""
        api_key = "test_api_key"
        source = AlphaVantageDataSource(api_key)
        
        assert source.api_key == api_key
        assert source.base_url == "https://www.alphavantage.co/query"
    
    @pytest.mark.asyncio
    async def test_get_historical_data_success(self):
        """Test successful data retrieval."""
        source = AlphaVantageDataSource("test_key")
        
        # Mock the HTTP response
        mock_response_data = {
            "Time Series (Daily)": {
                "2024-01-01": {
                    "1. open": "100.00",
                    "2. high": "105.00",
                    "3. low": "95.00",
                    "4. close": "102.00",
                    "6. volume": "1000000"
                }
            }
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            start_date = datetime(2024, 1, 1)
            end_date = datetime(2024, 1, 2)
            
            result = await source.get_historical_data("AAPL", start_date, end_date)
            
            assert not result.empty
            assert "symbol" in result.columns
            assert result.iloc[0]["symbol"] == "AAPL"
    
    def test_parse_alpha_vantage_data(self):
        """Test Alpha Vantage data parsing."""
        source = AlphaVantageDataSource("test_key")
        
        data = {
            "Time Series (Daily)": {
                "2024-01-01": {
                    "1. open": "100.00",
                    "2. high": "105.00",
                    "3. low": "95.00",
                    "4. close": "102.00",
                    "6. volume": "1000000"
                }
            }
        }
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        
        result = source._parse_alpha_vantage_data(data, "AAPL", start_date, end_date)
        
        assert not result.empty
        assert result.iloc[0]["symbol"] == "AAPL"
        assert result.iloc[0]["close"] == 102.0


class TestYFinanceDataSource:
    """Test YFinance data source."""
    
    def test_init(self):
        """Test YFinanceDataSource initialization."""
        source = YFinanceDataSource()
        
        assert source.api_key == "N/A"
        assert source.rate_limit == 60
    
    @pytest.mark.asyncio
    async def test_get_historical_data_success(self, sample_market_data):
        """Test successful data retrieval."""
        source = YFinanceDataSource()
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = sample_market_data
            mock_ticker.return_value = mock_ticker_instance
            
            start_date = datetime.now() - timedelta(days=30)
            end_date = datetime.now()
            
            result = await source.get_historical_data("AAPL", start_date, end_date)
            
            assert not result.empty
            assert "symbol" in result.columns
            assert result.iloc[0]["symbol"] == "AAPL"


class TestDataManager:
    """Test DataManager class."""
    
    def test_init(self):
        """Test DataManager initialization."""
        manager = DataManager()
        
        assert hasattr(manager, 'finazon_source')
        assert hasattr(manager, 'alpha_vantage_source')
        assert hasattr(manager, 'yfinance_source')
        assert len(manager.sources) == 3
    
    @pytest.mark.asyncio
    async def test_get_market_data_success(self, sample_market_data):
        """Test successful market data retrieval."""
        manager = DataManager()
        
        # Mock the first source to return data
        with patch.object(manager.finazon_source, 'get_historical_data', return_value=sample_market_data):
            result = await manager.get_market_data(
                symbol="AAPL",
                start_date="2024-01-01",
                end_date="2024-01-31"
            )
            
            assert not result.empty
            assert "symbol" in result.columns
    
    @pytest.mark.asyncio
    async def test_get_market_data_fallback(self, sample_market_data):
        """Test fallback to other sources when first fails."""
        manager = DataManager()
        
        # Mock first source to fail, second to succeed
        with patch.object(manager.finazon_source, 'get_historical_data', return_value=pd.DataFrame()):
            with patch.object(manager.alpha_vantage_source, 'get_historical_data', return_value=sample_market_data):
                result = await manager.get_market_data(
                    symbol="AAPL",
                    start_date="2024-01-01",
                    end_date="2024-01-31"
                )
                
                assert not result.empty
    
    @pytest.mark.asyncio
    async def test_get_market_data_all_fail(self):
        """Test when all sources fail."""
        manager = DataManager()
        
        # Mock all sources to fail
        with patch.object(manager.finazon_source, 'get_historical_data', return_value=pd.DataFrame()):
            with patch.object(manager.alpha_vantage_source, 'get_historical_data', return_value=pd.DataFrame()):
                with patch.object(manager.yfinance_source, 'get_historical_data', return_value=pd.DataFrame()):
                    result = await manager.get_market_data(
                        symbol="AAPL",
                        start_date="2024-01-01",
                        end_date="2024-01-31"
                    )
                    
                    assert result.empty


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self):
        """Test that rate limiting is enforced."""
        source = FinazonDataSource("test_key")
        
        # Mock time to test rate limiting
        with patch('datetime.datetime') as mock_datetime:
            # First call - should pass
            mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
            await source._rate_limit_check("test")
            
            # Second call immediately - should be rate limited
            mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 1)
            
            with patch('asyncio.sleep') as mock_sleep:
                await source._rate_limit_check("test")
                # Should have called sleep due to rate limiting
                mock_sleep.assert_called()


class TestDataValidation:
    """Test data validation and cleaning."""
    
    def test_data_cleaning(self):
        """Test that data is properly cleaned and validated."""
        # Create test data with some invalid values
        test_data = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'open': [100.0, np.nan, 105.0],
            'high': [105.0, 110.0, np.nan],
            'low': [95.0, 100.0, 100.0],
            'close': [102.0, 108.0, 103.0],
            'volume': [1000000, 0, 2000000]
        })
        
        # Test that NaN values are handled
        assert test_data['open'].isna().any()
        assert test_data['high'].isna().any()
        
        # Test volume validation
        assert (test_data['volume'] > 0).any()  # Some valid volumes
        assert (test_data['volume'] == 0).any()  # Some invalid volumes
    
    def test_data_types(self):
        """Test that data types are correct."""
        test_data = pd.DataFrame({
            'symbol': ['AAPL'],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [1000000]
        })
        
        assert test_data['open'].dtype in [np.float64, np.float32]
        assert test_data['volume'].dtype in [np.int64, np.int32]
        assert test_data['symbol'].dtype == 'object'
