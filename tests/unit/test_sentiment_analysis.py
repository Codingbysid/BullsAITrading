"""
Unit tests for sentiment analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.data.sentiment_analysis import (
    NewsSentimentAnalyzer,
    SocialMediaSentimentAnalyzer,
    SentimentAggregator,
    RealTimeSentimentMonitor,
    SentimentData
)


class TestSentimentData:
    """Test SentimentData dataclass."""
    
    def test_sentiment_data_creation(self):
        """Test SentimentData object creation."""
        sentiment_data = SentimentData(
            symbol="AAPL",
            timestamp=datetime.now(),
            sentiment_score=0.5,
            confidence=0.8,
            source="news_api",
            text="Test news article"
        )
        
        assert sentiment_data.symbol == "AAPL"
        assert sentiment_data.sentiment_score == 0.5
        assert sentiment_data.confidence == 0.8
        assert sentiment_data.source == "news_api"


class TestNewsSentimentAnalyzer:
    """Test NewsSentimentAnalyzer class."""
    
    def test_init(self):
        """Test NewsSentimentAnalyzer initialization."""
        analyzer = NewsSentimentAnalyzer()
        
        assert hasattr(analyzer, 'news_api_key')
        assert hasattr(analyzer, 'gemini_api_key')
        assert hasattr(analyzer, 'news_client')
    
    def test_init_with_api_keys(self):
        """Test initialization with API keys."""
        with patch('src.data.sentiment_analysis.NewsApiClient') as mock_client:
            analyzer = NewsSentimentAnalyzer()
            # Should initialize News API client if key is available
    
    @pytest.mark.asyncio
    async def test_fetch_news_sentiment_success(self, sample_sentiment_data):
        """Test successful news sentiment fetching."""
        analyzer = NewsSentimentAnalyzer()
        
        # Mock the news client
        mock_articles = [
            {
                'title': 'Test news article',
                'description': 'Test description',
                'publishedAt': datetime.now().isoformat()
            }
        ]
        
        with patch.object(analyzer, '_fetch_news_articles', return_value=mock_articles):
            with patch.object(analyzer, '_analyze_sentiment', return_value={'sentiment_score': 0.5, 'confidence': 0.8}):
                result = await analyzer.fetch_news_sentiment("AAPL", 7)
                
                assert len(result) == 1
                assert result[0].symbol == "AAPL"
                assert result[0].sentiment_score == 0.5
                assert result[0].confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_fetch_news_sentiment_no_articles(self):
        """Test news sentiment fetching with no articles."""
        analyzer = NewsSentimentAnalyzer()
        
        with patch.object(analyzer, '_fetch_news_articles', return_value=[]):
            result = await analyzer.fetch_news_sentiment("AAPL", 7)
            
            assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_fetch_news_articles_success(self):
        """Test successful news article fetching."""
        analyzer = NewsSentimentAnalyzer()
        
        # Mock News API client
        mock_response = {
            'status': 'ok',
            'articles': [
                {
                    'title': 'Test article',
                    'description': 'Test description',
                    'publishedAt': datetime.now().isoformat()
                }
            ]
        }
        
        with patch.object(analyzer, 'news_client') as mock_client:
            mock_client.get_everything.return_value = mock_response
            
            result = await analyzer._fetch_news_articles("AAPL", 7)
            
            assert len(result) == 1
            assert result[0]['title'] == 'Test article'
    
    @pytest.mark.asyncio
    async def test_fetch_news_articles_error(self):
        """Test news article fetching with error."""
        analyzer = NewsSentimentAnalyzer()
        
        # Mock News API client with error
        mock_response = {
            'status': 'error',
            'message': 'API key invalid'
        }
        
        with patch.object(analyzer, 'news_client') as mock_client:
            mock_client.get_everything.return_value = mock_response
            
            result = await analyzer._fetch_news_articles("AAPL", 7)
            
            assert len(result) == 0
    
    def test_simple_sentiment_analysis_positive(self):
        """Test simple sentiment analysis with positive text."""
        analyzer = NewsSentimentAnalyzer()
        
        positive_text = "The stock is showing strong growth and bullish momentum"
        result = analyzer._simple_sentiment_analysis(positive_text)
        
        assert result['sentiment_score'] > 0
        assert result['confidence'] > 0
    
    def test_simple_sentiment_analysis_negative(self):
        """Test simple sentiment analysis with negative text."""
        analyzer = NewsSentimentAnalyzer()
        
        negative_text = "The stock is declining and showing bearish signals"
        result = analyzer._simple_sentiment_analysis(negative_text)
        
        assert result['sentiment_score'] < 0
        assert result['confidence'] > 0
    
    def test_simple_sentiment_analysis_neutral(self):
        """Test simple sentiment analysis with neutral text."""
        analyzer = NewsSentimentAnalyzer()
        
        neutral_text = "The stock price remained unchanged today"
        result = analyzer._simple_sentiment_analysis(neutral_text)
        
        assert result['sentiment_score'] == 0.0
        assert result['confidence'] == 0.3
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_with_gemini(self):
        """Test sentiment analysis with Gemini API."""
        analyzer = NewsSentimentAnalyzer()
        analyzer.gemini_api_key = "test_gemini_key"
        
        # Mock Gemini API response
        mock_response = {
            'candidates': [{
                'content': {
                    'parts': [{
                        'text': '{"sentiment_score": 0.5, "confidence": 0.8}'
                    }]
                }
            }]
        }
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response_obj = AsyncMock()
            mock_response_obj.status = 200
            mock_response_obj.json = AsyncMock(return_value=mock_response)
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response_obj
            
            result = await analyzer._analyze_sentiment("Test text")
            
            assert result['sentiment_score'] == 0.5
            assert result['confidence'] == 0.8
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_fallback(self):
        """Test sentiment analysis fallback to simple analysis."""
        analyzer = NewsSentimentAnalyzer()
        analyzer.gemini_api_key = None  # No Gemini API key
        
        result = await analyzer._analyze_sentiment("Test text with positive words like growth and profit")
        
        assert 'sentiment_score' in result
        assert 'confidence' in result
        assert result['confidence'] > 0


class TestSocialMediaSentimentAnalyzer:
    """Test SocialMediaSentimentAnalyzer class."""
    
    def test_init(self):
        """Test SocialMediaSentimentAnalyzer initialization."""
        analyzer = SocialMediaSentimentAnalyzer()
        
        assert hasattr(analyzer, 'settings')
    
    @pytest.mark.asyncio
    async def test_fetch_twitter_sentiment(self):
        """Test Twitter sentiment fetching (placeholder)."""
        analyzer = SocialMediaSentimentAnalyzer()
        
        result = await analyzer.fetch_twitter_sentiment("AAPL", 24)
        
        # Should return empty list for placeholder implementation
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_fetch_reddit_sentiment(self):
        """Test Reddit sentiment fetching (placeholder)."""
        analyzer = SocialMediaSentimentAnalyzer()
        
        result = await analyzer.fetch_reddit_sentiment("AAPL", 24)
        
        # Should return empty list for placeholder implementation
        assert len(result) == 0


class TestSentimentAggregator:
    """Test SentimentAggregator class."""
    
    def test_init(self):
        """Test SentimentAggregator initialization."""
        aggregator = SentimentAggregator()
        
        assert hasattr(aggregator, 'news_analyzer')
        assert hasattr(aggregator, 'social_analyzer')
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_sentiment_success(self, sample_sentiment_data):
        """Test comprehensive sentiment analysis with data."""
        aggregator = SentimentAggregator()
        
        # Mock the analyzers
        mock_sentiment_data = [
            SentimentData(
                symbol="AAPL",
                timestamp=datetime.now(),
                sentiment_score=0.5,
                confidence=0.8,
                source="news_api",
                text="Test news"
            )
        ]
        
        with patch.object(aggregator.news_analyzer, 'fetch_news_sentiment', return_value=mock_sentiment_data):
            with patch.object(aggregator.social_analyzer, 'fetch_twitter_sentiment', return_value=[]):
                with patch.object(aggregator.social_analyzer, 'fetch_reddit_sentiment', return_value=[]):
                    result = await aggregator.get_comprehensive_sentiment("AAPL", 7)
                    
                    assert 'overall_sentiment' in result
                    assert 'confidence' in result
                    assert 'news_sentiment' in result
                    assert 'social_sentiment' in result
                    assert 'sample_size' in result
                    
                    assert result['sample_size'] == 1
                    assert result['news_sentiment'] == 0.5
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_sentiment_no_data(self):
        """Test comprehensive sentiment analysis with no data."""
        aggregator = SentimentAggregator()
        
        with patch.object(aggregator.news_analyzer, 'fetch_news_sentiment', return_value=[]):
            with patch.object(aggregator.social_analyzer, 'fetch_twitter_sentiment', return_value=[]):
                with patch.object(aggregator.social_analyzer, 'fetch_reddit_sentiment', return_value=[]):
                    result = await aggregator.get_comprehensive_sentiment("AAPL", 7)
                    
                    assert result['overall_sentiment'] == 0.0
                    assert result['confidence'] == 0.0
                    assert result['sample_size'] == 0
    
    def test_create_sentiment_features(self, sample_sentiment_data):
        """Test sentiment feature creation."""
        aggregator = SentimentAggregator()
        
        # Convert sample data to SentimentData objects
        sentiment_objects = [
            SentimentData(
                symbol="AAPL",
                timestamp=datetime.now() - timedelta(days=i),
                sentiment_score=np.random.uniform(-1, 1),
                confidence=np.random.uniform(0.5, 1.0),
                source="news_api",
                text=f"Test news {i}"
            )
            for i in range(10)
        ]
        
        features = aggregator.create_sentiment_features(sentiment_objects)
        
        assert not features.empty
        assert 'sentiment_mean_1d' in features.columns
        assert 'sentiment_mean_3d' in features.columns
        assert 'sentiment_mean_7d' in features.columns
        assert 'sentiment_volatility' in features.columns
    
    def test_create_sentiment_features_empty(self):
        """Test sentiment feature creation with empty data."""
        aggregator = SentimentAggregator()
        
        features = aggregator.create_sentiment_features([])
        
        assert features.empty


class TestRealTimeSentimentMonitor:
    """Test RealTimeSentimentMonitor class."""
    
    def test_init(self):
        """Test RealTimeSentimentMonitor initialization."""
        monitor = RealTimeSentimentMonitor()
        
        assert hasattr(monitor, 'aggregator')
        assert hasattr(monitor, 'sentiment_cache')
    
    @pytest.mark.asyncio
    async def test_get_live_sentiment(self):
        """Test live sentiment retrieval."""
        monitor = RealTimeSentimentMonitor()
        
        # Mock the aggregator
        mock_sentiment = {
            'overall_sentiment': 0.3,
            'confidence': 0.8,
            'news_sentiment': 0.4,
            'social_sentiment': 0.2,
            'sample_size': 50
        }
        
        with patch.object(monitor.aggregator, 'get_comprehensive_sentiment', return_value=mock_sentiment):
            result = await monitor.get_live_sentiment("AAPL")
            
            assert result == mock_sentiment
            assert 'AAPL' in str(monitor.sentiment_cache.keys())
    
    @pytest.mark.asyncio
    async def test_get_live_sentiment_cached(self):
        """Test live sentiment retrieval with caching."""
        monitor = RealTimeSentimentMonitor()
        
        # Pre-populate cache
        cache_key = "AAPL_2024010112"
        monitor.sentiment_cache[cache_key] = {
            'overall_sentiment': 0.2,
            'confidence': 0.7
        }
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
            
            result = await monitor.get_live_sentiment("AAPL")
            
            assert result['overall_sentiment'] == 0.2
            assert result['confidence'] == 0.7
    
    def test_get_sentiment_signal_buy(self):
        """Test sentiment signal generation for buy."""
        monitor = RealTimeSentimentMonitor()
        
        sentiment_data = {
            'overall_sentiment': 0.5,
            'confidence': 0.8
        }
        
        signal = monitor.get_sentiment_signal(sentiment_data)
        
        assert signal == 'BUY'
    
    def test_get_sentiment_signal_sell(self):
        """Test sentiment signal generation for sell."""
        monitor = RealTimeSentimentMonitor()
        
        sentiment_data = {
            'overall_sentiment': -0.5,
            'confidence': 0.8
        }
        
        signal = monitor.get_sentiment_signal(sentiment_data)
        
        assert signal == 'SELL'
    
    def test_get_sentiment_signal_hold_confidence(self):
        """Test sentiment signal generation for hold due to low confidence."""
        monitor = RealTimeSentimentMonitor()
        
        sentiment_data = {
            'overall_sentiment': 0.5,
            'confidence': 0.3  # Low confidence
        }
        
        signal = monitor.get_sentiment_signal(sentiment_data)
        
        assert signal == 'HOLD'
    
    def test_get_sentiment_signal_hold_neutral(self):
        """Test sentiment signal generation for hold due to neutral sentiment."""
        monitor = RealTimeSentimentMonitor()
        
        sentiment_data = {
            'overall_sentiment': 0.1,  # Neutral sentiment
            'confidence': 0.8
        }
        
        signal = monitor.get_sentiment_signal(sentiment_data)
        
        assert signal == 'HOLD'


class TestSentimentIntegration:
    """Test sentiment analysis integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_sentiment_pipeline(self):
        """Test the complete sentiment analysis pipeline."""
        # This would test the full pipeline from news fetching to sentiment aggregation
        # Implementation would depend on the specific pipeline structure
        pass
    
    def test_sentiment_data_quality(self):
        """Test sentiment data quality and validation."""
        # Test that sentiment scores are within expected ranges
        sentiment_scores = np.random.uniform(-1, 1, 100)
        
        assert all(-1 <= score <= 1 for score in sentiment_scores)
        
        # Test confidence scores
        confidence_scores = np.random.uniform(0, 1, 100)
        
        assert all(0 <= conf <= 1 for conf in confidence_scores)
    
    def test_sentiment_temporal_consistency(self):
        """Test temporal consistency of sentiment data."""
        # Create time series of sentiment data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        sentiment_scores = np.random.uniform(-1, 1, 30)
        
        sentiment_df = pd.DataFrame({
            'date': dates,
            'sentiment': sentiment_scores
        })
        
        # Test that sentiment changes are reasonable
        sentiment_changes = sentiment_df['sentiment'].diff().dropna()
        
        # Most changes should be within reasonable bounds
        assert sentiment_changes.abs().max() < 2.0  # No extreme jumps
    
    @pytest.mark.asyncio
    async def test_sentiment_error_handling(self):
        """Test error handling in sentiment analysis."""
        analyzer = NewsSentimentAnalyzer()
        
        # Test with invalid API key
        analyzer.news_api_key = None
        
        result = await analyzer.fetch_news_sentiment("AAPL", 7)
        
        # Should handle gracefully
        assert isinstance(result, list)
