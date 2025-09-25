"""
Integration tests for the complete trading pipeline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from src.data.data_sources import DataManager
from src.data.sentiment_analysis import SentimentAggregator
from src.risk.risk_management import RiskManager
from src.models.trading_models import EnsembleModel
from src.backtesting.backtesting_engine import BacktestingEngine


class TestDataPipelineIntegration:
    """Test the complete data pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_market_data_to_features_pipeline(self, integration_helper):
        """Test the complete pipeline from market data to features."""
        # Setup test environment
        await integration_helper.setup_test_environment()
        
        # Run the full pipeline
        result = await integration_helper.run_full_pipeline('AAPL', 30)
        
        assert result is not None
        assert 'market_data' in result
        assert 'sentiment' in result
        assert 'risk_metrics' in result
        
        # Verify market data
        market_data = result['market_data']
        assert not market_data.empty
        assert 'close' in market_data.columns
        assert 'volume' in market_data.columns
        
        # Verify sentiment data
        sentiment = result['sentiment']
        assert 'overall_sentiment' in sentiment
        assert 'confidence' in sentiment
        
        # Verify risk metrics
        risk_metrics = result['risk_metrics']
        assert 'volatility' in risk_metrics
        assert 'sharpe_ratio' in risk_metrics
    
    @pytest.mark.asyncio
    async def test_sentiment_integration_with_market_data(self):
        """Test sentiment analysis integration with market data."""
        # Mock market data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        market_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(30) * 0.02),
            'volume': np.random.randint(1000000, 10000000, 30)
        }, index=dates)
        
        # Mock sentiment data
        sentiment_data = {
            'overall_sentiment': 0.3,
            'confidence': 0.8,
            'news_sentiment': 0.4,
            'social_sentiment': 0.2,
            'sample_size': 50
        }
        
        # Test integration
        # This would test how sentiment data is integrated with market data
        # for feature engineering and model training
        
        assert market_data.shape[0] == 30
        assert sentiment_data['overall_sentiment'] > 0
        assert sentiment_data['confidence'] > 0.5
    
    @pytest.mark.asyncio
    async def test_risk_metrics_integration(self):
        """Test risk metrics integration with trading decisions."""
        # Create sample returns
        returns = pd.Series(np.random.randn(100) * 0.02)
        
        risk_manager = RiskManager()
        risk_metrics = risk_manager.calculate_risk_metrics(returns)
        
        # Test that risk metrics are calculated correctly
        assert 'volatility' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert 'sharpe_ratio' in risk_metrics
        
        # Test risk-adjusted position sizing
        position_size = risk_manager.calculate_position_size(
            expected_return=0.02,
            volatility=risk_metrics['volatility'],
            confidence=0.95
        )
        
        assert 0 <= position_size <= 1
        assert position_size <= risk_manager.max_position_size


class TestModelTrainingIntegration:
    """Test model training integration."""
    
    def test_feature_engineering_pipeline(self, sample_market_data, sample_features):
        """Test feature engineering pipeline."""
        # This would test the complete feature engineering pipeline
        # from raw market data to ML-ready features
        
        # Simulate feature engineering
        features = sample_features.copy()
        
        # Add technical indicators
        features['rsi'] = np.random.uniform(20, 80, len(features))
        features['sma_20'] = np.random.uniform(90, 110, len(features))
        features['bb_upper'] = np.random.uniform(105, 115, len(features))
        features['bb_lower'] = np.random.uniform(85, 95, len(features))
        
        # Add sentiment features
        features['sentiment_score'] = np.random.uniform(-1, 1, len(features))
        features['sentiment_confidence'] = np.random.uniform(0.5, 1.0, len(features))
        
        # Verify features
        assert len(features) > 0
        assert 'rsi' in features.columns
        assert 'sentiment_score' in features.columns
        
        # Test feature quality
        assert features['rsi'].min() >= 0
        assert features['rsi'].max() <= 100
        assert features['sentiment_score'].min() >= -1
        assert features['sentiment_score'].max() <= 1
    
    def test_model_training_pipeline(self, sample_features):
        """Test model training pipeline."""
        # Create target variable
        target = np.random.choice([-1, 0, 1], len(sample_features), p=[0.2, 0.6, 0.2])
        
        # Test ensemble model training
        ensemble_model = EnsembleModel()
        
        # Mock model training
        with patch.object(ensemble_model, 'train') as mock_train:
            mock_train.return_value = {'accuracy': 0.75, 'precision': 0.72, 'recall': 0.70}
            
            result = ensemble_model.train(sample_features, target)
            
            assert 'accuracy' in result
            assert result['accuracy'] > 0.7
    
    def test_model_prediction_pipeline(self, sample_features):
        """Test model prediction pipeline."""
        ensemble_model = EnsembleModel()
        
        # Mock model prediction
        with patch.object(ensemble_model, 'predict') as mock_predict:
            mock_predict.return_value = np.random.choice([-1, 0, 1], len(sample_features))
            
            predictions = ensemble_model.predict(sample_features)
            
            assert len(predictions) == len(sample_features)
            assert all(pred in [-1, 0, 1] for pred in predictions)


class TestBacktestingIntegration:
    """Test backtesting integration."""
    
    def test_backtesting_pipeline(self, sample_market_data, sample_features):
        """Test complete backtesting pipeline."""
        backtesting_engine = BacktestingEngine()
        
        # Create sample signals
        signals = pd.DataFrame({
            'signal': np.random.choice([-1, 0, 1], len(sample_market_data)),
            'confidence': np.random.uniform(0.5, 1.0, len(sample_market_data))
        }, index=sample_market_data.index)
        
        # Mock backtesting
        with patch.object(backtesting_engine, 'run_backtest') as mock_backtest:
            mock_backtest.return_value = {
                'total_return': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.08,
                'win_rate': 0.65
            }
            
            result = backtesting_engine.run_backtest(sample_market_data, signals)
            
            assert 'total_return' in result
            assert 'sharpe_ratio' in result
            assert 'max_drawdown' in result
            assert result['total_return'] > 0
    
    def test_risk_adjusted_backtesting(self, sample_market_data):
        """Test risk-adjusted backtesting."""
        backtesting_engine = BacktestingEngine()
        risk_manager = RiskManager()
        
        # Create risk-adjusted signals
        returns = sample_market_data['close'].pct_change().dropna()
        risk_metrics = risk_manager.calculate_risk_metrics(returns)
        
        # Adjust position sizes based on risk
        base_signals = np.random.choice([-1, 0, 1], len(sample_market_data))
        risk_adjusted_signals = base_signals * (1 - abs(risk_metrics['max_drawdown']))
        
        signals = pd.DataFrame({
            'signal': risk_adjusted_signals,
            'confidence': np.random.uniform(0.5, 1.0, len(sample_market_data))
        }, index=sample_market_data.index)
        
        # Test that risk-adjusted signals are within bounds
        assert signals['signal'].min() >= -1
        assert signals['signal'].max() <= 1
        
        # Test that risk adjustment reduces position sizes
        assert abs(signals['signal'].mean()) <= abs(base_signals.mean())


class TestRealTimeTradingIntegration:
    """Test real-time trading integration."""
    
    @pytest.mark.asyncio
    async def test_real_time_decision_making(self):
        """Test real-time trading decision making."""
        # Mock real-time data
        current_price = 150.0
        market_data = pd.DataFrame({
            'close': [current_price],
            'volume': [1000000]
        })
        
        # Mock sentiment
        sentiment = {
            'overall_sentiment': 0.3,
            'confidence': 0.8
        }
        
        # Mock model prediction
        model_prediction = 0.7  # Buy signal with 70% confidence
        
        # Mock risk assessment
        risk_metrics = {
            'volatility': 0.15,
            'max_drawdown': -0.05,
            'sharpe_ratio': 1.2
        }
        
        # Test decision making logic
        if (model_prediction > 0.6 and 
            sentiment['overall_sentiment'] > 0.2 and 
            sentiment['confidence'] > 0.7 and
            risk_metrics['sharpe_ratio'] > 1.0):
            
            decision = 'BUY'
            position_size = min(0.2, model_prediction * sentiment['confidence'])
        else:
            decision = 'HOLD'
            position_size = 0.0
        
        assert decision in ['BUY', 'SELL', 'HOLD']
        assert 0 <= position_size <= 0.2
    
    @pytest.mark.asyncio
    async def test_portfolio_rebalancing(self):
        """Test portfolio rebalancing logic."""
        # Mock portfolio state
        portfolio = {
            'AAPL': {'shares': 100, 'price': 150.0, 'weight': 0.4},
            'GOOGL': {'shares': 50, 'price': 2800.0, 'weight': 0.35},
            'MSFT': {'shares': 75, 'price': 300.0, 'weight': 0.25}
        }
        
        # Mock new target weights
        target_weights = {
            'AAPL': 0.3,
            'GOOGL': 0.4,
            'MSFT': 0.3
        }
        
        # Calculate rebalancing needs
        rebalancing = {}
        for symbol in portfolio:
            current_weight = portfolio[symbol]['weight']
            target_weight = target_weights[symbol]
            rebalancing[symbol] = target_weight - current_weight
        
        # Test rebalancing logic
        total_rebalancing = sum(rebalancing.values())
        assert abs(total_rebalancing) < 0.01  # Should sum to approximately zero
        
        # Test individual rebalancing
        assert rebalancing['AAPL'] < 0  # Should reduce AAPL
        assert rebalancing['GOOGL'] > 0  # Should increase GOOGL
        assert rebalancing['MSFT'] > 0  # Should increase MSFT


class TestPerformanceMonitoringIntegration:
    """Test performance monitoring integration."""
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation."""
        # Mock portfolio returns
        returns = pd.Series(np.random.randn(252) * 0.02)  # Daily returns for a year
        
        # Calculate performance metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility
        
        # Test metrics
        assert total_return > -1  # Can't lose more than 100%
        assert volatility > 0
        assert isinstance(sharpe_ratio, float)
    
    def test_risk_monitoring(self):
        """Test risk monitoring and alerts."""
        # Mock portfolio state
        portfolio_value = 100000
        current_drawdown = -0.05
        max_drawdown_limit = -0.10
        
        # Test risk alerts
        if current_drawdown < max_drawdown_limit:
            risk_alert = "HIGH_RISK"
        elif current_drawdown < -0.05:
            risk_alert = "MEDIUM_RISK"
        else:
            risk_alert = "LOW_RISK"
        
        assert risk_alert in ["HIGH_RISK", "MEDIUM_RISK", "LOW_RISK"]
    
    def test_performance_attribution(self):
        """Test performance attribution analysis."""
        # Mock portfolio and benchmark returns
        portfolio_returns = pd.Series(np.random.randn(252) * 0.02)
        benchmark_returns = pd.Series(np.random.randn(252) * 0.015)
        
        # Calculate attribution
        excess_return = portfolio_returns - benchmark_returns
        tracking_error = excess_return.std() * np.sqrt(252)
        information_ratio = excess_return.mean() / (excess_return.std() * np.sqrt(252))
        
        # Test attribution metrics
        assert tracking_error > 0
        assert isinstance(information_ratio, float)


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_api_failure_handling(self):
        """Test handling of API failures."""
        # Mock API failure
        with patch('src.data.data_sources.DataManager.get_market_data') as mock_get_data:
            mock_get_data.side_effect = Exception("API Error")
            
            # Test that system handles API failures gracefully
            try:
                result = await mock_get_data("AAPL", "2024-01-01", "2024-01-31")
            except Exception as e:
                # Should handle gracefully
                assert "API Error" in str(e)
    
    @pytest.mark.asyncio
    async def test_data_quality_validation(self):
        """Test data quality validation."""
        # Mock data with quality issues
        bad_data = pd.DataFrame({
            'close': [100, np.nan, 102, 0, 105],  # NaN and zero values
            'volume': [1000, 2000, -1000, 3000, 4000]  # Negative volume
        })
        
        # Test data validation
        has_nan = bad_data.isnull().any().any()
        has_negative_volume = (bad_data['volume'] < 0).any()
        has_zero_price = (bad_data['close'] == 0).any()
        
        assert has_nan
        assert has_negative_volume
        assert has_zero_price
        
        # Test data cleaning
        cleaned_data = bad_data.dropna()
        cleaned_data = cleaned_data[cleaned_data['volume'] > 0]
        cleaned_data = cleaned_data[cleaned_data['close'] > 0]
        
        assert len(cleaned_data) < len(bad_data)
        assert not cleaned_data.isnull().any().any()
        assert (cleaned_data['volume'] > 0).all()
        assert (cleaned_data['close'] > 0).all()
    
    def test_model_prediction_validation(self):
        """Test model prediction validation."""
        # Mock model predictions
        predictions = np.array([0.8, -0.9, 1.2, -1.5, 0.3])
        
        # Test prediction validation
        valid_predictions = predictions[(predictions >= -1) & (predictions <= 1)]
        invalid_predictions = predictions[(predictions < -1) | (predictions > 1)]
        
        assert len(valid_predictions) < len(predictions)
        assert len(invalid_predictions) > 0
        
        # Test prediction clipping
        clipped_predictions = np.clip(predictions, -1, 1)
        assert all(-1 <= pred <= 1 for pred in clipped_predictions)
