from src.utils.common_imports import *
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from .base_models import BaseModel, ModelOutput
from src.data.sentiment_analysis import SentimentAggregator, RealTimeSentimentMonitor
    import asyncio

#!/usr/bin/env python3
"""
Sentiment Analysis Model for the four-model decision engine.

This model analyzes market sentiment from multiple sources:
- News API integration
- Gemini AI processing
- Social media monitoring
- Earnings call analysis

Provides 25% input weight to the RL Decider Agent.
"""



logger = setup_logger()


class SentimentAnalysisModel(BaseModel):
    """Model 1: Sentiment Analysis with News API + Gemini AI"""
    
    def __init__(self):
        super().__init__("SentimentAnalysis", weight=0.25)
        
        # Initialize sentiment components
        self.sentiment_aggregator = SentimentAggregator()
        self.sentiment_monitor = RealTimeSentimentMonitor()
        
        # Model parameters
        self.confidence_threshold = 0.6
        self.sentiment_weights = {
            'news': 0.4,
            'social': 0.3,
            'earnings': 0.3
        }
        
        # Performance tracking
        self.sentiment_history = []
        self.accuracy_by_source = {
            'news': 0.0,
            'social': 0.0,
            'earnings': 0.0
        }
    
    async def predict(self, symbol: str, market_data: pd.DataFrame, 
                     features: pd.DataFrame, **kwargs) -> ModelOutput:
        """Generate sentiment-based prediction"""
        try:
            if not self.validate_inputs(market_data, features):
                return self._create_error_output("Invalid input data")
            
            # Get real-time sentiment data
            sentiment_data = await self._get_comprehensive_sentiment(symbol)
            
            # Calculate weighted sentiment signal
            sentiment_signal = self._calculate_sentiment_signal(sentiment_data)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(sentiment_data)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(sentiment_data, sentiment_signal)
            
            # Compile metrics
            metrics = self._compile_metrics(sentiment_data, sentiment_signal, confidence)
            
            # Update performance tracking
            self.performance_metrics['total_predictions'] += 1
            
            return ModelOutput(
                signal=sentiment_signal,
                confidence=confidence,
                reasoning=reasoning,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return self._create_error_output(f"Sentiment analysis failed: {e}")
    
    async def _get_comprehensive_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive sentiment data from all sources"""
        try:
            # Get real-time sentiment
            realtime_sentiment = await self.sentiment_monitor.get_live_sentiment(symbol)
            
            # Get aggregated sentiment
            aggregated_sentiment = await self.sentiment_aggregator.aggregate_sentiment(
                symbol=symbol,
                sources=['news', 'social', 'earnings']
            )
            
            # Combine sentiment data
            combined_sentiment = {
                'news_sentiment': realtime_sentiment.get('news_sentiment', 0.0),
                'social_sentiment': realtime_sentiment.get('social_sentiment', 0.0),
                'earnings_sentiment': aggregated_sentiment.get('earnings_sentiment', 0.0),
                'news_sample_size': realtime_sentiment.get('news_sample_size', 0),
                'social_sample_size': realtime_sentiment.get('social_sample_size', 0),
                'earnings_sample_size': aggregated_sentiment.get('earnings_sample_size', 0),
                'sentiment_momentum': self._calculate_sentiment_momentum(symbol),
                'sentiment_volatility': self._calculate_sentiment_volatility(symbol),
                'recent_trend': self._get_recent_sentiment_trend(symbol)
            }
            
            return combined_sentiment
            
        except Exception as e:
            logger.warning(f"Failed to get comprehensive sentiment for {symbol}: {e}")
            return self._get_fallback_sentiment()
    
    def _calculate_sentiment_signal(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate weighted sentiment signal (-1 to 1)"""
        try:
            # Extract sentiment scores
            news_score = sentiment_data.get('news_sentiment', 0.0)
            social_score = sentiment_data.get('social_sentiment', 0.0)
            earnings_score = sentiment_data.get('earnings_sentiment', 0.0)
            
            # Apply weights
            weighted_signal = (
                self.sentiment_weights['news'] * news_score +
                self.sentiment_weights['social'] * social_score +
                self.sentiment_weights['earnings'] * earnings_score
            )
            
            # Apply momentum adjustment
            momentum = sentiment_data.get('sentiment_momentum', 0.0)
            momentum_adjusted_signal = weighted_signal + (momentum * 0.2)
            
            # Clamp to [-1, 1] range
            return max(-1.0, min(1.0, momentum_adjusted_signal))
            
        except Exception as e:
            logger.error(f"Failed to calculate sentiment signal: {e}")
            return 0.0
    
    def _calculate_confidence(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate confidence based on data quality and consistency"""
        try:
            # Base confidence from sample sizes
            news_samples = sentiment_data.get('news_sample_size', 0)
            social_samples = sentiment_data.get('social_sample_size', 0)
            earnings_samples = sentiment_data.get('earnings_sample_size', 0)
            
            total_samples = news_samples + social_samples + earnings_samples
            sample_confidence = min(total_samples / 200, 1.0)  # Scale by total samples
            
            # Consistency confidence
            sentiments = [
                sentiment_data.get('news_sentiment', 0.0),
                sentiment_data.get('social_sentiment', 0.0),
                sentiment_data.get('earnings_sentiment', 0.0)
            ]
            
            # Remove zero values for consistency calculation
            non_zero_sentiments = [s for s in sentiments if s != 0.0]
            
            if len(non_zero_sentiments) > 1:
                consistency = 1 - (np.std(non_zero_sentiments) / 2)
                consistency = max(0.0, min(1.0, consistency))
            else:
                consistency = 0.5  # Default for insufficient data
            
            # Volatility adjustment (lower volatility = higher confidence)
            volatility = sentiment_data.get('sentiment_volatility', 0.5)
            volatility_confidence = max(0.0, 1 - volatility)
            
            # Combined confidence
            combined_confidence = (
                0.4 * sample_confidence +
                0.4 * consistency +
                0.2 * volatility_confidence
            )
            
            return max(0.0, min(1.0, combined_confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 0.0
    
    def _generate_reasoning(self, sentiment_data: Dict[str, Any], signal: float) -> str:
        """Generate human-readable reasoning"""
        try:
            news_score = sentiment_data.get('news_sentiment', 0.0)
            social_score = sentiment_data.get('social_sentiment', 0.0)
            earnings_score = sentiment_data.get('earnings_sentiment', 0.0)
            
            # Determine overall sentiment
            if signal > 0.3:
                overall_sentiment = "strongly positive"
            elif signal > 0.1:
                overall_sentiment = "positive"
            elif signal < -0.3:
                overall_sentiment = "strongly negative"
            elif signal < -0.1:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
            
            # Build reasoning components
            reasoning_parts = [f"Overall sentiment is {overall_sentiment}"]
            
            # Add source-specific insights
            if abs(news_score) > 0.3:
                news_direction = "positive" if news_score > 0 else "negative"
                reasoning_parts.append(f"News sentiment is {news_direction}")
            
            if abs(social_score) > 0.3:
                social_direction = "positive" if social_score > 0 else "negative"
                reasoning_parts.append(f"Social media sentiment is {social_direction}")
            
            if abs(earnings_score) > 0.3:
                earnings_direction = "positive" if earnings_score > 0 else "negative"
                reasoning_parts.append(f"Earnings sentiment is {earnings_direction}")
            
            # Add momentum insight
            momentum = sentiment_data.get('sentiment_momentum', 0.0)
            if abs(momentum) > 0.2:
                momentum_direction = "improving" if momentum > 0 else "deteriorating"
                reasoning_parts.append(f"Sentiment momentum is {momentum_direction}")
            
            # Add data quality insight
            total_samples = (
                sentiment_data.get('news_sample_size', 0) +
                sentiment_data.get('social_sample_size', 0) +
                sentiment_data.get('earnings_sample_size', 0)
            )
            
            if total_samples > 100:
                reasoning_parts.append("High data quality with substantial sample size")
            elif total_samples > 50:
                reasoning_parts.append("Moderate data quality")
            else:
                reasoning_parts.append("Limited data quality - low sample size")
            
            return ". ".join(reasoning_parts) + "."
            
        except Exception as e:
            logger.error(f"Failed to generate reasoning: {e}")
            return "Sentiment analysis completed with limited insights."
    
    def _compile_metrics(self, sentiment_data: Dict[str, Any], signal: float, confidence: float) -> Dict[str, float]:
        """Compile comprehensive metrics"""
        try:
            metrics = {
                # Core sentiment scores
                'news_sentiment': sentiment_data.get('news_sentiment', 0.0),
                'social_sentiment': sentiment_data.get('social_sentiment', 0.0),
                'earnings_sentiment': sentiment_data.get('earnings_sentiment', 0.0),
                
                # Sample sizes
                'news_sample_size': float(sentiment_data.get('news_sample_size', 0)),
                'social_sample_size': float(sentiment_data.get('social_sample_size', 0)),
                'earnings_sample_size': float(sentiment_data.get('earnings_sample_size', 0)),
                
                # Derived metrics
                'sentiment_momentum': sentiment_data.get('sentiment_momentum', 0.0),
                'sentiment_volatility': sentiment_data.get('sentiment_volatility', 0.0),
                'total_sample_size': float(
                    sentiment_data.get('news_sample_size', 0) +
                    sentiment_data.get('social_sample_size', 0) +
                    sentiment_data.get('earnings_sample_size', 0)
                ),
                
                # Model outputs
                'final_signal': signal,
                'confidence_score': confidence,
                
                # Performance metrics
                'model_accuracy': self.performance_metrics['accuracy'],
                'total_predictions': float(self.performance_metrics['total_predictions'])
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to compile metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_sentiment_momentum(self, symbol: str) -> float:
        """Calculate sentiment momentum over recent period"""
        try:
            # Get recent sentiment history
            recent_sentiments = self._get_recent_sentiment_history(symbol, days=7)
            
            if len(recent_sentiments) < 3:
                return 0.0
            
            # Calculate momentum as slope of recent sentiment
            x = np.arange(len(recent_sentiments))
            y = np.array(recent_sentiments)
            
            # Simple linear regression for slope
            slope = np.polyfit(x, y, 1)[0]
            
            return max(-1.0, min(1.0, slope * 10))  # Scale and clamp
            
        except Exception as e:
            logger.warning(f"Failed to calculate sentiment momentum for {symbol}: {e}")
            return 0.0
    
    def _calculate_sentiment_volatility(self, symbol: str) -> float:
        """Calculate sentiment volatility over recent period"""
        try:
            # Get recent sentiment history
            recent_sentiments = self._get_recent_sentiment_history(symbol, days=14)
            
            if len(recent_sentiments) < 5:
                return 0.5  # Default moderate volatility
            
            # Calculate standard deviation
            volatility = np.std(recent_sentiments)
            
            return max(0.0, min(1.0, volatility))
            
        except Exception as e:
            logger.warning(f"Failed to calculate sentiment volatility for {symbol}: {e}")
            return 0.5
    
    def _get_recent_sentiment_trend(self, symbol: str) -> str:
        """Get recent sentiment trend direction"""
        try:
            momentum = self._calculate_sentiment_momentum(symbol)
            
            if momentum > 0.1:
                return "improving"
            elif momentum < -0.1:
                return "deteriorating"
            else:
                return "stable"
                
        except Exception as e:
            logger.warning(f"Failed to get sentiment trend for {symbol}: {e}")
            return "unknown"
    
    def _get_recent_sentiment_history(self, symbol: str, days: int = 7) -> List[float]:
        """Get recent sentiment history for momentum/volatility calculations"""
        try:
            # This would typically query a sentiment history database
            # For now, return mock data
            return [0.1, 0.2, -0.1, 0.3, 0.0, -0.2, 0.1]
            
        except Exception as e:
            logger.warning(f"Failed to get sentiment history for {symbol}: {e}")
            return []
    
    def _get_fallback_sentiment(self) -> Dict[str, Any]:
        """Get fallback sentiment data when primary sources fail"""
        return {
            'news_sentiment': 0.0,
            'social_sentiment': 0.0,
            'earnings_sentiment': 0.0,
            'news_sample_size': 0,
            'social_sample_size': 0,
            'earnings_sample_size': 0,
            'sentiment_momentum': 0.0,
            'sentiment_volatility': 0.5,
            'recent_trend': 'unknown'
        }
    
    def _create_error_output(self, error_message: str) -> ModelOutput:
        """Create error output with standardized format"""
        return ModelOutput(
            signal=0.0,
            confidence=0.0,
            reasoning=f"Sentiment analysis error: {error_message}",
            metrics={'error': error_message}
        )
    
    def get_confidence(self) -> float:
        """Get model confidence score"""
        return self.performance_metrics['accuracy']
    
    def update(self, feedback: Dict[str, Any]) -> None:
        """Update model based on feedback and outcomes"""
        try:
            # Extract feedback data
            actual_return = feedback.get('actual_return', 0.0)
            predicted_signal = feedback.get('predicted_signal', 0.0)
            symbol = feedback.get('symbol', 'UNKNOWN')
            
            # Determine if prediction was correct
            predicted_direction = 1 if predicted_signal > 0.1 else (-1 if predicted_signal < -0.1 else 0)
            actual_direction = 1 if actual_return > 0.02 else (-1 if actual_return < -0.02 else 0)
            
            is_correct = predicted_direction == actual_direction
            
            # Update performance metrics
            self.performance_metrics['total_predictions'] += 1
            if is_correct:
                self.performance_metrics['correct_predictions'] += 1
            
            # Calculate new accuracy
            total = self.performance_metrics['total_predictions']
            correct = self.performance_metrics['correct_predictions']
            self.performance_metrics['accuracy'] = correct / total if total > 0 else 0.0
            
            # Store feedback for learning
            feedback_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'predicted_signal': predicted_signal,
                'actual_return': actual_return,
                'is_correct': is_correct
            }
            
            self.sentiment_history.append(feedback_record)
            
            # Keep only recent history
            if len(self.sentiment_history) > 1000:
                self.sentiment_history = self.sentiment_history[-1000:]
            
            logger.info(f"Updated sentiment model performance: {self.performance_metrics['accuracy']:.2%}")
            
        except Exception as e:
            logger.error(f"Failed to update sentiment model: {e}")


# Example usage and testing
if __name__ == "__main__":
    
    async def test_sentiment_model():
        """Test the sentiment analysis model"""
        model = SentimentAnalysisModel()
        
        # Create mock data
        market_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        features = pd.DataFrame({
            'rsi': [50, 55, 60],
            'macd': [0.1, 0.2, 0.3]
        })
        
        # Test prediction
        result = await model.predict("AAPL", market_data, features)
        
        print("Sentiment Analysis Model Test:")
        print(f"Signal: {result.signal:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Metrics: {result.metrics}")
        
        # Test update
        feedback = {
            'actual_return': 0.05,
            'predicted_signal': 0.3,
            'symbol': 'AAPL'
        }
        
        model.update(feedback)
        print(f"Updated accuracy: {model.get_confidence():.2%}")
    
    # Run test
    asyncio.run(test_sentiment_model())
