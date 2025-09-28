from src.utils.common_imports import *
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import logging
from .base_models import BaseModel, ModelOutput
from src.risk.risk_management import RiskManager, KellyCriterion

#!/usr/bin/env python3
"""
Quantitative Risk Model for the four-model decision engine.

This model analyzes risk-adjusted quantitative metrics:
- Sharpe ratio, MAR ratio, Alpha, Beta
- Sortino ratio, Calmar ratio, Information ratio
- Volatility, Maximum drawdown
- Risk-adjusted signal generation

Provides 25% input weight to the RL Decider Agent.
"""



logger = setup_logger()


class QuantitativeRiskModel(BaseModel):
    """Model 2: Risk-adjusted quantitative metrics analysis"""
    
    def __init__(self):
        super().__init__("QuantitativeRisk", weight=0.25)
        
        # Initialize risk management components
        self.risk_manager = RiskManager()
        self.kelly_criterion = KellyCriterion()
        
        # Model parameters
        self.lookback_period = 252  # 1 year of trading days
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.min_data_points = 30  # Minimum data points required
        
        # Risk thresholds
        self.risk_thresholds = {
            'sharpe_ratio': {'good': 1.5, 'poor': 0.5},
            'mar_ratio': {'good': 0.8, 'poor': 0.3},
            'max_drawdown': {'good': -0.05, 'poor': -0.15},
            'volatility': {'good': 0.15, 'poor': 0.35}
        }
        
        # Performance tracking
        self.risk_history = []
        self.metric_accuracy = {
            'sharpe_ratio': 0.0,
            'mar_ratio': 0.0,
            'alpha': 0.0,
            'beta': 0.0
        }
    
    def predict(self, symbol: str, market_data: pd.DataFrame, 
                features: pd.DataFrame, **kwargs) -> ModelOutput:
        """Generate quantitative risk-based prediction"""
        try:
            if not self.validate_inputs(market_data, features):
                return self._create_error_output("Invalid input data")
            
            # Calculate risk-adjusted metrics
            risk_metrics = self._calculate_risk_metrics(market_data)
            
            # Generate risk-adjusted signal
            risk_signal = self._generate_risk_signal(risk_metrics)
            
            # Calculate confidence based on metric consistency
            confidence = self._calculate_risk_confidence(risk_metrics)
            
            # Generate reasoning
            reasoning = self._generate_risk_reasoning(risk_metrics, risk_signal)
            
            # Compile metrics
            metrics = self._compile_risk_metrics(risk_metrics, risk_signal, confidence)
            
            # Update performance tracking
            self.performance_metrics['total_predictions'] += 1
            
            return ModelOutput(
                signal=risk_signal,
                confidence=confidence,
                reasoning=reasoning,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Quantitative risk analysis failed for {symbol}: {e}")
            return self._create_error_output(f"Quantitative analysis failed: {e}")
    
    def _calculate_risk_metrics(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        try:
            # Calculate returns
            returns = market_data['Close'].pct_change().dropna()
            
            if len(returns) < self.min_data_points:
                return self._get_fallback_metrics()
            
            # Core risk metrics
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            mar_ratio = self._calculate_mar_ratio(returns)
            alpha, beta = self._calculate_alpha_beta(returns, market_data)
            
            # Advanced metrics
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns)
            information_ratio = self._calculate_information_ratio(returns)
            
            # Volatility and drawdown
            volatility = returns.std() * np.sqrt(252)
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # Additional metrics
            var_95 = self._calculate_var(returns, 0.95)
            var_99 = self._calculate_var(returns, 0.99)
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'mar_ratio': mar_ratio,
                'alpha': alpha,
                'beta': beta,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'var_99': var_99,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'annual_return': returns.mean() * 252,
                'total_return': (market_data['Close'].iloc[-1] / market_data['Close'].iloc[0] - 1) if len(market_data) > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {e}")
            return self._get_fallback_metrics()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        try:
            excess_returns = returns.mean() * 252 - self.risk_free_rate
            volatility = returns.std() * np.sqrt(252)
            return excess_returns / volatility if volatility > 0 else 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_mar_ratio(self, returns: pd.Series) -> float:
        """Calculate MAR (Risk-Adjusted Return) ratio"""
        try:
            annual_return = returns.mean() * 252
            max_drawdown = abs(self._calculate_max_drawdown(returns))
            return annual_return / max_drawdown if max_drawdown > 0 else 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate MAR ratio: {e}")
            return 0.0
    
    def _calculate_alpha_beta(self, returns: pd.Series, market_data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate alpha and beta vs market benchmark"""
        try:
            # For simplicity, using SPY as market proxy
            # In production, this would use actual market index data
            market_returns = returns  # Simplified - should use market index
            
            if len(returns) > 30:
                # Calculate covariance and variance
                covariance = np.cov(returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                
                beta = covariance / market_variance if market_variance > 0 else 1.0
                alpha = (returns.mean() * 252) - beta * (market_returns.mean() * 252)
                
                return alpha, beta
            else:
                return 0.0, 1.0
                
        except Exception as e:
            logger.warning(f"Failed to calculate alpha/beta: {e}")
            return 0.0, 1.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        try:
            excess_returns = returns.mean() * 252 - self.risk_free_rate
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252)
            
            return excess_returns / downside_deviation if downside_deviation > 0 else 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate Sortino ratio: {e}")
            return 0.0
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        try:
            annual_return = returns.mean() * 252
            max_drawdown = abs(self._calculate_max_drawdown(returns))
            return annual_return / max_drawdown if max_drawdown > 0 else 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate Calmar ratio: {e}")
            return 0.0
    
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate Information ratio"""
        try:
            # Simplified calculation - in production would use benchmark
            benchmark_returns = returns  # Should use actual benchmark
            active_returns = returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(252)
            
            return active_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate Information ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            return drawdown.min()
        except Exception as e:
            logger.warning(f"Failed to calculate max drawdown: {e}")
            return 0.0
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        try:
            return np.percentile(returns, (1 - confidence_level) * 100)
        except Exception as e:
            logger.warning(f"Failed to calculate VaR: {e}")
            return 0.0
    
    def _generate_risk_signal(self, risk_metrics: Dict[str, float]) -> float:
        """Generate composite risk-adjusted signal"""
        try:
            # Extract key metrics
            sharpe = risk_metrics.get('sharpe_ratio', 0.0)
            mar = risk_metrics.get('mar_ratio', 0.0)
            alpha = risk_metrics.get('alpha', 0.0)
            beta = risk_metrics.get('beta', 1.0)
            sortino = risk_metrics.get('sortino_ratio', 0.0)
            calmar = risk_metrics.get('calmar_ratio', 0.0)
            
            # Normalize metrics to [-1, 1] range
            sharpe_score = np.tanh(sharpe / 2) if sharpe > 0 else -0.5
            mar_score = np.tanh(mar) if mar > 0 else -0.5
            alpha_score = np.tanh(alpha * 10) if alpha > 0 else -0.3
            sortino_score = np.tanh(sortino / 2) if sortino > 0 else -0.3
            calmar_score = np.tanh(calmar) if calmar > 0 else -0.3
            
            # Beta adjustment (lower beta = more defensive = positive for risk-adjusted returns)
            beta_score = 1.0 - min(beta, 2.0) / 2.0  # Normalize beta to [0, 1]
            
            # Weighted composite score
            composite_score = (
                0.25 * sharpe_score +
                0.20 * mar_score +
                0.20 * alpha_score +
                0.15 * sortino_score +
                0.10 * calmar_score +
                0.10 * beta_score
            )
            
            # Apply risk penalty for poor metrics
            risk_penalty = self._calculate_risk_penalty(risk_metrics)
            adjusted_score = composite_score * (1 - risk_penalty)
            
            return max(-1.0, min(1.0, adjusted_score))
            
        except Exception as e:
            logger.error(f"Failed to generate risk signal: {e}")
            return 0.0
    
    def _calculate_risk_penalty(self, risk_metrics: Dict[str, float]) -> float:
        """Calculate risk penalty based on poor risk metrics"""
        try:
            penalty = 0.0
            
            # Sharpe ratio penalty
            sharpe = risk_metrics.get('sharpe_ratio', 0.0)
            if sharpe < self.risk_thresholds['sharpe_ratio']['poor']:
                penalty += 0.3
            elif sharpe < self.risk_thresholds['sharpe_ratio']['good']:
                penalty += 0.1
            
            # MAR ratio penalty
            mar = risk_metrics.get('mar_ratio', 0.0)
            if mar < self.risk_thresholds['mar_ratio']['poor']:
                penalty += 0.2
            elif mar < self.risk_thresholds['mar_ratio']['good']:
                penalty += 0.05
            
            # Max drawdown penalty
            max_dd = risk_metrics.get('max_drawdown', 0.0)
            if max_dd < self.risk_thresholds['max_drawdown']['poor']:
                penalty += 0.2
            elif max_dd < self.risk_thresholds['max_drawdown']['good']:
                penalty += 0.05
            
            # Volatility penalty
            volatility = risk_metrics.get('volatility', 0.2)
            if volatility > self.risk_thresholds['volatility']['poor']:
                penalty += 0.15
            elif volatility > self.risk_thresholds['volatility']['good']:
                penalty += 0.05
            
            return min(0.5, penalty)  # Cap penalty at 50%
            
        except Exception as e:
            logger.warning(f"Failed to calculate risk penalty: {e}")
            return 0.0
    
    def _calculate_risk_confidence(self, risk_metrics: Dict[str, float]) -> float:
        """Calculate confidence based on metric consistency and quality"""
        try:
            # Base confidence from data quality
            data_quality = 0.8  # Assume good data quality for now
            
            # Metric consistency
            sharpe = risk_metrics.get('sharpe_ratio', 0.0)
            mar = risk_metrics.get('mar_ratio', 0.0)
            sortino = risk_metrics.get('sortino_ratio', 0.0)
            calmar = risk_metrics.get('calmar_ratio', 0.0)
            
            # Check if metrics are consistent (all positive or all negative)
            positive_metrics = sum(1 for m in [sharpe, mar, sortino, calmar] if m > 0)
            total_metrics = len([m for m in [sharpe, mar, sortino, calmar] if m != 0])
            
            if total_metrics > 0:
                consistency = abs(positive_metrics / total_metrics - 0.5) * 2  # 0 to 1
            else:
                consistency = 0.5
            
            # Risk level confidence (lower risk = higher confidence)
            volatility = risk_metrics.get('volatility', 0.2)
            max_dd = abs(risk_metrics.get('max_drawdown', 0.0))
            
            risk_confidence = 1.0 - min(1.0, (volatility + max_dd) / 2)
            
            # Combined confidence
            combined_confidence = (
                0.4 * data_quality +
                0.3 * consistency +
                0.3 * risk_confidence
            )
            
            return max(0.0, min(1.0, combined_confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate risk confidence: {e}")
            return 0.0
    
    def _generate_risk_reasoning(self, risk_metrics: Dict[str, float], signal: float) -> str:
        """Generate human-readable risk reasoning"""
        try:
            reasoning_parts = []
            
            # Overall assessment
            if signal > 0.3:
                overall_assessment = "strong risk-adjusted opportunity"
            elif signal > 0.1:
                overall_assessment = "moderate risk-adjusted opportunity"
            elif signal < -0.3:
                overall_assessment = "poor risk-adjusted profile"
            elif signal < -0.1:
                overall_assessment = "below-average risk-adjusted profile"
            else:
                overall_assessment = "neutral risk-adjusted profile"
            
            reasoning_parts.append(f"Overall assessment: {overall_assessment}")
            
            # Key metrics analysis
            sharpe = risk_metrics.get('sharpe_ratio', 0.0)
            if sharpe > 1.5:
                reasoning_parts.append(f"Excellent Sharpe ratio ({sharpe:.2f}) indicates strong risk-adjusted returns")
            elif sharpe > 1.0:
                reasoning_parts.append(f"Good Sharpe ratio ({sharpe:.2f}) shows solid risk-adjusted performance")
            elif sharpe < 0.5:
                reasoning_parts.append(f"Poor Sharpe ratio ({sharpe:.2f}) suggests weak risk-adjusted returns")
            
            mar = risk_metrics.get('mar_ratio', 0.0)
            if mar > 0.8:
                reasoning_parts.append(f"Strong MAR ratio ({mar:.2f}) indicates good return-to-drawdown ratio")
            elif mar < 0.3:
                reasoning_parts.append(f"Low MAR ratio ({mar:.2f}) suggests poor return-to-drawdown ratio")
            
            alpha = risk_metrics.get('alpha', 0.0)
            if alpha > 0.05:
                reasoning_parts.append(f"Positive alpha ({alpha:.3f}) shows outperformance vs benchmark")
            elif alpha < -0.05:
                reasoning_parts.append(f"Negative alpha ({alpha:.3f}) indicates underperformance vs benchmark")
            
            # Risk factors
            volatility = risk_metrics.get('volatility', 0.2)
            if volatility > 0.3:
                reasoning_parts.append(f"High volatility ({volatility:.1%}) increases risk")
            elif volatility < 0.15:
                reasoning_parts.append(f"Low volatility ({volatility:.1%}) provides stability")
            
            max_dd = risk_metrics.get('max_drawdown', 0.0)
            if max_dd < -0.15:
                reasoning_parts.append(f"Significant maximum drawdown ({max_dd:.1%}) indicates high downside risk")
            elif max_dd > -0.05:
                reasoning_parts.append(f"Low maximum drawdown ({max_dd:.1%}) shows good downside protection")
            
            return ". ".join(reasoning_parts) + "."
            
        except Exception as e:
            logger.error(f"Failed to generate risk reasoning: {e}")
            return "Risk analysis completed with limited insights."
    
    def _compile_risk_metrics(self, risk_metrics: Dict[str, float], signal: float, confidence: float) -> Dict[str, float]:
        """Compile comprehensive risk metrics"""
        try:
            metrics = risk_metrics.copy()
            
            # Add model outputs
            metrics.update({
                'final_signal': signal,
                'confidence_score': confidence,
                'model_accuracy': self.performance_metrics['accuracy'],
                'total_predictions': float(self.performance_metrics['total_predictions'])
            })
            
            # Add risk assessment
            metrics.update({
                'risk_level': self._assess_risk_level(risk_metrics),
                'risk_score': self._calculate_overall_risk_score(risk_metrics)
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to compile risk metrics: {e}")
            return {'error': str(e)}
    
    def _assess_risk_level(self, risk_metrics: Dict[str, float]) -> float:
        """Assess overall risk level (0-1, where 1 is highest risk)"""
        try:
            volatility = risk_metrics.get('volatility', 0.2)
            max_dd = abs(risk_metrics.get('max_drawdown', 0.0))
            var_95 = abs(risk_metrics.get('var_95', 0.0))
            
            # Normalize risk factors
            vol_risk = min(1.0, volatility / 0.4)  # 40% volatility = max risk
            dd_risk = min(1.0, max_dd / 0.3)  # 30% drawdown = max risk
            var_risk = min(1.0, var_95 / 0.1)  # 10% VaR = max risk
            
            # Weighted risk score
            risk_score = (
                0.4 * vol_risk +
                0.4 * dd_risk +
                0.2 * var_risk
            )
            
            return risk_score
            
        except Exception as e:
            logger.warning(f"Failed to assess risk level: {e}")
            return 0.5
    
    def _calculate_overall_risk_score(self, risk_metrics: Dict[str, float]) -> float:
        """Calculate overall risk score (0-1, where 1 is best risk profile)"""
        try:
            sharpe = risk_metrics.get('sharpe_ratio', 0.0)
            mar = risk_metrics.get('mar_ratio', 0.0)
            alpha = risk_metrics.get('alpha', 0.0)
            
            # Normalize to [0, 1] range
            sharpe_score = max(0, min(1, (sharpe + 1) / 3))  # -1 to 2 range -> 0 to 1
            mar_score = max(0, min(1, mar / 2))  # 0 to 2 range -> 0 to 1
            alpha_score = max(0, min(1, (alpha + 0.2) / 0.4))  # -0.2 to 0.2 range -> 0 to 1
            
            # Weighted score
            overall_score = (
                0.4 * sharpe_score +
                0.3 * mar_score +
                0.3 * alpha_score
            )
            
            return overall_score
            
        except Exception as e:
            logger.warning(f"Failed to calculate overall risk score: {e}")
            return 0.5
    
    def _get_fallback_metrics(self) -> Dict[str, float]:
        """Get fallback metrics when calculation fails"""
        return {
            'sharpe_ratio': 0.0,
            'mar_ratio': 0.0,
            'alpha': 0.0,
            'beta': 1.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'information_ratio': 0.0,
            'volatility': 0.2,
            'max_drawdown': 0.0,
            'var_95': 0.0,
            'var_99': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'annual_return': 0.0,
            'total_return': 0.0
        }
    
    def _create_error_output(self, error_message: str) -> ModelOutput:
        """Create error output with standardized format"""
        return ModelOutput(
            signal=0.0,
            confidence=0.0,
            reasoning=f"Quantitative risk analysis error: {error_message}",
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
                'is_correct': is_correct,
                'risk_metrics': feedback.get('risk_metrics', {})
            }
            
            self.risk_history.append(feedback_record)
            
            # Keep only recent history
            if len(self.risk_history) > 1000:
                self.risk_history = self.risk_history[-1000:]
            
            logger.info(f"Updated quantitative risk model performance: {self.performance_metrics['accuracy']:.2%}")
            
        except Exception as e:
            logger.error(f"Failed to update quantitative risk model: {e}")


# Example usage and testing
if __name__ == "__main__":
    def test_quantitative_model():
        """Test the quantitative risk model"""
        model = QuantitativeRiskModel()
        
        # Create mock data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, 252))
        
        market_data = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.005,
            'Low': prices * 0.995,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 252)
        }, index=dates)
        
        features = pd.DataFrame({
            'rsi': np.random.uniform(20, 80, 252),
            'macd': np.random.normal(0, 0.1, 252)
        }, index=dates)
        
        # Test prediction
        result = model.predict("AAPL", market_data, features)
        
        print("Quantitative Risk Model Test:")
        print(f"Signal: {result.signal:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Key Metrics:")
        for key, value in result.metrics.items():
            if key in ['sharpe_ratio', 'mar_ratio', 'alpha', 'beta', 'volatility', 'max_drawdown']:
                print(f"  {key}: {value:.3f}")
        
        # Test update
        feedback = {
            'actual_return': 0.05,
            'predicted_signal': 0.3,
            'symbol': 'AAPL',
            'risk_metrics': result.metrics
        }
        
        model.update(feedback)
        print(f"Updated accuracy: {model.get_confidence():.2%}")
    
    # Run test
    test_quantitative_model()
