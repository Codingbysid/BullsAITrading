from src.utils.common_imports import *
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

#!/usr/bin/env python3
"""
Base model interfaces for the four-model decision engine architecture.

This module provides the foundation for all four models in the QuantAI system:
1. Sentiment Analysis Model
2. Quantitative Risk Model  
3. ML Ensemble Model
4. RL Decider Agent

Each model implements a standardized interface for consistent integration.
"""



@dataclass
class ModelOutput:
    """Standardized output format for all models in the four-model architecture."""
    
    signal: float  # -1 to 1 (sell to buy strength)
    confidence: float  # 0 to 1
    reasoning: str  # Human-readable explanation
    metrics: Dict[str, float]  # Model-specific metrics
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'signal': self.signal,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'metrics': self.metrics,
            'timestamp': self.timestamp.isoformat()
        }


class BaseModel(ABC):
    """Base class for all four models in the decision engine."""
    
    def __init__(self, model_name: str, weight: float = 1.0):
        self.model_name = model_name
        self.weight = weight  # Input weight to RL agent
        self.is_trained = False
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'last_updated': None
        }
    
    @abstractmethod
    def predict(self, symbol: str, market_data: pd.DataFrame, 
                features: pd.DataFrame, **kwargs) -> ModelOutput:
        """Generate model prediction with standardized output."""
        pass
    
    @abstractmethod
    def get_confidence(self) -> float:
        """Get model confidence score (0-1)."""
        pass
    
    @abstractmethod
    def update(self, feedback: Dict[str, Any]) -> None:
        """Update model based on feedback and outcomes."""
        pass
    
    def validate_inputs(self, market_data: pd.DataFrame, features: pd.DataFrame) -> bool:
        """Validate input data quality."""
        if market_data.empty or features.empty:
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in market_data.columns for col in required_columns):
            return False
        
        return True
    
    def calculate_accuracy(self, predictions: list, actuals: list) -> float:
        """Calculate model accuracy from predictions vs actuals."""
        if len(predictions) != len(actuals) or len(predictions) == 0:
            return 0.0
        
        correct = sum(1 for p, a in zip(predictions, actuals) 
                     if (p > 0 and a > 0) or (p < 0 and a < 0) or (p == 0 and a == 0))
        
        accuracy = correct / len(predictions)
        self.performance_metrics['accuracy'] = accuracy
        self.performance_metrics['last_updated'] = datetime.now()
        
        return accuracy
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'model_name': self.model_name,
            'weight': self.weight,
            'is_trained': self.is_trained,
            'performance_metrics': self.performance_metrics.copy()
        }


class ModelEnsemble:
    """Utility class for combining multiple model outputs."""
    
    def __init__(self, models: list, weights: Optional[Dict[str, float]] = None):
        self.models = models
        self.weights = weights or {model.model_name: model.weight for model in models}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {name: weight/total_weight for name, weight in self.weights.items()}
    
    def combine_predictions(self, predictions: Dict[str, ModelOutput]) -> ModelOutput:
        """Combine multiple model predictions using weighted average."""
        if not predictions:
            return ModelOutput(
                signal=0.0,
                confidence=0.0,
                reasoning="No predictions available",
                metrics={}
            )
        
        # Weighted signal calculation
        weighted_signal = sum(
            pred.signal * self.weights.get(name, 0.0) 
            for name, pred in predictions.items()
        )
        
        # Weighted confidence calculation
        weighted_confidence = sum(
            pred.confidence * self.weights.get(name, 0.0) 
            for name, pred in predictions.items()
        )
        
        # Combine reasoning
        reasoning_parts = []
        for name, pred in predictions.items():
            weight = self.weights.get(name, 0.0)
            reasoning_parts.append(f"{name} ({weight:.1%}): {pred.reasoning}")
        
        combined_reasoning = " | ".join(reasoning_parts)
        
        # Combine metrics
        combined_metrics = {}
        for name, pred in predictions.items():
            for metric_name, value in pred.metrics.items():
                combined_metrics[f"{name}_{metric_name}"] = value
        
        return ModelOutput(
            signal=weighted_signal,
            confidence=weighted_confidence,
            reasoning=combined_reasoning,
            metrics=combined_metrics
        )


class RiskAdjuster:
    """Utility class for applying risk adjustments to model outputs."""
    
    @staticmethod
    def apply_risk_adjustment(output: ModelOutput, risk_metrics: Dict[str, float]) -> ModelOutput:
        """Apply risk-based adjustment to model output."""
        # Extract key risk metrics
        sharpe_ratio = risk_metrics.get('sharpe_ratio', 0.0)
        max_drawdown = risk_metrics.get('max_drawdown', 0.0)
        volatility = risk_metrics.get('volatility', 0.2)
        
        # Calculate risk adjustment factor
        risk_factor = 1.0
        
        # Adjust for Sharpe ratio
        if sharpe_ratio > 1.5:
            risk_factor *= 1.2  # Boost confidence for good risk-adjusted returns
        elif sharpe_ratio < 0.5:
            risk_factor *= 0.8  # Reduce confidence for poor risk-adjusted returns
        
        # Adjust for drawdown
        if max_drawdown < -0.15:  # High drawdown
            risk_factor *= 0.7  # Significant reduction
        elif max_drawdown > -0.05:  # Low drawdown
            risk_factor *= 1.1  # Slight boost
        
        # Adjust for volatility
        if volatility > 0.4:  # High volatility
            risk_factor *= 0.9  # Reduce confidence
        elif volatility < 0.15:  # Low volatility
            risk_factor *= 1.05  # Slight boost
        
        # Apply adjustment
        adjusted_confidence = min(1.0, output.confidence * risk_factor)
        
        # Create adjusted output
        adjusted_output = ModelOutput(
            signal=output.signal,
            confidence=adjusted_confidence,
            reasoning=f"{output.reasoning} [Risk-adjusted: {risk_factor:.2f}x]",
            metrics=output.metrics.copy()
        )
        
        # Add risk adjustment metrics
        adjusted_output.metrics.update({
            'risk_adjustment_factor': risk_factor,
            'original_confidence': output.confidence,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility
        })
        
        return adjusted_output


class ModelValidator:
    """Utility class for validating model outputs and consistency."""
    
    @staticmethod
    def validate_output(output: ModelOutput) -> bool:
        """Validate model output format and ranges."""
        # Check signal range
        if not (-1.0 <= output.signal <= 1.0):
            return False
        
        # Check confidence range
        if not (0.0 <= output.confidence <= 1.0):
            return False
        
        # Check required fields
        if not output.reasoning or not isinstance(output.metrics, dict):
            return False
        
        return True
    
    @staticmethod
    def check_model_consistency(predictions: Dict[str, ModelOutput]) -> Dict[str, Any]:
        """Check consistency across multiple model predictions."""
        if len(predictions) < 2:
            return {'consistent': True, 'agreement': 1.0}
        
        signals = [pred.signal for pred in predictions.values()]
        confidences = [pred.confidence for pred in predictions.values()]
        
        # Calculate agreement
        signal_std = np.std(signals)
        confidence_std = np.std(confidences)
        
        # Agreement score (0-1, higher is better)
        agreement = 1.0 - min(1.0, (signal_std + confidence_std) / 2)
        
        # Check for extreme disagreements
        consistent = agreement > 0.3  # Threshold for consistency
        
        return {
            'consistent': consistent,
            'agreement': agreement,
            'signal_std': signal_std,
            'confidence_std': confidence_std,
            'signal_range': max(signals) - min(signals),
            'confidence_range': max(confidences) - min(confidences)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test ModelOutput
    test_output = ModelOutput(
        signal=0.7,
        confidence=0.85,
        reasoning="Strong bullish signals detected",
        metrics={'rsi': 65.0, 'macd': 0.02}
    )
    
    print("Test ModelOutput:")
    print(test_output.to_dict())
    
    # Test ModelValidator
    validator = ModelValidator()
    is_valid = validator.validate_output(test_output)
    print(f"Output is valid: {is_valid}")
    
    # Test RiskAdjuster
    risk_metrics = {
        'sharpe_ratio': 1.8,
        'max_drawdown': -0.08,
        'volatility': 0.25
    }
    
    adjusted_output = RiskAdjuster.apply_risk_adjustment(test_output, risk_metrics)
    print(f"Risk-adjusted confidence: {adjusted_output.confidence:.3f}")
    print(f"Risk adjustment factor: {adjusted_output.metrics['risk_adjustment_factor']:.3f}")
