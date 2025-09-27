#!/usr/bin/env python3
"""
ML Ensemble Model for the four-model decision engine.

This model combines multiple machine learning algorithms:
- Random Forest (40% of ensemble)
- XGBoost (35% of ensemble)
- LSTM (25% of ensemble)

Provides 35% input weight to the RL Decider Agent.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

from .base_models import BaseModel, ModelOutput
from ..models.trading_models import EnsembleTradingModel, RandomForestTradingModel, XGBoostTradingModel, LSTMTradingModel

logger = logging.getLogger(__name__)


class MLEnsembleModel(BaseModel):
    """Model 3: Enhanced ML ensemble with Random Forest, XGBoost, LSTM"""
    
    def __init__(self):
        super().__init__("MLEnsemble", weight=0.35)
        
        # Initialize ensemble components
        self.ensemble_model = EnsembleTradingModel()
        
        # Model-specific weights within ensemble
        self.model_weights = {
            'random_forest': 0.40,   # 40% of ensemble
            'xgboost': 0.35,        # 35% of ensemble
            'lstm': 0.25            # 25% of ensemble
        }
        
        # Individual model instances
        self.individual_models = {
            'random_forest': RandomForestTradingModel(),
            'xgboost': XGBoostTradingModel(),
            'lstm': LSTMTradingModel()
        }
        
        # Model performance tracking
        self.model_performance = {
            'random_forest': {'accuracy': 0.0, 'predictions': 0},
            'xgboost': {'accuracy': 0.0, 'predictions': 0},
            'lstm': {'accuracy': 0.0, 'predictions': 0}
        }
        
        # Feature importance tracking
        self.feature_importance = {}
        
        # Training data storage
        self.training_data = None
        self.training_targets = None
        
    def predict(self, symbol: str, market_data: pd.DataFrame, 
                features: pd.DataFrame, **kwargs) -> ModelOutput:
        """Generate ML ensemble prediction"""
        try:
            if not self.validate_inputs(market_data, features):
                return self._create_error_output("Invalid input data")
            
            if not self.is_trained:
                return self._create_error_output("ML models not trained yet")
            
            # Get ensemble prediction
            ensemble_prediction = self._get_ensemble_prediction(features)
            
            # Get individual model predictions
            individual_predictions = self._get_individual_predictions(features)
            
            # Calculate ensemble signal
            ml_signal = self._calculate_ensemble_signal(ensemble_prediction, individual_predictions)
            
            # Calculate confidence
            confidence = self._calculate_ml_confidence(ensemble_prediction, individual_predictions)
            
            # Generate reasoning
            reasoning = self._generate_ml_reasoning(individual_predictions, ensemble_prediction)
            
            # Compile metrics
            metrics = self._compile_ml_metrics(ensemble_prediction, individual_predictions, confidence)
            
            # Update performance tracking
            self.performance_metrics['total_predictions'] += 1
            
            return ModelOutput(
                signal=ml_signal,
                confidence=confidence,
                reasoning=reasoning,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"ML ensemble prediction failed for {symbol}: {e}")
            return self._create_error_output(f"ML ensemble failed: {e}")
    
    def train_ensemble(self, training_data: pd.DataFrame, target: pd.Series):
        """Train the ML ensemble models"""
        try:
            # Store training data
            self.training_data = training_data.copy()
            self.training_targets = target.copy()
            
            # Train ensemble model
            self.ensemble_model.fit(training_data, target)
            
            # Train individual models
            for model_name, model in self.individual_models.items():
                try:
                    model.fit(training_data, target)
                    logger.info(f"Successfully trained {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to train {model_name}: {e}")
            
            # Calculate feature importance
            self._calculate_feature_importance()
            
            self.is_trained = True
            logger.info("ML ensemble training completed successfully")
            
        except Exception as e:
            logger.error(f"ML ensemble training failed: {e}")
            raise
    
    def _get_ensemble_prediction(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Get prediction from ensemble model"""
        try:
            # Get the latest features
            latest_features = features.iloc[-1:].copy()
            
            # Get ensemble prediction
            ensemble_pred = self.ensemble_model.predict(latest_features)
            ensemble_proba = self.ensemble_model.predict_proba(latest_features)
            
            return {
                'prediction': ensemble_pred[0] if len(ensemble_pred) > 0 else 0.0,
                'probability': ensemble_proba[0] if len(ensemble_proba) > 0 else [0.33, 0.34, 0.33],
                'confidence': max(ensemble_proba[0]) if len(ensemble_proba) > 0 else 0.33
            }
            
        except Exception as e:
            logger.warning(f"Failed to get ensemble prediction: {e}")
            return {
                'prediction': 0.0,
                'probability': [0.33, 0.34, 0.33],
                'confidence': 0.33
            }
    
    def _get_individual_predictions(self, features: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Get predictions from individual models"""
        predictions = {}
        
        try:
            latest_features = features.iloc[-1:].copy()
            
            for model_name, model in self.individual_models.items():
                try:
                    if hasattr(model, 'is_fitted') and model.is_fitted:
                        pred = model.predict(latest_features)
                        proba = model.predict_proba(latest_features) if hasattr(model, 'predict_proba') else None
                        
                        predictions[model_name] = {
                            'prediction': pred[0] if len(pred) > 0 else 0.0,
                            'probability': proba[0] if proba is not None and len(proba) > 0 else [0.33, 0.34, 0.33],
                            'confidence': max(proba[0]) if proba is not None and len(proba) > 0 else 0.33
                        }
                    else:
                        predictions[model_name] = {
                            'prediction': 0.0,
                            'probability': [0.33, 0.34, 0.33],
                            'confidence': 0.33
                        }
                        
                except Exception as e:
                    logger.warning(f"Failed to get prediction from {model_name}: {e}")
                    predictions[model_name] = {
                        'prediction': 0.0,
                        'probability': [0.33, 0.34, 0.33],
                        'confidence': 0.33
                    }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to get individual predictions: {e}")
            return {
                'random_forest': {'prediction': 0.0, 'probability': [0.33, 0.34, 0.33], 'confidence': 0.33},
                'xgboost': {'prediction': 0.0, 'probability': [0.33, 0.34, 0.33], 'confidence': 0.33},
                'lstm': {'prediction': 0.0, 'probability': [0.33, 0.34, 0.33], 'confidence': 0.33}
            }
    
    def _calculate_ensemble_signal(self, ensemble_pred: Dict[str, Any], 
                                 individual_preds: Dict[str, Dict[str, float]]) -> float:
        """Calculate final ensemble signal"""
        try:
            # Get ensemble prediction
            ensemble_signal = ensemble_pred.get('prediction', 0.0)
            
            # Get weighted individual predictions
            weighted_individual = 0.0
            total_weight = 0.0
            
            for model_name, pred_data in individual_preds.items():
                weight = self.model_weights.get(model_name, 0.0)
                prediction = pred_data.get('prediction', 0.0)
                
                weighted_individual += weight * prediction
                total_weight += weight
            
            # Normalize weighted individual
            if total_weight > 0:
                weighted_individual /= total_weight
            
            # Combine ensemble and weighted individual (70% ensemble, 30% weighted individual)
            final_signal = 0.7 * ensemble_signal + 0.3 * weighted_individual
            
            # Convert to [-1, 1] range using tanh
            return np.tanh(final_signal)
            
        except Exception as e:
            logger.error(f"Failed to calculate ensemble signal: {e}")
            return 0.0
    
    def _calculate_ml_confidence(self, ensemble_pred: Dict[str, Any], 
                               individual_preds: Dict[str, Dict[str, float]]) -> float:
        """Calculate ML confidence based on model agreement and individual confidences"""
        try:
            # Get ensemble confidence
            ensemble_confidence = ensemble_pred.get('confidence', 0.33)
            
            # Get individual model confidences
            individual_confidences = []
            for model_name, pred_data in individual_preds.items():
                confidence = pred_data.get('confidence', 0.33)
                individual_confidences.append(confidence)
            
            # Calculate agreement
            agreement = self._calculate_model_agreement(individual_preds)
            
            # Calculate average individual confidence
            avg_individual_confidence = np.mean(individual_confidences) if individual_confidences else 0.33
            
            # Combine confidences (50% ensemble, 30% individual average, 20% agreement)
            combined_confidence = (
                0.5 * ensemble_confidence +
                0.3 * avg_individual_confidence +
                0.2 * agreement
            )
            
            return max(0.0, min(1.0, combined_confidence))
            
        except Exception as e:
            logger.error(f"Failed to calculate ML confidence: {e}")
            return 0.33
    
    def _calculate_model_agreement(self, individual_preds: Dict[str, Dict[str, float]]) -> float:
        """Calculate agreement level between models"""
        try:
            if len(individual_preds) < 2:
                return 0.5
            
            # Get predictions
            predictions = [pred_data.get('prediction', 0.0) for pred_data in individual_preds.values()]
            
            # Calculate standard deviation
            pred_std = np.std(predictions)
            
            # Convert to agreement score (lower std = higher agreement)
            agreement = max(0.0, 1.0 - pred_std)
            
            return agreement
            
        except Exception as e:
            logger.warning(f"Failed to calculate model agreement: {e}")
            return 0.5
    
    def _generate_ml_reasoning(self, individual_preds: Dict[str, Dict[str, float]], 
                             ensemble_pred: Dict[str, Any]) -> str:
        """Generate human-readable ML reasoning"""
        try:
            reasoning_parts = []
            
            # Overall ensemble assessment
            ensemble_signal = ensemble_pred.get('prediction', 0.0)
            if ensemble_signal > 0.3:
                assessment = "strong bullish signal"
            elif ensemble_signal > 0.1:
                assessment = "moderate bullish signal"
            elif ensemble_signal < -0.3:
                assessment = "strong bearish signal"
            elif ensemble_signal < -0.1:
                assessment = "moderate bearish signal"
            else:
                assessment = "neutral signal"
            
            reasoning_parts.append(f"ML ensemble shows {assessment}")
            
            # Individual model contributions
            for model_name, pred_data in individual_preds.items():
                prediction = pred_data.get('prediction', 0.0)
                confidence = pred_data.get('confidence', 0.33)
                weight = self.model_weights.get(model_name, 0.0)
                
                if abs(prediction) > 0.2:
                    direction = "bullish" if prediction > 0 else "bearish"
                    reasoning_parts.append(f"{model_name} ({weight:.0%} weight) shows {direction} signal with {confidence:.1%} confidence")
            
            # Model agreement
            agreement = self._calculate_model_agreement(individual_preds)
            if agreement > 0.8:
                reasoning_parts.append("High model agreement indicates strong consensus")
            elif agreement < 0.4:
                reasoning_parts.append("Low model agreement suggests uncertainty")
            else:
                reasoning_parts.append("Moderate model agreement")
            
            # Feature importance insights
            if self.feature_importance:
                top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                if top_features:
                    feature_names = [f[0] for f in top_features]
                    reasoning_parts.append(f"Key features: {', '.join(feature_names)}")
            
            return ". ".join(reasoning_parts) + "."
            
        except Exception as e:
            logger.error(f"Failed to generate ML reasoning: {e}")
            return "ML ensemble analysis completed with limited insights."
    
    def _compile_ml_metrics(self, ensemble_pred: Dict[str, Any], 
                          individual_preds: Dict[str, Dict[str, float]], 
                          confidence: float) -> Dict[str, float]:
        """Compile comprehensive ML metrics"""
        try:
            metrics = {
                # Ensemble metrics
                'ensemble_prediction': ensemble_pred.get('prediction', 0.0),
                'ensemble_confidence': ensemble_pred.get('confidence', 0.33),
                'ensemble_probability': ensemble_pred.get('probability', [0.33, 0.34, 0.33]),
                
                # Individual model metrics
                'rf_prediction': individual_preds.get('random_forest', {}).get('prediction', 0.0),
                'rf_confidence': individual_preds.get('random_forest', {}).get('confidence', 0.33),
                'xgb_prediction': individual_preds.get('xgboost', {}).get('prediction', 0.0),
                'xgb_confidence': individual_preds.get('xgboost', {}).get('confidence', 0.33),
                'lstm_prediction': individual_preds.get('lstm', {}).get('prediction', 0.0),
                'lstm_confidence': individual_preds.get('lstm', {}).get('confidence', 0.33),
                
                # Model performance
                'rf_accuracy': self.model_performance['random_forest']['accuracy'],
                'xgb_accuracy': self.model_performance['xgboost']['accuracy'],
                'lstm_accuracy': self.model_performance['lstm']['accuracy'],
                
                # Agreement and confidence
                'model_agreement': self._calculate_model_agreement(individual_preds),
                'final_confidence': confidence,
                
                # Model outputs
                'final_signal': self._calculate_ensemble_signal(ensemble_pred, individual_preds),
                'model_accuracy': self.performance_metrics['accuracy'],
                'total_predictions': float(self.performance_metrics['total_predictions'])
            }
            
            # Add feature importance
            if self.feature_importance:
                for feature, importance in self.feature_importance.items():
                    metrics[f'feature_importance_{feature}'] = importance
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to compile ML metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_feature_importance(self):
        """Calculate feature importance from trained models"""
        try:
            if not self.is_trained or self.training_data is None:
                return
            
            # Get feature importance from Random Forest
            if hasattr(self.individual_models['random_forest'], 'feature_importances_'):
                rf_importance = self.individual_models['random_forest'].feature_importances_
                feature_names = self.training_data.columns
                
                # Store feature importance
                self.feature_importance = dict(zip(feature_names, rf_importance))
                
                # Sort by importance
                self.feature_importance = dict(sorted(
                    self.feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ))
                
                logger.info(f"Calculated feature importance for {len(self.feature_importance)} features")
            
        except Exception as e:
            logger.warning(f"Failed to calculate feature importance: {e}")
    
    def _create_error_output(self, error_message: str) -> ModelOutput:
        """Create error output with standardized format"""
        return ModelOutput(
            signal=0.0,
            confidence=0.0,
            reasoning=f"ML ensemble error: {error_message}",
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
            
            # Update individual model performance (simplified)
            for model_name in self.model_performance.keys():
                self.model_performance[model_name]['predictions'] += 1
                if is_correct:
                    # Update accuracy (simplified - in practice would track each model separately)
                    current_accuracy = self.model_performance[model_name]['accuracy']
                    total_preds = self.model_performance[model_name]['predictions']
                    new_accuracy = (current_accuracy * (total_preds - 1) + 1) / total_preds
                    self.model_performance[model_name]['accuracy'] = new_accuracy
            
            logger.info(f"Updated ML ensemble performance: {self.performance_metrics['accuracy']:.2%}")
            
        except Exception as e:
            logger.error(f"Failed to update ML ensemble model: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        return {
            'model_name': self.model_name,
            'weight': self.weight,
            'is_trained': self.is_trained,
            'model_weights': self.model_weights.copy(),
            'model_performance': self.model_performance.copy(),
            'feature_importance': self.feature_importance.copy(),
            'performance_metrics': self.performance_metrics.copy()
        }


# Example usage and testing
if __name__ == "__main__":
    def test_ml_ensemble_model():
        """Test the ML ensemble model"""
        model = MLEnsembleModel()
        
        # Create mock training data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Generate synthetic features
        X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Generate synthetic targets (future returns)
        y_train = pd.Series(np.random.randn(n_samples) * 0.02)  # 2% daily volatility
        
        # Train the model
        print("Training ML ensemble model...")
        model.train_ensemble(X_train, y_train)
        print("Training completed!")
        
        # Create mock test data
        market_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000000, 1100000, 1200000]
        })
        
        features = pd.DataFrame(
            np.random.randn(3, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Test prediction
        result = model.predict("AAPL", market_data, features)
        
        print("\nML Ensemble Model Test:")
        print(f"Signal: {result.signal:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Key Metrics:")
        for key, value in result.metrics.items():
            if key in ['ensemble_prediction', 'rf_prediction', 'xgb_prediction', 'lstm_prediction', 'model_agreement']:
                print(f"  {key}: {value:.3f}")
        
        # Test update
        feedback = {
            'actual_return': 0.05,
            'predicted_signal': 0.3,
            'symbol': 'AAPL'
        }
        
        model.update(feedback)
        print(f"Updated accuracy: {model.get_confidence():.2%}")
        
        # Show model summary
        summary = model.get_model_summary()
        print(f"\nModel Summary:")
        print(f"  Trained: {summary['is_trained']}")
        print(f"  Model weights: {summary['model_weights']}")
        print(f"  Feature importance (top 3): {list(summary['feature_importance'].items())[:3]}")
    
    # Run test
    test_ml_ensemble_model()
