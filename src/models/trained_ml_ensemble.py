#!/usr/bin/env python3
"""
Trained ML Ensemble Model Integration

This module integrates the trained ML ensemble models from the real dataset training
into the four-model decision engine.

Uses the trained models from models/ directory.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .base_models import BaseModel, ModelOutput
from ..utils.common_imports import setup_logger
from ..utils.feature_engineering import feature_engineer

logger = setup_logger(__name__)


class TrainedMLEnsembleModel:
    """Trained ML Ensemble Model using real dataset models."""
    
    def __init__(self, models_dir: str = "models"):
        self.name = "TrainedMLEnsemble"
        self.weight = 0.35
        
        self.models_dir = Path(models_dir)
        self.models = {}
        self.feature_columns = []
        self.ensemble_weights = {
            'linear_model': 0.50,
            'naive_bayes': 0.30,
            'decision_tree': 0.20
        }
        
        # Load trained models
        self._load_trained_models()
        
        # Model performance tracking
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'last_updated': datetime.now()
        }
    
    def _load_trained_models(self) -> None:
        """Load trained models from files."""
        try:
            # Load metadata
            metadata_path = self.models_dir / "simple_ensemble_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_columns = metadata.get('feature_columns', [])
                    self.ensemble_weights = metadata.get('ensemble_weights', self.ensemble_weights)
                    logger.info(f"✅ Loaded metadata: {len(self.feature_columns)} features")
            else:
                logger.warning("❌ Metadata file not found, using defaults")
            
            # Load individual models
            model_files = {
                'linear_model': 'linear_model_model.json',
                'naive_bayes': 'naive_bayes_model.json',
                'decision_tree': 'decision_tree_model.json'
            }
            
            for model_name, filename in model_files.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    with open(model_path, 'r') as f:
                        self.models[model_name] = json.load(f)
                    logger.info(f"✅ Loaded {model_name}")
                else:
                    logger.warning(f"❌ Model file not found: {model_path}")
            
            if self.models:
                logger.info(f"✅ Loaded {len(self.models)} trained models")
            else:
                logger.warning("❌ No trained models loaded")
                
        except Exception as e:
            logger.error(f"❌ Failed to load trained models: {e}")
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction using the same feature engineering."""
        try:
            # Use the same feature engineering as training
            features = feature_engineer.create_technical_features(data)
            
            # Select only the features used in training
            if self.feature_columns:
                available_features = [col for col in self.feature_columns if col in features.columns]
                if available_features:
                    features = features[available_features]
                else:
                    logger.warning("❌ No matching features found, using all available features")
            
            # Fill missing values
            features = features.fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Feature preparation failed: {e}")
            return pd.DataFrame()
    
    def _predict_linear_model(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions using the linear model."""
        try:
            if 'linear_model' not in self.models:
                return np.zeros(len(features))
            
            model = self.models['linear_model']
            weights = np.array(model['weights'], dtype=float)
            
            # Add bias term
            X = np.column_stack([np.ones(len(features)), features.values.astype(float)])
            
            # Make predictions
            predictions = X @ weights
            
            # Convert to binary predictions
            binary_predictions = (predictions > 0.5).astype(int)
            
            return binary_predictions
            
        except Exception as e:
            logger.error(f"❌ Linear model prediction failed: {e}")
            return np.zeros(len(features))
    
    def _predict_naive_bayes(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions using the Naive Bayes model."""
        try:
            if 'naive_bayes' not in self.models:
                return np.zeros(len(features))
            
            model = self.models['naive_bayes']
            class_priors = model['class_priors']
            feature_stats = model['feature_stats']
            
            predictions = []
            for _, sample in features.iterrows():
                class_probs = {}
                for class_label in class_priors.keys():
                    # Calculate likelihood for each feature
                    likelihood = 1.0
                    for feature in self.feature_columns:
                        if feature in sample.index and feature in feature_stats[class_label]['mean']:
                            mean = float(feature_stats[class_label]['mean'][feature])
                            std = float(feature_stats[class_label]['std'][feature])
                            # Simple Gaussian likelihood
                            likelihood *= np.exp(-0.5 * ((float(sample[feature]) - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
                    
                    # Calculate posterior probability
                    class_probs[class_label] = float(class_priors[class_label]) * likelihood
                
                # Predict class with highest probability
                predicted_class = max(class_probs, key=class_probs.get)
                predictions.append(predicted_class)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"❌ Naive Bayes prediction failed: {e}")
            return np.zeros(len(features))
    
    def _predict_decision_tree(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions using the decision tree model."""
        try:
            if 'decision_tree' not in self.models:
                return np.zeros(len(features))
            
            tree = self.models['decision_tree']['tree']
            
            def predict_tree(tree, sample):
                node = tree
                while 'prediction' not in node:
                    if float(sample[node['feature']]) <= float(node['threshold']):
                        node = node['left']
                    else:
                        node = node['right']
                return float(node['prediction'])
            
            predictions = []
            for _, sample in features.iterrows():
                pred = predict_tree(tree, sample)
                predictions.append(pred)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"❌ Decision tree prediction failed: {e}")
            return np.zeros(len(features))
    
    def predict(self, data: pd.DataFrame) -> ModelOutput:
        """Make ensemble prediction using all trained models."""
        try:
            # Prepare features
            features = self._prepare_features(data)
            
            if features.empty:
                return {
                    'signal': 0.0,
                    'confidence': 0.0,
                    'reasoning': "No features available for prediction",
                    'model_name': self.name,
                    'timestamp': datetime.now()
                }
            
            # Get predictions from each model
            linear_pred = self._predict_linear_model(features)
            naive_bayes_pred = self._predict_naive_bayes(features)
            decision_tree_pred = self._predict_decision_tree(features)
            
            # Calculate weighted ensemble prediction
            ensemble_pred = (
                self.ensemble_weights['linear_model'] * linear_pred +
                self.ensemble_weights['naive_bayes'] * naive_bayes_pred +
                self.ensemble_weights['decision_tree'] * decision_tree_pred
            )
            
            # Convert to signal (-1, 0, 1)
            if len(ensemble_pred) > 0:
                avg_pred = np.mean(ensemble_pred)
                if avg_pred > 0.6:
                    signal = 1.0  # Buy
                elif avg_pred < 0.4:
                    signal = -1.0  # Sell
                else:
                    signal = 0.0  # Hold
                
                # Calculate confidence based on prediction consistency
                confidence = min(1.0, abs(avg_pred - 0.5) * 2)
            else:
                signal = 0.0
                confidence = 0.0
            
            # Update performance metrics
            self.performance_metrics['total_predictions'] += 1
            self.performance_metrics['last_updated'] = datetime.now()
            
            # Create reasoning
            reasoning = f"Ensemble prediction: Linear={np.mean(linear_pred):.3f}, " \
                       f"NaiveBayes={np.mean(naive_bayes_pred):.3f}, " \
                       f"DecisionTree={np.mean(decision_tree_pred):.3f}, " \
                       f"Final={avg_pred:.3f}"
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reasoning': reasoning,
                'model_name': self.name,
                'timestamp': datetime.now(),
                'metadata': {
                    'ensemble_weights': self.ensemble_weights,
                    'individual_predictions': {
                        'linear_model': float(np.mean(linear_pred)),
                        'naive_bayes': float(np.mean(naive_bayes_pred)),
                        'decision_tree': float(np.mean(decision_tree_pred))
                    },
                    'feature_count': len(features.columns)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Ensemble prediction failed: {e}")
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'reasoning': f"Prediction failed: {str(e)}",
                'model_name': self.name,
                'timestamp': datetime.now()
            }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and performance."""
        return {
            'model_name': self.name,
            'weight': self.weight,
            'models_loaded': list(self.models.keys()),
            'feature_columns_count': len(self.feature_columns),
            'ensemble_weights': self.ensemble_weights,
            'performance_metrics': self.performance_metrics,
            'models_dir': str(self.models_dir),
            'status': 'ready' if self.models else 'not_loaded'
        }
    
    def update_performance(self, actual_outcome: float, prediction: float) -> None:
        """Update model performance based on actual outcomes."""
        try:
            # Calculate if prediction was correct (for binary classification)
            predicted_direction = 1 if prediction > 0.5 else 0
            actual_direction = 1 if actual_outcome > 0 else 0
            
            if predicted_direction == actual_direction:
                self.performance_metrics['correct_predictions'] += 1
            
            # Update accuracy
            if self.performance_metrics['total_predictions'] > 0:
                self.performance_metrics['accuracy'] = (
                    self.performance_metrics['correct_predictions'] / 
                    self.performance_metrics['total_predictions']
                )
            
            self.performance_metrics['last_updated'] = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Performance update failed: {e}")


def main():
    """Test the trained ML ensemble model."""
    # Create test data
    test_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=100),
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(100, 200, 100),
        'Low': np.random.uniform(100, 200, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.uniform(1000000, 10000000, 100)
    })
    
    # Initialize model
    model = TrainedMLEnsembleModel()
    
    # Test prediction
    result = model.predict(test_data)
    
    print("🎯 Trained ML Ensemble Model Test")
    print("=" * 50)
    print(f"Signal: {result.signal}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Model Status: {model.get_model_status()['status']}")
    print(f"Models Loaded: {len(model.get_model_status()['models_loaded'])}")


if __name__ == "__main__":
    main()
