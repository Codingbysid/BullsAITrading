"""
Trading models module implementing ensemble ML approaches.

This module provides various ML models for trading predictions including:
- Random Forest for classification and regression
- XGBoost for gradient boosting
- LSTM for time series prediction
- Ensemble methods for combining predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class BaseTradingModel:
    """Base class for all trading models."""
    
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period
        self.is_fitted = False
        self.feature_importance_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model to training data."""
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        raise NotImplementedError
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities."""
        raise NotImplementedError
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        return self.feature_importance_


class RandomForestTradingModel(BaseTradingModel):
    """Random Forest model for trading predictions."""
    
    def __init__(
        self, 
        lookback_period: int = 252,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        random_state: int = 42
    ):
        super().__init__(lookback_period)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        # Initialize models for classification and regression
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit both classification and regression models."""
        # Remove any NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            logger.error("No valid data for training")
            return
        
        # Fit classifier (for direction prediction)
        y_direction = (y_clean > 0).astype(int)
        self.classifier.fit(X_clean, y_direction)
        
        # Fit regressor (for return prediction)
        self.regressor.fit(X_clean, y_clean)
        
        self.is_fitted = True
        
        # Store feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': X_clean.columns,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Random Forest model fitted on {len(X_clean)} samples")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Use regression model for return predictions
        return self.regressor.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.classifier.predict_proba(X)
    
    def predict_direction(self, X: pd.DataFrame) -> np.ndarray:
        """Predict price direction (up/down)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.classifier.predict(X)


class XGBoostTradingModel(BaseTradingModel):
    """XGBoost model for trading predictions."""
    
    def __init__(
        self,
        lookback_period: int = 252,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        super().__init__(lookback_period)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Initialize models
        self.classifier = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric='logloss'
        )
        
        self.regressor = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric='rmse'
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit both classification and regression models."""
        # Remove any NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) == 0:
            logger.error("No valid data for training")
            return
        
        # Fit classifier
        y_direction = (y_clean > 0).astype(int)
        self.classifier.fit(X_clean, y_direction)
        
        # Fit regressor
        self.regressor.fit(X_clean, y_clean)
        
        self.is_fitted = True
        
        # Store feature importance
        self.feature_importance_ = pd.DataFrame({
            'feature': X_clean.columns,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"XGBoost model fitted on {len(X_clean)} samples")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.regressor.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.classifier.predict_proba(X)
    
    def predict_direction(self, X: pd.DataFrame) -> np.ndarray:
        """Predict price direction."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.classifier.predict(X)


class LSTMTradingModel(BaseTradingModel):
    """LSTM model for time series prediction."""
    
    def __init__(
        self,
        lookback_period: int = 252,
        sequence_length: int = 60,
        lstm_units: int = 50,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        super().__init__(lookback_period)
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
    
    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit LSTM model."""
        from sklearn.preprocessing import StandardScaler
        
        # Remove NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < self.sequence_length + 1:
            logger.error("Insufficient data for LSTM training")
            return
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_clean.values)
        
        # Build LSTM model
        self.model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, X_scaled.shape[1])),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        self.model.fit(
            X_seq, y_seq,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.is_fitted = True
        logger.info(f"LSTM model fitted on {len(X_seq)} sequences")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Create sequences for prediction
        if len(X_scaled) < self.sequence_length:
            # Pad with zeros if insufficient data
            padding = np.zeros((self.sequence_length - len(X_scaled), X_scaled.shape[1]))
            X_padded = np.vstack([padding, X_scaled])
        else:
            X_padded = X_scaled[-self.sequence_length:]
        
        # Reshape for LSTM
        X_seq = X_padded.reshape(1, self.sequence_length, X_scaled.shape[1])
        
        # Predict
        prediction = self.model.predict(X_seq, verbose=0)
        return prediction.flatten()


class EnsembleTradingModel(BaseTradingModel):
    """Ensemble model combining multiple base models."""
    
    def __init__(
        self,
        lookback_period: int = 252,
        models: Optional[List[BaseTradingModel]] = None,
        weights: Optional[List[float]] = None
    ):
        super().__init__(lookback_period)
        self.models = models or [
            RandomForestTradingModel(lookback_period),
            XGBoostTradingModel(lookback_period),
            LSTMTradingModel(lookback_period)
        ]
        self.weights = weights or [1/len(self.models)] * len(self.models)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit all ensemble models."""
        for i, model in enumerate(self.models):
            try:
                logger.info(f"Training model {i+1}/{len(self.models)}")
                model.fit(X, y)
            except Exception as e:
                logger.error(f"Error training model {i+1}: {e}")
        
        self.is_fitted = True
        logger.info("Ensemble model training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        for model in self.models:
            if model.is_fitted:
                pred = model.predict(X)
                predictions.append(pred)
        
        if not predictions:
            raise ValueError("No trained models available for prediction")
        
        # Weighted average of predictions
        ensemble_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += self.weights[i] * pred
        
        return ensemble_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble probability predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        probabilities = []
        for model in self.models:
            if model.is_fitted and hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probabilities.append(proba)
        
        if not probabilities:
            raise ValueError("No models with probability prediction available")
        
        # Weighted average of probabilities
        ensemble_proba = np.zeros_like(probabilities[0])
        for i, proba in enumerate(probabilities):
            ensemble_proba += self.weights[i] * proba
        
        return ensemble_proba
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return {
            f"model_{i}": weight 
            for i, weight in enumerate(self.weights)
        }
    
    def update_weights(self, new_weights: List[float]) -> None:
        """Update model weights based on performance."""
        if len(new_weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        total_weight = sum(new_weights)
        self.weights = [w / total_weight for w in new_weights]
        
        logger.info(f"Updated ensemble weights: {self.weights}")


def create_ensemble_model(lookback_period: int = 252) -> EnsembleTradingModel:
    """
    Create a default ensemble model with standard configurations.
    
    Args:
        lookback_period: Lookback period for models
        
    Returns:
        Configured ensemble model
    """
    return EnsembleTradingModel(lookback_period)
