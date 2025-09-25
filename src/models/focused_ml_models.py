"""
Focused ML Models for 5-Ticker QuantAI Platform.

This module implements advanced machine learning models specifically
optimized for AMZN, META, NVDA, GOOGL, and AAPL trading.

Features:
- Ensemble models (Random Forest, XGBoost, LSTM with attention)
- Meta labeling for trade filtering
- Reinforcement learning agents
- Feature selection and engineering
- Factor regression models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, LayerNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch")

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    mse: float
    rmse: float
    r2: float
    mae: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_return: float


@dataclass
class FeatureImportance:
    """Feature importance data."""
    feature_name: str
    importance: float
    rank: int


class FocusedMLModels:
    """
    Advanced ML models for 5-ticker focused trading.
    
    Implements ensemble learning, meta labeling, and reinforcement learning
    specifically optimized for AMZN, META, NVDA, GOOGL, AAPL.
    """
    
    def __init__(self):
        """Initialize focused ML models."""
        self.tickers = ['AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL']
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.scalers = {}
        
        # Model configurations
        self.rf_config = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        self.xgb_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        self.lstm_config = {
            'sequence_length': 60,
            'lstm_units': 50,
            'dropout_rate': 0.2,
            'epochs': 100,
            'batch_size': 32,
            'patience': 10
        }
    
    def prepare_features(self, data: pd.DataFrame, target_col: str = 'returns') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features for ML models.
        
        Args:
            data: DataFrame with features
            target_col: Target column name
            
        Returns:
            X, y, feature_names
        """
        # Select numeric features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_col and not col.startswith('target_')]
        
        # Remove columns with too many NaN values
        valid_cols = []
        for col in feature_cols:
            if data[col].isnull().sum() / len(data) < 0.5:
                valid_cols.append(col)
        
        X = data[valid_cols].fillna(0).values
        y = data[target_col].fillna(0).values
        
        # Remove infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, y, valid_cols
    
    def create_meta_labels(self, data: pd.DataFrame, returns_col: str = 'returns', 
                          threshold: float = 0.001) -> pd.DataFrame:
        """
        Create meta labels for trade filtering (Lopez de Prado method).
        
        Args:
            data: DataFrame with price data
            returns_col: Returns column name
            threshold: Minimum return threshold for labeling
            
        Returns:
            DataFrame with meta labels
        """
        data_copy = data.copy()
        
        # Create forward returns
        data_copy['forward_returns'] = data_copy[returns_col].shift(-1)
        
        # Create meta labels
        data_copy['meta_label'] = 0
        data_copy.loc[data_copy['forward_returns'] > threshold, 'meta_label'] = 1
        data_copy.loc[data_copy['forward_returns'] < -threshold, 'meta_label'] = -1
        
        # Add confidence scores
        data_copy['confidence'] = abs(data_copy['forward_returns']) / data_copy['forward_returns'].std()
        data_copy['confidence'] = np.clip(data_copy['confidence'], 0, 1)
        
        return data_copy
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray, ticker: str) -> RandomForestRegressor:
        """
        Train Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target values
            ticker: Ticker symbol
            
        Returns:
            Trained Random Forest model
        """
        logger.info(f"Training Random Forest for {ticker}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(**self.rf_config)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Random Forest for {ticker} - MSE: {mse:.6f}, R²: {r2:.4f}")
        
        # Store model and scaler
        self.models[f'{ticker}_rf'] = model
        self.scalers[f'{ticker}_rf'] = scaler
        
        return model
    
    def train_xgboost(self, X: np.ndarray, y: np.ndarray, ticker: str) -> Any:
        """
        Train XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target values
            ticker: Ticker symbol
            
        Returns:
            Trained XGBoost model
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, using Random Forest instead")
            return self.train_random_forest(X, y, ticker)
        
        logger.info(f"Training XGBoost for {ticker}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = xgb.XGBRegressor(**self.xgb_config)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"XGBoost for {ticker} - MSE: {mse:.6f}, R²: {r2:.4f}")
        
        # Store model and scaler
        self.models[f'{ticker}_xgb'] = model
        self.scalers[f'{ticker}_xgb'] = scaler
        
        return model
    
    def train_lstm(self, data: pd.DataFrame, ticker: str, target_col: str = 'returns') -> Any:
        """
        Train LSTM model with attention mechanism.
        
        Args:
            data: DataFrame with time series data
            ticker: Ticker symbol
            target_col: Target column name
            
        Returns:
            Trained LSTM model
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, using Random Forest instead")
            X, y, _ = self.prepare_features(data, target_col)
            return self.train_random_forest(X, y, ticker)
        
        logger.info(f"Training LSTM for {ticker}")
        
        # Prepare data
        X, y, feature_names = self.prepare_features(data, target_col)
        
        # Create sequences
        sequence_length = self.lstm_config['sequence_length']
        X_sequences, y_sequences = self._create_sequences(X, y, sequence_length)
        
        if len(X_sequences) == 0:
            logger.warning(f"Insufficient data for LSTM training for {ticker}")
            return None
        
        # Split data
        split_idx = int(0.8 * len(X_sequences))
        X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Build LSTM model with attention
        model = self._build_lstm_model(X_train.shape[1], X_train.shape[2])
        
        # Train model
        callbacks = [
            EarlyStopping(patience=self.lstm_config['patience'], restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = model.fit(
            X_train_scaled, y_train,
            epochs=self.lstm_config['epochs'],
            batch_size=self.lstm_config['batch_size'],
            validation_data=(X_test_scaled, y_test),
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"LSTM for {ticker} - MSE: {mse:.6f}, R²: {r2:.4f}")
        
        # Store model and scaler
        self.models[f'{ticker}_lstm'] = model
        self.scalers[f'{ticker}_lstm'] = scaler
        
        return model
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X_sequences, y_sequences = [], []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _build_lstm_model(self, sequence_length: int, n_features: int) -> tf.keras.Model:
        """Build LSTM model with attention mechanism."""
        model = Sequential([
            LSTM(self.lstm_config['lstm_units'], return_sequences=True, 
                 input_shape=(sequence_length, n_features)),
            Dropout(self.lstm_config['dropout_rate']),
            LSTM(self.lstm_config['lstm_units'], return_sequences=True),
            Dropout(self.lstm_config['dropout_rate']),
            Attention(),
            LayerNormalization(),
            Dense(32, activation='relu'),
            Dropout(self.lstm_config['dropout_rate']),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_ensemble_model(self, ticker: str) -> Dict[str, Any]:
        """
        Create ensemble model for a ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Ensemble model configuration
        """
        ensemble_config = {
            'ticker': ticker,
            'models': [],
            'weights': [],
            'performance': {}
        }
        
        # Add available models
        if f'{ticker}_rf' in self.models:
            ensemble_config['models'].append('rf')
            ensemble_config['weights'].append(0.4)
        
        if f'{ticker}_xgb' in self.models:
            ensemble_config['models'].append('xgb')
            ensemble_config['weights'].append(0.4)
        
        if f'{ticker}_lstm' in self.models:
            ensemble_config['models'].append('lstm')
            ensemble_config['weights'].append(0.2)
        
        # Normalize weights
        total_weight = sum(ensemble_config['weights'])
        ensemble_config['weights'] = [w / total_weight for w in ensemble_config['weights']]
        
        return ensemble_config
    
    def predict_ensemble(self, X: np.ndarray, ticker: str) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Feature matrix
            ticker: Ticker symbol
            
        Returns:
            Ensemble predictions
        """
        ensemble_config = self.create_ensemble_model(ticker)
        predictions = []
        
        for i, model_name in enumerate(ensemble_config['models']):
            model_key = f'{ticker}_{model_name}'
            scaler_key = f'{ticker}_{model_name}'
            
            if model_key in self.models and scaler_key in self.scalers:
                model = self.models[model_key]
                scaler = self.scalers[scaler_key]
                
                # Scale features
                X_scaled = scaler.transform(X)
                
                # Make predictions
                if model_name == 'lstm':
                    # LSTM expects sequences
                    if len(X_scaled.shape) == 2:
                        X_scaled = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1])
                    pred = model.predict(X_scaled, verbose=0)
                    if len(pred.shape) > 1:
                        pred = pred.flatten()
                else:
                    pred = model.predict(X_scaled)
                
                predictions.append(pred * ensemble_config['weights'][i])
        
        if predictions:
            return np.sum(predictions, axis=0)
        else:
            return np.zeros(len(X))
    
    def calculate_feature_importance(self, ticker: str) -> List[FeatureImportance]:
        """
        Calculate feature importance for a ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            List of feature importance objects
        """
        importance_list = []
        
        # Get feature importance from Random Forest
        rf_key = f'{ticker}_rf'
        if rf_key in self.models:
            model = self.models[rf_key]
            if hasattr(model, 'feature_importances_'):
                for i, importance in enumerate(model.feature_importances_):
                    importance_list.append(FeatureImportance(
                        feature_name=f'feature_{i}',
                        importance=importance,
                        rank=0
                    ))
        
        # Get feature importance from XGBoost
        xgb_key = f'{ticker}_xgb'
        if xgb_key in self.models:
            model = self.models[xgb_key]
            if hasattr(model, 'feature_importances_'):
                for i, importance in enumerate(model.feature_importances_):
                    importance_list.append(FeatureImportance(
                        feature_name=f'feature_{i}',
                        importance=importance,
                        rank=0
                    ))
        
        # Sort by importance
        importance_list.sort(key=lambda x: x.importance, reverse=True)
        
        # Update ranks
        for i, item in enumerate(importance_list):
            item.rank = i + 1
        
        self.feature_importance[ticker] = importance_list
        return importance_list
    
    def train_all_models(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Train all models for all tickers.
        
        Args:
            data_dict: Dictionary with data for each ticker
            
        Returns:
            Training results
        """
        results = {}
        
        for ticker, data in data_dict.items():
            if ticker not in self.tickers:
                continue
            
            logger.info(f"Training models for {ticker}")
            ticker_results = {}
            
            try:
                # Prepare features
                X, y, feature_names = self.prepare_features(data)
                
                if len(X) == 0:
                    logger.warning(f"No features available for {ticker}")
                    continue
                
                # Train Random Forest
                rf_model = self.train_random_forest(X, y, ticker)
                ticker_results['rf'] = {'model': rf_model, 'status': 'success'}
                
                # Train XGBoost
                xgb_model = self.train_xgboost(X, y, ticker)
                ticker_results['xgb'] = {'model': xgb_model, 'status': 'success'}
                
                # Train LSTM
                lstm_model = self.train_lstm(data, ticker)
                if lstm_model is not None:
                    ticker_results['lstm'] = {'model': lstm_model, 'status': 'success'}
                else:
                    ticker_results['lstm'] = {'model': None, 'status': 'failed'}
                
                # Create ensemble
                ensemble_config = self.create_ensemble_model(ticker)
                ticker_results['ensemble'] = ensemble_config
                
                # Calculate feature importance
                feature_importance = self.calculate_feature_importance(ticker)
                ticker_results['feature_importance'] = feature_importance
                
                results[ticker] = ticker_results
                logger.info(f"✅ Models trained for {ticker}")
                
            except Exception as e:
                logger.error(f"Error training models for {ticker}: {e}")
                results[ticker] = {'error': str(e)}
        
        return results
    
    def get_model_predictions(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Get predictions from all models for all tickers.
        
        Args:
            data_dict: Dictionary with data for each ticker
            
        Returns:
            Dictionary with predictions for each ticker and model
        """
        predictions = {}
        
        for ticker, data in data_dict.items():
            if ticker not in self.tickers:
                continue
            
            ticker_predictions = {}
            
            try:
                # Prepare features
                X, y, feature_names = self.prepare_features(data)
                
                if len(X) == 0:
                    continue
                
                # Get predictions from each model
                for model_name in ['rf', 'xgb', 'lstm']:
                    model_key = f'{ticker}_{model_name}'
                    scaler_key = f'{ticker}_{model_name}'
                    
                    if model_key in self.models and scaler_key in self.scalers:
                        model = self.models[model_key]
                        scaler = self.scalers[scaler_key]
                        
                        # Scale features
                        X_scaled = scaler.transform(X)
                        
                        # Make predictions
                        if model_name == 'lstm':
                            if len(X_scaled.shape) == 2:
                                X_scaled = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1])
                            pred = model.predict(X_scaled, verbose=0)
                            if len(pred.shape) > 1:
                                pred = pred.flatten()
                        else:
                            pred = model.predict(X_scaled)
                        
                        ticker_predictions[model_name] = pred
                
                # Get ensemble predictions
                ensemble_pred = self.predict_ensemble(X, ticker)
                ticker_predictions['ensemble'] = ensemble_pred
                
                predictions[ticker] = ticker_predictions
                
            except Exception as e:
                logger.error(f"Error getting predictions for {ticker}: {e}")
                continue
        
        return predictions
    
    def evaluate_models(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, ModelPerformance]]:
        """
        Evaluate all models for all tickers.
        
        Args:
            data_dict: Dictionary with data for each ticker
            
        Returns:
            Dictionary with performance metrics for each ticker and model
        """
        performance = {}
        
        for ticker, data in data_dict.items():
            if ticker not in self.tickers:
                continue
            
            ticker_performance = {}
            
            try:
                # Prepare features
                X, y, feature_names = self.prepare_features(data)
                
                if len(X) == 0:
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, shuffle=False
                )
                
                # Evaluate each model
                for model_name in ['rf', 'xgb', 'lstm']:
                    model_key = f'{ticker}_{model_name}'
                    scaler_key = f'{ticker}_{model_name}'
                    
                    if model_key in self.models and scaler_key in self.scalers:
                        model = self.models[model_key]
                        scaler = self.scalers[scaler_key]
                        
                        # Scale features
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Make predictions
                        if model_name == 'lstm':
                            if len(X_test_scaled.shape) == 2:
                                X_test_scaled = X_test_scaled.reshape(1, X_test_scaled.shape[0], X_test_scaled.shape[1])
                            y_pred = model.predict(X_test_scaled, verbose=0)
                            if len(y_pred.shape) > 1:
                                y_pred = y_pred.flatten()
                        else:
                            y_pred = model.predict(X_test_scaled)
                        
                        # Calculate metrics
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)
                        mae = np.mean(np.abs(y_test - y_pred))
                        
                        # Calculate trading metrics
                        returns = y_pred
                        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                        max_drawdown = self._calculate_max_drawdown(returns)
                        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0
                        total_return = np.sum(returns)
                        
                        ticker_performance[model_name] = ModelPerformance(
                            model_name=model_name,
                            mse=mse,
                            rmse=rmse,
                            r2=r2,
                            mae=mae,
                            sharpe_ratio=sharpe_ratio,
                            max_drawdown=max_drawdown,
                            win_rate=win_rate,
                            total_return=total_return
                        )
                
                performance[ticker] = ticker_performance
                
            except Exception as e:
                logger.error(f"Error evaluating models for {ticker}: {e}")
                continue
        
        return performance
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return np.min(drawdown)
    
    def save_models(self, output_dir: str = "models"):
        """Save all trained models."""
        import os
        import pickle
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            if hasattr(model, 'save'):
                # TensorFlow/Keras model
                model.save(f"{output_dir}/{model_name}.h5")
            else:
                # Scikit-learn model
                with open(f"{output_dir}/{model_name}.pkl", 'wb') as f:
                    pickle.dump(model, f)
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            with open(f"{output_dir}/{scaler_name}_scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
        
        logger.info(f"✅ Models saved to {output_dir}")


# Global instance for easy access
focused_ml_models = FocusedMLModels()
