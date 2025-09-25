"""
Simple ML Models for 5-Ticker QuantAI Platform.

This module implements simplified ML models without scipy dependencies
specifically optimized for AMZN, META, NVDA, GOOGL, and AAPL trading.

Features:
- Basic Random Forest models
- Simple feature selection
- No scipy dependencies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Basic ML libraries (no scipy dependencies)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    mse: float
    rmse: float
    r2: float
    mae: float
    training_time: float


class SimpleMLModels:
    """
    Simple ML models for 5-ticker focused trading.
    
    Implements basic machine learning without scipy dependencies
    specifically optimized for AMZN, META, NVDA, GOOGL, AAPL.
    """
    
    def __init__(self):
        """Initialize simple ML models."""
        self.tickers = ['AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL']
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.scalers = {}
        
        # Model configurations
        self.rf_config = {
            'n_estimators': 50,
            'max_depth': 8,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
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
        scaler = StandardScaler()
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
    
    def predict(self, X: np.ndarray, ticker: str) -> np.ndarray:
        """
        Make predictions for a ticker.
        
        Args:
            X: Feature matrix
            ticker: Ticker symbol
            
        Returns:
            Predictions
        """
        model_key = f'{ticker}_rf'
        scaler_key = f'{ticker}_rf'
        
        if model_key in self.models and scaler_key in self.scalers:
            model = self.models[model_key]
            scaler = self.scalers[scaler_key]
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make predictions
            return model.predict(X_scaled)
        else:
            return np.zeros(len(X))
    
    def calculate_feature_importance(self, ticker: str) -> Dict[str, float]:
        """
        Calculate feature importance for a ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary with feature importance
        """
        model_key = f'{ticker}_rf'
        if model_key in self.models:
            model = self.models[model_key]
            if hasattr(model, 'feature_importances_'):
                return dict(zip([f'feature_{i}' for i in range(len(model.feature_importances_))], 
                               model.feature_importances_))
        
        return {}
    
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
                ticker_results['RandomForest'] = {'model': rf_model, 'status': 'success'}
                
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
                
                # Get predictions from Random Forest
                rf_pred = self.predict(X, ticker)
                ticker_predictions['rf'] = rf_pred
                
                predictions[ticker] = ticker_predictions
                
            except Exception as e:
                logger.error(f"Error getting predictions for {ticker}: {e}")
                continue
        
        return predictions


# Global instance for easy access
simple_ml_models = SimpleMLModels()
