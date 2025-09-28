"""
Basic ML Models for 5-Ticker QuantAI Platform.

This module implements basic ML models without any scipy/sklearn dependencies
specifically optimized for AMZN, META, NVDA, GOOGL, and AAPL trading.

Features:
- Basic linear regression
- Simple feature selection
- No external ML dependencies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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


class BasicLinearRegression:
    """Basic linear regression implementation without scipy/sklearn."""
    
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.feature_names = None
        
    def fit(self, X, y):
        """Fit linear regression model."""
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Calculate coefficients using normal equation
        try:
            # (X'X)^(-1)X'y
            XtX = np.dot(X_with_intercept.T, X_with_intercept)
            XtX_inv = np.linalg.inv(XtX)
            Xty = np.dot(X_with_intercept.T, y)
            coefficients = np.dot(XtX_inv, Xty)
            
            self.intercept = coefficients[0]
            self.coefficients = coefficients[1:]
            
        except np.linalg.LinAlgError:
            # Fallback to simple mean
            self.intercept = np.mean(y)
            self.coefficients = np.zeros(X.shape[1])
    
    def predict(self, X):
        """Make predictions."""
        if self.coefficients is None:
            return np.zeros(X.shape[0])
        
        return self.intercept + np.dot(X, self.coefficients)
    
    def score(self, X, y):
        """Calculate R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)


class BasicMLModels:
    """
    Basic ML models for 5-ticker focused trading.
    
    Implements basic machine learning without any external dependencies
    specifically optimized for AMZN, META, NVDA, GOOGL, AAPL.
    """
    
    def __init__(self):
        """Initialize basic ML models."""
        self.tickers = ['AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL']
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.scalers = {}
    
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
    
    def scale_features(self, X):
        """Simple feature scaling."""
        # Z-score normalization
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        return (X - mean) / std
    
    def train_linear_regression(self, X: np.ndarray, y: np.ndarray, ticker: str) -> BasicLinearRegression:
        """
        Train linear regression model.
        
        Args:
            X: Feature matrix
            y: Target values
            ticker: Ticker symbol
            
        Returns:
            Trained linear regression model
        """
        logger.info(f"Training Linear Regression for {ticker}")
        
        # Scale features
        X_scaled = self.scale_features(X)
        
        # Train model
        model = BasicLinearRegression()
        model.fit(X_scaled, y)
        
        # Evaluate model
        y_pred = model.predict(X_scaled)
        mse = np.mean((y - y_pred) ** 2)
        r2 = model.score(X_scaled, y)
        
        logger.info(f"Linear Regression for {ticker} - MSE: {mse:.6f}, R²: {r2:.4f}")
        
        # Store model
        self.models[f'{ticker}_lr'] = model
        
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
        model_key = f'{ticker}_lr'
        
        if model_key in self.models:
            model = self.models[model_key]
            
            # Scale features
            X_scaled = self.scale_features(X)
            
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
        model_key = f'{ticker}_lr'
        if model_key in self.models:
            model = self.models[model_key]
            if model.coefficients is not None:
                # Use absolute coefficients as importance
                importance = np.abs(model.coefficients)
                return dict(zip([f'feature_{i}' for i in range(len(importance))], importance))
        
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
                
                # Train Linear Regression
                lr_model = self.train_linear_regression(X, y, ticker)
                ticker_results['LinearRegression'] = {'model': lr_model, 'status': 'success'}
                
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
                
                # Get predictions from Linear Regression
                lr_pred = self.predict(X, ticker)
                ticker_predictions['lr'] = lr_pred
                
                predictions[ticker] = ticker_predictions
                
            except Exception as e:
                logger.error(f"Error getting predictions for {ticker}: {e}")
                continue
        
        return predictions


# Global instance for easy access
basic_ml_models = BasicMLModels()
