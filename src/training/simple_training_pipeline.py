"""
Simple Training Pipeline for 5-Ticker QuantAI Platform.

This module implements simplified training without scipy dependencies
specifically optimized for AMZN, META, NVDA, GOOGL, and AAPL.

Features:
- Basic Random Forest training
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
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration for simple models."""
    # Data parameters
    lookback_period: int = 252  # 1 year
    test_size: float = 0.2
    
    # Cross-validation parameters
    n_splits: int = 5
    
    # Model parameters
    rf_n_estimators: int = 50
    rf_max_depth: int = 8
    
    # Feature selection
    n_features: int = 20


@dataclass
class ModelResults:
    """Model training results."""
    model_name: str
    ticker: str
    train_score: float
    test_score: float
    mse: float
    r2: float
    training_time: float


class SimpleTrainingPipeline:
    """
    Simple training pipeline for 5-ticker focused models.
    
    Implements basic training without scipy dependencies
    specifically optimized for AMZN, META, NVDA, GOOGL, AAPL.
    """
    
    def __init__(self, config: TrainingConfig = None):
        """Initialize simple training pipeline."""
        self.config = config or TrainingConfig()
        self.tickers = ['AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL']
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def prepare_training_data(
        self, 
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        Prepare training data for all tickers.
        
        Args:
            data: Dictionary with data for each ticker
            
        Returns:
            Dictionary with (X, y, feature_names) for each ticker
        """
        training_data = {}
        
        for ticker, df in data.items():
            if ticker not in self.tickers:
                continue
            
            logger.info(f"Preparing training data for {ticker}")
            
            try:
                # Select numeric features
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                feature_cols = [col for col in numeric_cols if col not in ['returns', 'target'] and not col.startswith('target_')]
                
                # Remove columns with too many NaN values
                valid_cols = []
                for col in feature_cols:
                    if df[col].isnull().sum() / len(df) < 0.5:
                        valid_cols.append(col)
                
                # Prepare features
                X = df[valid_cols].fillna(0).values
                
                # Create target variable (future returns)
                y = df['returns'].shift(-1).fillna(0).values
                
                # Remove infinite values
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Remove rows with invalid data
                valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X = X[valid_mask]
                y = y[valid_mask]
                
                if len(X) > 0:
                    training_data[ticker] = (X, y, valid_cols)
                    logger.info(f"✅ Prepared {len(X)} samples with {len(valid_cols)} features for {ticker}")
                else:
                    logger.warning(f"No valid training data for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error preparing training data for {ticker}: {e}")
                continue
        
        return training_data
    
    def select_features(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Select best features using simple methods.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            
        Returns:
            Selected features and names
        """
        # Simple feature selection - use top features by variance
        feature_vars = np.var(X, axis=0)
        top_features = np.argsort(feature_vars)[-self.config.n_features:]
        
        # Select features
        X_selected = X[:, top_features]
        feature_names_selected = [feature_names[i] for i in top_features]
        
        return X_selected, feature_names_selected
    
    def train_random_forest(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        ticker: str
    ) -> ModelResults:
        """Train Random Forest model."""
        logger.info(f"Training Random Forest for {ticker}")
        
        start_time = datetime.now()
        
        # Feature selection
        X_selected, feature_names = self.select_features(X, y, [f'feature_{i}' for i in range(X.shape[1])])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=self.config.test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate scores
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store model
        self.models[f'{ticker}_rf'] = model
        
        return ModelResults(
            model_name='RandomForest',
            ticker=ticker,
            train_score=train_score,
            test_score=test_score,
            mse=mse,
            r2=r2,
            training_time=training_time
        )
    
    def train_all_models(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, ModelResults]]:
        """
        Train all models for all tickers.
        
        Args:
            data_dict: Dictionary with data for each ticker
            
        Returns:
            Dictionary with training results for each ticker and model
        """
        logger.info("🚀 Starting simple training pipeline for 5 tickers")
        
        # Prepare training data
        training_data = self.prepare_training_data(data_dict)
        
        results = {}
        
        for ticker in self.tickers:
            if ticker not in training_data:
                logger.warning(f"No training data for {ticker}")
                continue
            
            logger.info(f"Training models for {ticker}")
            ticker_results = {}
            
            try:
                X, y, feature_names = training_data[ticker]
                
                # Train Random Forest
                rf_results = self.train_random_forest(X, y, ticker)
                ticker_results['RandomForest'] = rf_results
                
                results[ticker] = ticker_results
                logger.info(f"✅ Training completed for {ticker}")
                
            except Exception as e:
                logger.error(f"Error training models for {ticker}: {e}")
                continue
        
        self.results = results
        return results
    
    def evaluate_models(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate all trained models."""
        evaluation_results = {}
        
        for ticker, ticker_results in self.results.items():
            ticker_evaluation = {}
            
            for model_name, results in ticker_results.items():
                evaluation = {
                    'model_name': results.model_name,
                    'ticker': results.ticker,
                    'train_score': results.train_score,
                    'test_score': results.test_score,
                    'mse': results.mse,
                    'r2': results.r2,
                    'training_time': results.training_time
                }
                
                ticker_evaluation[model_name] = evaluation
            
            evaluation_results[ticker] = ticker_evaluation
        
        return evaluation_results
    
    def get_best_models(self) -> Dict[str, str]:
        """Get best model for each ticker based on test score."""
        best_models = {}
        
        for ticker, ticker_results in self.results.items():
            best_score = -np.inf
            best_model = None
            
            for model_name, result in ticker_results.items():
                if result.test_score > best_score:
                    best_score = result.test_score
                    best_model = model_name
            
            best_models[ticker] = best_model
        
        return best_models


# Global instance for easy access
simple_training_pipeline = SimpleTrainingPipeline()
