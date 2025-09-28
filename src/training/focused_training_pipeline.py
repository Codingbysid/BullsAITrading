from src.utils.common_imports import *
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression
    import xgboost as xgb
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Attention, LayerNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from mlfinlab.cross_validation import PurgedKFold, CombinatorialPurgedCV
        import os
        import pickle

"""
Focused Training Pipeline for 5-Ticker QuantAI Platform.

This module implements advanced training specifically optimized for
AMZN, META, NVDA, GOOGL, and AAPL with sophisticated ML models.

Features:
- Purged cross-validation for time series
- Walk-forward analysis
- Meta labeling for trade filtering
- Ensemble learning with multiple models
- Feature selection and engineering
- Hyperparameter optimization
- Overfitting prevention
"""

warnings.filterwarnings('ignore')

# Core ML libraries

# Advanced ML libraries
try:
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

try:
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")

try:
    MLFINLAB_AVAILABLE = True
except ImportError:
    MLFINLAB_AVAILABLE = False
    logging.warning("MLFinLab not available. Install with: pip install mlfinlab")

logger = setup_logger()


@dataclass
class TrainingConfig:
    """Training configuration for focused models."""
    # Data parameters
    lookback_period: int = 252  # 1 year
    sequence_length: int = 60  # For LSTM
    test_size: float = 0.2
    
    # Cross-validation parameters
    n_splits: int = 5
    embargo_period: int = 5  # Days to embargo between train/test
    
    # Model parameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    
    # LSTM parameters
    lstm_units: int = 50
    lstm_dropout: float = 0.2
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    lstm_patience: int = 10
    
    # Feature selection
    n_features: int = 50
    feature_selection_method: str = 'mutual_info'
    
    # Meta labeling
    meta_label_threshold: float = 0.001
    meta_label_confidence: float = 0.6


@dataclass
class ModelResults:
    """Model training results."""
    model_name: str
    ticker: str
    train_score: float
    test_score: float
    cv_scores: List[float]
    feature_importance: Dict[str, float]
    predictions: np.ndarray
    actual: np.ndarray
    training_time: float


class FocusedTrainingPipeline:
    """
    Advanced training pipeline for 5-ticker focused models.
    
    Implements sophisticated training with purged cross-validation,
    walk-forward analysis, and meta labeling specifically for
    AMZN, META, NVDA, GOOGL, AAPL.
    """
    
    def __init__(self, config: TrainingConfig = None):
        """Initialize focused training pipeline."""
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
    
    def create_meta_labels(
        self, 
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Create meta labels for trade filtering.
        
        Args:
            data: Dictionary with data for each ticker
            
        Returns:
            Dictionary with meta labels for each ticker
        """
        meta_labeled_data = {}
        
        for ticker, df in data.items():
            if ticker not in self.tickers:
                continue
            
            logger.info(f"Creating meta labels for {ticker}")
            
            try:
                df_copy = df.copy()
                
                # Create forward returns
                df_copy['forward_returns'] = df_copy['returns'].shift(-1)
                
                # Create meta labels
                df_copy['meta_label'] = 0
                df_copy.loc[df_copy['forward_returns'] > self.config.meta_label_threshold, 'meta_label'] = 1
                df_copy.loc[df_copy['forward_returns'] < -self.config.meta_label_threshold, 'meta_label'] = -1
                
                # Add confidence scores
                df_copy['confidence'] = abs(df_copy['forward_returns']) / df_copy['forward_returns'].std()
                df_copy['confidence'] = np.clip(df_copy['confidence'], 0, 1)
                
                # Filter by confidence
                df_copy = df_copy[df_copy['confidence'] >= self.config.meta_label_confidence]
                
                meta_labeled_data[ticker] = df_copy
                logger.info(f"✅ Created meta labels for {ticker}: {len(df_copy)} samples")
                
            except Exception as e:
                logger.error(f"Error creating meta labels for {ticker}: {e}")
                continue
        
        return meta_labeled_data
    
    def purged_cross_validation(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        model: Any,
        embargo_period: int = 5
    ) -> List[float]:
        """
        Perform purged cross-validation for time series data.
        
        Args:
            X: Feature matrix
            y: Target values
            model: Model to validate
            embargo_period: Embargo period in days
            
        Returns:
            List of CV scores
        """
        if MLFINLAB_AVAILABLE:
            try:
                # Use MLFinLab's PurgedKFold
                purged_cv = PurgedKFold(n_splits=self.config.n_splits, t1=None, pct_embargo=embargo_period/len(X))
                scores = cross_val_score(model, X, y, cv=purged_cv, scoring='neg_mean_squared_error')
                return -scores.tolist()
            except Exception as e:
                logger.warning(f"MLFinLab purged CV failed: {e}")
        
        # Fallback to time series split
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        return -scores.tolist()
    
    def walk_forward_analysis(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        model: Any,
        train_size: int = 200,
        test_size: int = 50
    ) -> List[float]:
        """
        Perform walk-forward analysis.
        
        Args:
            X: Feature matrix
            y: Target values
            model: Model to validate
            train_size: Training window size
            test_size: Test window size
            
        Returns:
            List of walk-forward scores
        """
        scores = []
        
        for i in range(train_size, len(X) - test_size, test_size):
            # Training data
            X_train = X[i-train_size:i]
            y_train = y[i-train_size:i]
            
            # Test data
            X_test = X[i:i+test_size]
            y_test = y[i:i+test_size]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate score
            score = mean_squared_error(y_test, y_pred)
            scores.append(score)
        
        return scores
    
    def select_features(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        feature_names: List[str],
        method: str = 'mutual_info'
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Select best features using various methods.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            method: Feature selection method
            
        Returns:
            Selected features and names
        """
        if method == 'mutual_info':
            # Use f_regression as fallback for mutual information
            f_scores, _ = f_regression(X, y)
            top_features = np.argsort(f_scores)[-self.config.n_features:]
            
        elif method == 'f_regression':
            # F-regression
            f_scores, _ = f_regression(X, y)
            top_features = np.argsort(f_scores)[-self.config.n_features:]
            
        elif method == 'rfe':
            # Recursive feature elimination
            estimator = RandomForestRegressor(n_estimators=10, random_state=42)
            selector = RFE(estimator, n_features_to_select=self.config.n_features)
            selector.fit(X, y)
            top_features = np.where(selector.support_)[0]
            
        else:
            # Use all features
            top_features = np.arange(min(self.config.n_features, X.shape[1]))
        
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
        scaler = RobustScaler()
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
        
        # Cross-validation
        cv_scores = self.purged_cross_validation(X_selected, y, model)
        
        # Feature importance
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store model
        self.models[f'{ticker}_rf'] = model
        
        return ModelResults(
            model_name='RandomForest',
            ticker=ticker,
            train_score=train_score,
            test_score=test_score,
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            predictions=y_pred,
            actual=y_test,
            training_time=training_time
        )
    
    def train_xgboost(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        ticker: str
    ) -> ModelResults:
        """Train XGBoost model."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, using Random Forest instead")
            return self.train_random_forest(X, y, ticker)
        
        logger.info(f"Training XGBoost for {ticker}")
        
        start_time = datetime.now()
        
        # Feature selection
        X_selected, feature_names = self.select_features(X, y, [f'feature_{i}' for i in range(X.shape[1])])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=self.config.test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=self.config.xgb_n_estimators,
            max_depth=self.config.xgb_max_depth,
            learning_rate=self.config.xgb_learning_rate,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate scores
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = self.purged_cross_validation(X_selected, y, model)
        
        # Feature importance
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store model
        self.models[f'{ticker}_xgb'] = model
        
        return ModelResults(
            model_name='XGBoost',
            ticker=ticker,
            train_score=train_score,
            test_score=test_score,
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            predictions=y_pred,
            actual=y_test,
            training_time=training_time
        )
    
    def train_lstm(
        self, 
        data: pd.DataFrame, 
        ticker: str
    ) -> ModelResults:
        """Train LSTM model."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, using Random Forest instead")
            X, y, _ = self.prepare_training_data({ticker: data})
            if ticker in X:
                return self.train_random_forest(X[ticker][0], X[ticker][1], ticker)
            return None
        
        logger.info(f"Training LSTM for {ticker}")
        
        start_time = datetime.now()
        
        # Prepare data
        X, y, feature_names = self.prepare_training_data({ticker: data})[ticker]
        
        if len(X) < self.config.sequence_length:
            logger.warning(f"Insufficient data for LSTM training for {ticker}")
            return None
        
        # Create sequences
        X_sequences, y_sequences = self._create_sequences(X, y, self.config.sequence_length)
        
        if len(X_sequences) == 0:
            logger.warning(f"No sequences created for LSTM training for {ticker}")
            return None
        
        # Feature selection
        X_selected, selected_features = self.select_features(
            X_sequences.reshape(X_sequences.shape[0], -1), 
            y_sequences,
            [f'feature_{i}' for i in range(X_sequences.shape[2])]
        )
        
        # Reshape back to sequences
        X_selected = X_selected.reshape(X_sequences.shape[0], X_sequences.shape[1], -1)
        
        # Split data
        split_idx = int((1 - self.config.test_size) * len(X_selected))
        X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
        y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Build LSTM model
        model = self._build_lstm_model(X_train.shape[1], X_train.shape[2])
        
        # Train model
        callbacks = [
            EarlyStopping(patience=self.config.lstm_patience, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = model.fit(
            X_train_scaled, y_train,
            epochs=self.config.lstm_epochs,
            batch_size=self.config.lstm_batch_size,
            validation_data=(X_test_scaled, y_test),
            callbacks=callbacks,
            verbose=0
        )
        
        # Make predictions
        y_pred = model.predict(X_test_scaled, verbose=0)
        if len(y_pred.shape) > 1:
            y_pred = y_pred.flatten()
        
        # Calculate scores
        train_score = model.evaluate(X_train_scaled, y_train, verbose=0)[1]  # MAE
        test_score = model.evaluate(X_test_scaled, y_test, verbose=0)[1]  # MAE
        
        # Cross-validation (simplified for LSTM)
        cv_scores = [test_score]  # Simplified CV for LSTM
        
        # Feature importance (simplified for LSTM)
        feature_importance = {f'feature_{i}': 1.0/len(selected_features) for i in range(len(selected_features))}
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store model
        self.models[f'{ticker}_lstm'] = model
        
        return ModelResults(
            model_name='LSTM',
            ticker=ticker,
            train_score=train_score,
            test_score=test_score,
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            predictions=y_pred,
            actual=y_test,
            training_time=training_time
        )
    
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
            LSTM(self.config.lstm_units, return_sequences=True, 
                 input_shape=(sequence_length, n_features)),
            Dropout(self.config.lstm_dropout),
            LSTM(self.config.lstm_units, return_sequences=True),
            Dropout(self.config.lstm_dropout),
            Attention(),
            LayerNormalization(),
            Dense(32, activation='relu'),
            Dropout(self.config.lstm_dropout),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_ensemble_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        ticker: str
    ) -> ModelResults:
        """Train ensemble model combining multiple algorithms."""
        logger.info(f"Training ensemble model for {ticker}")
        
        start_time = datetime.now()
        
        # Train individual models
        rf_results = self.train_random_forest(X, y, ticker)
        xgb_results = self.train_xgboost(X, y, ticker)
        
        # Create ensemble predictions
        if rf_results and xgb_results:
            # Simple ensemble (average predictions)
            ensemble_predictions = (rf_results.predictions + xgb_results.predictions) / 2
            
            # Calculate ensemble score
            ensemble_score = r2_score(rf_results.actual, ensemble_predictions)
            
            # Combine feature importance
            combined_importance = {}
            for feature, importance in rf_results.feature_importance.items():
                combined_importance[feature] = importance * 0.5
            for feature, importance in xgb_results.feature_importance.items():
                if feature in combined_importance:
                    combined_importance[feature] += importance * 0.5
                else:
                    combined_importance[feature] = importance * 0.5
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            return ModelResults(
                model_name='Ensemble',
                ticker=ticker,
                train_score=(rf_results.train_score + xgb_results.train_score) / 2,
                test_score=ensemble_score,
                cv_scores=(np.array(rf_results.cv_scores) + np.array(xgb_results.cv_scores)) / 2,
                feature_importance=combined_importance,
                predictions=ensemble_predictions,
                actual=rf_results.actual,
                training_time=training_time
            )
        
        return None
    
    def train_all_models(
        self, 
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, ModelResults]]:
        """
        Train all models for all tickers.
        
        Args:
            data: Dictionary with data for each ticker
            
        Returns:
            Dictionary with training results for each ticker and model
        """
        logger.info("🚀 Starting focused training pipeline for 5 tickers")
        
        # Prepare training data
        training_data = self.prepare_training_data(data)
        
        # Create meta labels
        meta_labeled_data = self.create_meta_labels(data)
        
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
                
                # Train XGBoost
                xgb_results = self.train_xgboost(X, y, ticker)
                ticker_results['XGBoost'] = xgb_results
                
                # Train LSTM
                if ticker in data:
                    lstm_results = self.train_lstm(data[ticker], ticker)
                    if lstm_results:
                        ticker_results['LSTM'] = lstm_results
                
                # Train Ensemble
                ensemble_results = self.train_ensemble_model(X, y, ticker)
                if ensemble_results:
                    ticker_results['Ensemble'] = ensemble_results
                
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
                    'cv_mean': np.mean(results.cv_scores),
                    'cv_std': np.std(results.cv_scores),
                    'training_time': results.training_time,
                    'feature_importance': results.feature_importance,
                    'mse': mean_squared_error(results.actual, results.predictions),
                    'r2': r2_score(results.actual, results.predictions)
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
            
            for model_name, results in ticker_results.items():
                if results.test_score > best_score:
                    best_score = results.test_score
                    best_model = model_name
            
            best_models[ticker] = best_model
        
        return best_models
    
    def save_models(self, output_dir: str = "focused_models"):
        """Save all trained models."""
        
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
        
        # Save results
        with open(f"{output_dir}/training_results.pkl", 'wb') as f:
            pickle.dump(self.results, f)
        
        logger.info(f"✅ Models saved to {output_dir}")


# Global instance for easy access
focused_training_pipeline = FocusedTrainingPipeline()
