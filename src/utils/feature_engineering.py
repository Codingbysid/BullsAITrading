"""
Unified feature engineering utilities for the QuantAI Trading Platform.

This module provides standardized feature engineering functions to eliminate
duplication across the codebase.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from .common_imports import setup_logger, validate_dataframe, validate_series
from .data_processing import data_processor

# Optional sklearn imports
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available - some ML features will be disabled")

logger = setup_logger(__name__)


class FeatureEngineer:
    """Unified feature engineering class."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_importance = {}
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive technical features."""
        if not data_processor.validate_price_data(data):
            logger.error("Invalid price data for feature engineering")
            return data
        
        df = data.copy()
        
        # Price-based features
        df['price_change'] = df['Close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        df['high_close_ratio'] = df['High'] / df['Close']
        df['low_close_ratio'] = df['Low'] / df['Close']
        
        # Volume-based features
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_price_ratio'] = df['Volume'] / df['Close']
        df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Price momentum features
        for period in [1, 2, 3, 5, 10, 20, 50]:
            df[f'price_momentum_{period}'] = df['Close'].pct_change(period)
            df[f'price_ma_ratio_{period}'] = df['Close'] / df['Close'].rolling(period).mean()
        
        # Volatility features
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['Close'].pct_change().rolling(period).std()
            df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df[f'volatility_{period}'].rolling(20).mean()
        
        # Moving average features
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'sma_ratio_{period}'] = df['Close'] / df[f'sma_{period}']
            df[f'ema_ratio_{period}'] = df['Close'] / df[f'ema_{period}']
        
        # Bollinger Bands features
        for period in [20, 50]:
            sma = df['Close'].rolling(period).mean()
            std = df['Close'].rolling(period).std()
            df[f'bb_upper_{period}'] = sma + (std * 2)
            df[f'bb_lower_{period}'] = sma - (std * 2)
            df[f'bb_position_{period}'] = (df['Close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
        
        # RSI features
        for period in [14, 21, 50]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            df[f'rsi_signal_{period}'] = np.where(df[f'rsi_{period}'] > 70, 1, np.where(df[f'rsi_{period}'] < 30, -1, 0))
        
        # MACD features
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_signal_line'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        # Stochastic features
        for period in [14, 21]:
            low_min = df['Low'].rolling(period).min()
            high_max = df['High'].rolling(period).max()
            df[f'stoch_k_{period}'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()
            df[f'stoch_signal_{period}'] = np.where(df[f'stoch_k_{period}'] > 80, 1, np.where(df[f'stoch_k_{period}'] < 20, -1, 0))
        
        # Williams %R features
        for period in [14, 21]:
            high_max = df['High'].rolling(period).max()
            low_min = df['Low'].rolling(period).min()
            df[f'williams_r_{period}'] = -100 * (high_max - df['Close']) / (high_max - low_min)
            df[f'williams_r_signal_{period}'] = np.where(df[f'williams_r_{period}'] > -20, 1, np.where(df[f'williams_r_{period}'] < -80, -1, 0))
        
        # ATR features
        for period in [14, 21]:
            df[f'atr_{period}'] = self._calculate_atr(df, period)
            df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['Close']
        
        # ADX features
        for period in [14, 21]:
            df[f'adx_{period}'] = self._calculate_adx(df, period)
            df[f'adx_signal_{period}'] = np.where(df[f'adx_{period}'] > 25, 1, 0)
        
        # CCI features
        for period in [20, 50]:
            df[f'cci_{period}'] = self._calculate_cci(df, period)
            df[f'cci_signal_{period}'] = np.where(df[f'cci_{period}'] > 100, 1, np.where(df[f'cci_{period}'] < -100, -1, 0))
        
        # Volume features
        df['obv'] = (df['Volume'] * np.where(df['Close'] > df['Close'].shift(1), 1, -1)).cumsum()
        df['obv_ma'] = df['obv'].rolling(20).mean()
        df['obv_ratio'] = df['obv'] / df['obv_ma']
        
        # Price patterns
        df['doji'] = np.where(abs(df['Open'] - df['Close']) / (df['High'] - df['Low']) < 0.1, 1, 0)
        df['hammer'] = np.where((df['Close'] > df['Open']) & 
                               ((df['Close'] - df['Low']) > 2 * (df['Open'] - df['Low'])) & 
                               ((df['High'] - df['Close']) < 0.3 * (df['Close'] - df['Low'])), 1, 0)
        df['shooting_star'] = np.where((df['Open'] > df['Close']) & 
                                     ((df['High'] - df['Open']) > 2 * (df['Open'] - df['Close'])) & 
                                     ((df['Open'] - df['Low']) < 0.3 * (df['High'] - df['Open'])), 1, 0)
        
        # Gap features
        df['gap_up'] = np.where(df['Open'] > df['High'].shift(1), 1, 0)
        df['gap_down'] = np.where(df['Open'] < df['Low'].shift(1), 1, 0)
        df['gap_size'] = np.where(df['gap_up'], df['Open'] - df['High'].shift(1), 
                                 np.where(df['gap_down'], df['Low'].shift(1) - df['Open'], 0))
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        # Calculate directional movement
        high_diff = df['High'].diff()
        low_diff = df['Low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Calculate smoothed averages
        atr = self._calculate_atr(df, period)
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=period).mean()
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        return (typical_price - sma_tp) / (0.015 * mad)
    
    def create_lag_features(self, data: pd.DataFrame, columns: List[str], 
                           lags: List[int]) -> pd.DataFrame:
        """Create lag features for specified columns."""
        df = data.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, data: pd.DataFrame, columns: List[str], 
                               windows: List[int], functions: List[str]) -> pd.DataFrame:
        """Create rolling features for specified columns."""
        df = data.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    for func in functions:
                        if func == 'mean':
                            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                        elif func == 'std':
                            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                        elif func == 'min':
                            df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                        elif func == 'max':
                            df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
                        elif func == 'skew':
                            df[f'{col}_rolling_skew_{window}'] = df[col].rolling(window).skew()
                        elif func == 'kurt':
                            df[f'{col}_rolling_kurt_{window}'] = df[col].rolling(window).kurt()
        
        return df
    
    def create_interaction_features(self, data: pd.DataFrame, 
                                   feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create interaction features between pairs of features."""
        df = data.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-8)  # Add small value to avoid division by zero
                df[f'{feat1}_plus_{feat2}'] = df[feat1] + df[feat2]
                df[f'{feat1}_minus_{feat2}'] = df[feat1] - df[feat2]
        
        return df
    
    def create_polynomial_features(self, data: pd.DataFrame, columns: List[str], 
                                  degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for specified columns."""
        df = data.copy()
        
        for col in columns:
            if col in df.columns:
                for d in range(2, degree + 1):
                    df[f'{col}_poly_{d}'] = df[col] ** d
        
        return df
    
    def create_time_features(self, data: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
        """Create time-based features."""
        df = data.copy()
        
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Extract time components
            df['year'] = df[date_column].dt.year
            df['month'] = df[date_column].dt.month
            df['day'] = df[date_column].dt.day
            df['dayofweek'] = df[date_column].dt.dayofweek
            df['dayofyear'] = df[date_column].dt.dayofyear
            df['quarter'] = df[date_column].dt.quarter
            df['is_month_end'] = df[date_column].dt.is_month_end.astype(int)
            df['is_quarter_end'] = df[date_column].dt.is_quarter_end.astype(int)
            df['is_year_end'] = df[date_column].dt.is_year_end.astype(int)
            
            # Cyclical encoding
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
            df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
        
        return df
    
    def scale_features(self, data: pd.DataFrame, columns: List[str], 
                      method: str = 'standard', fit: bool = True) -> pd.DataFrame:
        """Scale features using specified method."""
        df = data.copy()
        
        if not SKLEARN_AVAILABLE:
            # Simple scaling without sklearn
            for col in columns:
                if col in df.columns:
                    if method == 'standard':
                        df[col] = (df[col] - df[col].mean()) / df[col].std()
                    elif method == 'minmax':
                        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                    elif method == 'robust':
                        median = df[col].median()
                        mad = np.median(np.abs(df[col] - median))
                        df[col] = (df[col] - median) / mad
            return df
        
        if method not in self.scalers:
            if method == 'standard':
                self.scalers[method] = StandardScaler()
            elif method == 'minmax':
                self.scalers[method] = MinMaxScaler()
            elif method == 'robust':
                self.scalers[method] = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
        
        scaler = self.scalers[method]
        
        for col in columns:
            if col in df.columns:
                if fit:
                    df[col] = scaler.fit_transform(df[[col]]).flatten()
                else:
                    df[col] = scaler.transform(df[[col]]).flatten()
        
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       method: str = 'f_regression', k: int = 20) -> pd.DataFrame:
        """Select top k features using specified method."""
        if not SKLEARN_AVAILABLE:
            # Simple feature selection without sklearn
            if method == 'variance':
                # Select features with highest variance
                variances = X.var()
                selected_features = variances.nlargest(k).index.tolist()
                return X[selected_features]
            else:
                # Return first k features
                return X.iloc[:, :k]
        
        if method not in self.feature_selectors:
            if method == 'f_regression':
                self.feature_selectors[method] = SelectKBest(score_func=f_regression, k=k)
            elif method == 'mutual_info':
                self.feature_selectors[method] = SelectKBest(score_func=mutual_info_regression, k=k)
            else:
                raise ValueError(f"Unknown feature selection method: {method}")
        
        selector = self.feature_selectors[method]
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store feature importance
        self.feature_importance[method] = dict(zip(selected_features, selector.scores_))
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def create_target_variable(self, data: pd.DataFrame, target_column: str = 'Close',
                              method: str = 'future_return', periods: int = 1) -> pd.Series:
        """Create target variable for ML models."""
        if method == 'future_return':
            return data[target_column].shift(-periods) / data[target_column] - 1
        elif method == 'future_price':
            return data[target_column].shift(-periods)
        elif method == 'direction':
            future_return = data[target_column].shift(-periods) / data[target_column] - 1
            return np.where(future_return > 0, 1, 0)
        else:
            raise ValueError(f"Unknown target method: {method}")
    
    def prepare_ml_data(self, data: pd.DataFrame, target_column: str = 'Close',
                       feature_columns: Optional[List[str]] = None,
                       target_method: str = 'future_return',
                       target_periods: int = 1,
                       lookback_period: int = 20) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for ML model training."""
        # Create comprehensive features
        df = self.create_technical_features(data)
        df = self.create_time_features(df)
        
        # Create lag features
        price_columns = ['Close', 'Volume', 'price_change', 'volatility_20']
        df = self.create_lag_features(df, price_columns, [1, 2, 3, 5, 10])
        
        # Create rolling features
        df = self.create_rolling_features(df, ['Close', 'Volume'], [5, 10, 20], ['mean', 'std'])
        
        # Create interaction features
        interaction_pairs = [('Close', 'Volume'), ('price_change', 'volatility_20'), ('rsi_14', 'macd')]
        df = self.create_interaction_features(df, interaction_pairs)
        
        # Select feature columns
        if feature_columns is None:
            # Exclude target and basic price columns
            exclude_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
            feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Create target variable
        target = self.create_target_variable(df, target_column, target_method, target_periods)
        
        # Remove rows with NaN values
        df = df.dropna()
        target = target.dropna()
        
        # Align features and target
        common_index = df.index.intersection(target.index)
        X = df.loc[common_index, feature_columns]
        y = target.loc[common_index]
        
        return X, y
    
    def get_feature_importance(self, method: str = 'f_regression') -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance.get(method, {})
    
    def get_feature_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of all features in the dataset."""
        summary = {
            'total_features': len(data.columns),
            'numeric_features': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(data.select_dtypes(include=['object']).columns),
            'missing_values': data.isnull().sum().to_dict(),
            'feature_types': data.dtypes.to_dict(),
            'feature_stats': data.describe().to_dict()
        }
        
        return summary


# Global feature engineer instance
feature_engineer = FeatureEngineer()


# Convenience functions
def create_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive technical features."""
    return feature_engineer.create_technical_features(data)

def create_lag_features(data: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """Create lag features for specified columns."""
    return feature_engineer.create_lag_features(data, columns, lags)

def create_rolling_features(data: pd.DataFrame, columns: List[str], 
                           windows: List[int], functions: List[str]) -> pd.DataFrame:
    """Create rolling features for specified columns."""
    return feature_engineer.create_rolling_features(data, columns, windows, functions)

def create_interaction_features(data: pd.DataFrame, 
                               feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """Create interaction features between pairs of features."""
    return feature_engineer.create_interaction_features(data, feature_pairs)

def create_time_features(data: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
    """Create time-based features."""
    return feature_engineer.create_time_features(data, date_column)

def scale_features(data: pd.DataFrame, columns: List[str], 
                  method: str = 'standard', fit: bool = True) -> pd.DataFrame:
    """Scale features using specified method."""
    return feature_engineer.scale_features(data, columns, method, fit)

def select_features(X: pd.DataFrame, y: pd.Series, 
                   method: str = 'f_regression', k: int = 20) -> pd.DataFrame:
    """Select top k features using specified method."""
    return feature_engineer.select_features(X, y, method, k)

def prepare_ml_data(data: pd.DataFrame, target_column: str = 'Close',
                   feature_columns: Optional[List[str]] = None,
                   target_method: str = 'future_return',
                   target_periods: int = 1,
                   lookback_period: int = 20) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for ML model training."""
    return feature_engineer.prepare_ml_data(data, target_column, feature_columns, 
                                           target_method, target_periods, lookback_period)
