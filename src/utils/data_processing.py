"""
Unified data processing utilities for the QuantAI Trading Platform.

This module provides standardized data processing functions to eliminate
duplication across the codebase.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from .common_imports import setup_logger, validate_dataframe, validate_series

logger = setup_logger(__name__)


class DataProcessor:
    """Unified data processing class."""
    
    def __init__(self):
        self.required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def validate_price_data(self, data: pd.DataFrame) -> bool:
        """Validate price data structure."""
        if not validate_dataframe(data, self.required_columns):
            logger.error(f"Missing required columns: {self.required_columns}")
            return False
        
        # Check for negative prices
        if (data[self.required_columns[:4]] < 0).any().any():
            logger.error("Negative prices found in data")
            return False
        
        # Check for invalid OHLC relationships
        if not (data['High'] >= data['Low']).all():
            logger.error("High prices less than Low prices found")
            return False
        
        if not (data['High'] >= data['Open']).all():
            logger.error("High prices less than Open prices found")
            return False
        
        if not (data['High'] >= data['Close']).all():
            logger.error("High prices less than Close prices found")
            return False
        
        if not (data['Low'] <= data['Open']).all():
            logger.error("Low prices greater than Open prices found")
            return False
        
        if not (data['Low'] <= data['Close']).all():
            logger.error("Low prices greater than Close prices found")
            return False
        
        return True
    
    def clean_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean price data by removing invalid rows."""
        df = data.copy()
        
        # Remove rows with missing values
        df = df.dropna(subset=self.required_columns)
        
        # Remove rows with negative prices
        for col in self.required_columns[:4]:
            df = df[df[col] > 0]
        
        # Fix OHLC relationships
        df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'High', 'Low', 'Close']].min(axis=1)
        
        # Remove rows with zero volume
        df = df[df['Volume'] > 0]
        
        return df
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data."""
        if not self.validate_price_data(data):
            logger.error("Invalid price data for technical indicators")
            return data
        
        df = data.copy()
        
        # Moving averages
        df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['Close'].rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # Stochastic Oscillator
        df['Stoch_K'] = 100 * (df['Close'] - df['Low'].rolling(window=14).min()) / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Williams %R
        df['Williams_R'] = -100 * (df['High'].rolling(window=14).max() - df['Close']) / (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['OBV'] = (df['Volume'] * np.where(df['Close'] > df['Close'].shift(1), 1, -1)).cumsum()
        
        # Price momentum
        df['Price_Change_1D'] = df['Close'].pct_change(1)
        df['Price_Change_5D'] = df['Close'].pct_change(5)
        df['Price_Change_20D'] = df['Close'].pct_change(20)
        df['Price_Change_50D'] = df['Close'].pct_change(50)
        
        # Volatility indicators
        df['Volatility_20D'] = df['Close'].pct_change().rolling(window=20, min_periods=1).std()
        df['ATR'] = self._calculate_atr(df)
        
        # Trend indicators
        df['ADX'] = self._calculate_adx(df)
        df['CCI'] = self._calculate_cci(df)
        
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
    
    def resample_data(self, data: pd.DataFrame, frequency: str = 'D') -> pd.DataFrame:
        """Resample data to specified frequency."""
        if 'Date' in data.columns:
            data = data.set_index('Date')
        
        resampled = data.resample(frequency).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return resampled.reset_index()
    
    def calculate_returns(self, prices: pd.Series, method: str = 'simple') -> pd.Series:
        """Calculate returns from price series."""
        if method == 'simple':
            return prices.pct_change().dropna()
        elif method == 'log':
            return np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValueError(f"Unknown return method: {method}")
    
    def calculate_rolling_metrics(self, data: pd.Series, window: int = 20) -> pd.DataFrame:
        """Calculate rolling metrics for a data series."""
        return pd.DataFrame({
            'mean': data.rolling(window=window).mean(),
            'std': data.rolling(window=window).std(),
            'min': data.rolling(window=window).min(),
            'max': data.rolling(window=window).max(),
            'skew': data.rolling(window=window).skew(),
            'kurt': data.rolling(window=window).kurt()
        })
    
    def detect_outliers(self, data: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """Detect outliers in data series."""
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            return z_scores > threshold
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def remove_outliers(self, data: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from specified columns."""
        df = data.copy()
        
        for col in columns:
            if col in df.columns:
                outliers = self.detect_outliers(df[col], method, threshold)
                df = df[~outliers]
        
        return df
    
    def create_features(self, data: pd.DataFrame, target_column: str = 'Close') -> pd.DataFrame:
        """Create feature engineering for ML models."""
        df = data.copy()
        
        # Price-based features
        df['price_change'] = df[target_column].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Volume-based features
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_price_ratio'] = df['Volume'] / df[target_column]
        
        # Technical indicator features
        df = self.add_technical_indicators(df)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'price_lag_{lag}'] = df[target_column].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'price_mean_{window}'] = df[target_column].rolling(window=window).mean()
            df[f'price_std_{window}'] = df[target_column].rolling(window=window).std()
            df[f'volume_mean_{window}'] = df['Volume'].rolling(window=window).mean()
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def prepare_ml_data(self, data: pd.DataFrame, target_column: str = 'Close',
                       feature_columns: Optional[List[str]] = None,
                       lookback_period: int = 20) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for ML model training."""
        # Create features
        df = self.create_features(data, target_column)
        
        # Select feature columns
        if feature_columns is None:
            # Exclude target and basic price columns
            exclude_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']
            feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Create target variable (future returns)
        df['target'] = df[target_column].shift(-1) / df[target_column] - 1
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Select features and target
        X = df[feature_columns]
        y = df['target']
        
        return X, y
    
    def create_synthetic_data(self, symbols: List[str], start_date: datetime, 
                            end_date: datetime, base_price: float = 100.0) -> pd.DataFrame:
        """Create synthetic market data for testing."""
        # Generate trading days (exclude weekends)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = [d for d in dates if d.weekday() < 5]
        
        all_data = []
        
        for symbol in symbols:
            # Generate realistic price data for each symbol
            np.random.seed(42 + hash(symbol) % 1000)
            
            # Symbol-specific parameters
            symbol_base_price = base_price + hash(symbol) % 100
            volatility = 0.02 + (hash(symbol) % 10) * 0.005
            trend = (hash(symbol) % 20 - 10) * 0.0001
            
            prices = [symbol_base_price]
            volumes = []
            
            for i, date in enumerate(trading_days):
                # Generate price with trend and volatility
                daily_return = np.random.normal(trend, volatility)
                new_price = prices[-1] * (1 + daily_return)
                prices.append(new_price)
                
                # Generate volume
                base_volume = 1000000 + hash(symbol) % 500000
                volume_multiplier = 1 + abs(daily_return) * 5
                volume = int(base_volume * volume_multiplier * (0.8 + 0.4 * np.random.random()))
                volumes.append(volume)
            
            # Remove the extra price
            prices = prices[1:]
            
            # Create OHLCV data
            for i, (date, price, volume) in enumerate(zip(trading_days, prices, volumes)):
                daily_range = price * volatility * np.random.uniform(0.5, 2.0)
                
                open_price = price * (1 + np.random.normal(0, volatility * 0.3))
                high_price = max(open_price, price) + daily_range * np.random.uniform(0, 0.5)
                low_price = min(open_price, price) - daily_range * np.random.uniform(0, 0.5)
                close_price = price
                
                all_data.append({
                    'Date': date,
                    'Symbol': symbol,
                    'Open': round(open_price, 2),
                    'High': round(high_price, 2),
                    'Low': round(low_price, 2),
                    'Close': round(close_price, 2),
                    'Volume': volume
                })
        
        df = pd.DataFrame(all_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Date', 'Symbol']).reset_index(drop=True)
        
        return df


# Global instance for easy access
data_processor = DataProcessor()


# Convenience functions
def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to price data."""
    return data_processor.add_technical_indicators(data)

def validate_price_data(data: pd.DataFrame) -> bool:
    """Validate price data structure."""
    return data_processor.validate_price_data(data)

def clean_price_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean price data by removing invalid rows."""
    return data_processor.clean_price_data(data)

def create_synthetic_data(symbols: List[str], start_date: datetime, 
                         end_date: datetime, base_price: float = 100.0) -> pd.DataFrame:
    """Create synthetic market data for testing."""
    return data_processor.create_synthetic_data(symbols, start_date, end_date, base_price)
