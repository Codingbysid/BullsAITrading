"""
Scientific backtesting engine with purged cross-validation.

This module implements rigorous backtesting methodology including:
- Purged Cross-Validation to prevent data leakage
- Combinatorial Purged Cross-Validation (CPCV)
- Walk-forward analysis
- Comprehensive performance metrics
- Transaction cost modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """Results from backtesting."""
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    drawdown: pd.Series
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float


class PurgedCrossValidator(BaseCrossValidator):
    """
    Purged Cross-Validation to prevent data leakage in time series.
    
    This implementation ensures that training and test sets are properly
    separated in time, preventing look-ahead bias.
    """
    
    def __init__(self, n_splits: int = 5, purge_length: int = 1, embargo_length: int = 1):
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate train/test splits with purging."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate split sizes
        split_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Test set
            test_start = i * split_size
            test_end = min((i + 1) * split_size, n_samples)
            test_indices = indices[test_start:test_end]
            
            # Purge period (remove data before test set)
            purge_start = max(0, test_start - self.purge_length)
            purge_end = min(n_samples, test_end + self.embargo_length)
            purge_indices = indices[purge_start:purge_end]
            
            # Training set (exclude test and purge periods)
            train_indices = np.setdiff1d(indices, np.union1d(test_indices, purge_indices))
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X: pd.DataFrame = None, y: pd.Series = None, groups: pd.Series = None) -> int:
        """Return number of splits."""
        return self.n_splits


class CombinatorialPurgedCrossValidator(BaseCrossValidator):
    """
    Combinatorial Purged Cross-Validation (CPCV).
    
    More robust than standard purged CV, especially for financial time series.
    """
    
    def __init__(self, n_splits: int = 5, purge_length: int = 1, embargo_length: int = 1):
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate combinatorial train/test splits."""
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Create all possible combinations of train/test splits
        for i in range(self.n_splits):
            for j in range(i + 1, self.n_splits):
                # Test set
                test_start = i * (n_samples // self.n_splits)
                test_end = min((i + 1) * (n_samples // self.n_splits), n_samples)
                test_indices = indices[test_start:test_end]
                
                # Training set (exclude test period and purge/embargo)
                train_start = j * (n_samples // self.n_splits)
                train_end = min((j + 1) * (n_samples // self.n_splits), n_samples)
                train_indices = indices[train_start:train_end]
                
                # Apply purging and embargo
                purge_start = max(0, test_start - self.purge_length)
                purge_end = min(n_samples, test_end + self.embargo_length)
                purge_indices = indices[purge_start:purge_end]
                
                # Remove purged indices from training
                train_indices = np.setdiff1d(train_indices, purge_indices)
                
                if len(train_indices) > 0 and len(test_indices) > 0:
                    yield train_indices, test_indices
    
    def get_n_splits(self, X: pd.DataFrame = None, y: pd.Series = None, groups: pd.Series = None) -> int:
        """Return number of splits."""
        return self.n_splits * (self.n_splits - 1) // 2


class BacktestingEngine:
    """Scientific backtesting engine with comprehensive validation."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 0.2
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        
    def run_backtest(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        prices: pd.Series,
        cv_method: str = "purged"
    ) -> BacktestResults:
        """
        Run comprehensive backtest with cross-validation.
        
        Args:
            model: Trained ML model
            X: Feature matrix
            y: Target variable
            prices: Price series
            cv_method: Cross-validation method ("purged" or "combinatorial")
            
        Returns:
            BacktestResults object
        """
        if cv_method == "purged":
            cv = PurgedCrossValidator(n_splits=5)
        elif cv_method == "combinatorial":
            cv = CombinatorialPurgedCrossValidator(n_splits=5)
        else:
            raise ValueError("cv_method must be 'purged' or 'combinatorial'")
        
        # Store results for each fold
        all_returns = []
        all_positions = []
        all_trades = []
        
        for train_idx, test_idx in cv.split(X, y):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            prices_test = prices.iloc[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Generate predictions
            predictions = model.predict(X_test)
            
            # Run backtest for this fold
            fold_results = self._run_single_backtest(
                predictions, prices_test, y_test
            )
            
            all_returns.append(fold_results['returns'])
            all_positions.append(fold_results['positions'])
            all_trades.append(fold_results['trades'])
        
        # Combine results
        combined_returns = pd.concat(all_returns)
        combined_positions = pd.concat(all_positions)
        combined_trades = pd.concat(all_trades, ignore_index=True)
        
        # Calculate metrics
        metrics = self._calculate_metrics(combined_returns, combined_trades)
        
        return BacktestResults(
            returns=combined_returns,
            positions=combined_positions,
            trades=combined_trades,
            metrics=metrics,
            drawdown=self._calculate_drawdown(combined_returns),
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            calmar_ratio=metrics['calmar_ratio']
        )
    
    def _run_single_backtest(
        self,
        predictions: np.ndarray,
        prices: pd.Series,
        actual_returns: pd.Series
    ) -> Dict:
        """Run backtest for a single fold."""
        # Initialize portfolio
        capital = self.initial_capital
        position = 0.0
        positions = []
        returns = []
        trades = []
        
        for i, (pred, price, actual_ret) in enumerate(zip(predictions, prices, actual_returns)):
            # Calculate position size based on prediction
            if pred > 0.1:  # Buy signal
                target_position = min(self.max_position_size, pred)
            elif pred < -0.1:  # Sell signal
                target_position = max(-self.max_position_size, pred)
            else:  # Hold
                target_position = 0.0
            
            # Calculate position change
            position_change = target_position - position
            
            # Apply transaction costs
            if abs(position_change) > 0:
                cost = abs(position_change) * capital * self.transaction_cost
                capital -= cost
                
                # Record trade
                trades.append({
                    'timestamp': prices.index[i],
                    'price': price,
                    'position_change': position_change,
                    'cost': cost,
                    'prediction': pred,
                    'actual_return': actual_ret
                })
            
            # Update position
            position = target_position
            positions.append(position)
            
            # Calculate portfolio return
            portfolio_return = position * actual_ret
            capital *= (1 + portfolio_return)
            returns.append(portfolio_return)
        
        return {
            'returns': pd.Series(returns, index=prices.index),
            'positions': pd.Series(positions, index=prices.index),
            'trades': pd.DataFrame(trades)
        }
    
    def _calculate_metrics(self, returns: pd.Series, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade metrics
        if not trades.empty:
            winning_trades = trades[trades['actual_return'] > 0]
            losing_trades = trades[trades['actual_return'] < 0]
            
            win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
            
            gross_profit = winning_trades['actual_return'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['actual_return'].sum()) if len(losing_trades) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'num_trades': len(trades)
        }
    
    def _calculate_drawdown(self, returns: pd.Series) -> pd.Series:
        """Calculate drawdown series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    def walk_forward_analysis(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        prices: pd.Series,
        train_window: int = 252,
        test_window: int = 63
    ) -> List[BacktestResults]:
        """
        Perform walk-forward analysis.
        
        Args:
            model: ML model to test
            X: Feature matrix
            y: Target variable
            prices: Price series
            train_window: Training window size
            test_window: Test window size
            
        Returns:
            List of BacktestResults for each walk-forward period
        """
        results = []
        n_samples = len(X)
        
        for start in range(0, n_samples - train_window - test_window, test_window):
            # Training period
            train_end = start + train_window
            X_train = X.iloc[start:train_end]
            y_train = y.iloc[start:train_end]
            
            # Test period
            test_start = train_end
            test_end = min(test_start + test_window, n_samples)
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            prices_test = prices.iloc[test_start:test_end]
            
            if len(X_test) == 0:
                continue
            
            # Train model
            model.fit(X_train, y_train)
            
            # Generate predictions
            predictions = model.predict(X_test)
            
            # Run backtest
            fold_results = self._run_single_backtest(predictions, prices_test, y_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(fold_results['returns'], fold_results['trades'])
            
            result = BacktestResults(
                returns=fold_results['returns'],
                positions=fold_results['positions'],
                trades=fold_results['trades'],
                metrics=metrics,
                drawdown=self._calculate_drawdown(fold_results['returns']),
                sharpe_ratio=metrics.get('sharpe_ratio', 0),
                max_drawdown=metrics.get('max_drawdown', 0),
                win_rate=metrics.get('win_rate', 0),
                profit_factor=metrics.get('profit_factor', 0),
                calmar_ratio=metrics.get('calmar_ratio', 0)
            )
            
            results.append(result)
        
        return results


def run_backtest(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    prices: pd.Series,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001
) -> BacktestResults:
    """
    Convenience function for running backtests.
    
    Args:
        model: Trained ML model
        X: Feature matrix
        y: Target variable
        prices: Price series
        initial_capital: Initial capital
        transaction_cost: Transaction cost as fraction
        
    Returns:
        BacktestResults object
    """
    engine = BacktestingEngine(
        initial_capital=initial_capital,
        transaction_cost=transaction_cost
    )
    
    return engine.run_backtest(model, X, y, prices)
