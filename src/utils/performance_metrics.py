"""
Unified performance metrics calculation module.

This module provides a single source of truth for all performance metrics
calculations, eliminating duplication across the codebase.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .common_imports import (
    setup_logger, PerformanceMetrics, safe_divide, 
    DEFAULT_RISK_FREE_RATE, calculate_returns
)

logger = setup_logger(__name__)


class PerformanceCalculator:
    """Unified performance metrics calculator."""
    
    def __init__(self, risk_free_rate: float = DEFAULT_RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
    
    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio from returns series."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns.mean() * 252 - self.risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return safe_divide(excess_returns, volatility)
    
    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio from returns series."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns.mean() * 252 - self.risk_free_rate
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        return safe_divide(excess_returns, downside_deviation)
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio from returns series."""
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        max_dd = abs(self.calculate_max_drawdown(returns))
        return safe_divide(annual_return, max_dd)
    
    def calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate from returns series."""
        if len(returns) == 0:
            return 0.0
        
        winning_trades = (returns > 0).sum()
        total_trades = len(returns)
        return safe_divide(winning_trades, total_trades)
    
    def calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor from returns series."""
        if len(returns) == 0:
            return 0.0
        
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        return safe_divide(gross_profit, gross_loss)
    
    def calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio vs benchmark."""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align the series
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        if len(aligned_returns) == 0:
            return 0.0
        
        excess_returns = aligned_returns - aligned_benchmark
        tracking_error = excess_returns.std() * np.sqrt(252)
        return safe_divide(excess_returns.mean() * 252, tracking_error)
    
    def calculate_alpha_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float]:
        """Calculate alpha and beta vs benchmark."""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0, 1.0
        
        # Align the series
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        if len(aligned_returns) < 2:
            return 0.0, 1.0
        
        # Calculate beta using covariance
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        beta = safe_divide(covariance, benchmark_variance, 1.0)
        
        # Calculate alpha
        alpha = (aligned_returns.mean() * 252) - beta * (aligned_benchmark.mean() * 252)
        
        return alpha, beta
    
    def calculate_mar_ratio(self, returns: pd.Series) -> float:
        """Calculate MAR (Risk-Adjusted Return) ratio."""
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        max_drawdown = abs(self.calculate_max_drawdown(returns))
        return safe_divide(annual_return, max_drawdown)
    
    def calculate_volatility(self, returns: pd.Series, annualized: bool = True) -> float:
        """Calculate volatility from returns series."""
        if len(returns) == 0:
            return 0.0
        
        volatility = returns.std()
        return volatility * np.sqrt(252) if annualized else volatility
    
    def calculate_skewness(self, returns: pd.Series) -> float:
        """Calculate skewness of returns."""
        if len(returns) == 0:
            return 0.0
        
        return returns.skew()
    
    def calculate_kurtosis(self, returns: pd.Series) -> float:
        """Calculate kurtosis of returns."""
        if len(returns) == 0:
            return 0.0
        
        return returns.kurtosis()
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)."""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_comprehensive_metrics(self, returns: pd.Series, 
                                      benchmark_returns: Optional[pd.Series] = None,
                                      initial_value: float = 1.0) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if len(returns) == 0:
            return self._empty_metrics()
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = returns.mean() * 252
        volatility = self.calculate_volatility(returns)
        
        # Risk-adjusted metrics
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        calmar_ratio = self.calculate_calmar_ratio(returns)
        mar_ratio = self.calculate_mar_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(returns)
        
        # Trade statistics
        win_rate = self.calculate_win_rate(returns)
        profit_factor = self.calculate_profit_factor(returns)
        
        # Risk metrics
        var_95 = self.calculate_var(returns, 0.05)
        cvar_95 = self.calculate_cvar(returns, 0.05)
        skewness = self.calculate_skewness(returns)
        kurtosis = self.calculate_kurtosis(returns)
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'mar_ratio': mar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
        # Benchmark-relative metrics
        if benchmark_returns is not None:
            alpha, beta = self.calculate_alpha_beta(returns, benchmark_returns)
            information_ratio = self.calculate_information_ratio(returns, benchmark_returns)
            
            metrics.update({
                'alpha': alpha,
                'beta': beta,
                'information_ratio': information_ratio
            })
        
        return metrics
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics dictionary."""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'mar_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0
        }
    
    def calculate_portfolio_metrics(self, portfolio_values: pd.Series, 
                                  benchmark_values: Optional[pd.Series] = None) -> PerformanceMetrics:
        """Calculate metrics from portfolio values series."""
        if len(portfolio_values) == 0:
            return self._empty_metrics()
        
        # Calculate returns from portfolio values
        returns = calculate_returns(portfolio_values)
        
        # Calculate benchmark returns if provided
        benchmark_returns = None
        if benchmark_values is not None:
            benchmark_returns = calculate_returns(benchmark_values)
        
        return self.calculate_comprehensive_metrics(returns, benchmark_returns)
    
    def calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics from trade list."""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'gross_profit': 0.0,
                'gross_loss': 0.0
            }
        
        # Calculate trade returns
        trade_returns = []
        for trade in trades:
            if 'return' in trade:
                trade_returns.append(trade['return'])
            elif 'pnl' in trade:
                # Convert P&L to return if needed
                trade_returns.append(trade['pnl'])
        
        if not trade_returns:
            return {
                'total_trades': len(trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'gross_profit': 0.0,
                'gross_loss': 0.0
            }
        
        returns_series = pd.Series(trade_returns)
        
        # Calculate trade statistics
        winning_trades = (returns_series > 0).sum()
        losing_trades = (returns_series < 0).sum()
        win_rate = safe_divide(winning_trades, len(returns_series))
        
        avg_win = returns_series[returns_series > 0].mean() if winning_trades > 0 else 0.0
        avg_loss = returns_series[returns_series < 0].mean() if losing_trades > 0 else 0.0
        
        gross_profit = returns_series[returns_series > 0].sum()
        gross_loss = abs(returns_series[returns_series < 0].sum())
        profit_factor = safe_divide(gross_profit, gross_loss)
        
        return {
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }


# Global instance for easy access
performance_calculator = PerformanceCalculator()


# Convenience functions
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> float:
    """Calculate Sharpe ratio from returns series."""
    calculator = PerformanceCalculator(risk_free_rate)
    return calculator.calculate_sharpe_ratio(returns)

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from returns series."""
    return performance_calculator.calculate_max_drawdown(returns)

def calculate_performance_metrics(returns: pd.Series, 
                                benchmark_returns: Optional[pd.Series] = None) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics."""
    return performance_calculator.calculate_comprehensive_metrics(returns, benchmark_returns)

def calculate_portfolio_metrics(portfolio_values: pd.Series, 
                              benchmark_values: Optional[pd.Series] = None) -> PerformanceMetrics:
    """Calculate metrics from portfolio values series."""
    return performance_calculator.calculate_portfolio_metrics(portfolio_values, benchmark_values)
