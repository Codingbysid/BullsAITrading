"""
Unified risk management utilities for the QuantAI Trading Platform.

This module provides standardized risk management functions to eliminate
duplication across the codebase.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .common_imports import setup_logger, safe_divide, RiskMetrics

# Optional scipy imports
try:
    from scipy.optimize import minimize
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available - some optimization features will be disabled")

logger = setup_logger(__name__)


class RiskCalculator:
    """Unified risk management calculator."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_kelly_fraction(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Kelly Criterion fraction for position sizing."""
        if len(returns) == 0:
            return 0.0
        
        # Calculate win rate and average win/loss
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        if len(winning_trades) == 0 or len(losing_trades) == 0:
            return 0.0
        
        win_rate = len(winning_trades) / len(returns)
        avg_win = winning_trades.mean()
        avg_loss = abs(losing_trades.mean())
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply confidence adjustment
        kelly_fraction *= confidence
        
        # Cap at reasonable limits
        return max(0.0, min(0.25, kelly_fraction))
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)."""
        if len(returns) == 0:
            return 0.0
        
        if SCIPY_AVAILABLE:
            return stats.norm.ppf(confidence_level) * returns.std() + returns.mean()
        else:
            return np.percentile(returns, confidence_level * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_portfolio_var(self, weights: np.ndarray, returns: pd.DataFrame, 
                               confidence_level: float = 0.05) -> float:
        """Calculate portfolio VaR using variance-covariance method."""
        if len(weights) == 0 or returns.empty:
            return 0.0
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calculate portfolio variance
        portfolio_variance = np.dot(weights, np.dot(returns.cov(), weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Calculate VaR
        if SCIPY_AVAILABLE:
            z_score = stats.norm.ppf(confidence_level)
            var = -z_score * portfolio_std
        else:
            var = -1.645 * portfolio_std  # Approximate 95% VaR
        
        return var
    
    def calculate_portfolio_cvar(self, weights: np.ndarray, returns: pd.DataFrame, 
                                confidence_level: float = 0.05) -> float:
        """Calculate portfolio CVaR."""
        if len(weights) == 0 or returns.empty:
            return 0.0
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calculate CVaR
        var = self.calculate_var(portfolio_returns, confidence_level)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        
        return -cvar
    
    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta coefficient."""
        if len(asset_returns) == 0 or len(market_returns) == 0:
            return 1.0
        
        # Align the series
        aligned_asset, aligned_market = asset_returns.align(market_returns, join='inner')
        if len(aligned_asset) < 2:
            return 1.0
        
        # Calculate beta
        covariance = np.cov(aligned_asset, aligned_market)[0, 1]
        market_variance = np.var(aligned_market)
        
        return safe_divide(covariance, market_variance, 1.0)
    
    def calculate_alpha(self, asset_returns: pd.Series, market_returns: pd.Series, 
                       risk_free_rate: Optional[float] = None) -> float:
        """Calculate alpha coefficient."""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if len(asset_returns) == 0 or len(market_returns) == 0:
            return 0.0
        
        # Align the series
        aligned_asset, aligned_market = asset_returns.align(market_returns, join='inner')
        if len(aligned_asset) < 2:
            return 0.0
        
        # Calculate alpha
        beta = self.calculate_beta(aligned_asset, aligned_market)
        alpha = (aligned_asset.mean() * 252) - risk_free_rate - beta * (aligned_market.mean() * 252 - risk_free_rate)
        
        return alpha
    
    def calculate_tracking_error(self, portfolio_returns: pd.Series, 
                                benchmark_returns: pd.Series) -> float:
        """Calculate tracking error."""
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align the series
        aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
        if len(aligned_portfolio) == 0:
            return 0.0
        
        # Calculate tracking error
        excess_returns = aligned_portfolio - aligned_benchmark
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        return tracking_error
    
    def calculate_information_ratio(self, portfolio_returns: pd.Series, 
                                   benchmark_returns: pd.Series) -> float:
        """Calculate information ratio."""
        if len(portfolio_returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align the series
        aligned_portfolio, aligned_benchmark = portfolio_returns.align(benchmark_returns, join='inner')
        if len(aligned_portfolio) == 0:
            return 0.0
        
        # Calculate information ratio
        excess_returns = aligned_portfolio - aligned_benchmark
        tracking_error = self.calculate_tracking_error(aligned_portfolio, aligned_benchmark)
        
        return safe_divide(excess_returns.mean() * 252, tracking_error)
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        
        return drawdown.min()
    
    def calculate_max_drawdown_duration(self, returns: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods."""
        if len(returns) == 0:
            return 0
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        
        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def calculate_volatility(self, returns: pd.Series, annualized: bool = True) -> float:
        """Calculate volatility."""
        if len(returns) == 0:
            return 0.0
        
        volatility = returns.std()
        return volatility * np.sqrt(252) if annualized else volatility
    
    def calculate_correlation(self, returns1: pd.Series, returns2: pd.Series) -> float:
        """Calculate correlation between two return series."""
        if len(returns1) == 0 or len(returns2) == 0:
            return 0.0
        
        # Align the series
        aligned_returns1, aligned_returns2 = returns1.align(returns2, join='inner')
        if len(aligned_returns1) < 2:
            return 0.0
        
        return aligned_returns1.corr(aligned_returns2)
    
    def calculate_portfolio_risk_metrics(self, weights: np.ndarray, returns: pd.DataFrame,
                                       benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        if len(weights) == 0 or returns.empty:
            return {
                'portfolio_volatility': 0.0,
                'portfolio_var_95': 0.0,
                'portfolio_cvar_95': 0.0,
                'portfolio_beta': 1.0,
                'portfolio_alpha': 0.0,
                'tracking_error': 0.0,
                'information_ratio': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0
            }
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Basic risk metrics
        portfolio_volatility = self.calculate_volatility(portfolio_returns)
        portfolio_var_95 = self.calculate_var(portfolio_returns, 0.05)
        portfolio_cvar_95 = self.calculate_cvar(portfolio_returns, 0.05)
        max_drawdown = self.calculate_max_drawdown(portfolio_returns)
        max_drawdown_duration = self.calculate_max_drawdown_duration(portfolio_returns)
        
        metrics = {
            'portfolio_volatility': portfolio_volatility,
            'portfolio_var_95': portfolio_var_95,
            'portfolio_cvar_95': portfolio_cvar_95,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration
        }
        
        # Benchmark-relative metrics
        if benchmark_returns is not None:
            portfolio_beta = self.calculate_beta(portfolio_returns, benchmark_returns)
            portfolio_alpha = self.calculate_alpha(portfolio_returns, benchmark_returns)
            tracking_error = self.calculate_tracking_error(portfolio_returns, benchmark_returns)
            information_ratio = self.calculate_information_ratio(portfolio_returns, benchmark_returns)
            
            metrics.update({
                'portfolio_beta': portfolio_beta,
                'portfolio_alpha': portfolio_alpha,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio
            })
        else:
            metrics.update({
                'portfolio_beta': 1.0,
                'portfolio_alpha': 0.0,
                'tracking_error': 0.0,
                'information_ratio': 0.0
            })
        
        return metrics
    
    def optimize_portfolio_weights(self, returns: pd.DataFrame, 
                                  target_return: Optional[float] = None,
                                  risk_aversion: float = 1.0) -> np.ndarray:
        """Optimize portfolio weights using mean-variance optimization."""
        if returns.empty:
            return np.array([])
        
        n_assets = len(returns.columns)
        
        if not SCIPY_AVAILABLE:
            # Return equal weights if scipy is not available
            return np.array([1.0 / n_assets] * n_assets)
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Objective function (negative Sharpe ratio for minimization)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            if portfolio_std == 0:
                return -portfolio_return
            
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe_ratio * risk_aversion
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        if target_return is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda x: np.dot(x, expected_returns) - target_return
            })
        
        # Bounds (no short selling)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            # Return equal weights if optimization fails
            return x0
    
    def calculate_risk_budget(self, weights: np.ndarray, returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk budget for each asset."""
        if len(weights) == 0 or returns.empty:
            return {}
        
        # Calculate portfolio volatility
        cov_matrix = returns.cov() * 252
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate marginal contribution to risk
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_volatility
        
        # Calculate risk budget
        risk_budget = {}
        for i, asset in enumerate(returns.columns):
            risk_budget[asset] = weights[i] * marginal_contrib[i] / portfolio_volatility
        
        return risk_budget
    
    def calculate_risk_parity_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate risk parity portfolio weights."""
        if returns.empty:
            return np.array([])
        
        n_assets = len(returns.columns)
        
        if not SCIPY_AVAILABLE:
            # Return equal weights if scipy is not available
            return np.array([1.0 / n_assets] * n_assets)
        
        # Objective function (minimize sum of squared differences from equal risk contribution)
        def objective(weights):
            cov_matrix = returns.cov() * 252
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_volatility
            risk_contrib = weights * marginal_contrib / portfolio_volatility
            
            # Target equal risk contribution
            target_contrib = 1.0 / n_assets
            
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Bounds (no short selling)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x
        else:
            # Return equal weights if optimization fails
            return x0


# Global instance for easy access
risk_calculator = RiskCalculator()


# Convenience functions
def calculate_kelly_fraction(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Kelly Criterion fraction for position sizing."""
    return risk_calculator.calculate_kelly_fraction(returns, confidence)

def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk (VaR)."""
    return risk_calculator.calculate_var(returns, confidence_level)

def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (CVaR)."""
    return risk_calculator.calculate_cvar(returns, confidence_level)

def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculate beta coefficient."""
    return risk_calculator.calculate_beta(asset_returns, market_returns)

def calculate_alpha(asset_returns: pd.Series, market_returns: pd.Series, 
                   risk_free_rate: float = 0.02) -> float:
    """Calculate alpha coefficient."""
    return risk_calculator.calculate_alpha(asset_returns, market_returns, risk_free_rate)

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown."""
    return risk_calculator.calculate_max_drawdown(returns)

def calculate_portfolio_risk_metrics(weights: np.ndarray, returns: pd.DataFrame,
                                   benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
    """Calculate comprehensive portfolio risk metrics."""
    return risk_calculator.calculate_portfolio_risk_metrics(weights, returns, benchmark_returns)
