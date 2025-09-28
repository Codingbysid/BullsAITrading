from src.utils.common_imports import *
from src.utils.risk_utils import RiskCalculator
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from scipy import stats
import logging
from ..config.settings import get_settings

"""
Risk management module implementing Kelly Criterion and advanced risk controls.

This module provides comprehensive risk management including:
- Kelly Criterion for optimal position sizing
- Risk-constrained Kelly for drawdown control
- Portfolio-level risk metrics
- Dynamic risk adjustment based on market conditions
"""



logger = setup_logger()


class RiskManager:
    """Advanced risk management system."""
    
    def __init__(self):
        self.settings = get_settings()
        self.max_position_size = self.settings.max_position_size
        self.max_drawdown = self.settings.max_drawdown
        self.target_sharpe_ratio = self.settings.target_sharpe_ratio
        
    risk_calc = RiskCalculator()
# Use: risk_calc.calculate_kelly_fraction(win_prob, avg_win, avg_loss)
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Kelly Criterion optimal fraction.
        
        Kelly formula: f* = (bp - q) / b
        where:
        - b = odds received on the wager (return/risk ratio)
        - p = probability of winning
        - q = probability of losing (1-p)
        
        Args:
            returns: Series of historical returns
            confidence: Confidence level for probability calculation
            
        Returns:
            Optimal Kelly fraction (0-1)
        """
        if len(returns) < 10:
            return 0.0
        
        # Calculate win probability
        win_rate = (returns > 0).mean()
        
        if win_rate == 0 or win_rate == 1:
            return 0.0
        
        # Calculate average win and loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.0
        
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())
        
        # Kelly formula
        b = avg_win / avg_loss  # Return/risk ratio
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Ensure non-negative and reasonable bounds
        kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        return kelly_fraction
    
    def calculate_risk_constrained_kelly(
        self, 
        returns: pd.Series, 
        current_drawdown: float,
        max_drawdown: Optional[float] = None
    ) -> float:
        """
        Calculate risk-constrained Kelly fraction.
        
        Reduces position size when approaching maximum drawdown limits.
        
        Args:
            returns: Series of historical returns
            current_drawdown: Current portfolio drawdown
            max_drawdown: Maximum allowed drawdown
            
        Returns:
            Risk-constrained Kelly fraction
        """
        if max_drawdown is None:
            max_drawdown = self.max_drawdown
        
        # Calculate base Kelly fraction
        base_kelly = self.calculate_kelly_fraction(returns)
        
        # Risk adjustment factor based on current drawdown
        if current_drawdown >= max_drawdown:
            return 0.0  # Stop trading if at max drawdown
        
        # Linear reduction as drawdown approaches limit
        risk_factor = 1 - (current_drawdown / max_drawdown)
        risk_factor = max(0.1, risk_factor)  # Minimum 10% of base Kelly
        
        constrained_kelly = base_kelly * risk_factor
        
        # Apply maximum position size limit
        return min(constrained_kelly, self.max_position_size)
    
    def calculate_position_size(
        self,
        predicted_return: float,
        predicted_volatility: float,
        confidence: float,
        current_drawdown: float = 0.0,
        portfolio_value: float = 1.0
    ) -> float:
        """
        Calculate optimal position size using multiple risk criteria.
        
        Args:
            predicted_return: Expected return
            predicted_volatility: Expected volatility
            confidence: Model confidence (0-1)
            current_drawdown: Current portfolio drawdown
            portfolio_value: Total portfolio value
            
        Returns:
            Position size as fraction of portfolio
        """
        # Base Kelly calculation
        if predicted_volatility == 0:
            return 0.0
        
        # Sharpe ratio based position sizing
        sharpe_ratio = predicted_return / predicted_volatility if predicted_volatility > 0 else 0
        
        # Kelly fraction based on Sharpe ratio
        kelly_fraction = sharpe_ratio / 2  # Simplified Kelly for Sharpe ratio
        
        # Apply confidence adjustment
        confidence_adjusted = kelly_fraction * confidence
        
        # Apply risk constraints
        risk_constrained = self.calculate_risk_constrained_kelly(
            pd.Series([predicted_return]), 
            current_drawdown
        )
        
        # Take minimum of all constraints
        position_size = min(
            confidence_adjusted,
            risk_constrained,
            self.max_position_size
        )
        
        # Ensure non-negative
        return max(0.0, position_size)
    
    def calculate_portfolio_risk_metrics(
        self, 
        returns: pd.Series, 
        weights: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            returns: Portfolio returns
            weights: Asset weights (optional)
            
        Returns:
            Dictionary of risk metrics
        """
        if len(returns) < 2:
            return {}
        
        metrics = {}
        
        # Basic risk metrics
        metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized
        metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
        metrics['max_drawdown'] = self.calculate_max_drawdown(returns)
        metrics['var_95'] = np.percentile(returns, 5)  # 95% VaR
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()  # CVaR
        
        # Advanced metrics
        metrics['skewness'] = stats.skew(returns)
        metrics['kurtosis'] = stats.kurtosis(returns)
        metrics['calmar_ratio'] = returns.mean() / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Information ratio (if benchmark provided)
        if weights is not None:
            portfolio_return = (returns * weights).sum()
            metrics['information_ratio'] = portfolio_return / metrics['volatility']
        
        return metrics
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calculate_var(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.05
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            VaR value
        """
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_expected_shortfall(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.05
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level
            
        Returns:
            Expected shortfall value
        """
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def optimize_portfolio_weights(
        self, 
        expected_returns: pd.Series, 
        cov_matrix: pd.DataFrame,
        risk_tolerance: float = 1.0
    ) -> pd.Series:
        """
        Optimize portfolio weights using mean-variance optimization.
        
        Args:
            expected_returns: Expected returns for each asset
            cov_matrix: Covariance matrix
            risk_tolerance: Risk tolerance parameter
            
        Returns:
            Optimal weights
        """
        n_assets = len(expected_returns)
        
        # Objective function: maximize Sharpe ratio
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return / portfolio_volatility)  # Negative for minimization
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
        
        # Bounds: no short selling, max position size
        bounds = tuple((0, self.max_position_size) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            objective, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return pd.Series(result.x, index=expected_returns.index)
        else:
            logger.warning("Portfolio optimization failed, using equal weights")
            return pd.Series([1/n_assets] * n_assets, index=expected_returns.index)
    
    def check_risk_limits(
        self, 
        current_positions: Dict[str, float],
        predicted_returns: Dict[str, float],
        current_drawdown: float
    ) -> Dict[str, bool]:
        """
        Check if current positions violate risk limits.
        
        Args:
            current_positions: Current position sizes
            predicted_returns: Predicted returns for each position
            current_drawdown: Current portfolio drawdown
            
        Returns:
            Dictionary of risk limit violations
        """
        violations = {}
        
        # Check individual position limits
        for symbol, position in current_positions.items():
            violations[f'{symbol}_max_position'] = position > self.max_position_size
        
        # Check total exposure
        total_exposure = sum(current_positions.values())
        violations['total_exposure'] = total_exposure > 1.0
        
        # Check drawdown limit
        violations['max_drawdown'] = current_drawdown > self.max_drawdown
        
        # Check concentration risk
        max_position = max(current_positions.values()) if current_positions else 0
        violations['concentration_risk'] = max_position > 0.3  # Max 30% in single position
        
        return violations
    
    def adjust_positions_for_risk(
        self, 
        positions: Dict[str, float],
        risk_violations: Dict[str, bool]
    ) -> Dict[str, float]:
        """
        Adjust positions to comply with risk limits.
        
        Args:
            positions: Current positions
            risk_violations: Risk limit violations
            
        Returns:
            Adjusted positions
        """
        adjusted_positions = positions.copy()
        
        # Reduce oversized positions
        for symbol, position in adjusted_positions.items():
            if position > self.max_position_size:
                adjusted_positions[symbol] = self.max_position_size
                logger.warning(f"Reduced {symbol} position to max limit: {self.max_position_size}")
        
        # Reduce total exposure if needed
        total_exposure = sum(adjusted_positions.values())
        if total_exposure > 1.0:
            scale_factor = 1.0 / total_exposure
            adjusted_positions = {
                symbol: position * scale_factor 
                for symbol, position in adjusted_positions.items()
            }
            logger.warning(f"Scaled down all positions by factor: {scale_factor}")
        
        return adjusted_positions


risk_calc = RiskCalculator()
# Use: risk_calc.calculate_kelly_fraction(win_prob, avg_win, avg_loss)
    """
    Convenience function for Kelly Criterion calculation.
    
    Args:
        returns: Series of returns
        
    Returns:
        Kelly fraction
    """
    risk_manager = RiskManager()
    return risk_manager.calculate_kelly_fraction(returns)


def calculate_position_size(
    predicted_return: float,
    predicted_volatility: float,
    confidence: float,
    current_drawdown: float = 0.0
) -> float:
    """
    Convenience function for position size calculation.
    
    Args:
        predicted_return: Expected return
        predicted_volatility: Expected volatility
        confidence: Model confidence
        current_drawdown: Current drawdown
        
    Returns:
        Position size
    """
    risk_manager = RiskManager()
    return risk_manager.calculate_position_size(
        predicted_return, predicted_volatility, confidence, current_drawdown
    )
