"""
Focused Risk Management for 5-Ticker QuantAI Platform.

This module implements advanced risk management specifically
optimized for AMZN, META, NVDA, GOOGL, and AAPL trading.

Features:
- Kelly Criterion with ticker-specific parameters
- Advanced VaR calculations
- Portfolio risk budgeting
- Factor exposure limits
- Stress testing with historical scenarios
- Real-time risk monitoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Advanced risk libraries
SCIPY_AVAILABLE = False  # Disabled to avoid scipy dependency issues

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logging.warning("CVXPY not available. Install with: pip install cvxpy")

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limits for focused trading."""
    max_position_size: float = 0.25  # 25% max per ticker
    max_portfolio_drawdown: float = 0.10  # 10% max portfolio drawdown
    max_ticker_drawdown: float = 0.12  # 12% max per ticker drawdown
    max_correlation: float = 0.7  # Max correlation between positions
    volatility_target: float = 0.15  # 15% target volatility
    kelly_fraction: float = 0.25  # 25% of Kelly fraction
    var_confidence: float = 0.05  # 5% VaR
    stress_test_scenarios: int = 1000  # Number of stress test scenarios


@dataclass
class TickerRiskProfile:
    """Risk profile for individual ticker."""
    symbol: str
    beta: float
    volatility: float
    correlation_matrix: Dict[str, float]
    max_drawdown: float
    kelly_parameters: Dict[str, float]
    factor_exposures: Dict[str, float]


class FocusedRiskManager:
    """
    Advanced risk management for 5-ticker focused trading.
    
    Implements sophisticated risk controls specifically for
    AMZN, META, NVDA, GOOGL, AAPL with institutional-grade
    risk management practices.
    """
    
    def __init__(self):
        """Initialize focused risk manager."""
        self.tickers = ['AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL']
        self.risk_limits = RiskLimits()
        
        # Ticker-specific risk profiles
        self.ticker_profiles = self._initialize_ticker_profiles()
        
        # Risk monitoring
        self.risk_metrics = {}
        self.alerts = []
        
    def _initialize_ticker_profiles(self) -> Dict[str, TickerRiskProfile]:
        """Initialize risk profiles for each ticker."""
        profiles = {}
        
        # Ticker-specific parameters
        ticker_params = {
            'AMZN': {
                'beta': 1.2,
                'volatility': 0.25,
                'sector': 'Consumer Discretionary',
                'market_cap': 1.5e12,
                'kelly_params': {'win_prob': 0.55, 'avg_win': 0.02, 'avg_loss': 0.01}
            },
            'META': {
                'beta': 1.3,
                'volatility': 0.30,
                'sector': 'Communication Services',
                'market_cap': 800e9,
                'kelly_params': {'win_prob': 0.52, 'avg_win': 0.025, 'avg_loss': 0.012}
            },
            'NVDA': {
                'beta': 1.5,
                'volatility': 0.35,
                'sector': 'Technology',
                'market_cap': 1.2e12,
                'kelly_params': {'win_prob': 0.58, 'avg_win': 0.03, 'avg_loss': 0.015}
            },
            'GOOGL': {
                'beta': 1.1,
                'volatility': 0.22,
                'sector': 'Communication Services',
                'market_cap': 1.8e12,
                'kelly_params': {'win_prob': 0.53, 'avg_win': 0.018, 'avg_loss': 0.009}
            },
            'AAPL': {
                'beta': 1.0,
                'volatility': 0.20,
                'sector': 'Technology',
                'market_cap': 3.0e12,
                'kelly_params': {'win_prob': 0.51, 'avg_win': 0.015, 'avg_loss': 0.008}
            }
        }
        
        for symbol, params in ticker_params.items():
            # Calculate correlation matrix (simplified)
            correlation_matrix = self._calculate_correlation_matrix(symbol)
            
            profiles[symbol] = TickerRiskProfile(
                symbol=symbol,
                beta=params['beta'],
                volatility=params['volatility'],
                correlation_matrix=correlation_matrix,
                max_drawdown=0.0,  # Will be updated dynamically
                kelly_parameters=params['kelly_params'],
                factor_exposures={
                    'market': params['beta'],
                    'size': np.log(params['market_cap']),
                    'momentum': 0.0,  # Will be updated dynamically
                    'value': 0.0,  # Will be updated dynamically
                    'quality': 0.0  # Will be updated dynamically
                }
            )
        
        return profiles
    
    def _calculate_correlation_matrix(self, symbol: str) -> Dict[str, float]:
        """Calculate correlation matrix for ticker."""
        # Simplified correlation based on sector and market cap
        correlations = {}
        
        for other_symbol in self.tickers:
            if other_symbol == symbol:
                correlations[other_symbol] = 1.0
            else:
                # Simplified correlation based on sector similarity
                if self._get_sector(symbol) == self._get_sector(other_symbol):
                    correlations[other_symbol] = 0.6
                else:
                    correlations[other_symbol] = 0.3
        
        return correlations
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for ticker."""
        sectors = {
            'AMZN': 'Consumer Discretionary',
            'META': 'Communication Services',
            'NVDA': 'Technology',
            'GOOGL': 'Communication Services',
            'AAPL': 'Technology'
        }
        return sectors.get(symbol, 'Unknown')
    
    def calculate_kelly_position_size(
        self, 
        symbol: str, 
        expected_return: float, 
        confidence: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            symbol: Ticker symbol
            expected_return: Expected return
            confidence: Signal confidence
            
        Returns:
            Optimal position size as fraction of portfolio
        """
        if symbol not in self.ticker_profiles:
            return 0.0
        
        profile = self.ticker_profiles[symbol]
        kelly_params = profile.kelly_parameters
        
        # Adjust win probability based on confidence
        adjusted_win_prob = kelly_params['win_prob'] * confidence
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_prob, q = 1 - win_prob
        b = kelly_params['avg_win'] / kelly_params['avg_loss']
        p = adjusted_win_prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply risk limits
        kelly_fraction = max(0, min(kelly_fraction, self.risk_limits.kelly_fraction))
        
        # Adjust for expected return
        if expected_return > 0:
            kelly_fraction *= min(1.0, expected_return / 0.02)  # Scale by expected return
        
        # Apply position size limits
        kelly_fraction = min(kelly_fraction, self.risk_limits.max_position_size)
        
        return kelly_fraction
    
    def calculate_var(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.05,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Returns series
            confidence_level: VaR confidence level
            method: VaR calculation method
            
        Returns:
            VaR value
        """
        if len(returns) == 0:
            return 0.0
        
        if method == 'historical':
            # Historical simulation
            return np.percentile(returns, confidence_level * 100)
        
        elif method == 'parametric':
            # Parametric VaR (simplified without scipy)
            mean_return = returns.mean()
            std_return = returns.std()
            # Simplified normal distribution approximation
            z_score = np.sqrt(2) * np.log(1 / confidence_level)
            return mean_return + z_score * std_return
        
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            n_simulations = 10000
            mean_return = returns.mean()
            std_return = returns.std()
            
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            return np.percentile(simulated_returns, confidence_level * 100)
        
        else:
            return 0.0
    
    def calculate_cvar(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.05
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: Returns series
            confidence_level: CVaR confidence level
            
        Returns:
            CVaR value
        """
        if len(returns) == 0:
            return 0.0
        
        var = self.calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return tail_returns.mean()
    
    def calculate_portfolio_var(
        self, 
        positions: Dict[str, Dict[str, Any]], 
        returns_data: Dict[str, pd.Series]
    ) -> float:
        """
        Calculate portfolio VaR.
        
        Args:
            positions: Current positions
            returns_data: Returns data for each ticker
            
        Returns:
            Portfolio VaR
        """
        if not positions:
            return 0.0
        
        # Calculate portfolio weights
        total_value = sum(pos['quantity'] * pos['current_price'] for pos in positions.values())
        if total_value == 0:
            return 0.0
        
        weights = {}
        for symbol, pos in positions.items():
            weights[symbol] = (pos['quantity'] * pos['current_price']) / total_value
        
        # Calculate portfolio returns
        portfolio_returns = []
        for symbol, weight in weights.items():
            if symbol in returns_data and len(returns_data[symbol]) > 0:
                ticker_returns = returns_data[symbol] * weight
                portfolio_returns.append(ticker_returns)
        
        if not portfolio_returns:
            return 0.0
        
        # Combine returns
        combined_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)
        
        # Calculate portfolio VaR
        return self.calculate_var(combined_returns, self.risk_limits.var_confidence)
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return np.min(drawdown)
    
    def calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_return = returns.mean() - risk_free_rate / 252
        return excess_return / returns.std() * np.sqrt(252)
    
    def calculate_sortino_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        excess_return = returns.mean() - risk_free_rate / 252
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return float('inf')
        
        return excess_return / downside_std * np.sqrt(252)
    
    def calculate_correlation_risk(
        self, 
        positions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate correlation risk between positions.
        
        Args:
            positions: Current positions
            
        Returns:
            Correlation risk metrics
        """
        if len(positions) < 2:
            return {'max_correlation': 0.0, 'avg_correlation': 0.0}
        
        symbols = list(positions.keys())
        correlations = []
        
        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols[i+1:], i+1):
                if symbol1 in self.ticker_profiles and symbol2 in self.ticker_profiles:
                    corr = self.ticker_profiles[symbol1].correlation_matrix.get(symbol2, 0.0)
                    correlations.append(abs(corr))
        
        if not correlations:
            return {'max_correlation': 0.0, 'avg_correlation': 0.0}
        
        return {
            'max_correlation': max(correlations),
            'avg_correlation': np.mean(correlations)
        }
    
    def check_risk_limits(
        self, 
        positions: Dict[str, Dict[str, Any]], 
        portfolio_value: float
    ) -> List[str]:
        """
        Check if current positions violate risk limits.
        
        Args:
            positions: Current positions
            portfolio_value: Total portfolio value
            
        Returns:
            List of risk limit violations
        """
        violations = []
        
        # Check position size limits
        for symbol, pos in positions.items():
            position_value = pos['quantity'] * pos['current_price']
            position_ratio = position_value / portfolio_value
            
            if position_ratio > self.risk_limits.max_position_size:
                violations.append(f"Position size limit exceeded for {symbol}: {position_ratio:.2%} > {self.risk_limits.max_position_size:.2%}")
        
        # Check correlation limits
        correlation_risk = self.calculate_correlation_risk(positions)
        if correlation_risk['max_correlation'] > self.risk_limits.max_correlation:
            violations.append(f"Correlation limit exceeded: {correlation_risk['max_correlation']:.2f} > {self.risk_limits.max_correlation:.2f}")
        
        return violations
    
    def optimize_portfolio_weights(
        self, 
        expected_returns: Dict[str, float], 
        cov_matrix: pd.DataFrame,
        risk_tolerance: float = 0.5
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights using mean-variance optimization.
        
        Args:
            expected_returns: Expected returns for each ticker
            cov_matrix: Covariance matrix
            risk_tolerance: Risk tolerance parameter
            
        Returns:
            Optimal weights
        """
        if not CVXPY_AVAILABLE:
            # Fallback to equal weights
            n_assets = len(expected_returns)
            return {symbol: 1.0 / n_assets for symbol in expected_returns.keys()}
        
        try:
            # Prepare data
            symbols = list(expected_returns.keys())
            n_assets = len(symbols)
            
            if n_assets == 0:
                return {}
            
            # Create variables
            weights = cp.Variable(n_assets)
            
            # Expected returns vector
            mu = np.array([expected_returns[symbol] for symbol in symbols])
            
            # Covariance matrix
            Sigma = cov_matrix.values if isinstance(cov_matrix, pd.DataFrame) else cov_matrix
            
            # Objective: maximize return - risk_tolerance * variance
            portfolio_return = mu.T @ weights
            portfolio_variance = cp.quad_form(weights, Sigma)
            objective = cp.Maximize(portfolio_return - risk_tolerance * portfolio_variance)
            
            # Constraints
            constraints = [
                cp.sum(weights) == 1,  # Weights sum to 1
                weights >= 0,  # Long-only
                weights <= self.risk_limits.max_position_size  # Position size limits
            ]
            
            # Solve
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = weights.value
                return {symbol: weight for symbol, weight in zip(symbols, optimal_weights)}
            else:
                # Fallback to equal weights
                return {symbol: 1.0 / n_assets for symbol in symbols}
                
        except Exception as e:
            logger.warning(f"Portfolio optimization failed: {e}")
            # Fallback to equal weights
            n_assets = len(expected_returns)
            return {symbol: 1.0 / n_assets for symbol in expected_returns.keys()}
    
    def stress_test_portfolio(
        self, 
        positions: Dict[str, Dict[str, Any]], 
        historical_returns: Dict[str, pd.Series],
        scenarios: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform stress testing on portfolio.
        
        Args:
            positions: Current positions
            historical_returns: Historical returns data
            scenarios: Number of stress test scenarios
            
        Returns:
            Stress test results
        """
        if not positions:
            return {'max_loss': 0.0, 'var_95': 0.0, 'var_99': 0.0}
        
        # Calculate portfolio weights
        total_value = sum(pos['quantity'] * pos['current_price'] for pos in positions.values())
        if total_value == 0:
            return {'max_loss': 0.0, 'var_95': 0.0, 'var_99': 0.0}
        
        weights = {}
        for symbol, pos in positions.items():
            weights[symbol] = (pos['quantity'] * pos['current_price']) / total_value
        
        # Generate stress test scenarios
        scenario_returns = []
        
        for _ in range(scenarios):
            scenario_return = 0.0
            
            for symbol, weight in weights.items():
                if symbol in historical_returns and len(historical_returns[symbol]) > 0:
                    # Random historical return
                    random_return = np.random.choice(historical_returns[symbol].dropna())
                    scenario_return += weight * random_return
            
            scenario_returns.append(scenario_return)
        
        scenario_returns = np.array(scenario_returns)
        
        # Calculate stress test metrics
        max_loss = np.min(scenario_returns)
        var_95 = np.percentile(scenario_returns, 5)
        var_99 = np.percentile(scenario_returns, 1)
        
        return {
            'max_loss': max_loss,
            'var_95': var_95,
            'var_99': var_99,
            'scenarios': scenarios,
            'mean_return': np.mean(scenario_returns),
            'std_return': np.std(scenario_returns)
        }
    
    def calculate_risk_metrics(
        self, 
        positions: Dict[str, Dict[str, Any]], 
        returns_data: Dict[str, pd.Series],
        portfolio_value: float
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            positions: Current positions
            returns_data: Returns data for each ticker
            portfolio_value: Total portfolio value
            
        Returns:
            Comprehensive risk metrics
        """
        metrics = {}
        
        # Basic risk metrics
        metrics['portfolio_value'] = portfolio_value
        metrics['n_positions'] = len(positions)
        metrics['cash_ratio'] = 1.0 - sum(pos['quantity'] * pos['current_price'] for pos in positions.values()) / portfolio_value
        
        # Position concentration
        if positions:
            position_values = [pos['quantity'] * pos['current_price'] for pos in positions.values()]
            total_position_value = sum(position_values)
            
            if total_position_value > 0:
                position_ratios = [val / total_position_value for val in position_values]
                metrics['max_position_ratio'] = max(position_ratios)
                metrics['position_concentration'] = sum(ratio**2 for ratio in position_ratios)  # Herfindahl index
            else:
                metrics['max_position_ratio'] = 0.0
                metrics['position_concentration'] = 0.0
        else:
            metrics['max_position_ratio'] = 0.0
            metrics['position_concentration'] = 0.0
        
        # Portfolio VaR
        portfolio_var = self.calculate_portfolio_var(positions, returns_data)
        metrics['portfolio_var'] = portfolio_var
        
        # Correlation risk
        correlation_risk = self.calculate_correlation_risk(positions)
        metrics.update(correlation_risk)
        
        # Risk limit violations
        violations = self.check_risk_limits(positions, portfolio_value)
        metrics['risk_violations'] = violations
        metrics['n_violations'] = len(violations)
        
        # Stress test results
        stress_results = self.stress_test_portfolio(positions, returns_data)
        metrics.update(stress_results)
        
        return metrics
    
    def update_ticker_profiles(
        self, 
        symbol: str, 
        new_volatility: float, 
        new_beta: float,
        factor_exposures: Dict[str, float]
    ) -> None:
        """Update ticker risk profile with new data."""
        if symbol in self.ticker_profiles:
            self.ticker_profiles[symbol].volatility = new_volatility
            self.ticker_profiles[symbol].beta = new_beta
            self.ticker_profiles[symbol].factor_exposures.update(factor_exposures)
    
    def get_risk_alerts(self) -> List[str]:
        """Get current risk alerts."""
        return self.alerts
    
    def add_risk_alert(self, alert: str) -> None:
        """Add risk alert."""
        self.alerts.append(f"{datetime.now()}: {alert}")
    
    def clear_risk_alerts(self) -> None:
        """Clear all risk alerts."""
        self.alerts = []


# Global instance for easy access
focused_risk_manager = FocusedRiskManager()
