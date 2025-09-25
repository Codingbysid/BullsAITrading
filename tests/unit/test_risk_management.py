"""
Unit tests for risk management module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.risk.risk_management import (
    RiskManager,
    KellyCriterion,
    calculate_var,
    calculate_cvar,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio
)


class TestRiskManager:
    """Test RiskManager class."""
    
    def test_init(self):
        """Test RiskManager initialization."""
        risk_manager = RiskManager()
        
        assert hasattr(risk_manager, 'max_position_size')
        assert hasattr(risk_manager, 'max_drawdown')
        assert hasattr(risk_manager, 'target_sharpe_ratio')
    
    def test_calculate_risk_metrics(self, sample_returns):
        """Test risk metrics calculation."""
        risk_manager = RiskManager()
        
        metrics = risk_manager.calculate_risk_metrics(sample_returns)
        
        assert 'volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'var_95' in metrics
        assert 'cvar_95' in metrics
        
        # Test that metrics are reasonable
        assert metrics['volatility'] >= 0
        assert metrics['max_drawdown'] <= 0
        assert metrics['var_95'] <= 0
    
    def test_calculate_position_size(self, sample_returns):
        """Test position size calculation."""
        risk_manager = RiskManager()
        
        # Test with different confidence levels
        position_size = risk_manager.calculate_position_size(
            expected_return=0.02,
            volatility=0.15,
            confidence=0.95
        )
        
        assert 0 <= position_size <= 1
        assert position_size <= risk_manager.max_position_size
    
    def test_check_risk_limits(self, sample_returns):
        """Test risk limit checking."""
        risk_manager = RiskManager()
        
        # Test within limits
        metrics = risk_manager.calculate_risk_metrics(sample_returns)
        within_limits = risk_manager.check_risk_limits(metrics)
        
        assert isinstance(within_limits, bool)
    
    def test_optimize_portfolio(self, sample_returns):
        """Test portfolio optimization."""
        risk_manager = RiskManager()
        
        # Create sample portfolio data
        portfolio_data = pd.DataFrame({
            'AAPL': sample_returns,
            'GOOGL': sample_returns * 1.1,
            'MSFT': sample_returns * 0.9
        })
        
        optimized_weights = risk_manager.optimize_portfolio(portfolio_data)
        
        assert len(optimized_weights) == 3
        assert np.isclose(optimized_weights.sum(), 1.0, rtol=1e-3)
        assert all(0 <= w <= 1 for w in optimized_weights)


class TestKellyCriterion:
    """Test KellyCriterion class."""
    
    def test_init(self):
        """Test KellyCriterion initialization."""
        kelly = KellyCriterion()
        
        assert hasattr(kelly, 'min_bet_size')
        assert hasattr(kelly, 'max_bet_size')
    
    def test_calculate_kelly_fraction(self):
        """Test Kelly fraction calculation."""
        kelly = KellyCriterion()
        
        # Test with positive expected value
        fraction = kelly.calculate_kelly_fraction(
            win_probability=0.6,
            win_amount=1.0,
            loss_amount=1.0
        )
        
        assert 0 <= fraction <= 1
        assert fraction > 0  # Should be positive for profitable bet
    
    def test_calculate_kelly_fraction_negative_ev(self):
        """Test Kelly fraction with negative expected value."""
        kelly = KellyCriterion()
        
        # Test with negative expected value
        fraction = kelly.calculate_kelly_fraction(
            win_probability=0.3,
            win_amount=1.0,
            loss_amount=1.0
        )
        
        assert fraction == 0  # Should be 0 for unprofitable bet
    
    def test_apply_kelly_criterion(self, sample_returns):
        """Test applying Kelly criterion to returns."""
        kelly = KellyCriterion()
        
        # Create sample predictions and returns
        predictions = np.random.uniform(0, 1, len(sample_returns))
        actual_returns = sample_returns.values
        
        kelly_weights = kelly.apply_kelly_criterion(predictions, actual_returns)
        
        assert len(kelly_weights) == len(sample_returns)
        assert all(0 <= w <= 1 for w in kelly_weights)
    
    def test_kelly_with_uncertainty(self):
        """Test Kelly criterion with uncertainty."""
        kelly = KellyCriterion()
        
        # Test with confidence intervals
        win_prob = 0.6
        win_amount = 1.0
        loss_amount = 1.0
        confidence = 0.8
        
        fraction = kelly.calculate_kelly_fraction_with_uncertainty(
            win_probability=win_prob,
            win_amount=win_amount,
            loss_amount=loss_amount,
            confidence=confidence
        )
        
        assert 0 <= fraction <= 1


class TestRiskMetrics:
    """Test individual risk metric calculations."""
    
    def test_calculate_var(self, sample_returns):
        """Test VaR calculation."""
        var_95 = calculate_var(sample_returns, confidence_level=0.95)
        var_99 = calculate_var(sample_returns, confidence_level=0.99)
        
        assert var_95 >= var_99  # 99% VaR should be more extreme
        assert var_95 <= 0  # VaR should be negative (loss)
        assert var_99 <= 0
    
    def test_calculate_cvar(self, sample_returns):
        """Test CVaR calculation."""
        cvar_95 = calculate_cvar(sample_returns, confidence_level=0.95)
        cvar_99 = calculate_cvar(sample_returns, confidence_level=0.99)
        
        assert cvar_95 >= cvar_99  # 99% CVaR should be more extreme
        assert cvar_95 <= 0  # CVaR should be negative (loss)
        assert cvar_99 <= 0
    
    def test_calculate_max_drawdown(self, sample_returns):
        """Test maximum drawdown calculation."""
        max_dd = calculate_max_drawdown(sample_returns)
        
        assert max_dd <= 0  # Drawdown should be negative
        assert isinstance(max_dd, float)
    
    def test_calculate_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        sharpe = calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02)
        
        assert isinstance(sharpe, float)
        # Sharpe ratio can be positive or negative
    
    def test_calculate_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        sortino = calculate_sortino_ratio(sample_returns, risk_free_rate=0.02)
        
        assert isinstance(sortino, float)
        # Sortino ratio can be positive or negative
    
    def test_calculate_calmar_ratio(self, sample_returns):
        """Test Calmar ratio calculation."""
        calmar = calculate_calmar_ratio(sample_returns)
        
        assert isinstance(calmar, float)
        # Calmar ratio can be positive or negative


class TestRiskLimits:
    """Test risk limit enforcement."""
    
    def test_position_size_limits(self):
        """Test position size limit enforcement."""
        risk_manager = RiskManager()
        
        # Test normal position size
        normal_size = risk_manager.calculate_position_size(0.02, 0.15, 0.95)
        assert normal_size <= risk_manager.max_position_size
        
        # Test extreme position size
        extreme_size = risk_manager.calculate_position_size(0.5, 0.01, 0.99)
        assert extreme_size <= risk_manager.max_position_size
    
    def test_drawdown_limits(self, sample_returns):
        """Test drawdown limit enforcement."""
        risk_manager = RiskManager()
        
        metrics = risk_manager.calculate_risk_metrics(sample_returns)
        
        # Check if drawdown exceeds limits
        if metrics['max_drawdown'] < -risk_manager.max_drawdown:
            within_limits = risk_manager.check_risk_limits(metrics)
            assert not within_limits
    
    def test_sharpe_ratio_targets(self, sample_returns):
        """Test Sharpe ratio target enforcement."""
        risk_manager = RiskManager()
        
        metrics = risk_manager.calculate_risk_metrics(sample_returns)
        
        # If Sharpe ratio is below target, should flag
        if metrics['sharpe_ratio'] < risk_manager.target_sharpe_ratio:
            within_limits = risk_manager.check_risk_limits(metrics)
            # This might be False if other limits are also violated


class TestStressTesting:
    """Test stress testing scenarios."""
    
    def test_market_crash_scenario(self):
        """Test risk metrics during market crash."""
        # Simulate market crash with large negative returns
        crash_returns = pd.Series([-0.05, -0.08, -0.12, -0.15, -0.10])
        
        risk_manager = RiskManager()
        metrics = risk_manager.calculate_risk_metrics(crash_returns)
        
        # During crash, drawdown should be large
        assert metrics['max_drawdown'] < -0.1
        
        # VaR should be very negative
        assert metrics['var_95'] < -0.05
    
    def test_high_volatility_scenario(self):
        """Test risk metrics during high volatility."""
        # Simulate high volatility period
        high_vol_returns = pd.Series([0.1, -0.15, 0.2, -0.18, 0.12])
        
        risk_manager = RiskManager()
        metrics = risk_manager.calculate_risk_metrics(high_vol_returns)
        
        # Volatility should be high
        assert metrics['volatility'] > 0.1
        
        # Sharpe ratio might be low due to high volatility
        assert metrics['sharpe_ratio'] < 2.0
    
    def test_low_volatility_scenario(self):
        """Test risk metrics during low volatility."""
        # Simulate low volatility period
        low_vol_returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
        
        risk_manager = RiskManager()
        metrics = risk_manager.calculate_risk_metrics(low_vol_returns)
        
        # Volatility should be low
        assert metrics['volatility'] < 0.05
        
        # Sharpe ratio might be high due to low volatility
        if metrics['sharpe_ratio'] > 0:
            assert metrics['sharpe_ratio'] > 0.5


class TestPortfolioRisk:
    """Test portfolio-level risk calculations."""
    
    def test_portfolio_var(self):
        """Test portfolio VaR calculation."""
        # Create sample portfolio returns
        portfolio_returns = pd.DataFrame({
            'AAPL': np.random.randn(100) * 0.02,
            'GOOGL': np.random.randn(100) * 0.025,
            'MSFT': np.random.randn(100) * 0.018
        })
        
        risk_manager = RiskManager()
        
        # Test portfolio VaR
        portfolio_var = risk_manager.calculate_portfolio_var(
            portfolio_returns, 
            weights=[0.4, 0.3, 0.3]
        )
        
        assert portfolio_var <= 0
        assert isinstance(portfolio_var, float)
    
    def test_correlation_risk(self):
        """Test correlation-based risk calculations."""
        # Create correlated returns
        base_returns = np.random.randn(100)
        correlated_returns = pd.DataFrame({
            'Stock1': base_returns * 0.02,
            'Stock2': base_returns * 0.02 + np.random.randn(100) * 0.01,
            'Stock3': -base_returns * 0.02 + np.random.randn(100) * 0.01
        })
        
        risk_manager = RiskManager()
        
        # Calculate correlation matrix
        correlation_matrix = correlated_returns.corr()
        
        # Test that correlations are reasonable
        assert correlation_matrix.loc['Stock1', 'Stock2'] > 0.5  # High correlation
        assert correlation_matrix.loc['Stock1', 'Stock3'] < -0.5  # Negative correlation
    
    def test_diversification_benefits(self):
        """Test diversification benefits in risk reduction."""
        # Create uncorrelated returns
        uncorrelated_returns = pd.DataFrame({
            'Stock1': np.random.randn(100) * 0.02,
            'Stock2': np.random.randn(100) * 0.02,
            'Stock3': np.random.randn(100) * 0.02
        })
        
        risk_manager = RiskManager()
        
        # Calculate individual stock risks
        individual_risks = [uncorrelated_returns[col].std() for col in uncorrelated_returns.columns]
        avg_individual_risk = np.mean(individual_risks)
        
        # Calculate portfolio risk with equal weights
        portfolio_returns = uncorrelated_returns.mean(axis=1)
        portfolio_risk = portfolio_returns.std()
        
        # Portfolio risk should be lower than average individual risk
        assert portfolio_risk < avg_individual_risk