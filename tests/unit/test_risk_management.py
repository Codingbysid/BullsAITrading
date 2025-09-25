"""
Unit tests for risk management module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the modules to test
try:
    from src.risk.focused_risk_management import FocusedRiskManager
    RISK_MODULES_AVAILABLE = True
except ImportError:
    RISK_MODULES_AVAILABLE = False
    pytest.skip("Risk management modules not available", allow_module_level=True)


class TestFocusedRiskManager:
    """Test the FocusedRiskManager class."""
    
    @pytest.fixture
    def risk_manager(self):
        """Create a risk manager instance for testing."""
        return FocusedRiskManager(
            max_position_size=0.1,
            max_portfolio_drawdown=0.15,
            max_ticker_drawdown=0.20,
            var_confidence=0.05
        )
    
    def test_initialization(self, risk_manager):
        """Test risk manager initialization."""
        assert risk_manager.max_position_size == 0.1
        assert risk_manager.max_portfolio_drawdown == 0.15
        assert risk_manager.max_ticker_drawdown == 0.20
        assert risk_manager.var_confidence == 0.05
    
    def test_calculate_kelly_fraction(self, risk_manager):
        """Test Kelly Criterion calculation."""
        # Test case 1: Positive expected value
        kelly = risk_manager.calculate_kelly_fraction(
            win_probability=0.6,
            average_win=0.1,
            average_loss=0.05
        )
        assert kelly > 0, "Kelly fraction should be positive for positive expected value"
        assert kelly <= risk_manager.max_position_size, "Kelly fraction should not exceed max position size"
        
        # Test case 2: Negative expected value
        kelly = risk_manager.calculate_kelly_fraction(
            win_probability=0.4,
            average_win=0.05,
            average_loss=0.1
        )
        assert kelly == 0, "Kelly fraction should be 0 for negative expected value"
    
    def test_calculate_var(self, risk_manager, sample_price_data):
        """Test Value at Risk calculation."""
        returns = sample_price_data['close'].pct_change().dropna()
        
        var = risk_manager.calculate_var(returns)
        
        assert isinstance(var, float), "VaR should be a float"
        assert var < 0, "VaR should be negative (loss)"
        assert abs(var) < 1, "VaR should be reasonable (less than 100%)"
    
    def test_calculate_portfolio_metrics(self, risk_manager, sample_portfolio_data):
        """Test portfolio metrics calculation."""
        returns = sample_portfolio_data['portfolio_value'].pct_change().dropna()
        
        metrics = risk_manager.calculate_portfolio_metrics(returns)
        
        # Check required metrics
        required_metrics = [
            'total_return', 'annualized_return', 'volatility', 
            'sharpe_ratio', 'max_drawdown', 'var'
        ]
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Metric {metric} should be numeric"