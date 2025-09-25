"""
Advanced Quantitative Finance Module for QuantAI Trading Platform.

This module provides cutting-edge quantitative finance capabilities including:
- Advanced factor models and risk attribution
- Sophisticated portfolio optimization
- Regime detection and volatility forecasting
- Options pricing and Greeks calculation
- Purged cross-validation and walk-forward analysis
- Comprehensive performance metrics and statistical testing
"""

from .advanced_models import (
    AdvancedFactorModel,
    AdvancedPortfolioOptimizer,
    RegimeDetectionModel,
    VolatilityForecaster,
    OptionsPricer,
    factor_model,
    portfolio_optimizer,
    regime_detector,
    volatility_forecaster,
    options_pricer
)

from .advanced_validation import (
    PurgedTimeSeriesCV,
    AdvancedWalkForward,
    AdvancedPerformanceMetrics,
    CombinatorialPurgedCV,
    ValidationResult,
    WalkForwardResult,
    purged_cv,
    walk_forward,
    performance_metrics,
    combinatorial_cv
)

__all__ = [
    # Advanced Models
    'AdvancedFactorModel',
    'AdvancedPortfolioOptimizer', 
    'RegimeDetectionModel',
    'VolatilityForecaster',
    'OptionsPricer',
    'factor_model',
    'portfolio_optimizer',
    'regime_detector',
    'volatility_forecaster',
    'options_pricer',
    
    # Advanced Validation
    'PurgedTimeSeriesCV',
    'AdvancedWalkForward',
    'AdvancedPerformanceMetrics',
    'CombinatorialPurgedCV',
    'ValidationResult',
    'WalkForwardResult',
    'purged_cv',
    'walk_forward',
    'performance_metrics',
    'combinatorial_cv'
]
