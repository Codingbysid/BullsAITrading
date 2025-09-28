from src.utils.common_imports import *
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
    import QuantLib as ql
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation
    from pypfopt.cla import CLA
    import riskfolio as rp
    import empyrical as ep
    import statsmodels.api as sm
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    from arch import arch_model

"""
Advanced Quantitative Models for QuantAI Trading Platform.

This module implements cutting-edge quantitative finance models including:
- Factor models and risk attribution
- Advanced portfolio optimization
- Regime switching models
- Volatility forecasting
- Options pricing and Greeks
"""

warnings.filterwarnings('ignore')

# Advanced quantitative libraries
try:
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False
    logging.warning("QuantLib not available. Install with: pip install QuantLib-Python")

try:
    PYPORTFOLIO_AVAILABLE = True
except ImportError:
    PYPORTFOLIO_AVAILABLE = False
    logging.warning("PyPortfolioOpt not available. Install with: pip install PyPortfolioOpt")

try:
    RISKFOLIO_AVAILABLE = True
except ImportError:
    RISKFOLIO_AVAILABLE = False
    logging.warning("Riskfolio-Lib not available. Install with: pip install Riskfolio-Lib")

try:
    EMPYRICIAL_AVAILABLE = True
except ImportError:
    EMPYRICIAL_AVAILABLE = False
    logging.warning("Empyrical not available. Install with: pip install empyrical-reloaded")

try:
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not available. Install with: pip install statsmodels")

try:
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logging.warning("ARCH not available. Install with: pip install arch")

logger = setup_logger()


@dataclass
class FactorExposure:
    """Factor exposure data structure."""
    symbol: str
    market_beta: float
    size_factor: float
    value_factor: float
    momentum_factor: float
    quality_factor: float
    volatility_factor: float


@dataclass
class RiskAttribution:
    """Risk attribution data structure."""
    total_risk: float
    factor_risk: float
    specific_risk: float
    factor_contributions: Dict[str, float]
    risk_decomposition: Dict[str, float]


class AdvancedFactorModel:
    """
    Advanced multi-factor risk model implementation.
    
    Implements Fama-French style factor models with additional factors
    for comprehensive risk attribution and portfolio optimization.
    """
    
    def __init__(self, factors: List[str] = None):
        """
        Initialize factor model.
        
        Args:
            factors: List of factors to include in the model
        """
        self.factors = factors or [
            'market', 'size', 'value', 'momentum', 
            'quality', 'volatility', 'liquidity'
        ]
        self.factor_loadings = None
        self.factor_returns = None
        self.specific_returns = None
        self.factor_covariance = None
        
    def build_factor_model(
        self, 
        returns: pd.DataFrame,
        market_cap: pd.Series = None,
        book_to_market: pd.Series = None,
        momentum_scores: pd.Series = None,
        quality_scores: pd.Series = None,
        volatility_scores: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Build comprehensive multi-factor risk model.
        
        Args:
            returns: Asset returns DataFrame
            market_cap: Market capitalization data
            book_to_market: Book-to-market ratios
            momentum_scores: Momentum factor scores
            quality_scores: Quality factor scores
            volatility_scores: Volatility factor scores
            
        Returns:
            Dictionary containing factor model results
        """
        logger.info("Building advanced factor model...")
        
        try:
            # Calculate factor exposures
            factor_exposures = self._calculate_factor_exposures(
                returns, market_cap, book_to_market, 
                momentum_scores, quality_scores, volatility_scores
            )
            
            # Estimate factor returns
            factor_returns = self._estimate_factor_returns(returns, factor_exposures)
            
            # Calculate specific returns
            specific_returns = self._calculate_specific_returns(
                returns, factor_exposures, factor_returns
            )
            
            # Calculate factor covariance matrix
            factor_covariance = factor_returns.cov()
            
            # Store results
            self.factor_loadings = factor_exposures
            self.factor_returns = factor_returns
            self.specific_returns = specific_returns
            self.factor_covariance = factor_covariance
            
            logger.info("✅ Factor model built successfully")
            
            return {
                'factor_loadings': factor_exposures,
                'factor_returns': factor_returns,
                'specific_returns': specific_returns,
                'factor_covariance': factor_covariance,
                'r_squared': self._calculate_r_squared(returns, factor_exposures, factor_returns)
            }
            
        except Exception as e:
            logger.error(f"Error building factor model: {e}")
            return {'error': str(e)}
    
    def _calculate_factor_exposures(
        self, 
        returns: pd.DataFrame,
        market_cap: pd.Series = None,
        book_to_market: pd.Series = None,
        momentum_scores: pd.Series = None,
        quality_scores: pd.Series = None,
        volatility_scores: pd.Series = None
    ) -> pd.DataFrame:
        """Calculate factor exposures for each asset."""
        
        factor_exposures = pd.DataFrame(index=returns.columns)
        
        # Market factor (beta)
        market_returns = returns.mean(axis=1)  # Equal-weighted market proxy
        for asset in returns.columns:
            if len(returns[asset].dropna()) > 30:  # Minimum data requirement
                beta = returns[asset].cov(market_returns) / market_returns.var()
                factor_exposures.loc[asset, 'market'] = beta
            else:
                factor_exposures.loc[asset, 'market'] = 1.0  # Default beta
        
        # Size factor (negative log market cap)
        if market_cap is not None:
            factor_exposures['size'] = -np.log(market_cap.fillna(market_cap.median()))
        else:
            factor_exposures['size'] = 0.0
        
        # Value factor (log book-to-market)
        if book_to_market is not None:
            factor_exposures['value'] = np.log(book_to_market.fillna(book_to_market.median()))
        else:
            factor_exposures['value'] = 0.0
        
        # Momentum factor
        if momentum_scores is not None:
            factor_exposures['momentum'] = momentum_scores.fillna(0.0)
        else:
            # Calculate momentum from returns
            momentum = returns.rolling(252).apply(lambda x: (1 + x).prod() - 1, raw=False)
            factor_exposures['momentum'] = momentum.iloc[-1].fillna(0.0)
        
        # Quality factor
        if quality_scores is not None:
            factor_exposures['quality'] = quality_scores.fillna(0.0)
        else:
            factor_exposures['quality'] = 0.0
        
        # Volatility factor
        if volatility_scores is not None:
            factor_exposures['volatility'] = volatility_scores.fillna(0.0)
        else:
            # Calculate volatility from returns
            volatility = returns.rolling(252).std() * np.sqrt(252)
            factor_exposures['volatility'] = volatility.iloc[-1].fillna(0.0)
        
        # Liquidity factor (approximated by volume)
        factor_exposures['liquidity'] = 0.0  # Placeholder for volume-based liquidity
        
        return factor_exposures
    
    def _estimate_factor_returns(
        self, 
        returns: pd.DataFrame, 
        factor_exposures: pd.DataFrame
    ) -> pd.DataFrame:
        """Estimate factor returns using cross-sectional regression."""
        
        factor_returns = pd.DataFrame(index=returns.index)
        
        for date in returns.index:
            if date in returns.index and not returns.loc[date].isna().all():
                # Cross-sectional regression for each date
                y = returns.loc[date].dropna()
                X = factor_exposures.loc[y.index]
                
                if len(y) > len(X.columns) and not X.isna().all().all():
                    try:
                        # Add constant for intercept
                        X_with_const = sm.add_constant(X)
                        
                        # Fit regression
                        model = sm.OLS(y, X_with_const).fit()
                        
                        # Store factor returns (excluding intercept)
                        for factor in self.factors:
                            if factor in model.params.index:
                                factor_returns.loc[date, factor] = model.params[factor]
                            else:
                                factor_returns.loc[date, factor] = 0.0
                    except Exception as e:
                        logger.debug(f"Regression failed for {date}: {e}")
                        for factor in self.factors:
                            factor_returns.loc[date, factor] = 0.0
                else:
                    for factor in self.factors:
                        factor_returns.loc[date, factor] = 0.0
        
        return factor_returns.fillna(0.0)
    
    def _calculate_specific_returns(
        self, 
        returns: pd.DataFrame,
        factor_exposures: pd.DataFrame,
        factor_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate specific (idiosyncratic) returns."""
        
        specific_returns = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for asset in returns.columns:
            if asset in factor_exposures.index:
                # Calculate predicted returns from factor model
                factor_contributions = pd.Series(index=returns.index)
                for factor in self.factors:
                    if factor in factor_exposures.columns and factor in factor_returns.columns:
                        factor_contributions += (
                            factor_exposures.loc[asset, factor] * factor_returns[factor]
                        )
                
                # Specific returns = actual returns - factor contributions
                specific_returns[asset] = returns[asset] - factor_contributions
        
        return specific_returns
    
    def _calculate_r_squared(
        self, 
        returns: pd.DataFrame,
        factor_exposures: pd.DataFrame,
        factor_returns: pd.DataFrame
    ) -> pd.Series:
        """Calculate R-squared for each asset."""
        
        r_squared = pd.Series(index=returns.columns)
        
        for asset in returns.columns:
            if asset in factor_exposures.index:
                # Calculate predicted returns
                predicted_returns = pd.Series(index=returns.index)
                for factor in self.factors:
                    if factor in factor_exposures.columns and factor in factor_returns.columns:
                        predicted_returns += (
                            factor_exposures.loc[asset, factor] * factor_returns[factor]
                        )
                
                # Calculate R-squared
                actual_returns = returns[asset].dropna()
                pred_returns = predicted_returns.dropna()
                
                if len(actual_returns) > 0 and len(pred_returns) > 0:
                    # Align indices
                    common_idx = actual_returns.index.intersection(pred_returns.index)
                    if len(common_idx) > 10:
                        actual = actual_returns.loc[common_idx]
                        pred = pred_returns.loc[common_idx]
                        
                        ss_res = ((actual - pred) ** 2).sum()
                        ss_tot = ((actual - actual.mean()) ** 2).sum()
                        
                        if ss_tot > 0:
                            r_squared[asset] = 1 - (ss_res / ss_tot)
                        else:
                            r_squared[asset] = 0.0
                    else:
                        r_squared[asset] = 0.0
                else:
                    r_squared[asset] = 0.0
            else:
                r_squared[asset] = 0.0
        
        return r_squared.fillna(0.0)
    
    def calculate_risk_attribution(self, portfolio_weights: pd.Series) -> RiskAttribution:
        """
        Calculate risk attribution for a portfolio.
        
        Args:
            portfolio_weights: Portfolio weights for each asset
            
        Returns:
            RiskAttribution object with risk decomposition
        """
        if self.factor_loadings is None:
            raise ValueError("Factor model must be built first")
        
        # Portfolio factor exposures
        portfolio_exposures = self.factor_loadings.T @ portfolio_weights
        
        # Factor risk contribution
        factor_risk = portfolio_exposures.T @ self.factor_covariance @ portfolio_exposures
        
        # Specific risk contribution
        specific_variances = self.specific_returns.var()
        specific_risk = (portfolio_weights ** 2) @ specific_variances
        
        # Total risk
        total_risk = factor_risk + specific_risk
        
        # Factor contributions
        factor_contributions = {}
        for i, factor in enumerate(self.factors):
            if factor in self.factor_covariance.columns:
                factor_var = self.factor_covariance.loc[factor, factor]
                factor_exposure = portfolio_exposures.get(factor, 0.0)
                factor_contributions[factor] = factor_var * (factor_exposure ** 2)
        
        # Risk decomposition
        risk_decomposition = {
            'factor_risk_pct': (factor_risk / total_risk) * 100 if total_risk > 0 else 0,
            'specific_risk_pct': (specific_risk / total_risk) * 100 if total_risk > 0 else 0
        }
        
        return RiskAttribution(
            total_risk=total_risk,
            factor_risk=factor_risk,
            specific_risk=specific_risk,
            factor_contributions=factor_contributions,
            risk_decomposition=risk_decomposition
        )


class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimization using modern portfolio theory.
    
    Implements multiple optimization objectives and constraints
    for institutional-grade portfolio construction.
    """
    
    def __init__(self):
        """Initialize portfolio optimizer."""
        self.optimizer = None
        self.expected_returns = None
        self.covariance_matrix = None
        
    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        objective: str = 'max_sharpe',
        constraints: Dict[str, Any] = None,
        risk_free_rate: float = 0.02
    ) -> Dict[str, Any]:
        """
        Optimize portfolio using specified objective.
        
        Args:
            returns: Historical returns data
            objective: Optimization objective ('max_sharpe', 'min_vol', 'max_return')
            constraints: Additional constraints
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Dictionary with optimization results
        """
        if not PYPORTFOLIO_AVAILABLE:
            logger.warning("PyPortfolioOpt not available, using fallback optimization")
            return self._fallback_optimization(returns, objective)
        
        try:
            logger.info(f"Optimizing portfolio with objective: {objective}")
            
            # Calculate expected returns
            self.expected_returns = expected_returns.mean_historical_return(returns)
            
            # Calculate covariance matrix with shrinkage
            self.covariance_matrix = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
            
            # Create optimizer
            self.optimizer = EfficientFrontier(
                self.expected_returns, 
                self.covariance_matrix
            )
            
            # Add constraints
            if constraints:
                for constraint_type, value in constraints.items():
                    if constraint_type == 'max_weight':
                        self.optimizer.add_constraint(lambda w: w <= value)
                    elif constraint_type == 'min_weight':
                        self.optimizer.add_constraint(lambda w: w >= value)
            
            # Optimize based on objective
            if objective == 'max_sharpe':
                weights = self.optimizer.max_sharpe(risk_free_rate=risk_free_rate)
            elif objective == 'min_volatility':
                weights = self.optimizer.min_volatility()
            elif objective == 'max_return':
                weights = self.optimizer.max_quadratic_utility()
            else:
                weights = self.optimizer.max_sharpe(risk_free_rate=risk_free_rate)
            
            # Calculate performance metrics
            performance = self.optimizer.portfolio_performance(verbose=False)
            
            # Clean weights
            clean_weights = self.optimizer.clean_weights()
            
            logger.info("✅ Portfolio optimization completed")
            
            return {
                'weights': clean_weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2],
                'optimizer': self.optimizer
            }
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return {'error': str(e)}
    
    def _fallback_optimization(self, returns: pd.DataFrame, objective: str) -> Dict[str, Any]:
        """Fallback optimization when PyPortfolioOpt is not available."""
        
        # Simple equal-weight portfolio
        n_assets = len(returns.columns)
        weights = pd.Series(1/n_assets, index=returns.columns)
        
        # Calculate performance metrics
        portfolio_returns = (returns * weights).sum(axis=1)
        expected_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        
        return {
            'weights': weights,
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'method': 'equal_weight_fallback'
        }


class RegimeDetectionModel:
    """
    Market regime detection using Markov switching models.
    
    Identifies different market states (bull, bear, sideways) to
    enable regime-aware trading strategies.
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize regime detection model.
        
        Args:
            n_regimes: Number of regimes to detect
        """
        self.n_regimes = n_regimes
        self.model = None
        self.results = None
        
    def detect_regimes(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Detect market regimes using Markov switching model.
        
        Args:
            returns: Time series of returns
            
        Returns:
            Dictionary with regime detection results
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("Statsmodels not available, using simple regime detection")
            return self._simple_regime_detection(returns)
        
        try:
            logger.info(f"Detecting {self.n_regimes} market regimes...")
            
            # Prepare data
            returns_clean = returns.dropna()
            if len(returns_clean) < 100:
                logger.warning("Insufficient data for regime detection")
                return {'error': 'Insufficient data'}
            
            # Fit Markov switching model
            self.model = MarkovRegression(
                returns_clean.values.reshape(-1, 1),
                k_regimes=self.n_regimes,
                trend='c',
                switching_variance=True
            )
            
            self.results = self.model.fit(disp=False)
            
            # Get regime probabilities
            regime_probs = self.results.smoothed_marginal_probabilities
            predicted_regime = regime_probs.idxmax(axis=1)
            
            # Calculate regime statistics
            regime_stats = {}
            for regime in range(self.n_regimes):
                regime_mask = predicted_regime == regime
                regime_returns = returns_clean[regime_mask]
                
                if len(regime_returns) > 0:
                    regime_stats[f'regime_{regime}'] = {
                        'count': len(regime_returns),
                        'mean_return': regime_returns.mean(),
                        'volatility': regime_returns.std(),
                        'sharpe_ratio': regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0
                    }
            
            logger.info("✅ Regime detection completed")
            
            return {
                'model': self.results,
                'regime_probabilities': regime_probs,
                'predicted_regime': predicted_regime,
                'regime_statistics': regime_stats,
                'transition_matrix': self.results.regime_transition
            }
            
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            return {'error': str(e)}
    
    def _simple_regime_detection(self, returns: pd.Series) -> Dict[str, Any]:
        """Simple regime detection based on volatility and momentum."""
        
        # Calculate rolling statistics
        rolling_mean = returns.rolling(20).mean()
        rolling_std = returns.rolling(20).std()
        
        # Simple regime classification
        regimes = pd.Series(index=returns.index, dtype=int)
        
        # Bull market: positive mean, low volatility
        bull_mask = (rolling_mean > 0) & (rolling_std < rolling_std.quantile(0.33))
        regimes[bull_mask] = 0
        
        # Bear market: negative mean, high volatility
        bear_mask = (rolling_mean < 0) & (rolling_std > rolling_std.quantile(0.67))
        regimes[bear_mask] = 1
        
        # Sideways market: everything else
        regimes[regimes.isna()] = 2
        
        return {
            'predicted_regime': regimes,
            'method': 'simple_volatility_momentum'
        }


class VolatilityForecaster:
    """
    Advanced volatility forecasting using GARCH models.
    
    Provides volatility predictions for risk management and
    position sizing decisions.
    """
    
    def __init__(self, model_type: str = 'GARCH'):
        """
        Initialize volatility forecaster.
        
        Args:
            model_type: Type of GARCH model ('GARCH', 'EGARCH', 'GJR-GARCH')
        """
        self.model_type = model_type
        self.model = None
        self.results = None
        
    def forecast_volatility(
        self, 
        returns: pd.Series, 
        horizon: int = 22
    ) -> Dict[str, Any]:
        """
        Forecast volatility using GARCH models.
        
        Args:
            returns: Time series of returns
            horizon: Forecast horizon in days
            
        Returns:
            Dictionary with volatility forecasts
        """
        if not ARCH_AVAILABLE:
            logger.warning("ARCH not available, using simple volatility forecast")
            return self._simple_volatility_forecast(returns, horizon)
        
        try:
            logger.info(f"Forecasting volatility using {self.model_type} model...")
            
            # Clean returns data
            returns_clean = returns.dropna() * 100  # Convert to percentage
            
            if len(returns_clean) < 50:
                logger.warning("Insufficient data for GARCH modeling")
                return {'error': 'Insufficient data'}
            
            # Fit GARCH model
            if self.model_type == 'GARCH':
                self.model = arch_model(returns_clean, vol='GARCH', p=1, q=1)
            elif self.model_type == 'EGARCH':
                self.model = arch_model(returns_clean, vol='EGARCH', p=1, q=1)
            elif self.model_type == 'GJR-GARCH':
                self.model = arch_model(returns_clean, vol='GARCH', p=1, o=1, q=1)
            else:
                self.model = arch_model(returns_clean, vol='GARCH', p=1, q=1)
            
            self.results = self.model.fit(disp='off')
            
            # Generate forecasts
            forecasts = self.results.forecast(horizon=horizon, reindex=False)
            
            # Convert back to decimal
            volatility_forecast = np.sqrt(forecasts.variance.iloc[-1] / 10000)
            
            logger.info("✅ Volatility forecasting completed")
            
            return {
                'model': self.results,
                'conditional_volatility': self.results.conditional_volatility / 100,
                'volatility_forecast': volatility_forecast,
                'forecast_horizon': horizon,
                'model_summary': self.results.summary()
            }
            
        except Exception as e:
            logger.error(f"Volatility forecasting failed: {e}")
            return {'error': str(e)}
    
    def _simple_volatility_forecast(self, returns: pd.Series, horizon: int) -> Dict[str, Any]:
        """Simple volatility forecast using historical volatility."""
        
        # Calculate historical volatility
        historical_vol = returns.rolling(252).std() * np.sqrt(252)
        current_vol = historical_vol.iloc[-1]
        
        # Simple forecast (constant volatility)
        volatility_forecast = current_vol
        
        return {
            'volatility_forecast': volatility_forecast,
            'method': 'historical_volatility',
            'forecast_horizon': horizon
        }


class OptionsPricer:
    """
    Options pricing and Greeks calculation using QuantLib.
    
    Provides Black-Scholes pricing and Greeks for hedging strategies.
    """
    
    def __init__(self):
        """Initialize options pricer."""
        self.calendar = None
        self.day_counter = None
        
    def calculate_black_scholes_greeks(
        self,
        spot_price: float,
        strike_price: float,
        time_to_maturity: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate Black-Scholes option price and Greeks.
        
        Args:
            spot_price: Current stock price
            strike_price: Option strike price
            time_to_maturity: Time to maturity in years
            volatility: Annualized volatility
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            
        Returns:
            Dictionary with option price and Greeks
        """
        if not QUANTLIB_AVAILABLE:
            logger.warning("QuantLib not available, using simplified Black-Scholes")
            return self._simplified_black_scholes(
                spot_price, strike_price, time_to_maturity, 
                volatility, risk_free_rate, dividend_yield
            )
        
        try:
            # Set up QuantLib
            self.calendar = ql.UnitedStates()
            self.day_counter = ql.Actual365Fixed()
            
            # Create handles
            spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
            risk_free_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(0, self.calendar, risk_free_rate, self.day_counter)
            )
            dividend_handle = ql.YieldTermStructureHandle(
                ql.FlatForward(0, self.calendar, dividend_yield, self.day_counter)
            )
            volatility_handle = ql.BlackVolTermStructureHandle(
                ql.BlackConstantVol(0, self.calendar, volatility, self.day_counter)
            )
            
            # Create Black-Scholes process
            process = ql.BlackScholesMertonProcess(
                spot_handle, dividend_handle, risk_free_handle, volatility_handle
            )
            
            # Create option
            maturity_date = ql.Date.todaysDate() + int(time_to_maturity * 365)
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
            exercise = ql.EuropeanExercise(maturity_date)
            option = ql.EuropeanOption(payoff, exercise)
            
            # Set pricing engine
            engine = ql.AnalyticEuropeanEngine(process)
            option.setPricingEngine(engine)
            
            # Calculate price and Greeks
            price = option.NPV()
            delta = option.delta()
            gamma = option.gamma()
            theta = option.theta()
            vega = option.vega()
            rho = option.rho()
            
            return {
                'price': price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
            
        except Exception as e:
            logger.error(f"Options pricing failed: {e}")
            return {'error': str(e)}
    
    def _simplified_black_scholes(
        self,
        spot_price: float,
        strike_price: float,
        time_to_maturity: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0
    ) -> Dict[str, float]:
        """Simplified Black-Scholes calculation without QuantLib."""
        
        # Simplified Black-Scholes formula
        d1 = (np.log(spot_price / strike_price) + 
              (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_maturity) / \
             (volatility * np.sqrt(time_to_maturity))
        
        d2 = d1 - volatility * np.sqrt(time_to_maturity)
        
        # Normal CDF approximation
        def norm_cdf(x):
            return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
        
        def norm_pdf(x):
            return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        
        # Calculate price and Greeks
        price = spot_price * np.exp(-dividend_yield * time_to_maturity) * norm_cdf(d1) - \
                strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm_cdf(d2)
        
        delta = np.exp(-dividend_yield * time_to_maturity) * norm_cdf(d1)
        gamma = np.exp(-dividend_yield * time_to_maturity) * norm_pdf(d1) / \
                (spot_price * volatility * np.sqrt(time_to_maturity))
        theta = -spot_price * norm_pdf(d1) * volatility * np.exp(-dividend_yield * time_to_maturity) / \
                (2 * np.sqrt(time_to_maturity)) - \
                risk_free_rate * strike_price * np.exp(-risk_free_rate * time_to_maturity) * norm_cdf(d2)
        vega = spot_price * np.sqrt(time_to_maturity) * norm_pdf(d1) * np.exp(-dividend_yield * time_to_maturity)
        rho = strike_price * time_to_maturity * np.exp(-risk_free_rate * time_to_maturity) * norm_cdf(d2)
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }


# Global instances for easy access
factor_model = AdvancedFactorModel()
portfolio_optimizer = AdvancedPortfolioOptimizer()
regime_detector = RegimeDetectionModel()
volatility_forecaster = VolatilityForecaster()
options_pricer = OptionsPricer()
