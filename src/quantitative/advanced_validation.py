from src.utils.common_imports import *
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator
import warnings
    from mlfinlab.cross_validation import PurgedKFold, combinatorial_purged_cv
    from mlfinlab.data_structures import get_daily_vol
    import empyrical as ep
        from scipy import stats

"""
Advanced Cross-Validation and Backtesting for QuantAI Trading Platform.

This module implements sophisticated validation techniques including:
- Purged cross-validation for financial time series
- Walk-forward analysis with embargo periods
- Combinatorial purged cross-validation
- Advanced performance metrics
- Statistical significance testing
"""

warnings.filterwarnings('ignore')

# Advanced validation libraries
try:
    MLFINLAB_AVAILABLE = True
except ImportError:
    MLFINLAB_AVAILABLE = False
    logging.warning("MLFinLab not available. Install with: pip install mlfinlab")

try:
    EMPYRICIAL_AVAILABLE = True
except ImportError:
    EMPYRICIAL_AVAILABLE = False
    logging.warning("Empyrical not available. Install with: pip install empyrical-reloaded")

logger = setup_logger()


@dataclass
class ValidationResult:
    """Validation result data structure."""
    train_scores: List[float]
    test_scores: List[float]
    mean_train_score: float
    mean_test_score: float
    std_train_score: float
    std_test_score: float
    score_difference: float
    overfitting_risk: str
    statistical_significance: Dict[str, float]


@dataclass
class WalkForwardResult:
    """Walk-forward analysis result data structure."""
    periods: List[Dict[str, Any]]
    overall_performance: Dict[str, float]
    stability_metrics: Dict[str, float]
    degradation_analysis: Dict[str, float]


class PurgedTimeSeriesCV:
    """
    Advanced cross-validation for financial time series with purging.
    
    Implements purged k-fold cross-validation to prevent data leakage
    in financial time series data.
    """
    
    def __init__(
        self, 
        n_splits: int = 5,
        test_size: float = 0.2,
        embargo: pd.Timedelta = pd.Timedelta(days=1),
        purge: pd.Timedelta = pd.Timedelta(days=1)
    ):
        """
        Initialize purged time series cross-validator.
        
        Args:
            n_splits: Number of cross-validation splits
            test_size: Fraction of data to use for testing
            embargo: Embargo period to prevent data leakage
            purge: Purge period to remove overlapping samples
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.embargo = embargo
        self.purge = purge
        
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: pd.Series = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits with purging.
        
        Args:
            X: Feature matrix
            y: Target variable
            groups: Sample information sets (timestamps)
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        if groups is None:
            groups = X.index
        
        splits = []
        n_samples = len(X)
        test_size_samples = int(n_samples * self.test_size)
        
        for i in range(self.n_splits):
            # Calculate test period
            test_start = int(i * test_size_samples / self.n_splits)
            test_end = min(test_start + test_size_samples, n_samples)
            
            # Calculate train period (before test period)
            train_end = test_start - int(self.purge.total_seconds() / (24 * 3600))  # Convert to days
            train_start = max(0, train_end - (n_samples - test_size_samples))
            
            # Apply embargo
            if train_end > 0:
                train_end = max(0, train_end - int(self.embargo.total_seconds() / (24 * 3600)))
            
            if train_end > train_start:
                train_indices = np.arange(train_start, train_end)
                test_indices = np.arange(test_start, test_end)
                splits.append((train_indices, test_indices))
        
        return splits
    
    def purged_cv_score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: BaseEstimator,
        scoring: str = 'accuracy',
        sample_weight: pd.Series = None,
        groups: pd.Series = None
    ) -> ValidationResult:
        """
        Perform purged cross-validation scoring.
        
        Args:
            X: Feature matrix
            y: Target variable
            model: Model to validate
            scoring: Scoring metric
            sample_weight: Sample weights
            groups: Sample information sets
            
        Returns:
            ValidationResult with cross-validation scores
        """
        logger.info("Performing purged cross-validation...")
        
        train_scores = []
        test_scores = []
        
        splits = self.split(X, y, groups)
        
        for train_idx, test_idx in splits:
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Handle sample weights
            if sample_weight is not None:
                w_train = sample_weight.iloc[train_idx]
                w_test = sample_weight.iloc[test_idx]
            else:
                w_train = w_test = None
            
            # Fit model
            if w_train is not None:
                model.fit(X_train, y_train, sample_weight=w_train)
            else:
                model.fit(X_train, y_train)
            
            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate scores
            if scoring == 'accuracy':
                train_score = accuracy_score(y_train, y_train_pred)
                test_score = accuracy_score(y_test, y_test_pred)
            elif scoring == 'precision':
                train_score = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
                test_score = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            elif scoring == 'recall':
                train_score = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
                test_score = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            elif scoring == 'f1':
                train_score = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
                test_score = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            else:
                train_score = accuracy_score(y_train, y_train_pred)
                test_score = accuracy_score(y_test, y_test_pred)
            
            train_scores.append(train_score)
            test_scores.append(test_score)
        
        # Calculate statistics
        mean_train_score = np.mean(train_scores)
        mean_test_score = np.mean(test_scores)
        std_train_score = np.std(train_scores)
        std_test_score = np.std(test_scores)
        score_difference = mean_train_score - mean_test_score
        
        # Assess overfitting risk
        if score_difference > 0.1:
            overfitting_risk = "HIGH"
        elif score_difference > 0.05:
            overfitting_risk = "MEDIUM"
        else:
            overfitting_risk = "LOW"
        
        # Statistical significance testing
        if len(train_scores) > 1 and len(test_scores) > 1:
            t_stat, p_value = stats.ttest_rel(train_scores, test_scores)
            statistical_significance = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        else:
            statistical_significance = {
                't_statistic': 0.0,
                'p_value': 1.0,
                'significant': False
            }
        
        logger.info(f"Purged CV completed: Train={mean_train_score:.3f}, Test={mean_test_score:.3f}")
        
        return ValidationResult(
            train_scores=train_scores,
            test_scores=test_scores,
            mean_train_score=mean_train_score,
            mean_test_score=mean_test_score,
            std_train_score=std_train_score,
            std_test_score=std_test_score,
            score_difference=score_difference,
            overfitting_risk=overfitting_risk,
            statistical_significance=statistical_significance
        )


class AdvancedWalkForward:
    """
    Enhanced walk-forward analysis with purging and embargo periods.
    
    Implements sophisticated walk-forward analysis for realistic
    backtesting of trading strategies.
    """
    
    def __init__(
        self,
        initial_window: int = 252,
        step_size: int = 22,
        retrain_frequency: int = 66,
        embargo_days: int = 5,
        purge_days: int = 5
    ):
        """
        Initialize walk-forward analyzer.
        
        Args:
            initial_window: Initial training window in days
            step_size: Step size for forward movement in days
            retrain_frequency: Frequency of model retraining in days
            embargo_days: Embargo period in days
            purge_days: Purge period in days
        """
        self.initial_window = initial_window
        self.step_size = step_size
        self.retrain_frequency = retrain_frequency
        self.embargo_days = embargo_days
        self.purge_days = purge_days
        
    def walk_forward_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_class: type,
        model_kwargs: Dict[str, Any] = None,
        scoring: str = 'accuracy'
    ) -> WalkForwardResult:
        """
        Perform walk-forward analysis with purging.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_class: Model class to use
            model_kwargs: Model initialization parameters
            scoring: Scoring metric
            
        Returns:
            WalkForwardResult with analysis results
        """
        logger.info("Starting advanced walk-forward analysis...")
        
        if model_kwargs is None:
            model_kwargs = {}
        
        periods = []
        n_samples = len(X)
        
        for start in range(self.initial_window, n_samples, self.step_size):
            # Define training window
            train_end = start
            train_start = max(0, train_end - self.initial_window)
            
            # Define test window
            test_start = start
            test_end = min(n_samples, test_start + self.step_size)
            
            # Apply purging
            purge_end = min(train_end, test_start - self.purge_days)
            if purge_end <= train_start:
                continue
            
            # Apply embargo
            embargo_end = min(purge_end, test_start - self.embargo_days)
            if embargo_end <= train_start:
                continue
            
            # Split data
            X_train = X.iloc[train_start:embargo_end]
            y_train = y.iloc[train_start:embargo_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            if len(X_train) < 50 or len(X_test) < 5:
                continue
            
            # Train model
            model = model_class(**model_kwargs)
            model.fit(X_train, y_train)
            
            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate scores
            if scoring == 'accuracy':
                train_score = accuracy_score(y_train, y_train_pred)
                test_score = accuracy_score(y_test, y_test_pred)
            elif scoring == 'f1':
                train_score = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
                test_score = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            else:
                train_score = accuracy_score(y_train, y_train_pred)
                test_score = accuracy_score(y_test, y_test_pred)
            
            # Store period results
            period_result = {
                'period': len(periods),
                'train_period': (train_start, embargo_end),
                'test_period': (test_start, test_end),
                'train_score': train_score,
                'test_score': test_score,
                'score_difference': train_score - test_score,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'model': model
            }
            periods.append(period_result)
        
        # Calculate overall performance
        if periods:
            train_scores = [p['train_score'] for p in periods]
            test_scores = [p['test_score'] for p in periods]
            score_differences = [p['score_difference'] for p in periods]
            
            overall_performance = {
                'mean_train_score': np.mean(train_scores),
                'mean_test_score': np.mean(test_scores),
                'std_train_score': np.std(train_scores),
                'std_test_score': np.std(test_scores),
                'mean_score_difference': np.mean(score_differences),
                'score_difference_std': np.std(score_differences)
            }
            
            # Calculate stability metrics
            stability_metrics = {
                'score_stability': 1 - np.std(test_scores),
                'degradation_trend': self._calculate_degradation_trend(test_scores),
                'consistency': len([s for s in test_scores if s > np.mean(test_scores)]) / len(test_scores)
            }
            
            # Degradation analysis
            degradation_analysis = {
                'early_performance': np.mean(test_scores[:len(test_scores)//2]),
                'late_performance': np.mean(test_scores[len(test_scores)//2:]),
                'performance_drop': np.mean(test_scores[:len(test_scores)//2]) - np.mean(test_scores[len(test_scores)//2:])
            }
        else:
            overall_performance = {}
            stability_metrics = {}
            degradation_analysis = {}
        
        logger.info(f"Walk-forward analysis completed: {len(periods)} periods")
        
        return WalkForwardResult(
            periods=periods,
            overall_performance=overall_performance,
            stability_metrics=stability_metrics,
            degradation_analysis=degradation_analysis
        )
    
    def _calculate_degradation_trend(self, scores: List[float]) -> float:
        """Calculate performance degradation trend."""
        if len(scores) < 3:
            return 0.0
        
        # Linear regression on scores over time
        x = np.arange(len(scores))
        y = np.array(scores)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        return slope


class AdvancedPerformanceMetrics:
    """
    Advanced performance metrics for financial strategies.
    
    Implements comprehensive performance analysis including
    risk-adjusted returns, drawdown analysis, and statistical tests.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance metrics calculator.
        
        Args:
            risk_free_rate: Risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        
    def calculate_comprehensive_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info("Calculating comprehensive performance metrics...")
        
        metrics = {}
        
        # Basic return metrics
        metrics['total_return'] = (1 + returns).prod() - 1
        metrics['annualized_return'] = (1 + returns).prod() ** (252 / len(returns)) - 1
        metrics['volatility'] = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = (metrics['annualized_return'] - self.risk_free_rate) / metrics['volatility']
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        metrics['max_drawdown'] = drawdown.min()
        metrics['avg_drawdown'] = drawdown[drawdown < 0].mean()
        metrics['drawdown_duration'] = self._calculate_max_drawdown_duration(drawdown)
        
        # Advanced metrics using empyrical if available
        if EMPYRICIAL_AVAILABLE:
            try:
                metrics['calmar_ratio'] = ep.calmar_ratio(returns)
                metrics['sortino_ratio'] = ep.sortino_ratio(returns)
                metrics['omega_ratio'] = ep.omega_ratio(returns)
                metrics['tail_ratio'] = ep.tail_ratio(returns)
                metrics['stability'] = ep.stability_of_timeseries(returns)
                
                # VaR and CVaR
                metrics['var_95'] = ep.value_at_risk(returns, cutoff=0.05)
                metrics['cvar_95'] = ep.conditional_value_at_risk(returns, cutoff=0.05)
                
            except Exception as e:
                logger.warning(f"Empyrical metrics failed: {e}")
                # Fallback calculations
                metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
                metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
                metrics['omega_ratio'] = self._calculate_omega_ratio(returns)
                metrics['tail_ratio'] = self._calculate_tail_ratio(returns)
                metrics['stability'] = 1 - returns.std()
                metrics['var_95'] = np.percentile(returns, 5)
                metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        else:
            # Fallback calculations
            metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
            metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns)
            metrics['omega_ratio'] = self._calculate_omega_ratio(returns)
            metrics['tail_ratio'] = self._calculate_tail_ratio(returns)
            metrics['stability'] = 1 - returns.std()
            metrics['var_95'] = np.percentile(returns, 5)
            metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()
        
        # Benchmark comparison
        if benchmark_returns is not None:
            try:
                # Align returns
                aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
                
                if len(aligned_returns) > 0:
                    metrics['alpha'] = ep.alpha(aligned_returns, aligned_benchmark)
                    metrics['beta'] = ep.beta(aligned_returns, aligned_benchmark)
                    metrics['information_ratio'] = ep.excess_sharpe(aligned_returns, aligned_benchmark)
                    metrics['tracking_error'] = ep.tracking_error(aligned_returns, aligned_benchmark)
                    metrics['treynor_ratio'] = metrics['annualized_return'] / metrics['beta'] if metrics['beta'] != 0 else 0
                else:
                    metrics['alpha'] = 0.0
                    metrics['beta'] = 1.0
                    metrics['information_ratio'] = 0.0
                    metrics['tracking_error'] = 0.0
                    metrics['treynor_ratio'] = 0.0
            except Exception as e:
                logger.warning(f"Benchmark comparison failed: {e}")
                metrics['alpha'] = 0.0
                metrics['beta'] = 1.0
                metrics['information_ratio'] = 0.0
                metrics['tracking_error'] = 0.0
                metrics['treynor_ratio'] = 0.0
        else:
            metrics['alpha'] = 0.0
            metrics['beta'] = 1.0
            metrics['information_ratio'] = 0.0
            metrics['tracking_error'] = 0.0
            metrics['treynor_ratio'] = 0.0
        
        # Statistical significance
        metrics['statistical_significance'] = self._calculate_statistical_significance(returns)
        
        logger.info("✅ Performance metrics calculated")
        
        return metrics
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calculate maximum drawdown duration in days."""
        drawdown_periods = []
        current_period = 0
        
        for dd in drawdown:
            if dd < 0:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0
        
        if current_period > 0:
            drawdown_periods.append(current_period)
        
        return max(drawdown_periods) if drawdown_periods else 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = downside_returns.std() * np.sqrt(252)
        if downside_std == 0:
            return float('inf')
        
        return (returns.mean() * 252 - self.risk_free_rate) / downside_std
    
    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        excess_returns = returns - threshold
        positive_excess = excess_returns[excess_returns > 0].sum()
        negative_excess = abs(excess_returns[excess_returns < 0].sum())
        
        if negative_excess == 0:
            return float('inf')
        
        return positive_excess / negative_excess
    
    def _calculate_tail_ratio(self, returns: pd.Series) -> float:
        """Calculate tail ratio."""
        var_95 = np.percentile(returns, 5)
        var_5 = np.percentile(returns, 95)
        
        if abs(var_95) == 0:
            return float('inf')
        
        return abs(var_5) / abs(var_95)
    
    def _calculate_statistical_significance(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate statistical significance of returns."""
        
        # T-test for mean return
        t_stat, p_value = stats.ttest_1samp(returns, 0)
        
        # Jarque-Bera test for normality
        jb_stat, jb_p_value = stats.jarque_bera(returns)
        
        return {
            'mean_return_t_stat': t_stat,
            'mean_return_p_value': p_value,
            'mean_return_significant': p_value < 0.05,
            'normality_jb_stat': jb_stat,
            'normality_jb_p_value': jb_p_value,
            'normally_distributed': jb_p_value > 0.05
        }


class CombinatorialPurgedCV:
    """
    Combinatorial purged cross-validation for robust model validation.
    
    Implements combinatorial purged cross-validation to reduce
    variance in cross-validation estimates.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_combinations: int = 10,
        test_size: float = 0.2,
        embargo: pd.Timedelta = pd.Timedelta(days=1)
    ):
        """
        Initialize combinatorial purged CV.
        
        Args:
            n_splits: Number of CV splits
            n_combinations: Number of combinations to test
            test_size: Fraction of data for testing
            embargo: Embargo period
        """
        self.n_splits = n_splits
        self.n_combinations = n_combinations
        self.test_size = test_size
        self.embargo = embargo
        
    def combinatorial_cv_score(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: BaseEstimator,
        scoring: str = 'accuracy',
        groups: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Perform combinatorial purged cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            model: Model to validate
            scoring: Scoring metric
            groups: Sample information sets
            
        Returns:
            Dictionary with combinatorial CV results
        """
        if not MLFINLAB_AVAILABLE:
            logger.warning("MLFinLab not available, using standard purged CV")
            return self._standard_purged_cv(X, y, model, scoring, groups)
        
        try:
            logger.info("Performing combinatorial purged cross-validation...")
            
            # Use MLFinLab's combinatorial purged CV
            scores = combinatorial_purged_cv(
                X, y, model, self.n_splits, self.n_combinations,
                self.test_size, self.embargo, groups
            )
            
            # Calculate statistics
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            # Confidence intervals
            confidence_interval = 1.96 * std_score / np.sqrt(len(scores))
            lower_bound = mean_score - confidence_interval
            upper_bound = mean_score + confidence_interval
            
            logger.info(f"Combinatorial CV completed: Mean={mean_score:.3f}, Std={std_score:.3f}")
            
            return {
                'scores': scores,
                'mean_score': mean_score,
                'std_score': std_score,
                'min_score': min_score,
                'max_score': max_score,
                'confidence_interval': (lower_bound, upper_bound),
                'n_combinations': self.n_combinations,
                'n_splits': self.n_splits
            }
            
        except Exception as e:
            logger.error(f"Combinatorial CV failed: {e}")
            return {'error': str(e)}
    
    def _standard_purged_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model: BaseEstimator,
        scoring: str = 'accuracy',
        groups: pd.Series = None
    ) -> Dict[str, Any]:
        """Fallback to standard purged CV when MLFinLab is not available."""
        
        purged_cv = PurgedTimeSeriesCV(
            n_splits=self.n_splits,
            test_size=self.test_size,
            embargo=self.embargo
        )
        
        result = purged_cv.purged_cv_score(X, y, model, scoring, groups=groups)
        
        return {
            'scores': result.test_scores,
            'mean_score': result.mean_test_score,
            'std_score': result.std_test_score,
            'min_score': np.min(result.test_scores),
            'max_score': np.max(result.test_scores),
            'method': 'standard_purged_cv'
        }


# Global instances for easy access
purged_cv = PurgedTimeSeriesCV()
walk_forward = AdvancedWalkForward()
performance_metrics = AdvancedPerformanceMetrics()
combinatorial_cv = CombinatorialPurgedCV()
