from src.utils.common_imports import *
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import json
from pathlib import Path
from ..trading.decision_engine import decision_engine
from ..risk.risk_management import RiskManager
from ..config.settings import get_settings

"""
Performance monitoring dashboard for the QuantAI Trading Platform.

This module provides real-time monitoring of:
- Trading performance metrics
- Risk metrics and alerts
- Model performance
- System health
"""



logger = setup_logger()


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    var_95: float
    cvar_95: float


@dataclass
class RiskAlert:
    """Risk alert data structure."""
    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    symbol: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None


class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self):
        self.settings = get_settings()
        self.risk_manager = RiskManager()
        
        # Performance tracking
        self.performance_history = []
        self.risk_alerts = []
        self.model_performance = {}
        
        # Risk thresholds
        self.risk_thresholds = {
            'max_drawdown': -0.10,
            'min_sharpe_ratio': 1.0,
            'max_volatility': 0.25,
            'max_var_95': -0.05,
            'min_win_rate': 0.5
        }
        
        # Performance targets
        self.performance_targets = {
            'target_return': 0.15,
            'target_sharpe_ratio': 1.5,
            'target_win_rate': 0.6,
            'max_drawdown_limit': -0.08
        }
    
    def update_performance(
        self,
        portfolio_value: float,
        returns: pd.Series,
        positions: Dict[str, Dict[str, Any]],
        trading_signals: List[Any]
    ) -> PerformanceMetrics:
        """
        Update and calculate performance metrics.
        
        Args:
            portfolio_value: Current portfolio value
            returns: Historical returns
            positions: Current positions
            trading_signals: Recent trading signals
            
        Returns:
            Updated performance metrics
        """
        try:
            # Calculate basic metrics
            total_return = (1 + returns).prod() - 1 if len(returns) > 0 else 0
            annualized_return = (1 + returns.mean()) ** 252 - 1 if len(returns) > 0 else 0
            volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
            
            # Risk-adjusted metrics
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = annualized_return / abs(self._calculate_max_drawdown(returns)) if self._calculate_max_drawdown(returns) != 0 else 0
            
            # Drawdown metrics
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # Trading metrics
            win_rate = self._calculate_win_rate(returns)
            profit_factor = self._calculate_profit_factor(returns)
            
            # Risk metrics
            var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
            cvar_95 = returns[returns <= var_95].mean() if len(returns) > 0 else 0
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                var_95=var_95,
                cvar_95=cvar_95
            )
            
            # Store metrics
            self.performance_history.append(metrics)
            
            # Check for risk alerts
            self._check_risk_alerts(metrics, positions)
            
            # Update model performance
            self._update_model_performance(trading_signals)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
            return self._create_default_metrics()
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0
        
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        if downside_volatility == 0:
            return 0
        
        annualized_return = (1 + returns.mean()) ** 252 - 1
        return annualized_return / downside_volatility
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_win_rate(self, returns: pd.Series) -> float:
        """Calculate win rate (percentage of positive returns)."""
        if len(returns) == 0:
            return 0
        
        positive_returns = (returns > 0).sum()
        return positive_returns / len(returns)
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        if len(returns) == 0:
            return 0
        
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        
        if negative_returns == 0:
            return float('inf') if positive_returns > 0 else 0
        
        return positive_returns / negative_returns
    
    def _check_risk_alerts(self, metrics: PerformanceMetrics, positions: Dict[str, Dict[str, Any]]):
        """Check for risk alerts and add them to the alert list."""
        alerts = []
        
        # Check drawdown alert
        if metrics.max_drawdown < self.risk_thresholds['max_drawdown']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type='DRAWDOWN',
                severity='HIGH',
                message=f'Maximum drawdown exceeded: {metrics.max_drawdown:.2%}',
                value=metrics.max_drawdown,
                threshold=self.risk_thresholds['max_drawdown']
            ))
        
        # Check Sharpe ratio alert
        if metrics.sharpe_ratio < self.risk_thresholds['min_sharpe_ratio']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type='SHARPE_RATIO',
                severity='MEDIUM',
                message=f'Sharpe ratio below threshold: {metrics.sharpe_ratio:.2f}',
                value=metrics.sharpe_ratio,
                threshold=self.risk_thresholds['min_sharpe_ratio']
            ))
        
        # Check volatility alert
        if metrics.volatility > self.risk_thresholds['max_volatility']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type='VOLATILITY',
                severity='MEDIUM',
                message=f'Volatility too high: {metrics.volatility:.2%}',
                value=metrics.volatility,
                threshold=self.risk_thresholds['max_volatility']
            ))
        
        # Check VaR alert
        if metrics.var_95 < self.risk_thresholds['max_var_95']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type='VAR',
                severity='HIGH',
                message=f'VaR 95% exceeded: {metrics.var_95:.2%}',
                value=metrics.var_95,
                threshold=self.risk_thresholds['max_var_95']
            ))
        
        # Check win rate alert
        if metrics.win_rate < self.risk_thresholds['min_win_rate']:
            alerts.append(RiskAlert(
                timestamp=datetime.now(),
                alert_type='WIN_RATE',
                severity='MEDIUM',
                message=f'Win rate below threshold: {metrics.win_rate:.2%}',
                value=metrics.win_rate,
                threshold=self.risk_thresholds['min_win_rate']
            ))
        
        # Add new alerts
        self.risk_alerts.extend(alerts)
        
        # Keep only last 100 alerts
        if len(self.risk_alerts) > 100:
            self.risk_alerts = self.risk_alerts[-100:]
    
    def _update_model_performance(self, trading_signals: List[Any]):
        """Update model performance tracking."""
        if not trading_signals:
            return
        
        # Track model accuracy (simplified)
        recent_signals = trading_signals[-10:]  # Last 10 signals
        
        for signal in recent_signals:
            if hasattr(signal, 'model_prediction') and hasattr(signal, 'confidence'):
                model_key = f"{signal.symbol}_model"
                
                if model_key not in self.model_performance:
                    self.model_performance[model_key] = {
                        'predictions': [],
                        'confidences': [],
                        'accuracy': 0.0
                    }
                
                self.model_performance[model_key]['predictions'].append(signal.model_prediction)
                self.model_performance[model_key]['confidences'].append(signal.confidence)
    
    def _create_default_metrics(self) -> PerformanceMetrics:
        """Create default metrics when calculation fails."""
        return PerformanceMetrics(
            timestamp=datetime.now(),
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            var_95=0.0,
            cvar_95=0.0
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary."""
        if not self.performance_history:
            return {}
        
        latest_metrics = self.performance_history[-1]
        
        return {
            'timestamp': latest_metrics.timestamp.isoformat(),
            'total_return': f"{latest_metrics.total_return:.2%}",
            'annualized_return': f"{latest_metrics.annualized_return:.2%}",
            'volatility': f"{latest_metrics.volatility:.2%}",
            'sharpe_ratio': f"{latest_metrics.sharpe_ratio:.2f}",
            'sortino_ratio': f"{latest_metrics.sortino_ratio:.2f}",
            'calmar_ratio': f"{latest_metrics.calmar_ratio:.2f}",
            'max_drawdown': f"{latest_metrics.max_drawdown:.2%}",
            'win_rate': f"{latest_metrics.win_rate:.2%}",
            'profit_factor': f"{latest_metrics.profit_factor:.2f}",
            'var_95': f"{latest_metrics.var_95:.2%}",
            'cvar_95': f"{latest_metrics.cvar_95:.2%}",
            'active_alerts': len([a for a in self.risk_alerts if a.severity == 'HIGH']),
            'total_alerts': len(self.risk_alerts)
        }
    
    def get_risk_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get risk alerts, optionally filtered by severity."""
        alerts = self.risk_alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return [
            {
                'timestamp': alert.timestamp.isoformat(),
                'type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'symbol': alert.symbol,
                'value': alert.value,
                'threshold': alert.threshold
            }
            for alert in alerts
        ]
    
    def get_performance_trends(self, days: int = 30) -> Dict[str, List[float]]:
        """Get performance trends over time."""
        if not self.performance_history:
            return {}
        
        # Filter recent metrics
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [m for m in self.performance_history if m.timestamp >= cutoff_date]
        
        if not recent_metrics:
            return {}
        
        return {
            'timestamps': [m.timestamp.isoformat() for m in recent_metrics],
            'total_returns': [m.total_return for m in recent_metrics],
            'sharpe_ratios': [m.sharpe_ratio for m in recent_metrics],
            'max_drawdowns': [m.max_drawdown for m in recent_metrics],
            'win_rates': [m.win_rate for m in recent_metrics],
            'volatilities': [m.volatility for m in recent_metrics]
        }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        if not self.model_performance:
            return {}
        
        model_summary = {}
        
        for model_key, performance in self.model_performance.items():
            if performance['predictions']:
                avg_confidence = np.mean(performance['confidences'])
                prediction_std = np.std(performance['predictions'])
                
                model_summary[model_key] = {
                    'avg_confidence': f"{avg_confidence:.2f}",
                    'prediction_std': f"{prediction_std:.2f}",
                    'total_predictions': len(performance['predictions'])
                }
        
        return model_summary
    
    def export_performance_data(self, filepath: str):
        """Export performance data to JSON file."""
        try:
            export_data = {
                'performance_history': [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'total_return': m.total_return,
                        'annualized_return': m.annualized_return,
                        'volatility': m.volatility,
                        'sharpe_ratio': m.sharpe_ratio,
                        'sortino_ratio': m.sortino_ratio,
                        'calmar_ratio': m.calmar_ratio,
                        'max_drawdown': m.max_drawdown,
                        'win_rate': m.win_rate,
                        'profit_factor': m.profit_factor,
                        'var_95': m.var_95,
                        'cvar_95': m.cvar_95
                    }
                    for m in self.performance_history
                ],
                'risk_alerts': self.get_risk_alerts(),
                'model_performance': self.get_model_performance(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Performance data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting performance data: {e}")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def get_dashboard_data() -> Dict[str, Any]:
    """
    Get comprehensive dashboard data.
    
    Returns:
        Dictionary containing all dashboard data
    """
    return {
        'performance_summary': performance_monitor.get_performance_summary(),
        'risk_alerts': performance_monitor.get_risk_alerts(),
        'performance_trends': performance_monitor.get_performance_trends(),
        'model_performance': performance_monitor.get_model_performance(),
        'decision_engine_metrics': decision_engine.get_performance_metrics(),
        'timestamp': datetime.now().isoformat()
    }
