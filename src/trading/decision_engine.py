from src.utils.common_imports import *
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from ..data.sentiment_analysis import SentimentAggregator, RealTimeSentimentMonitor
from ..risk.risk_management import RiskManager, KellyCriterion
from ..models.trading_models import EnsembleModel
from ..config.settings import get_settings

"""
Advanced decision engine with risk-adjusted metrics integration.

This module implements the core decision-making logic that integrates:
- ML model predictions
- Sentiment analysis
- Risk management
- Market conditions
- Portfolio state
"""



logger = setup_logger()


@dataclass
class TradingSignal:
    """Trading signal with risk-adjusted metrics."""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    position_size: float
    risk_score: float
    expected_return: float
    max_loss: float
    sharpe_ratio: float
    kelly_fraction: float
    sentiment_score: float
    model_prediction: float
    timestamp: datetime
    reasoning: str


@dataclass
class PortfolioState:
    """Current portfolio state."""
    total_value: float
    cash: float
    positions: Dict[str, Dict[str, Any]]
    risk_metrics: Dict[str, float]
    last_rebalance: datetime


class RiskAdjustedDecisionEngine:
    """Advanced decision engine with comprehensive risk management."""
    
    def __init__(self):
        self.settings = get_settings()
        self.sentiment_aggregator = SentimentAggregator()
        self.sentiment_monitor = RealTimeSentimentMonitor()
        self.risk_manager = RiskManager()
        self.kelly_criterion = KellyCriterion()
        self.ensemble_model = EnsembleModel()
        
        # Risk thresholds
        self.max_position_size = self.settings.max_position_size
        self.max_drawdown = self.settings.max_drawdown
        self.target_sharpe_ratio = self.settings.target_sharpe_ratio
        self.min_confidence = 0.6
        self.max_risk_score = 0.7
        
        # Performance tracking
        self.performance_history = []
        self.risk_alerts = []
    
    async def generate_trading_signal(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        features: pd.DataFrame,
        portfolio_state: PortfolioState
    ) -> TradingSignal:
        """
        Generate a comprehensive trading signal with risk-adjusted metrics.
        
        Args:
            symbol: Stock symbol
            market_data: Recent market data
            features: Engineered features
            portfolio_state: Current portfolio state
            
        Returns:
            TradingSignal with risk-adjusted metrics
        """
        try:
            # 1. Get model prediction
            model_prediction = await self._get_model_prediction(symbol, features)
            
            # 2. Get sentiment analysis
            sentiment_data = await self._get_sentiment_analysis(symbol)
            
            # 3. Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(market_data, portfolio_state)
            
            # 4. Generate base signal
            base_signal = self._generate_base_signal(
                model_prediction, sentiment_data, risk_metrics
            )
            
            # 5. Apply risk adjustments
            risk_adjusted_signal = self._apply_risk_adjustments(
                base_signal, risk_metrics, portfolio_state
            )
            
            # 6. Calculate position size
            position_size = self._calculate_position_size(
                risk_adjusted_signal, risk_metrics, portfolio_state
            )
            
            # 7. Generate final signal
            final_signal = self._create_final_signal(
                symbol, risk_adjusted_signal, position_size, 
                model_prediction, sentiment_data, risk_metrics
            )
            
            # 8. Log decision
            self._log_decision(final_signal, risk_metrics)
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal for {symbol}: {e}")
            return self._create_hold_signal(symbol, f"Error: {e}")
    
    async def _get_model_prediction(self, symbol: str, features: pd.DataFrame) -> float:
        """Get ML model prediction."""
        try:
            # Use ensemble model for prediction
            prediction = self.ensemble_model.predict(features.iloc[-1:].values)
            return float(prediction[0])
        except Exception as e:
            logger.warning(f"Model prediction failed for {symbol}: {e}")
            return 0.0
    
    async def _get_sentiment_analysis(self, symbol: str) -> Dict[str, float]:
        """Get comprehensive sentiment analysis."""
        try:
            sentiment = await self.sentiment_monitor.get_live_sentiment(symbol)
            return sentiment
        except Exception as e:
            logger.warning(f"Sentiment analysis failed for {symbol}: {e}")
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'news_sentiment': 0.0,
                'social_sentiment': 0.0,
                'sample_size': 0
            }
    
    def _calculate_risk_metrics(
        self, 
        market_data: pd.DataFrame, 
        portfolio_state: PortfolioState
    ) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        try:
            # Calculate returns
            returns = market_data['close'].pct_change().dropna()
            
            # Basic risk metrics
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (returns.mean() * 252) / (volatility) if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            
            # Portfolio risk metrics
            portfolio_risk = self._calculate_portfolio_risk(portfolio_state)
            
            # Market risk metrics
            market_risk = self._calculate_market_risk(market_data)
            
            return {
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'portfolio_risk': portfolio_risk,
                'market_risk': market_risk,
                'risk_score': self._calculate_risk_score(volatility, max_drawdown, portfolio_risk)
            }
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return {
                'volatility': 0.2,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'var_95': -0.05,
                'cvar_95': -0.08,
                'portfolio_risk': 0.5,
                'market_risk': 0.5,
                'risk_score': 0.5
            }
    
    def _generate_base_signal(
        self,
        model_prediction: float,
        sentiment_data: Dict[str, float],
        risk_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate base trading signal."""
        # Combine model prediction and sentiment
        sentiment_score = sentiment_data.get('overall_sentiment', 0.0)
        sentiment_confidence = sentiment_data.get('confidence', 0.0)
        
        # Weighted combination
        model_weight = 0.6
        sentiment_weight = 0.4
        
        combined_score = (model_weight * model_prediction + 
                         sentiment_weight * sentiment_score)
        
        # Determine action
        if combined_score > 0.3 and sentiment_confidence > 0.6:
            action = 'BUY'
        elif combined_score < -0.3 and sentiment_confidence > 0.6:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'action': action,
            'score': combined_score,
            'confidence': (abs(model_prediction) + sentiment_confidence) / 2,
            'model_prediction': model_prediction,
            'sentiment_score': sentiment_score,
            'sentiment_confidence': sentiment_confidence
        }
    
    def _apply_risk_adjustments(
        self,
        base_signal: Dict[str, Any],
        risk_metrics: Dict[str, float],
        portfolio_state: PortfolioState
    ) -> Dict[str, Any]:
        """Apply risk-based adjustments to the signal."""
        adjusted_signal = base_signal.copy()
        
        # Check risk limits
        if risk_metrics['risk_score'] > self.max_risk_score:
            adjusted_signal['action'] = 'HOLD'
            adjusted_signal['reasoning'] = 'Risk score too high'
        
        # Check drawdown limits
        if risk_metrics['max_drawdown'] < -self.max_drawdown:
            adjusted_signal['action'] = 'HOLD'
            adjusted_signal['reasoning'] = 'Maximum drawdown exceeded'
        
        # Check Sharpe ratio
        if risk_metrics['sharpe_ratio'] < self.target_sharpe_ratio:
            adjusted_signal['confidence'] *= 0.8  # Reduce confidence
            adjusted_signal['reasoning'] = 'Sharpe ratio below target'
        
        # Check portfolio risk
        if risk_metrics['portfolio_risk'] > 0.8:
            adjusted_signal['action'] = 'HOLD'
            adjusted_signal['reasoning'] = 'Portfolio risk too high'
        
        return adjusted_signal
    
    def _calculate_position_size(
        self,
        signal: Dict[str, Any],
        risk_metrics: Dict[str, float],
        portfolio_state: PortfolioState
    ) -> float:
        """Calculate risk-adjusted position size."""
        if signal['action'] == 'HOLD':
            return 0.0
        
        # Base position size from signal confidence
        base_size = signal['confidence'] * self.max_position_size
        
        # Risk adjustment
        risk_adjustment = 1 - risk_metrics['risk_score']
        risk_adjusted_size = base_size * risk_adjustment
        
        # Kelly criterion adjustment
        kelly_fraction = self._calculate_kelly_fraction(signal, risk_metrics)
        kelly_adjusted_size = risk_adjusted_size * kelly_fraction
        
        # Portfolio risk adjustment
        portfolio_adjustment = 1 - risk_metrics['portfolio_risk']
        final_size = kelly_adjusted_size * portfolio_adjustment
        
        # Ensure within limits
        final_size = min(final_size, self.max_position_size)
        final_size = max(final_size, 0.0)
        
        return final_size
    
    def _calculate_kelly_fraction(
        self,
        signal: Dict[str, Any],
        risk_metrics: Dict[str, float]
    ) -> float:
        """Calculate Kelly criterion fraction."""
        try:
            # Estimate win probability from model prediction
            win_prob = (signal['model_prediction'] + 1) / 2  # Convert to 0-1 range
            win_amount = abs(signal['model_prediction']) * 0.1  # Expected return
            loss_amount = risk_metrics['var_95']  # Expected loss
            
            kelly_fraction = self.kelly_criterion.calculate_kelly_fraction(
                win_prob, win_amount, abs(loss_amount)
            )
            
            return min(kelly_fraction, 0.25)  # Cap at 25%
            
        except Exception as e:
            logger.warning(f"Kelly criterion calculation failed: {e}")
            return 0.1  # Default conservative fraction
    
    def _create_final_signal(
        self,
        symbol: str,
        signal: Dict[str, Any],
        position_size: float,
        model_prediction: float,
        sentiment_data: Dict[str, float],
        risk_metrics: Dict[str, float]
    ) -> TradingSignal:
        """Create the final trading signal."""
        return TradingSignal(
            symbol=symbol,
            action=signal['action'],
            confidence=signal['confidence'],
            position_size=position_size,
            risk_score=risk_metrics['risk_score'],
            expected_return=model_prediction * 0.1,  # Convert to expected return
            max_loss=risk_metrics['var_95'],
            sharpe_ratio=risk_metrics['sharpe_ratio'],
            kelly_fraction=self._calculate_kelly_fraction(signal, risk_metrics),
            sentiment_score=sentiment_data.get('overall_sentiment', 0.0),
            model_prediction=model_prediction,
            timestamp=datetime.now(),
            reasoning=signal.get('reasoning', 'Normal signal generation')
        )
    
    def _create_hold_signal(self, symbol: str, reason: str) -> TradingSignal:
        """Create a HOLD signal."""
        return TradingSignal(
            symbol=symbol,
            action='HOLD',
            confidence=0.0,
            position_size=0.0,
            risk_score=0.0,
            expected_return=0.0,
            max_loss=0.0,
            sharpe_ratio=0.0,
            kelly_fraction=0.0,
            sentiment_score=0.0,
            model_prediction=0.0,
            timestamp=datetime.now(),
            reasoning=reason
        )
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_portfolio_risk(self, portfolio_state: PortfolioState) -> float:
        """Calculate portfolio-level risk."""
        if not portfolio_state.positions:
            return 0.0
        
        # Simple portfolio risk calculation
        position_weights = [pos['weight'] for pos in portfolio_state.positions.values()]
        return np.std(position_weights) if len(position_weights) > 1 else 0.0
    
    def _calculate_market_risk(self, market_data: pd.DataFrame) -> float:
        """Calculate market-level risk."""
        if len(market_data) < 2:
            return 0.5
        
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Normalize volatility to 0-1 scale
        return min(volatility * 10, 1.0)
    
    def _calculate_risk_score(
        self, 
        volatility: float, 
        max_drawdown: float, 
        portfolio_risk: float
    ) -> float:
        """Calculate overall risk score."""
        # Weighted combination of risk factors
        volatility_score = min(volatility * 2, 1.0)
        drawdown_score = min(abs(max_drawdown) * 2, 1.0)
        
        risk_score = (volatility_score * 0.4 + 
                     drawdown_score * 0.4 + 
                     portfolio_risk * 0.2)
        
        return min(risk_score, 1.0)
    
    def _log_decision(self, signal: TradingSignal, risk_metrics: Dict[str, float]):
        """Log trading decision for monitoring."""
        decision_log = {
            'timestamp': signal.timestamp,
            'symbol': signal.symbol,
            'action': signal.action,
            'confidence': signal.confidence,
            'position_size': signal.position_size,
            'risk_score': signal.risk_score,
            'sharpe_ratio': signal.sharpe_ratio,
            'reasoning': signal.reasoning
        }
        
        self.performance_history.append(decision_log)
        
        # Keep only last 1000 decisions
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        logger.info(f"Trading decision: {signal.symbol} {signal.action} "
                   f"(confidence: {signal.confidence:.2f}, "
                   f"position: {signal.position_size:.2f}, "
                   f"risk: {signal.risk_score:.2f})")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from decision history."""
        if not self.performance_history:
            return {}
        
        df = pd.DataFrame(self.performance_history)
        
        return {
            'total_decisions': len(df),
            'buy_decisions': len(df[df['action'] == 'BUY']),
            'sell_decisions': len(df[df['action'] == 'SELL']),
            'hold_decisions': len(df[df['action'] == 'HOLD']),
            'avg_confidence': df['confidence'].mean(),
            'avg_risk_score': df['risk_score'].mean(),
            'avg_sharpe_ratio': df['sharpe_ratio'].mean(),
            'recent_decisions': df.tail(10).to_dict('records')
        }
    
    def get_risk_alerts(self) -> List[Dict[str, Any]]:
        """Get current risk alerts."""
        return self.risk_alerts.copy()


# Global decision engine instance
decision_engine = RiskAdjustedDecisionEngine()


async def generate_trading_signal(
    symbol: str,
    market_data: pd.DataFrame,
    features: pd.DataFrame,
    portfolio_state: PortfolioState
) -> TradingSignal:
    """
    Convenience function to generate trading signals.
    
    Args:
        symbol: Stock symbol
        market_data: Recent market data
        features: Engineered features
        portfolio_state: Current portfolio state
        
    Returns:
        TradingSignal with risk-adjusted metrics
    """
    return await decision_engine.generate_trading_signal(
        symbol, market_data, features, portfolio_state
    )
