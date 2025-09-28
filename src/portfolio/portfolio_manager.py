from src.utils.common_imports import *
from typing import Dict, List, Tuple
from src.utils.risk_utils import RiskCalculator
from datetime import datetime
import logging
    from ..data.data_sources import DataManager
    from ..trading.decision_engine import DecisionEngine
    from ..risk.risk_management import RiskManager

"""
Portfolio Manager for QuantAI Trading Bot.

This module manages user portfolios, generates AI recommendations,
and handles trade execution with risk management.
"""


# Import existing modules
try:
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    logging.warning("Some modules not available, using fallback implementations")

class PortfolioManager:
    """Advanced Portfolio Manager for QuantAI Trading Bot"""
    
    def __init__(self, db, data_manager=None):
        self.db = db
        self.data_manager = data_manager or self._create_fallback_data_manager()
        self.decision_engine = self._create_decision_engine()
        self.risk_manager = self._create_risk_manager()
        self.logger = setup_logger()
        self.symbols = ['AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL']
    
    def _create_fallback_data_manager(self):
        """Create fallback data manager for development"""
        class FallbackDataManager:
            def get_current_price(self, symbol):
                # Mock prices for development
                prices = {'AAPL': 150.0, 'GOOGL': 2800.0, 'AMZN': 3200.0, 'META': 300.0, 'NVDA': 400.0}
                return {'price': prices.get(symbol, 100.0)}
            
            def get_historical_data(self, symbol, days=100):
                # Mock historical data
                dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
                prices = 100 + np.random.randn(days).cumsum()
                return pd.DataFrame({
                    'close': prices,
                    'volume': np.random.randint(1000000, 10000000, days)
                }, index=dates)
        
        return FallbackDataManager()
    
    def _create_decision_engine(self):
        """Create decision engine with fallback"""
        class FallbackDecisionEngine:
            def prepare_features(self, data):
                # Mock feature preparation
                return pd.DataFrame({
                    'rsi': np.random.uniform(20, 80, len(data)),
                    'macd': np.random.uniform(-2, 2, len(data)),
                    'bb_position': np.random.uniform(0, 1, len(data))
                })
            
            def predict(self, features):
                # Mock prediction
                return {
                    'signal': np.random.choice(['BUY', 'SELL', 'HOLD']),
                    'confidence': np.random.uniform(0.5, 0.9),
                    'target_price': features['close'].iloc[-1] * np.random.uniform(0.95, 1.05),
                    'expected_return': np.random.uniform(-0.05, 0.1),
                    'reasoning': 'Technical analysis indicates favorable conditions'
                }
        
        return FallbackDecisionEngine()
    
    def _create_risk_manager(self):
        """Create risk manager with fallback"""
        class FallbackRiskManager:
            def calculate_position_risk(self, symbol, portfolio, target_price):
                return np.random.uniform(0.1, 0.8)
            
            risk_calc = RiskCalculator()
# Use: risk_calc.calculate_kelly_fraction(win_prob, avg_win, avg_loss)
                return max(0, (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win)
        
        return FallbackRiskManager()
    
    def get_user_portfolio_summary(self, user_id: int) -> Dict:
        """Get comprehensive portfolio summary"""
        portfolio = self.db.get_user_portfolio(user_id)
        
        if not portfolio:
            return {
                'total_value': 0,
                'total_pnl': 0,
                'total_return_pct': 0,
                'positions': [],
                'risk_metrics': {}
            }
        
        # Calculate summary metrics
        total_value = sum(p['market_value'] or 0 for p in portfolio)
        total_pnl = sum(p['unrealized_pnl'] or 0 for p in portfolio)
        total_invested = sum(p['shares'] * p['avg_purchase_price'] for p in portfolio)
        total_return_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        # Calculate risk metrics
        risk_metrics = self._calculate_portfolio_risk(portfolio)
        
        return {
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_return_pct': total_return_pct,
            'positions': portfolio,
            'risk_metrics': risk_metrics,
            'last_updated': datetime.now()
        }
    
    def _calculate_portfolio_risk(self, portfolio: List[Dict]) -> Dict:
        """Calculate portfolio risk metrics"""
        if not portfolio:
            return {}
        
        try:
            # Get historical data for risk calculation
            symbols = [p['symbol'] for p in portfolio]
            weights = []
            
            total_value = sum(p['market_value'] or 0 for p in portfolio)
            for p in portfolio:
                weight = (p['market_value'] or 0) / total_value if total_value > 0 else 0
                weights.append(weight)
            
            # Calculate basic risk metrics
            portfolio_beta = sum(w * self._get_stock_beta(s) for w, s in zip(weights, symbols))
            concentration_risk = max(weights) if weights else 0
            
            return {
                'portfolio_beta': portfolio_beta,
                'concentration_risk': concentration_risk,
                'diversification_score': 1 - sum(w**2 for w in weights),
                'risk_level': 'Low' if concentration_risk < 0.3 else 'Medium' if concentration_risk < 0.5 else 'High'
            }
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}")
            return {}
    
    def _get_stock_beta(self, symbol: str) -> float:
        """Get stock beta (simplified calculation)"""
        beta_map = {'AAPL': 1.2, 'GOOGL': 1.1, 'AMZN': 1.3, 'META': 1.4, 'NVDA': 1.8}
        return beta_map.get(symbol, 1.0)
    
    def update_all_portfolios(self):
        """Update current prices for all portfolios"""
        try:
            # Fetch current prices
            current_prices = {}
            for symbol in self.symbols:
                try:
                    data = self.data_manager.get_current_price(symbol)
                    current_prices[symbol] = data['price']
                except Exception as e:
                    self.logger.error(f"Failed to get price for {symbol}: {e}")
            
            if current_prices:
                self.db.update_portfolio_prices(current_prices)
                self.logger.info(f"Updated prices for {len(current_prices)} symbols")
        except Exception as e:
            self.logger.error(f"Failed to update portfolios: {e}")
    
    def generate_recommendations(self, user_id: int) -> List[Dict]:
        """Generate AI-powered trading recommendations"""
        try:
            portfolio = self.db.get_user_portfolio(user_id)
            recommendations = []
            
            for symbol in self.symbols:
                # Get latest market data and features
                market_data = self.data_manager.get_historical_data(symbol, days=100)
                features = self.decision_engine.prepare_features(market_data)
                
                # Generate prediction
                prediction = self.decision_engine.predict(features)
                signal = prediction['signal']
                confidence = prediction['confidence']
                
                # Risk assessment
                risk_score = self.risk_manager.calculate_position_risk(
                    symbol, portfolio, prediction['target_price']
                )
                
                # Generate recommendation
                if signal != 'HOLD' and confidence > 0.6:
                    recommended_shares = self._calculate_position_size(
                        user_id, symbol, signal, prediction, risk_score
                    )
                    
                    recommendation = {
                        'symbol': symbol,
                        'type': signal,
                        'shares': recommended_shares,
                        'target_price': prediction['target_price'],
                        'confidence': confidence,
                        'risk_score': risk_score,
                        'reasoning': prediction.get('reasoning', ''),
                        'expected_return': prediction.get('expected_return', 0)
                    }
                    
                    recommendations.append(recommendation)
                    
                    # Save to database
                    rec_id = self.db.save_recommendation(
                        user_id=user_id,
                        symbol=symbol,
                        rec_type=signal,
                        shares=recommended_shares,
                        price=prediction['target_price'],
                        confidence=confidence,
                        model_prediction=prediction['target_price'],
                        features=features.to_dict(),
                        risk_score=risk_score
                    )
                    recommendation['recommendation_id'] = rec_id
            
            self.logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations for user {user_id}: {e}")
            return []
    
    def _calculate_position_size(self, user_id: int, symbol: str, signal: str, 
                               prediction: Dict, risk_score: float) -> float:
        """Calculate optimal position size using risk management"""
        try:
            portfolio = self.db.get_user_portfolio(user_id)
            total_value = sum(p['market_value'] or 0 for p in portfolio)
            
            # Base position size using Kelly Criterion
            kelly_fraction = self.risk_manager.calculate_kelly_fraction(
                win_prob=prediction['confidence'],
                avg_win=prediction.get('expected_return', 0.1),
                avg_loss=0.05
            )
            
            # Adjust for risk score
            risk_adjusted_fraction = kelly_fraction * (1 - risk_score)
            
            # Position size limits
            max_position_value = total_value * 0.2  # Max 20% per position
            target_position_value = total_value * min(risk_adjusted_fraction, 0.2)
            
            # Calculate shares
            shares = target_position_value / prediction['target_price']
            
            return max(0, shares)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
    
    def execute_recommendation(self, recommendation_id: int, user_id: int, 
                             accept: bool, actual_shares: float = 0, 
                             actual_price: float = 0) -> Dict:
        """Execute user's decision on recommendation"""
        try:
            # Record user response
            self.db.record_user_response(
                recommendation_id, accept, actual_shares, actual_price
            )
            
            if accept and actual_shares > 0:
                # Get recommendation details
                cursor = self.db.connection.cursor(dictionary=True)
                cursor.execute(
                    "SELECT * FROM recommendations_feedback WHERE recommendation_id = %s",
                    (recommendation_id,)
                )
                rec = cursor.fetchone()
                cursor.close()
                
                if rec:
                    # Update user portfolio
                    if rec['recommendation_type'] == 'BUY':
                        self.db.add_portfolio_position(
                            user_id, rec['symbol'], actual_shares, actual_price
                        )
                    elif rec['recommendation_type'] == 'SELL':
                        self._handle_sell_order(user_id, rec['symbol'], actual_shares, actual_price)
                    
                    self.logger.info(f"Executed {rec['recommendation_type']} order for user {user_id}")
                    
                    return {
                        'success': True,
                        'message': f"Successfully executed {rec['recommendation_type']} order",
                        'shares': actual_shares,
                        'price': actual_price
                    }
            
            return {'success': True, 'message': 'Recommendation response recorded'}
            
        except Exception as e:
            self.logger.error(f"Failed to execute recommendation: {e}")
            return {'success': False, 'message': str(e)}
    
    def _handle_sell_order(self, user_id: int, symbol: str, shares: float, price: float):
        """Handle sell order execution"""
        cursor = self.db.connection.cursor()
        
        # Get current position
        cursor.execute(
            "SELECT shares, avg_purchase_price FROM user_portfolios WHERE user_id = %s AND symbol = %s",
            (user_id, symbol)
        )
        position = cursor.fetchone()
        
        if position and position[0] >= shares:
            current_shares, avg_price = position
            
            # Calculate realized P&L
            realized_pnl = shares * (price - avg_price)
            
            # Update position
            new_shares = current_shares - shares
            if new_shares > 0:
                cursor.execute(
                    "UPDATE user_portfolios SET shares = %s, realized_pnl = realized_pnl + %s WHERE user_id = %s AND symbol = %s",
                    (new_shares, realized_pnl, user_id, symbol)
                )
            else:
                # Close position
                cursor.execute(
                    "DELETE FROM user_portfolios WHERE user_id = %s AND symbol = %s",
                    (user_id, symbol)
                )
            
            self.db.connection.commit()
        
        cursor.close()

    def get_performance_analytics(self, user_id: int, days: int = 30) -> Dict:
        """Get portfolio performance analytics"""
        try:
            # Get recommendations and outcomes
            cursor = self.db.connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_recommendations,
                    SUM(user_accepted) as accepted_recommendations,
                    AVG(CASE WHEN user_accepted = TRUE THEN outcome_return_pct END) as avg_return,
                    AVG(confidence_score) as avg_confidence,
                    COUNT(CASE WHEN outcome_return_pct > 0 THEN 1 END) as winning_trades,
                    COUNT(CASE WHEN outcome_evaluated = TRUE THEN 1 END) as evaluated_trades
                FROM recommendations_feedback 
                WHERE user_id = %s AND recommendation_timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            """, (user_id, days))
            
            analytics = cursor.fetchone()
            cursor.close()
            
            # Calculate win rate
            win_rate = 0
            if analytics['evaluated_trades'] > 0:
                win_rate = analytics['winning_trades'] / analytics['evaluated_trades'] * 100
            
            # Get portfolio summary
            portfolio_summary = self.get_user_portfolio_summary(user_id)
            
            return {
                'total_recommendations': analytics['total_recommendations'] or 0,
                'acceptance_rate': (
                    (analytics['accepted_recommendations'] or 0) / 
                    max(analytics['total_recommendations'] or 1, 1) * 100
                ),
                'avg_return': analytics['avg_return'] or 0,
                'win_rate': win_rate,
                'portfolio_value': portfolio_summary['total_value'],
                'portfolio_return': portfolio_summary['total_return_pct'],
                'avg_confidence': analytics['avg_confidence'] or 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance analytics: {e}")
            return {}
