"""
User Journey Flow System for QuantAI Portfolio Manager.

This module implements the complete user journey from registration
to portfolio management with AI recommendations and learning.
"""

from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
try:
    from ..database.db_manager import QuantAIDatabase
    from ..portfolio.portfolio_manager import PortfolioManager
    from ..training.feedback_trainer import ReinforcementFeedbackTrainer
except ImportError:
    # Fallback for direct execution
    from database.db_manager import QuantAIDatabase
    from portfolio.portfolio_manager import PortfolioManager
    from training.feedback_trainer import ReinforcementFeedbackTrainer

class UserJourneyManager:
    """Manages the complete user journey flow"""
    
    def __init__(self, db: QuantAIDatabase, portfolio_manager: PortfolioManager):
        self.db = db
        self.portfolio_manager = portfolio_manager
        self.feedback_trainer = ReinforcementFeedbackTrainer(db)
        self.logger = logging.getLogger(__name__)
        self.symbols = ['AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL']
    
    def complete_user_onboarding(self, user_id: int) -> Dict:
        """Complete user onboarding process"""
        try:
            # Get user statistics
            user_stats = self.db.get_user_statistics(user_id)
            
            # Create initial portfolio recommendations
            initial_recommendations = self._generate_initial_recommendations(user_id)
            
            # Set up user preferences
            preferences = self._setup_user_preferences(user_id)
            
            return {
                'user_id': user_id,
                'onboarding_complete': True,
                'user_stats': user_stats,
                'initial_recommendations': initial_recommendations,
                'preferences': preferences,
                'next_steps': [
                    'Add your first stock position',
                    'Review AI recommendations',
                    'Set up risk preferences',
                    'Enable learning mode'
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to complete user onboarding: {e}")
            return {'error': str(e)}
    
    def _generate_initial_recommendations(self, user_id: int) -> List[Dict]:
        """Generate initial recommendations for new users"""
        recommendations = []
        
        for symbol in self.symbols:
            # Simple initial recommendation based on risk tolerance
            user_info = self.db.get_user_statistics(user_id)['user_info']
            risk_tolerance = user_info.get('risk_tolerance', 'Moderate')
            
            # Adjust confidence based on risk tolerance
            confidence_map = {
                'Conservative': 0.6,
                'Moderate': 0.7,
                'Aggressive': 0.8
            }
            
            confidence = confidence_map.get(risk_tolerance, 0.7)
            
            recommendation = {
                'symbol': symbol,
                'type': 'BUY',
                'confidence': confidence,
                'reasoning': f'Initial recommendation for {risk_tolerance} risk tolerance',
                'target_price': 100.0,  # Placeholder
                'shares': 10.0  # Placeholder
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _setup_user_preferences(self, user_id: int) -> Dict:
        """Set up user preferences and settings"""
        preferences = {
            'notification_frequency': 'daily',
            'risk_alerts': True,
            'learning_mode': True,
            'personalized_recommendations': True,
            'portfolio_rebalancing': 'monthly',
            'preferred_symbols': self.symbols
        }
        
        # Update user preferences in database
        cursor = self.db.connection.cursor()
        query = "UPDATE users SET preferences = %s WHERE user_id = %s"
        cursor.execute(query, (str(preferences), user_id))
        self.db.connection.commit()
        cursor.close()
        
        return preferences
    
    def execute_daily_workflow(self, user_id: int) -> Dict:
        """Execute daily workflow for active users"""
        try:
            # Update portfolio prices
            self.portfolio_manager.update_all_portfolios()
            
            # Update user portfolio value
            self.db.update_user_portfolio_value(user_id)
            
            # Generate daily recommendations
            recommendations = self.portfolio_manager.generate_recommendations(user_id)
            
            # Get personalized recommendations
            personalized_recs = self.feedback_trainer.generate_personalized_recommendations(
                user_id, {}
            )
            
            # Check for risk alerts
            risk_alerts = self._check_risk_alerts(user_id)
            
            # Generate daily summary
            daily_summary = self._generate_daily_summary(user_id)
            
            return {
                'user_id': user_id,
                'date': datetime.now().isoformat(),
                'recommendations': recommendations,
                'personalized_recommendations': personalized_recs,
                'risk_alerts': risk_alerts,
                'daily_summary': daily_summary,
                'workflow_complete': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute daily workflow: {e}")
            return {'error': str(e)}
    
    def _check_risk_alerts(self, user_id: int) -> List[Dict]:
        """Check for risk alerts and warnings"""
        alerts = []
        portfolio = self.db.get_user_portfolio(user_id)
        
        for position in portfolio:
            # Check for high drawdown
            if position['unrealized_pnl'] and position['unrealized_pnl'] < -1000:
                alerts.append({
                    'type': 'high_drawdown',
                    'symbol': position['symbol'],
                    'message': f"High drawdown detected for {position['symbol']}",
                    'severity': 'warning'
                })
            
            # Check for concentration risk
            total_value = sum(p['market_value'] or 0 for p in portfolio)
            position_weight = (position['market_value'] or 0) / total_value if total_value > 0 else 0
            
            if position_weight > 0.3:  # 30% concentration
                alerts.append({
                    'type': 'concentration_risk',
                    'symbol': position['symbol'],
                    'message': f"High concentration in {position['symbol']} ({position_weight:.1%})",
                    'severity': 'warning'
                })
        
        return alerts
    
    def _generate_daily_summary(self, user_id: int) -> Dict:
        """Generate daily portfolio summary"""
        portfolio_summary = self.portfolio_manager.get_user_portfolio_summary(user_id)
        
        return {
            'total_value': portfolio_summary['total_value'],
            'daily_pnl': portfolio_summary['total_pnl'],
            'daily_return': portfolio_summary['total_return_pct'],
            'positions_count': len(portfolio_summary['positions']),
            'risk_level': portfolio_summary['risk_metrics'].get('risk_level', 'Unknown')
        }
    
    def process_user_feedback(self, user_id: int, recommendation_id: int, 
                            feedback_data: Dict) -> Dict:
        """Process user feedback and update learning models"""
        try:
            # Record user feedback
            self.db.record_user_feedback(
                recommendation_id,
                feedback_data.get('score', 5),
                feedback_data.get('comment', '')
            )
            
            # Update learning model
            outcome = feedback_data.get('outcome', 0)
            self.feedback_trainer.update_model_weights(recommendation_id, outcome)
            
            # Generate learning insights
            insights = self._generate_learning_insights(user_id)
            
            return {
                'feedback_recorded': True,
                'model_updated': True,
                'insights': insights,
                'next_recommendations': 'Will be more personalized based on your feedback'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process user feedback: {e}")
            return {'error': str(e)}
    
    def _generate_learning_insights(self, user_id: int) -> Dict:
        """Generate insights from user behavior analysis"""
        try:
            user_analysis = self.feedback_trainer.analyze_user_behavior_patterns(user_id)
            
            if 'error' in user_analysis:
                return {'message': 'Insufficient data for insights'}
            
            insights = {
                'acceptance_rate': user_analysis['acceptance_rate'],
                'avg_performance': user_analysis['avg_performance'],
                'preferred_symbols': list(user_analysis['symbol_preferences'].keys()),
                'confidence_threshold': user_analysis['confidence_thresholds'],
                'recommendations': []
            }
            
            # Generate personalized recommendations
            if user_analysis['acceptance_rate'] > 0.7:
                insights['recommendations'].append(
                    "You have a high acceptance rate! The system will prioritize similar recommendations."
                )
            
            if user_analysis['avg_performance'] > 0.05:
                insights['recommendations'].append(
                    "Your portfolio is performing well! Consider increasing position sizes."
                )
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate learning insights: {e}")
            return {'error': str(e)}
    
    def generate_weekly_report(self, user_id: int) -> Dict:
        """Generate comprehensive weekly report"""
        try:
            # Get performance analytics
            analytics = self.portfolio_manager.get_performance_analytics(user_id, days=7)
            
            # Get learning performance
            learning_report = self.feedback_trainer.get_model_performance_report()
            
            # Get user behavior analysis
            user_analysis = self.feedback_trainer.analyze_user_behavior_patterns(user_id)
            
            # Generate recommendations for next week
            next_week_recommendations = self._generate_next_week_recommendations(user_id)
            
            return {
                'user_id': user_id,
                'week_ending': datetime.now().isoformat(),
                'performance_analytics': analytics,
                'learning_performance': learning_report,
                'user_behavior': user_analysis,
                'next_week_recommendations': next_week_recommendations,
                'report_generated': True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate weekly report: {e}")
            return {'error': str(e)}
    
    def _generate_next_week_recommendations(self, user_id: int) -> List[Dict]:
        """Generate recommendations for the next week"""
        recommendations = []
        
        # Get user's current portfolio
        portfolio = self.db.get_user_portfolio(user_id)
        portfolio_symbols = [p['symbol'] for p in portfolio]
        
        # Generate recommendations for symbols not in portfolio
        for symbol in self.symbols:
            if symbol not in portfolio_symbols:
                recommendation = {
                    'symbol': symbol,
                    'action': 'BUY',
                    'reasoning': 'Diversification opportunity',
                    'priority': 'medium'
                }
                recommendations.append(recommendation)
        
        return recommendations
