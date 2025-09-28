#!/usr/bin/env python3
"""
Unified Decision Engine for the four-model architecture.

This engine coordinates all four models:
1. Sentiment Analysis Model (25% input weight)
2. Quantitative Risk Model (25% input weight)
3. ML Ensemble Model (35% input weight)
4. RL Decider Agent (Final decision maker)

Provides comprehensive trading decisions with full transparency and risk management.
"""

import asyncio
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.models.sentiment_model import SentimentAnalysisModel
from src.models.quantitative_model import QuantitativeRiskModel  
from src.models.ml_ensemble_model import MLEnsembleModel
from src.models.trained_ml_ensemble import TrainedMLEnsembleModel
from src.models.rl_decider_agent import RLDeciderAgent
from src.models.base_models import ModelOutput, ModelValidator, RiskAdjuster

logger = logging.getLogger(__name__)


class FourModelDecisionEngine:
    """Unified decision engine with four-model architecture"""
    
    def __init__(self):
        # Initialize four models
        self.sentiment_model = SentimentAnalysisModel()
        self.quantitative_model = QuantitativeRiskModel()
        # Use trained ML ensemble model instead of the original
        self.ml_ensemble_model = TrainedMLEnsembleModel()
        self.rl_decider_agent = RLDeciderAgent()
        
        # Model weights for transparency (RL agent uses these as inputs)
        self.model_input_weights = {
            'sentiment': 0.25,
            'quantitative': 0.25, 
            'ml_ensemble': 0.35,
            'rl_final_decision': 1.0  # Final decision maker
        }
        
        # Decision history for learning and monitoring
        self.decision_history = []
        self.performance_metrics = {
            'total_decisions': 0,
            'model_accuracy': {
                'sentiment': 0.0, 
                'quantitative': 0.0, 
                'ml_ensemble': 0.0,
                'rl_agent': 0.0
            },
            'rl_learning_progress': 0.0,
            'overall_accuracy': 0.0
        }
        
        # Risk management
        self.risk_validator = ModelValidator()
        self.risk_adjuster = RiskAdjuster()
        
        # System status
        self.is_initialized = False
        self.last_decision_time = None
        
    async def initialize_models(self, training_data: Optional[Dict[str, pd.DataFrame]] = None):
        """Initialize all models with training data if provided"""
        try:
            logger.info("Initializing four-model decision engine...")
            
            # Train ML ensemble model if training data provided
            if training_data and 'ml_ensemble' in training_data:
                ml_data = training_data['ml_ensemble']
                features = ml_data['features']
                targets = ml_data['targets']
                
                self.ml_ensemble_model.train_ensemble(features, targets)
                logger.info("ML ensemble model trained successfully")
            
            self.is_initialized = True
            logger.info("Four-model decision engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def generate_trading_decision(self, symbol: str, market_data: pd.DataFrame, 
                                      features: pd.DataFrame, portfolio_state: Dict) -> Dict[str, Any]:
        """Generate comprehensive trading decision using four-model architecture"""
        
        try:
            if not self.is_initialized:
                await self.initialize_models()
            
            # Phase 1: Get outputs from first three models
            logger.info(f"Generating decision for {symbol} using four-model architecture")
            
            sentiment_output = await self.sentiment_model.predict(symbol, market_data, features)
            quantitative_output = self.quantitative_model.predict(symbol, market_data, features)
            ml_output = self.ml_ensemble_model.predict(symbol, market_data, features)
            
            # Phase 2: Validate model outputs
            model_outputs = [sentiment_output, quantitative_output, ml_output]
            validation_results = self._validate_model_outputs(model_outputs)
            
            # Phase 3: Prepare market features for RL agent
            market_features = self._extract_market_features(market_data, features)
            market_features['symbol'] = symbol
            
            # Phase 4: Get final decision from RL agent
            rl_decision = self.rl_decider_agent.predict(
                sentiment_output, quantitative_output, ml_output, 
                market_features, portfolio_state
            )
            
            # Phase 5: Apply risk adjustments
            risk_adjusted_decision = self._apply_risk_adjustments(rl_decision, quantitative_output)
            
            # Phase 6: Compile comprehensive decision output
            final_decision = self._compile_final_decision(
                symbol, sentiment_output, quantitative_output, 
                ml_output, risk_adjusted_decision, market_features, 
                portfolio_state, validation_results
            )
            
            # Phase 7: Log decision for monitoring and learning
            self._log_decision(final_decision)
            
            # Update performance metrics
            self._update_performance_metrics(final_decision)
            
            self.last_decision_time = datetime.now()
            
            return final_decision
            
        except Exception as e:
            logger.error(f"Failed to generate trading decision for {symbol}: {e}")
            return self._create_error_decision(symbol, str(e))
    
    def _validate_model_outputs(self, model_outputs: List[ModelOutput]) -> Dict[str, Any]:
        """Validate model outputs for consistency and quality"""
        try:
            validation_results = {
                'all_valid': True,
                'individual_validations': [],
                'consistency_check': {},
                'quality_scores': {}
            }
            
            # Individual validation
            for i, output in enumerate(model_outputs):
                is_valid = self.risk_validator.validate_output(output)
                validation_results['individual_validations'].append(is_valid)
                
                if not is_valid:
                    validation_results['all_valid'] = False
            
            # Consistency check
            if len(model_outputs) >= 2:
                consistency = self.risk_validator.check_model_consistency(
                    {f'model_{i}': output for i, output in enumerate(model_outputs)}
                )
                validation_results['consistency_check'] = consistency
            
            # Quality scores
            for i, output in enumerate(model_outputs):
                quality_score = self._calculate_output_quality(output)
                validation_results['quality_scores'][f'model_{i}'] = quality_score
            
            return validation_results
            
        except Exception as e:
            logger.warning(f"Failed to validate model outputs: {e}")
            return {
                'all_valid': False,
                'individual_validations': [False, False, False],
                'consistency_check': {'consistent': False, 'agreement': 0.0},
                'quality_scores': {'model_0': 0.0, 'model_1': 0.0, 'model_2': 0.0}
            }
    
    def _calculate_output_quality(self, output: ModelOutput) -> float:
        """Calculate quality score for model output"""
        try:
            # Base quality from confidence
            confidence_quality = output.confidence
            
            # Reasoning quality (length and detail)
            reasoning_quality = min(1.0, len(output.reasoning) / 100)
            
            # Metrics quality (number of metrics)
            metrics_quality = min(1.0, len(output.metrics) / 10)
            
            # Combined quality score
            quality_score = (
                0.5 * confidence_quality +
                0.3 * reasoning_quality +
                0.2 * metrics_quality
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Failed to calculate output quality: {e}")
            return 0.0
    
    def _extract_market_features(self, market_data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, float]:
        """Extract key market features for RL agent"""
        try:
            # Price momentum
            price_momentum = 0.0
            if len(market_data) >= 20:
                price_momentum = (market_data['Close'].iloc[-1] / market_data['Close'].iloc[-20] - 1)
            
            # Volume analysis
            volume_ratio = 1.0
            if len(market_data) >= 20:
                avg_volume = market_data['Volume'].rolling(20).mean().iloc[-1]
                current_volume = market_data['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volatility
            market_volatility = 0.2
            if len(market_data) >= 20:
                returns = market_data['Close'].pct_change().dropna()
                market_volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.2
            
            # Sector performance (simplified - would need sector index)
            sector_performance = 0.0
            
            return {
                'price_momentum': float(price_momentum),
                'volume_ratio': float(volume_ratio),
                'market_volatility': float(market_volatility),
                'sector_performance': float(sector_performance)
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract market features: {e}")
            return {
                'price_momentum': 0.0,
                'volume_ratio': 1.0,
                'market_volatility': 0.2,
                'sector_performance': 0.0
            }
    
    def _apply_risk_adjustments(self, rl_decision: ModelOutput, 
                              quantitative_output: ModelOutput) -> ModelOutput:
        """Apply additional risk adjustments to RL decision"""
        try:
            # Extract risk metrics
            risk_metrics = quantitative_output.metrics
            
            # Apply risk adjustment
            risk_adjusted_decision = self.risk_adjuster.apply_risk_adjustment(
                rl_decision, risk_metrics
            )
            
            return risk_adjusted_decision
            
        except Exception as e:
            logger.warning(f"Failed to apply risk adjustments: {e}")
            return rl_decision
    
    def _compile_final_decision(self, symbol: str, sentiment_output: ModelOutput, 
                              quantitative_output: ModelOutput, ml_output: ModelOutput,
                              rl_decision: ModelOutput, market_features: Dict, 
                              portfolio_state: Dict, validation_results: Dict) -> Dict[str, Any]:
        """Compile comprehensive final decision"""
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'final_decision': {
                'action': self._signal_to_action(rl_decision.signal),
                'signal_strength': rl_decision.signal,
                'confidence': rl_decision.confidence,
                'position_size': rl_decision.metrics.get('position_size', 0.0),
                'reasoning': rl_decision.reasoning
            },
            'model_inputs': {
                'sentiment_analysis': {
                    'signal': sentiment_output.signal,
                    'confidence': sentiment_output.confidence,
                    'reasoning': sentiment_output.reasoning,
                    'key_metrics': sentiment_output.metrics,
                    'weight': self.model_input_weights['sentiment']
                },
                'quantitative_risk': {
                    'signal': quantitative_output.signal,
                    'confidence': quantitative_output.confidence,
                    'reasoning': quantitative_output.reasoning,
                    'key_metrics': {
                        'sharpe_ratio': quantitative_output.metrics.get('sharpe_ratio', 0.0),
                        'mar_ratio': quantitative_output.metrics.get('mar_ratio', 0.0),
                        'alpha': quantitative_output.metrics.get('alpha', 0.0),
                        'beta': quantitative_output.metrics.get('beta', 1.0)
                    },
                    'weight': self.model_input_weights['quantitative']
                },
                'ml_ensemble': {
                    'signal': ml_output.signal,
                    'confidence': ml_output.confidence,
                    'reasoning': ml_output.reasoning,
                    'individual_models': ml_output.metrics,
                    'weight': self.model_input_weights['ml_ensemble']
                }
            },
            'rl_decision_details': {
                'q_values': rl_decision.metrics.get('q_values', []),
                'risk_adjusted_q_values': rl_decision.metrics.get('risk_adjusted_q_values', []),
                'epsilon': rl_decision.metrics.get('epsilon', 0.0),
                'risk_factors': rl_decision.metrics.get('risk_factors', {}),
                'learning_stats': rl_decision.metrics.get('learning_stats', {})
            },
            'risk_assessment': {
                'portfolio_risk': portfolio_state.get('portfolio_risk', 0.0),
                'position_risk': quantitative_output.metrics.get('volatility', 0.0),
                'market_risk': market_features.get('market_volatility', 0.0),
                'overall_risk_score': self._calculate_overall_risk_score(quantitative_output, portfolio_state)
            },
            'validation_results': validation_results,
            'market_context': market_features,
            'portfolio_context': portfolio_state
        }
    
    def _signal_to_action(self, signal: float) -> str:
        """Convert signal strength to action"""
        if signal > 0.3:
            return 'BUY'
        elif signal < -0.3:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_overall_risk_score(self, quantitative_output: ModelOutput, 
                                    portfolio_state: Dict) -> float:
        """Calculate overall risk score"""
        try:
            # Extract risk metrics
            sharpe_ratio = quantitative_output.metrics.get('sharpe_ratio', 0.0)
            max_drawdown = abs(quantitative_output.metrics.get('max_drawdown', 0.0))
            volatility = quantitative_output.metrics.get('volatility', 0.2)
            
            # Portfolio risk
            portfolio_risk = portfolio_state.get('portfolio_risk', 0.3)
            
            # Calculate risk score (0-1, where 1 is highest risk)
            risk_score = (
                0.3 * min(1.0, volatility / 0.4) +
                0.3 * min(1.0, max_drawdown / 0.3) +
                0.2 * min(1.0, portfolio_risk / 0.5) +
                0.2 * max(0.0, (1.0 - min(1.0, sharpe_ratio / 2.0)))
            )
            
            return max(0.0, min(1.0, risk_score))
            
        except Exception as e:
            logger.warning(f"Failed to calculate overall risk score: {e}")
            return 0.5
    
    def _log_decision(self, decision: Dict[str, Any]):
        """Log decision for monitoring and learning"""
        try:
            # Create decision record
            decision_record = {
                'timestamp': decision['timestamp'],
                'symbol': decision['symbol'],
                'action': decision['final_decision']['action'],
                'signal_strength': decision['final_decision']['signal_strength'],
                'confidence': decision['final_decision']['confidence'],
                'position_size': decision['final_decision']['position_size'],
                'model_inputs': decision['model_inputs'],
                'risk_assessment': decision['risk_assessment']
            }
            
            # Add to history
            self.decision_history.append(decision_record)
            
            # Keep only recent history
            if len(self.decision_history) > 1000:
                self.decision_history = self.decision_history[-1000:]
            
            logger.info(f"Logged decision for {decision['symbol']}: {decision['final_decision']['action']} "
                       f"(confidence: {decision['final_decision']['confidence']:.2%})")
            
        except Exception as e:
            logger.warning(f"Failed to log decision: {e}")
    
    def _update_performance_metrics(self, decision: Dict[str, Any]):
        """Update performance metrics"""
        try:
            self.performance_metrics['total_decisions'] += 1
            
            # Update model accuracies (simplified - would track actual performance)
            for model_name in self.performance_metrics['model_accuracy'].keys():
                # This would be updated based on actual performance tracking
                pass
            
            # Update RL learning progress
            rl_stats = decision['rl_decision_details'].get('learning_stats', {})
            if rl_stats:
                self.performance_metrics['rl_learning_progress'] = rl_stats.get('total_decisions', 0)
            
        except Exception as e:
            logger.warning(f"Failed to update performance metrics: {e}")
    
    def _create_error_decision(self, symbol: str, error_message: str) -> Dict[str, Any]:
        """Create error decision with standardized format"""
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'final_decision': {
                'action': 'HOLD',
                'signal_strength': 0.0,
                'confidence': 0.0,
                'position_size': 0.0,
                'reasoning': f"Decision engine error: {error_message}"
            },
            'model_inputs': {},
            'rl_decision_details': {},
            'risk_assessment': {},
            'validation_results': {'all_valid': False},
            'market_context': {},
            'portfolio_context': {},
            'error': error_message
        }
    
    def update_models_with_outcome(self, decision_record: Dict, outcome: Dict):
        """Update all models based on trading outcome"""
        try:
            # Update RL agent (primary learning)
            self.rl_decider_agent.learn_from_outcome(
                decision_record, 
                outcome.get('market_return', 0.0),
                outcome.get('portfolio_return', 0.0),
                outcome.get('days_held', 1)
            )
            
            # Update individual models
            feedback = {
                'actual_return': outcome.get('market_return', 0.0),
                'predicted_signal': decision_record.get('signal_strength', 0.0),
                'symbol': decision_record.get('symbol', 'UNKNOWN')
            }
            
            self.sentiment_model.update(feedback)
            self.quantitative_model.update(feedback)
            self.ml_ensemble_model.update(feedback)
            
            # Update performance metrics
            self._update_performance_metrics_from_outcome(decision_record, outcome)
            
            logger.info(f"Updated all models with outcome for {decision_record.get('symbol', 'UNKNOWN')}")
            
        except Exception as e:
            logger.error(f"Failed to update models with outcome: {e}")
    
    def _update_performance_metrics_from_outcome(self, decision_record: Dict, outcome: Dict):
        """Update performance metrics based on actual outcome"""
        try:
            # Calculate if decision was correct
            predicted_action = decision_record.get('action', 'HOLD')
            actual_return = outcome.get('market_return', 0.0)
            
            # Determine actual action based on return
            if actual_return > 0.02:
                actual_action = 'BUY'
            elif actual_return < -0.02:
                actual_action = 'SELL'
            else:
                actual_action = 'HOLD'
            
            is_correct = predicted_action == actual_action
            
            # Update overall accuracy
            if is_correct:
                self.performance_metrics['overall_accuracy'] = (
                    (self.performance_metrics['overall_accuracy'] * (self.performance_metrics['total_decisions'] - 1) + 1) /
                    self.performance_metrics['total_decisions']
                )
            else:
                self.performance_metrics['overall_accuracy'] = (
                    (self.performance_metrics['overall_accuracy'] * (self.performance_metrics['total_decisions'] - 1)) /
                    self.performance_metrics['total_decisions']
                )
            
        except Exception as e:
            logger.warning(f"Failed to update performance metrics from outcome: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'is_initialized': self.is_initialized,
            'last_decision_time': self.last_decision_time,
            'performance_metrics': self.performance_metrics.copy(),
            'model_status': {
                'sentiment_model': self.sentiment_model.get_performance_summary(),
                'quantitative_model': self.quantitative_model.get_performance_summary(),
                'ml_ensemble_model': self.ml_ensemble_model.get_model_summary(),
                'rl_decider_agent': self.rl_decider_agent.get_learning_summary()
            },
            'decision_history_size': len(self.decision_history),
            'model_weights': self.model_input_weights.copy()
        }
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent decisions for analysis"""
        return self.decision_history[-limit:] if self.decision_history else []


# Example usage and testing
if __name__ == "__main__":
    async def test_four_model_engine():
        """Test the four-model decision engine"""
        engine = FourModelDecisionEngine()
        
        # Create mock data
        market_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [99, 100, 101, 102, 103],
            'Close': [104, 105, 106, 107, 108],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        features = pd.DataFrame({
            'rsi': [50, 55, 60, 65, 70],
            'macd': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        portfolio_state = {
            'current_position': 0.1,
            'portfolio_risk': 0.3,
            'cash_ratio': 0.7
        }
        
        # Initialize engine
        await engine.initialize_models()
        
        # Generate decision
        decision = await engine.generate_trading_decision("AAPL", market_data, features, portfolio_state)
        
        print("Four-Model Decision Engine Test:")
        print(f"Symbol: {decision['symbol']}")
        print(f"Action: {decision['final_decision']['action']}")
        print(f"Signal Strength: {decision['final_decision']['signal_strength']:.3f}")
        print(f"Confidence: {decision['final_decision']['confidence']:.3f}")
        print(f"Position Size: {decision['final_decision']['position_size']:.2%}")
        print(f"Reasoning: {decision['final_decision']['reasoning']}")
        
        # Show model contributions
        print(f"\nModel Contributions:")
        for model_name, model_data in decision['model_inputs'].items():
            print(f"  {model_name}: Signal={model_data['signal']:.3f}, "
                  f"Confidence={model_data['confidence']:.3f}, Weight={model_data['weight']:.0%}")
        
        # Show system status
        status = engine.get_system_status()
        print(f"\nSystem Status:")
        print(f"  Initialized: {status['is_initialized']}")
        print(f"  Total Decisions: {status['performance_metrics']['total_decisions']}")
        print(f"  Overall Accuracy: {status['performance_metrics']['overall_accuracy']:.2%}")
    
    # Run test
    asyncio.run(test_four_model_engine())
