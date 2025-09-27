#!/usr/bin/env python3
"""
RL Decider Agent for the four-model decision engine.

This is the final decision maker that combines inputs from:
- Sentiment Analysis Model (25% weight)
- Quantitative Risk Model (25% weight)
- ML Ensemble Model (35% weight)

Uses Deep Q-Network (DQN) with risk-adjusted Q-values for final trading decisions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging

from .base_models import BaseModel, ModelOutput

logger = logging.getLogger(__name__)


class RLDeciderState:
    """State representation for RL agent"""
    
    def __init__(self, sentiment_output: ModelOutput, quantitative_output: ModelOutput, 
                 ml_output: ModelOutput, market_features: Dict[str, float], 
                 portfolio_state: Dict[str, float]):
        # Model outputs (12 features)
        self.sentiment_signal = sentiment_output.signal
        self.sentiment_confidence = sentiment_output.confidence
        self.quantitative_signal = quantitative_output.signal
        self.quantitative_confidence = quantitative_output.confidence
        self.ml_signal = ml_output.signal
        self.ml_confidence = ml_output.confidence
        
        # Key risk metrics (6 features)
        self.sharpe_ratio = quantitative_output.metrics.get('sharpe_ratio', 0.0)
        self.mar_ratio = quantitative_output.metrics.get('mar_ratio', 0.0)
        self.alpha = quantitative_output.metrics.get('alpha', 0.0)
        self.beta = quantitative_output.metrics.get('beta', 1.0)
        self.volatility = quantitative_output.metrics.get('volatility', 0.0)
        self.max_drawdown = quantitative_output.metrics.get('max_drawdown', 0.0)
        
        # Market features (4 features)
        self.price_momentum = market_features.get('price_momentum', 0.0)
        self.volume_ratio = market_features.get('volume_ratio', 1.0)
        self.market_volatility = market_features.get('market_volatility', 0.0)
        self.sector_performance = market_features.get('sector_performance', 0.0)
        
        # Portfolio state (3 features)
        self.current_position = portfolio_state.get('current_position', 0.0)
        self.portfolio_risk = portfolio_state.get('portfolio_risk', 0.0)
        self.cash_ratio = portfolio_state.get('cash_ratio', 1.0)
        
    def to_array(self) -> np.ndarray:
        """Convert state to numpy array for neural network"""
        return np.array([
            self.sentiment_signal, self.sentiment_confidence,
            self.quantitative_signal, self.quantitative_confidence,
            self.ml_signal, self.ml_confidence,
            self.sharpe_ratio, self.mar_ratio, self.alpha, self.beta,
            self.volatility, self.max_drawdown,
            self.price_momentum, self.volume_ratio, 
            self.market_volatility, self.sector_performance,
            self.current_position, self.portfolio_risk, self.cash_ratio
        ], dtype=np.float32)


class DQNNetwork(nn.Module):
    """Deep Q-Network for RL decision making"""
    
    def __init__(self, state_size: int = 19, hidden_size: int = 128, action_size: int = 3):
        super(DQNNetwork, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(state_size, hidden_size)
        self.input_activation = nn.ReLU()
        self.input_dropout = nn.Dropout(0.2)
        
        # Hidden layers
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden1_activation = nn.ReLU()
        self.hidden1_dropout = nn.Dropout(0.2)
        
        self.hidden2 = nn.Linear(hidden_size, 64)
        self.hidden2_activation = nn.ReLU()
        
        # Output layer
        self.output_layer = nn.Linear(64, action_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = self.input_dropout(self.input_activation(self.input_layer(x)))
        x = self.hidden1_dropout(self.hidden1_activation(self.hidden1(x)))
        x = self.hidden2_activation(self.hidden2(x))
        x = self.output_layer(x)
        return x


class RLDeciderAgent(BaseModel):
    """Model 4: Reinforcement Learning agent for final decision making"""
    
    def __init__(self, state_size: int = 19, action_size: int = 3, learning_rate: float = 0.001):
        super().__init__("RLDeciderAgent", weight=1.0)  # Final decision maker
        
        self.state_size = state_size
        self.action_size = action_size  # 0=SELL, 1=HOLD, 2=BUY
        self.learning_rate = learning_rate
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Exploration parameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Neural networks
        self.q_network = DQNNetwork(state_size, 128, action_size)
        self.target_network = DQNNetwork(state_size, 128, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Risk integration weights for explicit risk consideration
        self.risk_weights = {
            'sharpe_ratio': 0.25,
            'mar_ratio': 0.25,
            'alpha': 0.20,
            'beta': 0.15,
            'volatility': 0.15
        }
        
        # Performance tracking
        self.decision_history = deque(maxlen=1000)
        self.learning_stats = {
            'total_decisions': 0,
            'correct_decisions': 0,
            'total_reward': 0.0,
            'epsilon_history': [],
            'loss_history': [],
            'q_value_history': []
        }
        
        # Model input weights (from other models)
        self.model_input_weights = {
            'sentiment': 0.25,
            'quantitative': 0.25,
            'ml_ensemble': 0.35
        }
    
    def predict(self, sentiment_output: ModelOutput, quantitative_output: ModelOutput, 
                ml_output: ModelOutput, market_features: Dict[str, float], 
                portfolio_state: Dict[str, float]) -> ModelOutput:
        """Generate final trading decision using RL agent"""
        
        try:
            # Create state representation
            state = RLDeciderState(sentiment_output, quantitative_output, ml_output, 
                                 market_features, portfolio_state)
            
            # Get Q-values from neural network
            state_tensor = torch.FloatTensor(state.to_array()).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            
            # Risk-adjusted Q-values
            risk_adjusted_q_values = self._apply_risk_adjustment(q_values, quantitative_output)
            
            # Select action (epsilon-greedy for training)
            if random.random() > self.epsilon:
                action = risk_adjusted_q_values.argmax().item()
            else:
                action = random.randrange(self.action_size)
            
            # Convert action to trading signal
            action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            signal_map = {0: -1.0, 1: 0.0, 2: 1.0}
            
            decision_action = action_map[action]
            signal_strength = signal_map[action]
            
            # Calculate confidence based on Q-value spread and risk metrics
            confidence = self._calculate_rl_confidence(risk_adjusted_q_values, quantitative_output)
            
            # Calculate position size using risk-adjusted approach
            position_size = self._calculate_risk_adjusted_position_size(
                signal_strength, confidence, quantitative_output, portfolio_state
            )
            
            reasoning = self._generate_rl_reasoning(
                sentiment_output, quantitative_output, ml_output, 
                action, q_values, risk_adjusted_q_values
            )
            
            metrics = {
                'q_values': q_values.squeeze().tolist(),
                'risk_adjusted_q_values': risk_adjusted_q_values.squeeze().tolist(),
                'selected_action': action,
                'epsilon': self.epsilon,
                'model_inputs': {
                    'sentiment_signal': sentiment_output.signal,
                    'quantitative_signal': quantitative_output.signal, 
                    'ml_signal': ml_output.signal
                },
                'risk_factors': self._extract_risk_factors(quantitative_output),
                'position_size': position_size,
                'learning_stats': self.learning_stats.copy()
            }
            
            # Store decision for learning
            decision_record = {
                'state': state.to_array(),
                'action': action,
                'q_values': q_values.squeeze().detach().numpy(),
                'timestamp': datetime.now(),
                'symbol': market_features.get('symbol', 'UNKNOWN')
            }
            self.decision_history.append(decision_record)
            
            # Update performance tracking
            self.learning_stats['total_decisions'] += 1
            
            return ModelOutput(
                signal=signal_strength,
                confidence=confidence,
                reasoning=reasoning,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"RL agent prediction failed: {e}")
            return self._create_error_output(f"RL agent failed: {e}")
    
    def _apply_risk_adjustment(self, q_values: torch.Tensor, 
                             quantitative_output: ModelOutput) -> torch.Tensor:
        """Apply risk-based adjustment to Q-values"""
        try:
            risk_metrics = quantitative_output.metrics
            
            # Calculate risk penalty/bonus
            sharpe_factor = np.tanh(risk_metrics.get('sharpe_ratio', 0.0) / 2)  # -1 to 1
            mar_factor = np.tanh(risk_metrics.get('mar_ratio', 0.0))  # -1 to 1
            alpha_factor = np.tanh(risk_metrics.get('alpha', 0.0) * 10)  # -1 to 1
            
            # Composite risk score (-1 to 1, where 1 is best risk profile)
            risk_score = (
                self.risk_weights['sharpe_ratio'] * sharpe_factor +
                self.risk_weights['mar_ratio'] * mar_factor +
                self.risk_weights['alpha'] * alpha_factor
            )
            
            # Apply risk adjustment (reduce extreme actions if risk is poor)
            risk_multiplier = 1 + 0.5 * risk_score  # 0.5 to 1.5 multiplier
            
            adjusted_q_values = q_values.clone()
            adjusted_q_values[0, 0] *= risk_multiplier  # SELL adjustment
            adjusted_q_values[0, 2] *= risk_multiplier  # BUY adjustment
            # HOLD unchanged (conservative default)
            
            return adjusted_q_values
            
        except Exception as e:
            logger.warning(f"Failed to apply risk adjustment: {e}")
            return q_values
    
    def _calculate_rl_confidence(self, q_values: torch.Tensor, 
                               quantitative_output: ModelOutput) -> float:
        """Calculate RL confidence based on Q-value spread and risk metrics"""
        try:
            # Q-value spread (higher spread = higher confidence)
            q_spread = torch.max(q_values) - torch.min(q_values)
            q_spread_confidence = min(1.0, q_spread.item() / 2.0)  # Normalize
            
            # Risk-based confidence
            sharpe_ratio = quantitative_output.metrics.get('sharpe_ratio', 0.0)
            max_drawdown = abs(quantitative_output.metrics.get('max_drawdown', 0.0))
            
            risk_confidence = 1.0 - min(1.0, (max_drawdown + (1.0 - min(1.0, sharpe_ratio))) / 2)
            
            # Model agreement confidence (simplified)
            model_confidence = (
                self.model_input_weights['sentiment'] * quantitative_output.confidence +
                self.model_input_weights['quantitative'] * quantitative_output.confidence +
                self.model_input_weights['ml_ensemble'] * quantitative_output.confidence
            )
            
            # Combined confidence
            combined_confidence = (
                0.4 * q_spread_confidence +
                0.3 * risk_confidence +
                0.3 * model_confidence
            )
            
            return max(0.0, min(1.0, combined_confidence))
            
        except Exception as e:
            logger.warning(f"Failed to calculate RL confidence: {e}")
            return 0.5
    
    def _calculate_risk_adjusted_position_size(self, signal_strength: float, confidence: float,
                                             quantitative_output: ModelOutput, 
                                             portfolio_state: Dict[str, float]) -> float:
        """Calculate risk-adjusted position size"""
        try:
            # Base position size from signal strength and confidence
            base_position = abs(signal_strength) * confidence
            
            # Risk adjustment
            sharpe_ratio = quantitative_output.metrics.get('sharpe_ratio', 0.0)
            volatility = quantitative_output.metrics.get('volatility', 0.2)
            max_drawdown = abs(quantitative_output.metrics.get('max_drawdown', 0.0))
            
            # Risk multiplier (lower risk = higher position size)
            risk_multiplier = 1.0
            if sharpe_ratio > 1.5:
                risk_multiplier *= 1.2
            elif sharpe_ratio < 0.5:
                risk_multiplier *= 0.8
            
            if volatility < 0.15:
                risk_multiplier *= 1.1
            elif volatility > 0.35:
                risk_multiplier *= 0.9
            
            if max_drawdown < 0.05:
                risk_multiplier *= 1.1
            elif max_drawdown > 0.15:
                risk_multiplier *= 0.8
            
            # Portfolio constraints
            current_position = portfolio_state.get('current_position', 0.0)
            max_position = 0.3  # 30% max position
            
            # Calculate final position size
            adjusted_position = base_position * risk_multiplier
            final_position = min(adjusted_position, max_position)
            
            # Consider current position
            if current_position > 0 and signal_strength < 0:  # Selling
                final_position = min(final_position, current_position)
            elif current_position < 0 and signal_strength > 0:  # Buying
                final_position = min(final_position, abs(current_position))
            
            return max(0.0, min(0.3, final_position))  # Clamp to [0, 0.3]
            
        except Exception as e:
            logger.warning(f"Failed to calculate position size: {e}")
            return 0.1  # Default 10% position
    
    def _generate_rl_reasoning(self, sentiment_output: ModelOutput, quantitative_output: ModelOutput,
                             ml_output: ModelOutput, action: int, q_values: torch.Tensor,
                             risk_adjusted_q_values: torch.Tensor) -> str:
        """Generate human-readable RL reasoning"""
        try:
            action_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            action_name = action_names[action]
            
            reasoning_parts = [f"RL agent recommends {action_name} action"]
            
            # Q-value analysis
            q_values_list = q_values.squeeze().tolist()
            risk_adjusted_list = risk_adjusted_q_values.squeeze().tolist()
            
            max_q_idx = np.argmax(q_values_list)
            max_risk_q_idx = np.argmax(risk_adjusted_list)
            
            if max_q_idx != max_risk_q_idx:
                reasoning_parts.append("Risk adjustment changed the optimal action")
            
            # Model input analysis
            sentiment_signal = sentiment_output.signal
            quantitative_signal = quantitative_output.signal
            ml_signal = ml_output.signal
            
            # Count model agreement
            positive_models = sum(1 for s in [sentiment_signal, quantitative_signal, ml_signal] if s > 0.1)
            negative_models = sum(1 for s in [sentiment_signal, quantitative_signal, ml_signal] if s < -0.1)
            
            if positive_models >= 2:
                reasoning_parts.append("Majority of models show bullish signals")
            elif negative_models >= 2:
                reasoning_parts.append("Majority of models show bearish signals")
            else:
                reasoning_parts.append("Models show mixed signals")
            
            # Risk assessment
            sharpe_ratio = quantitative_output.metrics.get('sharpe_ratio', 0.0)
            if sharpe_ratio > 1.5:
                reasoning_parts.append("Strong risk-adjusted returns support the decision")
            elif sharpe_ratio < 0.5:
                reasoning_parts.append("Poor risk-adjusted returns suggest caution")
            
            # Exploration vs exploitation
            if self.epsilon > 0.1:
                reasoning_parts.append("Agent is still exploring (learning phase)")
            else:
                reasoning_parts.append("Agent is exploiting learned knowledge")
            
            return ". ".join(reasoning_parts) + "."
            
        except Exception as e:
            logger.warning(f"Failed to generate RL reasoning: {e}")
            return "RL agent decision based on learned patterns and risk assessment."
    
    def _extract_risk_factors(self, quantitative_output: ModelOutput) -> Dict[str, float]:
        """Extract key risk factors for analysis"""
        try:
            return {
                'sharpe_ratio': quantitative_output.metrics.get('sharpe_ratio', 0.0),
                'mar_ratio': quantitative_output.metrics.get('mar_ratio', 0.0),
                'alpha': quantitative_output.metrics.get('alpha', 0.0),
                'beta': quantitative_output.metrics.get('beta', 1.0),
                'volatility': quantitative_output.metrics.get('volatility', 0.2),
                'max_drawdown': quantitative_output.metrics.get('max_drawdown', 0.0)
            }
        except Exception as e:
            logger.warning(f"Failed to extract risk factors: {e}")
            return {}
    
    def learn_from_outcome(self, decision_record: Dict, market_return: float, 
                          portfolio_return: float, days_held: int):
        """Learn from trading decision outcome"""
        try:
            # Calculate reward based on risk-adjusted performance
            base_reward = portfolio_return
            
            # Risk adjustment to reward
            sharpe_bonus = 0.1 if decision_record.get('sharpe_ratio', 0) > 1.5 else 0
            drawdown_penalty = -0.1 if decision_record.get('max_drawdown', 0) < -0.1 else 0
            
            risk_adjusted_reward = base_reward + sharpe_bonus + drawdown_penalty
            
            # Store experience for replay learning
            experience = (
                decision_record['state'],
                decision_record['action'], 
                risk_adjusted_reward,
                decision_record.get('next_state'),
                decision_record.get('done', False)
            )
            
            self.memory.append(experience)
            
            # Update learning stats
            self.learning_stats['total_reward'] += risk_adjusted_reward
            if risk_adjusted_reward > 0:
                self.learning_stats['correct_decisions'] += 1
            
            # Perform experience replay learning
            if len(self.memory) > self.batch_size:
                self._experience_replay()
            
            logger.info(f"RL agent learned from outcome: reward={risk_adjusted_reward:.3f}, "
                       f"total_reward={self.learning_stats['total_reward']:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to learn from outcome: {e}")
    
    def _experience_replay(self):
        """Experience replay learning"""
        try:
            if len(self.memory) < self.batch_size:
                return
            
            # Sample batch from memory
            batch = random.sample(self.memory, self.batch_size)
            
            # Extract batch data
            states = torch.FloatTensor([e[0] for e in batch])
            actions = torch.LongTensor([e[1] for e in batch])
            rewards = torch.FloatTensor([e[2] for e in batch])
            next_states = torch.FloatTensor([e[3] for e in batch if e[3] is not None])
            dones = torch.BoolTensor([e[4] for e in batch])
            
            # Current Q-values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q-values
            next_q_values = torch.zeros(self.batch_size)
            if len(next_states) > 0:
                next_q_values = self.target_network(next_states).max(1)[0].detach()
            
            # Target Q-values
            target_q_values = rewards + (0.99 * next_q_values * ~dones)
            
            # Calculate loss
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.learning_stats['epsilon_history'].append(self.epsilon)
            
            # Store learning metrics
            self.learning_stats['loss_history'].append(loss.item())
            self.learning_stats['q_value_history'].append(current_q_values.mean().item())
            
            # Update target network periodically
            if self.learning_stats['total_decisions'] % 100 == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                logger.info(f"Updated target network at decision {self.learning_stats['total_decisions']}")
            
        except Exception as e:
            logger.error(f"Experience replay failed: {e}")
    
    def _create_error_output(self, error_message: str) -> ModelOutput:
        """Create error output with standardized format"""
        return ModelOutput(
            signal=0.0,
            confidence=0.0,
            reasoning=f"RL agent error: {error_message}",
            metrics={'error': error_message}
        )
    
    def get_confidence(self) -> float:
        """Get model confidence score"""
        if self.learning_stats['total_decisions'] > 0:
            return self.learning_stats['correct_decisions'] / self.learning_stats['total_decisions']
        return 0.0
    
    def update(self, feedback: Dict[str, Any]) -> None:
        """Update model based on feedback and outcomes"""
        try:
            # Extract feedback data
            actual_return = feedback.get('actual_return', 0.0)
            predicted_signal = feedback.get('predicted_signal', 0.0)
            symbol = feedback.get('symbol', 'UNKNOWN')
            days_held = feedback.get('days_held', 1)
            
            # Create decision record
            decision_record = {
                'state': feedback.get('state', np.zeros(19)),
                'action': feedback.get('action', 1),  # Default to HOLD
                'sharpe_ratio': feedback.get('sharpe_ratio', 0.0),
                'max_drawdown': feedback.get('max_drawdown', 0.0),
                'next_state': feedback.get('next_state'),
                'done': feedback.get('done', False)
            }
            
            # Learn from outcome
            self.learn_from_outcome(decision_record, actual_return, actual_return, days_held)
            
            # Update performance metrics
            self.performance_metrics['total_predictions'] += 1
            if actual_return > 0:
                self.performance_metrics['correct_predictions'] += 1
            
            # Calculate new accuracy
            total = self.performance_metrics['total_predictions']
            correct = self.performance_metrics['correct_predictions']
            self.performance_metrics['accuracy'] = correct / total if total > 0 else 0.0
            
            logger.info(f"Updated RL agent performance: {self.performance_metrics['accuracy']:.2%}")
            
        except Exception as e:
            logger.error(f"Failed to update RL agent: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary"""
        return {
            'model_name': self.model_name,
            'weight': self.weight,
            'is_trained': self.is_trained,
            'learning_stats': self.learning_stats.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'decision_history_size': len(self.decision_history)
        }


# Example usage and testing
if __name__ == "__main__":
    def test_rl_decider_agent():
        """Test the RL Decider Agent"""
        agent = RLDeciderAgent()
        
        # Create mock model outputs
        sentiment_output = ModelOutput(
            signal=0.3,
            confidence=0.8,
            reasoning="Positive sentiment detected",
            metrics={'news_sentiment': 0.4, 'social_sentiment': 0.2}
        )
        
        quantitative_output = ModelOutput(
            signal=0.2,
            confidence=0.7,
            reasoning="Good risk-adjusted returns",
            metrics={
                'sharpe_ratio': 1.8,
                'mar_ratio': 0.9,
                'alpha': 0.05,
                'beta': 1.1,
                'volatility': 0.25,
                'max_drawdown': -0.08
            }
        )
        
        ml_output = ModelOutput(
            signal=0.4,
            confidence=0.75,
            reasoning="ML models show bullish signals",
            metrics={'ensemble_prediction': 0.35, 'model_agreement': 0.8}
        )
        
        # Create mock market features and portfolio state
        market_features = {
            'price_momentum': 0.05,
            'volume_ratio': 1.2,
            'market_volatility': 0.18,
            'sector_performance': 0.03,
            'symbol': 'AAPL'
        }
        
        portfolio_state = {
            'current_position': 0.1,
            'portfolio_risk': 0.3,
            'cash_ratio': 0.7
        }
        
        # Test prediction
        result = agent.predict(sentiment_output, quantitative_output, ml_output, 
                             market_features, portfolio_state)
        
        print("RL Decider Agent Test:")
        print(f"Signal: {result.signal:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Position Size: {result.metrics['position_size']:.2%}")
        print(f"Selected Action: {result.metrics['selected_action']}")
        print(f"Epsilon: {result.metrics['epsilon']:.3f}")
        
        # Test learning
        feedback = {
            'actual_return': 0.05,
            'predicted_signal': result.signal,
            'symbol': 'AAPL',
            'days_held': 5,
            'state': result.metrics['model_inputs'],
            'action': result.metrics['selected_action'],
            'sharpe_ratio': 1.8,
            'max_drawdown': -0.08
        }
        
        agent.update(feedback)
        print(f"Updated accuracy: {agent.get_confidence():.2%}")
        
        # Show learning summary
        summary = agent.get_learning_summary()
        print(f"\nLearning Summary:")
        print(f"  Total decisions: {summary['learning_stats']['total_decisions']}")
        print(f"  Correct decisions: {summary['learning_stats']['correct_decisions']}")
        print(f"  Total reward: {summary['learning_stats']['total_reward']:.3f}")
        print(f"  Epsilon: {summary['epsilon']:.3f}")
    
    # Run test
    test_rl_decider_agent()
