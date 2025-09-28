from src.utils.common_imports import *
from ..utils.common_imports import setup_logger, PerformanceMetrics
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime, timedelta
import logging
import warnings
from ..utils.data_processing import data_processor
from ..utils.feature_engineering import feature_engineer
from ..utils.performance_metrics import performance_calculator
from ..utils.risk_utils import risk_calculator
from ..utils.config_manager import config_manager
from ..models.sentiment_model import SentimentAnalysisModel
from ..models.quantitative_model import QuantitativeRiskModel
from ..models.ml_ensemble_model import MLEnsembleModel
from ..models.rl_decider_agent import RLDeciderAgent
from ..decision_engine.four_model_engine import FourModelDecisionEngine
from ..data.data_sources import DataManager
from ..data.feature_engineering import FeatureEngineer
            import json

#!/usr/bin/env python3
"""
Training Pipeline for the four-model decision engine.

This pipeline trains all four models:
1. Sentiment Analysis Model (pre-trained, updates with feedback)
2. Quantitative Risk Model (pre-trained, updates with feedback)
3. ML Ensemble Model (requires training data)
4. RL Decider Agent (requires simulation training)

Provides comprehensive training with validation and performance tracking.
"""

warnings.filterwarnings('ignore')

# Import unified utilities

# Import models

# Legacy imports for compatibility

logger = setup_logger()


class FourModelTrainingPipeline:
    """Training pipeline for four-model architecture"""
    
    def __init__(self):
        # Initialize unified utilities
        self.config = config_manager.get_model_config()
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize individual models
        self.sentiment_model = SentimentAnalysisModel()
        self.quantitative_model = QuantitativeRiskModel()
        self.ml_ensemble_model = MLEnsembleModel()
        self.rl_agent = RLDeciderAgent()
        
        # Legacy components for compatibility
        self.decision_engine = FourModelDecisionEngine()
        self.data_manager = DataManager()
        self.feature_engineer = FeatureEngineer()
        
        # Training configuration using unified config
        self.training_config = {
            'training_period_days': 800,
            'validation_period_days': 200,
            'rl_episodes': 100,
            'rl_simulation_days': 30,
            'min_training_samples': 100,
            'lookback_period': self.config.lookback_period,
            'validation_split': self.config.validation_split,
            'test_split': self.config.test_split
        }
        
        # Model weights for final decision
        self.model_weights = {
            'sentiment': 0.25,
            'quantitative': 0.25,
            'ml_ensemble': 0.35,
            'rl_agent': 1.0  # Final decision maker
        }
        
        # Performance tracking
        self.training_metrics = {
            'training_start_time': None,
            'training_end_time': None,
            'models_trained': [],
            'training_errors': [],
            'validation_results': {},
            'performance_summary': {}
        }
    
    async def train_complete_system(self, symbols: List[str], 
                                   training_period_days: Optional[int] = None,
                                   validation_period_days: Optional[int] = None) -> Dict[str, Any]:
        """Train all four models with historical data"""
        
        # Update configuration if provided
        if training_period_days:
            self.training_config['training_period_days'] = training_period_days
        if validation_period_days:
            self.training_config['validation_period_days'] = validation_period_days
        
        self.training_metrics['training_start_time'] = datetime.now()
        
        print("🚀 Starting Four-Model Training Pipeline")
        print("=" * 60)
        print("Architecture:")
        print("  1️⃣ Sentiment Analysis Model (25% input weight)")
        print("  2️⃣ Quantitative Risk Model (25% input weight)")  
        print("  3️⃣ ML Ensemble Model (35% input weight)")
        print("  4️⃣ RL Decider Agent (Final decision maker)")
        print("=" * 60)
        
        try:
            for symbol in symbols:
                print(f"\n📊 Training models for {symbol}")
                print("-" * 40)
                
                # Step 1: Load historical data
                market_data, features, targets = await self._load_training_data(symbol)
                
                if market_data is None:
                    print(f"❌ No data available for {symbol}")
                    continue
                
                # Step 2: Train ML ensemble model
                print("  🧠 Training ML Ensemble Model...")
                ml_training_success = await self._train_ml_ensemble(symbol, features, targets)
                
                if ml_training_success:
                    print("  ✅ ML Ensemble training completed")
                else:
                    print("  ⚠️ ML Ensemble training failed, using fallback")
                
                # Step 3: Train RL agent with simulation
                print("  🎯 Training RL Decider Agent...")
                rl_training_success = await self._train_rl_agent(symbol, market_data, features)
                
                if rl_training_success:
                    print("  ✅ RL Agent training completed")
                else:
                    print("  ⚠️ RL Agent training failed, using default")
                
                # Step 4: Validate models
                print("  🔍 Validating models...")
                validation_results = await self._validate_models(symbol, market_data, features)
                
                print(f"  📊 Validation results: {validation_results}")
            
            # Step 5: Initialize decision engine
            print("\n🔧 Initializing decision engine...")
            await self.decision_engine.initialize_models()
            
            # Step 6: Generate training summary
            self.training_metrics['training_end_time'] = datetime.now()
            training_summary = self._generate_training_summary()
            
            print(f"\n🎉 Training pipeline completed for {len(symbols)} symbols")
            print("📊 Final training summary:")
            self._print_training_summary()
            
            return training_summary
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            self.training_metrics['training_errors'].append(str(e))
            raise
    
    async def _load_training_data(self, symbol: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load historical data for training"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.training_config['training_period_days'] + self.training_config['validation_period_days'])
            
            # Load market data
            market_data = self.data_manager.get_historical_data(
                symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            )
            
            if market_data.empty:
                logger.warning(f"No market data available for {symbol}")
                return None, None, None
            
            # Engineer features
            features = self.feature_engineer.create_all_features(market_data)
            
            if features.empty:
                logger.warning(f"No features generated for {symbol}")
                return None, None, None
            
            # Create targets (future returns)
            targets = market_data['Close'].pct_change().shift(-5).dropna()  # 5-day future returns
            
            # Align features and targets
            min_length = min(len(features), len(targets))
            features = features.iloc[:min_length]
            targets = targets.iloc[:min_length]
            
            if len(features) < self.training_config['min_training_samples']:
                logger.warning(f"Insufficient training samples for {symbol}: {len(features)}")
                return None, None, None
            
            logger.info(f"Loaded training data for {symbol}: {len(features)} samples")
            return market_data, features, targets
            
        except Exception as e:
            logger.error(f"Failed to load training data for {symbol}: {e}")
            return None, None, None
    
    async def _train_ml_ensemble(self, symbol: str, features: pd.DataFrame, targets: pd.Series) -> bool:
        """Train ML ensemble model"""
        try:
            # Split data for training and validation
            split_point = len(features) - self.training_config['validation_period_days']
            
            if split_point < 100:  # Need minimum samples for training
                logger.warning(f"Insufficient data for ML training: {split_point} samples")
                return False
            
            train_features = features.iloc[:split_point]
            train_targets = targets.iloc[:split_point]
            
            # Train the ensemble model
            self.decision_engine.ml_ensemble_model.train_ensemble(train_features, train_targets)
            
            # Store training success
            self.training_metrics['models_trained'].append(f"{symbol}_ml_ensemble")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train ML ensemble for {symbol}: {e}")
            self.training_metrics['training_errors'].append(f"ML ensemble training failed for {symbol}: {e}")
            return False
    
    async def _train_rl_agent(self, symbol: str, market_data: pd.DataFrame, features: pd.DataFrame) -> bool:
        """Train RL agent using historical simulation"""
        try:
            # Use validation period for RL training
            val_start = len(market_data) - self.training_config['validation_period_days']
            val_data = market_data.iloc[val_start:]
            val_features = features.iloc[val_start:]
            
            if len(val_data) < 20:  # Need minimum data for simulation
                logger.warning(f"Insufficient data for RL training: {len(val_data)} samples")
                return False
            
            # Run RL training episodes
            episodes = self.training_config['rl_episodes']
            simulation_days = min(self.training_config['rl_simulation_days'], len(val_data) - 10)
            
            for episode in range(episodes):
                # Simulate trading environment
                portfolio_state = {'current_position': 0.0, 'portfolio_risk': 0.0, 'cash_ratio': 1.0}
                episode_return = 0.0
                
                for i in range(10, min(10 + simulation_days, len(val_data) - 5)):  # Leave room for lookahead
                    # Get current market slice
                    current_data = val_data.iloc[:i+1]
                    current_features = val_features.iloc[:i+1]
                    
                    # Generate decision
                    decision = await self.decision_engine.generate_trading_decision(
                        symbol, current_data, current_features, portfolio_state
                    )
                    
                    # Simulate outcome (5-day holding period)
                    entry_price = val_data['Close'].iloc[i]
                    exit_price = val_data['Close'].iloc[min(i+5, len(val_data)-1)]
                    
                    # Calculate returns based on action
                    action = decision['final_decision']['action']
                    if action == 'BUY':
                        trade_return = (exit_price - entry_price) / entry_price
                    elif action == 'SELL':
                        trade_return = (entry_price - exit_price) / entry_price
                    else:  # HOLD
                        trade_return = 0.0
                    
                    # Update RL agent with outcome
                    outcome = {
                        'market_return': (exit_price - entry_price) / entry_price,
                        'portfolio_return': trade_return,
                        'days_held': 5
                    }
                    
                    # Create decision record for learning
                    decision_record = {
                        'state': decision.get('rl_decision_details', {}).get('q_values', []),
                        'action': decision['final_decision']['action'],
                        'sharpe_ratio': decision['risk_assessment'].get('overall_risk_score', 0.0),
                        'max_drawdown': decision['risk_assessment'].get('position_risk', 0.0)
                    }
                    
                    self.decision_engine.update_models_with_outcome(decision_record, outcome)
                    
                    episode_return += trade_return
                    
                    # Update portfolio state
                    portfolio_state['current_position'] = decision['final_decision']['position_size']
                
                if episode % 20 == 0:
                    print(f"    Episode {episode}/{episodes}: Return = {episode_return:.3f}, "
                          f"Epsilon = {self.decision_engine.rl_decider_agent.epsilon:.3f}")
            
            # Store training success
            self.training_metrics['models_trained'].append(f"{symbol}_rl_agent")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to train RL agent for {symbol}: {e}")
            self.training_metrics['training_errors'].append(f"RL agent training failed for {symbol}: {e}")
            return False
    
    async def _validate_models(self, symbol: str, market_data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate trained models"""
        try:
            validation_results = {
                'symbol': symbol,
                'validation_date': datetime.now(),
                'model_validation': {},
                'overall_score': 0.0
            }
            
            # Test model predictions
            test_data = market_data.iloc[-10:]  # Last 10 days
            test_features = features.iloc[-10:]
            portfolio_state = {'current_position': 0.0, 'portfolio_risk': 0.3, 'cash_ratio': 0.7}
            
            # Generate test decision
            test_decision = await self.decision_engine.generate_trading_decision(
                symbol, test_data, test_features, portfolio_state
            )
            
            # Validate decision quality
            decision_quality = self._assess_decision_quality(test_decision)
            validation_results['model_validation'] = decision_quality
            
            # Calculate overall score
            overall_score = (
                decision_quality.get('confidence_score', 0.0) +
                decision_quality.get('reasoning_quality', 0.0) +
                decision_quality.get('risk_assessment_quality', 0.0)
            ) / 3
            
            validation_results['overall_score'] = overall_score
            
            # Store validation results
            self.training_metrics['validation_results'][symbol] = validation_results
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate models for {symbol}: {e}")
            return {
                'symbol': symbol,
                'validation_date': datetime.now(),
                'model_validation': {'error': str(e)},
                'overall_score': 0.0
            }
    
    def _assess_decision_quality(self, decision: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality of a trading decision"""
        try:
            quality_metrics = {}
            
            # Confidence score
            confidence = decision['final_decision']['confidence']
            quality_metrics['confidence_score'] = confidence
            
            # Reasoning quality
            reasoning = decision['final_decision']['reasoning']
            reasoning_quality = min(1.0, len(reasoning) / 200)  # Normalize by expected length
            quality_metrics['reasoning_quality'] = reasoning_quality
            
            # Risk assessment quality
            risk_assessment = decision['risk_assessment']
            risk_quality = 1.0 - risk_assessment.get('overall_risk_score', 0.5)  # Lower risk = higher quality
            quality_metrics['risk_assessment_quality'] = risk_quality
            
            # Model agreement quality
            model_inputs = decision['model_inputs']
            if model_inputs:
                signals = [model_data['signal'] for model_data in model_inputs.values()]
                agreement = 1.0 - np.std(signals) / 2  # Lower std = higher agreement
                quality_metrics['model_agreement_quality'] = max(0.0, agreement)
            else:
                quality_metrics['model_agreement_quality'] = 0.0
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Failed to assess decision quality: {e}")
            return {
                'confidence_score': 0.0,
                'reasoning_quality': 0.0,
                'risk_assessment_quality': 0.0,
                'model_agreement_quality': 0.0
            }
    
    def _generate_training_summary(self) -> Dict[str, Any]:
        """Generate comprehensive training summary"""
        try:
            training_duration = None
            if self.training_metrics['training_start_time'] and self.training_metrics['training_end_time']:
                training_duration = (self.training_metrics['training_end_time'] - 
                                   self.training_metrics['training_start_time']).total_seconds()
            
            # Calculate average validation score
            validation_scores = [result['overall_score'] for result in self.training_metrics['validation_results'].values()]
            avg_validation_score = np.mean(validation_scores) if validation_scores else 0.0
            
            # Get system status
            system_status = self.decision_engine.get_system_status()
            
            summary = {
                'training_config': self.training_config.copy(),
                'training_metrics': self.training_metrics.copy(),
                'training_duration_seconds': training_duration,
                'models_trained_count': len(self.training_metrics['models_trained']),
                'validation_results_count': len(self.training_metrics['validation_results']),
                'average_validation_score': avg_validation_score,
                'training_errors_count': len(self.training_metrics['training_errors']),
                'system_status': system_status,
                'training_success': len(self.training_metrics['training_errors']) == 0
            }
            
            self.training_metrics['performance_summary'] = summary
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate training summary: {e}")
            return {'error': str(e)}
    
    def _print_training_summary(self):
        """Print comprehensive training summary"""
        try:
            summary = self.training_metrics.get('performance_summary', {})
            
            if not summary:
                print("  No training summary available")
                return
            
            # Training duration
            duration = summary.get('training_duration_seconds', 0)
            if duration:
                print(f"  ⏱️ Training Duration: {duration:.1f} seconds")
            
            # Models trained
            models_count = summary.get('models_trained_count', 0)
            print(f"  🧠 Models Trained: {models_count}")
            
            # Validation results
            validation_count = summary.get('validation_results_count', 0)
            avg_score = summary.get('average_validation_score', 0.0)
            print(f"  🔍 Validation Results: {validation_count} symbols, avg score: {avg_score:.3f}")
            
            # Training errors
            errors_count = summary.get('training_errors_count', 0)
            if errors_count > 0:
                print(f"  ⚠️ Training Errors: {errors_count}")
            else:
                print(f"  ✅ Training Errors: {errors_count}")
            
            # System status
            system_status = summary.get('system_status', {})
            if system_status:
                performance = system_status.get('performance_metrics', {})
                total_decisions = performance.get('total_decisions', 0)
                overall_accuracy = performance.get('overall_accuracy', 0.0)
                print(f"  📊 System Performance: {total_decisions} decisions, {overall_accuracy:.2%} accuracy")
            
            # Training success
            success = summary.get('training_success', False)
            status_emoji = "✅" if success else "⚠️"
            print(f"  {status_emoji} Training Status: {'SUCCESS' if success else 'PARTIAL SUCCESS'}")
            
        except Exception as e:
            logger.error(f"Failed to print training summary: {e}")
    
    def get_training_report(self) -> Dict[str, Any]:
        """Get comprehensive training report"""
        return {
            'training_config': self.training_config.copy(),
            'training_metrics': self.training_metrics.copy(),
            'system_status': self.decision_engine.get_system_status(),
            'recent_decisions': self.decision_engine.get_recent_decisions(5)
        }
    
    def save_training_results(self, filepath: str):
        """Save training results to file"""
        try:
            
            # Convert datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                return obj
            
            # Get training report
            report = self.get_training_report()
            
            # Convert all datetime objects
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {k: recursive_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(item) for item in obj]
                else:
                    return convert_datetime(obj)
            
            converted_report = recursive_convert(report)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(converted_report, f, indent=2)
            
            logger.info(f"Training results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save training results: {e}")


# Example usage and testing
if __name__ == "__main__":
    async def test_training_pipeline():
        """Test the four-model training pipeline"""
        pipeline = FourModelTrainingPipeline()
        
        # Test with FAANG+ stocks
        symbols = ['AAPL', 'AMZN', 'GOOGL', 'META', 'NVDA']
        
        print("🧪 Testing Four-Model Training Pipeline")
        print("=" * 50)
        
        try:
            # Run training
            training_summary = await pipeline.train_complete_system(
                symbols, 
                training_period_days=400, 
                validation_period_days=100
            )
            
            print(f"\n📋 Training Summary:")
            print(f"  Success: {training_summary.get('training_success', False)}")
            print(f"  Models Trained: {training_summary.get('models_trained_count', 0)}")
            print(f"  Validation Score: {training_summary.get('average_validation_score', 0.0):.3f}")
            
            # Get training report
            report = pipeline.get_training_report()
            print(f"\n📊 Training Report Generated")
            print(f"  System Initialized: {report['system_status']['is_initialized']}")
            print(f"  Total Decisions: {report['system_status']['performance_metrics']['total_decisions']}")
            
            # Save results
            pipeline.save_training_results("training_results.json")
            print(f"\n💾 Training results saved to training_results.json")
            
        except Exception as e:
            print(f"❌ Training pipeline test failed: {e}")
    
    # Run test
    asyncio.run(test_training_pipeline())
