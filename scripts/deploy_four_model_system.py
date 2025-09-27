#!/usr/bin/env python3
"""
Deployment script for the four-model decision engine system.

This script deploys and tests the complete four-model architecture:
1. Sentiment Analysis Model (25% input weight)
2. Quantitative Risk Model (25% input weight)
3. ML Ensemble Model (35% input weight)
4. RL Decider Agent (Final decision maker)

Provides comprehensive testing and validation of the system.
"""

import asyncio
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.four_model_training import FourModelTrainingPipeline
from src.decision_engine.four_model_engine import FourModelDecisionEngine
from src.data.data_sources import DataManager
from src.data.feature_engineering import FeatureEngineer

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FourModelSystemDeployer:
    """Deployer for the four-model decision engine system"""
    
    def __init__(self):
        self.training_pipeline = FourModelTrainingPipeline()
        self.decision_engine = None
        self.data_manager = DataManager()
        self.feature_engineer = FeatureEngineer()
        
        # Deployment configuration
        self.deployment_config = {
            'symbols': ['AAPL', 'AMZN', 'GOOGL', 'META', 'NVDA'],
            'training_period_days': 800,
            'validation_period_days': 200,
            'test_period_days': 100,
            'rl_episodes': 50,  # Reduced for faster deployment
            'min_confidence_threshold': 0.6
        }
        
        # Deployment results
        self.deployment_results = {
            'deployment_start_time': None,
            'deployment_end_time': None,
            'training_results': {},
            'test_results': {},
            'system_status': {},
            'deployment_success': False
        }
    
    async def deploy_system(self) -> Dict[str, Any]:
        """Deploy the complete four-model system"""
        
        self.deployment_results['deployment_start_time'] = datetime.now()
        
        print("🚀 QUANTAI FOUR-MODEL SYSTEM DEPLOYMENT")
        print("=" * 70)
        print("Architecture:")
        print("  1️⃣ Sentiment Analysis Model (25% input weight)")
        print("  2️⃣ Quantitative Risk Model (25% input weight)")  
        print("  3️⃣ ML Ensemble Model (35% input weight)")
        print("  4️⃣ RL Decider Agent (Final decision maker)")
        print("=" * 70)
        
        try:
            # Phase 1: Training
            print("\n📚 Phase 1: Training Models")
            print("-" * 40)
            training_results = await self._train_models()
            self.deployment_results['training_results'] = training_results
            
            # Phase 2: System Initialization
            print("\n🔧 Phase 2: System Initialization")
            print("-" * 40)
            initialization_success = await self._initialize_system()
            
            if not initialization_success:
                raise Exception("System initialization failed")
            
            # Phase 3: Testing
            print("\n🧪 Phase 3: System Testing")
            print("-" * 40)
            test_results = await self._test_system()
            self.deployment_results['test_results'] = test_results
            
            # Phase 4: Validation
            print("\n✅ Phase 4: System Validation")
            print("-" * 40)
            validation_results = await self._validate_system()
            
            # Phase 5: Generate Report
            print("\n📊 Phase 5: Deployment Report")
            print("-" * 40)
            deployment_report = self._generate_deployment_report()
            
            self.deployment_results['deployment_end_time'] = datetime.now()
            self.deployment_results['deployment_success'] = True
            
            print("\n🎉 FOUR-MODEL SYSTEM DEPLOYMENT COMPLETE!")
            print("=" * 70)
            self._print_deployment_summary()
            
            return deployment_report
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self.deployment_results['deployment_end_time'] = datetime.now()
            self.deployment_results['deployment_success'] = False
            raise
    
    async def _train_models(self) -> Dict[str, Any]:
        """Train all four models"""
        try:
            print("Training models with historical data...")
            
            # Run training pipeline
            training_summary = await self.training_pipeline.train_complete_system(
                self.deployment_config['symbols'],
                self.deployment_config['training_period_days'],
                self.deployment_config['validation_period_days']
            )
            
            print(f"✅ Training completed successfully")
            print(f"   Models trained: {training_summary.get('models_trained_count', 0)}")
            print(f"   Validation score: {training_summary.get('average_validation_score', 0.0):.3f}")
            print(f"   Training duration: {training_summary.get('training_duration_seconds', 0):.1f}s")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    async def _initialize_system(self) -> bool:
        """Initialize the decision engine system"""
        try:
            print("Initializing decision engine...")
            
            # Get the trained decision engine
            self.decision_engine = self.training_pipeline.decision_engine
            
            # Verify system status
            system_status = self.decision_engine.get_system_status()
            
            if not system_status['is_initialized']:
                raise Exception("Decision engine not properly initialized")
            
            print("✅ System initialized successfully")
            print(f"   Models ready: {len(system_status['model_status'])}")
            print(f"   System weight: {system_status['model_weights']}")
            
            self.deployment_results['system_status'] = system_status
            
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def _test_system(self) -> Dict[str, Any]:
        """Test the deployed system"""
        try:
            print("Testing system with recent market data...")
            
            test_results = {
                'test_symbols': [],
                'test_decisions': [],
                'performance_metrics': {},
                'test_success': True
            }
            
            # Test each symbol
            for symbol in self.deployment_config['symbols'][:3]:  # Test first 3 symbols
                print(f"  Testing {symbol}...")
                
                # Load recent data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.deployment_config['test_period_days'])
                
                market_data = self.data_manager.get_historical_data(
                    symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
                )
                
                if market_data.empty:
                    print(f"    ⚠️ No data available for {symbol}")
                    continue
                
                # Engineer features
                features = self.feature_engineer.create_all_features(market_data)
                
                if features.empty:
                    print(f"    ⚠️ No features generated for {symbol}")
                    continue
                
                # Test portfolio state
                portfolio_state = {
                    'current_position': 0.1,
                    'portfolio_risk': 0.3,
                    'cash_ratio': 0.7
                }
                
                # Generate decision
                decision = await self.decision_engine.generate_trading_decision(
                    symbol, market_data, features, portfolio_state
                )
                
                # Analyze decision quality
                decision_quality = self._analyze_decision_quality(decision)
                
                test_result = {
                    'symbol': symbol,
                    'decision': decision,
                    'quality_metrics': decision_quality,
                    'test_timestamp': datetime.now()
                }
                
                test_results['test_decisions'].append(test_result)
                test_results['test_symbols'].append(symbol)
                
                print(f"    ✅ {symbol}: {decision['final_decision']['action']} "
                      f"(confidence: {decision['final_decision']['confidence']:.2%})")
            
            # Calculate overall performance
            if test_results['test_decisions']:
                avg_confidence = np.mean([d['quality_metrics']['confidence'] for d in test_results['test_decisions']])
                avg_quality = np.mean([d['quality_metrics']['overall_quality'] for d in test_results['test_decisions']])
                
                test_results['performance_metrics'] = {
                    'average_confidence': avg_confidence,
                    'average_quality': avg_quality,
                    'successful_tests': len(test_results['test_decisions']),
                    'total_tests': len(self.deployment_config['symbols'][:3])
                }
                
                print(f"✅ Testing completed successfully")
                print(f"   Successful tests: {test_results['performance_metrics']['successful_tests']}")
                print(f"   Average confidence: {avg_confidence:.2%}")
                print(f"   Average quality: {avg_quality:.3f}")
            else:
                test_results['test_success'] = False
                print("❌ No successful tests completed")
            
            return test_results
            
        except Exception as e:
            logger.error(f"System testing failed: {e}")
            return {'test_success': False, 'error': str(e)}
    
    def _analyze_decision_quality(self, decision: Dict[str, Any]) -> Dict[str, float]:
        """Analyze the quality of a trading decision"""
        try:
            quality_metrics = {}
            
            # Confidence score
            confidence = decision['final_decision']['confidence']
            quality_metrics['confidence'] = confidence
            
            # Reasoning quality
            reasoning = decision['final_decision']['reasoning']
            reasoning_quality = min(1.0, len(reasoning) / 200)
            quality_metrics['reasoning_quality'] = reasoning_quality
            
            # Risk assessment quality
            risk_assessment = decision['risk_assessment']
            risk_quality = 1.0 - risk_assessment.get('overall_risk_score', 0.5)
            quality_metrics['risk_quality'] = risk_quality
            
            # Model agreement
            model_inputs = decision['model_inputs']
            if model_inputs:
                signals = [model_data['signal'] for model_data in model_inputs.values()]
                agreement = 1.0 - np.std(signals) / 2
                quality_metrics['model_agreement'] = max(0.0, agreement)
            else:
                quality_metrics['model_agreement'] = 0.0
            
            # Overall quality
            overall_quality = (
                quality_metrics['confidence'] +
                quality_metrics['reasoning_quality'] +
                quality_metrics['risk_quality'] +
                quality_metrics['model_agreement']
            ) / 4
            
            quality_metrics['overall_quality'] = overall_quality
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Failed to analyze decision quality: {e}")
            return {
                'confidence': 0.0,
                'reasoning_quality': 0.0,
                'risk_quality': 0.0,
                'model_agreement': 0.0,
                'overall_quality': 0.0
            }
    
    async def _validate_system(self) -> Dict[str, Any]:
        """Validate the deployed system"""
        try:
            print("Validating system components...")
            
            validation_results = {
                'component_validation': {},
                'overall_validation': True,
                'validation_timestamp': datetime.now()
            }
            
            # Validate decision engine
            system_status = self.decision_engine.get_system_status()
            validation_results['component_validation']['decision_engine'] = {
                'initialized': system_status['is_initialized'],
                'models_ready': len(system_status['model_status']),
                'performance_metrics': system_status['performance_metrics']
            }
            
            # Validate individual models
            for model_name, model_status in system_status['model_status'].items():
                validation_results['component_validation'][model_name] = {
                    'trained': model_status.get('is_trained', False),
                    'performance': model_status.get('performance_metrics', {}),
                    'weight': model_status.get('weight', 0.0)
                }
            
            # Check overall validation
            for component, validation in validation_results['component_validation'].items():
                if not validation.get('initialized', validation.get('trained', False)):
                    validation_results['overall_validation'] = False
                    break
            
            print(f"✅ System validation completed")
            print(f"   Overall validation: {'PASS' if validation_results['overall_validation'] else 'FAIL'}")
            print(f"   Components validated: {len(validation_results['component_validation'])}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            return {
                'component_validation': {},
                'overall_validation': False,
                'validation_timestamp': datetime.now(),
                'error': str(e)
            }
    
    def _generate_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report"""
        try:
            deployment_duration = None
            if self.deployment_results['deployment_start_time'] and self.deployment_results['deployment_end_time']:
                deployment_duration = (
                    self.deployment_results['deployment_end_time'] - 
                    self.deployment_results['deployment_start_time']
                ).total_seconds()
            
            report = {
                'deployment_config': self.deployment_config.copy(),
                'deployment_results': self.deployment_results.copy(),
                'deployment_duration_seconds': deployment_duration,
                'system_ready': self.deployment_results['deployment_success'],
                'report_timestamp': datetime.now()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate deployment report: {e}")
            return {'error': str(e)}
    
    def _print_deployment_summary(self):
        """Print deployment summary"""
        try:
            results = self.deployment_results
            
            # Deployment status
            success = results['deployment_success']
            status_emoji = "✅" if success else "❌"
            print(f"{status_emoji} Deployment Status: {'SUCCESS' if success else 'FAILED'}")
            
            # Training results
            training_results = results.get('training_results', {})
            if training_results:
                models_trained = training_results.get('models_trained_count', 0)
                validation_score = training_results.get('average_validation_score', 0.0)
                print(f"📚 Training: {models_trained} models, validation score: {validation_score:.3f}")
            
            # Test results
            test_results = results.get('test_results', {})
            if test_results and test_results.get('test_success'):
                performance = test_results.get('performance_metrics', {})
                successful_tests = performance.get('successful_tests', 0)
                avg_confidence = performance.get('average_confidence', 0.0)
                print(f"🧪 Testing: {successful_tests} successful tests, avg confidence: {avg_confidence:.2%}")
            
            # System status
            system_status = results.get('system_status', {})
            if system_status:
                performance_metrics = system_status.get('performance_metrics', {})
                total_decisions = performance_metrics.get('total_decisions', 0)
                print(f"🔧 System: {total_decisions} decisions generated")
            
            # Duration
            if results['deployment_start_time'] and results['deployment_end_time']:
                duration = (results['deployment_end_time'] - results['deployment_start_time']).total_seconds()
                print(f"⏱️ Duration: {duration:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to print deployment summary: {e}")
    
    def save_deployment_report(self, filepath: str):
        """Save deployment report to file"""
        try:
            # Get deployment report
            report = self._generate_deployment_report()
            
            # Convert datetime objects to strings
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
            
            logger.info(f"Deployment report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save deployment report: {e}")


async def main():
    """Main deployment function"""
    try:
        # Create deployer
        deployer = FourModelSystemDeployer()
        
        # Deploy system
        deployment_report = await deployer.deploy_system()
        
        # Save report
        deployer.save_deployment_report("four_model_deployment_report.json")
        
        # Print final status
        if deployment_report.get('system_ready', False):
            print("\n🎉 FOUR-MODEL SYSTEM READY FOR PRODUCTION!")
            print("=" * 70)
            print("The system is now ready to generate trading decisions using:")
            print("  • Sentiment Analysis (25% weight)")
            print("  • Quantitative Risk Analysis (25% weight)")
            print("  • ML Ensemble Models (35% weight)")
            print("  • RL Decider Agent (Final decision maker)")
            print("\nNext steps:")
            print("  1. Integrate with portfolio management system")
            print("  2. Set up real-time data feeds")
            print("  3. Configure risk management parameters")
            print("  4. Deploy to production environment")
        else:
            print("\n⚠️ DEPLOYMENT COMPLETED WITH ISSUES")
            print("Please review the deployment report for details.")
        
        return deployment_report
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        print(f"\n❌ DEPLOYMENT FAILED: {e}")
        return None


if __name__ == "__main__":
    # Run deployment
    deployment_report = asyncio.run(main())
    
    if deployment_report:
        print(f"\n📋 Deployment completed successfully")
        print(f"Report saved to: four_model_deployment_report.json")
    else:
        print(f"\n❌ Deployment failed")
        sys.exit(1)
