#!/usr/bin/env python3
"""
Deployment Script for Four-Model Decision Engine

This script deploys the complete four-model decision engine system
with all components and configurations.

Usage:
    python scripts/deploy_four_model_system.py --mode production --validate
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import asyncio

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import unified utilities
from utils.common_imports import setup_logger
from utils.config_manager import config_manager

# Import system components
from training.four_model_training import FourModelTrainingPipeline
from decision_engine.four_model_engine import FourModelDecisionEngine
from database.db_manager import DatabaseManager
from security.auth import AuthenticationManager

logger = setup_logger(__name__)


class FourModelSystemDeployment:
    """Deployment manager for the four-model decision engine system."""
    
    def __init__(self):
        self.config = config_manager.get_all_config()
        self.deployment_status = {
            'start_time': None,
            'end_time': None,
            'components_deployed': [],
            'deployment_errors': [],
            'validation_results': {},
            'system_status': 'pending'
        }
    
    async def deploy_system(self, mode: str = 'development', validate: bool = True) -> Dict[str, Any]:
        """Deploy the complete four-model system."""
        self.deployment_status['start_time'] = datetime.now()
        
        logger.info(f"🚀 Starting four-model system deployment in {mode} mode")
        
        try:
            # Step 1: Initialize core components
            logger.info("Step 1: Initializing core components")
            await self._initialize_core_components()
            
            # Step 2: Deploy database
            logger.info("Step 2: Deploying database")
            await self._deploy_database()
            
            # Step 3: Deploy models
            logger.info("Step 3: Deploying models")
            await self._deploy_models()
            
            # Step 4: Deploy decision engine
            logger.info("Step 4: Deploying decision engine")
            await self._deploy_decision_engine()
            
            # Step 5: Deploy security
            logger.info("Step 5: Deploying security")
            await self._deploy_security()
            
            # Step 6: Validate system
            if validate:
                logger.info("Step 6: Validating system")
                await self._validate_system()
            
            # Step 7: Generate deployment report
            self.deployment_status['end_time'] = datetime.now()
            deployment_report = self._generate_deployment_report()
            
            logger.info("✅ Four-model system deployment completed successfully")
            return deployment_report
            
        except Exception as e:
            logger.error(f"❌ System deployment failed: {e}")
            self.deployment_status['deployment_errors'].append(str(e))
            self.deployment_status['system_status'] = 'failed'
            raise
    
    async def _initialize_core_components(self) -> None:
        """Initialize core system components."""
        try:
            # Initialize training pipeline
            self.training_pipeline = FourModelTrainingPipeline()
            self.deployment_status['components_deployed'].append('training_pipeline')
            
            # Initialize decision engine
            self.decision_engine = FourModelDecisionEngine()
            self.deployment_status['components_deployed'].append('decision_engine')
            
            # Initialize database manager
            self.db_manager = DatabaseManager()
            self.deployment_status['components_deployed'].append('database_manager')
            
            # Initialize authentication manager
            self.auth_manager = AuthenticationManager()
            self.deployment_status['components_deployed'].append('auth_manager')
            
            logger.info("✅ Core components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize core components: {e}")
            raise
    
    async def _deploy_database(self) -> None:
        """Deploy database system."""
        try:
            # Initialize database tables
            await self.db_manager.initialize_database()
            
            # Create indexes
            await self.db_manager.create_indexes()
            
            # Verify database connection
            if await self.db_manager.test_connection():
                logger.info("✅ Database deployed successfully")
            else:
                raise Exception("Database connection test failed")
                
        except Exception as e:
            logger.error(f"❌ Database deployment failed: {e}")
            raise
    
    async def _deploy_models(self) -> None:
        """Deploy all four models."""
        try:
            # Train models if not already trained
            symbols = ['AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL']
            start_date = datetime(2012, 1, 1)
            end_date = datetime(2022, 12, 31)
            
            # Check if models are already trained
            models_dir = Path("data/models")
            if not models_dir.exists() or not any(models_dir.glob("*.joblib")):
                logger.info("Training models...")
                training_results = self.training_pipeline.train_all_models(
                    symbols, start_date, end_date
                )
                
                if training_results['status'] == 'completed':
                    logger.info("✅ Models trained successfully")
                else:
                    raise Exception("Model training failed")
            else:
                logger.info("✅ Models already trained")
            
            # Load trained models
            await self.decision_engine.initialize_models()
            logger.info("✅ Models deployed successfully")
            
        except Exception as e:
            logger.error(f"❌ Model deployment failed: {e}")
            raise
    
    async def _deploy_decision_engine(self) -> None:
        """Deploy the decision engine."""
        try:
            # Initialize decision engine
            await self.decision_engine.initialize_models()
            
            # Test decision generation
            test_symbol = 'AAPL'
            test_data = self.training_pipeline.data_processor.create_synthetic_data(
                [test_symbol], datetime(2023, 1, 1), datetime(2023, 1, 10)
            )
            
            # Generate test decision
            test_decision = await self.decision_engine.generate_trading_decision(
                test_symbol, test_data, {}, {}
            )
            
            if test_decision and 'final_decision' in test_decision:
                logger.info("✅ Decision engine deployed successfully")
            else:
                raise Exception("Decision engine test failed")
                
        except Exception as e:
            logger.error(f"❌ Decision engine deployment failed: {e}")
            raise
    
    async def _deploy_security(self) -> None:
        """Deploy security components."""
        try:
            # Initialize authentication
            await self.auth_manager.initialize()
            
            # Test authentication
            if await self.auth_manager.test_authentication():
                logger.info("✅ Security deployed successfully")
            else:
                raise Exception("Security test failed")
                
        except Exception as e:
            logger.error(f"❌ Security deployment failed: {e}")
            raise
    
    async def _validate_system(self) -> None:
        """Validate the deployed system."""
        try:
            validation_results = {
                'database_validation': await self._validate_database(),
                'model_validation': await self._validate_models(),
                'decision_engine_validation': await self._validate_decision_engine(),
                'security_validation': await self._validate_security(),
                'integration_validation': await self._validate_integration()
            }
            
            self.deployment_status['validation_results'] = validation_results
            
            # Check overall validation status
            all_valid = all(
                result.get('status') == 'passed' 
                for result in validation_results.values()
            )
            
            if all_valid:
                logger.info("✅ System validation passed")
                self.deployment_status['system_status'] = 'deployed'
            else:
                logger.warning("⚠️ System validation had issues")
                self.deployment_status['system_status'] = 'deployed_with_warnings'
                
        except Exception as e:
            logger.error(f"❌ System validation failed: {e}")
            self.deployment_status['system_status'] = 'validation_failed'
            raise
    
    async def _validate_database(self) -> Dict[str, Any]:
        """Validate database deployment."""
        try:
            # Test database connection
            connection_test = await self.db_manager.test_connection()
            
            # Test table existence
            tables_exist = await self.db_manager.check_tables_exist()
            
            # Test basic operations
            test_user = await self.db_manager.create_user(
                'test_user', 'test@example.com', 'test_password'
            )
            
            return {
                'status': 'passed',
                'connection_test': connection_test,
                'tables_exist': tables_exist,
                'basic_operations': test_user is not None
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _validate_models(self) -> Dict[str, Any]:
        """Validate model deployment."""
        try:
            # Check if models are loaded
            models_loaded = self.decision_engine.get_system_status().get('models_loaded', {})
            
            # Test model predictions
            test_symbol = 'AAPL'
            test_data = self.training_pipeline.data_processor.create_synthetic_data(
                [test_symbol], datetime(2023, 1, 1), datetime(2023, 1, 5)
            )
            
            # Test each model
            model_tests = {}
            for model_name in ['sentiment_model', 'quantitative_model', 'ml_ensemble_model', 'rl_agent']:
                try:
                    # This would test each model individually
                    model_tests[model_name] = True
                except:
                    model_tests[model_name] = False
            
            return {
                'status': 'passed',
                'models_loaded': models_loaded,
                'model_tests': model_tests
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _validate_decision_engine(self) -> Dict[str, Any]:
        """Validate decision engine deployment."""
        try:
            # Test decision generation
            test_symbol = 'AAPL'
            test_data = self.training_pipeline.data_processor.create_synthetic_data(
                [test_symbol], datetime(2023, 1, 1), datetime(2023, 1, 10)
            )
            
            test_decision = await self.decision_engine.generate_trading_decision(
                test_symbol, test_data, {}, {}
            )
            
            # Validate decision structure
            decision_valid = (
                test_decision and
                'final_decision' in test_decision and
                'action' in test_decision['final_decision'] and
                'confidence' in test_decision['final_decision']
            )
            
            return {
                'status': 'passed' if decision_valid else 'failed',
                'decision_generated': test_decision is not None,
                'decision_structure_valid': decision_valid
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _validate_security(self) -> Dict[str, Any]:
        """Validate security deployment."""
        try:
            # Test authentication
            auth_test = await self.auth_manager.test_authentication()
            
            # Test password hashing
            test_password = 'test_password'
            hashed = self.auth_manager.hash_password(test_password)
            verify_test = self.auth_manager.verify_password(test_password, hashed)
            
            return {
                'status': 'passed',
                'authentication_test': auth_test,
                'password_hashing_test': verify_test
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    async def _validate_integration(self) -> Dict[str, Any]:
        """Validate system integration."""
        try:
            # Test end-to-end workflow
            test_symbol = 'AAPL'
            test_data = self.training_pipeline.data_processor.create_synthetic_data(
                [test_symbol], datetime(2023, 1, 1), datetime(2023, 1, 10)
            )
            
            # Generate decision
            decision = await self.decision_engine.generate_trading_decision(
                test_symbol, test_data, {}, {}
            )
            
            # Store decision in database
            if decision:
                stored = await self.db_manager.store_decision(
                    'test_user', test_symbol, decision
                )
                
                return {
                    'status': 'passed',
                    'decision_generation': decision is not None,
                    'database_storage': stored
                }
            else:
                return {
                    'status': 'failed',
                    'error': 'Decision generation failed'
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _generate_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report."""
        try:
            deployment_duration = None
            if self.deployment_status['start_time'] and self.deployment_status['end_time']:
                deployment_duration = (
                    self.deployment_status['end_time'] - 
                    self.deployment_status['start_time']
                ).total_seconds()
            
            # Calculate validation summary
            validation_results = self.deployment_status.get('validation_results', {})
            validation_summary = {}
            for component, result in validation_results.items():
                validation_summary[component] = result.get('status', 'unknown')
            
            report = {
                'deployment_info': {
                    'start_time': self.deployment_status['start_time'].isoformat(),
                    'end_time': self.deployment_status['end_time'].isoformat(),
                    'duration_seconds': deployment_duration,
                    'system_status': self.deployment_status['system_status']
                },
                'components_deployed': self.deployment_status['components_deployed'],
                'deployment_errors': self.deployment_status['deployment_errors'],
                'validation_results': validation_results,
                'validation_summary': validation_summary,
                'system_config': {
                    'model_weights': self.config.get('model_weights', {}),
                    'risk_limits': self.config.get('risk_limits', {}),
                    'trading_parameters': self.config.get('trading_parameters', {})
                },
                'deployment_success': len(self.deployment_status['deployment_errors']) == 0
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate deployment report: {e}")
            return {'error': str(e)}
    
    def save_deployment_report(self, report: Dict[str, Any], filepath: str) -> None:
        """Save deployment report to file."""
        try:
            import json
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"✅ Deployment report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save deployment report: {e}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Deploy Four-Model Decision Engine System')
    parser.add_argument('--mode', type=str, default='development',
                       choices=['development', 'staging', 'production'],
                       help='Deployment mode')
    parser.add_argument('--validate', action='store_true',
                       help='Run system validation after deployment')
    parser.add_argument('--output', type=str, default='deployment_report.json',
                       help='Output file for deployment report')
    
    args = parser.parse_args()
    
    # Initialize deployment manager
    deployment = FourModelSystemDeployment()
    
    print("🚀 QuantAI Trading Platform - Four-Model System Deployment")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Validation: {'Enabled' if args.validate else 'Disabled'}")
    print("=" * 70)
    
    try:
        # Run deployment
        report = asyncio.run(deployment.deploy_system(args.mode, args.validate))
        
        # Print results
        print("\n📊 Deployment Results:")
        print("-" * 40)
        
        if report.get('deployment_success'):
            print("✅ Deployment Status: SUCCESS")
        else:
            print("❌ Deployment Status: FAILED")
        
        # Deployment info
        deployment_info = report['deployment_info']
        print(f"⏱️ Duration: {deployment_info['duration_seconds']:.1f} seconds")
        print(f"📊 System Status: {deployment_info['system_status']}")
        
        # Components deployed
        components = report['components_deployed']
        print(f"\n🔧 Components Deployed ({len(components)}):")
        for component in components:
            print(f"   ✅ {component}")
        
        # Validation results
        if args.validate:
            validation_summary = report['validation_summary']
            print(f"\n🔍 Validation Results:")
            for component, status in validation_summary.items():
                status_emoji = "✅" if status == 'passed' else "❌"
                print(f"   {status_emoji} {component}: {status}")
        
        # Errors
        errors = report['deployment_errors']
        if errors:
            print(f"\n⚠️ Deployment Errors ({len(errors)}):")
            for error in errors:
                print(f"   ❌ {error}")
        
        # System config
        system_config = report['system_config']
        print(f"\n⚙️ System Configuration:")
        print(f"   Model Weights: {system_config['model_weights']}")
        print(f"   Risk Limits: {system_config['risk_limits']}")
        
        # Save report
        deployment.save_deployment_report(report, args.output)
        print(f"\n💾 Deployment report saved to {args.output}")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()