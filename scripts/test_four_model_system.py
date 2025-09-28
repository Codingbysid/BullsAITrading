from src.utils.common_imports import *
import asyncio
import sys
import os
import json
import unittest
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time
from src.models.sentiment_model import SentimentAnalysisModel
from src.models.quantitative_model import QuantitativeRiskModel
from src.models.ml_ensemble_model import MLEnsembleModel
from src.models.rl_decider_agent import RLDeciderAgent
from src.decision_engine.four_model_engine import FourModelDecisionEngine
from src.training.four_model_training import FourModelTrainingPipeline
import logging
            from src.models.base_models import ModelOutput
            import psutil
            import os

#!/usr/bin/env python3
"""
Comprehensive testing framework for the four-model decision engine system.

This script provides extensive testing capabilities:
- Unit tests for individual models
- Integration tests for the complete system
- Performance tests and benchmarks
- Stress tests and edge cases
- Validation tests for decision quality

Ensures the four-model architecture is robust and production-ready.
"""


# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = setup_logger()


class FourModelSystemTester:
    """Comprehensive testing framework for the four-model system"""
    
    def __init__(self):
        self.test_results = {
            'test_start_time': None,
            'test_end_time': None,
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'stress_tests': {},
            'validation_tests': {},
            'overall_success': False
        }
        
        # Test configuration
        self.test_config = {
            'test_symbols': ['AAPL', 'AMZN', 'GOOGL', 'META', 'NVDA'],
            'test_data_points': 100,
            'performance_thresholds': {
                'min_confidence': 0.3,
                'max_response_time': 1.0,  # seconds
                'min_accuracy': 0.4
            },
            'stress_test_iterations': 50
        }
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        
        self.test_results['test_start_time'] = datetime.now()
        
        print("🧪 COMPREHENSIVE FOUR-MODEL SYSTEM TESTING")
        print("=" * 70)
        print("Testing Components:")
        print("  🔬 Unit Tests - Individual model validation")
        print("  🔗 Integration Tests - System-wide functionality")
        print("  ⚡ Performance Tests - Speed and efficiency")
        print("  💪 Stress Tests - Edge cases and limits")
        print("  ✅ Validation Tests - Decision quality assurance")
        print("=" * 70)
        
        try:
            # Phase 1: Unit Tests
            print("\n🔬 Phase 1: Unit Tests")
            print("-" * 40)
            unit_test_results = await self._run_unit_tests()
            self.test_results['unit_tests'] = unit_test_results
            
            # Phase 2: Integration Tests
            print("\n🔗 Phase 2: Integration Tests")
            print("-" * 40)
            integration_test_results = await self._run_integration_tests()
            self.test_results['integration_tests'] = integration_test_results
            
            # Phase 3: Performance Tests
            print("\n⚡ Phase 3: Performance Tests")
            print("-" * 40)
            performance_test_results = await self._run_performance_tests()
            self.test_results['performance_tests'] = performance_test_results
            
            # Phase 4: Stress Tests
            print("\n💪 Phase 4: Stress Tests")
            print("-" * 40)
            stress_test_results = await self._run_stress_tests()
            self.test_results['stress_tests'] = stress_test_results
            
            # Phase 5: Validation Tests
            print("\n✅ Phase 5: Validation Tests")
            print("-" * 40)
            validation_test_results = await self._run_validation_tests()
            self.test_results['validation_tests'] = validation_test_results
            
            # Phase 6: Generate Test Report
            print("\n📊 Phase 6: Test Report")
            print("-" * 40)
            test_report = self._generate_test_report()
            
            self.test_results['test_end_time'] = datetime.now()
            self.test_results['overall_success'] = self._evaluate_overall_success()
            
            print("\n🎉 COMPREHENSIVE TESTING COMPLETE!")
            print("=" * 70)
            self._print_test_summary()
            
            return test_report
            
        except Exception as e:
            logger.error(f"Comprehensive testing failed: {e}")
            self.test_results['test_end_time'] = datetime.now()
            self.test_results['overall_success'] = False
            raise
    
    async def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests for individual models"""
        try:
            print("Testing individual model components...")
            
            unit_test_results = {
                'sentiment_model': {},
                'quantitative_model': {},
                'ml_ensemble_model': {},
                'rl_decider_agent': {},
                'overall_success': True
            }
            
            # Test Sentiment Model
            print("  Testing Sentiment Analysis Model...")
            sentiment_test = await self._test_sentiment_model()
            unit_test_results['sentiment_model'] = sentiment_test
            
            # Test Quantitative Model
            print("  Testing Quantitative Risk Model...")
            quantitative_test = await self._test_quantitative_model()
            unit_test_results['quantitative_model'] = quantitative_test
            
            # Test ML Ensemble Model
            print("  Testing ML Ensemble Model...")
            ml_ensemble_test = await self._test_ml_ensemble_model()
            unit_test_results['ml_ensemble_model'] = ml_ensemble_test
            
            # Test RL Decider Agent
            print("  Testing RL Decider Agent...")
            rl_agent_test = await self._test_rl_decider_agent()
            unit_test_results['rl_decider_agent'] = rl_agent_test
            
            # Evaluate overall success
            all_tests = [sentiment_test, quantitative_test, ml_ensemble_test, rl_agent_test]
            unit_test_results['overall_success'] = all(test.get('success', False) for test in all_tests)
            
            print(f"✅ Unit tests completed: {'PASS' if unit_test_results['overall_success'] else 'FAIL'}")
            
            return unit_test_results
            
        except Exception as e:
            logger.error(f"Unit tests failed: {e}")
            return {'overall_success': False, 'error': str(e)}
    
    async def _test_sentiment_model(self) -> Dict[str, Any]:
        """Test sentiment analysis model"""
        try:
            model = SentimentAnalysisModel()
            
            # Create test data
            market_data = self._create_test_market_data()
            features = self._create_test_features()
            
            # Test prediction
            result = await model.predict("AAPL", market_data, features)
            
            # Validate output
            success = (
                isinstance(result.signal, (int, float)) and
                -1.0 <= result.signal <= 1.0 and
                0.0 <= result.confidence <= 1.0 and
                isinstance(result.reasoning, str) and
                len(result.reasoning) > 0
            )
            
            return {
                'success': success,
                'signal': result.signal,
                'confidence': result.confidence,
                'reasoning_length': len(result.reasoning),
                'metrics_count': len(result.metrics)
            }
            
        except Exception as e:
            logger.error(f"Sentiment model test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_quantitative_model(self) -> Dict[str, Any]:
        """Test quantitative risk model"""
        try:
            model = QuantitativeRiskModel()
            
            # Create test data
            market_data = self._create_test_market_data()
            features = self._create_test_features()
            
            # Test prediction
            result = model.predict("AAPL", market_data, features)
            
            # Validate output
            success = (
                isinstance(result.signal, (int, float)) and
                -1.0 <= result.signal <= 1.0 and
                0.0 <= result.confidence <= 1.0 and
                isinstance(result.reasoning, str) and
                len(result.reasoning) > 0
            )
            
            return {
                'success': success,
                'signal': result.signal,
                'confidence': result.confidence,
                'reasoning_length': len(result.reasoning),
                'risk_metrics': len(result.metrics)
            }
            
        except Exception as e:
            logger.error(f"Quantitative model test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_ml_ensemble_model(self) -> Dict[str, Any]:
        """Test ML ensemble model"""
        try:
            model = MLEnsembleModel()
            
            # Create test data
            market_data = self._create_test_market_data()
            features = self._create_test_features()
            
            # Test prediction (should work even if not trained)
            result = model.predict("AAPL", market_data, features)
            
            # Validate output
            success = (
                isinstance(result.signal, (int, float)) and
                -1.0 <= result.signal <= 1.0 and
                0.0 <= result.confidence <= 1.0 and
                isinstance(result.reasoning, str)
            )
            
            return {
                'success': success,
                'signal': result.signal,
                'confidence': result.confidence,
                'reasoning_length': len(result.reasoning),
                'is_trained': model.is_trained
            }
            
        except Exception as e:
            logger.error(f"ML ensemble model test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_rl_decider_agent(self) -> Dict[str, Any]:
        """Test RL decider agent"""
        try:
            agent = RLDeciderAgent()
            
            # Create mock model outputs
            
            sentiment_output = ModelOutput(
                signal=0.3, confidence=0.8, reasoning="Test sentiment", metrics={}
            )
            quantitative_output = ModelOutput(
                signal=0.2, confidence=0.7, reasoning="Test quantitative", 
                metrics={'sharpe_ratio': 1.5, 'mar_ratio': 0.8}
            )
            ml_output = ModelOutput(
                signal=0.4, confidence=0.75, reasoning="Test ML", metrics={}
            )
            
            # Test prediction
            result = agent.predict(
                sentiment_output, quantitative_output, ml_output,
                {'price_momentum': 0.05, 'volume_ratio': 1.2},
                {'current_position': 0.1, 'portfolio_risk': 0.3, 'cash_ratio': 0.7}
            )
            
            # Validate output
            success = (
                isinstance(result.signal, (int, float)) and
                -1.0 <= result.signal <= 1.0 and
                0.0 <= result.confidence <= 1.0 and
                isinstance(result.reasoning, str) and
                len(result.reasoning) > 0
            )
            
            return {
                'success': success,
                'signal': result.signal,
                'confidence': result.confidence,
                'reasoning_length': len(result.reasoning),
                'epsilon': result.metrics.get('epsilon', 0.0)
            }
            
        except Exception as e:
            logger.error(f"RL decider agent test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests for the complete system"""
        try:
            print("Testing system integration...")
            
            integration_test_results = {
                'decision_engine': {},
                'training_pipeline': {},
                'end_to_end': {},
                'overall_success': True
            }
            
            # Test Decision Engine
            print("  Testing Decision Engine...")
            decision_engine_test = await self._test_decision_engine()
            integration_test_results['decision_engine'] = decision_engine_test
            
            # Test Training Pipeline
            print("  Testing Training Pipeline...")
            training_pipeline_test = await self._test_training_pipeline()
            integration_test_results['training_pipeline'] = training_pipeline_test
            
            # Test End-to-End
            print("  Testing End-to-End Workflow...")
            end_to_end_test = await self._test_end_to_end_workflow()
            integration_test_results['end_to_end'] = end_to_end_test
            
            # Evaluate overall success
            all_tests = [decision_engine_test, training_pipeline_test, end_to_end_test]
            integration_test_results['overall_success'] = all(test.get('success', False) for test in all_tests)
            
            print(f"✅ Integration tests completed: {'PASS' if integration_test_results['overall_success'] else 'FAIL'}")
            
            return integration_test_results
            
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            return {'overall_success': False, 'error': str(e)}
    
    async def _test_decision_engine(self) -> Dict[str, Any]:
        """Test decision engine integration"""
        try:
            engine = FourModelDecisionEngine()
            
            # Initialize engine
            await engine.initialize_models()
            
            # Create test data
            market_data = self._create_test_market_data()
            features = self._create_test_features()
            portfolio_state = {'current_position': 0.1, 'portfolio_risk': 0.3, 'cash_ratio': 0.7}
            
            # Test decision generation
            decision = await engine.generate_trading_decision("AAPL", market_data, features, portfolio_state)
            
            # Validate decision
            success = (
                'final_decision' in decision and
                'model_inputs' in decision and
                'rl_decision_details' in decision and
                'risk_assessment' in decision
            )
            
            return {
                'success': success,
                'decision_keys': list(decision.keys()),
                'final_action': decision.get('final_decision', {}).get('action', 'UNKNOWN'),
                'confidence': decision.get('final_decision', {}).get('confidence', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Decision engine test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_training_pipeline(self) -> Dict[str, Any]:
        """Test training pipeline"""
        try:
            pipeline = FourModelTrainingPipeline()
            
            # Test with minimal data
            symbols = ['AAPL']
            
            # This should work even with limited data
            training_summary = await pipeline.train_complete_system(
                symbols, training_period_days=100, validation_period_days=50
            )
            
            success = isinstance(training_summary, dict) and 'training_success' in training_summary
            
            return {
                'success': success,
                'training_summary_keys': list(training_summary.keys()) if success else [],
                'training_success': training_summary.get('training_success', False) if success else False
            }
            
        except Exception as e:
            logger.error(f"Training pipeline test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test complete end-to-end workflow"""
        try:
            # Create training pipeline
            pipeline = FourModelTrainingPipeline()
            
            # Train system
            training_summary = await pipeline.train_complete_system(
                ['AAPL'], training_period_days=100, validation_period_days=50
            )
            
            # Get decision engine
            engine = pipeline.decision_engine
            
            # Generate decision
            market_data = self._create_test_market_data()
            features = self._create_test_features()
            portfolio_state = {'current_position': 0.1, 'portfolio_risk': 0.3, 'cash_ratio': 0.7}
            
            decision = await engine.generate_trading_decision("AAPL", market_data, features, portfolio_state)
            
            # Validate complete workflow
            success = (
                training_summary.get('training_success', False) and
                'final_decision' in decision and
                decision['final_decision']['action'] in ['BUY', 'SELL', 'HOLD']
            )
            
            return {
                'success': success,
                'training_success': training_summary.get('training_success', False),
                'decision_generated': 'final_decision' in decision,
                'decision_action': decision.get('final_decision', {}).get('action', 'UNKNOWN')
            }
            
        except Exception as e:
            logger.error(f"End-to-end workflow test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        try:
            print("Testing system performance...")
            
            performance_test_results = {
                'response_times': {},
                'throughput': {},
                'memory_usage': {},
                'overall_success': True
            }
            
            # Test response times
            print("  Testing response times...")
            response_time_test = await self._test_response_times()
            performance_test_results['response_times'] = response_time_test
            
            # Test throughput
            print("  Testing throughput...")
            throughput_test = await self._test_throughput()
            performance_test_results['throughput'] = throughput_test
            
            # Test memory usage
            print("  Testing memory usage...")
            memory_test = await self._test_memory_usage()
            performance_test_results['memory_usage'] = memory_test
            
            # Evaluate overall success
            thresholds = self.test_config['performance_thresholds']
            overall_success = (
                response_time_test.get('avg_response_time', 10.0) <= thresholds['max_response_time'] and
                throughput_test.get('decisions_per_second', 0) >= 1.0
            )
            
            performance_test_results['overall_success'] = overall_success
            
            print(f"✅ Performance tests completed: {'PASS' if overall_success else 'FAIL'}")
            
            return performance_test_results
            
        except Exception as e:
            logger.error(f"Performance tests failed: {e}")
            return {'overall_success': False, 'error': str(e)}
    
    async def _test_response_times(self) -> Dict[str, Any]:
        """Test response times for decision generation"""
        try:
            engine = FourModelDecisionEngine()
            await engine.initialize_models()
            
            # Create test data
            market_data = self._create_test_market_data()
            features = self._create_test_features()
            portfolio_state = {'current_position': 0.1, 'portfolio_risk': 0.3, 'cash_ratio': 0.7}
            
            # Measure response times
            response_times = []
            for i in range(10):  # Test 10 decisions
                start_time = time.time()
                decision = await engine.generate_trading_decision("AAPL", market_data, features, portfolio_state)
                end_time = time.time()
                
                response_time = end_time - start_time
                response_times.append(response_time)
            
            avg_response_time = np.mean(response_times)
            max_response_time = np.max(response_times)
            min_response_time = np.min(response_times)
            
            return {
                'avg_response_time': avg_response_time,
                'max_response_time': max_response_time,
                'min_response_time': min_response_time,
                'response_times': response_times
            }
            
        except Exception as e:
            logger.error(f"Response time test failed: {e}")
            return {'error': str(e)}
    
    async def _test_throughput(self) -> Dict[str, Any]:
        """Test system throughput"""
        try:
            engine = FourModelDecisionEngine()
            await engine.initialize_models()
            
            # Create test data
            market_data = self._create_test_market_data()
            features = self._create_test_features()
            portfolio_state = {'current_position': 0.1, 'portfolio_risk': 0.3, 'cash_ratio': 0.7}
            
            # Measure throughput
            start_time = time.time()
            decisions_generated = 0
            
            # Generate decisions for 5 seconds
            while time.time() - start_time < 5.0:
                decision = await engine.generate_trading_decision("AAPL", market_data, features, portfolio_state)
                decisions_generated += 1
            
            total_time = time.time() - start_time
            decisions_per_second = decisions_generated / total_time
            
            return {
                'decisions_generated': decisions_generated,
                'total_time': total_time,
                'decisions_per_second': decisions_per_second
            }
            
        except Exception as e:
            logger.error(f"Throughput test failed: {e}")
            return {'error': str(e)}
    
    async def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage"""
        try:
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create and use system
            engine = FourModelDecisionEngine()
            await engine.initialize_models()
            
            # Generate some decisions
            market_data = self._create_test_market_data()
            features = self._create_test_features()
            portfolio_state = {'current_position': 0.1, 'portfolio_risk': 0.3, 'cash_ratio': 0.7}
            
            for i in range(10):
                decision = await engine.generate_trading_decision("AAPL", market_data, features, portfolio_state)
            
            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            return {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': memory_increase
            }
            
        except Exception as e:
            logger.error(f"Memory usage test failed: {e}")
            return {'error': str(e)}
    
    async def _run_stress_tests(self) -> Dict[str, Any]:
        """Run stress tests"""
        try:
            print("Testing system under stress...")
            
            stress_test_results = {
                'high_volume': {},
                'edge_cases': {},
                'error_handling': {},
                'overall_success': True
            }
            
            # Test high volume
            print("  Testing high volume decisions...")
            high_volume_test = await self._test_high_volume()
            stress_test_results['high_volume'] = high_volume_test
            
            # Test edge cases
            print("  Testing edge cases...")
            edge_cases_test = await self._test_edge_cases()
            stress_test_results['edge_cases'] = edge_cases_test
            
            # Test error handling
            print("  Testing error handling...")
            error_handling_test = await self._test_error_handling()
            stress_test_results['error_handling'] = error_handling_test
            
            # Evaluate overall success
            overall_success = (
                high_volume_test.get('success', False) and
                edge_cases_test.get('success', False) and
                error_handling_test.get('success', False)
            )
            
            stress_test_results['overall_success'] = overall_success
            
            print(f"✅ Stress tests completed: {'PASS' if overall_success else 'FAIL'}")
            
            return stress_test_results
            
        except Exception as e:
            logger.error(f"Stress tests failed: {e}")
            return {'overall_success': False, 'error': str(e)}
    
    async def _test_high_volume(self) -> Dict[str, Any]:
        """Test system under high volume"""
        try:
            engine = FourModelDecisionEngine()
            await engine.initialize_models()
            
            # Create test data
            market_data = self._create_test_market_data()
            features = self._create_test_features()
            portfolio_state = {'current_position': 0.1, 'portfolio_risk': 0.3, 'cash_ratio': 0.7}
            
            # Generate many decisions
            decisions = []
            start_time = time.time()
            
            for i in range(self.test_config['stress_test_iterations']):
                decision = await engine.generate_trading_decision("AAPL", market_data, features, portfolio_state)
                decisions.append(decision)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Check if all decisions were generated successfully
            success = len(decisions) == self.test_config['stress_test_iterations']
            
            return {
                'success': success,
                'decisions_generated': len(decisions),
                'total_time': total_time,
                'avg_time_per_decision': total_time / len(decisions) if decisions else 0
            }
            
        except Exception as e:
            logger.error(f"High volume test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases"""
        try:
            engine = FourModelDecisionEngine()
            await engine.initialize_models()
            
            edge_case_results = []
            
            # Test with empty data
            try:
                empty_market_data = pd.DataFrame()
                empty_features = pd.DataFrame()
                decision = await engine.generate_trading_decision("AAPL", empty_market_data, empty_features, {})
                edge_case_results.append({'test': 'empty_data', 'success': True})
            except Exception as e:
                edge_case_results.append({'test': 'empty_data', 'success': False, 'error': str(e)})
            
            # Test with invalid portfolio state
            try:
                market_data = self._create_test_market_data()
                features = self._create_test_features()
                invalid_portfolio = {'invalid_key': 'invalid_value'}
                decision = await engine.generate_trading_decision("AAPL", market_data, features, invalid_portfolio)
                edge_case_results.append({'test': 'invalid_portfolio', 'success': True})
            except Exception as e:
                edge_case_results.append({'test': 'invalid_portfolio', 'success': False, 'error': str(e)})
            
            # Test with extreme values
            try:
                extreme_portfolio = {
                    'current_position': 10.0,  # Invalid: > 1.0
                    'portfolio_risk': -0.5,    # Invalid: negative
                    'cash_ratio': 2.0          # Invalid: > 1.0
                }
                decision = await engine.generate_trading_decision("AAPL", market_data, features, extreme_portfolio)
                edge_case_results.append({'test': 'extreme_values', 'success': True})
            except Exception as e:
                edge_case_results.append({'test': 'extreme_values', 'success': False, 'error': str(e)})
            
            # Overall success
            success = all(result['success'] for result in edge_case_results)
            
            return {
                'success': success,
                'edge_case_results': edge_case_results
            }
            
        except Exception as e:
            logger.error(f"Edge cases test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_error_handling(self) -> Dict[str, Any]:
        """Test error handling"""
        try:
            engine = FourModelDecisionEngine()
            await engine.initialize_models()
            
            error_handling_results = []
            
            # Test with None inputs
            try:
                decision = await engine.generate_trading_decision(None, None, None, None)
                error_handling_results.append({'test': 'none_inputs', 'success': True})
            except Exception as e:
                error_handling_results.append({'test': 'none_inputs', 'success': False, 'error': str(e)})
            
            # Test with malformed data
            try:
                malformed_data = pd.DataFrame({'invalid': [1, 2, 3]})
                decision = await engine.generate_trading_decision("AAPL", malformed_data, malformed_data, {})
                error_handling_results.append({'test': 'malformed_data', 'success': True})
            except Exception as e:
                error_handling_results.append({'test': 'malformed_data', 'success': False, 'error': str(e)})
            
            # Overall success (should handle errors gracefully)
            success = all(result['success'] for result in error_handling_results)
            
            return {
                'success': success,
                'error_handling_results': error_handling_results
            }
            
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _run_validation_tests(self) -> Dict[str, Any]:
        """Run validation tests for decision quality"""
        try:
            print("Validating decision quality...")
            
            validation_test_results = {
                'decision_quality': {},
                'consistency': {},
                'risk_management': {},
                'overall_success': True
            }
            
            # Test decision quality
            print("  Testing decision quality...")
            decision_quality_test = await self._test_decision_quality()
            validation_test_results['decision_quality'] = decision_quality_test
            
            # Test consistency
            print("  Testing decision consistency...")
            consistency_test = await self._test_decision_consistency()
            validation_test_results['consistency'] = consistency_test
            
            # Test risk management
            print("  Testing risk management...")
            risk_management_test = await self._test_risk_management()
            validation_test_results['risk_management'] = risk_management_test
            
            # Evaluate overall success
            overall_success = (
                decision_quality_test.get('success', False) and
                consistency_test.get('success', False) and
                risk_management_test.get('success', False)
            )
            
            validation_test_results['overall_success'] = overall_success
            
            print(f"✅ Validation tests completed: {'PASS' if overall_success else 'FAIL'}")
            
            return validation_test_results
            
        except Exception as e:
            logger.error(f"Validation tests failed: {e}")
            return {'overall_success': False, 'error': str(e)}
    
    async def _test_decision_quality(self) -> Dict[str, Any]:
        """Test decision quality"""
        try:
            engine = FourModelDecisionEngine()
            await engine.initialize_models()
            
            # Generate multiple decisions
            decisions = []
            market_data = self._create_test_market_data()
            features = self._create_test_features()
            portfolio_state = {'current_position': 0.1, 'portfolio_risk': 0.3, 'cash_ratio': 0.7}
            
            for i in range(10):
                decision = await engine.generate_trading_decision("AAPL", market_data, features, portfolio_state)
                decisions.append(decision)
            
            # Analyze decision quality
            confidences = [d['final_decision']['confidence'] for d in decisions]
            signals = [d['final_decision']['signal_strength'] for d in decisions]
            actions = [d['final_decision']['action'] for d in decisions]
            
            # Quality metrics
            avg_confidence = np.mean(confidences)
            signal_range = np.max(signals) - np.min(signals)
            action_diversity = len(set(actions))
            
            # Success criteria
            success = (
                avg_confidence >= self.test_config['performance_thresholds']['min_confidence'] and
                signal_range > 0 and  # Some variation in signals
                action_diversity > 1  # Multiple action types
            )
            
            return {
                'success': success,
                'avg_confidence': avg_confidence,
                'signal_range': signal_range,
                'action_diversity': action_diversity,
                'decisions_analyzed': len(decisions)
            }
            
        except Exception as e:
            logger.error(f"Decision quality test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_decision_consistency(self) -> Dict[str, Any]:
        """Test decision consistency"""
        try:
            engine = FourModelDecisionEngine()
            await engine.initialize_models()
            
            # Generate decisions with same inputs
            market_data = self._create_test_market_data()
            features = self._create_test_features()
            portfolio_state = {'current_position': 0.1, 'portfolio_risk': 0.3, 'cash_ratio': 0.7}
            
            decisions = []
            for i in range(5):
                decision = await engine.generate_trading_decision("AAPL", market_data, features, portfolio_state)
                decisions.append(decision)
            
            # Check consistency
            actions = [d['final_decision']['action'] for d in decisions]
            confidences = [d['final_decision']['confidence'] for d in decisions]
            
            # Consistency metrics
            action_consistency = len(set(actions)) == 1  # All same action
            confidence_consistency = np.std(confidences) < 0.1  # Low variance in confidence
            
            # Success (some variation is expected due to exploration)
            success = not action_consistency or confidence_consistency
            
            return {
                'success': success,
                'action_consistency': action_consistency,
                'confidence_consistency': confidence_consistency,
                'confidence_std': np.std(confidences),
                'actions': actions
            }
            
        except Exception as e:
            logger.error(f"Decision consistency test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _test_risk_management(self) -> Dict[str, Any]:
        """Test risk management"""
        try:
            engine = FourModelDecisionEngine()
            await engine.initialize_models()
            
            # Test with different risk levels
            risk_scenarios = [
                {'current_position': 0.0, 'portfolio_risk': 0.1, 'cash_ratio': 1.0},  # Low risk
                {'current_position': 0.2, 'portfolio_risk': 0.5, 'cash_ratio': 0.5},  # Medium risk
                {'current_position': 0.4, 'portfolio_risk': 0.8, 'cash_ratio': 0.2},  # High risk
            ]
            
            decisions = []
            market_data = self._create_test_market_data()
            features = self._create_test_features()
            
            for scenario in risk_scenarios:
                decision = await engine.generate_trading_decision("AAPL", market_data, features, scenario)
                decisions.append(decision)
            
            # Analyze risk management
            position_sizes = [d['final_decision']['position_size'] for d in decisions]
            risk_scores = [d['risk_assessment']['overall_risk_score'] for d in decisions]
            
            # Risk management should reduce position sizes for higher risk
            risk_management_working = position_sizes[0] >= position_sizes[1] >= position_sizes[2]
            
            success = risk_management_working and all(0 <= ps <= 0.3 for ps in position_sizes)
            
            return {
                'success': success,
                'position_sizes': position_sizes,
                'risk_scores': risk_scores,
                'risk_management_working': risk_management_working
            }
            
        except Exception as e:
            logger.error(f"Risk management test failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_test_market_data(self) -> pd.DataFrame:
        """Create test market data"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=self.test_config['test_data_points'], freq='D')
        prices = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, self.test_config['test_data_points']))
        
        return pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.005,
            'Low': prices * 0.995,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, self.test_config['test_data_points'])
        }, index=dates)
    
    def _create_test_features(self) -> pd.DataFrame:
        """Create test features"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=self.test_config['test_data_points'], freq='D')
        
        return pd.DataFrame({
            'rsi': np.random.uniform(20, 80, self.test_config['test_data_points']),
            'macd': np.random.normal(0, 0.1, self.test_config['test_data_points']),
            'bb_upper': np.random.uniform(105, 110, self.test_config['test_data_points']),
            'bb_lower': np.random.uniform(90, 95, self.test_config['test_data_points']),
            'sma_20': np.random.uniform(95, 105, self.test_config['test_data_points'])
        }, index=dates)
    
    def _evaluate_overall_success(self) -> bool:
        """Evaluate overall test success"""
        try:
            # Check all test phases
            phases = [
                self.test_results.get('unit_tests', {}).get('overall_success', False),
                self.test_results.get('integration_tests', {}).get('overall_success', False),
                self.test_results.get('performance_tests', {}).get('overall_success', False),
                self.test_results.get('stress_tests', {}).get('overall_success', False),
                self.test_results.get('validation_tests', {}).get('overall_success', False)
            ]
            
            return all(phases)
            
        except Exception as e:
            logger.error(f"Failed to evaluate overall success: {e}")
            return False
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        try:
            test_duration = None
            if self.test_results['test_start_time'] and self.test_results['test_end_time']:
                test_duration = (
                    self.test_results['test_end_time'] - 
                    self.test_results['test_start_time']
                ).total_seconds()
            
            report = {
                'test_config': self.test_config.copy(),
                'test_results': self.test_results.copy(),
                'test_duration_seconds': test_duration,
                'overall_success': self.test_results['overall_success'],
                'report_timestamp': datetime.now()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate test report: {e}")
            return {'error': str(e)}
    
    def _print_test_summary(self):
        """Print test summary"""
        try:
            results = self.test_results
            
            # Overall status
            success = results['overall_success']
            status_emoji = "✅" if success else "❌"
            print(f"{status_emoji} Overall Test Status: {'PASS' if success else 'FAIL'}")
            
            # Individual test phases
            phases = [
                ('Unit Tests', results.get('unit_tests', {}).get('overall_success', False)),
                ('Integration Tests', results.get('integration_tests', {}).get('overall_success', False)),
                ('Performance Tests', results.get('performance_tests', {}).get('overall_success', False)),
                ('Stress Tests', results.get('stress_tests', {}).get('overall_success', False)),
                ('Validation Tests', results.get('validation_tests', {}).get('overall_success', False))
            ]
            
            for phase_name, phase_success in phases:
                phase_emoji = "✅" if phase_success else "❌"
                print(f"  {phase_emoji} {phase_name}: {'PASS' if phase_success else 'FAIL'}")
            
            # Duration
            if results['test_start_time'] and results['test_end_time']:
                duration = (results['test_end_time'] - results['test_start_time']).total_seconds()
                print(f"⏱️ Test Duration: {duration:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to print test summary: {e}")
    
    def save_test_report(self, filepath: str):
        """Save test report to file"""
        try:
            # Get test report
            report = self._generate_test_report()
            
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
            
            logger.info(f"Test report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save test report: {e}")


async def main():
    """Main testing function"""
    try:
        # Create tester
        tester = FourModelSystemTester()
        
        # Run comprehensive tests
        test_report = await tester.run_comprehensive_tests()
        
        # Save report
        tester.save_test_report("four_model_test_report.json")
        
        # Print final status
        if test_report.get('overall_success', False):
            print("\n🎉 ALL TESTS PASSED!")
            print("=" * 70)
            print("The four-model system is ready for production deployment.")
        else:
            print("\n⚠️ SOME TESTS FAILED")
            print("Please review the test report for details.")
        
        return test_report
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        print(f"\n❌ TESTING FAILED: {e}")
        return None


if __name__ == "__main__":
    # Run comprehensive tests
    test_report = asyncio.run(main())
    
    if test_report:
        print(f"\n📋 Testing completed")
        print(f"Report saved to: four_model_test_report.json")
    else:
        print(f"\n❌ Testing failed")
        sys.exit(1)
