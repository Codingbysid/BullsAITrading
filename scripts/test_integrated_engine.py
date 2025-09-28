from src.utils.common_imports import *
from src.utils.common_imports import setup_logger
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import asyncio
from src.utils.data_processing import data_processor
from src.decision_engine.four_model_engine import FourModelDecisionEngine
        import json

#!/usr/bin/env python3
"""
Test Integrated Four-Model Decision Engine

This script tests the four-model decision engine with the trained ML ensemble model
integrated from real dataset training.

Usage:
    python scripts/test_integrated_engine.py
"""


# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import unified utilities

# Import decision engine

logger = setup_logger(__name__)


class IntegratedEngineTester:
    """Test the integrated four-model decision engine."""
    
    def __init__(self):
        self.engine = FourModelDecisionEngine()
        self.test_results = {}
    
    def create_test_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Create test data for a symbol."""
        # Create synthetic test data
        test_data = data_processor.create_synthetic_data(
            [symbol], 
            datetime(2023, 1, 1), 
            datetime(2023, 1, 1) + pd.Timedelta(days=days),
            base_price=100.0
        )
        
        return test_data
    
    def test_individual_models(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Test individual models in the decision engine."""
        logger.info("Testing individual models")
        
        results = {}
        
        # Test sentiment model
        try:
            sentiment_result = self.engine.sentiment_model.predict(test_data)
            results['sentiment_model'] = {
                'status': 'success',
                'signal': sentiment_result.signal,
                'confidence': sentiment_result.confidence,
                'reasoning': sentiment_result.reasoning
            }
            logger.info(f"✅ Sentiment model: signal={sentiment_result.signal}, confidence={sentiment_result.confidence:.3f}")
        except Exception as e:
            results['sentiment_model'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Sentiment model failed: {e}")
        
        # Test quantitative model
        try:
            quantitative_result = self.engine.quantitative_model.predict(test_data)
            results['quantitative_model'] = {
                'status': 'success',
                'signal': quantitative_result.signal,
                'confidence': quantitative_result.confidence,
                'reasoning': quantitative_result.reasoning
            }
            logger.info(f"✅ Quantitative model: signal={quantitative_result.signal}, confidence={quantitative_result.confidence:.3f}")
        except Exception as e:
            results['quantitative_model'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Quantitative model failed: {e}")
        
        # Test trained ML ensemble model
        try:
            ml_result = self.engine.ml_ensemble_model.predict(test_data)
            results['ml_ensemble_model'] = {
                'status': 'success',
                'signal': ml_result.signal,
                'confidence': ml_result.confidence,
                'reasoning': ml_result.reasoning,
                'metadata': ml_result.metadata
            }
            logger.info(f"✅ ML ensemble model: signal={ml_result.signal}, confidence={ml_result.confidence:.3f}")
        except Exception as e:
            results['ml_ensemble_model'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ ML ensemble model failed: {e}")
        
        return results
    
    async def test_decision_engine(self, symbol: str, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Test the complete decision engine."""
        logger.info(f"Testing decision engine for {symbol}")
        
        try:
            # Create portfolio state
            portfolio_state = {
                'current_position': 0.0,
                'portfolio_risk': 0.3,
                'cash_ratio': 0.7
            }
            
            # Generate trading decision
            decision = await self.engine.generate_trading_decision(
                symbol, test_data, {}, portfolio_state
            )
            
            return {
                'status': 'success',
                'symbol': symbol,
                'decision': decision,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Decision engine test failed for {symbol}: {e}")
            return {
                'status': 'failed',
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def test_multiple_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        """Test the decision engine with multiple symbols."""
        logger.info(f"Testing decision engine with {len(symbols)} symbols")
        
        results = {}
        
        for symbol in symbols:
            logger.info(f"Testing {symbol}")
            
            # Create test data
            test_data = self.create_test_data(symbol)
            
            # Test individual models
            individual_results = self.test_individual_models(test_data)
            
            # Test complete decision engine
            decision_result = asyncio.run(self.test_decision_engine(symbol, test_data))
            
            results[symbol] = {
                'individual_models': individual_results,
                'decision_engine': decision_result,
                'test_data_samples': len(test_data)
            }
        
        return results
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get the status of the decision engine."""
        try:
            # Get system status
            system_status = self.engine.get_system_status()
            
            # Get model statuses
            model_statuses = {}
            for model_name, model in [
                ('sentiment_model', self.engine.sentiment_model),
                ('quantitative_model', self.engine.quantitative_model),
                ('ml_ensemble_model', self.engine.ml_ensemble_model),
                ('rl_decider_agent', self.engine.rl_decider_agent)
            ]:
                if hasattr(model, 'get_model_status'):
                    model_statuses[model_name] = model.get_model_status()
                else:
                    model_statuses[model_name] = {'status': 'unknown'}
            
            return {
                'system_status': system_status,
                'model_statuses': model_statuses,
                'engine_initialized': True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get engine status: {e}")
            return {
                'system_status': {'error': str(e)},
                'model_statuses': {},
                'engine_initialized': False
            }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of the integrated decision engine."""
        logger.info("🚀 Starting comprehensive test of integrated decision engine")
        
        # Test symbols
        test_symbols = ['AAPL', 'AMZN', 'GOOGL', 'META', 'NVDA']
        
        # Get engine status
        engine_status = self.get_engine_status()
        
        # Test multiple symbols
        symbol_results = self.test_multiple_symbols(test_symbols)
        
        # Generate summary
        summary = {
            'test_date': datetime.now().isoformat(),
            'engine_status': engine_status,
            'symbol_results': symbol_results,
            'total_symbols_tested': len(test_symbols),
            'successful_tests': sum(1 for result in symbol_results.values() 
                                  if result['decision_engine']['status'] == 'success'),
            'failed_tests': sum(1 for result in symbol_results.values() 
                              if result['decision_engine']['status'] == 'failed')
        }
        
        logger.info("✅ Comprehensive test completed")
        return summary


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Test Integrated Four-Model Decision Engine')
    parser.add_argument('--symbols', type=str, default='AAPL,AMZN,GOOGL,META,NVDA',
                       help='Comma-separated list of symbols to test')
    parser.add_argument('--output', type=str, default='test_results.json',
                       help='Output file for test results')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Initialize tester
    tester = IntegratedEngineTester()
    
    print("🎯 Integrated Four-Model Decision Engine Test")
    print("=" * 60)
    print(f"Test Symbols: {', '.join(symbols)}")
    print("=" * 60)
    
    try:
        # Run comprehensive test
        results = tester.run_comprehensive_test()
        
        # Print results
        print("\n📊 Test Results:")
        print("-" * 40)
        
        # Engine status
        engine_status = results['engine_status']
        print(f"✅ Engine Initialized: {engine_status['engine_initialized']}")
        
        # Model statuses
        model_statuses = engine_status['model_statuses']
        print(f"\n🧠 Model Statuses:")
        for model_name, status in model_statuses.items():
            model_status = status.get('status', 'unknown')
            status_emoji = "✅" if model_status == 'ready' else "⚠️" if model_status == 'unknown' else "❌"
            print(f"   {status_emoji} {model_name}: {model_status}")
        
        # Symbol results
        symbol_results = results['symbol_results']
        print(f"\n📈 Symbol Test Results:")
        for symbol, result in symbol_results.items():
            decision_status = result['decision_engine']['status']
            status_emoji = "✅" if decision_status == 'success' else "❌"
            print(f"   {status_emoji} {symbol}: {decision_status}")
            
            if decision_status == 'success':
                decision = result['decision_engine']['decision']
                final_decision = decision.get('final_decision', {})
                action = final_decision.get('action', 'UNKNOWN')
                confidence = final_decision.get('confidence', 0.0)
                print(f"      Action: {action}, Confidence: {confidence:.3f}")
        
        # Summary
        print(f"\n📊 Test Summary:")
        print(f"   Total Symbols: {results['total_symbols_tested']}")
        print(f"   Successful: {results['successful_tests']}")
        print(f"   Failed: {results['failed_tests']}")
        print(f"   Success Rate: {results['successful_tests']/results['total_symbols_tested']:.1%}")
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.output}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
