#!/usr/bin/env python3
"""
Test Trained ML Ensemble Model

This script tests the trained ML ensemble model directly without the full decision engine
to avoid dependency issues.

Usage:
    python scripts/test_trained_ensemble.py
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import trained ML ensemble model
from src.models.trained_ml_ensemble import TrainedMLEnsembleModel
from src.utils.common_imports import setup_logger
from src.utils.data_processing import data_processor

logger = setup_logger(__name__)


class TrainedEnsembleTester:
    """Test the trained ML ensemble model."""
    
    def __init__(self):
        self.model = TrainedMLEnsembleModel()
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
    
    def test_model_prediction(self, symbol: str, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Test model prediction for a symbol."""
        logger.info(f"Testing model prediction for {symbol}")
        
        try:
            # Make prediction
            result = self.model.predict(test_data)
            
            return {
                'status': 'success',
                'symbol': symbol,
                'signal': result.signal,
                'confidence': result.confidence,
                'reasoning': result.reasoning,
                'metadata': result.metadata,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Model prediction failed for {symbol}: {e}")
            return {
                'status': 'failed',
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def test_multiple_symbols(self, symbols: List[str]) -> Dict[str, Any]:
        """Test the model with multiple symbols."""
        logger.info(f"Testing model with {len(symbols)} symbols")
        
        results = {}
        
        for symbol in symbols:
            logger.info(f"Testing {symbol}")
            
            # Create test data
            test_data = self.create_test_data(symbol)
            
            # Test model prediction
            prediction_result = self.test_model_prediction(symbol, test_data)
            
            results[symbol] = {
                'prediction': prediction_result,
                'test_data_samples': len(test_data)
            }
        
        return results
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get the status of the trained model."""
        try:
            status = self.model.get_model_status()
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get model status: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of the trained ML ensemble model."""
        logger.info("🚀 Starting comprehensive test of trained ML ensemble model")
        
        # Test symbols
        test_symbols = ['AAPL', 'AMZN', 'GOOGL', 'META', 'NVDA']
        
        # Get model status
        model_status = self.get_model_status()
        
        # Test multiple symbols
        symbol_results = self.test_multiple_symbols(test_symbols)
        
        # Generate summary
        summary = {
            'test_date': datetime.now().isoformat(),
            'model_status': model_status,
            'symbol_results': symbol_results,
            'total_symbols_tested': len(test_symbols),
            'successful_tests': sum(1 for result in symbol_results.values() 
                                  if result['prediction']['status'] == 'success'),
            'failed_tests': sum(1 for result in symbol_results.values() 
                              if result['prediction']['status'] == 'failed')
        }
        
        logger.info("✅ Comprehensive test completed")
        return summary


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Test Trained ML Ensemble Model')
    parser.add_argument('--symbols', type=str, default='AAPL,AMZN,GOOGL,META,NVDA',
                       help='Comma-separated list of symbols to test')
    parser.add_argument('--output', type=str, default='trained_ensemble_test_results.json',
                       help='Output file for test results')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Initialize tester
    tester = TrainedEnsembleTester()
    
    print("🎯 Trained ML Ensemble Model Test")
    print("=" * 60)
    print(f"Test Symbols: {', '.join(symbols)}")
    print("=" * 60)
    
    try:
        # Run comprehensive test
        results = tester.run_comprehensive_test()
        
        # Print results
        print("\n📊 Test Results:")
        print("-" * 40)
        
        # Model status
        model_status = results['model_status']
        print(f"✅ Model Status: {model_status.get('status', 'unknown')}")
        print(f"📊 Models Loaded: {len(model_status.get('models_loaded', []))}")
        print(f"🔧 Feature Columns: {model_status.get('feature_columns_count', 0)}")
        
        # Ensemble weights
        ensemble_weights = model_status.get('ensemble_weights', {})
        print(f"\n⚖️ Ensemble Weights:")
        for model_name, weight in ensemble_weights.items():
            print(f"   {model_name}: {weight:.1%}")
        
        # Symbol results
        symbol_results = results['symbol_results']
        print(f"\n📈 Symbol Test Results:")
        for symbol, result in symbol_results.items():
            prediction_status = result['prediction']['status']
            status_emoji = "✅" if prediction_status == 'success' else "❌"
            print(f"   {status_emoji} {symbol}: {prediction_status}")
            
            if prediction_status == 'success':
                prediction = result['prediction']
                signal = prediction.get('signal', 0.0)
                confidence = prediction.get('confidence', 0.0)
                print(f"      Signal: {signal}, Confidence: {confidence:.3f}")
                
                # Show individual model predictions
                metadata = prediction.get('metadata', {})
                individual_predictions = metadata.get('individual_predictions', {})
                if individual_predictions:
                    print(f"      Individual Predictions:")
                    for model_name, pred in individual_predictions.items():
                        print(f"        {model_name}: {pred:.3f}")
        
        # Summary
        print(f"\n📊 Test Summary:")
        print(f"   Total Symbols: {results['total_symbols_tested']}")
        print(f"   Successful: {results['successful_tests']}")
        print(f"   Failed: {results['failed_tests']}")
        print(f"   Success Rate: {results['successful_tests']/results['total_symbols_tested']:.1%}")
        
        # Performance metrics
        performance_metrics = model_status.get('performance_metrics', {})
        if performance_metrics:
            print(f"\n📊 Model Performance:")
            print(f"   Total Predictions: {performance_metrics.get('total_predictions', 0)}")
            print(f"   Accuracy: {performance_metrics.get('accuracy', 0.0):.3f}")
        
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
