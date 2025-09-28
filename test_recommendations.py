from src.utils.common_imports import *
import sys
from pathlib import Path
from datetime import datetime
        from decision_engine.four_model_engine import FourModelDecisionEngine
        from models.trained_ml_ensemble import TrainedMLEnsembleModel

#!/usr/bin/env python3
"""
Test Real AI Recommendations

This script tests the real AI recommendations from the four-model decision engine
and trained ML ensemble models to show why you're getting different results.
"""


# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_decision_engine():
    """Test the four-model decision engine directly."""
    print("🧪 Testing Four-Model Decision Engine")
    print("=" * 50)
    
    try:
        
        # Initialize decision engine
        engine = FourModelDecisionEngine()
        ml_ensemble = TrainedMLEnsembleModel()
        
        print("✅ Decision engine initialized")
        print(f"✅ ML ensemble models loaded: {len(ml_ensemble.models)}")
        
        # Test symbols
        symbols = ["AAPL", "AMZN", "GOOGL", "META", "NVDA"]
        
        print("\n🤖 REAL AI RECOMMENDATIONS:")
        print("-" * 40)
        
        for symbol in symbols:
            try:
                # Load market data
                data_path = Path(f"data/{symbol}_sample_data.csv")
                if data_path.exists():
                    df = pd.read_csv(data_path)
                    df['Date'] = pd.to_datetime(df['Date'])
                    
                    # Get recent data
                    recent_data = df.tail(30)
                    
                    # Create portfolio state
                    portfolio_state = {
                        'current_position': 0.0,
                        'portfolio_risk': 0.05,
                        'cash_ratio': 0.7
                    }
                    
                    # Get recommendation
                    recommendation = engine.generate_trading_decision(
                        symbol, recent_data, {}, portfolio_state
                    )
                    
                    print(f"{symbol}: {recommendation.get('action', 'UNKNOWN')}")
                    print(f"  Confidence: {recommendation.get('confidence', 0.0):.1%}")
                    print(f"  Reasoning: {recommendation.get('reasoning', 'No reasoning')}")
                    print()
                    
                else:
                    print(f"{symbol}: No data available")
                    
            except Exception as e:
                print(f"{symbol}: Error - {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Decision engine not available: {e}")
        return False

def test_ml_ensemble():
    """Test the trained ML ensemble model directly."""
    print("\n🧠 Testing Trained ML Ensemble Model")
    print("=" * 50)
    
    try:
        
        # Initialize ML ensemble
        ml_ensemble = TrainedMLEnsembleModel()
        
        print(f"✅ ML ensemble initialized")
        print(f"📊 Models loaded: {list(ml_ensemble.models.keys())}")
        print(f"🔧 Features: {len(ml_ensemble.feature_columns)}")
        
        # Test symbols
        symbols = ["AAPL", "AMZN", "GOOGL", "META", "NVDA"]
        
        print("\n🎯 ML ENSEMBLE PREDICTIONS:")
        print("-" * 40)
        
        for symbol in symbols:
            try:
                # Load market data
                data_path = Path(f"data/{symbol}_sample_data.csv")
                if data_path.exists():
                    df = pd.read_csv(data_path)
                    df['Date'] = pd.to_datetime(df['Date'])
                    
                    # Get recent data
                    recent_data = df.tail(30)
                    
                    # Get prediction
                    prediction = ml_ensemble.predict(recent_data)
                    
                    print(f"{symbol}: Signal {prediction.get('signal', 0.0):.3f}")
                    print(f"  Confidence: {prediction.get('confidence', 0.0):.1%}")
                    print(f"  Reasoning: {prediction.get('reasoning', 'No reasoning')}")
                    
                    # Show individual model predictions
                    metadata = prediction.get('metadata', {})
                    individual_predictions = metadata.get('individual_predictions', {})
                    if individual_predictions:
                        print("  Individual Models:")
                        for model_name, pred in individual_predictions.items():
                            print(f"    {model_name}: {pred:.3f}")
                    print()
                    
                else:
                    print(f"{symbol}: No data available")
                    
            except Exception as e:
                print(f"{symbol}: Error - {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ ML ensemble not available: {e}")
        return False

def test_fallback_recommendations():
    """Test fallback recommendations."""
    print("\n🔄 Testing Fallback Recommendations")
    print("=" * 50)
    
    symbols = ["AAPL", "AMZN", "GOOGL", "META", "NVDA"]
    
    print("📊 FALLBACK RECOMMENDATIONS:")
    print("-" * 40)
    
    for symbol in symbols:
        try:
            # Load market data
            data_path = Path(f"data/{symbol}_sample_data.csv")
            if data_path.exists():
                df = pd.read_csv(data_path)
                
                # Simple momentum analysis
                recent_data = df.tail(5)
                price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
                
                if price_change > 0.02:
                    action = "BUY"
                    confidence = min(0.8, 0.5 + abs(price_change) * 10)
                elif price_change < -0.02:
                    action = "SELL"
                    confidence = min(0.8, 0.5 + abs(price_change) * 10)
                else:
                    action = "HOLD"
                    confidence = 0.6
                
                print(f"{symbol}: {action} - Confidence: {confidence:.1%}")
                print(f"  Price Change: {price_change:.2%}")
                print(f"  Reasoning: Simple momentum analysis")
                print()
                
            else:
                print(f"{symbol}: No data available")
                
        except Exception as e:
            print(f"{symbol}: Error - {e}")

def main():
    """Main test function."""
    print("🚀 QuantAI Recommendation System Test")
    print("=" * 60)
    print("This test shows why you're getting different recommendations")
    print("and demonstrates the real AI vs fallback systems.")
    print("=" * 60)
    
    # Test decision engine
    decision_engine_available = test_decision_engine()
    
    # Test ML ensemble
    ml_ensemble_available = test_ml_ensemble()
    
    # Test fallback
    test_fallback_recommendations()
    
    print("\n📊 SUMMARY:")
    print("=" * 30)
    print(f"Decision Engine Available: {'✅' if decision_engine_available else '❌'}")
    print(f"ML Ensemble Available: {'✅' if ml_ensemble_available else '❌'}")
    
    if not decision_engine_available and not ml_ensemble_available:
        print("\n⚠️  WHY YOU'RE GETTING MOCK RECOMMENDATIONS:")
        print("The four-model decision engine and trained ML ensemble")
        print("are not available due to import issues. The system is")
        print("falling back to simple momentum-based recommendations.")
        print("\n🔧 TO GET REAL AI RECOMMENDATIONS:")
        print("1. Fix the import paths in the decision engine")
        print("2. Ensure all trained models are properly loaded")
        print("3. Verify the four-model architecture is working")
    else:
        print("\n✅ REAL AI RECOMMENDATIONS ARE WORKING!")
        print("The system is using the trained ML ensemble models")
        print("and four-model decision engine for recommendations.")

if __name__ == "__main__":
    main()
