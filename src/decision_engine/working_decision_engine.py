from src.utils.common_imports import *
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
            import sys
            from pathlib import Path
            from models.trained_ml_ensemble import TrainedMLEnsembleModel

#!/usr/bin/env python3
"""
Working Decision Engine for Local Environment

This is a simplified version of the four-model decision engine
that works in the local environment without import issues.
"""


# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

class WorkingDecisionEngine:
    """Working decision engine for local environment."""
    
    def __init__(self):
        self.name = "WorkingDecisionEngine"
        self.models_loaded = False
        self._load_models()
    
    def _load_models(self):
        """Load the trained ML ensemble models."""
        try:
            # Load ML ensemble models
            models_dir = Path("models")
            if models_dir.exists():
                # Check if models exist
                model_files = [
                    "linear_model_model.json",
                    "naive_bayes_model.json", 
                    "decision_tree_model.json",
                    "simple_ensemble_metadata.json"
                ]
                
                all_exist = all((models_dir / f).exists() for f in model_files)
                if all_exist:
                    self.models_loaded = True
                    print("✅ Trained ML models loaded successfully")
                else:
                    print("⚠️  Some trained models missing, using fallback")
            else:
                print("⚠️  Models directory not found, using fallback")
                
        except Exception as e:
            print(f"⚠️  Error loading models: {e}")
    
    def generate_trading_decision(self, symbol: str, market_data: pd.DataFrame, 
                                market_state: Dict, portfolio_state: Dict) -> Dict[str, Any]:
        """Generate trading decision using available models."""
        
        if not self.models_loaded:
            return self._get_fallback_decision(symbol, market_data)
        
        try:
            # Load the trained ML ensemble
            sys.path.append(str(Path(__file__).parent.parent))
            ml_ensemble = TrainedMLEnsembleModel()
            
            # Get ML ensemble prediction
            ml_prediction = ml_ensemble.predict(market_data)
            
            # Get sentiment analysis (simplified)
            sentiment_signal = self._get_sentiment_signal(symbol, market_data)
            
            # Get quantitative risk signal
            risk_signal = self._get_risk_signal(symbol, market_data)
            
            # Combine signals
            final_signal = self._combine_signals(
                ml_prediction, sentiment_signal, risk_signal
            )
            
            return {
                "action": final_signal["action"],
                "confidence": final_signal["confidence"],
                "reasoning": final_signal["reasoning"],
                "model_outputs": {
                    "ml_ensemble": ml_prediction,
                    "sentiment": sentiment_signal,
                    "risk": risk_signal
                }
            }
            
        except Exception as e:
            print(f"⚠️  Error in decision engine: {e}")
            return self._get_fallback_decision(symbol, market_data)
    
    def _get_sentiment_signal(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get sentiment signal (simplified)."""
        # Simple sentiment based on recent price movement
        recent_data = market_data.tail(5)
        price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
        
        if price_change > 0.01:
            sentiment = "positive"
            signal = 0.7
        elif price_change < -0.01:
            sentiment = "negative"
            signal = -0.7
        else:
            sentiment = "neutral"
            signal = 0.0
        
        return {
            "signal": signal,
            "confidence": 0.6,
            "reasoning": f"Sentiment: {sentiment} (price change: {price_change:.2%})"
        }
    
    def _get_risk_signal(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get risk signal (simplified)."""
        # Simple risk based on volatility
        recent_data = market_data.tail(10)
        returns = recent_data['Close'].pct_change().dropna()
        volatility = returns.std()
        
        if volatility > 0.03:
            risk_level = "high"
            signal = -0.3
        elif volatility < 0.01:
            risk_level = "low"
            signal = 0.3
        else:
            risk_level = "medium"
            signal = 0.0
        
        return {
            "signal": signal,
            "confidence": 0.7,
            "reasoning": f"Risk: {risk_level} (volatility: {volatility:.2%})"
        }
    
    def _combine_signals(self, ml_prediction: Dict, sentiment: Dict, risk: Dict) -> Dict[str, Any]:
        """Combine all signals into final decision."""
        
        # Extract signals
        ml_signal = ml_prediction.get("signal", 0.0)
        sentiment_signal = sentiment.get("signal", 0.0)
        risk_signal = risk.get("signal", 0.0)
        
        # Weighted combination
        weights = {"ml": 0.5, "sentiment": 0.3, "risk": 0.2}
        combined_signal = (
            weights["ml"] * ml_signal +
            weights["sentiment"] * sentiment_signal +
            weights["risk"] * risk_signal
        )
        
        # Determine action
        if combined_signal > 0.3:
            action = "BUY"
            confidence = min(0.9, 0.5 + abs(combined_signal))
        elif combined_signal < -0.3:
            action = "SELL"
            confidence = min(0.9, 0.5 + abs(combined_signal))
        else:
            action = "HOLD"
            confidence = 0.6
        
        # Create reasoning
        reasoning = f"ML: {ml_signal:.3f}, Sentiment: {sentiment_signal:.3f}, Risk: {risk_signal:.3f}, Final: {combined_signal:.3f}"
        
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning
        }
    
    def _get_fallback_decision(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get fallback decision when models are not available."""
        # Simple momentum-based decision
        recent_data = market_data.tail(5)
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
        
        return {
            "action": action,
            "confidence": confidence,
            "reasoning": f"Fallback momentum analysis (price change: {price_change:.2%})",
            "model_outputs": {}
        }

def main():
    """Test the working decision engine."""
    print("🧪 Testing Working Decision Engine")
    print("=" * 40)
    
    # Initialize engine
    engine = WorkingDecisionEngine()
    
    # Test with sample data
    symbols = ["AAPL", "AMZN", "GOOGL", "META", "NVDA"]
    
    for symbol in symbols:
        try:
            # Load market data
            data_path = Path(f"data/{symbol}_sample_data.csv")
            if data_path.exists():
                df = pd.read_csv(data_path)
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Get recent data
                recent_data = df.tail(30)
                
                # Get decision
                decision = engine.generate_trading_decision(
                    symbol, recent_data, {}, {}
                )
                
                print(f"{symbol}: {decision['action']} - Confidence: {decision['confidence']:.1%}")
                print(f"  Reasoning: {decision['reasoning']}")
                print()
                
        except Exception as e:
            print(f"{symbol}: Error - {e}")

if __name__ == "__main__":
    main()
