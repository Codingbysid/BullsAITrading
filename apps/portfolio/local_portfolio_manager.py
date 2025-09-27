#!/usr/bin/env python3
"""
Local Portfolio Manager with Real Four-Model Decision Engine

This portfolio manager integrates the trained ML ensemble models and four-model
decision engine for real AI recommendations in the local environment.
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

# Import the four-model decision engine
try:
    from decision_engine.four_model_engine import FourModelDecisionEngine
    from models.trained_ml_ensemble import TrainedMLEnsembleModel
    from utils.common_imports import setup_logger
    from utils.data_processing import data_processor
    from utils.performance_metrics import performance_calculator
    from utils.risk_utils import risk_calculator
    DECISION_ENGINE_AVAILABLE = True
    logger = setup_logger(__name__)
except ImportError as e:
    print(f"⚠️  Decision engine not available: {e}")
    DECISION_ENGINE_AVAILABLE = False
    logger = logging.getLogger(__name__)


class LocalPortfolioManager:
    """Local portfolio manager with real AI recommendations."""
    
    def __init__(self, user_id: str = "local_user", portfolio_id: str = "local_portfolio", 
                 initial_capital: float = 100000.0, risk_tolerance: float = 0.05):
        self.user_id = user_id
        self.portfolio_id = portfolio_id
        self.initial_capital = initial_capital
        self.risk_tolerance = risk_tolerance
        
        # Initialize decision engine
        if DECISION_ENGINE_AVAILABLE:
            self.decision_engine = FourModelDecisionEngine()
            self.ml_ensemble = TrainedMLEnsembleModel()
            print("✅ Four-model decision engine initialized")
        else:
            self.decision_engine = None
            self.ml_ensemble = None
            print("⚠️  Using fallback recommendations")
        
        # Portfolio state
        self.portfolio_data = self._load_portfolio()
        self.symbols = ["AAPL", "AMZN", "GOOGL", "META", "NVDA"]
        
        # Performance tracking
        self.trade_history = []
        self.recommendation_history = []
    
    def _load_portfolio(self) -> Dict[str, Any]:
        """Load portfolio data from file."""
        portfolio_path = Path("data/portfolios/local_portfolio.json")
        if portfolio_path.exists():
            with open(portfolio_path, 'r') as f:
                return json.load(f)
        else:
            return self._create_initial_portfolio()
    
    def _create_initial_portfolio(self) -> Dict[str, Any]:
        """Create initial portfolio data."""
        return {
            "portfolio_id": self.portfolio_id,
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "initial_capital": self.initial_capital,
            "current_value": self.initial_capital,
            "cash_balance": self.initial_capital,
            "positions": {},
            "performance_metrics": {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0
            },
            "risk_metrics": {
                "portfolio_risk": 0.0,
                "var_95": 0.0,
                "beta": 0.0,
                "correlation": {}
            }
        }
    
    def _save_portfolio(self):
        """Save portfolio data to file."""
        portfolio_path = Path("data/portfolios/local_portfolio.json")
        portfolio_path.parent.mkdir(parents=True, exist_ok=True)
        with open(portfolio_path, 'w') as f:
            json.dump(self.portfolio_data, f, indent=2)
    
    def _load_market_data(self, symbol: str) -> pd.DataFrame:
        """Load market data for a symbol."""
        data_path = Path(f"data/{symbol}_sample_data.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            # Generate synthetic data if not available
            return self._generate_synthetic_data(symbol)
    
    def _generate_synthetic_data(self, symbol: str) -> pd.DataFrame:
        """Generate synthetic market data."""
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        
        base_prices = {
            "AAPL": 150.0,
            "AMZN": 3200.0,
            "GOOGL": 2800.0,
            "META": 300.0,
            "NVDA": 400.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        np.random.seed(hash(symbol) % 2**32)
        
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, base_price * 0.5))
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            volatility = 0.01
            high = price * (1 + np.random.uniform(0, volatility))
            low = price * (1 - np.random.uniform(0, volatility))
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                "Date": date.strftime('%Y-%m-%d'),
                "Open": round(open_price, 2),
                "High": round(high, 2),
                "Low": round(low, 2),
                "Close": round(price, 2),
                "Volume": volume
            })
        
        return pd.DataFrame(data)
    
    def get_ai_recommendation(self, symbol: str) -> Dict[str, Any]:
        """Get AI recommendation using the four-model decision engine."""
        try:
            if not DECISION_ENGINE_AVAILABLE or not self.decision_engine:
                return self._get_fallback_recommendation(symbol)
            
            # Load market data
            market_data = self._load_market_data(symbol)
            if market_data.empty:
                return self._get_fallback_recommendation(symbol)
            
            # Get recent data for analysis
            recent_data = market_data.tail(30)  # Last 30 days
            
            # Create portfolio state
            portfolio_state = {
                'current_position': self.portfolio_data['positions'].get(symbol, {}).get('shares', 0),
                'portfolio_risk': self.portfolio_data['risk_metrics']['portfolio_risk'],
                'cash_ratio': self.portfolio_data['cash_balance'] / self.portfolio_data['current_value']
            }
            
            # Get recommendation from decision engine
            recommendation = self.decision_engine.generate_trading_decision(
                symbol, recent_data, {}, portfolio_state
            )
            
            # Store recommendation
            self.recommendation_history.append({
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'recommendation': recommendation
            })
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error getting AI recommendation for {symbol}: {e}")
            return self._get_fallback_recommendation(symbol)
    
    def _get_fallback_recommendation(self, symbol: str) -> Dict[str, Any]:
        """Get fallback recommendation when decision engine is not available."""
        # Simple fallback based on recent price movement
        try:
            market_data = self._load_market_data(symbol)
            if len(market_data) < 2:
                return {
                    "action": "HOLD",
                    "confidence": 0.5,
                    "reasoning": "Insufficient data for analysis"
                }
            
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
                "reasoning": f"Price change: {price_change:.2%}, Simple momentum analysis"
            }
            
        except Exception as e:
            return {
                "action": "HOLD",
                "confidence": 0.5,
                "reasoning": f"Fallback error: {str(e)}"
            }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary."""
        return {
            "total_value": self.portfolio_data["current_value"],
            "cash_balance": self.portfolio_data["cash_balance"],
            "total_return": self.portfolio_data["performance_metrics"]["total_return"],
            "risk_level": "Medium" if self.risk_tolerance > 0.03 else "Low",
            "positions": self.portfolio_data["positions"]
        }
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics."""
        return self.portfolio_data["performance_metrics"]
    
    def update_risk_tolerance(self, new_risk: float):
        """Update risk tolerance."""
        self.risk_tolerance = new_risk
        self.portfolio_data["risk_metrics"]["portfolio_risk"] = new_risk
        self._save_portfolio()
    
    def execute_trade(self, symbol: str, action: str, amount: float) -> Dict[str, Any]:
        """Execute a simulated trade."""
        try:
            # Get current price
            market_data = self._load_market_data(symbol)
            current_price = market_data['Close'].iloc[-1]
            
            if action == "BUY":
                shares = amount / current_price
                cost = shares * current_price
                
                if cost <= self.portfolio_data["cash_balance"]:
                    # Update portfolio
                    if symbol in self.portfolio_data["positions"]:
                        old_shares = self.portfolio_data["positions"][symbol]["shares"]
                        old_cost = self.portfolio_data["positions"][symbol]["avg_price"] * old_shares
                        new_shares = old_shares + shares
                        new_avg_price = (old_cost + cost) / new_shares
                        
                        self.portfolio_data["positions"][symbol] = {
                            "shares": new_shares,
                            "avg_price": new_avg_price
                        }
                    else:
                        self.portfolio_data["positions"][symbol] = {
                            "shares": shares,
                            "avg_price": current_price
                        }
                    
                    self.portfolio_data["cash_balance"] -= cost
                    self.portfolio_data["current_value"] = self.portfolio_data["cash_balance"] + sum(
                        pos["shares"] * current_price for pos in self.portfolio_data["positions"].values()
                    )
                    
                    # Record trade
                    self.trade_history.append({
                        "symbol": symbol,
                        "action": action,
                        "shares": shares,
                        "price": current_price,
                        "amount": cost,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    self._save_portfolio()
                    
                    return {
                        "status": "success",
                        "message": f"Bought {shares:.2f} shares of {symbol} at ${current_price:.2f}",
                        "cost": cost
                    }
                else:
                    return {
                        "status": "failed",
                        "message": "Insufficient cash balance"
                    }
            
            elif action == "SELL":
                if symbol in self.portfolio_data["positions"]:
                    position = self.portfolio_data["positions"][symbol]
                    shares_to_sell = min(shares, position["shares"])
                    proceeds = shares_to_sell * current_price
                    
                    # Update portfolio
                    if shares_to_sell == position["shares"]:
                        del self.portfolio_data["positions"][symbol]
                    else:
                        self.portfolio_data["positions"][symbol]["shares"] -= shares_to_sell
                    
                    self.portfolio_data["cash_balance"] += proceeds
                    self.portfolio_data["current_value"] = self.portfolio_data["cash_balance"] + sum(
                        pos["shares"] * current_price for pos in self.portfolio_data["positions"].values()
                    )
                    
                    # Record trade
                    self.trade_history.append({
                        "symbol": symbol,
                        "action": action,
                        "shares": shares_to_sell,
                        "price": current_price,
                        "amount": proceeds,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    self._save_portfolio()
                    
                    return {
                        "status": "success",
                        "message": f"Sold {shares_to_sell:.2f} shares of {symbol} at ${current_price:.2f}",
                        "proceeds": proceeds
                    }
                else:
                    return {
                        "status": "failed",
                        "message": f"No position in {symbol}"
                    }
            
            else:
                return {
                    "status": "failed",
                    "message": f"Invalid action: {action}"
                }
                
        except Exception as e:
            return {
                "status": "failed",
                "message": f"Trade execution error: {str(e)}"
            }


def main():
    """Test the local portfolio manager."""
    print("🧪 Testing Local Portfolio Manager")
    print("=" * 40)
    
    # Initialize manager
    manager = LocalPortfolioManager()
    
    # Test AI recommendations
    print("\n🤖 AI RECOMMENDATIONS:")
    for symbol in manager.symbols:
        recommendation = manager.get_ai_recommendation(symbol)
        print(f"{symbol}: {recommendation['action']} - Confidence: {recommendation['confidence']:.1%}")
        print(f"  Reasoning: {recommendation['reasoning']}")
    
    # Test portfolio summary
    print("\n📊 PORTFOLIO SUMMARY:")
    summary = manager.get_portfolio_summary()
    print(f"Total Value: ${summary['total_value']:,.2f}")
    print(f"Cash Balance: ${summary['cash_balance']:,.2f}")
    print(f"Total Return: {summary['total_return']:.2%}")


if __name__ == "__main__":
    main()
