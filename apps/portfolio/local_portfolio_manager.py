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
        """Get AI recommendation using the complete four-model decision engine."""
        try:
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
            
            # Get four-model recommendation
            recommendation = self._get_four_model_recommendation(symbol, recent_data, portfolio_state)
            
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
    
    def _get_four_model_recommendation(self, symbol: str, market_data: pd.DataFrame, portfolio_state: Dict) -> Dict[str, Any]:
        """Get recommendation using the complete four-model approach."""
        try:
            # Model 1: Sentiment Analysis Model (25% weight)
            sentiment_output = self._get_sentiment_analysis(symbol, market_data)
            
            # Model 2: Quantitative Risk Model (25% weight)
            quantitative_output = self._get_quantitative_analysis(symbol, market_data)
            
            # Model 3: ML Ensemble Model (35% weight)
            ml_ensemble_output = self._get_ml_ensemble_analysis(symbol, market_data)
            
            # Model 4: RL Decider Agent (Final decision maker)
            rl_decision = self._get_rl_decision(symbol, market_data, portfolio_state, {
                'sentiment': sentiment_output,
                'quantitative': quantitative_output,
                'ml_ensemble': ml_ensemble_output
            })
            
            # Combine all model outputs
            final_recommendation = {
                "action": rl_decision["action"],
                "confidence": rl_decision["confidence"],
                "reasoning": rl_decision["reasoning"],
                "model_outputs": {
                    "sentiment_model": sentiment_output,
                    "quantitative_model": quantitative_output,
                    "ml_ensemble_model": ml_ensemble_output,
                    "rl_decider_agent": rl_decision
                },
                "four_model_analysis": {
                    "sentiment_weight": 0.25,
                    "quantitative_weight": 0.25,
                    "ml_ensemble_weight": 0.35,
                    "rl_final_weight": 1.0
                }
            }
            
            return final_recommendation
            
        except Exception as e:
            logger.error(f"Error in four-model analysis for {symbol}: {e}")
            return self._get_fallback_recommendation(symbol)
    
    def _get_sentiment_analysis(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Model 1: Sentiment Analysis Model (25% weight)."""
        try:
            # Analyze price momentum for sentiment
            recent_data = market_data.tail(10)
            price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
            
            # Volume sentiment
            volume_change = recent_data['Volume'].pct_change().mean()
            
            # Price volatility sentiment
            volatility = recent_data['Close'].pct_change().std()
            
            # Calculate sentiment score
            sentiment_score = 0.0
            if price_change > 0.02:
                sentiment_score += 0.3
            elif price_change < -0.02:
                sentiment_score -= 0.3
            
            if volume_change > 0.1:
                sentiment_score += 0.2
            elif volume_change < -0.1:
                sentiment_score -= 0.2
            
            if volatility < 0.01:
                sentiment_score += 0.1  # Low volatility is positive
            elif volatility > 0.03:
                sentiment_score -= 0.1  # High volatility is negative
            
            # Determine sentiment signal
            if sentiment_score > 0.2:
                signal = 1.0
                sentiment = "positive"
            elif sentiment_score < -0.2:
                signal = -1.0
                sentiment = "negative"
            else:
                signal = 0.0
                sentiment = "neutral"
            
            return {
                "signal": signal,
                "confidence": min(0.9, 0.5 + abs(sentiment_score)),
                "reasoning": f"Sentiment: {sentiment} (price: {price_change:.2%}, volume: {volume_change:.2%}, volatility: {volatility:.2%})",
                "sentiment_score": sentiment_score
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error for {symbol}: {e}")
            return {"signal": 0.0, "confidence": 0.5, "reasoning": f"Sentiment analysis error: {e}"}
    
    def _get_quantitative_analysis(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Model 2: Quantitative Risk Model (25% weight)."""
        try:
            # Calculate risk metrics
            returns = market_data['Close'].pct_change().dropna()
            
            # Sharpe ratio (simplified)
            mean_return = returns.mean()
            std_return = returns.std()
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Volatility
            volatility = std_return
            
            # Risk-adjusted signal
            risk_score = 0.0
            if sharpe_ratio > 0.5:
                risk_score += 0.3
            elif sharpe_ratio < -0.5:
                risk_score -= 0.3
            
            if max_drawdown > -0.1:
                risk_score += 0.2
            elif max_drawdown < -0.2:
                risk_score -= 0.2
            
            if volatility < 0.02:
                risk_score += 0.1
            elif volatility > 0.05:
                risk_score -= 0.1
            
            # Determine risk signal
            if risk_score > 0.2:
                signal = 1.0
                risk_level = "low"
            elif risk_score < -0.2:
                signal = -1.0
                risk_level = "high"
            else:
                signal = 0.0
                risk_level = "medium"
            
            return {
                "signal": signal,
                "confidence": min(0.9, 0.5 + abs(risk_score)),
                "reasoning": f"Risk: {risk_level} (Sharpe: {sharpe_ratio:.2f}, MaxDD: {max_drawdown:.2%}, Vol: {volatility:.2%})",
                "risk_metrics": {
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "volatility": volatility
                }
            }
            
        except Exception as e:
            logger.error(f"Quantitative analysis error for {symbol}: {e}")
            return {"signal": 0.0, "confidence": 0.5, "reasoning": f"Quantitative analysis error: {e}"}
    
    def _get_ml_ensemble_analysis(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Model 3: ML Ensemble Model (35% weight)."""
        try:
            if not self.ml_ensemble or not self.ml_ensemble.models:
                return self._get_simple_ml_analysis(symbol, market_data)
            
            # Use trained ML ensemble
            ml_prediction = self.ml_ensemble.predict(market_data)
            
            return {
                "signal": ml_prediction.get("signal", 0.0),
                "confidence": ml_prediction.get("confidence", 0.5),
                "reasoning": ml_prediction.get("reasoning", "ML ensemble analysis"),
                "individual_predictions": ml_prediction.get("metadata", {}).get("individual_predictions", {})
            }
            
        except Exception as e:
            logger.error(f"ML ensemble analysis error for {symbol}: {e}")
            return self._get_simple_ml_analysis(symbol, market_data)
    
    def _get_simple_ml_analysis(self, symbol: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Simple ML analysis when trained models are not available."""
        try:
            # Technical indicators
            recent_data = market_data.tail(20)
            
            # Moving averages
            ma_5 = recent_data['Close'].rolling(5).mean().iloc[-1]
            ma_10 = recent_data['Close'].rolling(10).mean().iloc[-1]
            ma_20 = recent_data['Close'].rolling(20).mean().iloc[-1]
            current_price = recent_data['Close'].iloc[-1]
            
            # RSI (simplified)
            price_changes = recent_data['Close'].pct_change().dropna()
            gains = price_changes[price_changes > 0].mean() if len(price_changes[price_changes > 0]) > 0 else 0
            losses = abs(price_changes[price_changes < 0].mean()) if len(price_changes[price_changes < 0]) > 0 else 0
            rsi = 100 - (100 / (1 + gains / losses)) if losses > 0 else 50
            
            # Technical signal
            tech_score = 0.0
            if current_price > ma_5 > ma_10 > ma_20:
                tech_score += 0.4  # Strong uptrend
            elif current_price < ma_5 < ma_10 < ma_20:
                tech_score -= 0.4  # Strong downtrend
            
            if rsi > 70:
                tech_score -= 0.2  # Overbought
            elif rsi < 30:
                tech_score += 0.2  # Oversold
            
            # Determine signal
            if tech_score > 0.2:
                signal = 1.0
                trend = "bullish"
            elif tech_score < -0.2:
                signal = -1.0
                trend = "bearish"
            else:
                signal = 0.0
                trend = "neutral"
            
            return {
                "signal": signal,
                "confidence": min(0.9, 0.5 + abs(tech_score)),
                "reasoning": f"Technical: {trend} (MA trend, RSI: {rsi:.1f})",
                "technical_indicators": {
                    "ma_5": ma_5,
                    "ma_10": ma_10,
                    "ma_20": ma_20,
                    "rsi": rsi
                }
            }
            
        except Exception as e:
            logger.error(f"Simple ML analysis error for {symbol}: {e}")
            return {"signal": 0.0, "confidence": 0.5, "reasoning": f"Simple ML analysis error: {e}"}
    
    def _get_rl_decision(self, symbol: str, market_data: pd.DataFrame, portfolio_state: Dict, model_outputs: Dict) -> Dict[str, Any]:
        """Model 4: RL Decider Agent (Final decision maker)."""
        try:
            # Extract signals from other models
            sentiment_signal = model_outputs['sentiment']['signal']
            sentiment_confidence = model_outputs['sentiment']['confidence']
            
            quantitative_signal = model_outputs['quantitative']['signal']
            quantitative_confidence = model_outputs['quantitative']['confidence']
            
            ml_signal = model_outputs['ml_ensemble']['signal']
            ml_confidence = model_outputs['ml_ensemble']['confidence']
            
            # RL agent decision logic (simplified)
            # Weight the inputs based on confidence and model weights
            weighted_sentiment = sentiment_signal * sentiment_confidence * 0.25
            weighted_quantitative = quantitative_signal * quantitative_confidence * 0.25
            weighted_ml = ml_signal * ml_confidence * 0.35
            
            # Portfolio risk adjustment
            portfolio_risk = portfolio_state.get('portfolio_risk', 0.05)
            cash_ratio = portfolio_state.get('cash_ratio', 0.7)
            
            # Risk adjustment factor
            risk_adjustment = 1.0 - (portfolio_risk * 2)  # Reduce signal strength if high risk
            cash_adjustment = 1.0 + (cash_ratio - 0.5) * 0.5  # Increase signal if more cash available
            
            # Final weighted decision
            final_signal = (weighted_sentiment + weighted_quantitative + weighted_ml) * risk_adjustment * cash_adjustment
            
            # Determine action
            if final_signal > 0.3:
                action = "BUY"
                confidence = min(0.95, 0.6 + abs(final_signal) * 0.5)
            elif final_signal < -0.3:
                action = "SELL"
                confidence = min(0.95, 0.6 + abs(final_signal) * 0.5)
            else:
                action = "HOLD"
                confidence = 0.7
            
            # Create comprehensive reasoning
            reasoning = f"RL Decision: Sentiment={sentiment_signal:.2f}({sentiment_confidence:.1%}), " \
                       f"Quant={quantitative_signal:.2f}({quantitative_confidence:.1%}), " \
                       f"ML={ml_signal:.2f}({ml_confidence:.1%}), " \
                       f"Final={final_signal:.2f}, Risk={portfolio_risk:.1%}, Cash={cash_ratio:.1%}"
            
            return {
                "action": action,
                "confidence": confidence,
                "reasoning": reasoning,
                "final_signal": final_signal,
                "risk_adjustment": risk_adjustment,
                "cash_adjustment": cash_adjustment
            }
            
        except Exception as e:
            logger.error(f"RL decision error for {symbol}: {e}")
            return {
                "action": "HOLD",
                "confidence": 0.5,
                "reasoning": f"RL decision error: {e}",
                "final_signal": 0.0
            }
    
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
