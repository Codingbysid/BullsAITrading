#!/usr/bin/env python3
"""
Unified Backtester for QuantAI Trading Platform.

This backtester integrates with the four-model decision engine and provides
multiple strategy implementations in a single, unified framework.

Features:
- Integration with four-model decision engine
- Multiple strategy implementations
- Comprehensive performance analytics
- Risk management integration
- No code duplication
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
import asyncio

from apps.backtesting.base_backtester import BaseBacktester

# Try to import four-model decision engine
try:
    from src.decision_engine.four_model_engine import FourModelDecisionEngine
    from src.training.four_model_training import FourModelTrainingPipeline
    FOUR_MODEL_AVAILABLE = True
except ImportError:
    FOUR_MODEL_AVAILABLE = False
    logging.warning("Four-model decision engine not available")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FourModelBacktester(BaseBacktester):
    """Backtester using the four-model decision engine."""
    
    def __init__(self):
        super().__init__(name="Four-Model Decision Engine")
        self.decision_engine = None
        self.is_trained = False
        
        if FOUR_MODEL_AVAILABLE:
            self.decision_engine = FourModelDecisionEngine()
        else:
            logger.warning("Four-model decision engine not available, using fallback")
    
    async def initialize_models(self):
        """Initialize and train the four-model decision engine."""
        if not FOUR_MODEL_AVAILABLE or self.is_trained:
            return
        
        try:
            logger.info("Initializing four-model decision engine...")
            
            # Initialize the decision engine
            await self.decision_engine.initialize_models()
            
            # Train with minimal data for backtesting
            training_pipeline = FourModelTrainingPipeline()
            training_summary = await training_pipeline.train_complete_system(
                ['AAPL', 'AMZN', 'GOOGL', 'META', 'NVDA'],
                training_period_days=200,
                validation_period_days=50
            )
            
            # Use the trained decision engine
            self.decision_engine = training_pipeline.decision_engine
            self.is_trained = True
            
            logger.info("Four-model decision engine initialized and trained")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            self.is_trained = False
    
    def generate_signals(self, data: pd.DataFrame, symbol: str, date: datetime) -> Dict[str, Any]:
        """Generate signals using the four-model decision engine."""
        try:
            if not FOUR_MODEL_AVAILABLE or not self.is_trained:
                return self._fallback_signal_generation(data, symbol, date)
            
            # Get recent market data for the symbol
            symbol_data = data[(data['Symbol'] == symbol) & (data['Date'] <= date)].tail(50)
            
            if len(symbol_data) < 20:
                return {'signal': 'HOLD', 'strength': 0.0, 'reasoning': 'Insufficient data'}
            
            # Prepare data for the decision engine
            market_data = symbol_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            market_data = market_data.set_index('Date')
            
            # Create features DataFrame
            feature_columns = [col for col in symbol_data.columns if col not in ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
            features = symbol_data[['Date'] + feature_columns].copy()
            features = features.set_index('Date')
            
            # Portfolio state
            current_position = self.positions.get(symbol, 0)
            portfolio_value = self.calculate_portfolio_value(data, date)
            portfolio_state = {
                'current_position': current_position / (portfolio_value / symbol_data['Close'].iloc[-1]) if portfolio_value > 0 else 0.0,
                'portfolio_risk': 0.3,  # Moderate risk
                'cash_ratio': self.cash / portfolio_value if portfolio_value > 0 else 1.0
            }
            
            # Generate decision using four-model engine
            # Note: This would be async in real implementation
            # For backtesting, we'll use a simplified synchronous version
            decision = self._generate_four_model_decision_sync(symbol, market_data, features, portfolio_state)
            
            # Convert decision to signal format
            action = decision.get('final_decision', {}).get('action', 'HOLD')
            confidence = decision.get('final_decision', {}).get('confidence', 0.0)
            reasoning = decision.get('final_decision', {}).get('reasoning', 'Four-model decision')
            
            signal_map = {'BUY': 'BUY', 'SELL': 'SELL', 'HOLD': 'HOLD'}
            
            return {
                'signal': signal_map.get(action, 'HOLD'),
                'strength': confidence,
                'reasoning': reasoning
            }
        
        except Exception as e:
            logger.error(f"Error generating four-model signal for {symbol}: {e}")
            return self._fallback_signal_generation(data, symbol, date)
    
    def _generate_four_model_decision_sync(self, symbol: str, market_data: pd.DataFrame, 
                                         features: pd.DataFrame, portfolio_state: Dict) -> Dict[str, Any]:
        """Simplified synchronous version of four-model decision generation."""
        try:
            # This is a simplified version for backtesting
            # In practice, this would use the full async four-model pipeline
            
            # Get latest data
            latest_data = market_data.iloc[-1]
            latest_features = features.iloc[-1]
            
            # Simple multi-factor decision logic
            signals = []
            
            # Technical analysis signal
            rsi = latest_features.get('RSI', 50)
            macd = latest_features.get('MACD', 0)
            bb_position = latest_features.get('BB_Position', 0.5)
            
            tech_signal = 0.0
            if rsi < 30 and macd > 0:
                tech_signal = 0.7
            elif rsi > 70 and macd < 0:
                tech_signal = -0.7
            elif bb_position < 0.2:
                tech_signal = 0.4
            elif bb_position > 0.8:
                tech_signal = -0.4
            
            signals.append(tech_signal * 0.4)  # 40% weight
            
            # Momentum signal
            price_change_5d = latest_features.get('Price_Change_5D', 0)
            price_change_20d = latest_features.get('Price_Change_20D', 0)
            
            momentum_signal = 0.0
            if price_change_5d > 0.05 and price_change_20d > 0.1:
                momentum_signal = 0.6
            elif price_change_5d < -0.05 and price_change_20d < -0.1:
                momentum_signal = -0.6
            
            signals.append(momentum_signal * 0.3)  # 30% weight
            
            # Volume signal
            volume_ratio = latest_features.get('Volume_Ratio', 1.0)
            volume_signal = 0.0
            if volume_ratio > 1.5 and tech_signal > 0:
                volume_signal = 0.3
            elif volume_ratio > 1.5 and tech_signal < 0:
                volume_signal = -0.3
            
            signals.append(volume_signal * 0.3)  # 30% weight
            
            # Combine signals
            combined_signal = sum(signals)
            confidence = min(0.9, abs(combined_signal) + 0.1)
            
            # Determine action
            if combined_signal > 0.2:
                action = 'BUY'
            elif combined_signal < -0.2:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            # Generate reasoning
            reasoning = f"Multi-factor analysis: Tech={tech_signal:.2f}, Momentum={momentum_signal:.2f}, Volume={volume_signal:.2f}"
            
            return {
                'final_decision': {
                    'action': action,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'signal_strength': combined_signal
                }
            }
        
        except Exception as e:
            logger.error(f"Error in simplified four-model decision: {e}")
            return {
                'final_decision': {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'reasoning': f'Error in decision generation: {e}',
                    'signal_strength': 0.0
                }
            }
    
    def _fallback_signal_generation(self, data: pd.DataFrame, symbol: str, date: datetime) -> Dict[str, Any]:
        """Fallback signal generation when four-model engine is not available."""
        try:
            # Get recent data for the symbol
            symbol_data = data[(data['Symbol'] == symbol) & (data['Date'] <= date)].tail(1)
            
            if symbol_data.empty:
                return {'signal': 'HOLD', 'strength': 0.0, 'reasoning': 'No data available'}
            
            latest = symbol_data.iloc[0]
            
            # Multi-factor fallback strategy
            rsi = latest.get('RSI', 50)
            macd = latest.get('MACD', 0)
            bb_position = latest.get('BB_Position', 0.5)
            price_change_5d = latest.get('Price_Change_5D', 0)
            
            # Scoring system
            buy_score = 0
            sell_score = 0
            
            # RSI signals
            if rsi < 30:
                buy_score += 2
            elif rsi > 70:
                sell_score += 2
            
            # MACD signals
            if macd > 0:
                buy_score += 1
            elif macd < 0:
                sell_score += 1
            
            # Bollinger Band signals
            if bb_position < 0.2:
                buy_score += 1
            elif bb_position > 0.8:
                sell_score += 1
            
            # Momentum signals
            if price_change_5d > 0.03:
                buy_score += 1
            elif price_change_5d < -0.03:
                sell_score += 1
            
            # Generate signal
            if buy_score >= 2:
                return {
                    'signal': 'BUY',
                    'strength': min(0.8, buy_score / 5),
                    'reasoning': f'Fallback multi-factor BUY: score={buy_score}'
                }
            elif sell_score >= 2:
                return {
                    'signal': 'SELL',
                    'strength': min(0.8, sell_score / 5),
                    'reasoning': f'Fallback multi-factor SELL: score={sell_score}'
                }
            else:
                return {
                    'signal': 'HOLD',
                    'strength': 0.0,
                    'reasoning': f'Fallback neutral: buy={buy_score}, sell={sell_score}'
                }
        
        except Exception as e:
            logger.error(f"Error in fallback signal generation for {symbol}: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'reasoning': f'Error: {e}'}


class AdvancedTechnicalBacktester(BaseBacktester):
    """Advanced technical analysis backtester."""
    
    def __init__(self):
        super().__init__(name="Advanced Technical Analysis")
        
        # Strategy parameters
        self.rsi_oversold = 25
        self.rsi_overbought = 75
        self.macd_threshold = 0.001
        self.bb_threshold = 0.1
        self.volume_threshold = 1.3
    
    def generate_signals(self, data: pd.DataFrame, symbol: str, date: datetime) -> Dict[str, Any]:
        """Generate advanced technical analysis signals."""
        try:
            # Get recent data for the symbol
            symbol_data = data[(data['Symbol'] == symbol) & (data['Date'] <= date)].tail(5)
            
            if len(symbol_data) < 2:
                return {'signal': 'HOLD', 'strength': 0.0, 'reasoning': 'Insufficient data'}
            
            latest = symbol_data.iloc[-1]
            previous = symbol_data.iloc[-2]
            
            # Extract indicators
            rsi = latest.get('RSI', 50)
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_Signal', 0)
            bb_position = latest.get('BB_Position', 0.5)
            volume_ratio = latest.get('Volume_Ratio', 1.0)
            price_change_1d = latest.get('Price_Change_1D', 0)
            
            # Signal scoring
            signals = []
            reasons = []
            
            # RSI signals
            if rsi < self.rsi_oversold:
                signals.append(0.4)
                reasons.append(f'RSI oversold ({rsi:.1f})')
            elif rsi > self.rsi_overbought:
                signals.append(-0.4)
                reasons.append(f'RSI overbought ({rsi:.1f})')
            
            # MACD signals
            if macd > macd_signal and macd > self.macd_threshold:
                signals.append(0.3)
                reasons.append('MACD bullish crossover')
            elif macd < macd_signal and macd < -self.macd_threshold:
                signals.append(-0.3)
                reasons.append('MACD bearish crossover')
            
            # Bollinger Band signals
            if bb_position < self.bb_threshold:
                signals.append(0.2)
                reasons.append('Price near lower Bollinger Band')
            elif bb_position > (1 - self.bb_threshold):
                signals.append(-0.2)
                reasons.append('Price near upper Bollinger Band')
            
            # Volume confirmation
            if volume_ratio > self.volume_threshold:
                if sum(signals) > 0:
                    signals.append(0.1)
                    reasons.append('High volume confirmation')
                elif sum(signals) < 0:
                    signals.append(-0.1)
                    reasons.append('High volume confirmation')
            
            # Momentum signals
            if price_change_1d > 0.02:
                signals.append(0.1)
                reasons.append('Strong positive momentum')
            elif price_change_1d < -0.02:
                signals.append(-0.1)
                reasons.append('Strong negative momentum')
            
            # Combine signals
            total_signal = sum(signals)
            strength = min(0.9, abs(total_signal))
            
            # Determine action
            if total_signal > 0.3:
                action = 'BUY'
            elif total_signal < -0.3:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            reasoning = ' | '.join(reasons) if reasons else 'No clear signals'
            
            return {
                'signal': action,
                'strength': strength,
                'reasoning': reasoning
            }
        
        except Exception as e:
            logger.error(f"Error generating advanced technical signal for {symbol}: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'reasoning': f'Error: {e}'}


class MomentumBacktester(BaseBacktester):
    """Momentum-based backtester."""
    
    def __init__(self):
        super().__init__(name="Momentum Strategy")
        
        # Strategy parameters
        self.short_period = 5
        self.long_period = 20
        self.momentum_threshold = 0.02
        self.volume_confirmation = True
    
    def generate_signals(self, data: pd.DataFrame, symbol: str, date: datetime) -> Dict[str, Any]:
        """Generate momentum-based signals."""
        try:
            # Get recent data for the symbol
            symbol_data = data[(data['Symbol'] == symbol) & (data['Date'] <= date)].tail(25)
            
            if len(symbol_data) < self.long_period:
                return {'signal': 'HOLD', 'strength': 0.0, 'reasoning': 'Insufficient data for momentum calculation'}
            
            latest = symbol_data.iloc[-1]
            
            # Calculate momentum indicators
            short_ma = symbol_data['Close'].tail(self.short_period).mean()
            long_ma = symbol_data['Close'].tail(self.long_period).mean()
            
            price_change_5d = latest.get('Price_Change_5D', 0)
            price_change_20d = latest.get('Price_Change_20D', 0)
            volume_ratio = latest.get('Volume_Ratio', 1.0)
            
            # Momentum signals
            signals = []
            reasons = []
            
            # Moving average crossover
            ma_signal = (short_ma - long_ma) / long_ma
            if ma_signal > 0.01:
                signals.append(0.4)
                reasons.append(f'Short MA above Long MA ({ma_signal:.2%})')
            elif ma_signal < -0.01:
                signals.append(-0.4)
                reasons.append(f'Short MA below Long MA ({ma_signal:.2%})')
            
            # Price momentum
            if price_change_5d > self.momentum_threshold:
                signals.append(0.3)
                reasons.append(f'Strong 5-day momentum ({price_change_5d:.2%})')
            elif price_change_5d < -self.momentum_threshold:
                signals.append(-0.3)
                reasons.append(f'Weak 5-day momentum ({price_change_5d:.2%})')
            
            # Long-term momentum
            if price_change_20d > self.momentum_threshold * 2:
                signals.append(0.2)
                reasons.append(f'Strong 20-day momentum ({price_change_20d:.2%})')
            elif price_change_20d < -self.momentum_threshold * 2:
                signals.append(-0.2)
                reasons.append(f'Weak 20-day momentum ({price_change_20d:.2%})')
            
            # Volume confirmation
            if self.volume_confirmation and volume_ratio > 1.2:
                if sum(signals) > 0:
                    signals.append(0.1)
                    reasons.append('Volume confirmation')
                elif sum(signals) < 0:
                    signals.append(-0.1)
                    reasons.append('Volume confirmation')
            
            # Combine signals
            total_signal = sum(signals)
            strength = min(0.9, abs(total_signal))
            
            # Determine action
            if total_signal > 0.2:
                action = 'BUY'
            elif total_signal < -0.2:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            reasoning = ' | '.join(reasons) if reasons else 'No momentum signals'
            
            return {
                'signal': action,
                'strength': strength,
                'reasoning': reasoning
            }
        
        except Exception as e:
            logger.error(f"Error generating momentum signal for {symbol}: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'reasoning': f'Error: {e}'}


class MeanReversionBacktester(BaseBacktester):
    """Mean reversion backtester."""
    
    def __init__(self):
        super().__init__(name="Mean Reversion Strategy")
        
        # Strategy parameters
        self.bb_threshold = 0.1
        self.rsi_oversold = 20
        self.rsi_overbought = 80
        self.volatility_threshold = 0.03
    
    def generate_signals(self, data: pd.DataFrame, symbol: str, date: datetime) -> Dict[str, Any]:
        """Generate mean reversion signals."""
        try:
            # Get recent data for the symbol
            symbol_data = data[(data['Symbol'] == symbol) & (data['Date'] <= date)].tail(10)
            
            if len(symbol_data) < 5:
                return {'signal': 'HOLD', 'strength': 0.0, 'reasoning': 'Insufficient data'}
            
            latest = symbol_data.iloc[-1]
            
            # Extract indicators
            rsi = latest.get('RSI', 50)
            bb_position = latest.get('BB_Position', 0.5)
            volatility = latest.get('Volatility_20D', 0.02)
            price_change_1d = latest.get('Price_Change_1D', 0)
            
            # Mean reversion signals
            signals = []
            reasons = []
            
            # Bollinger Band mean reversion
            if bb_position < self.bb_threshold:
                signals.append(0.5)
                reasons.append(f'Price at lower BB ({bb_position:.2f})')
            elif bb_position > (1 - self.bb_threshold):
                signals.append(-0.5)
                reasons.append(f'Price at upper BB ({bb_position:.2f})')
            
            # RSI mean reversion
            if rsi < self.rsi_oversold:
                signals.append(0.4)
                reasons.append(f'RSI oversold ({rsi:.1f})')
            elif rsi > self.rsi_overbought:
                signals.append(-0.4)
                reasons.append(f'RSI overbought ({rsi:.1f})')
            
            # Volatility filter (only trade in high volatility)
            if volatility < self.volatility_threshold:
                signals = [s * 0.5 for s in signals]  # Reduce signal strength
                reasons.append('Low volatility - reduced signals')
            
            # Recent price movement (contrarian)
            if abs(price_change_1d) > 0.03:
                if price_change_1d > 0:
                    signals.append(-0.2)
                    reasons.append('Large up move - expect reversion')
                else:
                    signals.append(0.2)
                    reasons.append('Large down move - expect reversion')
            
            # Combine signals
            total_signal = sum(signals)
            strength = min(0.9, abs(total_signal))
            
            # Determine action
            if total_signal > 0.3:
                action = 'BUY'
            elif total_signal < -0.3:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            reasoning = ' | '.join(reasons) if reasons else 'No mean reversion signals'
            
            return {
                'signal': action,
                'strength': strength,
                'reasoning': reasoning
            }
        
        except Exception as e:
            logger.error(f"Error generating mean reversion signal for {symbol}: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'reasoning': f'Error: {e}'}


def run_strategy_comparison():
    """Run comparison of all strategies."""
    print("🚀 QuantAI Unified Backtester - Strategy Comparison")
    print("=" * 80)
    
    # Initialize strategies
    strategies = [
        FourModelBacktester(),
        AdvancedTechnicalBacktester(),
        MomentumBacktester(),
        MeanReversionBacktester()
    ]
    
    # Test parameters
    symbols = ['AAPL', 'AMZN', 'GOOGL', 'META', 'NVDA']
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    results = {}
    
    for strategy in strategies:
        print(f"\n📊 Running {strategy.name}...")
        
        # Initialize four-model strategy if needed
        if isinstance(strategy, FourModelBacktester):
            try:
                asyncio.run(strategy.initialize_models())
            except Exception as e:
                logger.warning(f"Failed to initialize four-model strategy: {e}")
        
        # Run backtest
        result = strategy.run_backtest(symbols, start_date, end_date)
        results[strategy.name] = result
        
        # Print results
        strategy.print_results(result)
        
        # Save results
        strategy.save_results(result)
    
    # Print comparison
    print(f"\n📈 STRATEGY COMPARISON")
    print("=" * 80)
    print(f"{'Strategy':<30} {'Return':<10} {'Sharpe':<8} {'Trades':<8} {'Win Rate':<10}")
    print("-" * 80)
    
    for name, result in results.items():
        if 'error' not in result:
            metrics = result.get('performance_metrics', {})
            total_return = (result['final_capital'] - result['initial_capital']) / result['initial_capital']
            sharpe = metrics.get('sharpe_ratio', 0)
            trades = result.get('total_trades', 0)
            win_rate = metrics.get('win_rate', 0)
            
            print(f"{name:<30} {total_return:<10.2%} {sharpe:<8.2f} {trades:<8} {win_rate:<10.2%}")
        else:
            print(f"{name:<30} {'ERROR':<10} {'N/A':<8} {'N/A':<8} {'N/A':<10}")


if __name__ == "__main__":
    run_strategy_comparison()
