#!/usr/bin/env python3
"""
Unified Backtesting System - DRY Principle Implementation

This module consolidates all backtesting functionality into a single, unified system
that eliminates code duplication and provides a single source of truth for all
backtesting operations across the QuantAI platform.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from src.utils.common_imports import *
from src.utils.performance_metrics import PerformanceCalculator
from src.utils.data_processing import DataProcessor
from src.utils.risk_utils import RiskCalculator

class UnifiedBacktestingSystem:
    """
    Unified backtesting system implementing DRY principle.
    Consolidates all backtesting functionality into a single, maintainable system.
    """
    
    def __init__(self, initial_capital: float = 100000, risk_free_rate: float = 0.02):
        """Initialize unified backtesting system."""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
        # Use unified utilities
        self.data_processor = DataProcessor()
        self.perf_calc = PerformanceCalculator(risk_free_rate)
        self.risk_calc = RiskCalculator()
        self.logger = setup_logger("unified_backtester")
        
        # Backtesting configuration
        self.config = {
            'max_position_size': 0.30,
            'max_portfolio_risk': 0.15,
            'transaction_cost': 0.001,  # 0.1% transaction cost
            'slippage': 0.0005,  # 0.05% slippage
            'min_trade_size': 100,  # Minimum trade size in dollars
        }
    
    def load_market_data(self, symbol: str, use_synthetic: bool = False, 
                        days: int = 1000) -> pd.DataFrame:
        """Load market data using unified data processor."""
        self.logger.info(f"Loading market data for {symbol}")
        
        if use_synthetic:
            data = self.data_processor.create_synthetic_data(symbol, days)
        else:
            # Try to load real data
            data_path = Path(f"data/{symbol}_sample_data.csv")
            if data_path.exists():
                data = pd.read_csv(data_path)
                data = self.data_processor.validate_and_clean(data, symbol)
            else:
                self.logger.warning(f"No real data found for {symbol}, using synthetic data")
                data = self.data_processor.create_synthetic_data(symbol, days)
        
        # Add technical indicators
        data = self.data_processor.add_technical_indicators(data)
        
        self.logger.info(f"Loaded {len(data)} records for {symbol}")
        return data
    
    def generate_trading_signals(self, data: pd.DataFrame, strategy: str = "unified") -> pd.Series:
        """Generate trading signals using specified strategy."""
        strategies = {
            'unified': self._unified_strategy,
            'momentum': self._momentum_strategy,
            'mean_reversion': self._mean_reversion_strategy,
            'ml_ensemble': self._ml_ensemble_strategy,
            'risk_aware': self._risk_aware_strategy
        }
        
        strategy_func = strategies.get(strategy, self._unified_strategy)
        signals = strategy_func(data)
        
        self.logger.info(f"Generated {len(signals[signals != 0])} signals using {strategy} strategy")
        return signals
    
    def _unified_strategy(self, data: pd.DataFrame) -> pd.Series:
        """Unified strategy combining multiple approaches."""
        signals = pd.Series(0, index=data.index)
        
        # Technical indicators
        rsi = data.get('RSI', 50)
        sma_20 = data.get('SMA_20', data['Close'])
        sma_50 = data.get('SMA_50', data['Close'])
        macd = data.get('MACD', 0)
        bb_position = data.get('BB_Position', 0.5)
        volume_ratio = data.get('Volume_Ratio', 1.0)
        
        # Buy conditions (multiple criteria)
        buy_condition = (
            (rsi < 30) &  # Oversold
            (data['Close'] > sma_20) &  # Above 20-day MA
            (data['Close'] > sma_50) &  # Above 50-day MA
            (macd > 0) &  # Positive MACD
            (bb_position < 0.2) &  # Near lower Bollinger Band
            (volume_ratio > 1.2)  # High volume
        )
        
        # Sell conditions
        sell_condition = (
            (rsi > 70) |  # Overbought
            (data['Close'] < sma_20) |  # Below 20-day MA
            (macd < 0) |  # Negative MACD
            (bb_position > 0.8)  # Near upper Bollinger Band
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    def _momentum_strategy(self, data: pd.DataFrame) -> pd.Series:
        """Momentum-based strategy."""
        signals = pd.Series(0, index=data.index)
        
        # Price momentum
        price_change_5d = data['Close'].pct_change(5)
        price_change_20d = data['Close'].pct_change(20)
        volume_ratio = data.get('Volume_Ratio', 1.0)
        
        # Buy: Strong upward momentum
        buy_condition = (
            (price_change_5d > 0.02) &  # 5-day return > 2%
            (price_change_20d > 0.05) &  # 20-day return > 5%
            (volume_ratio > 1.5)  # High volume
        )
        
        # Sell: Momentum reversal
        sell_condition = (
            (price_change_5d < -0.02) |  # 5-day return < -2%
            (price_change_20d < -0.05)  # 20-day return < -5%
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    def _mean_reversion_strategy(self, data: pd.DataFrame) -> pd.Series:
        """Mean reversion strategy."""
        signals = pd.Series(0, index=data.index)
        
        rsi = data.get('RSI', 50)
        bb_position = data.get('BB_Position', 0.5)
        sma_20 = data.get('SMA_20', data['Close'])
        
        # Buy: Oversold conditions
        buy_condition = (
            (rsi < 25) &  # Very oversold
            (bb_position < 0.1) &  # Near lower Bollinger Band
            (data['Close'] < sma_20 * 0.95)  # Below 20-day MA
        )
        
        # Sell: Overbought conditions
        sell_condition = (
            (rsi > 75) &  # Very overbought
            (bb_position > 0.9) &  # Near upper Bollinger Band
            (data['Close'] > sma_20 * 1.05)  # Above 20-day MA
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        
        return signals
    
    def _ml_ensemble_strategy(self, data: pd.DataFrame) -> pd.Series:
        """ML ensemble strategy (simplified)."""
        signals = pd.Series(0, index=data.index)
        
        # Simplified ML features
        features = [
            'RSI', 'MACD', 'BB_Position', 'Volume_Ratio',
            'Price_Change_1D', 'Price_Change_5D', 'Volatility_20D'
        ]
        
        # Simple ensemble logic
        feature_scores = []
        for feature in features:
            if feature in data.columns:
                # Normalize feature
                feature_data = data[feature].fillna(0)
                if feature in ['RSI', 'BB_Position']:
                    # RSI and BB_Position are already normalized
                    feature_scores.append(feature_data)
                else:
                    # Normalize other features
                    feature_scores.append((feature_data - feature_data.mean()) / feature_data.std())
        
        if feature_scores:
            # Simple ensemble voting
            ensemble_score = pd.concat(feature_scores, axis=1).mean(axis=1)
            
            # Generate signals based on ensemble score
            signals[ensemble_score > 0.5] = 1
            signals[ensemble_score < -0.5] = -1
        
        return signals
    
    def _risk_aware_strategy(self, data: pd.DataFrame) -> pd.Series:
        """Risk-aware strategy using risk metrics."""
        signals = pd.Series(0, index=data.index)
        
        # Calculate rolling risk metrics
        returns = data['Close'].pct_change()
        rolling_vol = returns.rolling(20).std()
        rolling_sharpe = returns.rolling(20).mean() / rolling_vol
        
        # Risk-adjusted signals
        base_signals = self._unified_strategy(data)
        
        # Adjust signals based on risk
        risk_adjustment = np.where(rolling_vol > rolling_vol.quantile(0.8), 0.5, 1.0)
        risk_adjustment = np.where(rolling_sharpe < 0, 0.3, risk_adjustment)
        
        signals = base_signals * risk_adjustment
        
        return signals
    
    def execute_backtest(self, data: pd.DataFrame, signals: pd.Series, 
                        strategy: str = "unified") -> Dict[str, Any]:
        """Execute backtest using unified risk management."""
        self.logger.info(f"Executing backtest with {strategy} strategy")
        
        # Reset state
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
        # Execute trades
        for date, signal in signals.items():
            if signal != 0 and date in data.index:
                self._execute_trade(date, signal, data.loc[date])
        
        # Calculate performance metrics
        returns = self._calculate_portfolio_returns(data)
        metrics = self.perf_calc.calculate_comprehensive_metrics(returns)
        
        # Generate comprehensive results
        results = {
            'strategy': strategy,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': (self.current_capital / self.initial_capital) - 1,
            'trades_count': len(self.trades),
            'metrics': metrics,
            'trade_history': self.trades,
            'portfolio_history': self.portfolio_history,
            'unified_utilities_used': {
                'data_processor': True,
                'performance_calculator': True,
                'risk_calculator': True,
                'common_imports': True
            }
        }
        
        self.logger.info(f"Backtest completed: {results['total_return']:.2%} return, {results['trades_count']} trades")
        return results
    
    def _execute_trade(self, date: pd.Timestamp, signal: float, market_data: pd.Series):
        """Execute trade with unified risk management."""
        price = market_data['Close']
        
        # Calculate position size using risk management
        position_size = self.risk_calc.calculate_position_size(
            signal_strength=abs(signal),
            confidence=0.7,  # Default confidence
            portfolio_value=self.current_capital,
            max_position=self.config['max_position_size']
        )
        
        # Apply transaction costs and slippage
        effective_price = price * (1 + self.config['slippage'] * np.sign(signal))
        transaction_cost = self.config['transaction_cost']
        
        # Calculate shares
        trade_value = position_size * self.current_capital
        shares = int(trade_value / effective_price)
        
        if shares > 0 and trade_value >= self.config['min_trade_size']:
            # Execute trade
            total_cost = shares * effective_price * (1 + transaction_cost)
            
            if signal > 0:  # Buy
                if total_cost <= self.current_capital:
                    self.current_capital -= total_cost
                    self.positions[date] = {
                        'shares': shares,
                        'price': effective_price,
                        'value': total_cost,
                        'type': 'buy'
                    }
            else:  # Sell
                # For simplicity, assume we can always sell
                proceeds = shares * effective_price * (1 - transaction_cost)
                self.current_capital += proceeds
                self.positions[date] = {
                    'shares': -shares,
                    'price': effective_price,
                    'value': proceeds,
                    'type': 'sell'
                }
            
            # Record trade
            self.trades.append({
                'date': date,
                'signal': signal,
                'shares': shares,
                'price': effective_price,
                'value': total_cost if signal > 0 else proceeds,
                'type': 'buy' if signal > 0 else 'sell'
            })
            
            # Update portfolio history
            self.portfolio_history.append({
                'date': date,
                'capital': self.current_capital,
                'positions': len(self.positions),
                'signal': signal
            })
    
    def _calculate_portfolio_returns(self, data: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns for performance analysis."""
        if not self.portfolio_history:
            return pd.Series(dtype=float)
        
        # Create portfolio value series
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df = portfolio_df.set_index('date')
        
        # Calculate returns
        returns = portfolio_df['capital'].pct_change().dropna()
        
        return returns
    
    def run_comprehensive_backtest(self, symbol: str, strategies: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive backtest with multiple strategies."""
        if strategies is None:
            strategies = ['unified', 'momentum', 'mean_reversion', 'ml_ensemble', 'risk_aware']
        
        self.logger.info(f"Running comprehensive backtest for {symbol}")
        
        # Load data
        data = self.load_market_data(symbol, use_synthetic=True)
        
        results = {}
        for strategy in strategies:
            self.logger.info(f"Testing {strategy} strategy")
            
            # Generate signals
            signals = self.generate_trading_signals(data, strategy)
            
            # Execute backtest
            strategy_results = self.execute_backtest(data, signals, strategy)
            results[strategy] = strategy_results
        
        # Compare strategies
        comparison = self._compare_strategies(results)
        
        return {
            'symbol': symbol,
            'strategies_tested': strategies,
            'individual_results': results,
            'comparison': comparison,
            'best_strategy': max(results.keys(), key=lambda k: results[k]['total_return']),
            'unified_system': True
        }
    
    def _compare_strategies(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare strategy performance."""
        comparison = {}
        
        for strategy, result in results.items():
            comparison[strategy] = {
                'total_return': result['total_return'],
                'sharpe_ratio': result['metrics'].get('sharpe_ratio', 0),
                'max_drawdown': result['metrics'].get('max_drawdown', 0),
                'trades_count': result['trades_count'],
                'win_rate': result['metrics'].get('win_rate', 0)
            }
        
        return comparison

def main():
    """Main function to demonstrate unified backtesting system."""
    print("🚀 QuantAI Unified Backtesting System")
    print("=" * 50)
    
    # Initialize unified system
    backtester = UnifiedBacktestingSystem()
    
    # Test symbols
    symbols = ["AAPL", "AMZN", "GOOGL", "META", "NVDA"]
    
    for symbol in symbols:
        print(f"\n📊 Testing {symbol}")
        print("-" * 30)
        
        # Run comprehensive backtest
        results = backtester.run_comprehensive_backtest(symbol)
        
        print(f"✅ {symbol} backtest completed")
        print(f"   Best strategy: {results['best_strategy']}")
        print(f"   Best return: {results['individual_results'][results['best_strategy']]['total_return']:.2%}")
        print(f"   Total trades: {results['individual_results'][results['best_strategy']]['trades_count']}")
    
    print(f"\n🎉 Unified Backtesting System Demo Complete!")
    print(f"✅ All strategies tested using unified utilities")
    print(f"✅ DRY principle implemented")
    print(f"✅ Single source of truth for backtesting")

if __name__ == "__main__":
    main()
