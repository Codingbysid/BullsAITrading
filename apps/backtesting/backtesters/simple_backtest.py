#!/usr/bin/env python3
"""
Simple backtester that works around scipy issues.

This script provides basic backtesting functionality without scipy dependencies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


class SimpleBacktester:
    """Simple backtester without scipy dependencies."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
    def create_synthetic_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Create synthetic market data for backtesting."""
        print("📊 Creating synthetic market data...")
        
        # Generate dates
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        all_data = []
        
        for symbol in symbols:
            # Generate synthetic price data
            np.random.seed(42 + hash(symbol) % 1000)  # Different seed for each symbol
            
            # Start with a base price
            base_price = 100 + hash(symbol) % 100
            
            # Generate random walk for price
            returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
            prices = [base_price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(new_price)
            
            # Create OHLCV data
            for i, (date, close) in enumerate(zip(dates, prices)):
                # Generate OHLC from close price
                volatility = np.random.uniform(0.01, 0.03)
                high = close * (1 + volatility)
                low = close * (1 - volatility)
                open_price = close * (1 + np.random.uniform(-0.01, 0.01))
                
                # Generate volume
                volume = np.random.randint(1000000, 10000000)
                
                all_data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume
                })
        
        df = pd.DataFrame(all_data)
        df.set_index('timestamp', inplace=True)
        
        print(f"✅ Created {len(df)} records for {len(symbols)} symbols")
        return df
    
    def create_simple_strategy(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Create a simple trading strategy."""
        symbol_data = data[data['symbol'] == symbol].copy()
        
        if len(symbol_data) < 20:
            return {'signal': 'HOLD', 'strength': 0.0}
        
        # Calculate simple indicators
        symbol_data['sma_5'] = symbol_data['close'].rolling(5).mean()
        symbol_data['sma_20'] = symbol_data['close'].rolling(20).mean()
        symbol_data['rsi'] = self._calculate_rsi(symbol_data['close'])
        symbol_data['bb_position'] = self._calculate_bb_position(symbol_data['close'])
        
        # Get latest values
        latest = symbol_data.iloc[-1]
        prev = symbol_data.iloc[-2]
        
        # IMPROVED strategy: More aggressive thresholds for more trades
        buy_conditions = [
            latest['rsi'] < 40,  # LOWERED from 30 to 40 (less oversold)
            latest['close'] > latest['sma_5'],  # Above short MA
            latest['bb_position'] < 0.3,  # LOWERED from 0.2 to 0.3
            latest['close'] > prev['close']  # Price increasing
        ]
        
        sell_conditions = [
            latest['rsi'] > 60,  # LOWERED from 70 to 60 (less overbought)
            latest['close'] < latest['sma_5'],  # Below short MA
            latest['bb_position'] > 0.7,  # LOWERED from 0.8 to 0.7
            latest['close'] < prev['close']  # Price decreasing
        ]
        
        # Count conditions
        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)
        
        # Generate signal - IMPROVED THRESHOLDS for more trades
        if buy_score >= 1:  # LOWERED from 3 to 1 (much more sensitive)
            return {'signal': 'BUY', 'strength': min(0.3, buy_score / 4)}  # INCREASED max strength
        elif sell_score >= 1:  # LOWERED from 3 to 1 (much more sensitive)
            return {'signal': 'SELL', 'strength': min(0.3, sell_score / 4)}  # INCREASED max strength
        else:
            return {'signal': 'HOLD', 'strength': 0.0}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI without scipy."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Fill NaN with neutral RSI
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Bollinger Band position without scipy."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        
        return bb_position.fillna(0.5)  # Fill NaN with neutral position
    
    def run_backtest(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run the backtest."""
        print("🚀 Starting Simple Backtest")
        print("=" * 50)
        print(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"📊 Symbols: {symbols}")
        print()
        
        try:
            # Create synthetic data
            data = self.create_synthetic_data(symbols, start_date, end_date)
            
            # Group by date
            daily_data = data.groupby(data.index.date)
            
            print("📈 Running backtest loop...")
            
            for date, day_data in daily_data:
                current_date = pd.Timestamp(date)
                
                # Skip weekends
                if current_date.weekday() >= 5:
                    continue
                
                # Process each symbol
                for symbol in symbols:
                    symbol_day_data = day_data[day_data['symbol'] == symbol]
                    
                    if symbol_day_data.empty:
                        continue
                    
                    # Generate signal
                    signal = self.create_simple_strategy(data, symbol)
                    
                    if signal['signal'] != 'HOLD':
                        self._execute_trade(symbol, signal, symbol_day_data.iloc[-1], current_date)
                
                # Update portfolio history
                self._update_portfolio_history(current_date)
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
            
            print("✅ Backtest completed successfully!")
            
            return {
                'status': 'success',
                'initial_capital': self.initial_capital,
                'final_capital': self._get_total_value(),
                'total_return': self._get_total_return(),
                'performance_metrics': self.performance_metrics,
                'trades': len(self.trades),
                'portfolio_history': self.portfolio_history
            }
            
        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _execute_trade(self, symbol: str, signal: Dict[str, Any], market_data: pd.Series, date: datetime):
        """Execute a trade."""
        try:
            action = signal['signal']
            price = market_data['close']
            strength = signal['strength']
            
            # Calculate position size - IMPROVED FOR MORE ACTIVITY
            max_position_value = self._get_total_value() * 0.3 * strength  # INCREASED from 20% to 30%
            quantity = int(max_position_value / price)
            
            if quantity == 0:
                return
            
            # Apply transaction costs
            commission = quantity * price * 0.001  # 0.1% commission
            slippage = quantity * price * 0.0005  # 0.05% slippage
            
            if action == 'BUY':
                total_cost = (quantity * price) + commission + slippage
                
                if total_cost <= self.cash:
                    self.cash -= total_cost
                    
                    if symbol in self.positions:
                        # Update existing position
                        old_quantity = self.positions[symbol]['quantity']
                        old_avg_price = self.positions[symbol]['avg_price']
                        new_quantity = old_quantity + quantity
                        new_avg_price = ((old_quantity * old_avg_price) + (quantity * price)) / new_quantity
                        
                        self.positions[symbol] = {
                            'quantity': new_quantity,
                            'avg_price': new_avg_price,
                            'current_price': price
                        }
                    else:
                        # Create new position
                        self.positions[symbol] = {
                            'quantity': quantity,
                            'avg_price': price,
                            'current_price': price
                        }
                    
                    # Record trade
                    self.trades.append({
                        'timestamp': date,
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'price': price,
                        'value': quantity * price,
                        'commission': commission,
                        'slippage': slippage
                    })
            
            elif action == 'SELL':
                if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
                    # Update position
                    self.positions[symbol]['quantity'] -= quantity
                    self.cash += (quantity * price) - commission - slippage
                    
                    # Remove position if quantity is zero
                    if self.positions[symbol]['quantity'] == 0:
                        del self.positions[symbol]
                    
                    # Record trade
                    self.trades.append({
                        'timestamp': date,
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'price': price,
                        'value': quantity * price,
                        'commission': commission,
                        'slippage': slippage
                    })
            
        except Exception as e:
            print(f"Error executing trade for {symbol}: {e}")
    
    def _get_total_value(self) -> float:
        """Calculate total portfolio value."""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            total_value += position['quantity'] * position['current_price']
        
        return total_value
    
    def _get_total_return(self) -> float:
        """Calculate total return."""
        return (self._get_total_value() / self.initial_capital) - 1
    
    def _update_portfolio_history(self, date: datetime):
        """Update portfolio history."""
        total_value = self._get_total_value()
        
        # Update position prices (simplified - would use actual market data)
        for symbol, position in self.positions.items():
            # Random price movement
            price_change = np.random.normal(0, 0.01)
            position['current_price'] *= (1 + price_change)
        
        # Calculate returns
        if self.portfolio_history:
            prev_value = self.portfolio_history[-1]['total_value']
            daily_return = (total_value - prev_value) / prev_value
        else:
            daily_return = 0.0
        
        portfolio_snapshot = {
            'timestamp': date,
            'total_value': total_value,
            'cash': self.cash,
            'daily_return': daily_return,
            'cumulative_return': (total_value / self.initial_capital) - 1,
            'positions': self.positions.copy()
        }
        
        self.portfolio_history.append(portfolio_snapshot)
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics."""
        print("📊 Calculating performance metrics...")
        
        if not self.portfolio_history:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.portfolio_history)
        df.set_index('timestamp', inplace=True)
        
        # Calculate metrics
        total_return = self._get_total_return()
        annualized_return = (1 + total_return) ** (252 / len(df)) - 1
        
        # Volatility
        returns = df['daily_return'].dropna()
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = [t for t in self.trades if t['action'] == 'SELL' and t['value'] > 0]
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t['value'] for t in winning_trades)
        gross_loss = sum(abs(t['value']) for t in self.trades if t['value'] < 0)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        self.performance_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'final_value': self._get_total_value()
        }
        
        print(f"✅ Performance metrics calculated")


def main():
    """Main function."""
    print("🚀 QuantAI Trading Platform - Simple Backtester")
    print("=" * 60)
    print("This backtester works around scipy dependency issues.")
    print()
    
    # Configuration
    symbols = ['AAPL', 'GOOGL', 'NVDA', 'META', 'AMZN']
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    initial_capital = 100000
    
    # Initialize backtester
    backtester = SimpleBacktester(initial_capital=initial_capital)
    
    # Run backtest
    start_time = datetime.now()
    results = backtester.run_backtest(symbols, start_date, end_date)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    
    if results['status'] == 'success':
        print("\n" + "="*60)
        print("📊 SIMPLE BACKTEST RESULTS")
        print("="*60)
        
        # Basic metrics
        initial_capital = results['initial_capital']
        final_capital = results['final_capital']
        total_return = results['total_return']
        
        print(f"\n💰 CAPITAL PERFORMANCE:")
        print(f"   Initial Capital: ${initial_capital:,.2f}")
        print(f"   Final Capital: ${final_capital:,.2f}")
        print(f"   Total Return: {total_return:.2%}")
        print(f"   Profit/Loss: ${final_capital - initial_capital:,.2f}")
        
        # Performance metrics
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            print(f"\n📈 PERFORMANCE METRICS:")
            print(f"   Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            print(f"   Volatility: {metrics.get('volatility', 0):.2%}")
            print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"   Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"   Total Trades: {metrics.get('total_trades', 0)}")
        
        print(f"\n⏱️  PERFORMANCE:")
        print(f"   Duration: {duration:.2f} seconds")
        
        print(f"\n🎉 Simple backtest completed successfully!")
        
        # Save results
        results_file = Path("simple_backtest_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📁 Results saved to {results_file}")
        
    else:
        print(f"\n❌ Backtest failed: {results.get('error', 'Unknown error')}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🚀 Next Steps:")
        print("1. Review the backtest results above")
        print("2. Check the saved results file")
        print("3. Run with different strategies")
        print("4. Integrate with real market data")
    else:
        print("\n❌ Backtest failed. Please check the error messages above.")
        sys.exit(1)
