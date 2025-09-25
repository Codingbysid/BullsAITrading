#!/usr/bin/env python3
"""
Standalone backtester that works around all dependency issues.

This script provides comprehensive backtesting functionality without any external dependencies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StandaloneBacktester:
    """Standalone backtester with no external dependencies."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.performance_metrics = {}
        
    def create_synthetic_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Create realistic synthetic market data."""
        logger.info("📊 Creating synthetic market data...")
        
        # Generate trading days (exclude weekends)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_days = [d for d in dates if d.weekday() < 5]  # Monday=0, Friday=4
        
        all_data = []
        
        for symbol in symbols:
            # Generate realistic price data for each symbol
            np.random.seed(42 + hash(symbol) % 1000)
            
            # Symbol-specific parameters
            base_price = 100 + hash(symbol) % 200
            volatility = 0.02 + (hash(symbol) % 10) * 0.005  # 2-7% daily volatility
            trend = (hash(symbol) % 20 - 10) * 0.0001  # Slight trend
            
            prices = [base_price]
            volumes = []
            
            for i in range(1, len(trading_days)):
                # Generate realistic price movement
                daily_return = np.random.normal(trend, volatility)
                new_price = prices[-1] * (1 + daily_return)
                prices.append(max(new_price, 1.0))  # Prevent negative prices
                
                # Generate volume (higher on volatile days)
                base_volume = 1000000 + hash(symbol) % 5000000
                volume_multiplier = 1 + abs(daily_return) * 10
                volumes.append(int(base_volume * volume_multiplier))
            
            # Create OHLCV data
            for i, (date, close) in enumerate(zip(trading_days, prices)):
                # Generate OHLC from close price
                daily_volatility = volatility * np.random.uniform(0.5, 1.5)
                
                high = close * (1 + daily_volatility * np.random.uniform(0.3, 1.0))
                low = close * (1 - daily_volatility * np.random.uniform(0.3, 1.0))
                open_price = close * (1 + np.random.uniform(-0.5, 0.5) * daily_volatility)
                
                # Ensure OHLC consistency
                high = max(high, open_price, close)
                low = min(low, open_price, close)
                
                volume = volumes[i] if i < len(volumes) else 1000000
                
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
        
        logger.info(f"✅ Created {len(df)} records for {len(symbols)} symbols")
        return df
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators without external dependencies."""
        data = data.copy()
        
        # Simple Moving Averages
        data['sma_5'] = data['close'].rolling(5).mean()
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        
        # RSI calculation
        data['rsi'] = self._calculate_rsi(data['close'])
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(20).mean()
        data['bb_std'] = data['close'].rolling(20).std()
        data['bb_upper'] = data['bb_middle'] + (2 * data['bb_std'])
        data['bb_lower'] = data['bb_middle'] - (2 * data['bb_std'])
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # MACD
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Volume indicators
        data['volume_sma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Price momentum
        data['momentum_5'] = data['close'].pct_change(5)
        data['momentum_10'] = data['close'].pct_change(10)
        
        # Volatility
        data['volatility'] = data['close'].rolling(20).std()
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI without external dependencies."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    def create_advanced_strategy(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Create an advanced trading strategy using multiple indicators."""
        symbol_data = data[data['symbol'] == symbol].copy()
        
        if len(symbol_data) < 50:
            return {'signal': 'HOLD', 'strength': 0.0, 'confidence': 0.0}
        
        # Calculate indicators
        symbol_data = self.calculate_technical_indicators(symbol_data)
        
        # Get latest values
        latest = symbol_data.iloc[-1]
        prev = symbol_data.iloc[-2]
        
        # Strategy signals
        buy_signals = []
        sell_signals = []
        
        # 1. RSI signals
        if latest['rsi'] < 30:
            buy_signals.append(('rsi_oversold', 0.3))
        elif latest['rsi'] > 70:
            sell_signals.append(('rsi_overbought', 0.3))
        
        # 2. Moving average signals
        if latest['close'] > latest['sma_5'] > latest['sma_20']:
            buy_signals.append(('ma_bullish', 0.2))
        elif latest['close'] < latest['sma_5'] < latest['sma_20']:
            sell_signals.append(('ma_bearish', 0.2))
        
        # 3. Bollinger Band signals
        if latest['bb_position'] < 0.2:
            buy_signals.append(('bb_oversold', 0.25))
        elif latest['bb_position'] > 0.8:
            sell_signals.append(('bb_overbought', 0.25))
        
        # 4. MACD signals
        if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            buy_signals.append(('macd_bullish', 0.2))
        elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            sell_signals.append(('macd_bearish', 0.2))
        
        # 5. Volume confirmation
        if latest['volume_ratio'] > 1.5:
            if buy_signals:
                buy_signals.append(('volume_confirmation', 0.1))
            if sell_signals:
                sell_signals.append(('volume_confirmation', 0.1))
        
        # 6. Momentum signals
        if latest['momentum_5'] > 0.02:
            buy_signals.append(('momentum_positive', 0.15))
        elif latest['momentum_5'] < -0.02:
            sell_signals.append(('momentum_negative', 0.15))
        
        # Calculate signal strength
        buy_strength = sum(weight for _, weight in buy_signals)
        sell_strength = sum(weight for _, weight in sell_signals)
        
        # Generate final signal
        if buy_strength > sell_strength and buy_strength > 0.3:
            signal = 'BUY'
            strength = min(0.2, buy_strength)
            confidence = min(0.9, buy_strength)
        elif sell_strength > buy_strength and sell_strength > 0.3:
            signal = 'SELL'
            strength = min(0.2, sell_strength)
            confidence = min(0.9, sell_strength)
        else:
            signal = 'HOLD'
            strength = 0.0
            confidence = 0.5
        
        return {
            'signal': signal,
            'strength': strength,
            'confidence': confidence,
            'price': latest['close'],
            'indicators': {
                'rsi': latest['rsi'],
                'bb_position': latest['bb_position'],
                'macd': latest['macd'],
                'volume_ratio': latest['volume_ratio'],
                'momentum_5': latest['momentum_5']
            }
        }
    
    def run_backtest(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run the comprehensive backtest."""
        logger.info("🚀 Starting Standalone Backtest")
        logger.info("=" * 60)
        logger.info(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"📊 Symbols: {symbols}")
        logger.info("")
        
        try:
            # Create synthetic data
            data = self.create_synthetic_data(symbols, start_date, end_date)
            
            # Group by date
            daily_data = data.groupby(data.index.date)
            
            logger.info("📈 Running backtest loop...")
            
            for date, day_data in daily_data:
                current_date = pd.Timestamp(date)
                
                # Process each symbol
                for symbol in symbols:
                    symbol_day_data = day_data[day_data['symbol'] == symbol]
                    
                    if symbol_day_data.empty:
                        continue
                    
                    # Generate signal
                    signal = self.create_advanced_strategy(data, symbol)
                    
                    if signal['signal'] != 'HOLD':
                        self._execute_trade(symbol, signal, symbol_day_data.iloc[-1], current_date)
                
                # Update portfolio history
                self._update_portfolio_history(current_date)
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
            
            logger.info("✅ Backtest completed successfully!")
            
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
            logger.error(f"❌ Backtest failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _execute_trade(self, symbol: str, signal: Dict[str, Any], market_data: pd.Series, date: datetime):
        """Execute a trade with realistic transaction costs."""
        try:
            action = signal['signal']
            price = market_data['close']
            strength = signal['strength']
            confidence = signal['confidence']
            
            # Calculate position size based on confidence and strength
            max_position_value = self._get_total_value() * 0.2 * strength * confidence
            quantity = int(max_position_value / price)
            
            if quantity == 0:
                return
            
            # Apply transaction costs
            commission = quantity * price * 0.001  # 0.1% commission
            slippage = quantity * price * 0.0005 * (1 + np.random.uniform(0, 0.5))  # Variable slippage
            
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
                            'current_price': price,
                            'unrealized_pnl': 0.0
                        }
                    else:
                        # Create new position
                        self.positions[symbol] = {
                            'quantity': quantity,
                            'avg_price': price,
                            'current_price': price,
                            'unrealized_pnl': 0.0
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
                        'slippage': slippage,
                        'confidence': confidence
                    })
            
            elif action == 'SELL':
                if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
                    # Calculate realized P&L
                    realized_pnl = (price - self.positions[symbol]['avg_price']) * quantity
                    
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
                        'slippage': slippage,
                        'realized_pnl': realized_pnl,
                        'confidence': confidence
                    })
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
    
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
        """Update portfolio history with realistic price movements."""
        total_value = self._get_total_value()
        
        # Update position prices with realistic movements
        for symbol, position in self.positions.items():
            # Generate realistic price movement
            price_change = np.random.normal(0, 0.02)  # 2% daily volatility
            position['current_price'] *= (1 + price_change)
            position['unrealized_pnl'] = (position['current_price'] - position['avg_price']) * position['quantity']
        
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
            'positions': {symbol: {
                'quantity': pos['quantity'],
                'avg_price': pos['avg_price'],
                'current_price': pos['current_price'],
                'unrealized_pnl': pos['unrealized_pnl']
            } for symbol, pos in self.positions.items()}
        }
        
        self.portfolio_history.append(portfolio_snapshot)
    
    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics."""
        logger.info("📊 Calculating performance metrics...")
        
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
        
        # Win rate and profit factor
        winning_trades = [t for t in self.trades if t['action'] == 'SELL' and t.get('realized_pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t['action'] == 'SELL' and t.get('realized_pnl', 0) < 0]
        
        total_trades = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        gross_profit = sum(t.get('realized_pnl', 0) for t in winning_trades)
        gross_loss = sum(abs(t.get('realized_pnl', 0)) for t in losing_trades)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Additional metrics
        avg_trade_return = (gross_profit + sum(t.get('realized_pnl', 0) for t in losing_trades)) / total_trades if total_trades > 0 else 0
        
        self.performance_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades,
            'avg_trade_return': avg_trade_return,
            'final_value': self._get_total_value()
        }
        
        logger.info("✅ Performance metrics calculated")


def main():
    """Main function."""
    print("🚀 QuantAI Trading Platform - Standalone Backtester")
    print("=" * 70)
    print("Advanced backtesting with no external dependencies")
    print()
    
    # Configuration
    symbols = ['AAPL', 'GOOGL', 'NVDA', 'META', 'AMZN', 'TSLA', 'MSFT', 'NFLX']
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    initial_capital = 100000
    
    # Initialize backtester
    backtester = StandaloneBacktester(initial_capital=initial_capital)
    
    # Run backtest
    start_time = datetime.now()
    results = backtester.run_backtest(symbols, start_date, end_date)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    
    if results['status'] == 'success':
        print("\n" + "="*70)
        print("📊 STANDALONE BACKTEST RESULTS")
        print("="*70)
        
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
            print(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"   Total Trades: {metrics.get('total_trades', 0)}")
            print(f"   Avg Trade Return: {metrics.get('avg_trade_return', 0):.2%}")
        
        print(f"\n⏱️  PERFORMANCE:")
        print(f"   Duration: {duration:.2f} seconds")
        
        print(f"\n🎉 Standalone backtest completed successfully!")
        
        # Save results
        results_file = Path("standalone_backtest_results.json")
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
        print("3. Analyze trade performance")
        print("4. Optimize strategy parameters")
        print("5. Integrate with real market data")
    else:
        print("\n❌ Backtest failed. Please check the error messages above.")
        import sys
        sys.exit(1)
