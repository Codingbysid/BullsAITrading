from src.utils.common_imports import *
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path
import logging
from abc import ABC, abstractmethod

#!/usr/bin/env python3
"""
Base Backtester for QuantAI Trading Platform.

This module provides a unified base class for all backtesting implementations,
eliminating code duplication and ensuring consistency across all backtesters.

Features:
- Standardized backtesting interface
- Common data generation and management
- Unified performance metrics calculation
- Consistent trade execution logic
- Integrated with four-model decision engine
"""


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = setup_logger()


class BaseBacktester(ABC):
    """Base class for all backtesting implementations."""
    
    def __init__(self, initial_capital: float = 100000, name: str = "BaseBacktester"):
        self.name = name
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        self.performance_metrics = {}
        
        # Risk management parameters
        self.max_position_size = 0.30  # 30% max per ticker
        self.max_portfolio_drawdown = 0.15  # 15% max portfolio drawdown
        self.transaction_cost = 0.001  # 0.1% transaction cost
        
        logger.info(f"Initialized {self.name} with ${initial_capital:,.2f} capital")
    
    def create_synthetic_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Create realistic synthetic market data for backtesting."""
        logger.info(f"Creating synthetic data for {len(symbols)} symbols from {start_date} to {end_date}")
        
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
            
            for i, date in enumerate(trading_days):
                # Generate price with trend and volatility
                daily_return = np.random.normal(trend, volatility)
                new_price = prices[-1] * (1 + daily_return)
                prices.append(new_price)
                
                # Generate volume (higher volume on price changes)
                base_volume = 1000000 + hash(symbol) % 500000
                volume_multiplier = 1 + abs(daily_return) * 5  # Higher volume on big moves
                volume = int(base_volume * volume_multiplier * (0.8 + 0.4 * np.random.random()))
                volumes.append(volume)
            
            # Remove the extra price (we added one too many)
            prices = prices[1:]
            
            # Create OHLCV data
            for i, (date, price, volume) in enumerate(zip(trading_days, prices, volumes)):
                # Generate realistic OHLC from close price
                daily_range = price * volatility * np.random.uniform(0.5, 2.0)
                
                open_price = price * (1 + np.random.normal(0, volatility * 0.3))
                high_price = max(open_price, price) + daily_range * np.random.uniform(0, 0.5)
                low_price = min(open_price, price) - daily_range * np.random.uniform(0, 0.5)
                close_price = price
                
                all_data.append({
                    'Date': date,
                    'Symbol': symbol,
                    'Open': round(open_price, 2),
                    'High': round(high_price, 2),
                    'Low': round(low_price, 2),
                    'Close': round(close_price, 2),
                    'Volume': volume
                })
        
        df = pd.DataFrame(all_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Date', 'Symbol']).reset_index(drop=True)
        
        logger.info(f"Generated {len(df)} data points for {len(symbols)} symbols")
        return df
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the data."""
        logger.info("Calculating technical indicators...")
        
        data = data.copy()
        
        # Calculate indicators for each symbol
        for symbol in data['Symbol'].unique():
            symbol_mask = data['Symbol'] == symbol
            symbol_data = data[symbol_mask].copy()
            
            # Sort by date to ensure proper calculation
            symbol_data = symbol_data.sort_values('Date')
            
            # Moving averages
            symbol_data['SMA_5'] = symbol_data['Close'].rolling(window=5, min_periods=1).mean()
            symbol_data['SMA_20'] = symbol_data['Close'].rolling(window=20, min_periods=1).mean()
            symbol_data['SMA_50'] = symbol_data['Close'].rolling(window=50, min_periods=1).mean()
            
            # RSI
            delta = symbol_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            symbol_data['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = symbol_data['Close'].ewm(span=12).mean()
            ema_26 = symbol_data['Close'].ewm(span=26).mean()
            symbol_data['MACD'] = ema_12 - ema_26
            symbol_data['MACD_Signal'] = symbol_data['MACD'].ewm(span=9).mean()
            symbol_data['MACD_Histogram'] = symbol_data['MACD'] - symbol_data['MACD_Signal']
            
            # Bollinger Bands
            symbol_data['BB_Middle'] = symbol_data['Close'].rolling(window=20, min_periods=1).mean()
            bb_std = symbol_data['Close'].rolling(window=20, min_periods=1).std()
            symbol_data['BB_Upper'] = symbol_data['BB_Middle'] + (bb_std * 2)
            symbol_data['BB_Lower'] = symbol_data['BB_Middle'] - (bb_std * 2)
            symbol_data['BB_Position'] = (symbol_data['Close'] - symbol_data['BB_Lower']) / (symbol_data['BB_Upper'] - symbol_data['BB_Lower'])
            
            # Volume indicators
            symbol_data['Volume_SMA'] = symbol_data['Volume'].rolling(window=20, min_periods=1).mean()
            symbol_data['Volume_Ratio'] = symbol_data['Volume'] / symbol_data['Volume_SMA']
            
            # Price momentum
            symbol_data['Price_Change_1D'] = symbol_data['Close'].pct_change(1)
            symbol_data['Price_Change_5D'] = symbol_data['Close'].pct_change(5)
            symbol_data['Price_Change_20D'] = symbol_data['Close'].pct_change(20)
            
            # Volatility
            symbol_data['Volatility_20D'] = symbol_data['Close'].pct_change().rolling(window=20, min_periods=1).std()
            
            # Update the main dataframe
            data.loc[symbol_mask, symbol_data.columns] = symbol_data
        
        # Fill any remaining NaN values
        data = data.fillna(method='ffill').fillna(0)
        
        logger.info(f"Calculated technical indicators for {len(data['Symbol'].unique())} symbols")
        return data
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, symbol: str, date: datetime) -> Dict[str, Any]:
        """Generate trading signals for a specific symbol and date.
        
        Must be implemented by subclasses.
        
        Returns:
            Dict with keys: 'signal' ('BUY', 'SELL', 'HOLD'), 'strength' (0-1), 'reasoning'
        """
        pass
    
    def execute_trade(self, symbol: str, signal: Dict[str, Any], market_data: pd.Series, date: datetime):
        """Execute a trade based on the signal."""
        try:
            action = signal.get('signal', 'HOLD')
            strength = signal.get('strength', 0.0)
            reasoning = signal.get('reasoning', 'No reasoning provided')
            
            current_price = market_data['Close']
            current_position = self.positions.get(symbol, 0)
            
            if action == 'BUY' and strength > 0.1:
                # Calculate position size based on signal strength and risk management
                max_position_value = self.initial_capital * self.max_position_size
                position_value = max_position_value * strength
                shares_to_buy = int(position_value / current_price)
                
                # Apply transaction costs
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                
                if cost <= self.cash and shares_to_buy > 0:
                    self.cash -= cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + shares_to_buy
                    
                    trade = {
                        'date': date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': current_price,
                        'cost': cost,
                        'reasoning': reasoning,
                        'signal_strength': strength
                    }
                    self.trades.append(trade)
                    
                    logger.debug(f"BUY {shares_to_buy} shares of {symbol} at ${current_price:.2f}")
            
            elif action == 'SELL' and strength > 0.1 and current_position > 0:
                # Sell based on signal strength
                shares_to_sell = int(current_position * strength)
                
                if shares_to_sell > 0:
                    # Apply transaction costs
                    proceeds = shares_to_sell * current_price * (1 - self.transaction_cost)
                    
                    self.cash += proceeds
                    self.positions[symbol] -= shares_to_sell
                    
                    trade = {
                        'date': date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': current_price,
                        'proceeds': proceeds,
                        'reasoning': reasoning,
                        'signal_strength': strength
                    }
                    self.trades.append(trade)
                    
                    logger.debug(f"SELL {shares_to_sell} shares of {symbol} at ${current_price:.2f}")
        
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
    
    def calculate_portfolio_value(self, data: pd.DataFrame, date: datetime) -> float:
        """Calculate total portfolio value at a given date."""
        try:
            portfolio_value = self.cash
            
            # Get current prices for all positions
            current_data = data[data['Date'] == date]
            
            for symbol, shares in self.positions.items():
                if shares > 0:
                    symbol_data = current_data[current_data['Symbol'] == symbol]
                    if not symbol_data.empty:
                        current_price = symbol_data['Close'].iloc[0]
                        portfolio_value += shares * current_price
            
            return portfolio_value
        
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return self.cash
    
    def run_backtest(self, symbols: List[str], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run the backtest with the specified parameters."""
        try:
            logger.info(f"Starting {self.name} backtest for {len(symbols)} symbols")
            
            # Generate or load data
            data = self.create_synthetic_data(symbols, start_date, end_date)
            data = self.calculate_technical_indicators(data)
            
            # Get unique dates for iteration
            dates = sorted(data['Date'].unique())
            
            # Run backtest day by day
            for i, date in enumerate(dates):
                if i < 20:  # Skip first 20 days to allow indicators to stabilize
                    continue
                
                daily_data = data[data['Date'] == date]
                
                # Calculate portfolio value
                portfolio_value = self.calculate_portfolio_value(data, date)
                
                # Store portfolio history
                self.portfolio_history.append({
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'cash': self.cash,
                    'positions': self.positions.copy()
                })
                
                # Generate signals and execute trades for each symbol
                for symbol in symbols:
                    symbol_data = daily_data[daily_data['Symbol'] == symbol]
                    if not symbol_data.empty:
                        market_data = symbol_data.iloc[0]
                        
                        # Generate signal using the specific strategy
                        signal = self.generate_signals(data, symbol, date)
                        
                        # Execute trade if signal is generated
                        if signal:
                            self.execute_trade(symbol, signal, market_data, date)
            
            # Calculate final performance metrics
            self.performance_metrics = self.calculate_performance_metrics()
            
            # Prepare results
            results = {
                'backtest_name': self.name,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'symbols': symbols,
                'initial_capital': self.initial_capital,
                'final_capital': self.portfolio_history[-1]['portfolio_value'] if self.portfolio_history else self.initial_capital,
                'total_trades': len(self.trades),
                'performance_metrics': self.performance_metrics,
                'trades': self.trades,
                'portfolio_history': self.portfolio_history
            }
            
            logger.info(f"Backtest completed: {len(self.trades)} trades executed")
            return results
        
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {
                'backtest_name': self.name,
                'error': str(e),
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_trades': 0,
                'performance_metrics': {}
            }
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        try:
            if not self.portfolio_history:
                return {}
            
            # Extract portfolio values
            portfolio_values = [entry['portfolio_value'] for entry in self.portfolio_history]
            dates = [entry['date'] for entry in self.portfolio_history]
            
            if len(portfolio_values) < 2:
                return {}
            
            # Calculate returns
            returns = pd.Series(portfolio_values).pct_change().dropna()
            
            # Basic metrics
            total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
            annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
            
            # Risk metrics
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (np.array(portfolio_values) - peak) / peak
            max_drawdown = np.min(drawdown)
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
            
            # Trade statistics
            winning_trades = [t for t in self.trades if self._calculate_trade_pnl(t) > 0]
            losing_trades = [t for t in self.trades if self._calculate_trade_pnl(t) < 0]
            
            win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
            
            # Profit factor
            gross_profit = sum(self._calculate_trade_pnl(t) for t in winning_trades)
            gross_loss = abs(sum(self._calculate_trade_pnl(t) for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'final_portfolio_value': portfolio_values[-1],
                'cash_ratio': self.cash / portfolio_values[-1] if portfolio_values[-1] > 0 else 1.0
            }
        
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_trade_pnl(self, trade: Dict[str, Any]) -> float:
        """Calculate P&L for a single trade."""
        try:
            if trade['action'] == 'BUY':
                return -trade['cost']  # Cost is negative for P&L
            elif trade['action'] == 'SELL':
                return trade['proceeds']  # Proceeds are positive for P&L
            return 0.0
        except Exception:
            return 0.0
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted backtest results."""
        print(f"\n🚀 {self.name} Results")
        print("=" * 70)
        
        if 'error' in results:
            print(f"❌ Backtest failed: {results['error']}")
            return
        
        # Capital performance
        initial = results['initial_capital']
        final = results['final_capital']
        total_return = (final - initial) / initial
        
        print(f"💰 CAPITAL PERFORMANCE:")
        print(f"Initial Capital: ${initial:,.2f}")
        print(f"Final Capital: ${final:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Profit/Loss: ${final - initial:,.2f}")
        
        # Performance metrics
        metrics = results.get('performance_metrics', {})
        if metrics:
            print(f"\n📈 PERFORMANCE METRICS:")
            print(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            print(f"Volatility: {metrics.get('volatility', 0):.2%}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
            print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
            print(f"Cash Ratio: {metrics.get('cash_ratio', 0):.2%}")
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save backtest results to JSON file."""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.name.lower().replace(' ', '_')}_results_{timestamp}.json"
            
            filepath = Path(filename)
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {filepath}")
            print(f"📁 Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")


# Example implementation for testing
class SimpleStrategy(BaseBacktester):
    """Simple strategy implementation for testing."""
    
    def __init__(self):
        super().__init__(name="Simple Strategy")
    
    def generate_signals(self, data: pd.DataFrame, symbol: str, date: datetime) -> Dict[str, Any]:
        """Generate simple RSI-based signals."""
        try:
            # Get recent data for the symbol
            symbol_data = data[(data['Symbol'] == symbol) & (data['Date'] <= date)].tail(1)
            
            if symbol_data.empty:
                return {'signal': 'HOLD', 'strength': 0.0, 'reasoning': 'No data available'}
            
            latest = symbol_data.iloc[0]
            
            # Simple RSI strategy
            rsi = latest.get('RSI', 50)
            
            if rsi < 30:  # Oversold
                return {
                    'signal': 'BUY',
                    'strength': min(0.5, (30 - rsi) / 30),
                    'reasoning': f'RSI oversold at {rsi:.1f}'
                }
            elif rsi > 70:  # Overbought
                return {
                    'signal': 'SELL',
                    'strength': min(0.5, (rsi - 70) / 30),
                    'reasoning': f'RSI overbought at {rsi:.1f}'
                }
            else:
                return {
                    'signal': 'HOLD',
                    'strength': 0.0,
                    'reasoning': f'RSI neutral at {rsi:.1f}'
                }
        
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return {'signal': 'HOLD', 'strength': 0.0, 'reasoning': f'Error: {e}'}


# Test the base backtester
if __name__ == "__main__":
    # Test with simple strategy
    backtester = SimpleStrategy()
    
    symbols = ['AAPL', 'AMZN', 'GOOGL', 'META', 'NVDA']
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    results = backtester.run_backtest(symbols, start_date, end_date)
    backtester.print_results(results)
    backtester.save_results(results)
