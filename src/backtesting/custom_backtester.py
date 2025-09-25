"""
Custom backtester for QuantAI Trading Platform.

This module implements a comprehensive backtesting system with:
- Portfolio simulation
- Risk management
- Performance analytics
- Transaction cost modeling
- Multiple strategy support
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from pathlib import Path

from ..config.settings import get_settings
from ..trading.decision_engine import DecisionEngine
from ..risk.risk_management import RiskManager
from ..data.data_sources import data_manager
from ..data.feature_engineering import FeatureEngineer
from ..data.sentiment_analysis import sentiment_aggregator

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trade record."""
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    price: float
    value: float
    commission: float
    slippage: float


@dataclass
class Position:
    """Position record."""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class Portfolio:
    """Portfolio state."""
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, Position]
    trades: List[Trade]
    returns: float
    cumulative_return: float


class BacktestStrategy(ABC):
    """Abstract base class for backtesting strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.decision_engine = DecisionEngine()
        self.risk_manager = RiskManager()
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> Dict[str, Any]:
        """Generate trading signals for given data."""
        pass
    
    def get_position_size(self, signal: Dict[str, Any], portfolio: Portfolio) -> int:
        """Calculate position size based on signal and portfolio."""
        if signal['signal'] == 'HOLD':
            return 0
        
        # Risk-adjusted position sizing
        max_position_value = portfolio.total_value * 0.2  # Max 20% per position
        signal_strength = signal.get('strength', 0.1)
        
        # Calculate position size
        position_value = max_position_value * signal_strength
        quantity = int(position_value / signal['price'])
        
        return quantity


class QuantAIStrategy(BacktestStrategy):
    """QuantAI Trading Strategy for backtesting."""
    
    def __init__(self):
        super().__init__("QuantAI Strategy")
        self.feature_engineer = FeatureEngineer()
        self.lookback_period = 20
    
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> Dict[str, Any]:
        """Generate signals using QuantAI decision engine."""
        signals = {}
        
        for symbol in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else [data.columns[0]]:
            try:
                # Get symbol data
                symbol_data = data[symbol] if isinstance(data.columns, pd.MultiIndex) else data
                
                # Create features
                features = self.feature_engineer.create_all_features(symbol_data, symbol)
                
                if features.empty:
                    continue
                
                # Get sentiment data
                sentiment_score = 0.0  # Placeholder - would integrate with sentiment analysis
                
                # Get portfolio state
                portfolio_state = {
                    'total_value': 100000,  # Placeholder
                    'cash': 100000,
                    'positions': {}
                }
                
                # Generate signal
                signal_data = self.decision_engine.generate_trading_signal(
                    symbol=symbol,
                    current_market_data=features,
                    sentiment_score=sentiment_score,
                    portfolio_state=portfolio_state
                )
                
                signals[symbol] = {
                    'signal': signal_data['signal'],
                    'strength': signal_data['position_size'],
                    'price': symbol_data['close'].iloc[-1] if 'close' in symbol_data.columns else 100.0,
                    'confidence': 0.8  # Placeholder
                }
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        return signals


class MomentumStrategy(BacktestStrategy):
    """Momentum trading strategy."""
    
    def __init__(self):
        super().__init__("Momentum Strategy")
        self.short_window = 5
        self.long_window = 20
    
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> Dict[str, Any]:
        """Generate momentum signals."""
        signals = {}
        
        for symbol in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else [data.columns[0]]:
            try:
                symbol_data = data[symbol] if isinstance(data.columns, pd.MultiIndex) else data
                
                if len(symbol_data) < self.long_window:
                    continue
                
                # Calculate moving averages
                short_ma = symbol_data['close'].rolling(self.short_window).mean().iloc[-1]
                long_ma = symbol_data['close'].rolling(self.long_window).mean().iloc[-1]
                current_price = symbol_data['close'].iloc[-1]
                
                # Generate signal
                if short_ma > long_ma and current_price > short_ma:
                    signal = 'BUY'
                    strength = min(0.2, (short_ma - long_ma) / long_ma)
                elif short_ma < long_ma and current_price < short_ma:
                    signal = 'SELL'
                    strength = min(0.2, (long_ma - short_ma) / long_ma)
                else:
                    signal = 'HOLD'
                    strength = 0.0
                
                signals[symbol] = {
                    'signal': signal,
                    'strength': strength,
                    'price': current_price,
                    'confidence': 0.7
                }
                
            except Exception as e:
                logger.error(f"Error generating momentum signal for {symbol}: {e}")
                continue
        
        return signals


class MeanReversionStrategy(BacktestStrategy):
    """Mean reversion trading strategy."""
    
    def __init__(self):
        super().__init__("Mean Reversion Strategy")
        self.lookback_period = 20
        self.entry_threshold = 2.0  # Standard deviations
        self.exit_threshold = 0.5
    
    def generate_signals(self, data: pd.DataFrame, timestamp: datetime) -> Dict[str, Any]:
        """Generate mean reversion signals."""
        signals = {}
        
        for symbol in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else [data.columns[0]]:
            try:
                symbol_data = data[symbol] if isinstance(data.columns, pd.MultiIndex) else data
                
                if len(symbol_data) < self.lookback_period:
                    continue
                
                # Calculate Bollinger Bands
                prices = symbol_data['close']
                mean_price = prices.rolling(self.lookback_period).mean().iloc[-1]
                std_price = prices.rolling(self.lookback_period).std().iloc[-1]
                current_price = prices.iloc[-1]
                
                # Calculate z-score
                z_score = (current_price - mean_price) / std_price
                
                # Generate signal
                if z_score < -self.entry_threshold:
                    signal = 'BUY'
                    strength = min(0.2, abs(z_score) / self.entry_threshold)
                elif z_score > self.entry_threshold:
                    signal = 'SELL'
                    strength = min(0.2, abs(z_score) / self.entry_threshold)
                else:
                    signal = 'HOLD'
                    strength = 0.0
                
                signals[symbol] = {
                    'signal': signal,
                    'strength': strength,
                    'price': current_price,
                    'confidence': 0.6
                }
                
            except Exception as e:
                logger.error(f"Error generating mean reversion signal for {symbol}: {e}")
                continue
        
        return signals


class CustomBacktester:
    """Custom backtester implementation."""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 max_position_size: float = 0.2):
        """
        Initialize custom backtester.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate (0.1% default)
            slippage: Slippage rate (0.05% default)
            max_position_size: Maximum position size as fraction of portfolio
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        
        # Portfolio state
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
        # Performance tracking
        self.performance_metrics = {}
        
        # Settings
        self.settings = get_settings()
    
    def run_backtest(self, 
                    strategy: BacktestStrategy,
                    start_date: datetime,
                    end_date: datetime,
                    symbols: List[str]) -> Dict[str, Any]:
        """
        Run backtest with given strategy.
        
        Args:
            strategy: Trading strategy to use
            start_date: Backtest start date
            end_date: Backtest end date
            symbols: List of symbols to backtest
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting custom backtest with {strategy.name}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        
        try:
            # Get market data
            market_data = self._get_market_data(symbols, start_date, end_date)
            
            if market_data.empty:
                raise ValueError("No market data available")
            
            # Run backtest
            self._run_backtest_loop(strategy, market_data, start_date, end_date)
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
            
            logger.info("Custom backtest completed successfully")
            
            return {
                'status': 'success',
                'strategy': strategy.name,
                'initial_capital': self.initial_capital,
                'final_capital': self._get_total_value(),
                'total_return': self._get_total_return(),
                'performance_metrics': self.performance_metrics,
                'trades': len(self.trades),
                'portfolio_history': self.portfolio_history
            }
            
        except Exception as e:
            logger.error(f"Custom backtest failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _get_market_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get market data for backtesting."""
        logger.info("Fetching market data...")
        
        all_data = []
        
        for symbol in symbols:
            try:
                # Get data from data manager
                data = data_manager.get_market_data(
                    symbol=symbol,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                if not data.empty:
                    # Add symbol column
                    data['symbol'] = symbol
                    all_data.append(data)
                    logger.info(f"✅ Added {len(data)} records for {symbol}")
                else:
                    logger.warning(f"⚠️  No data for {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Error fetching data for {symbol}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No market data available")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data.set_index('timestamp', inplace=True)
        
        logger.info(f"📊 Combined dataset: {len(combined_data)} records")
        
        return combined_data
    
    def _run_backtest_loop(self, strategy: BacktestStrategy, data: pd.DataFrame, start_date: datetime, end_date: datetime):
        """Run the main backtest loop."""
        logger.info("Running backtest loop...")
        
        # Group data by date
        daily_data = data.groupby(data.index.date)
        
        for date, day_data in daily_data:
            current_date = pd.Timestamp(date)
            
            if current_date < start_date or current_date > end_date:
                continue
            
            # Skip weekends
            if current_date.weekday() >= 5:
                continue
            
            # Process each day
            self._process_day(strategy, day_data, current_date)
            
            # Update portfolio history
            self._update_portfolio_history(current_date)
    
    def _process_day(self, strategy: BacktestStrategy, day_data: pd.DataFrame, date: datetime):
        """Process a single trading day."""
        try:
            # Generate signals
            signals = strategy.generate_signals(day_data, date)
            
            # Execute trades
            for symbol, signal in signals.items():
                if signal['signal'] != 'HOLD':
                    self._execute_trade(symbol, signal, date)
            
        except Exception as e:
            logger.error(f"Error processing day {date}: {e}")
    
    def _execute_trade(self, symbol: str, signal: Dict[str, Any], date: datetime):
        """Execute a trade."""
        try:
            action = signal['signal']
            price = signal['price']
            strength = signal['strength']
            
            # Calculate position size
            position_size = strategy.get_position_size(signal, self._get_portfolio_state())
            
            if position_size == 0:
                return
            
            # Apply slippage
            if action == 'BUY':
                execution_price = price * (1 + self.slippage)
            else:
                execution_price = price * (1 - self.slippage)
            
            # Calculate costs
            trade_value = position_size * execution_price
            commission = trade_value * self.commission
            total_cost = trade_value + commission
            
            # Check if we have enough cash for buy orders
            if action == 'BUY' and total_cost > self.cash:
                logger.warning(f"Insufficient cash for {symbol} buy order")
                return
            
            # Execute trade
            if action == 'BUY':
                self.cash -= total_cost
                if symbol in self.positions:
                    # Update existing position
                    old_quantity = self.positions[symbol].quantity
                    old_avg_price = self.positions[symbol].avg_price
                    new_quantity = old_quantity + position_size
                    new_avg_price = ((old_quantity * old_avg_price) + (position_size * execution_price)) / new_quantity
                    
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=new_quantity,
                        avg_price=new_avg_price,
                        current_price=execution_price,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0
                    )
                else:
                    # Create new position
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=position_size,
                        avg_price=execution_price,
                        current_price=execution_price,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0
                    )
            
            elif action == 'SELL':
                if symbol in self.positions and self.positions[symbol].quantity >= position_size:
                    # Update position
                    self.positions[symbol].quantity -= position_size
                    self.cash += trade_value - commission
                    
                    # Calculate realized P&L
                    realized_pnl = (execution_price - self.positions[symbol].avg_price) * position_size
                    self.positions[symbol].realized_pnl += realized_pnl
                    
                    # Remove position if quantity is zero
                    if self.positions[symbol].quantity == 0:
                        del self.positions[symbol]
            
            # Record trade
            trade = Trade(
                timestamp=date,
                symbol=symbol,
                action=action,
                quantity=position_size,
                price=execution_price,
                value=trade_value,
                commission=commission,
                slippage=self.slippage
            )
            self.trades.append(trade)
            
            logger.debug(f"Executed {action} {position_size} {symbol} at {execution_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
    
    def _get_portfolio_state(self) -> Portfolio:
        """Get current portfolio state."""
        total_value = self._get_total_value()
        
        return Portfolio(
            timestamp=datetime.now(),
            total_value=total_value,
            cash=self.cash,
            positions=self.positions.copy(),
            trades=self.trades.copy(),
            returns=0.0,
            cumulative_return=(total_value / self.initial_capital) - 1
        )
    
    def _get_total_value(self) -> float:
        """Calculate total portfolio value."""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            total_value += position.quantity * position.current_price
        
        return total_value
    
    def _get_total_return(self) -> float:
        """Calculate total return."""
        return (self._get_total_value() / self.initial_capital) - 1
    
    def _update_portfolio_history(self, date: datetime):
        """Update portfolio history."""
        total_value = self._get_total_value()
        
        # Update position prices (simplified - would use actual market data)
        for symbol, position in self.positions.items():
            position.current_price *= (1 + np.random.normal(0, 0.01))  # Random price movement
        
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
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl
            } for symbol, pos in self.positions.items()}
        }
        
        self.portfolio_history.append(portfolio_snapshot)
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics."""
        logger.info("Calculating performance metrics...")
        
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
        winning_trades = [t for t in self.trades if t.action == 'SELL' and t.value > 0]
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.value for t in winning_trades)
        gross_loss = sum(abs(t.value) for t in self.trades if t.value < 0)
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
        
        logger.info(f"Performance metrics calculated: {self.performance_metrics}")


# Global backtester instance
custom_backtester = CustomBacktester()


async def run_custom_backtest(start_date: datetime = None, 
                             end_date: datetime = None,
                             initial_capital: float = 100000,
                             strategy_name: str = "quantai") -> Dict[str, Any]:
    """
    Run custom backtest.
    
    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        strategy_name: Strategy to use
        
    Returns:
        Dictionary with backtest results
    """
    try:
        # Set default dates
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        # Initialize backtester
        backtester = CustomBacktester(initial_capital=initial_capital)
        
        # Create strategy
        if strategy_name == "quantai":
            strategy = QuantAIStrategy()
        elif strategy_name == "momentum":
            strategy = MomentumStrategy()
        elif strategy_name == "mean_reversion":
            strategy = MeanReversionStrategy()
        else:
            strategy = QuantAIStrategy()
        
        # Get symbols
        settings = get_settings()
        symbols = settings.target_symbols
        
        # Run backtest
        results = backtester.run_backtest(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Custom backtest failed: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }
