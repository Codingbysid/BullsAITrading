"""
Event-driven backtester using QF-Lib.

This module implements a comprehensive backtesting system with:
- Event-driven architecture
- Portfolio management
- Risk management
- Performance analytics
- Transaction cost modeling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import queue
import threading
from enum import Enum

# QF-Lib imports (install with: pip install qf-lib)
try:
    from qf_lib.containers.series.qf_series import QFSeries
    from qf_lib.containers.dataframe.qf_dataframe import QFDataFrame
    from qf_lib.common.enums.frequency import Frequency
    from qf_lib.common.enums.price_field import PriceField
    from qf_lib.data_providers.data_provider import DataProvider
    from qf_lib.backtesting.events.event import Event
    from qf_lib.backtesting.events.signal_event import SignalEvent
    from qf_lib.backtesting.events.order_event import OrderEvent
    from qf_lib.backtesting.events.fill_event import FillEvent
    from qf_lib.backtesting.portfolio.portfolio import Portfolio
    from qf_lib.backtesting.execution.execution_model import ExecutionModel
    from qf_lib.backtesting.execution.simulated.simulated_execution_model import SimulatedExecutionModel
    from qf_lib.backtesting.portfolio.portfolio_factory import PortfolioFactory
    from qf_lib.backtesting.strategy import Strategy
    from qf_lib.backtesting.broker.broker import Broker
    from qf_lib.backtesting.broker.simulated_broker import SimulatedBroker
    from qf_lib.backtesting.data_handler.data_handler import DataHandler
    from qf_lib.backtesting.data_handler.daily_data_handler import DailyDataHandler
    from qf_lib.backtesting.data_handler.data_provider_daily_data_handler import DataProviderDailyDataHandler
    from qf_lib.backtesting.portfolio.portfolio import Portfolio
    from qf_lib.backtesting.portfolio.portfolio_factory import PortfolioFactory
    from qf_lib.backtesting.portfolio.portfolio import Portfolio
    from qf_lib.backtesting.portfolio.portfolio import Portfolio
    QF_LIB_AVAILABLE = True
except ImportError:
    QF_LIB_AVAILABLE = False
    logging.warning("QF-Lib not available. Install with: pip install qf-lib")

from ..config.settings import get_settings
from ..trading.decision_engine import DecisionEngine
from ..risk.risk_management import RiskManager

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event types for the backtester."""
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


@dataclass
class BacktestEvent:
    """Base event class for the backtester."""
    timestamp: datetime
    event_type: EventType
    data: Dict[str, Any]


@dataclass
class MarketEvent(BacktestEvent):
    """Market data event."""
    def __post_init__(self):
        self.event_type = EventType.MARKET


@dataclass
class SignalEvent(BacktestEvent):
    """Trading signal event."""
    symbol: str
    signal: str  # BUY, SELL, HOLD
    strength: float
    timestamp: datetime
    
    def __post_init__(self):
        self.event_type = EventType.SIGNAL


@dataclass
class OrderEvent(BacktestEvent):
    """Order event."""
    symbol: str
    order_type: str  # MARKET, LIMIT
    quantity: int
    direction: str  # BUY, SELL
    
    def __post_init__(self):
        self.event_type = EventType.ORDER


@dataclass
class FillEvent(BacktestEvent):
    """Order fill event."""
    symbol: str
    quantity: int
    direction: str
    fill_cost: float
    commission: float
    
    def __post_init__(self):
        self.event_type = EventType.FILL


class EventQueue:
    """Thread-safe event queue for the backtester."""
    
    def __init__(self):
        self._queue = queue.Queue()
        self._lock = threading.Lock()
    
    def put(self, event: BacktestEvent):
        """Add event to queue."""
        with self._lock:
            self._queue.put(event)
    
    def get(self, timeout: float = None) -> Optional[BacktestEvent]:
        """Get event from queue."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()
    
    def size(self) -> int:
        """Get queue size."""
        return self._queue.qsize()


class QFBacktester:
    """Event-driven backtester using QF-Lib."""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 start_date: datetime = None,
                 end_date: datetime = None,
                 symbols: List[str] = None):
        """
        Initialize the QF-Lib backtester.
        
        Args:
            initial_capital: Starting capital
            start_date: Backtest start date
            end_date: Backtest end date
            symbols: List of symbols to backtest
        """
        if not QF_LIB_AVAILABLE:
            raise ImportError("QF-Lib not available. Install with: pip install qf-lib")
        
        self.settings = get_settings()
        self.initial_capital = initial_capital
        self.start_date = start_date or datetime.now() - timedelta(days=365)
        self.end_date = end_date or datetime.now()
        self.symbols = symbols or self.settings.target_symbols
        
        # Initialize components
        self.event_queue = EventQueue()
        self.portfolio = None
        self.broker = None
        self.data_handler = None
        self.execution_model = None
        self.strategy = None
        
        # Performance tracking
        self.performance_metrics = {}
        self.trade_log = []
        self.portfolio_history = []
        
        # Initialize QF-Lib components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize QF-Lib components."""
        logger.info("Initializing QF-Lib components...")
        
        # Initialize data handler
        self.data_handler = self._create_data_handler()
        
        # Initialize broker
        self.broker = SimulatedBroker(
            initial_cash=self.initial_capital,
            commission=0.001,  # 0.1% commission
            slippage=0.0005   # 0.05% slippage
        )
        
        # Initialize execution model
        self.execution_model = SimulatedExecutionModel(
            broker=self.broker,
            data_handler=self.data_handler
        )
        
        # Initialize portfolio
        self.portfolio = PortfolioFactory.create_portfolio(
            initial_cash=self.initial_capital,
            data_handler=self.data_handler,
            execution_model=self.execution_model
        )
        
        logger.info("QF-Lib components initialized successfully")
    
    def _create_data_handler(self):
        """Create data handler for backtesting."""
        # This would integrate with your existing data sources
        # For now, we'll create a mock data handler
        logger.info("Creating data handler...")
        return None  # Placeholder - would integrate with your data sources
    
    def run_backtest(self, strategy: 'TradingStrategy') -> Dict[str, Any]:
        """
        Run the backtest with the given strategy.
        
        Args:
            strategy: Trading strategy to backtest
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        logger.info(f"Symbols: {self.symbols}")
        
        try:
            # Initialize strategy
            self.strategy = strategy
            self.strategy.initialize(self.data_handler, self.portfolio)
            
            # Run event loop
            self._run_event_loop()
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
            
            logger.info("Backtest completed successfully")
            
            return {
                'status': 'success',
                'initial_capital': self.initial_capital,
                'final_capital': self.portfolio.get_total_value(),
                'total_return': self.portfolio.get_total_return(),
                'performance_metrics': self.performance_metrics,
                'trade_log': self.trade_log,
                'portfolio_history': self.portfolio_history
            }
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _run_event_loop(self):
        """Run the main event loop."""
        logger.info("Running event loop...")
        
        current_date = self.start_date
        
        while current_date <= self.end_date:
            # Generate market event
            market_data = self._get_market_data(current_date)
            if market_data is not None:
                market_event = MarketEvent(
                    timestamp=current_date,
                    event_type=EventType.MARKET,
                    data=market_data
                )
                self.event_queue.put(market_event)
                
                # Process events
                self._process_events()
            
            # Move to next trading day
            current_date += timedelta(days=1)
            # Skip weekends
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
    
    def _get_market_data(self, date: datetime) -> Optional[Dict[str, Any]]:
        """Get market data for a specific date."""
        # This would integrate with your data sources
        # For now, return mock data
        return {
            'date': date,
            'symbols': {symbol: {'close': 100.0, 'volume': 1000000} for symbol in self.symbols}
        }
    
    def _process_events(self):
        """Process events from the queue."""
        while not self.event_queue.empty():
            event = self.event_queue.get()
            
            if event.event_type == EventType.MARKET:
                self._handle_market_event(event)
            elif event.event_type == EventType.SIGNAL:
                self._handle_signal_event(event)
            elif event.event_type == EventType.ORDER:
                self._handle_order_event(event)
            elif event.event_type == EventType.FILL:
                self._handle_fill_event(event)
    
    def _handle_market_event(self, event: MarketEvent):
        """Handle market data event."""
        # Update portfolio with current market data
        self.portfolio.update_timeindex(event.timestamp)
        
        # Generate signals from strategy
        if self.strategy:
            signals = self.strategy.calculate_signals(event)
            for signal in signals:
                self.event_queue.put(signal)
    
    def _handle_signal_event(self, event: SignalEvent):
        """Handle trading signal event."""
        # Convert signal to order
        if event.signal in ['BUY', 'SELL']:
            order = OrderEvent(
                timestamp=event.timestamp,
                event_type=EventType.ORDER,
                symbol=event.symbol,
                order_type='MARKET',
                quantity=int(event.strength * 100),  # Convert to shares
                direction=event.signal,
                data={}
            )
            self.event_queue.put(order)
    
    def _handle_order_event(self, event: OrderEvent):
        """Handle order event."""
        # Execute order through broker
        fill_event = self.broker.execute_order(event)
        if fill_event:
            self.event_queue.put(fill_event)
    
    def _handle_fill_event(self, event: FillEvent):
        """Handle order fill event."""
        # Update portfolio with fill
        self.portfolio.update_fill(event)
        
        # Log trade
        self.trade_log.append({
            'timestamp': event.timestamp,
            'symbol': event.symbol,
            'quantity': event.quantity,
            'direction': event.direction,
            'fill_cost': event.fill_cost,
            'commission': event.commission
        })
        
        # Update portfolio history
        self.portfolio_history.append({
            'timestamp': event.timestamp,
            'total_value': self.portfolio.get_total_value(),
            'cash': self.portfolio.get_cash(),
            'positions': self.portfolio.get_positions()
        })
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics."""
        logger.info("Calculating performance metrics...")
        
        if not self.portfolio_history:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.portfolio_history)
        df['returns'] = df['total_value'].pct_change()
        
        # Calculate metrics
        total_return = (df['total_value'].iloc[-1] / df['total_value'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(df)) - 1
        volatility = df['returns'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = df['total_value'].expanding().max()
        drawdown = (df['total_value'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = len([t for t in self.trade_log if t['direction'] == 'SELL' and t['fill_cost'] > 0])
        total_trades = len(self.trade_log)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        self.performance_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'final_value': df['total_value'].iloc[-1]
        }
        
        logger.info(f"Performance metrics calculated: {self.performance_metrics}")


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self):
        self.data_handler = None
        self.portfolio = None
        self.decision_engine = DecisionEngine()
        self.risk_manager = RiskManager()
    
    def initialize(self, data_handler, portfolio):
        """Initialize the strategy."""
        self.data_handler = data_handler
        self.portfolio = portfolio
    
    @abstractmethod
    def calculate_signals(self, market_event: MarketEvent) -> List[SignalEvent]:
        """Calculate trading signals based on market data."""
        pass


class QuantAIStrategy(TradingStrategy):
    """QuantAI Trading Strategy implementation."""
    
    def __init__(self):
        super().__init__()
        self.symbols = ['AAPL', 'GOOGL', 'NVDA', 'META', 'AMZN']
        self.lookback_period = 20
    
    def calculate_signals(self, market_event: MarketEvent) -> List[SignalEvent]:
        """Calculate signals using QuantAI decision engine."""
        signals = []
        
        for symbol in self.symbols:
            try:
                # Get historical data for the symbol
                historical_data = self._get_historical_data(symbol, market_event.timestamp)
                
                if historical_data is None or historical_data.empty:
                    continue
                
                # Create features
                features = self._create_features(historical_data)
                
                # Get sentiment data
                sentiment_score = self._get_sentiment_score(symbol)
                
                # Get portfolio state
                portfolio_state = {
                    'total_value': self.portfolio.get_total_value(),
                    'cash': self.portfolio.get_cash(),
                    'positions': self.portfolio.get_positions()
                }
                
                # Generate signal using decision engine
                signal_data = self.decision_engine.generate_trading_signal(
                    symbol=symbol,
                    current_market_data=features,
                    sentiment_score=sentiment_score,
                    portfolio_state=portfolio_state
                )
                
                # Create signal event
                signal = SignalEvent(
                    timestamp=market_event.timestamp,
                    event_type=EventType.SIGNAL,
                    symbol=symbol,
                    signal=signal_data['signal'],
                    strength=signal_data['position_size'],
                    data=signal_data
                )
                
                signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error calculating signal for {symbol}: {e}")
                continue
        
        return signals
    
    def _get_historical_data(self, symbol: str, timestamp: datetime) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol."""
        # This would integrate with your data sources
        # For now, return mock data
        return pd.DataFrame({
            'close': np.random.randn(20).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, 20)
        })
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from historical data."""
        # Basic feature engineering
        features = data.copy()
        features['returns'] = features['close'].pct_change()
        features['sma_5'] = features['close'].rolling(5).mean()
        features['sma_20'] = features['close'].rolling(20).mean()
        features['volatility'] = features['returns'].rolling(10).std()
        return features.fillna(0)
    
    def _get_sentiment_score(self, symbol: str) -> float:
        """Get sentiment score for a symbol."""
        # This would integrate with your sentiment analysis
        # For now, return random sentiment
        return np.random.uniform(-1, 1)


# Global backtester instance
qf_backtester = QFBacktester()


async def run_qf_backtest(start_date: datetime = None, 
                         end_date: datetime = None,
                         initial_capital: float = 100000) -> Dict[str, Any]:
    """
    Run QF-Lib backtest.
    
    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        
    Returns:
        Dictionary with backtest results
    """
    if not QF_LIB_AVAILABLE:
        return {
            'status': 'error',
            'error': 'QF-Lib not available. Install with: pip install qf-lib'
        }
    
    try:
        # Initialize backtester
        backtester = QFBacktester(
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date
        )
        
        # Create strategy
        strategy = QuantAIStrategy()
        
        # Run backtest
        results = backtester.run_backtest(strategy)
        
        return results
        
    except Exception as e:
        logger.error(f"QF-Lib backtest failed: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }
