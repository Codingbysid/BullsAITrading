# 📋 QuantAI Trading Platform - Coding Standards

This document outlines the coding standards and best practices for the QuantAI Trading Platform project.

## 🎯 **Core Principles**

### 1. **Consistent Naming Conventions**

#### **Variables and Functions**
```python
# ✅ Good: Descriptive, snake_case
def calculate_portfolio_return(portfolio_data: pd.DataFrame) -> float:
    initial_value = portfolio_data.iloc[0]['value']
    final_value = portfolio_data.iloc[-1]['value']
    return (final_value - initial_value) / initial_value

# ❌ Bad: Unclear, inconsistent
def calc_ret(data):
    v1 = data.iloc[0]['v']
    v2 = data.iloc[-1]['v']
    return (v2 - v1) / v1
```

#### **Classes**
```python
# ✅ Good: PascalCase, descriptive
class PortfolioManager:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital

# ❌ Bad: Unclear purpose
class PM:
    def __init__(self, ic):
        self.ic = ic
```

#### **Constants**
```python
# ✅ Good: UPPER_SNAKE_CASE
MAX_POSITION_SIZE = 0.1
RISK_FREE_RATE = 0.02
DEFAULT_COMMISSION = 0.001

# ❌ Bad: Inconsistent casing
maxPositionSize = 0.1
risk_free_rate = 0.02
```

### 2. **Code Readability**

#### **Function Structure**
```python
# ✅ Good: Single responsibility, clear parameters
def calculate_sharpe_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio for a series of returns.
    
    Args:
        returns: Series of portfolio returns
        risk_free_rate: Risk-free rate (default: 2%)
        periods_per_year: Number of periods per year (default: 252)
        
    Returns:
        Sharpe ratio as float
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    return excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)

# ❌ Bad: Multiple responsibilities, unclear parameters
def calc_ratio(r, rf=0.02, ppy=252):
    er = r - rf / ppy
    return er.mean() / r.std() * np.sqrt(ppy)
```

#### **Class Structure**
```python
# ✅ Good: Clear separation of concerns
class RiskManager:
    """Manages portfolio risk and position sizing."""
    
    def __init__(self, max_position_size: float = 0.1):
        self.max_position_size = max_position_size
        self.positions = {}
    
    def calculate_position_size(
        self, 
        signal_strength: float, 
        volatility: float
    ) -> float:
        """Calculate position size based on signal and volatility."""
        base_size = signal_strength * self.max_position_size
        volatility_adjustment = 1.0 / (1.0 + volatility)
        return base_size * volatility_adjustment
```

### 3. **DRY Principle (Don't Repeat Yourself)**

#### **Extract Common Logic**
```python
# ✅ Good: Reusable utility functions
def calculate_technical_indicators(price_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate common technical indicators."""
    data = price_data.copy()
    
    # Moving averages
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    
    # RSI
    data['rsi'] = calculate_rsi(data['close'])
    
    # Bollinger Bands
    data['bb_upper'], data['bb_lower'] = calculate_bollinger_bands(data['close'])
    
    return data

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

### 4. **Comprehensive Documentation**

#### **Module Documentation**
```python
"""
Risk Management Module for QuantAI Trading Platform.

This module provides comprehensive risk management capabilities including:
- Position sizing using Kelly Criterion
- Value at Risk (VaR) calculations
- Portfolio optimization
- Risk budgeting and allocation

Classes:
    RiskManager: Main risk management class
    VaRCalculator: Value at Risk calculations
    PortfolioOptimizer: Portfolio optimization

Functions:
    calculate_kelly_fraction: Kelly Criterion position sizing
    calculate_var: Value at Risk calculation
    optimize_portfolio: Portfolio optimization
"""
```

#### **Function Documentation**
```python
def calculate_kelly_fraction(
    win_probability: float,
    average_win: float,
    average_loss: float,
    max_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly Criterion position size.
    
    The Kelly Criterion determines the optimal position size based on:
    - Win probability
    - Average win amount
    - Average loss amount
    
    Args:
        win_probability: Probability of winning (0.0 to 1.0)
        average_win: Average win amount
        average_loss: Average loss amount
        max_fraction: Maximum position size (default: 25%)
        
    Returns:
        Optimal position size as fraction of capital
        
    Raises:
        ValueError: If win_probability is not between 0 and 1
        ValueError: If average_win or average_loss is negative
        
    Example:
        >>> kelly = calculate_kelly_fraction(0.6, 0.1, 0.05)
        >>> print(f"Kelly fraction: {kelly:.2%}")
        Kelly fraction: 20.00%
    """
    if not 0 <= win_probability <= 1:
        raise ValueError("Win probability must be between 0 and 1")
    
    if average_win < 0 or average_loss < 0:
        raise ValueError("Average win and loss must be positive")
    
    kelly = (win_probability * average_win - (1 - win_probability) * average_loss) / average_win
    return min(max(kelly, 0), max_fraction)
```

### 5. **Error Handling**

#### **Structured Error Handling**
```python
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class TradingError(Exception):
    """Base exception for trading-related errors."""
    pass

class InsufficientDataError(TradingError):
    """Raised when insufficient data is available."""
    pass

class RiskLimitExceededError(TradingError):
    """Raised when risk limits are exceeded."""
    pass

def execute_trade(
    symbol: str,
    quantity: int,
    order_type: str,
    price: Optional[float] = None
) -> Dict[str, Any]:
    """
    Execute a trade with comprehensive error handling.
    
    Args:
        symbol: Trading symbol
        quantity: Number of shares
        order_type: Type of order ('market', 'limit')
        price: Limit price (required for limit orders)
        
    Returns:
        Dictionary containing trade result
        
    Raises:
        InsufficientDataError: If market data is unavailable
        RiskLimitExceededError: If trade exceeds risk limits
        ValueError: If parameters are invalid
    """
    try:
        # Validate inputs
        if not symbol or quantity <= 0:
            raise ValueError("Invalid symbol or quantity")
        
        if order_type == 'limit' and price is None:
            raise ValueError("Price required for limit orders")
        
        # Check market data availability
        if not is_market_data_available(symbol):
            raise InsufficientDataError(f"Market data unavailable for {symbol}")
        
        # Check risk limits
        if exceeds_risk_limits(symbol, quantity):
            raise RiskLimitExceededError(f"Trade exceeds risk limits for {symbol}")
        
        # Execute trade
        result = _execute_trade_internal(symbol, quantity, order_type, price)
        
        logger.info(f"Trade executed: {symbol} {quantity} shares at {result['price']}")
        return result
        
    except (InsufficientDataError, RiskLimitExceededError) as e:
        logger.error(f"Trade execution failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in trade execution: {e}")
        raise TradingError(f"Trade execution failed: {e}") from e
```

### 6. **Type Hints**

#### **Comprehensive Type Annotations**
```python
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

class OrderType(Enum):
    """Order types for trading."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

@dataclass
class Trade:
    """Represents a single trade."""
    symbol: str
    quantity: int
    price: float
    timestamp: pd.Timestamp
    order_type: OrderType
    commission: float = 0.0

def calculate_portfolio_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio metrics.
    
    Args:
        returns: Portfolio returns
        benchmark_returns: Benchmark returns for comparison
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {
        'total_return': (1 + returns).prod() - 1,
        'annualized_return': (1 + returns).prod() ** (252 / len(returns)) - 1,
        'volatility': returns.std() * np.sqrt(252),
        'sharpe_ratio': (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)
    }
    
    if benchmark_returns is not None:
        excess_returns = returns - benchmark_returns
        metrics['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        metrics['beta'] = returns.cov(benchmark_returns) / benchmark_returns.var()
        metrics['alpha'] = metrics['annualized_return'] - risk_free_rate - metrics['beta'] * (benchmark_returns.mean() * 252 - risk_free_rate)
    
    return metrics
```

## 🛠️ **Automated Tools**

### **Pre-commit Hooks**
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

### **Code Formatting**
```bash
# Format code with Black
black src/ apps/ tests/

# Sort imports with isort
isort src/ apps/ tests/

# Check code style with flake8
flake8 src/ apps/ tests/
```

### **Type Checking**
```bash
# Run type checking with mypy
mypy src/ apps/

# Run with strict mode
mypy --strict src/ apps/
```

### **Testing**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=apps

# Run specific test types
pytest -m unit
pytest -m integration
pytest -m performance
```

## 📊 **Code Quality Metrics**

### **Target Metrics**
- **Cyclomatic Complexity**: < 10 per function
- **Function Length**: < 50 lines
- **Class Length**: < 200 lines
- **Test Coverage**: > 80%
- **Documentation Coverage**: > 90%

### **Monitoring Tools**
- **SonarQube**: Code quality gates
- **CodeClimate**: Maintainability scores
- **Coverage.py**: Test coverage
- **Bandit**: Security issues

## 🚀 **Implementation Checklist**

- [ ] Consistent naming conventions
- [ ] Clear function and class documentation
- [ ] Comprehensive type hints
- [ ] Error handling with custom exceptions
- [ ] Unit tests for all functions
- [ ] Integration tests for workflows
- [ ] Performance tests for critical paths
- [ ] Code formatting with Black
- [ ] Import sorting with isort
- [ ] Linting with flake8
- [ ] Type checking with mypy
- [ ] Security scanning with bandit
- [ ] Pre-commit hooks configured
- [ ] CI/CD pipeline with quality gates

## 📚 **Resources**

- [PEP 8 - Python Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Black Code Formatter](https://black.readthedocs.io/)
- [MyPy Type Checker](https://mypy.readthedocs.io/)
- [Pytest Testing Framework](https://docs.pytest.org/)
- [Pre-commit Hooks](https://pre-commit.com/)
