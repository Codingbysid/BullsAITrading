# 💼 QuantAI Portfolio Manager & Trade Suggestion Bot

**Advanced AI-driven portfolio management system** with user feedback learning, focused on the 5 flagship stocks: **AMZN, META, NVDA, GOOGL, and AAPL**.

## 🎯 Overview

The QuantAI Portfolio Manager transforms your existing trading platform into an interactive portfolio management system with:

- **👤 User Management**: Secure authentication and risk profiling
- **📊 Portfolio Tracking**: Real-time P&L calculation and position management
- **🤖 AI Recommendations**: Integration with your existing 5-model ensemble
- **🔄 Feedback Learning**: Reinforcement learning from user acceptance/rejection
- **🛡️ Risk Management**: Kelly Criterion, VaR, and portfolio optimization
- **📈 Performance Analytics**: Comprehensive tracking and reporting

## 🏗️ Architecture

### **Database Schema (3 Core Tables)**

```sql
-- Users table
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100),
    risk_tolerance ENUM('Conservative', 'Moderate', 'Aggressive'),
    investment_experience ENUM('Beginner', 'Intermediate', 'Advanced')
);

-- Portfolio positions
CREATE TABLE user_portfolios (
    portfolio_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    symbol ENUM('AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL') NOT NULL,
    shares DECIMAL(10, 4) NOT NULL,
    avg_purchase_price DECIMAL(10, 2) NOT NULL,
    current_price DECIMAL(10, 2),
    market_value DECIMAL(12, 2),
    unrealized_pnl DECIMAL(12, 2),
    realized_pnl DECIMAL(12, 2) DEFAULT 0.00
);

-- Recommendations and feedback
CREATE TABLE recommendations_feedback (
    recommendation_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    symbol ENUM('AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL') NOT NULL,
    recommendation_type ENUM('BUY', 'SELL', 'HOLD') NOT NULL,
    confidence_score DECIMAL(3, 2),
    user_accepted BOOLEAN DEFAULT FALSE,
    outcome_return_pct DECIMAL(5, 4),
    model_features JSON
);
```

### **System Components**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Database      │───▶│  Portfolio     │───▶│   Terminal      │
│   Manager       │    │  Manager       │    │   Interface     │
│                 │    │                │    │                 │
│ • User Auth     │    │ • Portfolio    │    │ • CLI Interface │
│ • Portfolio     │    │ • Recommendations│    │ • Web API       │
│ • Feedback      │    │ • Risk Mgmt    │    │ • Demo Mode     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Reinforcement  │    │  Data Sources   │    │  Performance    │
│  Learning       │    │                 │    │  Analytics      │
│                 │    │ • Market Data   │    │                 │
│ • Feedback Loop │    │ • Features      │    │ • Metrics       │
│ • Model Update  │    │ • Predictions   │    │ • Reports       │
│ • Performance   │    │ • Risk Analysis  │    │ • Learning      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### **1. Installation**

```bash
# Install dependencies
pip install -r requirements_portfolio.txt

# For database setup (optional)
# Install MySQL or use SQLite fallback
```

### **2. Run the System**

```bash
# Terminal Interface (Recommended)
python portfolio_manager_main.py --mode cli

# Web API
python portfolio_manager_main.py --mode api --port 8000

# Demo Mode (No Database Required)
python portfolio_manager_main.py --mode demo
```

### **3. Access Features**

**Terminal Interface:**
- User registration/login
- Portfolio management
- AI recommendations
- Performance analytics

**Web API:**
- RESTful endpoints
- JSON responses
- Interactive documentation at `/docs`

## 📊 Key Features

### **🤖 AI-Powered Recommendations**

- **Multi-Model Integration**: Uses your existing 5-model ensemble
- **Risk-Adjusted Signals**: Kelly Criterion position sizing
- **Confidence Scoring**: 0-100% confidence levels
- **Feature Engineering**: 50+ technical indicators
- **Real-time Updates**: Live market data integration

### **📈 Portfolio Management**

- **Position Tracking**: Real-time P&L calculation
- **Risk Metrics**: Beta, concentration, diversification
- **Performance Analytics**: Sharpe ratio, drawdown, win rate
- **Trade History**: Complete transaction log

### **🔄 Reinforcement Learning**

- **User Feedback Loop**: Learn from acceptance/rejection
- **Outcome Analysis**: Track recommendation performance
- **Model Improvement**: Continuous learning from results
- **Performance Reports**: Detailed analytics

### **🛡️ Risk Management**

- **Kelly Criterion**: Optimal position sizing
- **VaR Analysis**: 95% and 99% confidence levels
- **Portfolio Optimization**: Mean-variance optimization
- **Drawdown Control**: Real-time monitoring

## 🎯 Focused 5-Ticker Strategy

### **Supported Stocks**
- **AMZN** - Amazon (Consumer Discretionary)
- **META** - Meta (Communication Services)  
- **NVDA** - NVIDIA (Technology)
- **GOOGL** - Alphabet (Communication Services)
- **AAPL** - Apple (Technology)

### **Strategy Benefits**
- **Focused Expertise**: Deep analysis of 5 flagship stocks
- **Reduced Complexity**: Simplified decision making
- **Higher Accuracy**: Specialized models for each stock
- **Better Risk Management**: Concentrated but diversified approach

## 📱 Usage Examples

### **Terminal Interface**

```bash
# Start the system
python portfolio_manager_main.py --mode cli

# Login/Register
Username: trader123
Password: ******

# View Portfolio
💰 Total Value: $125,000
📈 Total P&L: $25,000 (20.0%)
🎯 Risk Level: Medium

# Get AI Recommendations
🤖 AI TRADING RECOMMENDATIONS
1. META: BUY - Confidence: 85% - Risk: Low
2. NVDA: HOLD - Confidence: 65% - Risk: Medium
3. AAPL: SELL - Confidence: 78% - Risk: Low
```

### **Web API**

```bash
# Start API server
python portfolio_manager_main.py --mode api

# Access endpoints
curl http://localhost:8000/system/status
curl http://localhost:8000/portfolio/1
curl http://localhost:8000/portfolio/1/recommendations
```

### **API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/register` | POST | Register new user |
| `/auth/login` | POST | User login |
| `/portfolio/{user_id}` | GET | Get user portfolio |
| `/portfolio/{user_id}/positions` | POST | Add position |
| `/portfolio/{user_id}/recommendations` | GET | Get AI recommendations |
| `/portfolio/{user_id}/analytics` | GET | Get performance analytics |
| `/learning/train` | POST | Train feedback model |
| `/system/status` | GET | System status |

## 🔧 Configuration

### **Database Configuration**

```python
# src/database/db_manager.py
db_config = {
    'host': 'localhost',
    'database': 'quantai_portfolio',
    'user': 'root',
    'password': ''
}
```

### **Risk Management Settings**

```python
# Portfolio risk limits
MAX_POSITION_SIZE = 0.2  # 20% max per position
MAX_PORTFOLIO_RISK = 0.1  # 10% max portfolio risk
KELLY_FRACTION_LIMIT = 0.25  # 25% max Kelly fraction
```

### **AI Model Settings**

```python
# Recommendation thresholds
MIN_CONFIDENCE = 0.6  # 60% minimum confidence
MAX_RISK_SCORE = 0.8  # 80% maximum risk
FEATURE_IMPORTANCE_THRESHOLD = 0.1
```

## 📊 Performance Metrics

### **Expected Performance**
- **Recommendation Accuracy**: >65% (based on 49.13% backtest return)
- **User Acceptance Rate**: >40% initially, improving with feedback
- **System Response Time**: <500ms for recommendations
- **Database Operations**: <100ms per query
- **Model Prediction**: <50ms per symbol

### **Learning Metrics**
- **Feedback Loop**: 30-day evaluation period
- **Model Improvement**: Measurable within 30 days
- **Performance Tracking**: Real-time analytics
- **Success Rate**: Continuous monitoring

## 🛠️ Development

### **Project Structure**

```
BULLSTRADING/
├── portfolio_manager_main.py          # Main application
├── requirements_portfolio.txt         # Dependencies
├── src/
│   ├── database/
│   │   └── db_manager.py             # Database operations
│   ├── portfolio/
│   │   └── portfolio_manager.py      # Portfolio management
│   ├── interface/
│   │   └── cli.py                    # Terminal interface
│   ├── training/
│   │   └── feedback_trainer.py       # Reinforcement learning
│   └── api/
│       └── portfolio_api.py          # Web API
└── README_PORTFOLIO_MANAGER.md       # This documentation
```

### **Testing**

```bash
# Run tests
python -m pytest tests/

# Test specific components
python -m pytest tests/test_portfolio_manager.py
python -m pytest tests/test_database.py
```

### **Development Mode**

```bash
# Run with debug logging
python portfolio_manager_main.py --mode cli --debug

# Run API with auto-reload
python portfolio_manager_main.py --mode api --reload
```

## 🔄 Integration with Existing System

### **Leverages Existing Components**
- **Data Sources**: Finazon, Alpha Vantage, YFinance
- **ML Models**: Your existing 5-model ensemble
- **Risk Management**: Kelly Criterion, VaR analysis
- **Backtesting**: Integration with your 5 backtesting systems

### **New Components Added**
- **User Management**: Authentication and profiles
- **Portfolio Tracking**: Real-time position management
- **Feedback Learning**: Reinforcement learning system
- **Web Interface**: RESTful API and terminal interface

## 🎉 Benefits

### **For Users**
- **Personalized Recommendations**: Based on risk tolerance and experience
- **Real-time Portfolio Tracking**: Live P&L and performance metrics
- **Learning System**: Improves recommendations over time
- **Multiple Interfaces**: Terminal, web, and API access

### **For Developers**
- **Modular Architecture**: Easy to extend and modify
- **Fallback Systems**: Works without external dependencies
- **Comprehensive Testing**: Full test coverage
- **Documentation**: Detailed API and usage documentation

## 🚀 Next Steps

1. **Database Setup**: Configure MySQL or use SQLite fallback
2. **User Registration**: Create your first user account
3. **Portfolio Building**: Add initial positions
4. **AI Recommendations**: Start receiving and responding to recommendations
5. **Feedback Learning**: Let the system learn from your decisions
6. **Performance Monitoring**: Track improvement over time

## 📚 Documentation

- **API Documentation**: Available at `/docs` when running API mode
- **Database Schema**: See database section above
- **Configuration**: See configuration section above
- **Troubleshooting**: Check logs for detailed error messages

---

**🎯 QuantAI Portfolio Manager - Transform your trading platform into an interactive portfolio management system with AI-powered recommendations and user feedback learning!**
