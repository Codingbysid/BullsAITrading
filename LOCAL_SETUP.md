# 🚀 QuantAI Portfolio Manager - Local Environment Setup

This guide will help you set up and run the QuantAI Portfolio Manager in your local environment with the integrated four-model decision engine and trained ML ensemble models.

## 📋 Prerequisites

- Python 3.8 or higher
- Terminal/Command Prompt access
- Internet connection (for package installation)

## 🛠️ Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Make the setup script executable and run it
chmod +x setup_local.sh
./setup_local.sh
```

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install packages
pip install --upgrade pip
pip install pandas numpy scipy scikit-learn matplotlib seaborn

# 4. Run initial setup
python run_portfolio.py --setup
```

## 🚀 Running the Portfolio Manager

### First Time Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Run initial setup
python run_portfolio.py --setup
```

### Run the Portfolio Manager
```bash
# Activate virtual environment
source venv/bin/activate

# Start the portfolio manager
python run_portfolio.py --run
```

### Demo Mode
```bash
# Activate virtual environment
source venv/bin/activate

# Run demo mode
python run_portfolio.py --demo
```

## 📊 Features Available

### 🧠 Four-Model Decision Engine
- **Sentiment Analysis Model** (25% weight)
- **Quantitative Risk Model** (25% weight)
- **Trained ML Ensemble Model** (35% weight)
- **RL Decider Agent** (Final decision maker)

### 📈 Portfolio Management
- Real-time portfolio tracking
- Performance analytics
- Risk management (Kelly Criterion, VaR)
- Position sizing and rebalancing

### 🤖 AI Recommendations
- Buy/Sell/Hold recommendations
- Confidence scoring
- Risk-adjusted suggestions
- Model attribution and reasoning

### 📊 Analytics & Reporting
- Sharpe ratio, max drawdown
- Win rate, profit factor
- Portfolio performance metrics
- Risk-adjusted returns

## 🎯 Supported Symbols

The system is optimized for these 5 flagship stocks:
- **AAPL** (Apple Inc.)
- **AMZN** (Amazon.com Inc.)
- **GOOGL** (Alphabet Inc.)
- **META** (Meta Platforms Inc.)
- **NVDA** (NVIDIA Corporation)

## 📁 Project Structure

```
QuantAI-Trading-Platform/
├── run_portfolio.py              # Main launcher script
├── setup_local.sh               # Automated setup script
├── config/local/                # Local configuration
│   └── portfolio_config.json    # Portfolio settings
├── data/                        # Data storage
│   ├── portfolios/              # Portfolio data
│   ├── recommendations/         # AI recommendations
│   └── learning/                # Learning data
├── models/                      # Trained ML models
│   ├── linear_model_model.json
│   ├── naive_bayes_model.json
│   ├── decision_tree_model.json
│   └── simple_ensemble_metadata.json
└── src/                         # Source code
    ├── models/                  # ML models
    ├── decision_engine/         # Decision engine
    ├── utils/                   # Utilities
    └── apps/portfolio/          # Portfolio management
```

## 🖥️ Interactive Commands

When running the portfolio manager, you'll have access to these commands:

1. **📊 View Portfolio Status** - Current portfolio value, positions, performance
2. **🤖 Get AI Recommendations** - AI-powered buy/sell/hold suggestions
3. **📈 View Performance Analytics** - Detailed performance metrics
4. **⚙️ Update Portfolio Settings** - Adjust risk tolerance and preferences
5. **🔄 Simulate Trading** - Execute simulated trades based on recommendations
6. **📊 View Market Data** - Display market data for symbols
7. **🎯 Test Decision Engine** - Test the four-model decision engine
8. **❌ Exit** - Close the application

## 🔧 Configuration

### Portfolio Settings
- **Initial Capital**: $100,000 (configurable)
- **Risk Tolerance**: 5% (configurable)
- **Max Position Size**: 30% per symbol
- **Max Portfolio Risk**: 15%
- **Rebalance Frequency**: Weekly

### Model Weights
- **Sentiment Model**: 25%
- **Quantitative Model**: 25%
- **ML Ensemble Model**: 35%
- **RL Agent**: Final decision maker

## 📊 Sample Data

The system includes sample market data for all 5 symbols:
- Historical price data (OHLCV)
- Technical indicators
- Market sentiment data
- Risk metrics

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure virtual environment is activated
   source venv/bin/activate
   
   # Reinstall packages
   pip install -r requirements.txt
   ```

2. **Permission Errors**
   ```bash
   # Make scripts executable
   chmod +x setup_local.sh
   chmod +x run_portfolio.py
   ```

3. **Python Version Issues**
   ```bash
   # Check Python version
   python3 --version
   
   # Use python3 explicitly
   python3 run_portfolio.py --setup
   ```

### Getting Help

If you encounter issues:
1. Check the logs in the `logs/` directory
2. Verify all dependencies are installed
3. Ensure you're in the correct directory
4. Try running in demo mode first

## 🎉 Getting Started

1. **Clone the repository** (if not already done)
2. **Run the setup script**: `./setup_local.sh`
3. **Activate virtual environment**: `source venv/bin/activate`
4. **Start the portfolio manager**: `python run_portfolio.py --run`
5. **Explore the features** using the interactive menu

## 📈 Next Steps

Once you're comfortable with the local setup:
- Customize the configuration in `config/local/portfolio_config.json`
- Add your own market data sources
- Integrate with real trading APIs
- Deploy to a cloud environment

Happy trading! 🚀📊🤖
