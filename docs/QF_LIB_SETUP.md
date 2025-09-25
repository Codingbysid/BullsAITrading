# 🐍 QF-Lib Setup Guide

This guide helps you set up QF-Lib properly for the QuantAI Trading Platform based on the [official QF-Lib documentation](https://qf-lib.readthedocs.io/en/latest/configuration.html).

## 🔧 Environment Setup

### 1. Create QF-Lib Environment

```bash
# Create new environment with Python 3.11 (recommended by QF-Lib)
conda create --name qf_project python=3.11 -y

# Activate the environment
conda activate qf_project

# Install QF-Lib
pip install qf-lib
```

### 2. Verify Installation

```bash
# Test QF-Lib import
python -c "import qf_lib; print('QF-Lib installed successfully')"
```

## 📋 Configuration Setup

### 1. Environment Variables

QF-Lib requires specific environment variables to be set:

```bash
# Set starting directory (required by QF-Lib)
export QF_STARTING_DIRECTORY="/path/to/your/project"

# Optional: Set other directories
export QF_OUTPUT_DIRECTORY="/path/to/your/project/output"
export QF_DATA_DIRECTORY="/path/to/your/project/data"
export QF_CACHE_DIRECTORY="/path/to/your/project/cache"
export QF_LOGS_DIRECTORY="/path/to/your/project/logs"
```

### 2. Settings Files

QF-Lib requires two configuration files:

#### `config/qf_lib_settings.json`
```json
{
  "company_name": "QuantAI Trading Platform",
  "output_directory": "output",
  "data_directory": "data",
  "cache_directory": "cache",
  "logs_directory": "logs",
  "backtest_settings": {
    "initial_capital": 100000,
    "commission": 0.001,
    "slippage": 0.0005
  }
}
```

#### `config/qf_lib_secret_settings.json`
```json
{
  "data_providers": {
    "alpha_vantage": {
      "api_key": "YOUR_API_KEY_HERE"
    }
  }
}
```

## 🚀 Running QF-Lib Backtester

### Method 1: Using Setup Script

```bash
# Activate environment
conda activate qf_project

# Run setup script
python scripts/setup_qf_lib.py

# Run QF-Lib backtester
python scripts/run_qf_backtest.py
```

### Method 2: Direct Execution

```bash
# Activate environment
conda activate qf_project

# Set environment variables
export QF_STARTING_DIRECTORY="$(pwd)"

# Run backtester directly
python apps/backtesting/backtesters/qf_lib_backtester.py
```

### Method 3: Using Main Launcher

```bash
# Activate environment
conda activate qf_project

# Run with main launcher
python run_quantai.py qf
```

## 🧪 Testing QF-Lib

### Test Environment

```bash
# Test QF-Lib environment
python scripts/test_qf_env.py
```

### Test Configuration

```bash
# Test QF-Lib configuration
python scripts/setup_qf_lib.py
```

## 📊 QF-Lib Features

### Event-Driven Architecture
- **Event Queue**: Manages all market events
- **Portfolio**: Tracks positions and cash
- **Broker**: Executes trades with realistic costs
- **Data Handler**: Provides market data
- **Strategy**: Implements trading logic

### Professional-Grade Backtesting
- **Realistic Market Simulation**: Includes slippage and commissions
- **Risk Management**: Position sizing and risk controls
- **Performance Analytics**: Comprehensive metrics
- **Transaction Costs**: Realistic cost modeling

## 🔧 Troubleshooting

### Common Issues

1. **ImportError: No module named 'qf_lib'**
   ```bash
   # Solution: Install QF-Lib
   pip install qf-lib
   ```

2. **Settings not found**
   ```bash
   # Solution: Create settings files
   python scripts/setup_qf_lib.py
   ```

3. **Starting directory not set**
   ```bash
   # Solution: Set environment variable
   export QF_STARTING_DIRECTORY="$(pwd)"
   ```

4. **Permission errors**
   ```bash
   # Solution: Create directories
   mkdir -p output data cache logs
   ```

### Debug Mode

```bash
# Run with debug logging
export QF_DEBUG=1
python scripts/run_qf_backtest.py
```

## 📚 QF-Lib Documentation

- **Official Documentation**: https://qf-lib.readthedocs.io/
- **Configuration Guide**: https://qf-lib.readthedocs.io/en/latest/configuration.html
- **API Reference**: https://qf-lib.readthedocs.io/en/latest/api_reference.html

## 🎯 Next Steps

1. **Setup Environment**: Follow the environment setup steps
2. **Configure Settings**: Create the required settings files
3. **Test Installation**: Run the test scripts
4. **Run Backtester**: Execute the QF-Lib backtester
5. **Analyze Results**: Review the backtest results

## 🚀 Ready to Go!

Once you've completed the setup, you can run the QF-Lib backtester:

```bash
conda activate qf_project
python run_quantai.py qf
```

The QF-Lib backtester will provide professional-grade backtesting with realistic market simulation, comprehensive risk management, and detailed performance analytics.
