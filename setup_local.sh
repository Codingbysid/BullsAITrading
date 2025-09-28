#!/bin/bash

# QuantAI Portfolio Manager - Local Environment Setup Script

echo "🚀 QuantAI Portfolio Manager - Local Environment Setup"
echo "======================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if we're in the right directory
if [ ! -f "run_portfolio.py" ]; then
    echo "❌ run_portfolio.py not found. Please run this script from the project root directory."
    exit 1
fi

echo "✅ Project directory confirmed"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "📦 Installing required packages..."
pip install --upgrade pip

# Install core packages
pip install pandas numpy scipy scikit-learn matplotlib seaborn

# Install optional packages (with fallbacks)
pip install xgboost || echo "⚠️  XGBoost installation failed (optional)"
pip install tensorflow || echo "⚠️  TensorFlow installation failed (optional)"
pip install uvicorn fastapi || echo "⚠️  Web API packages installation failed (optional)"

echo "✅ Package installation completed"

# Run initial setup
echo "🔄 Running initial setup..."
python run_portfolio.py --setup

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the portfolio manager: python run_portfolio.py --run"
echo "3. Or try demo mode: python run_portfolio.py --demo"
echo ""
echo "🚀 Happy trading!"
