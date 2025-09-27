#!/usr/bin/env python3
"""
QuantAI Portfolio Manager - Local Environment Setup

This script sets up and runs the QuantAI Portfolio Manager in your local environment
with the integrated four-model decision engine and trained ML ensemble models.

Usage:
    python run_portfolio.py --setup    # First time setup
    python run_portfolio.py --run      # Run the portfolio manager
    python run_portfolio.py --demo     # Run demo mode
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

def setup_local_environment():
    """Set up the local environment for portfolio management."""
    print("🚀 Setting up QuantAI Portfolio Manager for Local Environment")
    print("=" * 70)
    
    # Create necessary directories
    directories = [
        "data/portfolios",
        "data/recommendations", 
        "data/learning",
        "logs",
        "config/local"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create local configuration
    local_config = {
        "user_id": "local_user",
        "portfolio_id": "local_portfolio",
        "initial_capital": 100000.0,
        "risk_tolerance": 0.05,
        "symbols": ["AAPL", "AMZN", "GOOGL", "META", "NVDA"],
        "model_weights": {
            "sentiment": 0.25,
            "quantitative": 0.25,
            "ml_ensemble": 0.35,
            "rl_agent": 1.0
        },
        "trading_settings": {
            "max_position_size": 0.30,
            "max_portfolio_risk": 0.15,
            "rebalance_frequency": "weekly",
            "stop_loss": 0.10,
            "take_profit": 0.20
        },
        "data_sources": {
            "use_real_data": False,  # Use synthetic data for local testing
            "update_frequency": "daily"
        }
    }
    
    config_path = Path("config/local/portfolio_config.json")
    with open(config_path, 'w') as f:
        json.dump(local_config, f, indent=2)
    print(f"✅ Created local configuration: {config_path}")
    
    # Create initial portfolio
    initial_portfolio = {
        "portfolio_id": "local_portfolio",
        "user_id": "local_user",
        "created_at": datetime.now().isoformat(),
        "initial_capital": 100000.0,
        "current_value": 100000.0,
        "cash_balance": 100000.0,
        "positions": {},
        "performance_metrics": {
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0
        },
        "risk_metrics": {
            "portfolio_risk": 0.0,
            "var_95": 0.0,
            "beta": 0.0,
            "correlation": {}
        }
    }
    
    portfolio_path = Path("data/portfolios/local_portfolio.json")
    with open(portfolio_path, 'w') as f:
        json.dump(initial_portfolio, f, indent=2)
    print(f"✅ Created initial portfolio: {portfolio_path}")
    
    # Create sample market data
    create_sample_market_data()
    
    print("\n🎉 Local environment setup complete!")
    print("📁 Configuration files created in config/local/")
    print("📊 Sample data created in data/")
    print("🚀 Ready to run the portfolio manager!")

def create_sample_market_data():
    """Create sample market data for local testing."""
    print("📊 Creating sample market data...")
    
    symbols = ["AAPL", "AMZN", "GOOGL", "META", "NVDA"]
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    for symbol in symbols:
        # Generate realistic price data
        np.random.seed(hash(symbol) % 2**32)  # Consistent random data per symbol
        
        base_price = {
            "AAPL": 150.0,
            "AMZN": 3200.0, 
            "GOOGL": 2800.0,
            "META": 300.0,
            "NVDA": 400.0
        }[symbol]
        
        # Generate price series with trend and volatility
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, base_price * 0.5))  # Prevent negative prices
        
        # Create OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            volatility = 0.01
            high = price * (1 + np.random.uniform(0, volatility))
            low = price * (1 - np.random.uniform(0, volatility))
            open_price = prices[i-1] if i > 0 else price
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                "Date": date.strftime('%Y-%m-%d'),
                "Open": round(open_price, 2),
                "High": round(high, 2),
                "Low": round(low, 2),
                "Close": round(price, 2),
                "Volume": volume
            })
        
        # Save to CSV
        df = pd.DataFrame(data)
        data_path = Path(f"data/{symbol}_sample_data.csv")
        df.to_csv(data_path, index=False)
        print(f"✅ Created sample data for {symbol}: {len(data)} records")

def run_portfolio_manager():
    """Run the portfolio manager with local configuration."""
    print("🚀 Starting QuantAI Portfolio Manager")
    print("=" * 50)
    
    try:
        # Import and initialize the portfolio manager
        from src.apps.portfolio.enhanced_portfolio_manager import EnhancedPortfolioManager
        
        # Initialize with local configuration
        config_path = Path("config/local/portfolio_config.json")
        if not config_path.exists():
            print("❌ Local configuration not found. Run with --setup first.")
            return
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize portfolio manager
        portfolio_manager = EnhancedPortfolioManager(
            user_id=config["user_id"],
            portfolio_id=config["portfolio_id"],
            initial_capital=config["initial_capital"],
            risk_tolerance=config["risk_tolerance"]
        )
        
        print("✅ Portfolio manager initialized")
        print(f"💰 Initial capital: ${config['initial_capital']:,.2f}")
        print(f"🎯 Risk tolerance: {config['risk_tolerance']:.1%}")
        print(f"📊 Symbols: {', '.join(config['symbols'])}")
        
        # Start interactive mode
        run_interactive_mode(portfolio_manager, config)
        
    except ImportError as e:
        print(f"❌ Failed to import portfolio manager: {e}")
        print("🔄 Running in fallback mode...")
        run_fallback_mode()
    except Exception as e:
        print(f"❌ Error starting portfolio manager: {e}")
        run_fallback_mode()

def run_interactive_mode(portfolio_manager, config):
    """Run interactive portfolio management mode."""
    print("\n🖥️  Interactive Portfolio Manager")
    print("=" * 40)
    
    while True:
        print("\n📋 QUANTAI PORTFOLIO MANAGER")
        print("1. 📊 View Portfolio Status")
        print("2. 🤖 Get AI Recommendations")
        print("3. 📈 View Performance Analytics")
        print("4. ⚙️  Update Portfolio Settings")
        print("5. 🔄 Simulate Trading")
        print("6. 📊 View Market Data")
        print("7. 🎯 Test Decision Engine")
        print("8. ❌ Exit")
        
        choice = input("\nSelect option (1-8): ").strip()
        
        if choice == '1':
            view_portfolio_status(portfolio_manager)
        elif choice == '2':
            get_ai_recommendations(portfolio_manager, config)
        elif choice == '3':
            view_performance_analytics(portfolio_manager)
        elif choice == '4':
            update_portfolio_settings(portfolio_manager, config)
        elif choice == '5':
            simulate_trading(portfolio_manager, config)
        elif choice == '6':
            view_market_data(config)
        elif choice == '7':
            test_decision_engine(portfolio_manager, config)
        elif choice == '8':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Please try again.")

def view_portfolio_status(portfolio_manager):
    """View current portfolio status."""
    print("\n📊 PORTFOLIO STATUS")
    print("=" * 30)
    
    try:
        # Get portfolio data
        portfolio_data = portfolio_manager.get_portfolio_summary()
        
        print(f"💰 Total Value: ${portfolio_data.get('total_value', 0):,.2f}")
        print(f"💵 Cash Balance: ${portfolio_data.get('cash_balance', 0):,.2f}")
        print(f"📈 Total Return: {portfolio_data.get('total_return', 0):.2%}")
        print(f"🎯 Risk Level: {portfolio_data.get('risk_level', 'Unknown')}")
        
        positions = portfolio_data.get('positions', {})
        if positions:
            print("\n📋 Current Positions:")
            for symbol, position in positions.items():
                print(f"  {symbol}: {position.get('shares', 0)} shares @ ${position.get('avg_price', 0):.2f}")
        else:
            print("\n📋 No current positions")
            
    except Exception as e:
        print(f"❌ Error getting portfolio status: {e}")

def get_ai_recommendations(portfolio_manager, config):
    """Get AI recommendations for the portfolio."""
    print("\n🤖 AI RECOMMENDATIONS")
    print("=" * 30)
    
    try:
        symbols = config["symbols"]
        recommendations = []
        
        for symbol in symbols:
            # Get recommendation from the four-model decision engine
            recommendation = portfolio_manager.get_ai_recommendation(symbol)
            recommendations.append({
                "symbol": symbol,
                "action": recommendation.get("action", "HOLD"),
                "confidence": recommendation.get("confidence", 0.0),
                "reasoning": recommendation.get("reasoning", "No reasoning available")
            })
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['symbol']}: {rec['action']}")
            print(f"   Confidence: {rec['confidence']:.1%}")
            print(f"   Reasoning: {rec['reasoning']}")
            print()
            
    except Exception as e:
        print(f"❌ Error getting recommendations: {e}")

def view_performance_analytics(portfolio_manager):
    """View performance analytics."""
    print("\n📈 PERFORMANCE ANALYTICS")
    print("=" * 30)
    
    try:
        analytics = portfolio_manager.get_performance_analytics()
        
        print(f"📊 Total Return: {analytics.get('total_return', 0):.2%}")
        print(f"🎯 Sharpe Ratio: {analytics.get('sharpe_ratio', 0):.2f}")
        print(f"⚠️  Max Drawdown: {analytics.get('max_drawdown', 0):.2%}")
        print(f"🏆 Win Rate: {analytics.get('win_rate', 0):.1%}")
        print(f"📈 Total Trades: {analytics.get('total_trades', 0)}")
        
    except Exception as e:
        print(f"❌ Error getting analytics: {e}")

def update_portfolio_settings(portfolio_manager, config):
    """Update portfolio settings."""
    print("\n⚙️  PORTFOLIO SETTINGS")
    print("=" * 30)
    
    print(f"Current risk tolerance: {config['risk_tolerance']:.1%}")
    new_risk = input("Enter new risk tolerance (0.01-0.20): ").strip()
    
    try:
        new_risk = float(new_risk)
        if 0.01 <= new_risk <= 0.20:
            config['risk_tolerance'] = new_risk
            portfolio_manager.update_risk_tolerance(new_risk)
            print(f"✅ Risk tolerance updated to {new_risk:.1%}")
        else:
            print("❌ Invalid risk tolerance. Must be between 0.01 and 0.20.")
    except ValueError:
        print("❌ Invalid input. Please enter a number.")

def simulate_trading(portfolio_manager, config):
    """Simulate trading based on recommendations."""
    print("\n🔄 SIMULATE TRADING")
    print("=" * 30)
    
    symbol = input("Enter symbol to trade (AAPL, AMZN, GOOGL, META, NVDA): ").strip().upper()
    
    if symbol not in config["symbols"]:
        print("❌ Invalid symbol.")
        return
    
    try:
        # Get recommendation
        recommendation = portfolio_manager.get_ai_recommendation(symbol)
        action = recommendation.get("action", "HOLD")
        confidence = recommendation.get("confidence", 0.0)
        
        print(f"\n🤖 AI Recommendation for {symbol}:")
        print(f"Action: {action}")
        print(f"Confidence: {confidence:.1%}")
        
        if action != "HOLD":
            proceed = input(f"\nExecute {action} order for {symbol}? (y/n): ").strip().lower()
            if proceed == 'y':
                # Simulate the trade
                result = portfolio_manager.execute_trade(symbol, action, 1000)  # $1000 position
                print(f"✅ Trade executed: {result}")
            else:
                print("❌ Trade cancelled.")
        else:
            print("No action recommended.")
            
    except Exception as e:
        print(f"❌ Error simulating trade: {e}")

def view_market_data(config):
    """View market data for symbols."""
    print("\n📊 MARKET DATA")
    print("=" * 30)
    
    symbol = input("Enter symbol to view (AAPL, AMZN, GOOGL, META, NVDA): ").strip().upper()
    
    if symbol not in config["symbols"]:
        print("❌ Invalid symbol.")
        return
    
    try:
        data_path = Path(f"data/{symbol}_sample_data.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
            print(f"\n📈 {symbol} Market Data (Last 10 days):")
            print(df.tail(10).to_string(index=False))
        else:
            print(f"❌ No data found for {symbol}")
            
    except Exception as e:
        print(f"❌ Error loading market data: {e}")

def test_decision_engine(portfolio_manager, config):
    """Test the four-model decision engine."""
    print("\n🎯 TESTING DECISION ENGINE")
    print("=" * 30)
    
    symbol = input("Enter symbol to test (AAPL, AMZN, GOOGL, META, NVDA): ").strip().upper()
    
    if symbol not in config["symbols"]:
        print("❌ Invalid symbol.")
        return
    
    try:
        # Test the decision engine
        recommendation = portfolio_manager.get_ai_recommendation(symbol)
        
        print(f"\n🧠 Four-Model Decision Engine Results for {symbol}:")
        print(f"Final Action: {recommendation.get('action', 'UNKNOWN')}")
        print(f"Confidence: {recommendation.get('confidence', 0.0):.1%}")
        print(f"Reasoning: {recommendation.get('reasoning', 'No reasoning available')}")
        
        # Show model breakdown if available
        model_outputs = recommendation.get('model_outputs', {})
        if model_outputs:
            print("\n📊 Model Breakdown:")
            for model_name, output in model_outputs.items():
                print(f"  {model_name}: {output.get('signal', 0.0):.3f} (confidence: {output.get('confidence', 0.0):.1%})")
        
    except Exception as e:
        print(f"❌ Error testing decision engine: {e}")

def run_fallback_mode():
    """Run fallback mode when main components are not available."""
    print("\n🔄 FALLBACK MODE")
    print("=" * 30)
    print("Running basic portfolio simulation...")
    
    # Simple portfolio simulation
    portfolio_value = 100000.0
    symbols = ["AAPL", "AMZN", "GOOGL", "META", "NVDA"]
    
    print(f"\n💰 Portfolio Value: ${portfolio_value:,.2f}")
    print("📊 Available Symbols:", ", ".join(symbols))
    
    while True:
        print("\n📋 FALLBACK PORTFOLIO MANAGER")
        print("1. View Portfolio")
        print("2. Get Mock Recommendations")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            print(f"\n💰 Portfolio Value: ${portfolio_value:,.2f}")
            print("📊 No positions (cash only)")
        elif choice == '2':
            print("\n🤖 MOCK RECOMMENDATIONS:")
            for symbol in symbols:
                print(f"{symbol}: BUY - Confidence: 75% - Risk: Medium")
        elif choice == '3':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option.")

def run_demo_mode():
    """Run demonstration mode."""
    print("🎮 DEMO MODE")
    print("=" * 30)
    print("Demonstrating QuantAI Portfolio Manager capabilities...")
    
    print("\n🚀 SYSTEM CAPABILITIES:")
    print("✅ Four-Model Decision Engine")
    print("✅ Trained ML Ensemble Models")
    print("✅ Risk Management (Kelly Criterion, VaR)")
    print("✅ Portfolio Tracking & Analytics")
    print("✅ Real-time Market Data Integration")
    print("✅ Reinforcement Learning")
    
    print("\n🤖 SAMPLE RECOMMENDATIONS:")
    symbols = ["AAPL", "AMZN", "GOOGL", "META", "NVDA"]
    for i, symbol in enumerate(symbols, 1):
        print(f"{i}. {symbol}: BUY - Confidence: 82% - Risk: Low")
        print(f"   Reasoning: Strong technical signals, positive sentiment")
    
    print("\n📊 SAMPLE PERFORMANCE:")
    print("💰 Portfolio Value: $125,000")
    print("📈 Total Return: 25.0%")
    print("🎯 Sharpe Ratio: 2.1")
    print("⚠️  Max Drawdown: -5.2%")
    print("🏆 Win Rate: 72%")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='QuantAI Portfolio Manager - Local Environment')
    parser.add_argument('--setup', action='store_true', help='Set up local environment')
    parser.add_argument('--run', action='store_true', help='Run portfolio manager')
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_local_environment()
    elif args.run:
        run_portfolio_manager()
    elif args.demo:
        run_demo_mode()
    else:
        print("🚀 QuantAI Portfolio Manager - Local Environment")
        print("=" * 50)
        print("Usage:")
        print("  python run_portfolio.py --setup    # First time setup")
        print("  python run_portfolio.py --run      # Run portfolio manager")
        print("  python run_portfolio.py --demo     # Run demo mode")
        print()
        print("For first time setup, run: python run_portfolio.py --setup")

if __name__ == "__main__":
    main()
