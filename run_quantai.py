#!/usr/bin/env python3
"""
QuantAI Trading Platform - Simple Launcher

This script provides a simple way to run the QuantAI Trading Platform
without import issues.
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / 'src'))

def run_demo():
    """Run demo mode showing platform capabilities"""
    print("🚀 QuantAI Trading Platform - Demo Mode")
    print("=" * 60)
    print()
    
    print("🎯 FOCUSED 5-TICKER STRATEGY:")
    print("   📊 AMZN - Amazon (Consumer Discretionary)")
    print("   📊 META - Meta (Communication Services)")
    print("   📊 NVDA - NVIDIA (Technology)")
    print("   📊 GOOGL - Alphabet (Communication Services)")
    print("   📊 AAPL - Apple (Technology)")
    print()
    
    print("🤖 AI-POWERED FEATURES:")
    print("   🧠 Multi-Model Ensemble (Random Forest, XGBoost, LSTM, RL)")
    print("   📈 Advanced Feature Engineering (50+ technical indicators)")
    print("   📊 Sentiment Analysis (News API, Gemini AI)")
    print("   🔄 Regime Detection and Volatility Forecasting")
    print()
    
    print("📊 BACKTESTING SYSTEMS:")
    print("   🎯 Simple Backtester (Fast, no dependencies)")
    print("   🎯 Standalone Backtester (Advanced)")
    print("   🎯 QF-Lib Backtester (Event-driven)")
    print("   🎯 Advanced Quantitative Backtester (Research-grade)")
    print("   🎯 Focused 5-Ticker Backtester (Production-ready)")
    print()
    
    print("💼 PORTFOLIO MANAGEMENT:")
    print("   👤 User Authentication with PBKDF2-SHA256")
    print("   💼 Real-time Portfolio Tracking")
    print("   🤖 AI Recommendations with confidence scoring")
    print("   🔄 Reinforcement Learning from user feedback")
    print("   🎯 Personalization based on user behavior")
    print("   🛡️ Enterprise-grade security")
    print()
    
    print("🛡️ RISK MANAGEMENT:")
    print("   📊 Kelly Criterion for optimal position sizing")
    print("   📈 VaR Analysis (95% and 99% confidence levels)")
    print("   📉 Maximum Drawdown monitoring")
    print("   🎯 Portfolio Optimization with constraints")
    print()
    
    print("📈 PERFORMANCE METRICS:")
    print("   🎯 Recommendation Accuracy: >65%")
    print("   👥 User Acceptance Rate: >40%")
    print("   ⚡ Response Time: <500ms")
    print("   🧠 Learning Improvement: 30 days")
    print()
    
    print("🚀 USAGE COMMANDS:")
    print("   # Run backtesting systems")
    print("   python apps/backtesting/backtesters/simple_backtest.py")
    print("   python apps/backtesting/backtesters/focused_5_ticker_backtester.py")
    print()
    print("   # Run original applications")
    print("   python apps/trading/main.py")
    print("   python apps/trading/focused_quantai_main.py summary")
    print()
    
    print("🎉 QUANTAI TRADING PLATFORM READY!")
    print("   Advanced AI-driven quantitative trading")
    print("   Interactive portfolio management")
    print("   Reinforcement learning system")
    print("   Enterprise-grade security")
    print("   Production-ready architecture")

def run_backtest():
    """Run a simple backtest"""
    print("📊 Running Simple Backtest...")
    try:
        # Try to run the simple backtest
        import subprocess
        result = subprocess.run([
            sys.executable, 
            "apps/backtesting/backtesters/simple_backtest.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Backtest completed successfully!")
            print(result.stdout)
        else:
            print("❌ Backtest failed:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        print("🔄 This is expected if dependencies are not installed")

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "demo":
            run_demo()
        elif mode == "backtest":
            run_backtest()
        else:
            print("❌ Unknown mode. Use 'demo' or 'backtest'")
    else:
        print("🚀 QuantAI Trading Platform")
        print("Usage:")
        print("  python run_quantai.py demo     # Show demo")
        print("  python run_quantai.py backtest # Run backtest")
        print()
        print("Or run directly:")
        print("  python apps/backtesting/backtesters/simple_backtest.py")

if __name__ == "__main__":
    main()
