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
    print("   🎯 QF-Lib Backtester (Event-driven) - Requires qf_env")
    print("   🎯 Advanced Quantitative Backtester (Research-grade)")
    print("   🎯 Focused 5-Ticker Backtester (Production-ready)")
    print()
    
    print("🐍 QF-LIB ENVIRONMENT:")
    print("   conda activate qf_env")
    print("   python scripts/run_qf_backtest.py")
    print("   python scripts/test_qf_env.py")
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

def run_qf_backtest():
    """Run QF-Lib backtester in qf_env environment"""
    print("🎯 QuantAI Trading Platform - QF-Lib Backtester")
    print("=" * 60)
    print()
    
    print("🐍 QF-LIB ENVIRONMENT REQUIRED:")
    print("   conda activate qf_env")
    print()
    
    print("🚀 RUNNING QF-LIB BACKTESTER:")
    print("   python scripts/run_qf_backtest.py")
    print()
    
    # Try to run the QF-Lib backtester
    try:
        import subprocess
        result = subprocess.run([
            'python', 'scripts/run_qf_backtest.py'
        ], check=True)
        print("✅ QF-Lib backtester completed successfully!")
    except subprocess.CalledProcessError:
        print("❌ QF-Lib backtester failed")
        print("Please ensure qf_env is activated: conda activate qf_env")
    except FileNotFoundError:
        print("❌ QF-Lib runner script not found")
        print("Please check the scripts/run_qf_backtest.py file")

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "demo":
            run_demo()
        elif mode == "backtest":
            run_backtest()
        elif mode == "qf":
            run_qf_backtest()
        else:
            print("❌ Unknown mode. Use 'demo', 'backtest', or 'qf'")
    else:
        print("🚀 QuantAI Trading Platform")
        print("Usage:")
        print("  python run_quantai.py demo     # Show demo")
        print("  python run_quantai.py backtest # Run backtest")
        print("  python run_quantai.py qf       # Run QF-Lib backtester")
        print()
        print("Or run directly:")
        print("  python apps/backtesting/backtesters/simple_backtest.py")
        print("  conda activate qf_env && python scripts/run_qf_backtest.py")

if __name__ == "__main__":
    main()
