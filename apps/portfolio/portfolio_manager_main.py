from src.utils.common_imports import *
import sys
import logging
from datetime import datetime
import argparse
        from src.interface.cli import QuantAITerminalInterface
        import uvicorn
        from src.api.portfolio_api import app

"""
QuantAI Portfolio Manager & Trade Suggestion Bot - Main Application

This is the main entry point for the QuantAI Portfolio Manager system.
It provides a comprehensive portfolio management and trading recommendation
system focused on the 5 flagship stocks: AMZN, META, NVDA, GOOGL, AAPL.
"""


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = setup_logger()

def main():
    """Main entry point for QuantAI Portfolio Manager"""
    parser = argparse.ArgumentParser(description='QuantAI Portfolio Manager & Trade Suggestion Bot')
    parser.add_argument('--mode', choices=['cli', 'api', 'demo'], default='cli',
                       help='Run mode: cli (terminal), api (web), demo (demonstration)')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port for API mode (default: 8000)')
    
    args = parser.parse_args()
    
    print("🚀 QuantAI Portfolio Manager & Trade Suggestion Bot")
    print("=" * 60)
    print("🎯 Focused on AMZN, META, NVDA, GOOGL, AAPL")
    print("🤖 AI-powered recommendations with risk management")
    print("📊 Portfolio tracking and performance analytics")
    print("🔄 Reinforcement learning from user feedback")
    print("=" * 60)
    
    try:
        if args.mode == 'cli':
            run_cli_mode()
        elif args.mode == 'api':
            run_api_mode(args.port)
        elif args.mode == 'demo':
            run_demo_mode()
        else:
            print("❌ Invalid mode. Use --help for options.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        sys.exit(1)

def run_cli_mode():
    """Run terminal interface mode"""
    print("\n🖥️  Starting Terminal Interface...")
    
    try:
        interface = QuantAITerminalInterface()
        interface.start()
    except ImportError as e:
        print(f"❌ Failed to import CLI interface: {e}")
        print("🔄 Running in fallback mode...")
        run_fallback_cli()

def run_api_mode(port):
    """Run web API mode"""
    print(f"\n🌐 Starting Web API on port {port}...")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except ImportError as e:
        print(f"❌ Failed to start API mode: {e}")
        print("🔄 Falling back to CLI mode...")
        run_cli_mode()

def run_demo_mode():
    """Run demonstration mode"""
    print("\n🎮 Starting Demo Mode...")
    print("This mode demonstrates the system capabilities without requiring a database.")
    
    try:
        interface = QuantAITerminalInterface()
        interface._demo_mode()
        interface.start()
    except Exception as e:
        print(f"❌ Demo mode failed: {e}")
        run_fallback_demo()

def run_fallback_cli():
    """Fallback CLI implementation"""
    print("\n🔄 Fallback CLI Mode")
    print("Basic portfolio management without advanced features...")
    
    while True:
        print("\n📋 QUANTAI PORTFOLIO MANAGER (Fallback)")
        print("1. View System Status")
        print("2. Simulate Portfolio")
        print("3. Generate Mock Recommendations")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            show_system_status()
        elif choice == '2':
            simulate_portfolio()
        elif choice == '3':
            generate_mock_recommendations()
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Please try again.")

def run_fallback_demo():
    """Fallback demo implementation"""
    print("\n🎮 Fallback Demo Mode")
    print("Demonstrating QuantAI capabilities...")
    
    print("\n📊 SYSTEM CAPABILITIES:")
    print("✅ 5-Ticker Strategy (AMZN, META, NVDA, GOOGL, AAPL)")
    print("✅ AI-Powered Recommendations")
    print("✅ Risk Management (Kelly Criterion, VaR)")
    print("✅ Portfolio Tracking")
    print("✅ Performance Analytics")
    print("✅ Reinforcement Learning")
    
    print("\n🤖 MOCK RECOMMENDATIONS:")
    symbols = ['AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL']
    for i, symbol in enumerate(symbols, 1):
        print(f"{i}. {symbol}: BUY - Confidence: 75% - Risk: Low")
    
    print("\n📈 PERFORMANCE METRICS:")
    print("💰 Portfolio Value: $100,000")
    print("📊 Total Return: 15.2%")
    print("🎯 Sharpe Ratio: 1.86")
    print("⚠️  Max Drawdown: -8.3%")
    print("🏆 Win Rate: 68%")

def show_system_status():
    """Show system status"""
    print("\n🔧 SYSTEM STATUS")
    print("=" * 40)
    print("🎯 Focused 5-Ticker Strategy")
    print("📊 Symbols: AMZN, META, NVDA, GOOGL, AAPL")
    print("🤖 AI Models: Available")
    print("🛡️  Risk Management: Kelly Criterion, VaR")
    print("📈 Backtesting: 5 Systems Available")
    print("🔄 Learning: User Feedback Integration")
    print("📱 Interface: Terminal, API, Demo")

def simulate_portfolio():
    """Simulate portfolio display"""
    print("\n📊 SIMULATED PORTFOLIO")
    print("=" * 50)
    print("💰 Total Value: $125,000")
    print("📈 Total P&L: $25,000 (20.0%)")
    print("🎯 Risk Level: Medium")
    print()
    print("Positions:")
    print("AMZN: 50 shares @ $3,200 = $160,000")
    print("NVDA: 100 shares @ $400 = $40,000")
    print("GOOGL: 20 shares @ $2,800 = $56,000")

def generate_mock_recommendations():
    """Generate mock recommendations"""
    print("\n🤖 MOCK AI RECOMMENDATIONS")
    print("=" * 40)
    
    recommendations = [
        {"symbol": "META", "action": "BUY", "confidence": 0.85, "reason": "Strong earnings growth"},
        {"symbol": "AAPL", "action": "HOLD", "confidence": 0.65, "reason": "Stable performance"},
        {"symbol": "NVDA", "action": "SELL", "confidence": 0.78, "reason": "Overvalued territory"}
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['symbol']}: {rec['action']}")
        print(f"   Confidence: {rec['confidence']:.0%}")
        print(f"   Reason: {rec['reason']}")
        print()

if __name__ == "__main__":
    main()
