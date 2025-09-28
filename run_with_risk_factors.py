from src.utils.common_imports import *
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
        from apps.portfolio.local_portfolio_manager import LocalPortfolioManager
        import json

#!/usr/bin/env python3
"""
Run QuantAI with Enhanced Risk Factors

This script demonstrates the four-model decision engine with comprehensive
risk factor analysis including volatility, drawdown, VaR, and position sizing.
"""


# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def run_with_risk_factors():
    """Run the four-model decision engine with enhanced risk factors."""
    print("🚀 QuantAI Four-Model Decision Engine with Risk Factors")
    print("=" * 70)
    print("Enhanced risk analysis including:")
    print("• Volatility adjustments")
    print("• Maximum drawdown analysis")
    print("• Value at Risk (VaR) calculations")
    print("• Kelly Criterion position sizing")
    print("• Correlation risk assessment")
    print("=" * 70)
    
    try:
        
        # Initialize portfolio manager
        portfolio_manager = LocalPortfolioManager()
        
        # Test symbols
        symbols = ["AAPL", "AMZN", "GOOGL", "META", "NVDA"]
        
        print(f"\n🤖 ENHANCED AI RECOMMENDATIONS WITH RISK FACTORS:")
        print("=" * 60)
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n📊 {i}. {symbol} Analysis:")
            print("-" * 40)
            
            # Get four-model recommendation with risk factors
            recommendation = portfolio_manager.get_ai_recommendation(symbol)
            
            print(f"🎯 Action: {recommendation['action']}")
            print(f"📈 Confidence: {recommendation['confidence']:.1%}")
            print(f"💭 Reasoning: {recommendation['reasoning']}")
            
            # Show risk factor analysis
            if 'model_outputs' in recommendation:
                rl_agent = recommendation['model_outputs'].get('rl_decider_agent', {})
                risk_factors = rl_agent.get('risk_factors', {})
                
                print(f"\n⚠️  RISK FACTOR ANALYSIS:")
                print(f"   📊 Volatility: {risk_factors.get('volatility', 0.0):.2%}")
                print(f"   📉 Max Drawdown: {risk_factors.get('max_drawdown', 0.0):.2%}")
                print(f"   💰 VaR (95%): {risk_factors.get('var_95', 0.0):.2%}")
                print(f"   🔗 Correlation Risk: {risk_factors.get('correlation_risk', 0.0):.2%}")
                print(f"   ⚖️  Risk Score: {risk_factors.get('risk_score', 0.0):.2%}")
                print(f"   📈 Sharpe Ratio: {risk_factors.get('sharpe_ratio', 0.0):.2f}")
                
                # Show position sizing
                position_size = rl_agent.get('position_size', 0.0)
                print(f"   💼 Position Size: {position_size:.1%}")
                
                # Show risk adjustments
                print(f"\n🔧 RISK ADJUSTMENTS:")
                print(f"   Risk Adjustment: {rl_agent.get('risk_adjustment', 1.0):.2f}")
                print(f"   Volatility Adjustment: {rl_agent.get('volatility_adjustment', 1.0):.2f}")
                print(f"   Correlation Adjustment: {rl_agent.get('correlation_adjustment', 1.0):.2f}")
                print(f"   Cash Adjustment: {rl_agent.get('cash_adjustment', 1.0):.2f}")
        
        print(f"\n📊 PORTFOLIO RISK SUMMARY:")
        print("=" * 40)
        summary = portfolio_manager.get_portfolio_summary()
        print(f"💰 Total Value: ${summary['total_value']:,.2f}")
        print(f"💵 Cash Balance: ${summary['cash_balance']:,.2f}")
        print(f"📈 Total Return: {summary['total_return']:.2%}")
        print(f"🎯 Risk Level: {summary['risk_level']}")
        
        print(f"\n🎉 ENHANCED RISK ANALYSIS COMPLETE!")
        print("=" * 50)
        print("✅ All risk factors calculated and applied")
        print("✅ Position sizing based on Kelly Criterion")
        print("✅ Volatility and correlation adjustments")
        print("✅ VaR and drawdown protection")
        print("✅ Risk-adjusted final decisions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running enhanced risk analysis: {e}")
        return False

def show_risk_configuration():
    """Show the current risk configuration."""
    print("\n⚙️  RISK CONFIGURATION:")
    print("=" * 30)
    
    try:
        config_path = Path("config/local/portfolio_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"Risk Tolerance: {config.get('risk_tolerance', 0.05):.1%}")
            
            risk_factors = config.get('risk_factors', {})
            print(f"Portfolio Risk: {risk_factors.get('portfolio_risk', 0.05):.1%}")
            print(f"Max Drawdown: {risk_factors.get('max_drawdown', 0.15):.1%}")
            print(f"Volatility Threshold: {risk_factors.get('volatility_threshold', 0.03):.1%}")
            print(f"Correlation Limit: {risk_factors.get('correlation_limit', 0.7):.1%}")
            print(f"VaR Confidence: {risk_factors.get('var_confidence', 0.95):.1%}")
            print(f"Kelly Criterion: {risk_factors.get('kelly_criterion', True)}")
            print(f"Position Sizing: {risk_factors.get('position_sizing', 'kelly')}")
            
            trading_settings = config.get('trading_settings', {})
            print(f"\nTrading Settings:")
            print(f"Max Position Size: {trading_settings.get('max_position_size', 0.3):.1%}")
            print(f"Max Portfolio Risk: {trading_settings.get('max_portfolio_risk', 0.15):.1%}")
            print(f"Stop Loss: {trading_settings.get('stop_loss', 0.1):.1%}")
            print(f"Take Profit: {trading_settings.get('take_profit', 0.2):.1%}")
            
        else:
            print("❌ Configuration file not found")
            
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")

def main():
    """Main function."""
    print("🚀 QuantAI Trading Platform - Enhanced Risk Analysis")
    print("=" * 70)
    print("This script demonstrates the four-model decision engine")
    print("with comprehensive risk factor analysis.")
    print("=" * 70)
    
    # Show current risk configuration
    show_risk_configuration()
    
    # Run enhanced risk analysis
    success = run_with_risk_factors()
    
    if success:
        print("\n✅ Enhanced risk analysis completed successfully!")
        print("\n📋 Key Risk Features:")
        print("• Volatility-based signal adjustments")
        print("• Maximum drawdown protection")
        print("• Value at Risk (VaR) calculations")
        print("• Kelly Criterion position sizing")
        print("• Correlation risk assessment")
        print("• Risk-adjusted final decisions")
    else:
        print("\n❌ Enhanced risk analysis failed")

if __name__ == "__main__":
    main()
