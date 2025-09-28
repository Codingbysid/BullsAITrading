#!/usr/bin/env python3
"""
Adjust Risk Factors for QuantAI Trading Platform

This script allows you to modify risk factors and see how they affect
the four-model decision engine recommendations.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load current configuration."""
    config_path = Path("config/local/portfolio_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def save_config(config: Dict[str, Any]):
    """Save configuration."""
    config_path = Path("config/local/portfolio_config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def show_current_risk_factors(config: Dict[str, Any]):
    """Show current risk factors."""
    print("📊 CURRENT RISK FACTORS:")
    print("=" * 30)
    
    risk_tolerance = config.get('risk_tolerance', 0.05)
    print(f"Risk Tolerance: {risk_tolerance:.1%}")
    
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

def adjust_risk_factors():
    """Interactive risk factor adjustment."""
    config = load_config()
    
    print("🚀 QuantAI Risk Factor Adjustment")
    print("=" * 40)
    
    show_current_risk_factors(config)
    
    print("\n🔧 ADJUSTMENT OPTIONS:")
    print("1. Conservative (Low Risk)")
    print("2. Moderate (Medium Risk)")
    print("3. Aggressive (High Risk)")
    print("4. Custom Settings")
    print("5. Reset to Default")
    print("6. Exit")
    
    try:
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == "1":
            # Conservative settings
            config['risk_tolerance'] = 0.02
            config['risk_factors'] = {
                "portfolio_risk": 0.02,
                "max_drawdown": 0.08,
                "volatility_threshold": 0.02,
                "correlation_limit": 0.5,
                "var_confidence": 0.99,
                "kelly_criterion": True,
                "position_sizing": "kelly"
            }
            config['trading_settings'] = {
                "max_position_size": 0.15,
                "max_portfolio_risk": 0.08,
                "rebalance_frequency": "weekly",
                "stop_loss": 0.05,
                "take_profit": 0.15
            }
            print("✅ Set to Conservative (Low Risk)")
            
        elif choice == "2":
            # Moderate settings
            config['risk_tolerance'] = 0.05
            config['risk_factors'] = {
                "portfolio_risk": 0.05,
                "max_drawdown": 0.15,
                "volatility_threshold": 0.03,
                "correlation_limit": 0.7,
                "var_confidence": 0.95,
                "kelly_criterion": True,
                "position_sizing": "kelly"
            }
            config['trading_settings'] = {
                "max_position_size": 0.3,
                "max_portfolio_risk": 0.15,
                "rebalance_frequency": "weekly",
                "stop_loss": 0.1,
                "take_profit": 0.2
            }
            print("✅ Set to Moderate (Medium Risk)")
            
        elif choice == "3":
            # Aggressive settings
            config['risk_tolerance'] = 0.1
            config['risk_factors'] = {
                "portfolio_risk": 0.1,
                "max_drawdown": 0.25,
                "volatility_threshold": 0.05,
                "correlation_limit": 0.8,
                "var_confidence": 0.90,
                "kelly_criterion": True,
                "position_sizing": "kelly"
            }
            config['trading_settings'] = {
                "max_position_size": 0.5,
                "max_portfolio_risk": 0.25,
                "rebalance_frequency": "weekly",
                "stop_loss": 0.15,
                "take_profit": 0.3
            }
            print("✅ Set to Aggressive (High Risk)")
            
        elif choice == "4":
            # Custom settings
            print("\n🔧 CUSTOM RISK SETTINGS:")
            print("Enter values as percentages (e.g., 5 for 5%)")
            
            try:
                risk_tolerance = float(input("Risk Tolerance (%): ")) / 100
                max_drawdown = float(input("Max Drawdown (%): ")) / 100
                volatility_threshold = float(input("Volatility Threshold (%): ")) / 100
                max_position_size = float(input("Max Position Size (%): ")) / 100
                
                config['risk_tolerance'] = risk_tolerance
                config['risk_factors'] = {
                    "portfolio_risk": risk_tolerance,
                    "max_drawdown": max_drawdown,
                    "volatility_threshold": volatility_threshold,
                    "correlation_limit": 0.7,
                    "var_confidence": 0.95,
                    "kelly_criterion": True,
                    "position_sizing": "kelly"
                }
                config['trading_settings']['max_position_size'] = max_position_size
                
                print("✅ Custom settings applied")
                
            except ValueError:
                print("❌ Invalid input, using default values")
                
        elif choice == "5":
            # Reset to default
            config['risk_tolerance'] = 0.05
            config['risk_factors'] = {
                "portfolio_risk": 0.05,
                "max_drawdown": 0.15,
                "volatility_threshold": 0.03,
                "correlation_limit": 0.7,
                "var_confidence": 0.95,
                "kelly_criterion": True,
                "position_sizing": "kelly"
            }
            config['trading_settings'] = {
                "max_position_size": 0.3,
                "max_portfolio_risk": 0.15,
                "rebalance_frequency": "weekly",
                "stop_loss": 0.1,
                "take_profit": 0.2
            }
            print("✅ Reset to Default Settings")
            
        elif choice == "6":
            print("👋 Goodbye!")
            return
            
        else:
            print("❌ Invalid choice")
            return
        
        # Save configuration
        save_config(config)
        print(f"\n💾 Configuration saved to config/local/portfolio_config.json")
        
        # Show updated settings
        print(f"\n📊 UPDATED RISK FACTORS:")
        show_current_risk_factors(config)
        
        print(f"\n🚀 To test the new settings, run:")
        print(f"python run_with_risk_factors.py")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function."""
    print("🚀 QuantAI Risk Factor Adjustment Tool")
    print("=" * 50)
    print("This tool allows you to adjust risk factors")
    print("and see how they affect AI recommendations.")
    print("=" * 50)
    
    adjust_risk_factors()

if __name__ == "__main__":
    main()
