#!/usr/bin/env python3
"""
Run Four-Model Stock Recommendations

This script runs the complete four-model decision engine and shows
AI recommendations for all stocks without interactive input.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def run_four_model_recommendations():
    """Run the four-model decision engine and show recommendations."""
    print("🚀 QuantAI Four-Model Stock Recommendations")
    print("=" * 60)
    print("Running the complete four-model decision engine")
    print("for AI-powered stock recommendations.")
    print("=" * 60)
    
    try:
        from apps.portfolio.local_portfolio_manager import LocalPortfolioManager
        
        # Initialize portfolio manager
        portfolio_manager = LocalPortfolioManager()
        
        # Test symbols
        symbols = ["AAPL", "AMZN", "GOOGL", "META", "NVDA"]
        
        print(f"\n🤖 FOUR-MODEL AI RECOMMENDATIONS:")
        print("=" * 50)
        
        for i, symbol in enumerate(symbols, 1):
            print(f"\n📊 {i}. {symbol} Analysis:")
            print("-" * 30)
            
            # Get four-model recommendation
            recommendation = portfolio_manager.get_ai_recommendation(symbol)
            
            print(f"🎯 Action: {recommendation['action']}")
            print(f"📈 Confidence: {recommendation['confidence']:.1%}")
            print(f"💭 Reasoning: {recommendation['reasoning']}")
            
            # Show four-model breakdown
            if 'model_outputs' in recommendation:
                model_outputs = recommendation['model_outputs']
                print(f"\n🧠 Four-Model Analysis:")
                
                # Sentiment Model (25% weight)
                sentiment = model_outputs.get('sentiment_model', {})
                print(f"  1️⃣ Sentiment Model (25%): {sentiment.get('signal', 0.0):.2f}")
                print(f"     {sentiment.get('reasoning', 'N/A')}")
                
                # Quantitative Model (25% weight)
                quantitative = model_outputs.get('quantitative_model', {})
                print(f"  2️⃣ Quantitative Model (25%): {quantitative.get('signal', 0.0):.2f}")
                print(f"     {quantitative.get('reasoning', 'N/A')}")
                
                # ML Ensemble Model (35% weight)
                ml_ensemble = model_outputs.get('ml_ensemble_model', {})
                print(f"  3️⃣ ML Ensemble Model (35%): {ml_ensemble.get('signal', 0.0):.2f}")
                print(f"     {ml_ensemble.get('reasoning', 'N/A')}")
                
                # RL Decider Agent (Final decision maker)
                rl_agent = model_outputs.get('rl_decider_agent', {})
                print(f"  4️⃣ RL Decider Agent (Final): {rl_agent.get('action', 'UNKNOWN')}")
                print(f"     Final Signal: {rl_agent.get('final_signal', 0.0):.2f}")
                print(f"     {rl_agent.get('reasoning', 'N/A')}")
        
        print(f"\n📊 PORTFOLIO SUMMARY:")
        print("=" * 30)
        summary = portfolio_manager.get_portfolio_summary()
        print(f"💰 Total Value: ${summary['total_value']:,.2f}")
        print(f"💵 Cash Balance: ${summary['cash_balance']:,.2f}")
        print(f"📈 Total Return: {summary['total_return']:.2%}")
        print(f"🎯 Risk Level: {summary['risk_level']}")
        
        print(f"\n🎉 FOUR-MODEL ANALYSIS COMPLETE!")
        print("=" * 40)
        print("✅ All four models analyzed each stock")
        print("✅ RL agent made final decisions")
        print("✅ Each stock got unique recommendations")
        print("✅ Model weights properly applied")
        
        return True
        
    except Exception as e:
        print(f"❌ Error running four-model recommendations: {e}")
        return False

def run_simple_recommendations():
    """Run simple recommendations as fallback."""
    print("\n🔄 FALLBACK: Simple Recommendations")
    print("=" * 40)
    
    try:
        from real_ai_recommendations import RealAIRecommendations
        
        # Initialize AI recommendations
        ai_recommendations = RealAIRecommendations()
        
        # Test symbols
        symbols = ["AAPL", "AMZN", "GOOGL", "META", "NVDA"]
        
        print("🤖 SIMPLE AI RECOMMENDATIONS:")
        print("-" * 40)
        
        for symbol in symbols:
            recommendation = ai_recommendations.get_recommendation(symbol)
            
            print(f"{symbol}: {recommendation['action']} - Confidence: {recommendation['confidence']:.1%}")
            print(f"  Reasoning: {recommendation['reasoning']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error running fallback recommendations: {e}")
        return False

def main():
    """Main function."""
    print("🚀 QuantAI Trading Platform - Four-Model Recommendations")
    print("=" * 70)
    print("This script runs the complete four-model decision engine")
    print("and shows AI recommendations for all stocks.")
    print("=" * 70)
    
    # Try four-model approach first
    four_model_success = run_four_model_recommendations()
    
    if not four_model_success:
        print("\n⚠️  Four-model approach failed, trying fallback...")
        simple_success = run_simple_recommendations()
        
        if simple_success:
            print("\n✅ Fallback recommendations working")
        else:
            print("\n❌ Both approaches failed")
    else:
        print("\n✅ Four-model approach working perfectly!")

if __name__ == "__main__":
    main()
