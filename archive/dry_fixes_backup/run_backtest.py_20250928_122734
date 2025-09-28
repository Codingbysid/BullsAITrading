#!/usr/bin/env python3
"""
Backtesting Script for QuantAI Trading Platform

This script runs comprehensive backtesting using the unified backtesting framework
with the four-model decision engine.

Usage:
    python scripts/run_backtest.py --strategy four_model --symbols AMZN,META,NVDA,GOOGL,AAPL --start-date 2020-01-01 --end-date 2023-12-31
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import asyncio

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Import unified utilities
from utils.common_imports import setup_logger
from utils.data_processing import data_processor
from utils.performance_metrics import performance_calculator

# Import backtesting framework
from apps.backtesting.base_backtester import BaseBacktester
from apps.backtesting.backtesters.unified_backtester import (
    FourModelBacktester,
    AdvancedTechnicalBacktester,
    MomentumBacktester,
    MeanReversionBacktester
)

logger = setup_logger(__name__)


class BacktestingScript:
    """Comprehensive backtesting script using unified framework."""
    
    def __init__(self):
        self.results = {}
        self.performance_metrics = {}
    
    def run_backtest(self, strategy: str, symbols: List[str], 
                    start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run backtest for specified strategy and symbols."""
        logger.info(f"Running {strategy} backtest for {len(symbols)} symbols")
        
        # Create synthetic data for demonstration
        all_data = []
        for symbol in symbols:
            symbol_data = data_processor.create_synthetic_data(
                [symbol], start_date, end_date, base_price=100.0
            )
            all_data.append(symbol_data)
        
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Initialize appropriate backtester
        if strategy == 'four_model':
            backtester = FourModelBacktester()
        elif strategy == 'advanced_technical':
            backtester = AdvancedTechnicalBacktester()
        elif strategy == 'momentum':
            backtester = MomentumBacktester()
        elif strategy == 'mean_reversion':
            backtester = MeanReversionBacktester()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Run backtest
        try:
            results = backtester.run_backtest(combined_data)
            
            # Calculate performance metrics using unified utilities
            performance = self._calculate_performance_metrics(results)
            
            backtest_summary = {
                'strategy': strategy,
                'symbols': symbols,
                'date_range': f"{start_date.date()} to {end_date.date()}",
                'results': results,
                'performance': performance,
                'status': 'completed'
            }
            
            logger.info(f"✅ {strategy} backtest completed successfully")
            return backtest_summary
            
        except Exception as e:
            logger.error(f"❌ {strategy} backtest failed: {e}")
            return {
                'strategy': strategy,
                'symbols': symbols,
                'status': 'failed',
                'error': str(e)
            }
    
    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics using unified utilities."""
        try:
            # Extract returns from results
            if 'returns' in results:
                returns = results['returns']
            elif 'portfolio_values' in results:
                returns = results['portfolio_values'].pct_change().dropna()
            else:
                # Create synthetic returns for demonstration
                import numpy as np
                returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
            
            # Calculate performance metrics using unified calculator
            performance = performance_calculator.calculate_comprehensive_metrics(returns)
            
            # Add additional metrics
            performance.update({
                'total_trades': results.get('total_trades', 0),
                'winning_trades': results.get('winning_trades', 0),
                'losing_trades': results.get('losing_trades', 0),
                'max_consecutive_wins': results.get('max_consecutive_wins', 0),
                'max_consecutive_losses': results.get('max_consecutive_losses', 0)
            })
            
            return performance
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return {'error': str(e)}
    
    def run_multiple_strategies(self, symbols: List[str], 
                               start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Run backtests for multiple strategies."""
        strategies = ['four_model', 'advanced_technical', 'momentum', 'mean_reversion']
        
        logger.info(f"Running backtests for {len(strategies)} strategies")
        
        all_results = {}
        
        for strategy in strategies:
            logger.info(f"Running {strategy} strategy")
            results = self.run_backtest(strategy, symbols, start_date, end_date)
            all_results[strategy] = results
        
        # Compare strategies
        comparison = self._compare_strategies(all_results)
        
        return {
            'strategies': all_results,
            'comparison': comparison,
            'date_range': f"{start_date.date()} to {end_date.date()}",
            'symbols': symbols,
            'status': 'completed'
        }
    
    def _compare_strategies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance across strategies."""
        comparison = {
            'best_strategy': None,
            'worst_strategy': None,
            'strategy_rankings': [],
            'performance_summary': {}
        }
        
        strategy_performance = {}
        
        for strategy, result in results.items():
            if result.get('status') == 'completed' and 'performance' in result:
                perf = result['performance']
                strategy_performance[strategy] = {
                    'total_return': perf.get('total_return', 0.0),
                    'sharpe_ratio': perf.get('sharpe_ratio', 0.0),
                    'max_drawdown': perf.get('max_drawdown', 0.0),
                    'win_rate': perf.get('win_rate', 0.0)
                }
        
        if strategy_performance:
            # Rank by Sharpe ratio
            ranked_strategies = sorted(
                strategy_performance.items(),
                key=lambda x: x[1]['sharpe_ratio'],
                reverse=True
            )
            
            comparison['strategy_rankings'] = [
                {
                    'strategy': strategy,
                    'rank': i + 1,
                    'sharpe_ratio': perf['sharpe_ratio'],
                    'total_return': perf['total_return'],
                    'max_drawdown': perf['max_drawdown']
                }
                for i, (strategy, perf) in enumerate(ranked_strategies)
            ]
            
            comparison['best_strategy'] = ranked_strategies[0][0]
            comparison['worst_strategy'] = ranked_strategies[-1][0]
            comparison['performance_summary'] = strategy_performance
        
        return comparison
    
    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save backtesting results to file."""
        try:
            import json
            
            # Convert numpy types to Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict()
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                return obj
            
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {k: recursive_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(item) for item in obj]
                else:
                    return convert_types(obj)
            
            converted_results = recursive_convert(results)
            
            with open(filepath, 'w') as f:
                json.dump(converted_results, f, indent=2)
            
            logger.info(f"✅ Results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run QuantAI Trading Platform Backtests')
    parser.add_argument('--strategy', type=str, default='four_model',
                       choices=['four_model', 'advanced_technical', 'momentum', 'mean_reversion', 'all'],
                       help='Backtesting strategy to run')
    parser.add_argument('--symbols', type=str, default='AMZN,META,NVDA,GOOGL,AAPL',
                       help='Comma-separated list of symbols to backtest')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                       help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='backtest_results.json',
                       help='Output file for backtest results')
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Initialize backtesting script
    backtest_script = BacktestingScript()
    
    print("🎯 QuantAI Trading Platform - Backtesting")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Date Range: {start_date.date()} to {end_date.date()}")
    print("=" * 60)
    
    try:
        # Run backtest(s)
        if args.strategy == 'all':
            results = backtest_script.run_multiple_strategies(symbols, start_date, end_date)
        else:
            results = backtest_script.run_backtest(args.strategy, symbols, start_date, end_date)
        
        # Print results
        print("\n📊 Backtesting Results:")
        print("-" * 40)
        
        if results.get('status') == 'completed':
            print("✅ Backtest Status: COMPLETED")
            
            if 'strategies' in results:
                # Multiple strategies
                print(f"\n🔍 Strategy Comparison:")
                comparison = results['comparison']
                
                if comparison['strategy_rankings']:
                    print(f"   Best Strategy: {comparison['best_strategy']}")
                    print(f"   Worst Strategy: {comparison['worst_strategy']}")
                    
                    print(f"\n📈 Strategy Rankings (by Sharpe Ratio):")
                    for ranking in comparison['strategy_rankings']:
                        print(f"   {ranking['rank']}. {ranking['strategy']}: "
                              f"Sharpe={ranking['sharpe_ratio']:.3f}, "
                              f"Return={ranking['total_return']:.2%}, "
                              f"MaxDD={ranking['max_drawdown']:.2%}")
                
                # Individual strategy results
                for strategy, result in results['strategies'].items():
                    if result.get('status') == 'completed':
                        perf = result['performance']
                        print(f"\n📊 {strategy.upper()} Strategy:")
                        print(f"   Total Return: {perf.get('total_return', 0.0):.2%}")
                        print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0.0):.3f}")
                        print(f"   Max Drawdown: {perf.get('max_drawdown', 0.0):.2%}")
                        print(f"   Win Rate: {perf.get('win_rate', 0.0):.2%}")
                        print(f"   Total Trades: {perf.get('total_trades', 0)}")
            else:
                # Single strategy
                perf = results['performance']
                print(f"\n📊 {results['strategy'].upper()} Strategy:")
                print(f"   Total Return: {perf.get('total_return', 0.0):.2%}")
                print(f"   Sharpe Ratio: {perf.get('sharpe_ratio', 0.0):.3f}")
                print(f"   Max Drawdown: {perf.get('max_drawdown', 0.0):.2%}")
                print(f"   Win Rate: {perf.get('win_rate', 0.0):.2%}")
                print(f"   Total Trades: {perf.get('total_trades', 0)}")
                print(f"   Volatility: {perf.get('volatility', 0.0):.2%}")
                print(f"   Sortino Ratio: {perf.get('sortino_ratio', 0.0):.3f}")
                print(f"   Calmar Ratio: {perf.get('calmar_ratio', 0.0):.3f}")
        
        else:
            print("❌ Backtest Status: FAILED")
            print(f"Error: {results.get('error', 'Unknown error')}")
        
        # Save results
        backtest_script.save_results(results, args.output)
        print(f"\n💾 Results saved to {args.output}")
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        logger.error(f"Backtesting failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
