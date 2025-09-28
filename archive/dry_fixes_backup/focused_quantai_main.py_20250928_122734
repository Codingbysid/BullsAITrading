#!/usr/bin/env python3
"""
Focused QuantAI Trading Platform - Main Application.

This is the main entry point for the focused 5-ticker QuantAI platform
specifically optimized for AMZN, META, NVDA, GOOGL, and AAPL.

Features:
- Advanced data pipeline for 5 tickers
- Sophisticated ML models with ensemble learning
- Risk management with Kelly Criterion
- Walk-forward validation
- Meta labeling for trade filtering
- Comprehensive backtesting
"""

import asyncio
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import focused modules
try:
    from src.data.focused_data_pipeline import FocusedDataPipeline
    from src.models.basic_ml_models import BasicMLModels
    from src.risk.focused_risk_management import FocusedRiskManager
    FOCUSED_MODULES_AVAILABLE = True
except ImportError:
    FOCUSED_MODULES_AVAILABLE = False
    logging.warning("Focused modules not available")

# Import focused backtester
try:
    from backtesters.focused_5_ticker_backtester import Focused5TickerBacktester
    FOCUSED_BACKTESTER_AVAILABLE = True
except ImportError:
    FOCUSED_BACKTESTER_AVAILABLE = False
    logging.warning("Focused backtester not available")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('focused_quantai.log')
    ]
)
logger = logging.getLogger(__name__)


class FocusedQuantAIPlatform:
    """
    Main platform class for focused 5-ticker QuantAI trading.
    
    Integrates all components specifically optimized for
    AMZN, META, NVDA, GOOGL, AAPL.
    """
    
    def __init__(self):
        """Initialize focused QuantAI platform."""
        self.tickers = ['AMZN', 'META', 'NVDA', 'GOOGL', 'AAPL']
        
        # Initialize components
        if FOCUSED_MODULES_AVAILABLE:
            self.data_pipeline = FocusedDataPipeline()
            self.ml_models = BasicMLModels()
            self.risk_manager = FocusedRiskManager()
        else:
            logger.error("Focused modules not available")
            sys.exit(1)
        
        # Platform state
        self.market_data = {}
        self.features = {}
        self.trained_models = {}
        self.performance_metrics = {}
        
    async def run_data_pipeline(self, days_back: int = 365) -> bool:
        """
        Run focused data pipeline.
        
        Args:
            days_back: Number of days to fetch data
            
        Returns:
            Success status
        """
        logger.info("🚀 Starting focused data pipeline for 5 tickers")
        
        try:
            # Fetch historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            logger.info(f"📊 Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Fetch historical data
            historical_data = self.data_pipeline.fetch_historical_data(
                start_date=start_date,
                end_date=end_date,
                include_intraday=False
            )
            
            if not historical_data:
                logger.error("No historical data fetched")
                return False
            
            # Fetch sentiment data
            sentiment_data = self.data_pipeline.fetch_sentiment_data(days_back=30)
            
            # Fetch fundamental data
            fundamental_data = self.data_pipeline.fetch_fundamental_data()
            
            # Create advanced features
            features = self.data_pipeline.create_advanced_features()
            
            if not features:
                logger.error("No features created")
                return False
            
            # Store data
            self.market_data = historical_data
            self.features = features
            
            # Save data
            self.data_pipeline.save_data("focused_data")
            
            logger.info("✅ Focused data pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data pipeline failed: {e}")
            return False
    
    async def run_model_training(self) -> bool:
        """
        Run focused model training.
        
        Returns:
            Success status
        """
        logger.info("🤖 Starting focused model training for 5 tickers")
        
        try:
            if not self.features:
                logger.error("No features available for training")
                return False
            
            # Train all models
            training_results = self.training_pipeline.train_all_models(self.features)
            
            if not training_results:
                logger.error("No models trained")
                return False
            
            # Evaluate models
            evaluation_results = self.training_pipeline.evaluate_models()
            
            # Get best models
            best_models = self.training_pipeline.get_best_models()
            
            # Store results
            self.trained_models = training_results
            
            # Save models
            self.training_pipeline.save_models("focused_models")
            
            # Log results
            logger.info("📊 Training Results Summary:")
            for ticker, results in training_results.items():
                logger.info(f"  {ticker}:")
                for model_name, result in results.items():
                    logger.info(f"    {model_name}: Test Score = {result.test_score:.4f}")
            
            logger.info("✅ Focused model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return False
    
    async def run_risk_analysis(self) -> bool:
        """
        Run focused risk analysis.
        
        Returns:
            Success status
        """
        logger.info("🛡️ Starting focused risk analysis for 5 tickers")
        
        try:
            if not self.market_data:
                logger.error("No market data available for risk analysis")
                return False
            
            # Calculate risk metrics for each ticker
            risk_metrics = {}
            
            for ticker, data in self.market_data.items():
                if ticker not in self.tickers:
                    continue
                
                # Calculate returns
                returns = data['Close'].pct_change().dropna()
                
                if len(returns) == 0:
                    continue
                
                # Calculate risk metrics
                ticker_risk = {
                    'volatility': returns.std() * (252 ** 0.5),
                    'sharpe_ratio': self.risk_manager.calculate_sharpe_ratio(returns),
                    'sortino_ratio': self.risk_manager.calculate_sortino_ratio(returns),
                    'max_drawdown': self.risk_manager.calculate_max_drawdown(returns),
                    'var_95': self.risk_manager.calculate_var(returns, 0.05),
                    'cvar_95': self.risk_manager.calculate_cvar(returns, 0.05)
                }
                
                risk_metrics[ticker] = ticker_risk
                
                # Update ticker profile
                self.risk_manager.update_ticker_profiles(
                    ticker,
                    ticker_risk['volatility'],
                    1.0,  # Beta (simplified)
                    {'momentum': 0.0, 'value': 0.0, 'quality': 0.0}
                )
            
            # Calculate portfolio risk metrics
            if len(risk_metrics) > 1:
                # Calculate correlation matrix
                returns_data = {}
                for ticker, data in self.market_data.items():
                    if ticker in self.tickers:
                        returns_data[ticker] = data['Close'].pct_change().dropna()
                
                # Portfolio VaR
                portfolio_var = self.risk_manager.calculate_portfolio_var({}, returns_data)
                
                # Stress test
                stress_results = self.risk_manager.stress_test_portfolio({}, returns_data)
                
                risk_metrics['portfolio'] = {
                    'portfolio_var': portfolio_var,
                    'stress_test': stress_results
                }
            
            self.performance_metrics = risk_metrics
            
            # Log risk metrics
            logger.info("📊 Risk Analysis Results:")
            for ticker, metrics in risk_metrics.items():
                if ticker != 'portfolio':
                    logger.info(f"  {ticker}:")
                    logger.info(f"    Volatility: {metrics['volatility']:.2%}")
                    logger.info(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    logger.info(f"    Max Drawdown: {metrics['max_drawdown']:.2%}")
                    logger.info(f"    VaR (95%): {metrics['var_95']:.2%}")
            
            logger.info("✅ Focused risk analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Risk analysis failed: {e}")
            return False
    
    async def run_backtesting(self) -> bool:
        """
        Run focused backtesting.
        
        Returns:
            Success status
        """
        logger.info("📈 Starting focused backtesting for 5 tickers")
        
        try:
            if not FOCUSED_BACKTESTER_AVAILABLE:
                logger.error("Focused backtester not available")
                return False
            
            # Initialize backtester
            backtester = Focused5TickerBacktester(initial_capital=100000)
            
            # Run backtest
            start_date = datetime.now() - timedelta(days=365)
            end_date = datetime.now()
            
            results = backtester.run_focused_backtest(start_date, end_date)
            
            if results['status'] != 'success':
                logger.error(f"Backtest failed: {results.get('error', 'Unknown error')}")
                return False
            
            # Log results
            logger.info("📊 Backtest Results:")
            logger.info(f"  Initial Capital: ${results['initial_capital']:,.2f}")
            logger.info(f"  Final Capital: ${results['final_capital']:,.2f}")
            logger.info(f"  Total Return: {results['total_return']:.2%}")
            logger.info(f"  Total Trades: {results['trades']}")
            
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            
            # Save results
            with open('focused_backtest_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info("✅ Focused backtesting completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backtesting failed: {e}")
            return False
    
    async def run_complete_pipeline(self) -> bool:
        """
        Run complete focused pipeline.
        
        Returns:
            Success status
        """
        logger.info("🚀 Starting complete focused QuantAI pipeline")
        logger.info("=" * 70)
        logger.info("Focused 5-Ticker QuantAI Trading Platform")
        logger.info("Optimized for: AMZN, META, NVDA, GOOGL, AAPL")
        logger.info("=" * 70)
        
        try:
            # Step 1: Data Pipeline
            logger.info("Step 1: Data Pipeline")
            if not await self.run_data_pipeline():
                return False
            
            # Step 2: Model Training
            logger.info("Step 2: Model Training")
            if not await self.run_model_training():
                return False
            
            # Step 3: Risk Analysis
            logger.info("Step 3: Risk Analysis")
            if not await self.run_risk_analysis():
                return False
            
            # Step 4: Backtesting
            logger.info("Step 4: Backtesting")
            if not await self.run_backtesting():
                return False
            
            logger.info("✅ Complete focused pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Complete pipeline failed: {e}")
            return False
    
    def print_summary(self):
        """Print platform summary."""
        print("\n" + "=" * 70)
        print("📊 FOCUSED QUANTAI PLATFORM SUMMARY")
        print("=" * 70)
        
        print(f"\n🎯 FOCUSED TICKERS:")
        for ticker in self.tickers:
            print(f"   {ticker}")
        
        print(f"\n📊 DATA STATUS:")
        print(f"   Market Data: {'✅' if self.market_data else '❌'}")
        print(f"   Features: {'✅' if self.features else '❌'}")
        print(f"   Trained Models: {'✅' if self.trained_models else '❌'}")
        print(f"   Risk Metrics: {'✅' if self.performance_metrics else '❌'}")
        
        if self.performance_metrics:
            print(f"\n🛡️ RISK METRICS:")
            for ticker, metrics in self.performance_metrics.items():
                if ticker != 'portfolio':
                    print(f"   {ticker}:")
                    print(f"     Volatility: {metrics.get('volatility', 0):.2%}")
                    print(f"     Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                    print(f"     Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        
        print(f"\n🚀 NEXT STEPS:")
        print("1. Review risk metrics and adjust parameters")
        print("2. Implement real-time data feeds")
        print("3. Add reinforcement learning agents")
        print("4. Deploy to production environment")
        print("5. Monitor performance and rebalance")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Focused QuantAI Trading Platform")
    parser.add_argument(
        "command",
        choices=["data", "train", "risk", "backtest", "all", "summary"],
        help="Command to run"
    )
    parser.add_argument("--days", type=int, default=365, help="Days of data to fetch")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    
    args = parser.parse_args()
    
    # Initialize platform
    platform = FocusedQuantAIPlatform()
    
    try:
        if args.command == "data":
            success = await platform.run_data_pipeline(args.days)
        elif args.command == "train":
            success = await platform.run_model_training()
        elif args.command == "risk":
            success = await platform.run_risk_analysis()
        elif args.command == "backtest":
            success = await platform.run_backtesting()
        elif args.command == "all":
            success = await platform.run_complete_pipeline()
        elif args.command == "summary":
            platform.print_summary()
            success = True
        else:
            logger.error(f"Unknown command: {args.command}")
            success = False
        
        if success:
            logger.info("✅ Command completed successfully")
            platform.print_summary()
        else:
            logger.error("❌ Command failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
