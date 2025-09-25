"""
Main application entry point for QuantAI Trading Platform.
"""

import asyncio
import sys
import argparse
import logging

from src.config.settings import get_settings
from src.utils.logger import setup_logging, get_logger
from src.api.main import app
import uvicorn


def setup_application():
    """Setup application configuration and logging."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level, log_format=settings.log_format)
    logger = get_logger("main")
    logger.info("QuantAI Trading Platform starting up")
    return settings, logger


async def run_data_pipeline():
    """Run the data pipeline to fetch and process market data."""
    from src.data.data_sources import data_manager
    from src.data.feature_engineering import FeatureEngineer
    
    logger = get_logger("data_pipeline")
    settings = get_settings()
    
    logger.info("Starting data pipeline")
    
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365)
    
    for symbol in settings.target_symbols:
        try:
            logger.info(f"Fetching data for {symbol}")
            data = await data_manager.get_market_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if not data.empty:
                logger.info(f"Successfully fetched {len(data)} records for {symbol}")
                feature_engineer = FeatureEngineer()
                features = feature_engineer.create_all_features(data)
                
                if not features.empty:
                    logger.info(f"Created {len(features.columns)} features for {symbol}")
                else:
                    logger.warning(f"Failed to create features for {symbol}")
            else:
                logger.warning(f"No data available for {symbol}")
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    logger.info("Data pipeline completed")


async def run_model_training():
    """Run advanced model training pipeline."""
    from src.training.training_pipeline import AdvancedTrainingPipeline, TrainingConfig
    
    logger = get_logger("model_training")
    logger.info("Starting advanced model training pipeline")
    
    try:
        # Create training configuration
        config = TrainingConfig()
        
        # Initialize training pipeline
        pipeline = AdvancedTrainingPipeline(config)
        
        # Run complete training
        results = await pipeline.run_complete_training()
        
        if results['status'] == 'success':
            logger.info("✅ Model training completed successfully!")
            logger.info(f"📊 Training samples: {results['training_samples']}")
            logger.info(f"🔧 Features: {results['features_count']}")
            
            if 'evaluation' in results:
                eval_results = results['evaluation']
                logger.info(f"🏆 Best model: {eval_results['model_name']}")
                logger.info(f"📈 F1 Score: {eval_results['metrics']['f1_score']:.3f}")
                logger.info(f"🎯 Accuracy: {eval_results['metrics']['accuracy']:.3f}")
        else:
            logger.error(f"❌ Model training failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise


async def run_backtesting():
    """Run backtesting on trained models."""
    logger = get_logger("backtesting")
    logger.info("Starting backtesting")
    
    try:
        # Run simple backtest (always works)
        logger.info("🚀 Running Simple Backtester...")
        
        # Import and run simple backtest
        import subprocess
        import sys
        
        result = subprocess.run([sys.executable, "backtesters/simple_backtest.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Simple backtesting completed successfully!")
            logger.info("📊 Check simple_backtest_results.json for detailed results")
        else:
            logger.error(f"❌ Simple backtesting failed: {result.stderr}")
            
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        raise


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    logger = get_logger("api_server")
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )


def run_dashboard_server(host: str = "0.0.0.0", port: int = 8001):
    """Run the dashboard server."""
    from src.dashboard.dashboard_api import run_dashboard
    logger = get_logger("dashboard")
    logger.info(f"Starting dashboard server on {host}:{port}")
    run_dashboard(host=host, port=port)


def run_tests(test_suite: str = "all"):
    """Run the test suite."""
    import subprocess
    
    logger = get_logger("testing")
    logger.info(f"Running test suite: {test_suite}")
    
    try:
        result = subprocess.run([
            sys.executable, "tests/run_tests.py", test_suite
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            logger.info("All tests passed!")
        else:
            logger.error(f"Tests failed with return code: {result.returncode}")
            
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="QuantAI Trading Platform")
    parser.add_argument(
        "command",
        choices=["api", "data-pipeline", "train", "backtest", "dashboard", "test", "all"],
        help="Command to run"
    )
    parser.add_argument("--host", default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument("--dashboard-port", type=int, default=8001, help="Dashboard port")
    parser.add_argument("--test-suite", default="all", help="Test suite to run")
    
    args = parser.parse_args()
    
    # Setup application
    settings, logger = setup_application()
    
    try:
        if args.command == "api":
            run_api_server(args.host, args.port)
        elif args.command == "data-pipeline":
            asyncio.run(run_data_pipeline())
        elif args.command == "train":
            asyncio.run(run_model_training())
        elif args.command == "backtest":
            asyncio.run(run_backtesting())
        elif args.command == "dashboard":
            run_dashboard_server(args.host, args.dashboard_port)
        elif args.command == "test":
            success = run_tests(args.test_suite)
            sys.exit(0 if success else 1)
        elif args.command == "all":
            logger.info("Running full pipeline")
            asyncio.run(run_data_pipeline())
            asyncio.run(run_model_training())
            asyncio.run(run_backtesting())
            run_api_server(args.host, args.port)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()