#!/usr/bin/env python3
"""
QF-Lib Configuration Setup Script

This script sets up the proper QF-Lib configuration based on the official documentation.
"""

import os
import sys
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_qf_lib_environment():
    """Set up QF-Lib environment variables and configuration."""
    logger.info("🔧 Setting up QF-Lib environment...")
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    project_root_abs = str(project_root.absolute())
    
    # Set QF-Lib starting directory
    os.environ['QF_STARTING_DIRECTORY'] = project_root_abs
    logger.info(f"✅ Set QF_STARTING_DIRECTORY to: {project_root_abs}")
    
    # Set other QF-Lib environment variables
    os.environ['QF_OUTPUT_DIRECTORY'] = str(project_root / "output")
    os.environ['QF_DATA_DIRECTORY'] = str(project_root / "data")
    os.environ['QF_CACHE_DIRECTORY'] = str(project_root / "cache")
    os.environ['QF_LOGS_DIRECTORY'] = str(project_root / "logs")
    
    # Create necessary directories
    directories = [
        project_root / "output",
        project_root / "data",
        project_root / "cache",
        project_root / "logs",
        project_root / "assets",
        project_root / "assets" / "css"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")
    
    return project_root_abs

def setup_qf_lib_settings():
    """Set up QF-Lib settings and configuration files."""
    logger.info("📋 Setting up QF-Lib settings...")
    
    project_root = Path(__file__).parent.parent
    
    # Settings file path
    settings_path = project_root / "config" / "qf_lib_settings.json"
    secret_settings_path = project_root / "config" / "qf_lib_secret_settings.json"
    
    if not settings_path.exists():
        logger.error(f"❌ Settings file not found: {settings_path}")
        return False
    
    if not secret_settings_path.exists():
        logger.error(f"❌ Secret settings file not found: {secret_settings_path}")
        return False
    
    logger.info(f"✅ Settings file: {settings_path}")
    logger.info(f"✅ Secret settings file: {secret_settings_path}")
    
    return str(settings_path), str(secret_settings_path)

def test_qf_lib_import():
    """Test QF-Lib import and basic functionality."""
    logger.info("🧪 Testing QF-Lib import...")
    
    try:
        # Import QF-Lib core components
        from qf_lib.settings import Settings
        from qf_lib.containers import QFSeries, QFDataFrame
        from qf_lib.data_providers import DataProvider
        from qf_lib.backtesting import Portfolio, Broker
        
        logger.info("✅ QF-Lib core components imported successfully")
        
        # Test Settings creation
        project_root = Path(__file__).parent.parent
        settings_path = project_root / "config" / "qf_lib_settings.json"
        secret_settings_path = project_root / "config" / "qf_lib_secret_settings.json"
        
        settings = Settings(str(settings_path), str(secret_settings_path))
        logger.info("✅ QF-Lib Settings created successfully")
        
        # Test DataContainer
        data_container = DataContainer()
        logger.info("✅ QF-Lib DataContainer created successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ QF-Lib import failed: {e}")
        logger.error("Please ensure QF-Lib is installed: pip install qf-lib")
        return False
    except Exception as e:
        logger.error(f"❌ QF-Lib setup failed: {e}")
        return False

def create_qf_lib_demo():
    """Create a simple QF-Lib demo to test functionality."""
    logger.info("🚀 Creating QF-Lib demo...")
    
    try:
        from qf_lib.settings import Settings
        from qf_lib.containers.data_container import DataContainer
        from qf_lib.containers.series.qf_series import QFSeries
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Setup
        project_root = Path(__file__).parent.parent
        settings_path = project_root / "config" / "qf_lib_settings.json"
        secret_settings_path = project_root / "config" / "qf_lib_secret_settings.json"
        
        # Create Settings
        settings = Settings(str(settings_path), str(secret_settings_path))
        
        # Create DataContainer
        data_container = DataContainer()
        
        # Create sample data
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)
        
        # Create QFSeries
        qf_series = QFSeries(data=prices, index=dates, name='AAPL')
        
        logger.info(f"✅ QF-Lib demo created successfully")
        logger.info(f"   Data points: {len(qf_series)}")
        logger.info(f"   Date range: {qf_series.index.min()} to {qf_series.index.max()}")
        logger.info(f"   Price range: ${qf_series.min():.2f} to ${qf_series.max():.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ QF-Lib demo failed: {e}")
        return False

def main():
    """Main function to set up QF-Lib configuration."""
    print("🔧 QF-Lib Configuration Setup")
    print("=" * 50)
    print()
    
    # Setup environment
    starting_dir = setup_qf_lib_environment()
    print(f"✅ Starting directory: {starting_dir}")
    
    # Setup settings
    settings_result = setup_qf_lib_settings()
    if not settings_result:
        print("❌ Settings setup failed")
        return False
    
    settings_path, secret_settings_path = settings_result
    print(f"✅ Settings file: {settings_path}")
    print(f"✅ Secret settings file: {secret_settings_path}")
    
    # Test QF-Lib import
    if not test_qf_lib_import():
        print("❌ QF-Lib import test failed")
        return False
    
    # Create demo
    if not create_qf_lib_demo():
        print("❌ QF-Lib demo failed")
        return False
    
    print("\n🎉 QF-Lib configuration completed successfully!")
    print("🚀 You can now run the QF-Lib backtester")
    print("   python scripts/run_qf_backtest.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
