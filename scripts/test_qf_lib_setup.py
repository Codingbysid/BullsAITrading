#!/usr/bin/env python3
"""
Test QF-Lib Setup and Configuration

This script tests the QF-Lib setup and configuration to ensure
everything is working correctly.
"""

import sys
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment_variables():
    """Test QF-Lib environment variables."""
    logger.info("🔧 Testing QF-Lib environment variables...")
    
    required_vars = ['QF_STARTING_DIRECTORY']
    optional_vars = ['QF_OUTPUT_DIRECTORY', 'QF_DATA_DIRECTORY', 'QF_CACHE_DIRECTORY', 'QF_LOGS_DIRECTORY']
    
    # Check required variables
    for var in required_vars:
        if var in os.environ:
            logger.info(f"✅ {var}: {os.environ[var]}")
        else:
            logger.warning(f"⚠️ {var} not set")
    
    # Check optional variables
    for var in optional_vars:
        if var in os.environ:
            logger.info(f"✅ {var}: {os.environ[var]}")
        else:
            logger.info(f"ℹ️ {var} not set (optional)")
    
    return True

def test_qf_lib_import():
    """Test QF-Lib import and basic functionality."""
    logger.info("🧪 Testing QF-Lib import...")
    
    try:
        # Test basic import
        import qf_lib
        logger.info("✅ QF-Lib imported successfully")
        
        # Test core components
        from qf_lib.settings import Settings
        from qf_lib.containers import QFSeries, QFDataFrame
        from qf_lib.data_providers import DataProvider
        from qf_lib.backtesting import Portfolio, Broker
        
        logger.info("✅ QF-Lib core components imported successfully")
        
        # Test Settings creation
        project_root = Path(__file__).parent.parent
        settings_path = project_root / "config" / "qf_lib_settings.json"
        secret_settings_path = project_root / "config" / "qf_lib_secret_settings.json"
        
        if settings_path.exists() and secret_settings_path.exists():
            settings = Settings(str(settings_path), str(secret_settings_path))
            logger.info("✅ QF-Lib Settings created successfully")
        else:
            logger.warning("⚠️ Settings files not found, creating default settings...")
            # Create default settings
            settings = Settings()
            logger.info("✅ QF-Lib Settings created with defaults")
        
        # Test DataContainer
        data_container = DataContainer()
        logger.info("✅ QF-Lib DataContainer created successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ QF-Lib import failed: {e}")
        logger.error("Please install QF-Lib: pip install qf-lib")
        return False
    except Exception as e:
        logger.error(f"❌ QF-Lib setup failed: {e}")
        return False

def test_qf_lib_directories():
    """Test QF-Lib directory structure."""
    logger.info("📁 Testing QF-Lib directory structure...")
    
    project_root = Path(__file__).parent.parent
    required_dirs = ['output', 'data', 'cache', 'logs']
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            logger.info(f"✅ {dir_name}/ directory exists")
        else:
            logger.warning(f"⚠️ {dir_name}/ directory not found, creating...")
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ {dir_name}/ directory created")
    
    return True

def test_qf_lib_settings():
    """Test QF-Lib settings files."""
    logger.info("📋 Testing QF-Lib settings files...")
    
    project_root = Path(__file__).parent.parent
    settings_path = project_root / "config" / "qf_lib_settings.json"
    secret_settings_path = project_root / "config" / "qf_lib_secret_settings.json"
    
    if settings_path.exists():
        logger.info(f"✅ Settings file found: {settings_path}")
    else:
        logger.warning(f"⚠️ Settings file not found: {settings_path}")
    
    if secret_settings_path.exists():
        logger.info(f"✅ Secret settings file found: {secret_settings_path}")
    else:
        logger.warning(f"⚠️ Secret settings file not found: {secret_settings_path}")
    
    return True

def test_qf_lib_demo():
    """Test QF-Lib with a simple demo."""
    logger.info("🚀 Testing QF-Lib with demo...")
    
    try:
        from qf_lib.settings import Settings
        from qf_lib.containers.data_container import DataContainer
        from qf_lib.containers.series.qf_series import QFSeries
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create demo data
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
    """Main test function."""
    print("🧪 QF-Lib Setup Test")
    print("=" * 40)
    print()
    
    # Test environment variables
    print("1. Testing environment variables...")
    test_environment_variables()
    print()
    
    # Test QF-Lib import
    print("2. Testing QF-Lib import...")
    if not test_qf_lib_import():
        print("❌ QF-Lib import test failed")
        return False
    print()
    
    # Test directories
    print("3. Testing directory structure...")
    test_qf_lib_directories()
    print()
    
    # Test settings
    print("4. Testing settings files...")
    test_qf_lib_settings()
    print()
    
    # Test demo
    print("5. Testing QF-Lib demo...")
    if not test_qf_lib_demo():
        print("❌ QF-Lib demo test failed")
        return False
    print()
    
    print("🎉 QF-Lib setup test completed successfully!")
    print("🚀 You can now run the QF-Lib backtester")
    print("   python scripts/run_qf_backtest.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
