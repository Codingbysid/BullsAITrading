#!/usr/bin/env python3
"""
Simple QF-Lib Test Script

This script tests QF-Lib with the correct imports for version 4.0.3.
"""

import sys
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_qf_lib_basic():
    """Test basic QF-Lib functionality."""
    logger.info("🧪 Testing QF-Lib basic functionality...")
    
    try:
        # Test basic import
        import qf_lib
        logger.info(f"✅ QF-Lib imported successfully")
        logger.info(f"   Version: {getattr(qf_lib, '__version__', 'Unknown')}")
        
        # Test Settings
        from qf_lib.settings import Settings
        logger.info("✅ Settings imported successfully")
        
        # Test containers
        from qf_lib.containers import QFSeries, QFDataFrame
        logger.info("✅ QFSeries and QFDataFrame imported successfully")
        
        # Test data providers
        from qf_lib.data_providers import DataProvider
        logger.info("✅ DataProvider imported successfully")
        
        # Test backtesting
        from qf_lib.backtesting import Portfolio, Broker
        logger.info("✅ Portfolio and Broker imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ QF-Lib import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def test_qf_lib_settings():
    """Test QF-Lib Settings creation."""
    logger.info("📋 Testing QF-Lib Settings...")
    
    try:
        from qf_lib.settings import Settings
        
        # Get project root
        project_root = Path(__file__).parent.parent
        settings_path = project_root / "config" / "qf_lib_settings.json"
        secret_settings_path = project_root / "config" / "qf_lib_secret_settings.json"
        
        # Create Settings
        if settings_path.exists() and secret_settings_path.exists():
            settings = Settings(str(settings_path), str(secret_settings_path))
            logger.info("✅ QF-Lib Settings created with config files")
        else:
            settings = Settings()
            logger.info("✅ QF-Lib Settings created with defaults")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ QF-Lib Settings test failed: {e}")
        return False

def test_qf_lib_containers():
    """Test QF-Lib containers."""
    logger.info("📦 Testing QF-Lib containers...")
    
    try:
        from qf_lib.containers import QFSeries, QFDataFrame
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample data
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)
        
        # Create QFSeries
        qf_series = QFSeries(data=prices, index=dates, name='AAPL')
        logger.info(f"✅ QFSeries created successfully")
        logger.info(f"   Data points: {len(qf_series)}")
        logger.info(f"   Date range: {qf_series.index.min()} to {qf_series.index.max()}")
        
        # Create QFDataFrame
        data = {
            'AAPL': prices,
            'GOOGL': prices * 1.1,
            'MSFT': prices * 0.9
        }
        qf_dataframe = QFDataFrame(data, index=dates)
        logger.info(f"✅ QFDataFrame created successfully")
        logger.info(f"   Columns: {list(qf_dataframe.columns)}")
        logger.info(f"   Shape: {qf_dataframe.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ QF-Lib containers test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 QF-Lib Simple Test")
    print("=" * 40)
    print()
    
    # Test basic imports
    print("1. Testing basic imports...")
    if not test_qf_lib_basic():
        print("❌ Basic import test failed")
        return False
    print()
    
    # Test Settings
    print("2. Testing Settings...")
    if not test_qf_lib_settings():
        print("❌ Settings test failed")
        return False
    print()
    
    # Test containers
    print("3. Testing containers...")
    if not test_qf_lib_containers():
        print("❌ Containers test failed")
        return False
    print()
    
    print("🎉 QF-Lib simple test completed successfully!")
    print("🚀 QF-Lib is working correctly")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
