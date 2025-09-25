#!/usr/bin/env python3
"""
Test script to verify QF-Lib environment setup
"""

import sys
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_qf_lib_import():
    """Test if QF-Lib can be imported."""
    try:
        import qf_lib
        logger.info("✅ QF-Lib imported successfully")
        logger.info(f"   Version: {qf_lib.__version__ if hasattr(qf_lib, '__version__') else 'Unknown'}")
        return True
    except ImportError as e:
        logger.error(f"❌ QF-Lib import failed: {e}")
        return False

def test_qf_lib_components():
    """Test QF-Lib components."""
    try:
        from qf_lib.containers.data_container import DataContainer
        from qf_lib.containers.series.qf_series import QFSeries
        from qf_lib.containers.dataframe.qf_dataframe import QFDataFrame
        from qf_lib.backtesting.portfolio.portfolio import Portfolio
        from qf_lib.backtesting.broker.broker import Broker
        from qf_lib.backtesting.execution.execution_model import ExecutionModel
        
        logger.info("✅ QF-Lib components imported successfully")
        logger.info("   - DataContainer")
        logger.info("   - QFSeries")
        logger.info("   - QFDataFrame")
        logger.info("   - Portfolio")
        logger.info("   - Broker")
        logger.info("   - ExecutionModel")
        return True
    except ImportError as e:
        logger.error(f"❌ QF-Lib components import failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 QF-Lib Environment Test")
    print("=" * 40)
    print()
    
    # Test QF-Lib import
    print("Testing QF-Lib import...")
    if not test_qf_lib_import():
        print("❌ QF-Lib not available in current environment")
        print("Please activate qf_env: conda activate qf_env")
        return False
    
    # Test QF-Lib components
    print("\nTesting QF-Lib components...")
    if not test_qf_lib_components():
        print("❌ QF-Lib components not available")
        return False
    
    print("\n✅ QF-Lib environment is ready!")
    print("🚀 You can now run the QF-Lib backtester")
    print("   python scripts/run_qf_backtest.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
