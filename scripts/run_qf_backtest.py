#!/usr/bin/env python3
"""
QuantAI Trading Platform - QF-Lib Backtester Runner

This script runs the QF-Lib backtester in the qf_env environment.
"""

import sys
import os
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_qf_env():
    """Check if qf_env is available and QF-Lib is installed."""
    try:
        # Check if conda is available
        result = subprocess.run(['conda', 'info', '--envs'], 
                              capture_output=True, text=True, check=True)
        
        if 'qf_env' in result.stdout:
            logger.info("✅ qf_env environment found")
            return True
        else:
            logger.error("❌ qf_env environment not found")
            logger.info("Please create the environment with: conda create -n qf_env python=3.8")
            return False
            
    except subprocess.CalledProcessError:
        logger.error("❌ conda not found. Please install Anaconda or Miniconda")
        return False
    except FileNotFoundError:
        logger.error("❌ conda command not found. Please install Anaconda or Miniconda")
        return False

def run_qf_backtester():
    """Run the QF-Lib backtester in the qf_env environment."""
    logger.info("🚀 Starting QF-Lib Backtester in qf_env environment")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # QF-Lib backtester path
    backtester_path = project_root / "apps" / "backtesting" / "backtesters" / "qf_lib_backtester.py"
    
    if not backtester_path.exists():
        logger.error(f"❌ QF-Lib backtester not found at {backtester_path}")
        return False
    
    try:
        # Run the backtester in qf_env environment
        cmd = [
            'conda', 'run', '-n', 'qf_env', 
            'python', str(backtester_path)
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Change to project directory
        os.chdir(project_root)
        
        # Run the backtester
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        logger.info("✅ QF-Lib backtester completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error running QF-Lib backtester: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function to run QF-Lib backtester."""
    print("🎯 QuantAI Trading Platform - QF-Lib Backtester Runner")
    print("=" * 60)
    print()
    
    # Check if qf_env is available
    if not check_qf_env():
        print("❌ qf_env environment not available")
        print("Please create it with: conda create -n qf_env python=3.8")
        print("Then install QF-Lib: conda activate qf_env && pip install qf-lib")
        return False
    
    print("✅ qf_env environment is available")
    print("🚀 Starting QF-Lib backtester...")
    print()
    
    # Run the QF-Lib backtester
    success = run_qf_backtester()
    
    if success:
        print("\n🎉 QF-Lib backtester completed successfully!")
        print("📊 Check the results in the output above")
        print("📁 Results are saved to JSON files")
    else:
        print("\n❌ QF-Lib backtester failed")
        print("Please check the error messages above")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
