#!/usr/bin/env python3
"""
Check QF-Lib structure and available modules
"""

import sys
import os

def check_qf_lib_structure():
    """Check QF-Lib structure and available modules."""
    print("🔍 Checking QF-Lib structure...")
    
    try:
        import qf_lib
        print(f"✅ QF-Lib imported successfully")
        print(f"   Version: {getattr(qf_lib, '__version__', 'Unknown')}")
        print(f"   Location: {qf_lib.__file__}")
        print()
        
        # Check available modules
        import pkgutil
        print("📦 Available modules in qf_lib:")
        modules = []
        for importer, modname, ispkg in pkgutil.walk_packages(qf_lib.__path__, qf_lib.__name__ + '.'):
            if not modname.startswith('qf_lib.tests'):
                modules.append(modname)
                print(f"   {modname}")
        
        print(f"\n📊 Total modules found: {len(modules)}")
        
        # Check specific imports that we need
        print("\n🧪 Testing specific imports:")
        
        # Test Settings
        try:
            from qf_lib.settings import Settings
            print("✅ Settings imported successfully")
        except ImportError as e:
            print(f"❌ Settings import failed: {e}")
        
        # Test containers
        try:
            from qf_lib.containers import QFSeries, QFDataFrame
            print("✅ QFSeries and QFDataFrame imported successfully")
        except ImportError as e:
            print(f"❌ QFSeries/QFDataFrame import failed: {e}")
        
        # Test backtesting
        try:
            from qf_lib.backtesting import Portfolio, Broker
            print("✅ Portfolio and Broker imported successfully")
        except ImportError as e:
            print(f"❌ Portfolio/Broker import failed: {e}")
        
        # Test data providers
        try:
            from qf_lib.data_providers import DataProvider
            print("✅ DataProvider imported successfully")
        except ImportError as e:
            print(f"❌ DataProvider import failed: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ QF-Lib import failed: {e}")
        print("Please ensure you're in the qf_project environment:")
        print("  conda activate qf_project")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function."""
    print("🔍 QF-Lib Structure Checker")
    print("=" * 40)
    print()
    
    success = check_qf_lib_structure()
    
    if success:
        print("\n✅ QF-Lib structure check completed")
    else:
        print("\n❌ QF-Lib structure check failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
