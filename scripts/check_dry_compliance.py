#!/usr/bin/env python3
"""
Check DRY principle compliance across the codebase.
Identifies code duplication and violations of the DRY principle.
"""

import ast
import re
from pathlib import Path
from collections import defaultdict
import sys

def check_dry_compliance():
    """Check for code duplication and DRY violations."""
    
    print("🔍 Checking DRY Principle Compliance")
    print("=" * 50)
    
    violations = []
    duplicate_patterns = []
    
    # Check for duplicate function definitions
    python_files = list(Path(".").rglob("*.py"))
    
    # Skip utility files and test files for now
    skip_patterns = ["src/utils/", "tests/", "__pycache__/", "archive/"]
    
    for file_path in python_files:
        if any(pattern in str(file_path) for pattern in skip_patterns):
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for duplicate imports (should use common_imports)
            if re.search(r'import pandas as pd\s+import numpy as np', content, re.MULTILINE):
                violations.append(f"{file_path}: Should use 'from src.utils.common_imports import *'")
            
            # Check for duplicate logging setup
            if re.search(r'logger = logging\.getLogger\(__name__\)', content):
                violations.append(f"{file_path}: Should use unified logging from common_imports")
            
            # Check for duplicate performance calculations
            if re.search(r'def calculate_sharpe_ratio', content):
                violations.append(f"{file_path}: Should use PerformanceCalculator from utils")
            
            # Check for duplicate risk calculations
            if re.search(r'def calculate_kelly', content):
                violations.append(f"{file_path}: Should use RiskCalculator from utils")
            
            # Check for duplicate data processing
            if re.search(r'def validate_market_data', content):
                violations.append(f"{file_path}: Should use DataProcessor from utils")
            
            # Check for duplicate configuration loading
            if re.search(r'def load_config', content):
                violations.append(f"{file_path}: Should use ConfigManager from utils")
                
        except Exception as e:
            print(f"Warning: Could not check {file_path}: {e}")
    
    # Check for duplicate files
    duplicate_files = find_duplicate_files()
    
    # Report results
    print(f"\n📊 DRY Compliance Analysis:")
    print(f"   Files analyzed: {len(python_files)}")
    print(f"   Violations found: {len(violations)}")
    print(f"   Duplicate files: {len(duplicate_files)}")
    
    if violations:
        print(f"\n❌ DRY Violations Found:")
        for i, violation in enumerate(violations, 1):
            print(f"   {i}. {violation}")
    
    if duplicate_files:
        print(f"\n⚠️ Duplicate Files Found:")
        for dup_file in duplicate_files:
            print(f"   - {dup_file}")
    
    if not violations and not duplicate_files:
        print(f"\n✅ No DRY violations found!")
        print(f"✅ Codebase follows DRY principle!")
        return True
    else:
        print(f"\n❌ DRY violations detected - needs cleanup")
        return False

def find_duplicate_files():
    """Find potentially duplicate files."""
    duplicate_files = []
    
    # Look for files with similar names
    file_groups = defaultdict(list)
    
    for file_path in Path(".").rglob("*.py"):
        if "src/utils/" in str(file_path) or "tests/" in str(file_path):
            continue
            
        # Group by filename
        file_groups[file_path.name].append(file_path)
    
    # Find groups with multiple files
    for filename, files in file_groups.items():
        if len(files) > 1:
            duplicate_files.extend(files)
    
    return duplicate_files

def check_unified_imports_usage():
    """Check if files are using unified imports."""
    print("\n🔍 Checking Unified Imports Usage:")
    
    python_files = list(Path(".").rglob("*.py"))
    skip_patterns = ["src/utils/", "tests/", "__pycache__/", "archive/"]
    
    using_unified = 0
    not_using_unified = 0
    
    for file_path in python_files:
        if any(pattern in str(file_path) for pattern in skip_patterns):
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if "from src.utils.common_imports import" in content:
                using_unified += 1
            elif "import pandas" in content or "import numpy" in content:
                not_using_unified += 1
                
        except Exception as e:
            continue
    
    print(f"   Files using unified imports: {using_unified}")
    print(f"   Files not using unified imports: {not_using_unified}")
    
    if not_using_unified > 0:
        print(f"   ⚠️ {not_using_unified} files still need to be updated to use unified imports")
    else:
        print(f"   ✅ All files are using unified imports!")

def check_code_duplication():
    """Check for code duplication patterns."""
    print("\n🔍 Checking Code Duplication Patterns:")
    
    # Common duplication patterns
    patterns = {
        "logging setup": r"logger = logging\.getLogger",
        "pandas import": r"import pandas as pd",
        "numpy import": r"import numpy as np",
        "json loading": r"def load_json",
        "data validation": r"def validate_.*data",
        "performance calc": r"def calculate_.*ratio",
        "risk calc": r"def calculate_.*risk"
    }
    
    python_files = list(Path(".").rglob("*.py"))
    skip_patterns = ["src/utils/", "tests/", "__pycache__/", "archive/"]
    
    pattern_counts = defaultdict(int)
    
    for file_path in python_files:
        if any(pattern in str(file_path) for pattern in skip_patterns):
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            for pattern_name, pattern in patterns.items():
                if re.search(pattern, content):
                    pattern_counts[pattern_name] += 1
                    
        except Exception as e:
            continue
    
    print(f"   Duplication patterns found:")
    for pattern_name, count in pattern_counts.items():
        if count > 1:
            print(f"   - {pattern_name}: {count} occurrences (should be 1 in utils)")
        else:
            print(f"   - {pattern_name}: {count} occurrences ✅")

def main():
    """Main function to run all DRY compliance checks."""
    print("🚀 QuantAI DRY Principle Compliance Checker")
    print("=" * 60)
    
    # Run all checks
    compliance_ok = check_dry_compliance()
    check_unified_imports_usage()
    check_code_duplication()
    
    print(f"\n🎯 Summary:")
    if compliance_ok:
        print(f"✅ DRY principle compliance: PASSED")
        print(f"✅ Codebase follows best practices")
    else:
        print(f"❌ DRY principle compliance: FAILED")
        print(f"❌ Code cleanup needed")
    
    return compliance_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
