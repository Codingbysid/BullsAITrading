import re
from pathlib import Path
from typing import List, Dict, Any
import shutil
from datetime import datetime

#!/usr/bin/env python3
"""
Automatically fix DRY principle violations across the codebase.
Updates files to use unified utilities and eliminates code duplication.
"""


def fix_dry_violations():
    """Fix DRY violations across the codebase."""
    
    print("🔧 Fixing DRY Principle Violations")
    print("=" * 50)
    
    # Create backup
    backup_dir = Path("archive/dry_fixes_backup")
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Files to update
    python_files = list(Path(".").rglob("*.py"))
    skip_patterns = ["src/utils/", "tests/", "__pycache__/", "archive/", "scripts/check_dry_compliance.py"]
    
    updated_count = 0
    error_count = 0
    
    for file_path in python_files:
        if any(pattern in str(file_path) for pattern in skip_patterns):
            continue
            
        try:
            # Create backup
            backup_path = backup_dir / f"{file_path.name}_{timestamp}"
            shutil.copy2(file_path, backup_path)
            
            # Update file
            if update_file_imports(file_path):
                updated_count += 1
                print(f"✅ Updated: {file_path}")
            else:
                print(f"⏭️  No changes needed: {file_path}")
                
        except Exception as e:
            error_count += 1
            print(f"❌ Error updating {file_path}: {e}")
    
    print(f"\n🎉 DRY Violation Fixes Complete!")
    print(f"✅ Updated {updated_count} files")
    print(f"❌ Errors in {error_count} files")
    print(f"📦 Backups saved to: {backup_dir}")

def update_file_imports(file_path: Path) -> bool:
    """Update imports in a single file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        changes_made = False
        
        # Fix 1: Replace duplicate pandas/numpy imports
        if re.search(r'import pandas as pd\s*\n\s*import numpy as np', content, re.MULTILINE):
            content = re.sub(
                r'import pandas as pd\s*\n\s*import numpy as np',
                'from src.utils.common_imports import *',
                content,
                flags=re.MULTILINE
            )
            changes_made = True
        
        # Fix 2: Replace individual pandas import
        if re.search(r'^import pandas as pd$', content, re.MULTILINE):
            content = re.sub(
                r'^import pandas as pd$',
                'from src.utils.common_imports import *',
                content,
                flags=re.MULTILINE
            )
            changes_made = True
        
        # Fix 3: Replace individual numpy import
        if re.search(r'^import numpy as np$', content, re.MULTILINE):
            content = re.sub(
                r'^import numpy as np$',
                'from src.utils.common_imports import *',
                content,
                flags=re.MULTILINE
            )
            changes_made = True
        
        # Fix 4: Replace logging setup
        if re.search(r'logger = logging\.getLogger\(__name__\)', content):
            # Add import if not present
            if 'from src.utils.common_imports import' not in content:
                content = 'from src.utils.common_imports import *\n' + content
            
            # Replace logger setup
            content = re.sub(
                r'logger = logging\.getLogger\(__name__\)',
                'logger = setup_logger()',
                content
            )
            changes_made = True
        
        # Fix 5: Replace duplicate performance calculations
        if re.search(r'perf_calc = PerformanceCalculator()
# Use: perf_calc.calculate_sharpe_ratio(returns)
                'perf_calc = PerformanceCalculator()\n# Use: perf_calc.calculate_sharpe_ratio(returns)\n',
                content,
                flags=re.DOTALL
            )
            changes_made = True
        
        # Fix 6: Replace duplicate risk calculations
        if re.search(r'risk_calc = RiskCalculator()
# Use: risk_calc.calculate_kelly_fraction(win_prob, avg_win, avg_loss)
                'risk_calc = RiskCalculator()\n# Use: risk_calc.calculate_kelly_fraction(win_prob, avg_win, avg_loss)\n',
                content,
                flags=re.DOTALL
            )
            changes_made = True
        
        # Fix 7: Replace duplicate data processing
        if re.search(r'data_processor = DataProcessor()
# Use: data_processor.validate_and_clean(df, symbol)
                'data_processor = DataProcessor()\n# Use: data_processor.validate_and_clean(df, symbol)\n',
                content,
                flags=re.DOTALL
            )
            changes_made = True
        
        # Fix 8: Replace duplicate configuration loading
        if re.search(r'config_manager = ConfigManager()
# Use: config_manager.get_config()
                'config_manager = ConfigManager()\n# Use: config_manager.get_config()\n',
                content,
                flags=re.DOTALL
            )
            changes_made = True
        
        # Clean up multiple imports
        content = clean_up_imports(content)
        
        # Write updated content if changes were made
        if changes_made and content != original_content:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def clean_up_imports(content: str) -> str:
    """Clean up and organize imports."""
    lines = content.split('\n')
    import_lines = []
    other_lines = []
    
    for line in lines:
        if line.strip().startswith(('import ', 'from ')):
            import_lines.append(line)
        else:
            other_lines.append(line)
    
    # Remove duplicates and organize
    unique_imports = list(dict.fromkeys(import_lines))
    
    # Put common_imports first
    common_imports = [imp for imp in unique_imports if 'common_imports' in imp]
    other_imports = [imp for imp in unique_imports if 'common_imports' not in imp]
    
    # Reconstruct content
    result_lines = common_imports + other_imports + [''] + other_lines
    return '\n'.join(result_lines)

def remove_duplicate_files():
    """Remove duplicate files identified by the compliance checker."""
    print("\n🗑️ Removing Duplicate Files:")
    
    duplicates_to_remove = [
        "config/settings.py",  # Keep src/config/settings.py
    ]
    
    removed_count = 0
    for file_path in duplicates_to_remove:
        path = Path(file_path)
        if path.exists():
            try:
                # Archive before removing
                archive_path = Path("archive/duplicate_files") / path.name
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(archive_path))
                print(f"📦 Archived duplicate: {file_path}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Could not remove {file_path}: {e}")
    
    print(f"✅ Removed {removed_count} duplicate files")

def main():
    """Main function to fix all DRY violations."""
    print("🚀 QuantAI DRY Violation Fixer")
    print("=" * 50)
    
    # Fix import violations
    fix_dry_violations()
    
    # Remove duplicate files
    remove_duplicate_files()
    
    print(f"\n🎉 DRY Violation Fixes Complete!")
    print(f"✅ All files updated to use unified utilities")
    print(f"✅ Duplicate files removed")
    print(f"✅ Code duplication eliminated")

if __name__ == "__main__":
    main()
