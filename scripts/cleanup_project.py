#!/usr/bin/env python3
"""
Project Cleanup Script - Remove unnecessary files and consolidate duplicates.
Implements Phase 1 of the DRY principle action plan.
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime

def cleanup_project():
    """Remove unnecessary files and organize project structure."""
    
    print("🧹 QuantAI Project Cleanup - Phase 1")
    print("=" * 50)
    
    # Files to remove (temporary, duplicate, or unnecessary)
    files_to_remove = [
        "fix_imports.py",           # Temporary fix file
        "real_ai_recommendations.py",  # Duplicate functionality
        "trained_ensemble_test_results.json",  # Old test results
        "run_quantai_commands.md",  # Redundant documentation
        "LOCAL_SETUP.md",           # Redundant documentation
    ]
    
    # Remove unnecessary files
    removed_count = 0
    for file_name in files_to_remove:
        file_paths = list(Path(".").rglob(file_name))
        for file_path in file_paths:
            try:
                if file_path.exists():
                    os.remove(file_path)
                    print(f"✅ Removed: {file_path}")
                    removed_count += 1
            except Exception as e:
                print(f"⚠️ Could not remove {file_path}: {e}")
    
    # Create new directory structure
    new_dirs = [
        "results/backtests",
        "results/models", 
        "results/analysis",
        "archive/old_results",
        "archive/duplicate_docs"
    ]
    
    created_count = 0
    for dir_path in new_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
        created_count += 1
    
    # Archive old results files
    archive_old_results()
    
    # Consolidate documentation
    consolidate_documentation()
    
    print(f"\n🎉 Phase 1 Cleanup Complete!")
    print(f"✅ Removed {removed_count} unnecessary files")
    print(f"✅ Created {created_count} new directories")
    print(f"✅ Archived old results")
    print(f"✅ Consolidated documentation")

def archive_old_results():
    """Archive old result files."""
    print("\n📦 Archiving old results...")
    
    archive_dir = Path("archive/old_results")
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Find and archive old result files
    result_patterns = [
        "*_results_*.json",
        "*_test_results.json", 
        "working_results_*.json",
        "success_results_*.json"
    ]
    
    archived_count = 0
    for pattern in result_patterns:
        for file_path in Path(".").rglob(pattern):
            if file_path.is_file() and "archive" not in str(file_path):
                try:
                    # Create archive filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    archive_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
                    archive_path = archive_dir / archive_name
                    
                    shutil.move(str(file_path), str(archive_path))
                    print(f"📦 Archived: {file_path} → {archive_path}")
                    archived_count += 1
                except Exception as e:
                    print(f"⚠️ Could not archive {file_path}: {e}")
    
    print(f"✅ Archived {archived_count} old result files")

def consolidate_documentation():
    """Consolidate duplicate documentation files."""
    print("\n📚 Consolidating documentation...")
    
    # Check for duplicate README files
    readme_files = list(Path(".").rglob("README.md"))
    if len(readme_files) > 1:
        print(f"Found {len(readme_files)} README files, keeping main one")
        for readme_file in readme_files[1:]:  # Keep first, archive others
            if "docs" not in str(readme_file):
                try:
                    archive_path = Path("archive/duplicate_docs") / f"README_{readme_file.parent.name}.md"
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(readme_file), str(archive_path))
                    print(f"📦 Archived duplicate: {readme_file}")
                except Exception as e:
                    print(f"⚠️ Could not archive {readme_file}: {e}")
    
    # Check for duplicate ABOUT files
    about_files = list(Path(".").rglob("ABOUT.md"))
    if len(about_files) > 1:
        print(f"Found {len(about_files)} ABOUT files, keeping main one")
        for about_file in about_files[1:]:  # Keep first, archive others
            if "docs" not in str(about_file):
                try:
                    archive_path = Path("archive/duplicate_docs") / f"ABOUT_{about_file.parent.name}.md"
                    archive_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(about_file), str(archive_path))
                    print(f"📦 Archived duplicate: {about_file}")
                except Exception as e:
                    print(f"⚠️ Could not archive {about_file}: {e}")

def analyze_project_structure():
    """Analyze current project structure for cleanup opportunities."""
    print("\n🔍 Analyzing project structure...")
    
    # Count files by type
    file_counts = {}
    total_files = 0
    
    for file_path in Path(".").rglob("*"):
        if file_path.is_file():
            total_files += 1
            suffix = file_path.suffix.lower()
            file_counts[suffix] = file_counts.get(suffix, 0) + 1
    
    print(f"📊 Project Analysis:")
    print(f"   Total files: {total_files}")
    print(f"   Python files: {file_counts.get('.py', 0)}")
    print(f"   JSON files: {file_counts.get('.json', 0)}")
    print(f"   Markdown files: {file_counts.get('.md', 0)}")
    print(f"   CSV files: {file_counts.get('.csv', 0)}")
    
    # Identify potential duplicates
    potential_duplicates = []
    for file_path in Path(".").rglob("*.py"):
        if file_path.name in ["__init__.py", "main.py", "test.py"]:
            # Check for multiple files with same name
            same_name_files = list(Path(".").rglob(file_path.name))
            if len(same_name_files) > 1:
                potential_duplicates.extend(same_name_files)
    
    if potential_duplicates:
        print(f"\n⚠️ Potential duplicate files found:")
        for dup_file in set(potential_duplicates):
            print(f"   - {dup_file}")
    else:
        print(f"\n✅ No obvious duplicate files found")

if __name__ == "__main__":
    analyze_project_structure()
    cleanup_project()
