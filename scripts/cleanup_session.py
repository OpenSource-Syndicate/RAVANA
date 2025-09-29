#!/usr/bin/env python3
"""
RAVANA Session Cleanup Script
Deletes session files for a fresh start
"""

import os
import shutil
from pathlib import Path


def delete_file_or_directory(path):
    """Safely delete a file or directory"""
    try:
        if os.path.isfile(path):
            os.remove(path)
            print(f"Deleted file: {path}")
            return True
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Deleted directory: {path}")
            return True
    except Exception as e:
        print(f"Error deleting {path}: {e}")
        return False
    return False


def delete_pycache_dirs(root_path):
    """Delete all __pycache__ directories"""
    deleted_count = 0
    for pycache_dir in Path(root_path).rglob('__pycache__'):
        if delete_file_or_directory(str(pycache_dir)):
            deleted_count += 1
    return deleted_count


def delete_logs(root_path):
    """Delete log files"""
    deleted_count = 0
    log_patterns = ["*.log", "*.jsonl"]

    for pattern in log_patterns:
        for log_file in root_path.glob(pattern):
            if delete_file_or_directory(str(log_file)):
                deleted_count += 1

    return deleted_count


def delete_test_files(root_path):
    """Delete pytest cache directories"""
    deleted_count = 0
    for pytest_cache in Path(root_path).rglob('.pytest_cache'):
        if delete_file_or_directory(str(pytest_cache)):
            deleted_count += 1
    return deleted_count


def main():
    """Main cleanup function"""
    project_root = Path(__file__).parent.absolute(
    )  # Use script location as root
    print(f"Cleaning RAVANA session files in: {project_root}")

    # Confirm before proceeding
    confirm = input("\nThis will delete all session files. Continue? (y/N): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return

    deleted_items = 0

    # Files and directories to delete
    targets = [
        # Database files
        "ravana_agi.db",
        "trends.db",
        "chroma_db",

        # Profile and memory files
        "profiles",
        "shared_memory",

        # Knowledge files
        "knowledge_id_map.pkl",
        "knowledge_index.faiss",

        # Log files
        "ravana_agi.log",
    ]

    # Delete specific targets
    for target in targets:
        target_path = project_root / target
        if target_path.exists() and delete_file_or_directory(str(target_path)):
            deleted_items += 1

    # Delete __pycache__ directories
    pycache_count = delete_pycache_dirs(project_root)
    deleted_items += pycache_count

    # Delete pytest cache directories
    test_cache_count = delete_test_files(project_root)
    deleted_items += test_cache_count

    # Delete log files
    log_count = delete_logs(project_root)
    deleted_items += log_count

    print(f"\nCleanup complete. Deleted {deleted_items} items.")
    print("RAVANA is now ready for a fresh start.")


if __name__ == "__main__":
    main()
