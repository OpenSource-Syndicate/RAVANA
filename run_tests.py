#!/usr/bin/env python3
"""Test runner for RAVANA AGI system."""

import sys
import subprocess
import os
from pathlib import Path


def main():
    """Run the complete test suite."""
    print("="*60)
    print("RAVANA AGI - Comprehensive Test Suite")
    print("="*60)
    print()
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Run pytest with coverage
    cmd = [
        "pytest",
        "tests/",
        "-v",                    # Verbose output
        "--tb=short",            # Short traceback format
        "--cov=core",            # Coverage for core
        "--cov=services",        # Coverage for services
        "--cov=modules",         # Coverage for modules
        "--cov-report=term-missing",  # Show missing lines
        "--cov-report=html",     # HTML report
        "--asyncio-mode=auto",   # Auto-detect asyncio tests
        "-n", "auto",            # Run tests in parallel
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=False)
        
        print()
        print("="*60)
        if result.returncode == 0:
            print("✅ All tests passed!")
            print("📊 Coverage report: htmlcov/index.html")
        else:
            print("❌ Some tests failed!")
            print(f"Exit code: {result.returncode}")
        print("="*60)
        
        return result.returncode
        
    except FileNotFoundError:
        print("❌ Error: pytest not found!")
        print("Please install test dependencies:")
        print("  uv pip install -e '.[dev]'")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
