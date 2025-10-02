#!/usr/bin/env python3
"""Comprehensive test runner for RAVANA AGI System"""

import sys
import os
import subprocess
import time
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Comprehensive test runner with reporting"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_suites": {},
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
                "duration": 0
            }
        }
    
    def run_test_suite(self, name, test_path, markers=None, timeout=300):
        """Run a specific test suite"""
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}\n")
        
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_path,
            "-v",
            "--tb=short",
            "--json-report",
            f"--json-report-file=test_results_{name.replace(' ', '_').lower()}.json"
        ]
        
        if markers:
            cmd.extend(["-m", markers])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            # Parse results
            self.results["test_suites"][name] = {
                "status": "passed" if result.returncode == 0 else "failed",
                "return_code": result.returncode,
                "duration": duration,
                "stdout": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
                "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
            }
            
            # Load JSON report if available
            json_file = project_root / f"test_results_{name.replace(' ', '_').lower()}.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                        summary = json_data.get('summary', {})
                        self.results["summary"]["total_tests"] += summary.get('total', 0)
                        self.results["summary"]["passed"] += summary.get('passed', 0)
                        self.results["summary"]["failed"] += summary.get('failed', 0)
                except Exception as e:
                    print(f"Warning: Could not parse JSON report: {e}")
            
            self.results["summary"]["duration"] += duration
            
            if result.returncode == 0:
                print(f"\n✓ {name} PASSED ({duration:.2f}s)")
            else:
                print(f"\n✗ {name} FAILED ({duration:.2f}s)")
                print(f"\nError output:\n{result.stderr}")
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"\n⚠ {name} TIMEOUT after {duration:.2f}s")
            self.results["test_suites"][name] = {
                "status": "timeout",
                "return_code": -1,
                "duration": duration
            }
            self.results["summary"]["errors"] += 1
            return False
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"\n✗ {name} ERROR: {e}")
            self.results["test_suites"][name] = {
                "status": "error",
                "return_code": -1,
                "duration": duration,
                "error": str(e)
            }
            self.results["summary"]["errors"] += 1
            return False
    
    def generate_report(self, output_file="test_report.json"):
        """Generate comprehensive test report"""
        report_path = project_root / output_file
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n{'='*60}")
        print("TEST REPORT GENERATED")
        print(f"{'='*60}")
        print(f"Report saved to: {report_path}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        summary = self.results["summary"]
        
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests:    {summary['total_tests']}")
        print(f"Passed:         {summary['passed']} ✓")
        print(f"Failed:         {summary['failed']} ✗")
        print(f"Errors:         {summary['errors']} ⚠")
        print(f"Total Duration: {summary['duration']:.2f}s")
        print(f"{'='*60}\n")
        
        # Print suite-by-suite results
        print("\nTest Suite Results:")
        print("-" * 60)
        for name, result in self.results["test_suites"].items():
            status_symbol = "✓" if result["status"] == "passed" else "✗" if result["status"] == "failed" else "⚠"
            print(f"{status_symbol} {name:<40} {result['duration']:>6.2f}s")
        print("-" * 60)


def main():
    """Main test runner"""
    print("\n" + "="*60)
    print("RAVANA AGI COMPREHENSIVE TEST SUITE")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    runner = TestRunner()
    
    # Define test suites
    test_suites = [
        {
            "name": "Unit Tests - Core",
            "path": "tests/core",
            "markers": "unit",
            "timeout": 120
        },
        {
            "name": "Unit Tests - Services",
            "path": "tests/services",
            "markers": "unit",
            "timeout": 120
        },
        {
            "name": "Unit Tests - Modules",
            "path": "tests/modules",
            "markers": "unit",
            "timeout": 120
        },
        {
            "name": "Integration Tests - System",
            "path": "tests/integration/test_system_integration.py",
            "markers": "integration",
            "timeout": 300
        },
        {
            "name": "Integration Tests - Modules",
            "path": "tests/integration/test_modules_integration.py",
            "markers": "integration",
            "timeout": 300
        },
        {
            "name": "End-to-End Tests",
            "path": "tests/test_system_e2e.py",
            "markers": "e2e",
            "timeout": 600
        }
    ]
    
    # Run test suites
    all_passed = True
    for suite in test_suites:
        passed = runner.run_test_suite(
            suite["name"],
            suite["path"],
            suite.get("markers"),
            suite.get("timeout", 300)
        )
        all_passed = all_passed and passed
        
        # Small delay between suites
        time.sleep(2)
    
    # Generate report
    runner.generate_report()
    
    # Return appropriate exit code
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
