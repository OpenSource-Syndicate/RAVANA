"""
Testing & Validation Framework for Snake Agent
This module provides testing and validation capabilities with rollback functionality
for the Snake Agent's autonomous changes.
"""

import os
import shutil
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import subprocess
import sys

logger = logging.getLogger(__name__)


class ChangeBackupManager:
    """
    Manages backups of files before changes are applied,
    allowing for rollback if tests fail.
    """
    
    def __init__(self, backup_dir: str = None):
        self.backup_dir = backup_dir or os.path.join(tempfile.gettempdir(), "snake_agent_backups")
        self._ensure_backup_dir()
        self.backup_manifest = {}  # Tracks file changes for potential rollback
        
    def _ensure_backup_dir(self):
        """Ensure the backup directory exists."""
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def create_backup(self, file_path: str) -> str:
        """
        Create a backup of a file before modification.
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            Path to the backup file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        # Create a unique backup filename with timestamp
        file_name = Path(file_path).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_file = os.path.join(self.backup_dir, f"{file_name}_{timestamp}.bak")
        
        # Copy the file to backup location
        shutil.copy2(file_path, backup_file)
        
        # Track this backup in the manifest
        if file_path not in self.backup_manifest:
            self.backup_manifest[file_path] = []
        
        self.backup_manifest[file_path].append({
            'backup_path': backup_file,
            'timestamp': timestamp,
            'original_path': file_path
        })
        
        logger.info(f"Created backup: {file_path} -> {backup_file}")
        return backup_file
    
    def rollback_file(self, original_path: str) -> bool:
        """
        Rollback a file to its last backup state.
        
        Args:
            original_path: Path to the original file to restore
            
        Returns:
            True if rollback was successful
        """
        if original_path not in self.backup_manifest or not self.backup_manifest[original_path]:
            logger.warning(f"No backup available for {original_path}")
            return False
        
        # Get the most recent backup
        backup_info = self.backup_manifest[original_path][-1]
        backup_path = backup_info['backup_path']
        
        try:
            # Copy the backup back to the original location
            shutil.copy2(backup_path, original_path)
            
            # Remove the backup entry from manifest
            self.backup_manifest[original_path].pop()
            
            logger.info(f"Rolled back {original_path} from {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to rollback {original_path}: {e}")
            return False
    
    def rollback_all(self) -> bool:
        """
        Rollback all changed files to their backup states.
        
        Returns:
            True if all rollbacks were successful
        """
        all_success = True
        
        for original_path in list(self.backup_manifest.keys()):
            while self.backup_manifest[original_path]:
                success = self.rollback_file(original_path)
                if not success:
                    all_success = False
        
        return all_success
    
    def cleanup_old_backups(self, days: int = 7):
        """
        Clean up backup files older than specified days.
        
        Args:
            days: Number of days to keep backups
        """
        import time
        
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        for filename in os.listdir(self.backup_dir):
            file_path = os.path.join(self.backup_dir, filename)
            if os.path.getctime(file_path) < cutoff_time:
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up old backup: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to clean up backup {file_path}: {e}")
    
    def save_manifest(self, manifest_path: str = None) -> str:
        """
        Save the backup manifest to a file for persistence.
        
        Args:
            manifest_path: Path to save manifest (if None, creates auto-named file)
            
        Returns:
            Path to the saved manifest
        """
        if manifest_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            manifest_path = os.path.join(self.backup_dir, f"manifest_{timestamp}.json")
        
        with open(manifest_path, 'w') as f:
            json.dump(self.backup_manifest, f, indent=2, default=str)
        
        logger.info(f"Saved backup manifest to {manifest_path}")
        return manifest_path
    
    def load_manifest(self, manifest_path: str):
        """
        Load a backup manifest from a file.
        
        Args:
            manifest_path: Path to the manifest file to load
        """
        with open(manifest_path, 'r') as f:
            self.backup_manifest = json.load(f)
        
        logger.info(f"Loaded backup manifest from {manifest_path}")


class TestValidator:
    """
    Validates changes by running tests and checking code quality.
    """
    
    def __init__(self):
        self.backup_manager = ChangeBackupManager()
    
    def validate_change(
        self, 
        file_path: str, 
        new_content: str, 
        test_strategy: str = "auto"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a change by applying it temporarily and running tests.
        
        Args:
            file_path: Path to the file to modify
            new_content: New content to write to the file
            test_strategy: How to run tests ('auto', 'targeted', 'all')
            
        Returns:
            Tuple of (success, validation_result)
        """
        # Create backup of original file
        backup_path = self.backup_manager.create_backup(file_path)
        
        validation_result = {
            'syntax_valid': False,
            'tests_passed': False,
            'import_valid': False,
            'overall_success': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Apply the change temporarily
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # 1. Validate syntax
            validation_result['syntax_valid'] = self._validate_syntax(file_path)
            if not validation_result['syntax_valid']:
                validation_result['errors'].append("Syntax validation failed")
                return False, validation_result
            
            # 2. Validate import (try to import the module)
            validation_result['import_valid'] = self._validate_import(file_path)
            if not validation_result['import_valid']:
                validation_result['errors'].append("Import validation failed")
                return False, validation_result
            
            # 3. Run tests based on strategy
            validation_result['tests_passed'], test_results = self._run_appropriate_tests(
                file_path, test_strategy
            )
            validation_result['test_results'] = test_results
            
            if not validation_result['tests_passed']:
                validation_result['errors'].append("Tests failed after change")
                return False, validation_result
            
            # All validations passed
            validation_result['overall_success'] = True
            return True, validation_result
            
        except Exception as e:
            validation_result['errors'].append(f"Exception during validation: {str(e)}")
            return False, validation_result
        finally:
            # Always restore the original file
            self.backup_manager.rollback_file(file_path)
    
    def _validate_syntax(self, file_path: str) -> bool:
        """
        Validate the syntax of a Python file.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if syntax is valid
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            compile(source, file_path, 'exec')
            logger.info(f"Syntax validation passed for: {file_path}")
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating syntax for {file_path}: {e}")
            return False
    
    def _validate_import(self, file_path: str) -> bool:
        """
        Validate that a Python file can be imported without errors.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if import is successful
        """
        try:
            # Get the module path relative to project root
            abs_path = os.path.abspath(file_path)
            project_root = os.path.dirname(os.path.dirname(abs_path))  # Assuming file is in core/ or similar
            
            # Add project root to Python path temporarily
            sys.path.insert(0, project_root)
            
            # Extract module name from file path
            rel_path = os.path.relpath(abs_path, project_root)
            module_name = rel_path.replace(os.sep, '.')[:-3]  # Remove .py extension
            
            # Import the module
            __import__(module_name)
            
            # Remove the project root from path
            sys.path.remove(project_root)
            
            logger.info(f"Import validation passed for: {module_name}")
            return True
        except ImportError as e:
            logger.error(f"Import error for {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error importing {file_path}: {e}")
            return False
    
    def _run_appropriate_tests(self, file_path: str, strategy: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Run appropriate tests based on the strategy.
        
        Args:
            file_path: Path to the file that was changed
            strategy: Test strategy to use ('auto', 'targeted', 'all')
            
        Returns:
            Tuple of (success, test_results)
        """
        if strategy == 'all':
            # Run all tests in the project
            return self._run_all_tests()
        elif strategy == 'targeted':
            # Run tests related to the changed file
            return self._run_targeted_tests(file_path)
        else:  # auto
            # Try targeted tests first, fall back to syntax/import validation
            success, results = self._run_targeted_tests(file_path)
            if success or results.get('total_tests', 0) > 0:
                return success, results
            else:
                # If no targeted tests were found, just do syntax and import validation
                logger.info("No targeted tests found, relying on syntax and import validation")
                return True, {
                    'passed': 0,
                    'failed': 0,
                    'total_tests': 0,
                    'test_files': []
                }
    
    def _run_targeted_tests(self, file_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Run tests that are related to the changed file.
        
        Args:
            file_path: Path to the file that was changed
            
        Returns:
            Tuple of (success, test_results)
        """
        # Identify related test files
        related_tests = self._find_related_tests(file_path)
        
        if not related_tests:
            logger.info(f"No related tests found for {file_path}")
            return True, {
                'passed': 0,
                'failed': 0,
                'total_tests': 0,
                'test_files': [],
                'message': 'No related tests found'
            }
        
        logger.info(f"Running {len(related_tests)} related tests for {file_path}")
        
        # Run the tests using subprocess to avoid import issues
        test_results = {
            'passed': 0,
            'failed': 0,
            'total_tests': len(related_tests),
            'test_files': related_tests,
            'details': []
        }
        
        all_success = True
        
        for test_file in related_tests:
            if test_file.startswith('.') or '__pycache__' in test_file:
                continue
                
            logger.info(f"Running test: {test_file}")
            
            try:
                # Use subprocess to run the test in isolation
                result = subprocess.run([
                    sys.executable, '-m', 'pytest', test_file, 
                    '-v', '--tb=short', '-x'  # Stop on first failure
                ], capture_output=True, text=True, timeout=120)
                
                test_detail = {
                    'file': test_file,
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                
                if result.returncode == 0:
                    test_results['passed'] += 1
                    logger.info(f"Test passed: {test_file}")
                else:
                    test_results['failed'] += 1
                    logger.warning(f"Test failed: {test_file}")
                    all_success = False
                
                test_results['details'].append(test_detail)
                
            except subprocess.TimeoutExpired:
                logger.error(f"Test timed out: {test_file}")
                test_results['failed'] += 1
                test_results['details'].append({
                    'file': test_file,
                    'error': 'timeout'
                })
                all_success = False
        
        return all_success, test_results
    
    def _find_related_tests(self, file_path: str) -> List[str]:
        """
        Find test files that are related to the changed file.
        
        Args:
            file_path: Path to the file that was changed
            
        Returns:
            List of related test files
        """
        # Convert module path to test path (e.g., core/config.py -> test_config.py)
        module_name = Path(file_path).stem
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(file_path)))
        
        test_candidates = [
            os.path.join(project_root, f"test_{module_name}.py"),
            os.path.join(project_root, "tests", f"test_{module_name}.py"),
            os.path.join(project_root, "test", f"test_{module_name}.py"),
            os.path.join(project_root, f"tests", f"{module_name}_test.py")
        ]
        
        # Look for test files in common test locations
        related_tests = []
        for candidate in test_candidates:
            if os.path.exists(candidate):
                # Convert to relative path from project root
                rel_path = os.path.relpath(candidate, project_root)
                related_tests.append(rel_path)
        
        return related_tests
    
    def _run_all_tests(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Run all tests in the project.
        
        Returns:
            Tuple of (success, test_results)
        """
        # This would run all tests in the project
        # For now, we'll implement a basic version
        project_root = os.getcwd()
        test_results = {
            'passed': 0,
            'failed': 0,
            'total_tests': 0,
            'test_files': [],
            'message': 'All project tests run not implemented yet'
        }
        
        logger.warning("Running all tests not fully implemented yet")
        return True, test_results


class ValidationReport:
    """
    Generates reports on validation results.
    """
    
    def __init__(self):
        pass
    
    def generate_report(self, validation_result: Dict[str, Any], change_description: str = "") -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            validation_result: The result from validation
            change_description: Description of the change being validated
            
        Returns:
            Formatted validation report string
        """
        report = f"Validation Report for: {change_description}\n"
        report += "=" * 50 + "\n"
        
        report += f"Syntax Valid: {'✓' if validation_result.get('syntax_valid') else '✗'}\n"
        report += f"Import Valid: {'✓' if validation_result.get('import_valid') else '✗'}\n"
        report += f"Tests Passed: {'✓' if validation_result.get('tests_passed') else '✗'}\n"
        report += f"Overall Success: {'✓' if validation_result.get('overall_success') else '✗'}\n"
        
        if 'test_results' in validation_result:
            test_results = validation_result['test_results']
            report += f"\nTest Results:\n"
            report += f"  Passed: {test_results.get('passed', 0)}\n"
            report += f"  Failed: {test_results.get('failed', 0)}\n"
            report += f"  Total: {test_results.get('total_tests', 0)}\n"
        
        if validation_result.get('errors'):
            report += f"\nErrors:\n"
            for error in validation_result['errors']:
                report += f"  - {error}\n"
        
        if validation_result.get('warnings'):
            report += f"\nWarnings:\n"
            for warning in validation_result['warnings']:
                report += f"  - {warning}\n"
        
        return report