"""
Testing Environment for Snake Agent
This module provides an isolated environment for the Snake Agent to test changes
before implementing them in the main codebase.
"""

import os
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import importlib.util
import asyncio
import time

logger = logging.getLogger(__name__)


class TestingEnvironment:
    """
    An isolated testing environment that allows the Snake Agent to test
    changes before implementing them in the main codebase.
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getcwd()
        self.temp_dir = None
        self._original_files = {}
        
    def create_isolated_environment(self) -> str:
        """
        Create an isolated environment with a copy of the project.
        
        Returns:
            str: Path to the temporary directory
        """
        self.temp_dir = tempfile.mkdtemp(prefix="snake_agent_test_")
        
        # Copy the entire project to the temp directory
        project_name = os.path.basename(self.project_root)
        target_dir = os.path.join(self.temp_dir, project_name)
        
        # Use copytree to copy the project
        try:
            shutil.copytree(
                self.project_root,
                target_dir,
                ignore=shutil.ignore_patterns(
                    '.git', '__pycache__', '*.pyc', '.pytest_cache',
                    'venv', '.venv', 'node_modules', '.idea', '.vscode',
                    'snake_logs', 'ravana_agi.log', 'test_*.py'
                )
            )
        except Exception as e:
            logger.error(f"Failed to copy project to test environment: {e}")
            raise
        
        logger.info(f"Created isolated testing environment at: {target_dir}")
        return target_dir
    
    def apply_change(self, file_path: str, new_content: str) -> bool:
        """
        Apply a change to a file in the temporary environment.
        
        Args:
            file_path: Path to the file to modify (relative to project root)
            new_content: The new content to write to the file
            
        Returns:
            bool: True if the change was applied successfully
        """
        if not self.temp_dir:
            logger.error("Testing environment not created yet")
            return False
            
        project_name = os.path.basename(self.project_root)
        full_file_path = os.path.join(self.temp_dir, project_name, file_path)
        
        try:
            # Backup the original file if not already backed up
            if full_file_path not in self._original_files:
                with open(full_file_path, 'r', encoding='utf-8') as f:
                    self._original_files[full_file_path] = f.read()
            
            # Write the new content
            with open(full_file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.info(f"Applied change to: {full_file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to apply change to {full_file_path}: {e}")
            # Restore original file if write failed
            self._restore_file(full_file_path)
            return False
    
    def run_tests(self, test_files: List[str] = None) -> Dict[str, Any]:
        """
        Run tests in the isolated environment.
        
        Args:
            test_files: List of test files to run (if None, run all tests)
            
        Returns:
            Dict with test results
        """
        if not self.temp_dir:
            logger.error("Testing environment not created yet")
            return {"success": False, "error": "No testing environment"}
        
        project_name = os.path.basename(self.project_root)
        test_dir = os.path.join(self.temp_dir, project_name)
        
        try:
            # Change to the test directory
            original_cwd = os.getcwd()
            os.chdir(test_dir)
            
            # Find all test files if none specified
            if not test_files:
                test_files = []
                for root, dirs, files in os.walk('.'):
                    for file in files:
                        if file.startswith('test_') and file.endswith('.py'):
                            test_files.append(os.path.relpath(os.path.join(root, file), '.'))
            
            # Run tests using pytest if available, otherwise try direct execution
            test_results = {"passed": [], "failed": [], "errors": []}
            
            for test_file in test_files:
                if test_file.startswith('.') or '__pycache__' in test_file:
                    continue
                    
                logger.info(f"Running test: {test_file}")
                
                # Use subprocess to run the test in isolation
                result = subprocess.run([
                    sys.executable, '-m', 'pytest', test_file, 
                    '-v', '--tb=short'
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    test_results["passed"].append(test_file)
                    logger.info(f"Test passed: {test_file}")
                else:
                    test_results["failed"].append(test_file)
                    logger.warning(f"Test failed: {test_file}")
                    logger.warning(f"Stdout: {result.stdout}")
                    logger.warning(f"Stderr: {result.stderr}")
            
            # Calculate success based on whether any tests failed
            success = len(test_results["failed"]) == 0 and len(test_results["errors"]) == 0
            
            return {
                "success": success,
                "test_results": test_results,
                "total_tests": len(test_files),
                "passed": len(test_results["passed"]),
                "failed": len(test_results["failed"])
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Tests timed out")
            return {"success": False, "error": "Tests timed out"}
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return {"success": False, "error": str(e)}
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
    
    def validate_syntax(self, file_path: str) -> bool:
        """
        Validate the syntax of a Python file.
        
        Args:
            file_path: Path to the file to validate (relative to project root)
            
        Returns:
            bool: True if syntax is valid
        """
        if not self.temp_dir:
            logger.error("Testing environment not created yet")
            return False
        
        project_name = os.path.basename(self.project_root)
        full_file_path = os.path.join(self.temp_dir, project_name, file_path)
        
        try:
            with open(full_file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            # This will raise an exception if syntax is invalid
            compile(source, full_file_path, 'exec')
            logger.info(f"Syntax validation passed for: {full_file_path}")
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in {full_file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error validating syntax for {full_file_path}: {e}")
            return False
    
    def _restore_file(self, file_path: str):
        """Restore a file to its original state."""
        if file_path in self._original_files:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self._original_files[file_path])
            # Remove from backup dict
            del self._original_files[file_path]
    
    def cleanup(self):
        """Clean up the temporary environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up testing environment: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Failed to clean up testing environment: {e}")
            finally:
                self.temp_dir = None


class TestRunner:
    """
    A class to run specific tests related to changes made by the Snake Agent.
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getcwd()
        self.testing_env = TestingEnvironment(self.project_root)
    
    async def run_targeted_tests(self, changed_file: str) -> Dict[str, Any]:
        """
        Run tests that are related to the changed file.
        
        Args:
            changed_file: The file that was changed (relative path)
            
        Returns:
            Dict with test results
        """
        # Create isolated environment
        test_dir = self.testing_env.create_isolated_environment()
        
        try:
            # Identify related test files
            related_tests = await self._find_related_tests(changed_file)
            
            # Run the tests
            results = self.testing_env.run_tests(related_tests)
            
            # Return results
            return results
        finally:
            # Clean up the testing environment
            self.testing_env.cleanup()
    
    async def _find_related_tests(self, changed_file: str) -> List[str]:
        """
        Find test files that are related to the changed file.
        
        Args:
            changed_file: The file that was changed (relative path)
            
        Returns:
            List of related test files
        """
        # Convert module path to test path (e.g., core/config.py -> test_config.py)
        module_name = Path(changed_file).stem
        test_candidates = [
            f"test_{module_name}.py",
            f"tests/test_{module_name}.py",
            f"test/test_{module_name}.py",
            f"tests/{module_name}_test.py"
        ]
        
        # Look for test files in common test locations
        test_files = []
        for candidate in test_candidates:
            test_path = os.path.join(self.project_root, candidate)
            if os.path.exists(test_path):
                test_files.append(candidate)
        
        # Also look for any test files that might import the changed module
        for root, dirs, files in os.walk(os.path.join(self.project_root, 'tests')):
            for file in files:
                if file.startswith('test_') and file.endswith('.py'):
                    test_file_path = os.path.relpath(os.path.join(root, file), self.project_root)
                    if self._could_be_related(test_file_path, changed_file):
                        test_files.append(test_file_path)
        
        return test_files
    
    def _could_be_related(self, test_file: str, changed_file: str) -> bool:
        """
        Check if a test file could be related to the changed file.
        
        Args:
            test_file: Path to the test file
            changed_file: Path to the changed file
            
        Returns:
            True if the files could be related
        """
        # Get module name without path and extension
        module_name = Path(changed_file).stem
        
        try:
            with open(os.path.join(self.project_root, test_file), 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check if the test file imports the changed module
            import_patterns = [
                f"from {module_name}",
                f"import {module_name}",
                f"from .{module_name}",
                f"from ..{module_name}"
            ]
            
            return any(pattern in content for pattern in import_patterns)
        except Exception:
            return False