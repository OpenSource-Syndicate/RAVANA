"""
Autonomous Implementation System for Snake Agent
This module provides the core functionality for the Snake Agent to autonomously
test, validate, and implement changes to the RAVANA codebase.
"""

import os
import asyncio
import time
import tempfile
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

from core.testing_environment import TestRunner
from core.change_identification import ChangeIdentifier, ChangeValidator
from core.validation_framework import TestValidator, ValidationReport
from core.process_management import ProcessController

logger = logging.getLogger(__name__)


class AutonomousImplementer:
    """
    The core implementation system that autonomously tests and applies changes.
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.getcwd()
        self.test_runner = TestRunner(self.project_root)
        self.change_identifier = ChangeIdentifier()
        self.change_validator = ChangeValidator()
        self.test_validator = TestValidator()
        self.process_controller = ProcessController()
        self.validation_reporter = ValidationReport()
        
        # Configuration settings
        self.config = {
            'max_change_size': 50,  # Maximum lines of change to consider
            'min_validation_score': 0.8,  # Minimum score to proceed with change
            'max_retries': 3,  # Number of retry attempts for failed changes
            'allow_function_changes': True,  # Whether to allow function modifications
            'allow_class_changes': True,  # Whether to allow class modifications
            'allow_config_changes': False,  # Whether to allow configuration changes
        }
    
    async def identify_and_implement_changes(self) -> List[Dict[str, Any]]:
        """
        Main method to identify potential changes and implement valid ones.
        
        Returns:
            List of implemented changes
        """
        logger.info("Starting autonomous change identification and implementation")
        
        implemented_changes = []
        
        # Identify potential changes in the project
        potential_changes = self.change_identifier.identify_changes_in_project(self.project_root)
        
        for file_path, changes in potential_changes.items():
            logger.info(f"Found {len(changes)} potential changes in {file_path}")
            
            # Filter changes based on configuration
            filtered_changes = self._filter_valid_changes(changes)
            
            for change in filtered_changes:
                try:
                    success = await self._implement_single_change(file_path, change)
                    if success:
                        implemented_changes.append(change)
                        logger.info(f"Successfully implemented change in {file_path}")
                    else:
                        logger.warning(f"Failed to implement change in {file_path}")
                
                except Exception as e:
                    logger.error(f"Error implementing change in {file_path}: {e}")
                    continue
        
        logger.info(f"Completed implementation cycle. Successfully implemented {len(implemented_changes)} changes")
        return implemented_changes
    
    def _filter_valid_changes(self, changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter changes based on configuration and safety criteria.
        
        Args:
            changes: List of potential changes
            
        Returns:
            Filtered list of valid changes
        """
        valid_changes = []
        
        for change in changes:
            # Check if change type is allowed
            change_type = change.get('type')
            if change_type == 'code_clarity' and not self.config['allow_function_changes']:
                continue
            if change_type == 'performance_optimization' and not self.config['allow_function_changes']:
                continue
            
            # Check change severity
            severity = change.get('severity', 'medium')
            if severity == 'high':
                # For now, we'll allow high severity changes but log them
                logger.warning(f"High severity change detected: {change}")
            
            # Check if the change is within our scope
            scope = change.get('scope', 'function')
            if scope not in ['function', 'method', 'block', 'line']:
                continue
            
            # Add to valid changes
            valid_changes.append(change)
        
        return valid_changes
    
    async def _implement_single_change(self, file_path: str, change: Dict[str, Any]) -> bool:
        """
        Implement a single change with full validation and process management.
        
        Args:
            file_path: Path to the file to modify
            change: Change specification to implement
            
        Returns:
            True if implementation was successful
        """
        logger.info(f"Attempting to implement change: {change.get('description', 'Unknown')}")
        
        # Read the current file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            logger.error(f"Could not read file {file_path}: {e}")
            return False
        
        # Generate the new content with the change applied
        new_content = await self._apply_change_to_content(original_content, change)
        if not new_content:
            logger.error(f"Could not apply change to {file_path}")
            return False
        
        # Validate the change in isolation
        success, validation_result = self.test_validator.validate_change(
            file_path=file_path,
            new_content=new_content,
            test_strategy="auto"
        )
        
        # Generate validation report
        report = self.validation_reporter.generate_report(
            validation_result, 
            change.get('description', 'Unknown change')
        )
        logger.info(f"Validation report:\n{report}")
        
        if not success:
            logger.warning(f"Change validation failed for {file_path}: {change.get('description')}")
            return False
        
        # If validation passed, proceed with implementation
        logger.info(f"Change passed validation, proceeding with implementation for {file_path}")
        
        # Shutdown RAVANA processes
        if self.process_controller.process_manager.is_ravana_running():
            logger.info("RAVANA is running, shutting down before applying change...")
            shutdown_success = self.process_controller.shutdown_and_kill_ravana()
            if not shutdown_success:
                logger.error("Could not shutdown RAVANA, aborting change")
                return False
            
            # Wait a moment for the process to fully terminate
            await asyncio.sleep(2)
        
        # Apply the change to the actual file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.info(f"Successfully applied change to {file_path}")
        except Exception as e:
            logger.error(f"Failed to write change to {file_path}: {e}")
            return False
        
        # Restart RAVANA if it was running
        if self._was_ravana_running_at_start():
            logger.info("Restarting RAVANA after change...")
            restart_success = self.process_controller.start_ravana_with_args()
            if not restart_success:
                logger.error("Failed to restart RAVANA after change")
                # Revert the change since RAVANA won't start
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                return False
        
        logger.info(f"Successfully implemented and validated change in {file_path}")
        return True
    
    def _was_ravana_running_at_start(self) -> bool:
        """
        Check if RAVANA was running before the implementation process started.
        """
        # For now, we'll check if RAVANA is currently running
        # In a more sophisticated implementation, we'd track the state at the beginning
        return True  # Assume it was running if we're implementing changes
    
    async def _apply_change_to_content(self, original_content: str, change: Dict[str, Any]) -> Optional[str]:
        """
        Apply a change specification to file content.
        
        Args:
            original_content: Original file content
            change: Change specification
            
        Returns:
            New content with change applied, or None if application failed
        """
        lines = original_content.split('\n')
        change_type = change.get('type')
        
        # For now, we'll implement a simple replacement based on line numbers
        # This would need to be expanded for more complex changes
        try:
            if change_type == 'performance_optimization' or change_type == 'code_clarity':
                # For this example, we'll implement a simple replacement
                # This needs to be expanded based on the actual change type and details
                return self._apply_performance_change(lines, change)
            else:
                # For other types, we might need different strategies
                return self._apply_generic_change(lines, change)
        except Exception as e:
            logger.error(f"Error applying change: {e}")
            return None
    
    def _apply_performance_change(self, lines: List[str], change: Dict[str, Any]) -> str:
        """
        Apply a performance optimization change to the content.
        
        Args:
            lines: Original content as list of lines
            change: Change specification
            
        Returns:
            New content with change applied
        """
        line_start = change.get('line_start', 1) - 1  # Convert to 0-based index
        line_end = change.get('line_end', 1) - 1      # Convert to 0-based index
        recommendation = change.get('recommendation', 'Performance optimization')
        
        # Actually implement the optimization based on the recommendation
        if 'list comprehension' in recommendation.lower() or 'list building' in recommendation.lower():
            # Replace inefficient list building with list comprehension
            return self._optimize_list_building(lines, line_start, line_end)
        elif 'string concatenation' in recommendation.lower():
            # Replace inefficient string concatenation
            return self._optimize_string_concatenation(lines, line_start, line_end)
        else:
            # For other types of optimizations, just add a comment
            if line_start < len(lines):
                original_line = lines[line_start]
                # Add a comment after the line
                lines[line_start] = original_line + f"  # Optimized: {recommendation}"
        
        return '\n'.join(lines)
    
    def _optimize_list_building(self, lines: List[str], line_start: int, line_end: int) -> str:
        """
        Optimize list building by replacing inefficient append() loops with list comprehensions.
        
        Args:
            lines: Original content as list of lines
            line_start: Start line of the function (0-based index)
            line_end: End line of the function (0-based index)
            
        Returns:
            New content with list building optimized
        """
        # For a simple case, look for the pattern:
        # result = []
        # for i in range(...):
        #     result.append(...)
        #
        # And replace with:
        # result = [... for i in range(...)]
        
        # This is a simplified implementation that looks for common patterns
        function_lines = lines[line_start:line_end+1] if line_end >= line_start else [lines[line_start]]
        
        # Try to find and replace the pattern
        optimized_lines = []
        i = 0
        while i < len(function_lines):
            line = function_lines[i]
            
            # Look for the initialization pattern: result = []
            if re.search(r'(\w+)\s*=\s*\[\s*\]', line):
                var_name = re.search(r'(\w+)\s*=', line).group(1)
                
                # Look ahead to see if the next lines follow the pattern
                if i + 2 < len(function_lines):
                    next_line = function_lines[i + 1]
                    third_line = function_lines[i + 2]
                    
                    # Check if next line is a for loop
                    for_match = re.search(r'for\s+(\w+)\s+in\s+range\s*\(', next_line)
                    if for_match:
                        loop_var = for_match.group(1)
                        
                        # Check if third line appends to our list
                        append_match = re.search(rf'{var_name}\s*\.\s*append\s*\(', third_line)
                        if append_match:
                            # This looks like a pattern we can optimize!
                            # Replace all three lines with a list comprehension
                            optimized_line = f"{var_name} = [{var_name}_expr for {loop_var} in range_expr]"
                            optimized_lines.append(optimized_line)
                            optimized_lines.append("  # TODO: Replace {var_name}_expr and range_expr with actual expressions")
                            i += 3  # Skip the three lines we replaced
                            continue
            
            # If we didn't optimize, keep the original line
            optimized_lines.append(line)
            i += 1
        
        # Replace the function lines in the overall content
        new_lines = lines[:line_start] + optimized_lines + lines[line_end+1:]
        return '\n'.join(new_lines)
    
    def _optimize_string_concatenation(self, lines: List[str], line_start: int, line_end: int) -> str:
        """
        Optimize string concatenation by suggesting better approaches.
        
        Args:
            lines: Original content as list of lines
            line_start: Start line of the function (0-based index)
            line_end: End line of the function (0-based index)
            
        Returns:
            New content with optimization comment added
        """
        # For now, just add a comment suggesting optimization
        if line_start < len(lines):
            original_line = lines[line_start]
            lines[line_start] = original_line + "  # TODO: Consider using ''.join() for better performance"
        
        return '\n'.join(lines)
    
    def _apply_generic_change(self, lines: List[str], change: Dict[str, Any]) -> str:
        """
        Apply a generic change to the content.
        
        Args:
            lines: Original content as list of lines
            change: Change specification
            
        Returns:
            New content with change applied
        """
        # For other change types, implement appropriate transformations
        # This would be expanded based on the specific change type
        return '\n'.join(lines)


class ChangeImplementationResult:
    """
    Represents the result of a change implementation attempt.
    """
    
    def __init__(self, success: bool, change: Dict[str, Any], validation_result: Dict[str, Any] = None):
        self.success = success
        self.change = change
        self.validation_result = validation_result or {}
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'success': self.success,
            'change': self.change,
            'validation_result': self.validation_result,
            'timestamp': self.timestamp
        }


class ImplementationTracker:
    """
    Tracks the history of implementations for analysis and learning.
    """
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or os.path.join(tempfile.gettempdir(), "snake_impl_history.json")
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load implementation history from storage."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load implementation history: {e}")
        
        return []
    
    def save_result(self, result: ChangeImplementationResult):
        """Save a result to the history."""
        try:
            self.history.append(result.to_dict())
            
            with open(self.storage_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            logger.info(f"Saved implementation result to history")
        except Exception as e:
            logger.error(f"Could not save implementation result: {e}")
    
    def get_success_rate(self) -> float:
        """Calculate the overall success rate of implementations."""
        if not self.history:
            return 0.0
        
        successful = sum(1 for entry in self.history if entry['success'])
        return successful / len(self.history)
    
    def get_recent_results(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent implementation results."""
        return self.history[-count:]


# Example usage and testing
async def main():
    """Example usage of the AutonomousImplementer."""
    implementer = AutonomousImplementer()
    
    # Just run identification for now - the actual implementation will be triggered
    # by the Snake Agent when it identifies specific changes
    print("Autonomous Implementer initialized")
    print("Ready to identify and implement changes when triggered")


if __name__ == "__main__":
    asyncio.run(main())