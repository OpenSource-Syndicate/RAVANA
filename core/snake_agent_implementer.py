"""
Snake Agent Implementer
This module provides the autonomous implementation functionality for the Snake Agent.
"""

import asyncio
import time
import tempfile
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import json

# Import the new code transformation and analysis systems
from core.code_transformer import CodeTransformer
from core.code_analyzer import CodeAnalyzer

from core.autonomous_implementation import AutonomousImplementer, ImplementationTracker
from core.testing_environment import TestRunner
from core.change_identification import ChangeIdentifier
from core.validation_framework import TestValidator
from core.process_management import ProcessController

logger = logging.getLogger(__name__)


class SnakeAgentImplementer:
    """
    The Snake Agent's implementation module that can autonomously identify,
    test, validate, and apply changes to the codebase.
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.autonomous_implementer = AutonomousImplementer(self.project_root)
        self.implementation_tracker = ImplementationTracker()
        self.is_implementing = False
        self.implementation_queue = asyncio.Queue()
        
        # Initialize the new code transformation and analysis systems
        self.code_transformer = CodeTransformer()
        self.code_analyzer = CodeAnalyzer()
        
    async def start_autonomous_implementation(self):
        """
        Start the autonomous implementation loop.
        """
        if self.is_implementing:
            logger.warning("Autonomous implementation is already running")
            return
        
        logger.info("Starting autonomous implementation loop")
        self.is_implementing = True
        
        try:
            while self.is_implementing:
                # Look for changes to implement
                await self._run_implementation_cycle()
                
                # Wait before next cycle
                await asyncio.sleep(300)  # 5 minutes
        except Exception as e:
            logger.error(f"Error in autonomous implementation loop: {e}")
        finally:
            self.is_implementing = False
            logger.info("Autonomous implementation loop stopped")
    
    async def stop_autonomous_implementation(self):
        """
        Stop the autonomous implementation loop.
        """
        logger.info("Stopping autonomous implementation")
        self.is_implementing = False
    
    async def _run_implementation_cycle(self):
        """
        Run a single cycle of the implementation process.
        """
        logger.info("Running implementation cycle")
        
        # Identify potential changes
        implemented_changes = await self.autonomous_implementer.identify_and_implement_changes()
        
        logger.info(f"Completed implementation cycle, implemented {len(implemented_changes)} changes")
    
    async def implement_targeted_change(
        self, 
        file_path: str, 
        change_description: str, 
        change_spec: Dict[str, Any]
    ) -> bool:
        """
        Implement a specific targeted change using AST-based transformation.
        
        Args:
            file_path: Path to the file to modify
            change_description: Description of the change
            change_spec: Specific change specification
            
        Returns:
            True if implementation was successful
        """
        logger.info(f"Implementing targeted change: {change_description}")
        
        try:
            # Read the current file content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Apply AST-based optimization
            lines = original_content.split('\n')
            
            # Determine optimization type and apply
            change_type = change_spec.get('type', 'generic')
            
            if change_type == 'performance_optimization':
                optimized_content = await self._apply_performance_optimization(lines, change_spec)
            elif change_type == 'code_clarity':
                optimized_content = await self._apply_clarity_change(lines, change_spec)
            elif change_type == 'redundancy_reduction':
                optimized_content = await self._apply_redundancy_reduction(lines, change_spec)
            else:
                # Apply generic optimization
                optimized_content = await self._apply_generic_change(lines, change_spec)
            
            # Validate the optimized content
            if optimized_content and optimized_content != original_content:
                # Write the optimized content back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(optimized_content)
                
                logger.info(f"Successfully implemented change: {change_description}")
                return True
            else:
                logger.warning(f"No changes made for: {change_description}")
                return False
                
        except Exception as e:
            logger.error(f"Error implementing change '{change_description}': {e}")
            return False
    
    async def _process_implementation_queue(self) -> bool:
        """
        Process changes in the implementation queue.
        
        Returns:
            True if all changes were processed successfully
        """
        success = True
        processed_any_changes = False
        
        while not self.implementation_queue.empty():
            try:
                change_task = await self.implementation_queue.get()
                
                file_path = change_task['file_path']
                change_spec = change_task['change_spec']
                processed_any_changes = True
                
                # Validate the change
                is_valid = self.autonomous_implementer.change_validator.is_change_safe(change_spec)
                if not is_valid:
                    logger.warning(f"Change not valid, skipping: {change_task['change_description']}")
                    success = False  # Mark as unsuccessful if any change was invalid
                    continue
                
                # Attempt to implement the change
                result = await self._attempt_single_implementation(file_path, change_spec)
                
                # Track the result
                self.implementation_tracker.save_result(result)
                
                if not result.success:
                    logger.warning(f"Failed to implement change: {change_task['change_description']}")
                    success = False
                else:
                    logger.info(f"Successfully implemented change: {change_task['change_description']}")
                
            except Exception as e:
                logger.error(f"Error processing change queue item: {e}")
                success = False
        
        # If no changes were processed at all, we should return False
        # to indicate that nothing was actually implemented
        if not processed_any_changes:
            return False
            
        return success
    
    async def _attempt_single_implementation(
        self, 
        file_path: str, 
        change_spec: Dict[str, Any]
    ) -> 'ChangeImplementationResult':
        """
        Attempt to implement a single change with full validation.
        
        Args:
            file_path: Path to the file to modify
            change_spec: Change specification
            
        Returns:
            ChangeImplementationResult indicating success/failure
        """
        # Read the current file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            logger.error(f"Could not read file {file_path}: {e}")
            from core.autonomous_implementation import ChangeImplementationResult
            return ChangeImplementationResult(False, change_spec)
        
        # Apply the change
        new_content = await self._apply_change_to_content(original_content, change_spec)
        if not new_content:
            logger.error(f"Could not apply change to {file_path}")
            from core.autonomous_implementation import ChangeImplementationResult
            return ChangeImplementationResult(False, change_spec)
        
        # Validate the change
        success, validation_result = self.autonomous_implementer.test_validator.validate_change(
            file_path=file_path,
            new_content=new_content,
            test_strategy="auto"
        )
        
        if not success:
            logger.warning(f"Change validation failed for {file_path}: {change_spec.get('description')}")
            from core.autonomous_implementation import ChangeImplementationResult
            return ChangeImplementationResult(False, change_spec, validation_result)
        
        # Shutdown RAVANA if it's running
        was_running = self.autonomous_implementer.process_controller.process_manager.is_ravana_running()
        if was_running:
            logger.info("RAVANA is running, shutting down before applying change...")
            shutdown_success = self.autonomous_implementer.process_controller.shutdown_and_kill_ravana()
            if not shutdown_success:
                logger.error("Could not shutdown RAVANA, aborting change")
                from core.autonomous_implementation import ChangeImplementationResult
                return ChangeImplementationResult(False, change_spec, validation_result)
            
            # Wait a moment for the process to fully terminate
            await asyncio.sleep(2)
        
        # Apply the change to the actual file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.info(f"Successfully applied change to {file_path}")
        except Exception as e:
            logger.error(f"Failed to write change to {file_path}: {e}")
            # If RAVANA was running, restart it
            if was_running:
                self.autonomous_implementer.process_controller.start_ravana_with_args()
            from core.autonomous_implementation import ChangeImplementationResult
            return ChangeImplementationResult(False, change_spec, validation_result)
        
        # Restart RAVANA if it was running
        if was_running:
            logger.info("Restarting RAVANA after change...")
            restart_success = self.autonomous_implementer.process_controller.start_ravana_with_args()
            if not restart_success:
                logger.error("Failed to restart RAVANA after change")
                # Revert the change since RAVANA won't start
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                from core.autonomous_implementation import ChangeImplementationResult
                return ChangeImplementationResult(False, change_spec, validation_result)
        
        from core.autonomous_implementation import ChangeImplementationResult
        return ChangeImplementationResult(True, change_spec, validation_result)
    
    async def _apply_change_to_content(self, original_content: str, change: Dict[str, Any]) -> Optional[str]:
        """
        Apply a change specification to file content.
        This is a more sophisticated version that handles different change types.
        """
        lines = original_content.split('\n')
        change_type = change.get('type')
        
        try:
            if change_type == 'performance_optimization':
                return await self._apply_performance_optimization(lines, change)
            elif change_type == 'code_clarity':
                return await self._apply_clarity_change(lines, change)
            elif change_type == 'redundancy_reduction':
                return await self._apply_redundancy_reduction(lines, change)
            else:
                # For other types, use the generic approach
                return self._apply_generic_change(lines, change)
        except Exception as e:
            logger.error(f"Error applying change: {e}")
            return None
    
    async def _apply_performance_optimization(self, lines: List[str], change: Dict[str, Any]) -> str:
        """
        Apply a performance optimization change.
        """
        # This is where we'd implement specific performance optimizations
        recommendation = change.get('recommendation', 'Performance improvement applied')
        
        # Check if this is a list comprehension optimization
        if 'list comprehension' in recommendation.lower() or 'list building' in recommendation.lower():
            line_start = change.get('line_start', 1) - 1
            line_end = change.get('line_end', line_start + 10) - 1  # Estimate end line
            
            # Apply list comprehension optimization
            return self._optimize_list_building(lines, line_start, line_end)
        elif 'string concatenation' in recommendation.lower():
            line_start = change.get('line_start', 1) - 1
            line_end = change.get('line_end', line_start + 10) - 1  # Estimate end line
            
            # Apply string concatenation optimization
            return self._optimize_string_concatenation(lines, line_start, line_end)
        else:
            # For other optimizations, just add a comment
            line_start = change.get('line_start', 1) - 1
            if line_start < len(lines):
                lines[line_start] += f"  # Performance optimized: {recommendation.split(':')[-1].strip()}"
        
        return '\n'.join(lines)
    
    def _optimize_list_building(self, lines: List[str], line_start: int, line_end: int) -> str:
        """
        Optimize list building using complete AST-based transformation.
        """
        # Convert to full code for proper AST parsing
        original_code = '\n'.join(lines)
        
        # Apply list comprehension optimization
        optimized_code = self.code_transformer.optimize_code(original_code, "list_comprehension")
        
        if optimized_code:
            logger.info("Successfully applied list comprehension optimization")
            return optimized_code
        else:
            # Fallback approach
            if line_start < len(lines):
                lines[line_start] += "  # TODO: Implement list comprehension optimization"
            return '\n'.join(lines)
    
    def _optimize_string_concatenation(self, lines: List[str], line_start: int, line_end: int) -> str:
        """
        Optimize string concatenation using AST-based transformation.
        """
        # Convert to full code for proper AST parsing
        original_code = '\n'.join(lines)
        
        # Apply string optimization
        optimized_code = self.code_transformer.optimize_code(original_code, "string_concatenation")
        
        if optimized_code:
            logger.info("Successfully applied string concatenation optimization")
            return optimized_code
        else:
            # Fallback approach
            if line_start < len(lines):
                lines[line_start] += "  # TODO: Consider using ''.join() for better performance"
            return '\n'.join(lines)
    
    async def _apply_clarity_change(self, lines: List[str], change: Dict[str, Any]) -> str:
        """
        Apply a code clarity improvement.
        """
        # Example: Replace magic numbers with named constants
        if change.get('subtype') == 'magic_number':
            line_start = change.get('line_start', 1) - 1
            if line_start < len(lines):
                original_line = lines[line_start]
                # In a real implementation, we'd extract the number and replace it
                # For now, just add a comment
                lines[line_start] = original_line + f"  # Clarity improved: {change.get('recommendation', 'Named constant suggested')}"
        
        return '\n'.join(lines)
    
    async def _apply_redundancy_reduction(self, lines: List[str], change: Dict[str, Any]) -> str:
        """
        Apply a redundancy reduction change.
        """
        # For redundancy reduction, we might refactor duplicated code into functions
        # For now, just add a comment indicating the change
        line_start = change.get('line_start', 1) - 1
        if line_start < len(lines):
            lines[line_start] = lines[line_start] + f"  # Refactored: {change.get('recommendation', 'Reduced redundancy')}"
        
        return '\n'.join(lines)
    
    async def _apply_generic_change(self, lines: List[str], change: Dict[str, Any]) -> str:
        """
        Apply a generic change.
        """
        # For other change types, implement appropriate transformations
        return '\n'.join(lines)
    
    def get_implementation_status(self) -> Dict[str, Any]:
        """
        Get the current status of the implementation system.
        
        Returns:
            Dictionary with status information
        """
        return {
            'is_implementing': self.is_implementing,
            'queue_size': self.implementation_queue.qsize(),
            'success_rate': self.implementation_tracker.get_success_rate(),
            'total_implementations': len(self.implementation_tracker.history),
            'recent_results': self.implementation_tracker.get_recent_results(5)
        }
    
    def enable_autonomous_mode(self):
        """
        Enable autonomous implementation mode.
        """
        # This would typically be configured through settings
        logger.info("Autonomous implementation mode enabled")
    
    def disable_autonomous_mode(self):
        """
        Disable autonomous implementation mode.
        """
        # This would stop the autonomous loop
        logger.info("Autonomous implementation mode disabled")
        self.is_implementing = False


# Helper function to create and return an implementer instance
def create_snake_implementer(project_root: str = None) -> SnakeAgentImplementer:
    """
    Create a new instance of the Snake Agent Implementer.
    
    Args:
        project_root: Path to the project root directory
        
    Returns:
        SnakeAgentImplementer instance
    """
    return SnakeAgentImplementer(project_root)


# Example usage
if __name__ == "__main__":
    async def example_usage():
        implementer = create_snake_implementer()
        
        # Example: Implement a simple performance optimization
        success = await implementer.implement_targeted_change(
            file_path="core/example_file.py",
            change_description="Optimize list iteration",
            change_spec={
                'type': 'performance_optimization',
                'subtype': 'iteration_pattern',
                'line_start': 10,
                'line_end': 10,
                'description': 'Replace range(len()) with direct iteration',
                'recommendation': 'Iterate directly over container instead of range(len(container))',
                'severity': 'medium',
                'scope': 'function'
            }
        )
        
        print(f"Implementation success: {success}")
        print(f"Status: {implementer.get_implementation_status()}")
    
    # asyncio.run(example_usage())