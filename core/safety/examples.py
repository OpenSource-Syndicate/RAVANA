"""
Safety Examples for RAVANA AGI System

This module provides examples of how to use the safety mechanisms.
"""

import asyncio
import logging
from typing import Dict, Any

from core.safety.decorators import safety_checked, privacy_protected, emergency_shutdown_on_violations
from core.safety.safety_manager import (
    initialize_safety_manager, SafetyLevel, SafetyViolationType,
    check_action_safety, monitor_execution
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# Example 1: Basic safety-checked function
@safety_checked(
    component="example_module",
    action="process_user_data",
    required_params=["user_id", "data"],
    safety_level=SafetyLevel.MEDIUM,
    block_on_violation=True
)
async def process_user_data(user_id: str, data: Dict[str, Any]) -> str:
    """
    Process user data with safety checks.
    
    Args:
        user_id: User identifier
        data: User data to process
        
    Returns:
        Processing result
    """
    logger.info(f"Processing data for user {user_id}")
    
    # Simulate data processing
    await asyncio.sleep(0.1)  # Simulate async work
    
    # Example processing that might raise privacy concerns
    if "ssn" in str(data).lower():
        logger.warning("SSN detected in user data - this should be flagged by safety checks")
    
    return f"Processed data for user {user_id}: {len(str(data))} characters"


# Example 2: Privacy-protected function
@privacy_protected(
    component="data_service",
    pii_handling="redact",
    data_retention_days=30
)
async def store_user_profile(user_id: str, profile_data: Dict[str, Any]) -> bool:
    """
    Store user profile with privacy protection.
    
    Args:
        user_id: User identifier
        profile_data: Profile data to store
        
    Returns:
        True if stored successfully
    """
    logger.info(f"Storing profile for user {user_id}")
    
    # Simulate profile storage
    await asyncio.sleep(0.1)  # Simulate async work
    
    # Log that sensitive data should be protected
    logger.info("Sensitive data in profile should be redacted by privacy protection")
    
    return True


# Example 3: Function with emergency shutdown protection
@emergency_shutdown_on_violations(max_violations=5)
@safety_checked(
    component="critical_service",
    action="critical_operation",
    safety_level=SafetyLevel.HIGH,
    block_on_violation=True
)
async def critical_operation(operation_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform a critical operation with emergency shutdown protection.
    
    Args:
        operation_id: Operation identifier
        parameters: Operation parameters
        
    Returns:
        Operation result
    """
    logger.info(f"Performing critical operation {operation_id}")
    
    # Simulate critical operation
    await asyncio.sleep(0.2)  # Simulate async work
    
    # Example operation that might trigger safety violations
    if "dangerous" in str(parameters).lower():
        logger.warning("Dangerous operation detected - safety system should flag this")
        # In a real implementation, this would be caught by safety checks
    
    result = {
        "operation_id": operation_id,
        "status": "completed",
        "result": f"Operation {operation_id} completed successfully",
        "parameters_processed": len(parameters)
    }
    
    return result


# Example 4: Manual safety checking
async def manually_checked_function(component: str, action: str, **kwargs) -> str:
    """
    Function that performs manual safety checking.
    
    Args:
        component: Component performing the action
        action: Action being performed
        **kwargs: Action parameters
        
    Returns:
        Function result
    """
    # Perform manual safety check
    safety_result = check_action_safety(
        component=component,
        action=action,
        params=kwargs,
        context={"manual_check": True, "caller": "manual_example"}
    )
    
    # Handle safety violations
    if not safety_result.get("approved", True):
        violations = safety_result.get("violations", [])
        logger.warning(f"Manual safety check failed: {len(violations)} violations")
        raise PermissionError(f"Safety check failed: {len(violations)} violations")
    
    # Proceed with function execution
    logger.info(f"Manual safety check passed for {component}.{action}")
    
    # Simulate work
    await asyncio.sleep(0.1)  # Simulate async work
    
    result = f"Manual check successful for {component}.{action}"
    
    # Monitor execution
    monitor_execution(
        component=component,
        action=action,
        params=kwargs,
        execution_result={"success": True, "result": result}
    )
    
    return result


# Example 5: Component with integrated safety monitoring
class SafetyAwareComponent:
    """Example component with integrated safety monitoring."""
    
    def __init__(self, name: str):
        self.name = name
        self.operations_count = 0
        self.violations_count = 0
        
    @safety_checked(
        component="safety_aware_component",
        safety_level=SafetyLevel.MEDIUM,
        block_on_violation=True
    )
    async def perform_safe_operation(self, operation: str, data: Dict[str, Any]) -> str:
        """
        Perform a safe operation with integrated monitoring.
        
        Args:
            operation: Operation to perform
            data: Operation data
            
        Returns:
            Operation result
        """
        self.operations_count += 1
        logger.info(f"[{self.name}] Performing safe operation: {operation}")
        
        # Simulate work
        await asyncio.sleep(0.1)  # Simulate async work
        
        # Simulate potential safety violations
        if "unsafe" in operation.lower():
            self.violations_count += 1
            logger.warning(f"[{self.name}] Unsafe operation detected: {operation}")
            # In a real implementation, safety system would catch this
        
        result = f"[{self.name}] Operation '{operation}' completed successfully"
        return result
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get component status with safety metrics."""
        return {
            "name": self.name,
            "operations_count": self.operations_count,
            "violations_count": self.violations_count,
            "safety_compliance": (self.operations_count - self.violations_count) / max(1, self.operations_count),
            "healthy": self.violations_count < 3  # Healthy if less than 3 violations
        }


# Demonstration function
async def demonstrate_safety_mechanisms():
    """Demonstrate various safety mechanisms."""
    logger.info("=== Safety Mechanisms Demonstration ===")
    
    # Initialize safety manager
    logger.info("\n1. Initializing safety manager...")
    # In a real implementation, this would be done during system startup
    # For this example, we'll simulate it
    logger.info("Safety manager initialized (simulated)")
    
    # Example 1: Basic safety-checked function
    logger.info("\n2. Testing basic safety-checked function...")
    try:
        result = await process_user_data(
            user_id="user123",
            data={"name": "John Doe", "age": 30}
        )
        logger.info(f"Result: {result}")
    except Exception as e:
        logger.error(f"Error in safety-checked function: {e}")
    
    # Example 2: Privacy-protected function
    logger.info("\n3. Testing privacy-protected function...")
    try:
        result = await store_user_profile(
            user_id="user456",
            profile_data={"name": "Jane Smith", "email": "jane@example.com"}
        )
        logger.info(f"Privacy-protected result: {result}")
    except Exception as e:
        logger.error(f"Error in privacy-protected function: {e}")
    
    # Example 3: Critical operation with emergency shutdown
    logger.info("\n4. Testing critical operation with emergency shutdown...")
    try:
        result = await critical_operation(
            operation_id="op789",
            parameters={"priority": "high", "data": "sensitive_info"}
        )
        logger.info(f"Critical operation result: {result}")
    except SystemExit:
        logger.critical("Emergency shutdown triggered!")
    except Exception as e:
        logger.error(f"Error in critical operation: {e}")
    
    # Example 4: Manual safety checking
    logger.info("\n5. Testing manual safety checking...")
    try:
        result = await manually_checked_function(
            component="manual_example",
            action="manual_check",
            param1="value1",
            param2="value2"
        )
        logger.info(f"Manual check result: {result}")
    except Exception as e:
        logger.error(f"Error in manual safety check: {e}")
    
    # Example 5: Safety-aware component
    logger.info("\n6. Testing safety-aware component...")
    component = SafetyAwareComponent("demo_component")
    
    # Perform safe operations
    for i in range(5):
        try:
            result = await component.perform_safe_operation(
                operation=f"safe_operation_{i}",
                data={"iteration": i, "value": f"data_{i}"}
            )
            logger.info(f"Safe operation result: {result}")
        except Exception as e:
            logger.error(f"Error in safe operation {i}: {e}")
        
        # Check component status periodically
        if i % 2 == 0:
            status = component.get_component_status()
            logger.info(f"Component status: {status}")
    
    # Perform unsafe operation to demonstrate safety violations
    logger.info("\n7. Testing unsafe operation (should trigger safety violations)...")
    try:
        result = await component.perform_safe_operation(
            operation="unsafe_operation",
            data={"risk_level": "high", "dangerous": True}
        )
        logger.info(f"Unsafe operation result: {result}")
    except Exception as e:
        logger.error(f"Unsafe operation blocked: {e}")
    
    # Final component status
    final_status = component.get_component_status()
    logger.info(f"Final component status: {final_status}")
    
    logger.info("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_safety_mechanisms())