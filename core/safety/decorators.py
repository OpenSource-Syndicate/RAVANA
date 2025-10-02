"""
Safety Decorators for RAVANA AGI System

This module provides decorators for applying safety checks to autonomous actions.
"""

import asyncio
import functools
import logging
from typing import Callable, Any, Optional, Dict
import json

from core.safety.safety_manager import (
    check_action_safety, monitor_execution, 
    SafetyLevel, SafetyViolationType
)

logger = logging.getLogger(__name__)


def safety_checked(component: str, 
                  action: Optional[str] = None,
                  required_params: Optional[list] = None,
                  safety_level: SafetyLevel = SafetyLevel.MEDIUM,
                  block_on_violation: bool = True):
    """
    Decorator for applying safety checks to functions.
    
    Args:
        component: Component name for safety tracking
        action: Specific action name (defaults to function name)
        required_params: List of required parameter names
        safety_level: Required safety level
        block_on_violation: Whether to block execution on safety violations
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Async wrapper with safety checks."""
            try:
                # Determine action name
                func_action = action or func.__name__
                
                # Prepare parameters for safety check
                params = kwargs.copy()
                
                # Add positional arguments if possible
                try:
                    # Get function signature to map positional args to parameter names
                    import inspect
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    params.update(bound_args.arguments)
                except Exception as e:
                    logger.debug(f"Could not bind positional args to parameter names: {e}")
                    # Fallback: add args as generic parameters
                    for i, arg in enumerate(args):
                        params[f"arg_{i}"] = arg
                
                # Check required parameters
                if required_params:
                    missing_params = [p for p in required_params if p not in params]
                    if missing_params:
                        logger.warning(f"Missing required parameters for {component}.{func_action}: {missing_params}")
                        # Still proceed but log the issue
                
                # Perform safety check
                safety_context = {
                    "function": func.__name__,
                    "caller": "decorator",
                    "required_safety_level": safety_level.value
                }
                
                safety_result = check_action_safety(
                    component=component,
                    action=func_action,
                    params=params,
                    context=safety_context
                )
                
                # Handle safety violations
                if not safety_result.get("approved", True):
                    violations = safety_result.get("violations", [])
                    logger.warning(f"Safety violations detected for {component}.{func_action}: {len(violations)} violations")
                    
                    # Log each violation
                    for violation in violations:
                        logger.warning(f"VIOLATION: {violation.get('violation_type')} - {violation.get('description')}")
                    
                    # Block execution if configured to do so
                    if block_on_violation:
                        logger.error(f"Execution blocked due to safety violations for {component}.{func_action}")
                        raise PermissionError(f"Action blocked due to safety violations: {len(violations)} violations detected")
                
                # Proceed with function execution
                logger.debug(f"Safety check passed for {component}.{func_action}, proceeding with execution")
                
                # Execute the function
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        # For sync functions in async context
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, func, *args, **kwargs)
                    
                    # Monitor successful execution
                    execution_result = {
                        "success": True,
                        "result_preview": str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                    }
                    monitor_execution(component, func_action, params, execution_result)
                    
                    return result
                    
                except Exception as execution_error:
                    # Monitor failed execution
                    execution_result = {
                        "success": False,
                        "error": str(execution_error)
                    }
                    monitor_execution(component, func_action, params, execution_result)
                    
                    # Re-raise the execution error
                    raise execution_error
                    
            except PermissionError:
                # Re-raise permission errors without additional handling
                raise
            except Exception as e:
                logger.error(f"Error in safety-checked function {component}.{func.__name__}: {e}")
                # Still raise the error, but don't block on safety check errors
                raise e
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            """Sync wrapper with safety checks."""
            try:
                # Determine action name
                func_action = action or func.__name__
                
                # Prepare parameters for safety check
                params = kwargs.copy()
                
                # Add positional arguments if possible
                try:
                    # Get function signature to map positional args to parameter names
                    import inspect
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    params.update(bound_args.arguments)
                except Exception as e:
                    logger.debug(f"Could not bind positional args to parameter names: {e}")
                    # Fallback: add args as generic parameters
                    for i, arg in enumerate(args):
                        params[f"arg_{i}"] = arg
                
                # Check required parameters
                if required_params:
                    missing_params = [p for p in required_params if p not in params]
                    if missing_params:
                        logger.warning(f"Missing required parameters for {component}.{func_action}: {missing_params}")
                        # Still proceed but log the issue
                
                # Perform safety check
                safety_context = {
                    "function": func.__name__,
                    "caller": "decorator",
                    "required_safety_level": safety_level.value
                }
                
                safety_result = check_action_safety(
                    component=component,
                    action=func_action,
                    params=params,
                    context=safety_context
                )
                
                # Handle safety violations
                if not safety_result.get("approved", True):
                    violations = safety_result.get("violations", [])
                    logger.warning(f"Safety violations detected for {component}.{func_action}: {len(violations)} violations")
                    
                    # Log each violation
                    for violation in violations:
                        logger.warning(f"VIOLATION: {violation.get('violation_type')} - {violation.get('description')}")
                    
                    # Block execution if configured to do so
                    if block_on_violation:
                        logger.error(f"Execution blocked due to safety violations for {component}.{func_action}")
                        raise PermissionError(f"Action blocked due to safety violations: {len(violations)} violations detected")
                
                # Proceed with function execution
                logger.debug(f"Safety check passed for {component}.{func_action}, proceeding with execution")
                
                # Execute the function
                try:
                    result = func(*args, **kwargs)
                    
                    # Monitor successful execution
                    execution_result = {
                        "success": True,
                        "result_preview": str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                    }
                    monitor_execution(component, func_action, params, execution_result)
                    
                    return result
                    
                except Exception as execution_error:
                    # Monitor failed execution
                    execution_result = {
                        "success": False,
                        "error": str(execution_error)
                    }
                    monitor_execution(component, func_action, params, execution_result)
                    
                    # Re-raise the execution error
                    raise execution_error
                    
            except PermissionError:
                # Re-raise permission errors without additional handling
                raise
            except Exception as e:
                logger.error(f"Error in safety-checked function {component}.{func.__name__}: {e}")
                # Still raise the error, but don't block on safety check errors
                raise e
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


def privacy_protected(component: str,
                     pii_handling: str = "redact",
                     data_retention_days: Optional[int] = None):
    """
    Decorator for functions that handle sensitive data with privacy protection.
    
    Args:
        component: Component name for privacy tracking
        pii_handling: How to handle PII (redact, encrypt, anonymize)
        data_retention_days: How long to retain data (None for default)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Async wrapper with privacy protection."""
            try:
                # Log privacy-sensitive operation
                logger.info(f"Privacy-protected operation in {component}.{func.__name__}")
                
                # Apply PII handling
                processed_args, processed_kwargs = _apply_pii_handling(
                    args, kwargs, pii_handling
                )
                
                # Execute function with protected data
                if asyncio.iscoroutinefunction(func):
                    result = await func(*processed_args, **processed_kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, func, *processed_args, **processed_kwargs)
                
                # Apply post-processing for privacy
                protected_result = _protect_result_privacy(result, pii_handling)
                
                # Monitor privacy-protected execution
                privacy_context = {
                    "pii_handling": pii_handling,
                    "data_retention_days": data_retention_days,
                    "operation": func.__name__
                }
                
                monitor_execution(
                    component, 
                    f"privacy_protected_{func.__name__}", 
                    {"args_count": len(args), "kwargs_count": len(kwargs)},
                    {
                        "success": True,
                        "privacy_handled": True,
                        "pii_handling": pii_handling
                    }
                )
                
                return protected_result
                
            except Exception as e:
                logger.error(f"Error in privacy-protected function {component}.{func.__name__}: {e}")
                
                # Monitor failed privacy-protected execution
                monitor_execution(
                    component, 
                    f"privacy_protected_{func.__name__}", 
                    {"args_count": len(args), "kwargs_count": len(kwargs)},
                    {
                        "success": False,
                        "error": str(e),
                        "privacy_handled": False
                    }
                )
                
                raise e
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            """Sync wrapper with privacy protection."""
            try:
                # Log privacy-sensitive operation
                logger.info(f"Privacy-protected operation in {component}.{func.__name__}")
                
                # Apply PII handling
                processed_args, processed_kwargs = _apply_pii_handling(
                    args, kwargs, pii_handling
                )
                
                # Execute function with protected data
                result = func(*processed_args, **processed_kwargs)
                
                # Apply post-processing for privacy
                protected_result = _protect_result_privacy(result, pii_handling)
                
                # Monitor privacy-protected execution
                privacy_context = {
                    "pii_handling": pii_handling,
                    "data_retention_days": data_retention_days,
                    "operation": func.__name__
                }
                
                monitor_execution(
                    component, 
                    f"privacy_protected_{func.__name__}", 
                    {"args_count": len(args), "kwargs_count": len(kwargs)},
                    {
                        "success": True,
                        "privacy_handled": True,
                        "pii_handling": pii_handling
                    }
                )
                
                return protected_result
                
            except Exception as e:
                logger.error(f"Error in privacy-protected function {component}.{func.__name__}: {e}")
                
                # Monitor failed privacy-protected execution
                monitor_execution(
                    component, 
                    f"privacy_protected_{func.__name__}", 
                    {"args_count": len(args), "kwargs_count": len(kwargs)},
                    {
                        "success": False,
                        "error": str(e),
                        "privacy_handled": False
                    }
                )
                
                raise e
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


def _apply_pii_handling(args: tuple, kwargs: dict, pii_handling: str) -> tuple:
    """
    Apply PII handling to function arguments.
    
    Args:
        args: Positional arguments
        kwargs: Keyword arguments
        pii_handling: How to handle PII
        
    Returns:
        Processed arguments
    """
    try:
        if pii_handling == "redact":
            # Redact sensitive information
            processed_args = _redact_sensitive_data(args)
            processed_kwargs = _redact_sensitive_data(kwargs)
            return processed_args, processed_kwargs
        elif pii_handling == "encrypt":
            # Encrypt sensitive information (placeholder)
            logger.debug("PII encryption would be applied here")
            return args, kwargs
        elif pii_handling == "anonymize":
            # Anonymize sensitive information (placeholder)
            logger.debug("PII anonymization would be applied here")
            return args, kwargs
        else:
            # No PII handling
            return args, kwargs
    except Exception as e:
        logger.warning(f"Error applying PII handling: {e}")
        return args, kwargs


def _redact_sensitive_data(data: Any) -> Any:
    """
    Redact sensitive data from arguments.
    
    Args:
        data: Data to redact
        
    Returns:
        Data with sensitive information redacted
    """
    try:
        if isinstance(data, str):
            # Redact common PII patterns
            import re
            
            # Social Security Numbers
            data = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', data)
            
            # Credit Card Numbers
            data = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[REDACTED_CC]', data)
            
            # Email addresses
            data = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', data)
            
            # Phone numbers
            data = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[REDACTED_PHONE]', data)
            
            return data
        elif isinstance(data, dict):
            # Recursively redact dictionary values
            redacted_dict = {}
            for key, value in data.items():
                redacted_dict[key] = _redact_sensitive_data(value)
            return redacted_dict
        elif isinstance(data, (list, tuple)):
            # Recursively redact list/tuple values
            redacted_list = [_redact_sensitive_data(item) for item in data]
            return type(data)(redacted_list)  # Preserve original type (list/tuple)
        else:
            # Return unchanged for other types
            return data
    except Exception as e:
        logger.warning(f"Error redacting sensitive data: {e}")
        return data


def _protect_result_privacy(result: Any, pii_handling: str) -> Any:
    """
    Protect privacy in function results.
    
    Args:
        result: Function result
        pii_handling: How to handle PII
        
    Returns:
        Privacy-protected result
    """
    try:
        if pii_handling == "redact":
            return _redact_sensitive_data(result)
        elif pii_handling in ["encrypt", "anonymize"]:
            # Placeholder for encryption/anonymization
            logger.debug(f"Privacy protection ({pii_handling}) would be applied to result")
            return result
        else:
            return result
    except Exception as e:
        logger.warning(f"Error protecting result privacy: {e}")
        return result


def emergency_shutdown_on_violations(max_violations: int = 10):
    """
    Decorator that triggers emergency shutdown if safety violations exceed threshold.
    
    Args:
        max_violations: Maximum violations before shutdown
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Async wrapper with emergency shutdown protection."""
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, func, *args, **kwargs)
                
                # Check violation count and trigger shutdown if needed
                from core.safety.safety_manager import get_safety_manager
                safety_manager = get_safety_manager()
                if safety_manager:
                    violation_count = len(safety_manager.safety_violations)
                    if violation_count >= max_violations:
                        logger.critical(f"Emergency shutdown triggered: {violation_count} violations >= {max_violations} threshold")
                        # In a real implementation, this would trigger actual shutdown
                        # For now, we'll just log and raise an exception
                        raise SystemExit(f"Emergency shutdown triggered due to {violation_count} safety violations")
                
                return result
                
            except SystemExit:
                # Re-raise system exit for emergency shutdown
                raise
            except Exception as e:
                # Check for emergency shutdown even on errors
                from core.safety.safety_manager import get_safety_manager
                safety_manager = get_safety_manager()
                if safety_manager:
                    violation_count = len(safety_manager.safety_violations)
                    if violation_count >= max_violations:
                        logger.critical(f"Emergency shutdown triggered on error: {violation_count} violations >= {max_violations} threshold")
                        raise SystemExit(f"Emergency shutdown triggered due to {violation_count} safety violations")
                
                # Re-raise original exception
                raise e
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            """Sync wrapper with emergency shutdown protection."""
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Check violation count and trigger shutdown if needed
                from core.safety.safety_manager import get_safety_manager
                safety_manager = get_safety_manager()
                if safety_manager:
                    violation_count = len(safety_manager.safety_violations)
                    if violation_count >= max_violations:
                        logger.critical(f"Emergency shutdown triggered: {violation_count} violations >= {max_violations} threshold")
                        # In a real implementation, this would trigger actual shutdown
                        # For now, we'll just log and raise an exception
                        raise SystemExit(f"Emergency shutdown triggered due to {violation_count} safety violations")
                
                return result
                
            except SystemExit:
                # Re-raise system exit for emergency shutdown
                raise
            except Exception as e:
                # Check for emergency shutdown even on errors
                from core.safety.safety_manager import get_safety_manager
                safety_manager = get_safety_manager()
                if safety_manager:
                    violation_count = len(safety_manager.safety_violations)
                    if violation_count >= max_violations:
                        logger.critical(f"Emergency shutdown triggered on error: {violation_count} violations >= {max_violations} threshold")
                        raise SystemExit(f"Emergency shutdown triggered due to {violation_count} safety violations")
                
                # Re-raise original exception
                raise e
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator