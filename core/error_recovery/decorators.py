"""
Error Handling Decorators for RAVANA AGI System

This module provides decorators for enhanced error handling and recovery.
"""

import asyncio
import functools
import logging
from typing import Callable, Any, Optional
import traceback

from core.error_recovery.error_recovery_manager import (
    register_error, ErrorSeverity, RecoveryStrategy
)

logger = logging.getLogger(__name__)


def error_handler(component: str, 
                  severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                  recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
                  max_retries: int = 3,
                  fallback_func: Optional[Callable] = None,
                  reraise: bool = True):
    """
    Decorator for handling errors with recovery mechanisms.
    
    Args:
        component: Component name for error tracking
        severity: Severity level of errors
        recovery_strategy: Strategy for error recovery
        max_retries: Maximum number of retry attempts
        fallback_func: Optional fallback function to call on failure
        reraise: Whether to re-raise the exception after handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Async wrapper for async functions."""
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        # For sync functions called in async context
                        return func(*args, **kwargs)
                        
                except Exception as e:
                    last_exception = e
                    error_id = register_error(
                        component=component,
                        error=e,
                        severity=severity,
                        recovery_strategy=recovery_strategy,
                        metadata={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "args_count": len(args),
                            "kwargs_keys": list(kwargs.keys()) if kwargs else []
                        }
                    )
                    
                    logger.warning(f"Error in {component}.{func.__name__} (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    
                    # If this was the last attempt, try fallback or re-raise
                    if attempt == max_retries:
                        if fallback_func:
                            try:
                                logger.info(f"Calling fallback function for {component}.{func.__name__}")
                                if asyncio.iscoroutinefunction(fallback_func):
                                    return await fallback_func(*args, **kwargs)
                                else:
                                    return fallback_func(*args, **kwargs)
                            except Exception as fallback_error:
                                logger.error(f"Fallback function failed: {fallback_error}")
                                # Register fallback error
                                register_error(
                                    component=f"{component}_fallback",
                                    error=fallback_error,
                                    severity=ErrorSeverity.HIGH,
                                    recovery_strategy=RecoveryStrategy.TERMINATE,
                                    metadata={
                                        "original_function": func.__name__,
                                        "fallback_function": fallback_func.__name__ if hasattr(fallback_func, '__name__') else 'unknown'
                                    }
                                )
                        
                        # If we should re-raise, do so with original exception
                        if reraise:
                            raise e
                        else:
                            return None
                    
                    # Wait before retry with exponential backoff
                    if attempt > 0:
                        delay = 0.1 * (2 ** attempt)  # Exponential backoff starting at 0.1s
                        logger.info(f"Waiting {delay:.2f}s before retry {attempt + 1}")
                        await asyncio.sleep(delay)
            
            # This should never be reached, but just in case
            if reraise and last_exception:
                raise last_exception
            return None
            
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            """Sync wrapper for synchronous functions."""
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_id = register_error(
                        component=component,
                        error=e,
                        severity=severity,
                        recovery_strategy=recovery_strategy,
                        metadata={
                            "function": func.__name__,
                            "attempt": attempt + 1,
                            "max_retries": max_retries,
                            "args_count": len(args),
                            "kwargs_keys": list(kwargs.keys()) if kwargs else []
                        }
                    )
                    
                    logger.warning(f"Error in {component}.{func.__name__} (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    
                    # If this was the last attempt, try fallback or re-raise
                    if attempt == max_retries:
                        if fallback_func:
                            try:
                                logger.info(f"Calling fallback function for {component}.{func.__name__}")
                                return fallback_func(*args, **kwargs)
                            except Exception as fallback_error:
                                logger.error(f"Fallback function failed: {fallback_error}")
                                # Register fallback error
                                register_error(
                                    component=f"{component}_fallback",
                                    error=fallback_error,
                                    severity=ErrorSeverity.HIGH,
                                    recovery_strategy=RecoveryStrategy.TERMINATE,
                                    metadata={
                                        "original_function": func.__name__,
                                        "fallback_function": fallback_func.__name__ if hasattr(fallback_func, '__name__') else 'unknown'
                                    }
                                )
                        
                        # If we should re-raise, do so with original exception
                        if reraise:
                            raise e
                        else:
                            return None
                    
                    # Wait before retry with exponential backoff
                    if attempt > 0:
                        delay = 0.1 * (2 ** attempt)  # Exponential backoff starting at 0.1s
                        logger.info(f"Waiting {delay:.2f}s before retry {attempt + 1}")
                        asyncio.sleep(delay)  # Note: This is blocking in sync context
            
            # This should never be reached, but just in case
            if reraise and last_exception:
                raise last_exception
            return None
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


def graceful_degradation(component: str, 
                        degraded_function: Optional[Callable] = None,
                        timeout: float = 30.0):
    """
    Decorator for functions that should gracefully degrade on timeout or error.
    
    Args:
        component: Component name for error tracking
        degraded_function: Optional degraded version of the function to call on failure
        timeout: Timeout in seconds before degrading
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Async wrapper with timeout and graceful degradation."""
            try:
                # Try the original function with timeout
                if asyncio.iscoroutinefunction(func):
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                else:
                    # For sync functions in async context
                    loop = asyncio.get_event_loop()
                    return await asyncio.wait_for(
                        loop.run_in_executor(None, func, *args, **kwargs), 
                        timeout=timeout
                    )
                    
            except asyncio.TimeoutError as e:
                logger.warning(f"Timeout in {component}.{func.__name__} after {timeout}s, attempting graceful degradation")
                
                # Register timeout error
                register_error(
                    component=component,
                    error=e,
                    severity=ErrorSeverity.MEDIUM,
                    recovery_strategy=RecoveryStrategy.DEGRADED,
                    metadata={
                        "function": func.__name__,
                        "timeout": timeout,
                        "error_type": "timeout"
                    }
                )
                
                # Try degraded function if provided
                if degraded_function:
                    try:
                        logger.info(f"Calling degraded function for {component}.{func.__name__}")
                        if asyncio.iscoroutinefunction(degraded_function):
                            return await degraded_function(*args, **kwargs)
                        else:
                            loop = asyncio.get_event_loop()
                            return await loop.run_in_executor(None, degraded_function, *args, **kwargs)
                    except Exception as degraded_error:
                        logger.error(f"Degraded function failed: {degraded_error}")
                        # Register degraded function error
                        register_error(
                            component=f"{component}_degraded",
                            error=degraded_error,
                            severity=ErrorSeverity.HIGH,
                            recovery_strategy=RecoveryStrategy.TERMINATE,
                            metadata={
                                "original_function": func.__name__,
                                "degraded_function": degraded_function.__name__ if hasattr(degraded_function, '__name__') else 'unknown'
                            }
                        )
                        raise degraded_error
                
                # If no degraded function, re-raise timeout
                raise e
                
            except Exception as e:
                logger.error(f"Error in {component}.{func.__name__}: {e}")
                
                # Register general error
                register_error(
                    component=component,
                    error=e,
                    severity=ErrorSeverity.HIGH,
                    recovery_strategy=RecoveryStrategy.DEGRADED,
                    metadata={
                        "function": func.__name__,
                        "error_type": type(e).__name__
                    }
                )
                
                # Try degraded function if provided
                if degraded_function:
                    try:
                        logger.info(f"Calling degraded function for {component}.{func.__name__}")
                        if asyncio.iscoroutinefunction(degraded_function):
                            return await degraded_function(*args, **kwargs)
                        else:
                            loop = asyncio.get_event_loop()
                            return await loop.run_in_executor(None, degraded_function, *args, **kwargs)
                    except Exception as degraded_error:
                        logger.error(f"Degraded function failed: {degraded_error}")
                        # Register degraded function error
                        register_error(
                            component=f"{component}_degraded",
                            error=degraded_error,
                            severity=ErrorSeverity.HIGH,
                            recovery_strategy=RecoveryStrategy.TERMINATE,
                            metadata={
                                "original_function": func.__name__,
                                "degraded_function": degraded_function.__name__ if hasattr(degraded_function, '__name__') else 'unknown'
                            }
                        )
                        raise degraded_error
                
                # If no degraded function, re-raise original error
                raise e
                
        return async_wrapper
    return decorator


def circuit_breaker(component: str,
                    failure_threshold: int = 5,
                    recovery_timeout: float = 60.0,
                    fallback_func: Optional[Callable] = None):
    """
    Circuit breaker decorator to prevent cascading failures.
    
    Args:
        component: Component name for error tracking
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time in seconds before attempting to close circuit
        fallback_func: Optional fallback function to call when circuit is open
    """
    # Circuit breaker state tracking
    circuit_state = {
        'failures': 0,
        'last_failure_time': None,
        'open': False
    }
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Async wrapper with circuit breaker protection."""
            # Check if circuit is open
            if circuit_state['open']:
                # Check if recovery timeout has passed
                if (circuit_state['last_failure_time'] and 
                    (asyncio.get_event_loop().time() - circuit_state['last_failure_time']) > recovery_timeout):
                    logger.info(f"Circuit breaker for {component} entering half-open state")
                    circuit_state['open'] = False
                    circuit_state['failures'] = 0
                else:
                    # Circuit is still open, use fallback or raise exception
                    logger.warning(f"Circuit breaker for {component} is OPEN, rejecting call to {func.__name__}")
                    if fallback_func:
                        try:
                            if asyncio.iscoroutinefunction(fallback_func):
                                return await fallback_func(*args, **kwargs)
                            else:
                                loop = asyncio.get_event_loop()
                                return await loop.run_in_executor(None, fallback_func, *args, **kwargs)
                        except Exception as fallback_error:
                            logger.error(f"Fallback function failed: {fallback_error}")
                            raise fallback_error
                    else:
                        raise Exception(f"Circuit breaker is OPEN for {component}.{func.__name__}")
            
            try:
                # Call the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, func, *args, **kwargs)
                
                # Success - reset failure count
                circuit_state['failures'] = 0
                circuit_state['last_failure_time'] = None
                circuit_state['open'] = False
                
                return result
                
            except Exception as e:
                # Record failure
                circuit_state['failures'] += 1
                circuit_state['last_failure_time'] = asyncio.get_event_loop().time()
                
                # Register error
                register_error(
                    component=component,
                    error=e,
                    severity=ErrorSeverity.HIGH,
                    recovery_strategy=RecoveryStrategy.ISOLATE,
                    metadata={
                        "function": func.__name__,
                        "circuit_failures": circuit_state['failures'],
                        "failure_threshold": failure_threshold
                    }
                )
                
                # Check if we should open the circuit
                if circuit_state['failures'] >= failure_threshold:
                    circuit_state['open'] = True
                    logger.error(f"Circuit breaker for {component} OPENED after {circuit_state['failures']} failures")
                
                # Re-raise the exception
                raise e
                
        return async_wrapper
    return decorator