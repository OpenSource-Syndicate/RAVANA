"""
Error Recovery Examples for RAVANA AGI System

This module provides examples of how to use the error recovery mechanisms.
"""

import asyncio
import logging
from typing import Optional

from core.error_recovery.decorators import (
    error_handler, graceful_degradation, circuit_breaker
)
from core.error_recovery.error_recovery_manager import (
    ErrorSeverity, RecoveryStrategy
)

logger = logging.getLogger(__name__)


# Example 1: Basic error handling with retry
@error_handler(
    component="example_module",
    severity=ErrorSeverity.MEDIUM,
    recovery_strategy=RecoveryStrategy.RETRY,
    max_retries=3,
    reraise=True
)
async def risky_network_operation(url: str) -> str:
    """
    Example of a network operation that might fail and needs retry logic.
    
    Args:
        url: URL to fetch data from
        
    Returns:
        Fetched data as string
        
    Raises:
        Exception: If all retry attempts fail
    """
    # Simulate network operation that might fail
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise Exception(f"Network error connecting to {url}")
    
    return f"Data from {url}"


# Example 2: Error handling with fallback
def fallback_data_source(query: str) -> str:
    """Fallback data source when primary source fails."""
    return f"Fallback data for query: {query}"

@error_handler(
    component="data_service",
    severity=ErrorSeverity.HIGH,
    recovery_strategy=RecoveryStrategy.FALLBACK,
    max_retries=2,
    fallback_func=fallback_data_source,
    reraise=False
)
async def primary_data_source(query: str) -> str:
    """
    Primary data source that might fail.
    
    Args:
        query: Query to execute
        
    Returns:
        Query results as string
    """
    # Simulate primary data source that sometimes fails
    import random
    if random.random() < 0.5:  # 50% chance of failure
        raise Exception(f"Primary data source error for query: {query}")
    
    return f"Primary data for query: {query}"


# Example 3: Graceful degradation
async def degraded_search_algorithm(query: str) -> str:
    """Simplified search algorithm for degraded mode."""
    return f"Simplified search results for: {query}"

@graceful_degradation(
    component="search_service",
    degraded_function=degraded_search_algorithm,
    timeout=5.0
)
async def advanced_search_algorithm(query: str) -> str:
    """
    Advanced search algorithm that might timeout.
    
    Args:
        query: Search query
        
    Returns:
        Search results as string
    """
    # Simulate complex search that might timeout
    import asyncio
    import random
    
    # Random delay that might exceed timeout
    delay = random.uniform(1.0, 10.0)
    await asyncio.sleep(delay)
    
    return f"Advanced search results for: {query}"


# Example 4: Circuit breaker pattern
@circuit_breaker(
    component="external_api",
    failure_threshold=3,
    recovery_timeout=30.0,
    fallback_func=lambda *args, **kwargs: "Fallback API response"
)
async def external_api_call(endpoint: str, data: dict) -> dict:
    """
    Call to external API that might be unreliable.
    
    Args:
        endpoint: API endpoint
        data: Data to send
        
    Returns:
        API response as dictionary
    """
    # Simulate external API that sometimes fails
    import random
    if random.random() < 0.6:  # 60% chance of failure
        raise Exception(f"External API error at endpoint: {endpoint}")
    
    return {"status": "success", "data": data, "endpoint": endpoint}


# Example 5: Component with health monitoring
class MonitoredComponent:
    """Example component with built-in health monitoring."""
    
    def __init__(self, name: str):
        self.name = name
        self.operation_count = 0
        self.error_count = 0
        
    @error_handler(
        component="monitored_component",
        severity=ErrorSeverity.MEDIUM,
        recovery_strategy=RecoveryStrategy.RESTART,
        max_retries=1,
        reraise=False
    )
    async def perform_operation(self, data: str) -> str:
        """
        Perform an operation that might fail.
        
        Args:
            data: Input data
            
        Returns:
            Processed data
        """
        self.operation_count += 1
        
        # Simulate occasional failures
        import random
        if random.random() < 0.3:  # 30% chance of failure
            self.error_count += 1
            raise Exception(f"Operation failed for data: {data}")
        
        return f"Processed: {data}"
    
    def get_health_status(self) -> dict:
        """Get health status of this component."""
        return {
            "name": self.name,
            "operations": self.operation_count,
            "errors": self.error_count,
            "success_rate": (self.operation_count - self.error_count) / max(1, self.operation_count),
            "healthy": self.error_count < 5  # Healthy if less than 5 errors
        }


# Example usage demonstration
async def demonstrate_error_recovery():
    """Demonstrate various error recovery mechanisms."""
    logger.info("=== Error Recovery Demonstration ===")
    
    # Example 1: Basic retry with error handling
    logger.info("\n1. Basic retry with error handling:")
    try:
        result = await risky_network_operation("https://example.com/api")
        logger.info(f"Success: {result}")
    except Exception as e:
        logger.error(f"Final failure after retries: {e}")
    
    # Example 2: Fallback mechanism
    logger.info("\n2. Fallback mechanism:")
    result = await primary_data_source("important_query")
    logger.info(f"Data source result: {result}")
    
    # Example 3: Graceful degradation
    logger.info("\n3. Graceful degradation:")
    try:
        result = await advanced_search_algorithm("complex search query")
        logger.info(f"Search result: {result}")
    except Exception as e:
        logger.error(f"Search failed: {e}")
    
    # Example 4: Circuit breaker
    logger.info("\n4. Circuit breaker pattern:")
    for i in range(10):
        try:
            result = await external_api_call(f"/api/endpoint/{i}", {"data": f"value_{i}"})
            logger.info(f"API call {i+1}: {result}")
        except Exception as e:
            logger.error(f"API call {i+1} failed: {e}")
        await asyncio.sleep(1)  # Small delay between calls
    
    # Example 5: Monitored component
    logger.info("\n5. Monitored component:")
    component = MonitoredComponent("example_component")
    
    # Perform multiple operations
    for i in range(10):
        result = await component.perform_operation(f"data_{i}")
        logger.info(f"Operation {i+1}: {result}")
        
        # Check health periodically
        if i % 3 == 0:
            health = component.get_health_status()
            logger.info(f"Health check: {health}")
    
    # Final health status
    final_health = component.get_health_status()
    logger.info(f"Final health status: {final_health}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demonstration
    asyncio.run(demonstrate_error_recovery())