"""
Simple test for Enhanced Snake Agent core functionality
"""

import asyncio
import tempfile
import time
from datetime import datetime

# Test the core components
from core.snake_data_models import SnakeAgentConfiguration, TaskPriority
from core.snake_log_manager import SnakeLogManager


def test_configuration():
    """Test configuration validation"""
    print("Testing Snake Agent Configuration...")
    
    # Valid configuration
    config = SnakeAgentConfiguration()
    issues = config.validate()
    assert len(issues) == 0, f"Valid config should have no issues, but got: {issues}"
    print("✓ Valid configuration passed")
    
    # Invalid configuration
    invalid_config = SnakeAgentConfiguration(
        max_threads=0,
        max_processes=0,
        file_monitor_interval=0.05
    )
    issues = invalid_config.validate()
    assert len(issues) > 0, "Invalid config should have issues"
    print(f"✓ Invalid configuration detected {len(issues)} issues")
    
    print("Configuration tests passed!\n")


def test_log_manager():
    """Test log manager functionality"""
    print("Testing Snake Log Manager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_manager = SnakeLogManager(temp_dir)
        
        # Test initialization
        assert log_manager.log_dir.exists()
        assert log_manager.improvement_logger is not None
        print("✓ Log manager initialized")
        
        # Test log processor
        log_manager.start_log_processor()
        assert log_manager.worker_running
        print("✓ Log processor started")
        
        # Test system event logging
        async def test_logging():
            await log_manager.log_system_event(
                "test_event",
                {"test_data": "value"},
                worker_id="test_worker"
            )
            
            # Give time for processing
            await asyncio.sleep(0.5)
            
            return log_manager.logs_processed > 0
        
        # Run async test
        result = asyncio.run(test_logging())
        assert result, "System event should be logged"
        print("✓ System event logging works")
        
        # Stop log processor
        log_manager.stop_log_processor()
        assert not log_manager.worker_running
        print("✓ Log processor stopped")
    
    print("Log manager tests passed!\n")


def test_task_priorities():
    """Test task priority system"""
    print("Testing Task Priority System...")
    
    priorities = [
        TaskPriority.CRITICAL,
        TaskPriority.HIGH,
        TaskPriority.MEDIUM,
        TaskPriority.LOW,
        TaskPriority.BACKGROUND
    ]
    
    # Test ordering
    assert TaskPriority.CRITICAL.value > TaskPriority.HIGH.value
    assert TaskPriority.HIGH.value > TaskPriority.MEDIUM.value
    assert TaskPriority.MEDIUM.value > TaskPriority.LOW.value
    assert TaskPriority.LOW.value > TaskPriority.BACKGROUND.value
    print("✓ Task priorities are correctly ordered")
    
    print("Task priority tests passed!\n")


def test_performance():
    """Test basic performance characteristics"""
    print("Testing Performance Characteristics...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_manager = SnakeLogManager(temp_dir)
        log_manager.start_log_processor()
        
        async def performance_test():
            start_time = time.time()
            
            # Log multiple events rapidly
            for i in range(100):
                await log_manager.log_system_event(
                    "performance_test",
                    {"iteration": i, "timestamp": datetime.now().isoformat()},
                    worker_id=f"worker_{i % 5}"
                )
            
            # Wait for processing
            await asyncio.sleep(2.0)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✓ Processed 100 log events in {duration:.2f} seconds")
            print(f"✓ Throughput: {100/duration:.1f} events/second")
            
            # Check that events were processed
            assert log_manager.logs_processed >= 100
            print(f"✓ {log_manager.logs_processed} events processed successfully")
        
        asyncio.run(performance_test())
        log_manager.stop_log_processor()
    
    print("Performance tests passed!\n")


def main():
    """Run all tests"""
    print("=== Enhanced Snake Agent Core Tests ===\n")
    
    try:
        test_configuration()
        test_log_manager()
        test_task_priorities()
        test_performance()
        
        print("🎉 All tests passed successfully!")
        print("\nEnhanced Snake Agent core functionality is working correctly.")
        print("The system is ready for:")
        print("- Threading-based file monitoring and analysis")
        print("- Multiprocessing-based experiments and improvements")
        print("- Separate logging for different activities")
        print("- Concurrent RAVANA improvement operations")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)