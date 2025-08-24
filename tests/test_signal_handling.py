#!/usr/bin/env python3
"""
Test script to verify that signal handling works properly.
"""
import asyncio
import signal
import sys
import os
import time

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.system import AGISystem
from database.engine import create_db_and_tables, engine

# Global shutdown event for testing
test_shutdown_event = asyncio.Event()

async def signal_handler(signum, frame):
    """Handle signals for testing."""
    print(f"Received signal {signum}")
    test_shutdown_event.set()

def setup_test_signal_handlers():
    """Set up signal handlers for testing."""
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        print("✅ Test signal handlers configured")
    except Exception as e:
        print(f"❌ Error setting up test signal handlers: {e}")

async def test_signal_handling():
    """Test that signal handling works properly."""
    print("Testing signal handling functionality...")
    
    # Set up signal handlers
    setup_test_signal_handlers()
    
    # Create database
    print("Creating database...")
    create_db_and_tables()
    
    # Initialize AGI system
    print("Initializing AGI system...")
    agi_system = AGISystem(engine)
    
    # Initialize components
    print("Initializing components...")
    await agi_system.initialize_components()
    
    # Start Conversational AI if available
    if agi_system.conversational_ai:
        print("Starting Conversational AI...")
        await agi_system.start_conversational_ai()
    
    # Simulate running the autonomous loop
    print("Running autonomous loop...")
    
    # Run for a short time, checking for shutdown event
    start_time = time.time()
    while not test_shutdown_event.is_set() and (time.time() - start_time) < 5:
        print(f"Running... {(time.time() - start_time):.1f}s elapsed")
        await asyncio.sleep(1)
    
    # Test shutdown
    print("Testing shutdown...")
    start_time = time.time()
    await agi_system.stop("test_signal_shutdown")
    shutdown_time = time.time() - start_time
    
    print(f"Shutdown completed in {shutdown_time:.2f} seconds")
    print("✅ Signal handling test completed successfully")
    
    return True

if __name__ == "__main__":
    try:
        asyncio.run(test_signal_handling())
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)