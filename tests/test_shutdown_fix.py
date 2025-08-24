#!/usr/bin/env python3
"""
Test script to verify that the shutdown fixes work properly.
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

async def test_shutdown():
    """Test that shutdown works properly."""
    print("Testing shutdown functionality...")
    
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
    
    # Simulate running for a short time
    print("Running system for 3 seconds...")
    await asyncio.sleep(3)
    
    # Test shutdown
    print("Testing shutdown...")
    start_time = time.time()
    await agi_system.stop("test_shutdown")
    shutdown_time = time.time() - start_time
    
    print(f"Shutdown completed in {shutdown_time:.2f} seconds")
    print("✅ Shutdown test completed successfully")
    
    return True

if __name__ == "__main__":
    try:
        asyncio.run(test_shutdown())
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)