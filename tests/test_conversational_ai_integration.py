#!/usr/bin/env python3
"""
Test script for the Conversational AI Module integration with the main AGI system
"""
import asyncio
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from core.system import AGISystem
from database.engine import create_db_and_tables, engine
from core.config import Config

async def test_conversational_ai_integration():
    """Test the conversational AI module integration with the AGI system."""
    print("Testing Conversational AI Module integration with AGI system...")
    
    try:
        # Create database and tables
        print("Initializing database...")
        create_db_and_tables()
        print("Database initialized successfully")
        
        # Initialize the AGI system
        print("Initializing AGI system...")
        agi_system = AGISystem(engine)
        
        # Initialize components
        print("Initializing AGI system components...")
        initialization_success = await agi_system.initialize_components()
        if not initialization_success:
            print("❌ AGI system initialization failed")
            return False
            
        print("✅ AGI system initialized successfully")
        
        # Check if conversational AI is available
        if not agi_system.conversational_ai:
            print("ℹ️  Conversational AI module not available or not enabled")
            return True
            
        # Start conversational AI
        print("Starting Conversational AI...")
        await agi_system.start_conversational_ai()
        print("✅ Conversational AI started successfully")
        
        # Check status
        status = agi_system.get_conversational_ai_status()
        print(f"Conversational AI Status: {status}")
        
        # Let it run for a few seconds to ensure bots are initialized
        await asyncio.sleep(3)
        
        # Stop conversational AI
        print("Stopping Conversational AI...")
        # Signal shutdown
        if agi_system.conversational_ai:
            agi_system.conversational_ai._shutdown.set()
        print("✅ Conversational AI stopped successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Conversational AI integration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_conversational_ai_integration())
        if success:
            print("\n✅ All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    except Exception as e:
        print(f"Error running test: {e}")
        sys.exit(1)