#!/usr/bin/env python3
import sys
import os
import asyncio

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.engine import create_db_and_tables, engine
from core.system import AGISystem

async def main():
    print("Creating database...")
    create_db_and_tables()
    print("Database created")
    
    print("Initializing AGI system...")
    agi_system = AGISystem(engine)
    print("AGI system initialized")
    
    print("Initializing components...")
    initialization_success = await agi_system.initialize_components()
    print(f"Component initialization success: {initialization_success}")
    
    # Check if conversational AI was initialized
    print(f"Conversational AI initialized: {agi_system.conversational_ai is not None}")
    if agi_system.conversational_ai:
        print("Conversational AI object exists")
    else:
        print("Conversational AI object is None")
        
    # Check config values
    from core.config import Config
    print(f"CONVERSATIONAL_AI_ENABLED: {Config.CONVERSATIONAL_AI_ENABLED}")
    
    # Try to start conversational AI
    if agi_system.conversational_ai:
        try:
            print("Starting conversational AI...")
            await agi_system.start_conversational_ai()
            print("Conversational AI started")
        except Exception as e:
            print(f"Error starting conversational AI: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping conversational AI start - not initialized")

if __name__ == "__main__":
    asyncio.run(main())