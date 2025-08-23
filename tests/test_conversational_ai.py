#!/usr/bin/env python3
"""
Test script for the Conversational AI Module startup and shutdown
"""
import asyncio
import sys
import os
import signal
import time

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.conversational_ai.main import ConversationalAI

async def test_conversational_ai():
    """Test the conversational AI module startup and shutdown."""
    print("Testing Conversational AI Module...")
    
    # Initialize the conversational AI
    conversational_ai = ConversationalAI()
    
    # Start the module in standalone mode
    print("Starting Conversational AI Module...")
    task = asyncio.create_task(conversational_ai.start(standalone=True))
    
    # Let it run for a few seconds
    await asyncio.sleep(3)
    
    # Signal shutdown
    print("Shutting down Conversational AI Module...")
    conversational_ai._shutdown.set()
    
    # Wait for shutdown to complete
    try:
        await asyncio.wait_for(task, timeout=10)
        print("Conversational AI Module shut down successfully")
    except asyncio.TimeoutError:
        print("Conversational AI Module shutdown timed out")
        task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(test_conversational_ai())
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    except Exception as e:
        print(f"Error running test: {e}")
        sys.exit(1)