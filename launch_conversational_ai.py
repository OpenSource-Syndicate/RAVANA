#!/usr/bin/env python3
"""
Launcher script for the Conversational AI Module
"""
import asyncio
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.conversational_ai.main import ConversationalAI

async def main():
    """Main entry point for the Conversational AI Module."""
    print("Starting Conversational AI Module...")
    
    # Initialize the conversational AI
    conversational_ai = ConversationalAI()
    
    # Start the module in standalone mode
    await conversational_ai.start(standalone=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nConversational AI Module stopped by user.")
    except Exception as e:
        print(f"Error running Conversational AI Module: {e}")
        sys.exit(1)