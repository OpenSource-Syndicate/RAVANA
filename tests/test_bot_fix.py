#!/usr/bin/env python3
"""
Test script to verify bot connection fixes
"""
import asyncio
import sys
import os
import logging

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from modules.conversational_ai.main import ConversationalAI

async def main():
    """Main entry point for testing bot connections."""
    logger.info("Starting bot connection test...")
    
    try:
        # Initialize the conversational AI
        conversational_ai = ConversationalAI()
        logger.info("Conversational AI instance created")
        
        # Check configuration
        logger.info("Configuration:")
        logger.info(f"  Discord enabled: {conversational_ai.config.get('platforms', {}).get('discord', {}).get('enabled', False)}")
        logger.info(f"  Telegram enabled: {conversational_ai.config.get('platforms', {}).get('telegram', {}).get('enabled', False)}")
        
        # Start the module in standalone mode
        logger.info("Starting Conversational AI in standalone mode...")
        await conversational_ai.start(standalone=True)
            
    except Exception as e:
        logger.error(f"Error running Conversational AI Module: {e}")
        logger.exception("Full traceback:")
        return 1
        
    logger.info("Conversational AI Module stopped.")
    return 0

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nConversational AI Module stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error running Conversational AI Module: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)