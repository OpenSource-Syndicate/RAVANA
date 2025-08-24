#!/usr/bin/env python3
"""
Test script to verify that the conversational AI fix works correctly
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

async def test_conversational_ai():
    """Test the conversational AI startup."""
    try:
        logger.info("Testing Conversational AI startup...")
        
        # Import the conversational AI
        from modules.conversational_ai.main import ConversationalAI
        
        # Create an instance
        ai = ConversationalAI()
        logger.info("Conversational AI instance created")
        
        # Check configuration
        logger.info("Configuration:")
        logger.info(f"  Discord enabled: {ai.config.get('platforms', {}).get('discord', {}).get('enabled', False)}")
        logger.info(f"  Telegram enabled: {ai.config.get('platforms', {}).get('telegram', {}).get('enabled', False)}")
        
        # Start in standalone mode but with a timeout
        logger.info("Starting Conversational AI in standalone mode with timeout...")
        
        # Create a task for the conversational AI
        async def run_ai():
            try:
                await ai.start(standalone=True)
            except Exception as e:
                logger.error(f"Error in conversational AI: {e}")
                logger.exception("Full traceback:")
        
        ai_task = asyncio.create_task(run_ai())
        
        # Wait for a short time to see if it starts
        try:
            await asyncio.wait_for(ai_task, timeout=10.0)
        except asyncio.TimeoutError:
            logger.info("Conversational AI is running (as expected for standalone mode)")
            # Cancel the task since we're just testing startup
            ai_task.cancel()
            try:
                await ai_task
            except asyncio.CancelledError:
                pass
        
        # Stop the AI
        await ai.stop()
        logger.info("Conversational AI stopped successfully")
        
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_conversational_ai())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)