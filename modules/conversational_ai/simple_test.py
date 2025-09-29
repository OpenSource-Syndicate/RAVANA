#!/usr/bin/env python3
"""
Simple test script to isolate the issue with the enhanced communication system.
"""
from modules.conversational_ai.communication.message_queue_channel import MessageQueueChannel
from modules.conversational_ai.communication.shared_state_channel import SharedStateChannel
from modules.conversational_ai.communication.memory_service_channel import MemoryServiceChannel
from modules.conversational_ai.communication.data_models import CommunicationMessage, CommunicationType, Priority
import asyncio
import sys
import os
import logging

# Add the project root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', '..'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Local imports


async def test_channels():
    """Test the communication channels individually."""
    logger.info("Starting simple channel test...")

    try:
        # Test Memory Service Channel
        logger.info("Testing Memory Service Channel...")
        memory_channel = MemoryServiceChannel("test_memory_channel")
        await memory_channel.start()
        logger.info("Memory Service Channel started")

        # Test Shared State Channel
        logger.info("Testing Shared State Channel...")
        shared_state_channel = SharedStateChannel("test_shared_state_channel")
        await shared_state_channel.start()
        logger.info("Shared State Channel started")

        # Test Message Queue Channel
        logger.info("Testing Message Queue Channel...")
        message_queue_channel = MessageQueueChannel(
            "test_message_queue_channel")
        await message_queue_channel.start()
        logger.info("Message Queue Channel started")

        # Wait a short time
        await asyncio.sleep(0.5)

        # Stop all channels
        logger.info("Stopping channels...")
        await memory_channel.stop()
        await shared_state_channel.stop()
        await message_queue_channel.stop()
        logger.info("All channels stopped")

        logger.info("Simple channel test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during simple channel test: {e}")
        logger.exception("Full traceback:")
        return False


async def main():
    """Main function with timeout."""
    try:
        # Run the test with a timeout of 5 seconds
        logger.info("Starting simple test with 5 second timeout...")
        result = await asyncio.wait_for(test_channels(), timeout=5.0)
        logger.info("Simple test completed successfully")
        return result
    except asyncio.TimeoutError:
        logger.error("Simple test timed out after 5 seconds")
        return False
    except Exception as e:
        logger.error(f"Error in simple test: {e}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
