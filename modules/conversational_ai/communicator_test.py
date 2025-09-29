#!/usr/bin/env python3
"""
Targeted test script for the RAVANACommunicator.
"""
from modules.conversational_ai.communication.ravana_bridge import RAVANACommunicator
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


class MockConversationalAI:
    """Mock ConversationalAI for testing."""

    def __init__(self):
        pass


async def test_communicator():
    """Test the RAVANACommunicator."""
    logger.info("Starting RAVANACommunicator test...")

    try:
        # Create a mock ConversationalAI
        mock_ai = MockConversationalAI()

        # Test RAVANA Communicator
        logger.info("Creating RAVANACommunicator...")
        communicator = RAVANACommunicator("test_channel", mock_ai)
        logger.info("RAVANACommunicator created")

        # Start the communicator
        logger.info("Starting RAVANACommunicator...")
        await communicator.start()
        logger.info("RAVANACommunicator started")

        # Wait a short time
        await asyncio.sleep(0.5)

        # Stop the communicator
        logger.info("Stopping RAVANACommunicator...")
        await communicator.stop()
        logger.info("RAVANACommunicator stopped")

        logger.info("RAVANACommunicator test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during RAVANACommunicator test: {e}")
        logger.exception("Full traceback:")
        return False


async def main():
    """Main function with timeout."""
    try:
        # Run the test with a timeout of 5 seconds
        logger.info("Starting RAVANACommunicator test with 5 second timeout...")
        result = await asyncio.wait_for(test_communicator(), timeout=5.0)
        logger.info("RAVANACommunicator test completed successfully")
        return result
    except asyncio.TimeoutError:
        logger.error("RAVANACommunicator test timed out after 5 seconds")
        return False
    except Exception as e:
        logger.error(f"Error in RAVANACommunicator test: {e}")
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
