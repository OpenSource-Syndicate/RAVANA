#!/usr/bin/env python3
"""
Test script for the communication channels between conversational AI and RAVANA AGI.
"""
import asyncio
import sys
import os
import logging
from datetime import datetime

# Add the project root directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Local imports
from modules.conversational_ai.communication.ravana_bridge import RAVANACommunicator
from modules.conversational_ai.communication.data_models import CommunicationMessage, CommunicationType, Priority


class MockConversationalAI:
    """Mock ConversationalAI for testing."""
    def __init__(self):
        pass
    
    async def send_message_to_user(self, user_id: str, message: str, platform: str = None):
        """Mock method to send message to user."""
        logger.info(f"Mock: Sending message to user {user_id}: {message}")


async def test_communication_channels():
    """Test the communication channels."""
    logger.info("Starting communication channels test...")
    
    try:
        # Create a mock ConversationalAI
        mock_ai = MockConversationalAI()
        
        # Create RAVANA Communicator
        logger.info("Creating RAVANACommunicator...")
        communicator = RAVANACommunicator("test_channel", mock_ai)
        logger.info("RAVANACommunicator created")
        
        # Start the communicator
        logger.info("Starting RAVANACommunicator...")
        await communicator.start()
        logger.info("RAVANACommunicator started")
        
        # Test Memory Service Channel
        logger.info("Testing Memory Service Channel...")
        memory_message = CommunicationMessage(
            id="test_memory_001",
            type=CommunicationType.THOUGHT_EXCHANGE,
            priority=Priority.LOW,
            timestamp=datetime.now(),
            sender="test",
            recipient="main_system",
            subject="Test thought exchange",
            content={"test": "This is a test message for memory service channel"}
        )
        
        memory_success = communicator.memory_service_channel.send_message(memory_message)
        logger.info(f"Memory Service Channel test: {'PASSED' if memory_success else 'FAILED'}")
        
        # Test Shared State Channel
        logger.info("Testing Shared State Channel...")
        shared_state_message = CommunicationMessage(
            id="test_shared_001",
            type=CommunicationType.STATUS_UPDATE,
            priority=Priority.HIGH,
            timestamp=datetime.now(),
            sender="test",
            recipient="main_system",
            subject="Test status update",
            content={"test": "This is a test message for shared state channel"}
        )
        
        shared_state_success = communicator.shared_state_channel.send_message(shared_state_message)
        logger.info(f"Shared State Channel test: {'PASSED' if shared_state_success else 'FAILED'}")
        
        # Test Message Queue Channel
        logger.info("Testing Message Queue Channel...")
        queue_message = CommunicationMessage(
            id="test_queue_001",
            type=CommunicationType.TASK_RESULT,
            priority=Priority.MEDIUM,
            timestamp=datetime.now(),
            sender="test",
            recipient="main_system",
            subject="Test task result",
            content={"test": "This is a test message for message queue channel"}
        )
        
        queue_success = communicator.message_queue_channel.send_message(queue_message)
        logger.info(f"Message Queue Channel test: {'PASSED' if queue_success else 'FAILED'}")
        
        # Wait a short time to allow any async operations to complete
        logger.info("Waiting for async operations to complete...")
        await asyncio.sleep(0.5)
        
        # Stop the communicator
        logger.info("Stopping RAVANACommunicator...")
        await communicator.stop()
        logger.info("RAVANACommunicator stopped")
        
        logger.info("Communication channels test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during communication channels test: {e}")
        logger.exception("Full traceback:")
        return False


async def main():
    """Main function with timeout."""
    try:
        # Run the test with a timeout of 10 seconds
        logger.info("Starting communication channels test with 10 second timeout...")
        result = await asyncio.wait_for(test_communication_channels(), timeout=10.0)
        logger.info("Communication channels test completed successfully")
        return result
    except asyncio.TimeoutError:
        logger.error("Communication channels test timed out after 10 seconds")
        return False
    except Exception as e:
        logger.error(f"Error in communication channels test: {e}")
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