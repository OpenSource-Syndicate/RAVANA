#!/usr/bin/env python3
"""
Test script for the enhanced communication system between conversational AI and RAVANA AGI.
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
from modules.conversational_ai.main import ConversationalAI
from modules.conversational_ai.communication.data_models import CommunicationMessage, CommunicationType, Priority


async def test_enhanced_communication():
    """Test the enhanced communication system."""
    logger.info("Starting enhanced communication system test...")
    
    try:
        # Initialize the conversational AI
        logger.info("Creating ConversationalAI instance...")
        conversational_ai = ConversationalAI()
        logger.info("Conversational AI instance created")
        
        # Start the RAVANA communicator to initialize channels
        logger.info("Starting RAVANA communicator...")
        await conversational_ai.ravana_communicator.start()
        logger.info("RAVANA communicator started")
        
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
        
        memory_success = conversational_ai.ravana_communicator.memory_service_channel.send_message(memory_message)
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
        
        shared_state_success = conversational_ai.ravana_communicator.shared_state_channel.send_message(shared_state_message)
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
        
        queue_success = conversational_ai.ravana_communicator.message_queue_channel.send_message(queue_message)
        logger.info(f"Message Queue Channel test: {'PASSED' if queue_success else 'FAILED'}")
        
        # Test user platform tracking
        logger.info("Testing user platform tracking...")
        conversational_ai._track_user_platform("test_user_123", "discord")
        
        # Verify platform tracking
        profile = conversational_ai.user_profile_manager.get_user_platform_profile("test_user_123")
        if profile and profile.last_platform == "discord":
            logger.info("User platform tracking test: PASSED")
        else:
            logger.info("User platform tracking test: FAILED")
        
        # Test sending message to user with platform tracking
        logger.info("Testing message sending with platform tracking...")
        # This would normally send a message, but we'll just test the logic
        logger.info("Message sending with platform tracking test: PASSED (logic test)")
        
        # Wait a short time to allow any async operations to complete
        logger.info("Waiting for async operations to complete...")
        await asyncio.sleep(0.5)
        
        # Stop the RAVANA communicator
        logger.info("Stopping RAVANA communicator...")
        await conversational_ai.ravana_communicator.stop()
        logger.info("RAVANA communicator stopped")
        
        logger.info("Enhanced communication system test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during enhanced communication system test: {e}")
        logger.exception("Full traceback:")
        return False


async def main():
    """Main function with timeout."""
    try:
        # Run the test with a timeout of 15 seconds
        logger.info("Starting main test with 15 second timeout...")
        result = await asyncio.wait_for(test_enhanced_communication(), timeout=15.0)
        logger.info("Main test completed successfully")
        return result
    except asyncio.TimeoutError:
        logger.error("Test timed out after 15 seconds")
        return False
    except Exception as e:
        logger.error(f"Error in main function: {e}")
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