import asyncio
import logging
import threading
from typing import Dict, Any, Optional, Callable
from datetime import timedelta

# Import RAVANA's shared state if available
try:
    from core.state import SharedState
    SHARED_STATE_AVAILABLE = True
except ImportError:
    SHARED_STATE_AVAILABLE = False
    logging.warning("SharedState not available, using simulated shared state")

from .data_models import CommunicationMessage, CommunicationType, Priority

logger = logging.getLogger(__name__)


class SharedStateChannel:
    """Channel that uses RAVANA's shared state for real-time communication."""

    def __init__(self, channel_name: str = "conversational_ai_shared_state_channel"):
        """
        Initialize the Shared State Channel.

        Args:
            channel_name: Name of the channel for identification
        """
        self.channel_name = channel_name
        self.running = False
        self._shutdown = asyncio.Event()

        # Message expiration time (5 minutes)
        self.message_expiration = timedelta(minutes=5)

        # Callbacks for message handling
        self.message_callbacks = {}

        # Initialize shared state or simulated version
        if SHARED_STATE_AVAILABLE:
            # Create a basic initial mood for the shared state
            initial_mood = {
                "happiness": 0.5,
                "sadness": 0.1,
                "anger": 0.1,
                "fear": 0.1,
                "surprise": 0.1,
                "disgust": 0.1
            }
            self.shared_state = SharedState(initial_mood)
        else:
            self.shared_state = SimulatedSharedState()

        # Thread-safe message queue for incoming messages
        self.incoming_messages = asyncio.Queue()

        logger.info(f"Shared State Channel '{self.channel_name}' initialized")

    async def start(self):
        """Start the Shared State Channel."""
        if self._shutdown.is_set():
            return
        self.running = True

        # Start message processing loop
        self._processing_task = asyncio.create_task(self._process_messages())

        logger.info(f"Shared State Channel '{self.channel_name}' started")

    async def stop(self):
        """Stop the Shared State Channel."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self.running = False

        # Cancel processing task
        if hasattr(self, '_processing_task'):
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Shared State Channel '{self.channel_name}' stopped")

    async def _process_messages(self):
        """Process messages from shared state."""
        while self.running and not self._shutdown.is_set():
            try:
                # Check for new messages in shared state
                await self._check_for_new_messages()

                # Process any incoming messages
                try:
                    message = await asyncio.wait_for(self.incoming_messages.get(), timeout=0.1)
                    if not self._shutdown.is_set():
                        await self._handle_incoming_message(message)
                except asyncio.TimeoutError:
                    # No messages to process, continue loop
                    continue

            except Exception as e:
                if not self._shutdown.is_set():
                    logger.error(
                        f"Error in shared state message processing loop: {e}")
                    await asyncio.sleep(1)  # Prevent tight loop on error
                else:
                    break  # Exit loop if shutdown is set

    async def _check_for_new_messages(self):
        """Check for new messages in shared state."""
        try:
            # In a real implementation, we would monitor shared state for new messages
            # For now, we'll just check periodically
            pass
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error checking for new messages: {e}")

    async def _handle_incoming_message(self, message: CommunicationMessage):
        """
        Handle an incoming message.

        Args:
            message: Incoming message to handle
        """
        try:
            # Get message type
            message_type = message.type.value

            # Call appropriate callback if registered
            if message_type in self.message_callbacks:
                callback = self.message_callbacks[message_type]
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            else:
                logger.debug(
                    f"No callback registered for message type: {message_type}")

        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error handling incoming message: {e}")

    def send_message(self, message: CommunicationMessage) -> bool:
        """
        Send a message through the Shared State Channel.

        Args:
            message: Message to send

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if self._shutdown.is_set():
            return False

        try:
            # In a real implementation, this would be more complex, maybe with pub/sub
            # For simulation, we'll store it under a key related to the recipient.
            recipient_key = f"{self.channel_name}:{message.recipient}"
            # We'll just overwrite the last message for a recipient for simplicity
            self.shared_state.set(recipient_key, message.to_dict())
            logger.info(
                f"Message {message.id} sent through shared state channel to {message.recipient}")
            return True

        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(
                    f"Error sending message through Shared State Channel: {e}")
            return False

    def register_message_callback(self, message_type: str, callback: Callable):
        """
        Register a callback for handling specific message types.

        Args:
            message_type: Type of message to handle
            callback: Function to call when message is received
        """
        self.message_callbacks[message_type] = callback
        logger.info(f"Registered callback for message type: {message_type}")

    def remove_expired_messages(self):
        """Remove expired messages from shared state."""
        try:
            # In a real implementation, we would check expiration keys and remove expired messages
            # For now, we'll just log that this should be done
            logger.debug("Checking for expired messages (simulated)")
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error removing expired messages: {e}")


class SimulatedSharedState:
    """Simulated shared state for testing when actual service is not available."""

    def __init__(self):
        """Initialize the simulated shared state."""
        self.storage = {}
        self.lock = threading.Lock()
        logger.info("Simulated Shared State initialized")

    def set(self, key: str, value: Any):
        """
        Set a value in the simulated shared state.

        Args:
            key: Key for the value
            value: Value to store
        """
        with self.lock:
            self.storage[key] = value
            logger.debug(f"Set key '{key}' in shared state")

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the simulated shared state.

        Args:
            key: Key for the value

        Returns:
            Value if found, None otherwise
        """
        with self.lock:
            return self.storage.get(key)

    def store(self, key: str, data: Dict[str, Any], collection: str = "default"):
        """
        Store data in the simulated shared state (for compatibility with MemoryService interface).

        Args:
            key: Key for the data
            data: Data to store
            collection: Collection name
        """
        self.set(key, data)

    def retrieve(self, key: str, collection: str = "default") -> Optional[Dict[str, Any]]:
        """
        Retrieve data from the simulated shared state (for compatibility with MemoryService interface).

        Args:
            key: Key for the data
            collection: Collection name

        Returns:
            Data if found, None otherwise
        """
        return self.get(key)

    def delete(self, key: str):
        """
        Delete a value from the simulated shared state.

        Args:
            key: Key to delete
        """
        with self.lock:
            if key in self.storage:
                del self.storage[key]
                logger.debug(f"Deleted key '{key}' from shared state")
