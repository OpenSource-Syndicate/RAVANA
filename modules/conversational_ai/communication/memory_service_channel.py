import asyncio
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import RAVANA's memory service
try:
    from services.memory_service import MemoryService
    MEMORY_SERVICE_AVAILABLE = True
except ImportError:
    MEMORY_SERVICE_AVAILABLE = False
    logging.warning("MemoryService not available, using simulated memory service")

from .data_models import CommunicationMessage, CommunicationType, Priority

logger = logging.getLogger(__name__)


class MemoryServiceChannel:
    """Channel that uses RAVANA's memory service for message storage."""
    
    def __init__(self, channel_name: str = "conversational_ai_memory_channel"):
        """
        Initialize the Memory Service Channel.
        
        Args:
            channel_name: Name of the channel for identification
        """
        self.channel_name = channel_name
        self.running = False
        self._shutdown = asyncio.Event()
        
        # Initialize memory service or simulated version
        if MEMORY_SERVICE_AVAILABLE:
            self.memory_service = MemoryService()
        else:
            self.memory_service = SimulatedMemoryService()
            
        logger.info(f"Memory Service Channel '{self.channel_name}' initialized")
    
    async def start(self):
        """Start the Memory Service Channel."""
        if self._shutdown.is_set():
            return
        self.running = True
        logger.info(f"Memory Service Channel '{self.channel_name}' started")
    
    async def stop(self):
        """Stop the Memory Service Channel."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self.running = False
        logger.info(f"Memory Service Channel '{self.channel_name}' stopped")
    
    def send_message(self, message: CommunicationMessage) -> bool:
        """
        Send a message through the Memory Service Channel.
        
        Args:
            message: Message to send
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if self._shutdown.is_set():
            return False
            
        try:
            # For now, we'll just log that we're trying to send a message
            # In a real implementation, we would store this in the memory service
            logger.info(f"Message {message.id} would be stored in memory service")
            return True
            
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error sending message through Memory Service Channel: {e}")
            return False
    
    def get_messages_for_recipient(self, recipient: str, limit: int = 100) -> List[CommunicationMessage]:
        """
        Retrieve messages for a specific recipient.
        
        Args:
            recipient: Recipient identifier
            limit: Maximum number of messages to retrieve
            
        Returns:
            List of CommunicationMessage objects
        """
        try:
            # In a real implementation, we would query the memory service
            # For now, we'll return an empty list since this is a simulated implementation
            return []
            
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error retrieving messages for recipient {recipient}: {e}")
            return []
    
    def acknowledge_message(self, message_id: str) -> bool:
        """
        Acknowledge receipt of a message.
        
        Args:
            message_id: ID of the message to acknowledge
            
        Returns:
            bool: True if acknowledged successfully, False otherwise
        """
        try:
            # In a real implementation, we would mark the message as acknowledged
            # For now, we'll just log it
            logger.info(f"Message {message_id} acknowledged")
            return True
            
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error acknowledging message {message_id}: {e}")
            return False


class SimulatedMemoryService:
    """Simulated memory service for testing when actual service is not available."""
    
    def __init__(self):
        """Initialize the simulated memory service."""
        self.storage = {}
        logger.info("Simulated Memory Service initialized")
    
    def store(self, key: str, data: Dict[str, Any], collection: str = "default"):
        """
        Store data in the simulated memory service.
        
        Args:
            key: Key for the data
            data: Data to store
            collection: Collection name
        """
        if collection not in self.storage:
            self.storage[collection] = {}
        self.storage[collection][key] = data
        logger.debug(f"Stored data with key '{key}' in collection '{collection}'")
    
    def set(self, key: str, data: Dict[str, Any]):
        """
        Set data in the simulated memory service (for compatibility with SharedState interface).
        
        Args:
            key: Key for the data
            data: Data to store
        """
        self.store(key, data, "default")
    
    def retrieve(self, key: str, collection: str = "default") -> Optional[Dict[str, Any]]:
        """
        Retrieve data from the simulated memory service.
        
        Args:
            key: Key for the data
            collection: Collection name
            
        Returns:
            Data if found, None otherwise
        """
        if collection in self.storage and key in self.storage[collection]:
            return self.storage[collection][key]
        return None
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from the simulated memory service (for compatibility with SharedState interface).
        
        Args:
            key: Key for the data
            
        Returns:
            Data if found, None otherwise
        """
        return self.retrieve(key, "default")
    
    def query(self, collection: str = "default", filter_func=None) -> List[Dict[str, Any]]:
        """
        Query data from the simulated memory service.
        
        Args:
            collection: Collection name
            filter_func: Function to filter results
            
        Returns:
            List of matching data entries
        """
        if collection not in self.storage:
            return []
        
        results = list(self.storage[collection].values())
        if filter_func:
            results = [item for item in results if filter_func(item)]
        return results
