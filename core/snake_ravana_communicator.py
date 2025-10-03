"""
Snake RAVANA Communicator

This module handles communication between the Snake Agent and the main RAVANA system,
providing structured protocols for proposals, approvals, and status updates.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from core.config import Config

logger = logging.getLogger(__name__)


class CommunicationType(Enum):
    """Types of communication messages"""
    PROPOSAL = "proposal"
    STATUS_UPDATE = "status_update"
    EMERGENCY = "emergency"
    REQUEST_APPROVAL = "request_approval"
    EXPERIMENT_RESULT = "experiment_result"
    LEARNING_UPDATE = "learning_update"


class Priority(Enum):
    """Message priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CommunicationMessage:
    """Structured communication message"""
    id: str
    type: CommunicationType
    priority: Priority
    timestamp: datetime
    sender: str = "snake_agent"
    recipient: str = "ravana_main"
    subject: str = ""
    content: Dict[str, Any] = None
    requires_response: bool = False
    response_timeout: Optional[int] = None

    def __post_init__(self):
        if self.content is None:
            self.content = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "sender": self.sender,
            "recipient": self.recipient,
            "subject": self.subject,
            "content": self.content,
            "requires_response": self.requires_response,
            "response_timeout": self.response_timeout
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CommunicationMessage':
        return cls(
            id=data["id"],
            type=CommunicationType(data["type"]),
            priority=Priority(data["priority"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            sender=data.get("sender", "snake_agent"),
            recipient=data.get("recipient", "ravana_main"),
            subject=data.get("subject", ""),
            content=data.get("content", {}),
            requires_response=data.get("requires_response", False),
            response_timeout=data.get("response_timeout")
        )


class CommunicationChannel(ABC):
    """Base class for communication channels"""

    def __init__(self, channel_name: str):
        self.channel_name = channel_name
        self.message_queue: List[CommunicationMessage] = []
        self.sent_messages: Dict[str, CommunicationMessage] = {}
        self.received_messages: Dict[str, CommunicationMessage] = {}

    @abstractmethod
    async def send_message(self, message: CommunicationMessage) -> bool:
        """Send a message through this channel"""
        pass

    @abstractmethod
    async def receive_message(self) -> Optional[CommunicationMessage]:
        """Receive a message from this channel"""
        pass

    @abstractmethod
    async def get_response(self, message_id: str, timeout: int = 30) -> Optional[CommunicationMessage]:
        """Wait for a response to a specific message"""
        pass


class MemoryServiceChannel(CommunicationChannel):
    """Communication channel through RAVANA's memory service"""

    def __init__(self, agi_system):
        super().__init__("memory_service")
        self.agi_system = agi_system
        self.memory_service = agi_system.memory_service

    async def send_message(self, message: CommunicationMessage) -> bool:
        """Send message via memory service"""
        try:
            # Store message in memory service for RAVANA to process
            memory_data = {
                "source": "snake_agent",
                "type": "agent_communication",
                "priority": message.priority.value,
                "message": message.to_dict(),
                "timestamp": message.timestamp.isoformat()
            }

            # Use memory service to store the communication
            await self.memory_service.add_episodic_memory(
                content=json.dumps(memory_data),
                metadata=memory_data,
                embedding_text=f"Snake Agent communication: {message.subject}"
            )

            self.sent_messages[message.id] = message
            logger.info(f"Sent message {message.id} via memory service")
            return True

        except Exception as e:
            logger.error(f"Failed to send message via memory service: {e}")
            return False

    async def receive_message(self) -> Optional[CommunicationMessage]:
        """Check for messages from RAVANA via memory service"""
        try:
            # Query memory service for messages addressed to snake agent
            memories = await self.memory_service.search_memories(
                query="message for snake_agent",
                limit=10
            )

            for memory in memories:
                if memory.metadata and memory.metadata.get("recipient") == "snake_agent":
                    message_data = memory.metadata
                    if message_data["id"] not in self.received_messages:
                        message = CommunicationMessage.from_dict(message_data)
                        self.received_messages[message.id] = message
                        return message

            return None

        except Exception as e:
            logger.error(f"Error receiving message via memory service: {e}")
            return None

    async def get_response(self, message_id: str, timeout: int = 30) -> Optional[CommunicationMessage]:
        """Wait for response to a specific message"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = await self.receive_message()
            if response and response.content.get("response_to") == message_id:
                return response

            await asyncio.sleep(1)

        return None


class SharedStateChannel(CommunicationChannel):
    """Communication channel through RAVANA's shared state"""

    def __init__(self, agi_system):
        super().__init__("shared_state")
        self.agi_system = agi_system
        self.shared_state = agi_system.shared_state

    async def send_message(self, message: CommunicationMessage) -> bool:
        """Send message via shared state"""
        try:
            # Add message to shared state for RAVANA to process
            if not hasattr(self.shared_state, 'snake_communications'):
                self.shared_state.snake_communications = []

            self.shared_state.snake_communications.append(message.to_dict())

            # Limit queue size
            if len(self.shared_state.snake_communications) > 50:
                self.shared_state.snake_communications = self.shared_state.snake_communications[-50:]

            self.sent_messages[message.id] = message
            logger.info(f"Sent message {message.id} via shared state")
            return True

        except Exception as e:
            logger.error(f"Failed to send message via shared state: {e}")
            return False

    async def receive_message(self) -> Optional[CommunicationMessage]:
        """Check for messages from RAVANA via shared state"""
        try:
            if hasattr(self.shared_state, 'ravana_to_snake_messages'):
                messages = self.shared_state.ravana_to_snake_messages

                for message_data in messages:
                    if message_data["id"] not in self.received_messages:
                        message = CommunicationMessage.from_dict(message_data)
                        self.received_messages[message.id] = message
                        return message

            return None

        except Exception as e:
            logger.error(f"Error receiving message via shared state: {e}")
            return None

    async def get_response(self, message_id: str, timeout: int = 30) -> Optional[CommunicationMessage]:
        """Wait for response to a specific message via shared state"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            response = await self.receive_message()
            if response and response.content.get("response_to") == message_id:
                return response

            await asyncio.sleep(1)

        return None


class LoggingChannel(CommunicationChannel):
    """Communication channel through structured logging"""

    def __init__(self, agi_system):
        super().__init__("logging")
        self.agi_system = agi_system

    async def send_message(self, message: CommunicationMessage) -> bool:
        """Send message via structured logging"""
        try:
            log_data = {
                "snake_agent_message": message.to_dict(),
                "priority": message.priority.value,
                "requires_attention": message.priority in [Priority.HIGH, Priority.CRITICAL]
            }

            if message.priority == Priority.CRITICAL:
                logger.critical(
                    f"SNAKE AGENT CRITICAL: {message.subject}", extra=log_data)
            elif message.priority == Priority.HIGH:
                logger.warning(
                    f"SNAKE AGENT HIGH: {message.subject}", extra=log_data)
            else:
                logger.info(f"SNAKE AGENT: {message.subject}", extra=log_data)

            self.sent_messages[message.id] = message
            return True

        except Exception as e:
            logger.error(f"Failed to send message via logging: {e}")
            return False

    async def receive_message(self) -> Optional[CommunicationMessage]:
        """Logging channel is send-only"""
        return None

    async def get_response(self, message_id: str, timeout: int = 30) -> Optional[CommunicationMessage]:
        """Logging channel is send-only, so no response capability"""
        # Logging channel is one-way only, so it can't receive responses
        logger.debug(f"Logging channel cannot receive response for message {message_id}")
        return None


class SnakeRavanaCommunicator:
    """Main communicator for Snake Agent to RAVANA system
    
    This class manages communication between the Snake Agent and the main RAVANA system,
    providing structured protocols for proposals, approvals, status updates, and other
    inter-agent communications.
    """

    def __init__(self, reasoning_llm, agi_system):
        self.reasoning_llm = reasoning_llm
        self.agi_system = agi_system

        # Initialize communication channels
        self.channels = {
            "memory_service": MemoryServiceChannel(agi_system),
            "shared_state": SharedStateChannel(agi_system),
            "logging": LoggingChannel(agi_system)
        }

        self.primary_channel = Config().SNAKE_COMM_CHANNEL
        self.message_counter = 0
        self.pending_responses: Dict[str, CommunicationMessage] = {}

    async def send_communication(self, comm_item: Dict[str, Any]) -> bool:
        """Send a communication item to RAVANA
        
        Args:
            comm_item: Dictionary containing communication details with keys:
                - type: CommunicationType value (proposal, status_update, etc.)
                - priority: Priority level (low, medium, high, critical)
                - content: The actual message content
                - other optional metadata
        
        Returns:
            bool: True if communication was successfully sent, False otherwise
        """
        if not isinstance(comm_item, dict):
            logger.error(f"Invalid communication item type: {type(comm_item)}, expected dict")
            return False

        if not comm_item:
            logger.error("Empty communication item provided")
            return False

        try:
            # Create structured message
            message = await self._create_message_from_item(comm_item)

            # Determine communication strategy
            strategy = await self._plan_communication_strategy(message)

            # Send via appropriate channels
            success = await self._send_via_channels(message, strategy)

            # Handle response if required
            if message.requires_response:
                self.pending_responses[message.id] = message

            logger.info(
                f"Communication sent: {message.id} - {message.subject}")
            return success

        except ValueError as e:
            logger.error(f"Value error in communication: {e}")
            return False
        except KeyError as e:
            logger.error(f"Missing required key in communication item: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending communication: {e}")
            logger.exception("Full traceback:")  # Log the full stack trace
            return False

    async def _create_message_from_item(self, comm_item: Dict[str, Any]) -> CommunicationMessage:
        """Create structured message from communication item
        
        Args:
            comm_item: Dictionary containing communication details
            
        Returns:
            CommunicationMessage: Structured communication message
            
        Raises:
            ValueError: If the communication item contains invalid types or priorities
        """
        self.message_counter += 1

        # Validate and extract message type
        type_str = comm_item.get("type", "status_update")
        try:
            message_type = CommunicationType(type_str)
        except ValueError:
            logger.warning(f"Invalid communication type '{type_str}', defaulting to STATUS_UPDATE")
            message_type = CommunicationType.STATUS_UPDATE

        # Validate and extract priority
        priority_str = comm_item.get("priority", "medium")
        try:
            priority = Priority(priority_str)
        except ValueError:
            logger.warning(f"Invalid priority '{priority_str}', defaulting to MEDIUM")
            priority = Priority.MEDIUM

        message = CommunicationMessage(
            id=f"snake_{int(time.time())}_{self.message_counter}",
            type=message_type,
            priority=priority,
            timestamp=datetime.now(),
            subject=self._generate_subject(comm_item),
            content=comm_item,
            requires_response=self._requires_response(comm_item),
            response_timeout=self._get_response_timeout(priority)
        )

        return message

    def _generate_subject(self, comm_item: Dict[str, Any]) -> str:
        """Generate subject line for communication"""
        comm_type = comm_item.get("type", "update")

        if comm_type == "experiment_result":
            experiment_id = comm_item.get("experiment_id", "unknown")
            success = comm_item.get("result", {}).get("success", False)
            status = "Success" if success else "Failed"
            return f"Experiment {experiment_id}: {status}"

        elif comm_type == "proposal":
            return f"Code Improvement Proposal: {comm_item.get('file_path', 'Unknown')}"

        elif comm_type == "emergency":
            return f"Emergency Alert: {comm_item.get('issue', 'Critical Issue')}"

        else:
            return f"Snake Agent {comm_type.replace('_', ' ').title()}"

    def _requires_response(self, comm_item: Dict[str, Any]) -> bool:
        """Determine if communication requires response"""
        comm_type = comm_item.get("type", "")
        priority = comm_item.get("priority", "medium")

        return (comm_type == "request_approval" or
                priority == "critical" or
                Config().SNAKE_APPROVAL_REQUIRED)

    def _get_response_timeout(self, priority: Priority) -> int:
        """Get response timeout based on priority"""
        timeouts = {
            Priority.CRITICAL: 300,  # 5 minutes
            Priority.HIGH: 1800,     # 30 minutes
            Priority.MEDIUM: 3600,   # 1 hour
            Priority.LOW: 7200       # 2 hours
        }
        return timeouts.get(priority, 3600)

    async def _plan_communication_strategy(self, message: CommunicationMessage) -> Dict[str, Any]:
        """Plan communication strategy using reasoning LLM
        
        Args:
            message: The message for which to plan a communication strategy
            
        Returns:
            Dict[str, Any]: Strategy dictionary with channels, retry count, and escalation flag
        """
        try:
            strategy_input = {
                "message": message.to_dict(),
                "available_channels": list(self.channels.keys()),
                "primary_channel": self.primary_channel,
                "priority_threshold": Config().SNAKE_COMM_PRIORITY_THRESHOLD
            }

            # Check if reasoning LLM has the required method
            if not hasattr(self.reasoning_llm, 'plan_communication'):
                logger.warning("reasoning_llm does not have plan_communication method, using fallback")
                raise AttributeError("reasoning_llm.plan_communication method not available")

            strategy = await self.reasoning_llm.plan_communication(strategy_input)

            # Validate the returned strategy
            if not isinstance(strategy, dict):
                logger.warning(f"Invalid strategy returned from reasoning_llm: {strategy}, using fallback")
                raise TypeError("strategy must be a dictionary")

            return {
                "channels": strategy.get("channels", [self.primary_channel]),
                "retry_count": strategy.get("retry_count", 1),
                "escalation": strategy.get("escalation", False)
            }

        except (AttributeError, TypeError) as e:
            logger.error(f"Strategy planning failed due to LLM configuration: {e}")
            # Fallback strategy
            return {
                "channels": [self.primary_channel, "logging"],
                "retry_count": 1,
                "escalation": message.priority in [Priority.HIGH, Priority.CRITICAL]
            }
        except Exception as e:
            logger.error(f"Unexpected error planning communication strategy: {e}")
            # General fallback strategy
            return {
                "channels": [self.primary_channel, "logging"],
                "retry_count": 1,
                "escalation": message.priority in [Priority.HIGH, Priority.CRITICAL]
            }

    async def _send_via_channels(self, message: CommunicationMessage,
                                 strategy: Dict[str, Any]) -> bool:
        """Send message via specified channels
        
        Args:
            message: The message to send
            strategy: Strategy dictionary with channels, retry count, etc.
            
        Returns:
            bool: True if at least one channel successfully sent the message
        """
        channels_to_use = strategy.get("channels", [self.primary_channel])
        retry_count = strategy.get("retry_count", 1)

        if not channels_to_use:
            logger.warning("No channels specified in strategy, using primary channel")
            channels_to_use = [self.primary_channel]

        success_count = 0

        for channel_name in channels_to_use:
            if channel_name not in self.channels:
                logger.error(f"Channel {channel_name} not available, skipping")
                continue

            channel = self.channels[channel_name]

            for attempt in range(retry_count):
                try:
                    send_result = await channel.send_message(message)
                    if send_result:
                        success_count += 1
                        logger.info(f"Successfully sent message via {channel_name} on attempt {attempt + 1}")
                        break
                    else:
                        logger.warning(
                            f"Failed to send via {channel_name}, attempt {attempt + 1}")
                        if attempt < retry_count - 1:  # Don't sleep after the last attempt
                            await asyncio.sleep(1)
                except asyncio.TimeoutError:
                    logger.error(f"Timeout sending via {channel_name}, attempt {attempt + 1}")
                except Exception as e:
                    logger.error(f"Unexpected error sending via {channel_name}: {e}")
                    logger.exception("Full traceback:")  # Log the full stack trace

        # Consider successful if at least one channel worked
        return success_count > 0

    async def check_for_responses(self) -> List[CommunicationMessage]:
        """Check for responses from RAVANA
        
        This method polls all available communication channels to check for
        responses to previously sent messages.
        
        Returns:
            List[CommunicationMessage]: List of received response messages
        """
        responses = []

        try:
            # Check each channel for responses
            for channel in self.channels.values():
                response = await channel.receive_message()
                if response:
                    responses.append(response)

                    # Remove from pending if this is a response
                    response_to = response.content.get("response_to")
                    if response_to and response_to in self.pending_responses:
                        del self.pending_responses[response_to]
                        logger.info(
                            f"Received response to message {response_to}")

            # Check for timed out responses
            await self._handle_response_timeouts()

        except Exception as e:
            logger.error(f"Error checking for responses: {e}")

        return responses

    async def _handle_response_timeouts(self):
        """Handle response timeouts"""
        current_time = datetime.now()
        timed_out_messages = []

        for message_id, message in self.pending_responses.items():
            if message.response_timeout:
                timeout_time = message.timestamp + \
                    timedelta(seconds=message.response_timeout)
                if current_time > timeout_time:
                    timed_out_messages.append(message_id)

        for message_id in timed_out_messages:
            message = self.pending_responses.pop(message_id)
            logger.warning(
                f"Response timeout for message {message_id}: {message.subject}")

            # Send timeout notification if critical
            if message.priority == Priority.CRITICAL:
                await self._send_timeout_notification(message)

    async def _send_timeout_notification(self, original_message: CommunicationMessage):
        """Send timeout notification for critical messages"""
        timeout_message = CommunicationMessage(
            id=f"timeout_{original_message.id}",
            type=CommunicationType.EMERGENCY,
            priority=Priority.CRITICAL,
            timestamp=datetime.now(),
            subject=f"Response Timeout: {original_message.subject}",
            content={
                "type": "timeout_notification",
                "original_message_id": original_message.id,
                "timeout_duration": original_message.response_timeout
            },
            requires_response=False
        )

        # Send via logging channel for immediate attention
        await self.channels["logging"].send_message(timeout_message)

    def get_communication_status(self) -> Dict[str, Any]:
        """Get current communication status
        
        Returns:
            Dict[str, Any]: Dictionary containing communication status with keys:
                - pending_responses: Number of messages waiting for responses
                - pending_message_ids: List of IDs of messages waiting for responses
                - available_channels: List of available communication channels
                - primary_channel: Current primary communication channel
                - message_count: Total number of messages sent
        """
        return {
            "pending_responses": len(self.pending_responses),
            "pending_message_ids": list(self.pending_responses.keys()),
            "available_channels": list(self.channels.keys()),
            "primary_channel": self.primary_channel,
            "message_count": self.message_counter
        }
