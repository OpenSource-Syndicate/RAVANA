import asyncio
import logging
import uuid
from typing import Dict, Any
from datetime import datetime

# Import the new communication channels
from .memory_service_channel import MemoryServiceChannel
from .shared_state_channel import SharedStateChannel
from .message_queue_channel import MessageQueueChannel
from .data_models import CommunicationMessage, CommunicationType, Priority

logger = logging.getLogger(__name__)


class RAVANACommunicator:
    def __init__(self, channel: str, conversational_ai):
        """
        Initialize the RAVANA communication bridge.

        Args:
            channel: IPC channel name for communication
            conversational_ai: Reference to the main ConversationalAI instance
        """
        self.channel = channel
        self.conversational_ai = conversational_ai
        self.running = False
        self.message_queue = asyncio.Queue()

        # For graceful shutdown
        self._shutdown = asyncio.Event()

        # Initialize communication channels
        self.memory_service_channel = MemoryServiceChannel(f"{channel}_memory")
        self.shared_state_channel = SharedStateChannel(
            f"{channel}_shared_state")
        self.message_queue_channel = MessageQueueChannel(f"{channel}_queue")

        # Register callbacks for handling messages from RAVANA
        self._register_message_callbacks()

        # In a real implementation, this would connect to an actual IPC system
        # For now, we'll simulate the communication
        self.pending_tasks = {}

    async def start(self):
        """Start the RAVANA communication bridge."""
        if self._shutdown.is_set():
            return
        self.running = True

        # Start communication channels
        await self.memory_service_channel.start()
        await self.shared_state_channel.start()
        await self.message_queue_channel.start()

        # Start message processing loop as a separate task
        self._processing_task = asyncio.create_task(self._process_messages())

        logger.info("RAVANA communication bridge started")

    async def stop(self):
        """Stop the RAVANA communication bridge."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self.running = False

        # Cancel processing task
        if hasattr(self, '_processing_task') and self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        # Stop communication channels
        await self.memory_service_channel.stop()
        await self.shared_state_channel.stop()
        await self.message_queue_channel.stop()

        logger.info("RAVANA communication bridge stopped")

    def _register_message_callbacks(self):
        """Register callbacks for handling messages from RAVANA."""
        # Register callbacks for different message types
        self.message_queue_channel.register_message_callback(
            CommunicationType.TASK_RESULT.value,
            self._handle_task_result_message
        )
        self.message_queue_channel.register_message_callback(
            CommunicationType.NOTIFICATION.value,
            self._handle_notification_message
        )
        self.message_queue_channel.register_message_callback(
            CommunicationType.USER_MESSAGE.value,
            self._handle_user_message_from_ravana_message
        )
        self.message_queue_channel.register_message_callback(
            CommunicationType.THOUGHT_EXCHANGE.value,
            self._handle_thought_exchange_message
        )

        # Register callbacks for shared state channel as well
        self.shared_state_channel.register_message_callback(
            CommunicationType.TASK_RESULT.value,
            self._handle_task_result_message
        )
        self.shared_state_channel.register_message_callback(
            CommunicationType.NOTIFICATION.value,
            self._handle_notification_message
        )
        self.shared_state_channel.register_message_callback(
            CommunicationType.USER_MESSAGE.value,
            self._handle_user_message_from_ravana_message
        )
        self.shared_state_channel.register_message_callback(
            CommunicationType.THOUGHT_EXCHANGE.value,
            self._handle_thought_exchange_message
        )

    async def _process_messages(self):
        """Process messages from RAVANA."""
        while self.running and not self._shutdown.is_set():
            try:
                # In a real implementation, this would receive messages from RAVANA
                # For now, we'll just process the message queue
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                    if not self._shutdown.is_set():
                        await self._handle_message(message)
                except asyncio.TimeoutError:
                    # No messages to process, continue loop
                    continue
                except Exception as e:
                    if not self._shutdown.is_set():
                        logger.error(f"Error processing message: {e}")

            except Exception as e:
                if not self._shutdown.is_set():
                    logger.error(f"Error in message processing loop: {e}")
                    await asyncio.sleep(1)  # Prevent tight loop on error
                else:
                    break  # Exit loop if shutdown is set
            # Check shutdown status periodically
            if self._shutdown.is_set():
                break

        logger.info("Message processing loop stopped")

    async def _handle_message(self, message: Dict[str, Any]):
        """
        Handle a message from RAVANA.

        Args:
            message: Message from RAVANA
        """
        if self._shutdown.is_set():
            return
        try:
            message_type = message.get("type")

            if message_type == "task_result":
                await self._handle_task_result(message)
            elif message_type == "notification":
                await self._handle_notification(message)
            elif message_type == "user_message":
                await self._handle_user_message_from_ravana(message)
            elif message_type == "thought_exchange":
                await self._handle_thought_exchange(message)
            else:
                if not self._shutdown.is_set():
                    logger.warning(f"Unknown message type: {message_type}")

        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error handling message: {e}")

    async def _handle_task_result(self, message: Dict[str, Any]):
        """
        Handle a task result from RAVANA.

        Args:
            message: Task result message
        """
        if self._shutdown.is_set():
            return
        try:
            user_id = message.get("user_id")
            task_id = message.get("task_id")
            result = message.get("result")

            # Send result to user
            notification = f"Task completed: {result}"
            await self.conversational_ai.send_message_to_user(user_id, notification)

            logger.info(f"Task result sent to user {user_id}")
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error handling task result: {e}")

    async def _handle_notification(self, message: Dict[str, Any]):
        """
        Handle a notification from RAVANA.

        Args:
            message: Notification message
        """
        if self._shutdown.is_set():
            return
        try:
            user_id = message.get("user_id")
            content = message.get("content")

            # Send notification to user
            await self.conversational_ai.send_message_to_user(user_id, content)

            logger.info(f"Notification sent to user {user_id}")
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error handling notification: {e}")

    async def _handle_user_message_from_ravana(self, message: Dict[str, Any]):
        """
        Handle a user message initiated by RAVANA.

        Args:
            message: User message from RAVANA
        """
        if self._shutdown.is_set():
            return
        try:
            user_id = message.get("user_id")
            content = message.get("content")
            platform = message.get("platform", "discord")

            # Send message to user
            # Note: This would typically go through the appropriate bot interface
            # For simplicity, we'll just log it
            logger.info(
                f"RAVANA wants to send message to user {user_id}: {content}")

        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error handling user message from RAVANA: {e}")

    async def _handle_thought_exchange(self, message: Dict[str, Any]):
        """
        Handle a thought exchange message from RAVANA.

        Args:
            message: Thought exchange message
        """
        if self._shutdown.is_set():
            return
        try:
            source = message.get("source")
            destination = message.get("destination")
            content = message.get("content", {})
            thought_type = content.get("thought_type")

            logger.info(
                f"Received thought exchange from {source} to {destination}: {thought_type}")

            # Process the thought based on its type
            if thought_type == "insight":
                # Process insight from RAVANA
                await self._process_ravana_insight(content)
            elif thought_type == "goal_adjustment":
                # Process goal adjustment from RAVANA
                await self._process_goal_adjustment(content)
            elif thought_type == "collaboration_proposal":
                # Process collaboration proposal from RAVANA
                await self._process_collaboration_proposal(content)
            # Add more thought types as needed

        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error handling thought exchange: {e}")

    async def _process_ravana_insight(self, content: Dict[str, Any]):
        """
        Process an insight from RAVANA.

        Args:
            content: Insight content
        """
        try:
            insight = content.get("payload", {})
            user_id = content.get("metadata", {}).get("user_id")

            if user_id:
                # Send insight to user
                message = f"I've had an insight I wanted to share: {insight.get('description', 'No description provided')}"
                await self.conversational_ai.send_message_to_user(user_id, message)

            logger.info(f"Processed RAVANA insight: {insight}")
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error processing RAVANA insight: {e}")

    async def _process_goal_adjustment(self, content: Dict[str, Any]):
        """
        Process a goal adjustment from RAVANA.

        Args:
            content: Goal adjustment content
        """
        try:
            adjustment = content.get("payload", {})
            user_id = content.get("metadata", {}).get("user_id")

            if user_id:
                # Notify user about goal adjustment
                message = f"I've adjusted my goals based on our conversation: {adjustment.get('reason', 'No reason provided')}"
                await self.conversational_ai.send_message_to_user(user_id, message)

            logger.info(f"Processed goal adjustment: {adjustment}")
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error processing goal adjustment: {e}")

    async def _process_collaboration_proposal(self, content: Dict[str, Any]):
        """
        Process a collaboration proposal from RAVANA.

        Args:
            content: Collaboration proposal content
        """
        try:
            proposal = content.get("payload", {})
            user_id = content.get("metadata", {}).get("user_id")

            if user_id:
                # Send collaboration proposal to user
                message = f"I have a collaboration idea: {proposal.get('description', 'No description provided')}"
                await self.conversational_ai.send_message_to_user(user_id, message)

            logger.info(f"Processed collaboration proposal: {proposal}")
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error processing collaboration proposal: {e}")

    def send_task_to_ravana(self, task: Dict[str, Any]):
        """
        Send a task to RAVANA.

        Args:
            task: Task to send to RAVANA
        """
        if self._shutdown.is_set():
            return
        try:
            # Create a communication message
            message_id = str(uuid.uuid4())
            communication_message = CommunicationMessage(
                id=message_id,
                type=CommunicationType.TASK_RESULT,
                priority=Priority.MEDIUM,
                timestamp=datetime.now(),
                sender="conversational_ai",
                recipient="main_system",
                subject=f"Task: {task.get('task_description', 'No description')}",
                content=task,
                requires_response=True,
                response_timeout=300  # 5 minutes
            )

            # Send through message queue channel for reliability
            success = self.message_queue_channel.send_message(
                communication_message)

            if success:
                # Add to pending tasks for tracking
                task_id = task.get("task_id", message_id)
                task["task_id"] = task_id
                task["status"] = "sent"
                task["sent_at"] = datetime.now().isoformat()
                self.pending_tasks[task_id] = task

                logger.info(
                    f"Task {task_id} sent to RAVANA: {task.get('task_description', 'No description')}")
            else:
                logger.error(
                    f"Failed to send task to RAVANA: {task.get('task_description', 'No description')}")

        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error sending task to RAVANA: {e}")

    def send_thought_to_ravana(self, thought: Dict[str, Any]):
        """
        Send a thought to RAVANA.

        Args:
            thought: Thought to send to RAVANA
        """
        if self._shutdown.is_set():
            return
        try:
            # Create a communication message
            message_id = str(uuid.uuid4())
            communication_message = CommunicationMessage(
                id=message_id,
                type=CommunicationType.THOUGHT_EXCHANGE,
                priority=Priority.LOW,
                timestamp=datetime.now(),
                sender="conversational_ai",
                recipient="main_system",
                subject=f"Thought: {thought.get('thought_type', 'unknown')}",
                content=thought,
                requires_response=False
            )

            # Send through memory service channel for persistence
            success = self.memory_service_channel.send_message(
                communication_message)

            if success:
                logger.info(
                    f"Thought sent to RAVANA: {thought.get('thought_type', 'unknown')}")
            else:
                logger.error(
                    f"Failed to send thought to RAVANA: {thought.get('thought_type', 'unknown')}")

        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error sending thought to RAVANA: {e}")

    def send_emotional_context_to_ravana(self, emotional_data: Dict[str, Any]):
        """
        Send emotional context to RAVANA.

        Args:
            emotional_data: Emotional context data to send to RAVANA
        """
        if self._shutdown.is_set():
            return
        try:
            # Create a communication message
            message_id = str(uuid.uuid4())
            communication_message = CommunicationMessage(
                id=message_id,
                type=CommunicationType.STATUS_UPDATE,
                priority=Priority.HIGH,
                timestamp=datetime.now(),
                sender="conversational_ai",
                recipient="main_system",
                subject=f"Emotional context update for user {emotional_data.get('user_id', 'unknown')}",
                content=emotional_data,
                requires_response=False
            )

            # Send through shared state channel for real-time communication
            success = self.shared_state_channel.send_message(
                communication_message)

            if success:
                logger.info(
                    f"Emotional context sent to RAVANA for user {emotional_data.get('user_id', 'unknown')}")
            else:
                logger.error(
                    f"Failed to send emotional context to RAVANA for user {emotional_data.get('user_id', 'unknown')}")

        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error sending emotional context to RAVANA: {e}")

    def send_notification_to_ravana(self, notification: Dict[str, Any]):
        """
        Send a notification to RAVANA.

        Args:
            notification: Notification to send to RAVANA
        """
        if self._shutdown.is_set():
            return
        try:
            # In a real implementation, this would send the notification through an IPC mechanism
            logger.info(f"Notification sent to RAVANA: {notification}")
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error sending notification to RAVANA: {e}")

    def notify_user(self, user_id: str, message: str, platform: str = None):
        """
        Notify a user through the appropriate platform.

        Args:
            user_id: User to notify
            message: Message to send
            platform: Platform to use (if None, will use user's last platform)
        """
        if self._shutdown.is_set():
            return
        try:
            # This would typically be called by RAVANA to notify users
            # We'll just log it for now
            logger.info(f"RAVANA notification for user {user_id}: {message}")

            # In a real implementation, this would send the message through the appropriate bot
            # For now, we'll just add it to the message queue to be processed
            notification = {
                "type": "notification",
                "user_id": user_id,
                "content": message,
                "platform": platform
            }

            # Process immediately rather than queueing
            if not self._shutdown.is_set():
                asyncio.create_task(self.conversational_ai.send_message_to_user(
                    user_id, message, platform))

        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error notifying user {user_id}: {e}")

    # New handler methods for the communication channels
    async def _handle_task_result_message(self, message: CommunicationMessage):
        """
        Handle a task result message from RAVANA.

        Args:
            message: Task result message
        """
        if self._shutdown.is_set():
            return
        try:
            await self._handle_task_result(message.content)
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error handling task result message: {e}")

    async def _handle_notification_message(self, message: CommunicationMessage):
        """
        Handle a notification message from RAVANA.

        Args:
            message: Notification message
        """
        if self._shutdown.is_set():
            return
        try:
            await self._handle_notification(message.content)
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error handling notification message: {e}")

    async def _handle_user_message_from_ravana_message(self, message: CommunicationMessage):
        """
        Handle a user message from RAVANA.

        Args:
            message: User message from RAVANA
        """
        if self._shutdown.is_set():
            return
        try:
            await self._handle_user_message_from_ravana(message.content)
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(
                    f"Error handling user message from RAVANA message: {e}")

    async def _handle_thought_exchange_message(self, message: CommunicationMessage):
        """
        Handle a thought exchange message from RAVANA.

        Args:
            message: Thought exchange message
        """
        if self._shutdown.is_set():
            return
        try:
            await self._handle_thought_exchange(message.content)
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error handling thought exchange message: {e}")
