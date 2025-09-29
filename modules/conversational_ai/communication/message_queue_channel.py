import asyncio
import logging
import json
import sqlite3
import os
from typing import Dict, Callable
from datetime import datetime
from enum import Enum
import queue

from .data_models import CommunicationMessage, CommunicationType, Priority

logger = logging.getLogger(__name__)


class MessageStatus(Enum):
    """Status of a message in the queue."""
    PENDING = "pending"
    PROCESSING = "processing"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"


class MessageQueueChannel:
    """Persistent message queue system with prioritization."""

    def __init__(self, channel_name: str = "conversational_ai_message_queue", db_path: str = None):
        """
        Initialize the Message Queue Channel.

        Args:
            channel_name: Name of the channel for identification
            db_path: Path to SQLite database for persistence (None for in-memory)
        """
        self.channel_name = channel_name
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), f"{channel_name}.db")
        self.running = False
        self._shutdown = asyncio.Event()

        # Callbacks for message handling
        self.message_callbacks = {}

        # Retry settings
        self.max_retries = 3
        self.retry_delay = 1  # seconds

        # Thread-safe message queues for each priority level
        self.queues = {
            Priority.CRITICAL: queue.PriorityQueue(),
            Priority.HIGH: queue.PriorityQueue(),
            Priority.MEDIUM: queue.PriorityQueue(),
            Priority.LOW: queue.PriorityQueue()
        }

        # Database connection
        self.db_conn = None
        self._init_database()

        # Processing task
        self._processing_task = None

        logger.info(
            f"Message Queue Channel '{self.channel_name}' initialized with DB: {self.db_path}")

    def _init_database(self):
        """Initialize the SQLite database for message persistence."""
        try:
            self.db_conn = sqlite3.connect(
                self.db_path, check_same_thread=False)
            self.db_conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    requires_response BOOLEAN NOT NULL,
                    response_timeout INTEGER,
                    status TEXT NOT NULL,
                    retry_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            self.db_conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_recipient ON messages(recipient)
            ''')
            self.db_conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_status ON messages(status)
            ''')
            self.db_conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_priority ON messages(priority)
            ''')
            self.db_conn.commit()
            logger.info("Message queue database initialized")
        except Exception as e:
            logger.error(f"Error initializing message queue database: {e}")
            raise

    async def start(self):
        """Start the Message Queue Channel."""
        if self._shutdown.is_set():
            return
        self.running = True

        # Load pending messages from database
        await self._load_pending_messages()

        # Start message processing loop
        self._processing_task = asyncio.create_task(self._process_messages())

        logger.info(f"Message Queue Channel '{self.channel_name}' started")

    async def stop(self):
        """Stop the Message Queue Channel."""
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self.running = False

        # Cancel processing task
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        # Close database connection
        if self.db_conn:
            self.db_conn.close()

        logger.info(f"Message Queue Channel '{self.channel_name}' stopped")

    async def _load_pending_messages(self):
        """Load pending messages from database into queues."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT id, type, priority, timestamp, sender, recipient, subject, content,
                       requires_response, response_timeout, retry_count
                FROM messages WHERE status = ? ORDER BY priority DESC, timestamp ASC
            ''', (MessageStatus.PENDING.value,))

            rows = cursor.fetchall()
            loaded_count = 0

            for row in rows:
                try:
                    # Create message from database row
                    message_data = {
                        'id': row[0],
                        'type': CommunicationType(row[1]),
                        'priority': Priority(row[2]),
                        'timestamp': datetime.fromisoformat(row[3]),
                        'sender': row[4],
                        'recipient': row[5],
                        'subject': row[6],
                        'content': json.loads(row[7]),
                        'requires_response': bool(row[8]),
                        'response_timeout': row[9]
                    }

                    message = CommunicationMessage(**message_data)

                    # Add to appropriate queue
                    priority_queue = self.queues[message.priority]
                    # Use timestamp as priority (older messages first within priority level)
                    priority_queue.put(
                        (message.timestamp.timestamp(), message))
                    loaded_count += 1

                except Exception as e:
                    logger.error(
                        f"Error loading message {row[0]} from database: {e}")

            logger.info(
                f"Loaded {loaded_count} pending messages from database")

        except Exception as e:
            logger.error(f"Error loading pending messages: {e}")

    async def _process_messages(self):
        """Process messages from queues."""
        while self.running and not self._shutdown.is_set():
            try:
                message = None

                # Check queues in priority order (critical -> high -> medium -> low)
                for priority in [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
                    priority_queue = self.queues[priority]
                    if not priority_queue.empty():
                        try:
                            _, message = priority_queue.get_nowait()
                            break
                        except queue.Empty:
                            continue

                if message:
                    # Process the message
                    await self._handle_message(message)
                else:
                    # No messages to process, wait a bit to yield control
                    await asyncio.sleep(0.1)

            except Exception as e:
                if not self._shutdown.is_set():
                    logger.error(f"Error in message processing loop: {e}")
                    await asyncio.sleep(1)  # Prevent tight loop on error
                else:
                    break  # Exit loop if shutdown is set

    async def _handle_message(self, message: CommunicationMessage):
        """
        Handle a message from the queue.

        Args:
            message: Message to handle
        """
        try:
            # Update message status to processing
            self._update_message_status(message.id, MessageStatus.PROCESSING)

            # Get message type
            message_type = message.type.value

            # Call appropriate callback if registered
            if message_type in self.message_callbacks:
                callback = self.message_callbacks[message_type]

                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(message)
                    else:
                        callback(message)

                    # Mark as acknowledged
                    self._update_message_status(
                        message.id, MessageStatus.ACKNOWLEDGED)
                    logger.info(f"Message {message.id} processed successfully")

                except Exception as e:
                    # Handle callback error
                    retry_count = self._get_message_retry_count(message.id)
                    if retry_count < self.max_retries:
                        # Retry the message
                        await self._retry_message(message, retry_count + 1)
                    else:
                        # Mark as failed
                        self._update_message_status(
                            message.id, MessageStatus.FAILED)
                        logger.error(
                            f"Message {message.id} failed after {self.max_retries} retries: {e}")
            else:
                logger.warning(
                    f"No callback registered for message type: {message_type}")
                # Mark as acknowledged since there's nothing to do
                self._update_message_status(
                    message.id, MessageStatus.ACKNOWLEDGED)

        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error handling message {message.id}: {e}")

    async def _retry_message(self, message: CommunicationMessage, retry_count: int):
        """
        Retry a failed message.

        Args:
            message: Message to retry
            retry_count: Current retry count
        """
        try:
            # Update retry count
            self._update_message_retry_count(message.id, retry_count)

            # Add back to queue with delay
            await asyncio.sleep(self.retry_delay * retry_count)

            # Add to appropriate queue based on priority
            priority_queue = self.queues[message.priority]
            priority_queue.put((datetime.now().timestamp(), message))

            logger.info(
                f"Message {message.id} scheduled for retry #{retry_count}")

        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error retrying message {message.id}: {e}")

    def send_message(self, message: CommunicationMessage) -> bool:
        """
        Send a message through the Message Queue Channel.

        Args:
            message: Message to send

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if self._shutdown.is_set():
            return False

        try:
            # Store message in database
            self._store_message(message)

            # Add to appropriate queue
            priority_queue = self.queues[message.priority]
            # Use timestamp as priority (older messages first within priority level)
            priority_queue.put((message.timestamp.timestamp(), message))

            logger.info(
                f"Message {message.id} added to queue with priority {message.priority.value}")
            return True

        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(
                    f"Error sending message through Message Queue Channel: {e}")
            return False

    def _store_message(self, message: CommunicationMessage):
        """
        Store a message in the database.

        Args:
            message: Message to store
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO messages
                (id, type, priority, timestamp, sender, recipient, subject, content,
                 requires_response, response_timeout, status, retry_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                message.id,
                message.type.value,
                message.priority.value,
                message.timestamp.isoformat(),
                message.sender,
                message.recipient,
                message.subject,
                json.dumps(message.content),
                message.requires_response,
                message.response_timeout,
                MessageStatus.PENDING.value,
                0,  # retry_count
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            self.db_conn.commit()
        except Exception as e:
            logger.error(
                f"Error storing message {message.id} in database: {e}")
            raise

    def _update_message_status(self, message_id: str, status: MessageStatus):
        """
        Update the status of a message in the database.

        Args:
            message_id: ID of the message
            status: New status
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                UPDATE messages SET status = ?, updated_at = ? WHERE id = ?
            ''', (status.value, datetime.now().isoformat(), message_id))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Error updating message {message_id} status: {e}")

    def _update_message_retry_count(self, message_id: str, retry_count: int):
        """
        Update the retry count of a message in the database.

        Args:
            message_id: ID of the message
            retry_count: New retry count
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                UPDATE messages SET retry_count = ?, updated_at = ? WHERE id = ?
            ''', (retry_count, datetime.now().isoformat(), message_id))
            self.db_conn.commit()
        except Exception as e:
            logger.error(
                f"Error updating message {message_id} retry count: {e}")

    def _get_message_retry_count(self, message_id: str) -> int:
        """
        Get the retry count of a message from the database.

        Args:
            message_id: ID of the message

        Returns:
            Current retry count
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(
                'SELECT retry_count FROM messages WHERE id = ?', (message_id,))
            row = cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            logger.error(
                f"Error getting retry count for message {message_id}: {e}")
            return 0

    def register_message_callback(self, message_type: str, callback: Callable):
        """
        Register a callback for handling specific message types.

        Args:
            message_type: Type of message to handle
            callback: Function to call when message is received
        """
        self.message_callbacks[message_type] = callback
        logger.info(f"Registered callback for message type: {message_type}")

    def get_pending_message_count(self) -> Dict[str, int]:
        """
        Get the count of pending messages by priority.

        Returns:
            Dictionary with priority levels and their message counts
        """
        counts = {}
        for priority, queue_obj in self.queues.items():
            counts[priority.value] = queue_obj.qsize()
        return counts
