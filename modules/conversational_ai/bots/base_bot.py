import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.conversational_ai.main import ConversationalAI

logger = logging.getLogger(__name__)


class BaseBot(ABC):
    """Abstract base class for conversational bots."""

    def __init__(self, token: str, command_prefix: str, conversational_ai: "ConversationalAI"):
        """
        Initialize the base bot.

        Args:
            token: Bot token
            command_prefix: Command prefix for the bot
            conversational_ai: Reference to the main ConversationalAI instance
        """
        if not token:
            raise ValueError("Bot token cannot be empty.")
            
        self.token = token
        self.command_prefix = command_prefix
        self.conversational_ai = conversational_ai
        self.connected = False
        self._shutdown = asyncio.Event()
        self._started = False

    @abstractmethod
    async def start(self):
        """Start the bot."""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the bot."""
        pass

    @abstractmethod
    async def send_message(self, user_id: str, message: str):
        """Send a message to a user."""
        pass

    def handle_task_from_user(self, user_id: str, task_description: str):
        """Common handler for task delegation."""
        if self._shutdown.is_set():
            return
        self.conversational_ai.handle_task_from_user(user_id, task_description)
