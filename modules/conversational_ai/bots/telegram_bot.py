import asyncio
import logging
from telegram import Update, Bot
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackContext, filters

logger = logging.getLogger(__name__)


class TelegramBot:
    # Class variables to track instances
    _active_instances = {}  # Track instances by token
    _lock = asyncio.Lock()  # Async lock for thread safety

    def __init__(self, token: str, command_prefix: str, conversational_ai):
        """
        Initialize the Telegram bot.

        Args:
            token: Telegram bot token
            command_prefix: Command prefix (not used in Telegram but kept for consistency)
            conversational_ai: Reference to the main ConversationalAI instance
        """
        self.token = token
        self.command_prefix = command_prefix
        self.conversational_ai = conversational_ai
        self.connected = False
        self._running = False

        # Initialize Telegram application
        self.application = Application.builder().token(token).build()

        # For graceful shutdown
        self._shutdown = asyncio.Event()
        # Track if the bot has been started
        self._started = False

        # Register handlers
        self._register_handlers()

    @classmethod
    async def get_instance(cls, token: str, command_prefix: str, conversational_ai):
        """
        Get a singleton instance of the TelegramBot for a specific token.

        Args:
            token: Telegram bot token
            command_prefix: Command prefix
            conversational_ai: Reference to the main ConversationalAI instance

        Returns:
            TelegramBot instance
        """
        async with cls._lock:
            if token in cls._active_instances:
                logger.warning(
                    f"TelegramBot instance already exists for token {token[:10]}..., returning existing instance")
                return cls._active_instances[token]

            instance = cls(token, command_prefix, conversational_ai)
            cls._active_instances[token] = instance
            logger.info(
                f"Created new TelegramBot instance for token {token[:10]}...")
            return instance

    @classmethod
    async def remove_instance(cls, token: str):
        """
        Remove an instance from the active instances dictionary.

        Args:
            token: Telegram bot token
        """
        async with cls._lock:
            if token in cls._active_instances:
                del cls._active_instances[token]
                logger.info(
                    f"Removed TelegramBot instance for token {token[:10]}...")

    def _register_handlers(self):
        """Register Telegram bot handlers."""
        # Message handler for regular messages
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND,
                           self._handle_message)
        )

        # Command handlers
        self.application.add_handler(
            CommandHandler("start", self._start_command))
        self.application.add_handler(
            CommandHandler("help", self._help_command))
        self.application.add_handler(
            CommandHandler("task", self._task_command))
        self.application.add_handler(
            CommandHandler("mood", self._mood_command))

    async def _handle_message(self, update: Update, context: CallbackContext):
        """Handle incoming messages."""
        if self._shutdown.is_set():
            return
        try:
            user_id = str(update.effective_user.id)
            username = update.effective_user.username or f"user_{user_id}"
            message_text = update.message.text

            # Update user profile with username
            self.conversational_ai.user_profile_manager.update_username(
                user_id, username)

            # Create a task to process the message asynchronously to prevent blocking
            asyncio.create_task(self._process_telegram_message(
                update, user_id, message_text))
        except Exception as e:
            logger.error(f"Error handling Telegram message: {e}")
            if not self._shutdown.is_set():
                await update.message.reply_text("Sorry, I encountered an error processing your message.")

    async def _process_telegram_message(self, update: Update, user_id: str, message_text: str):
        """Process a Telegram message asynchronously."""
        try:
            # Process message with conversational AI using async version to prevent blocking
            response = await self.conversational_ai.process_user_message_async(
                platform="telegram",
                user_id=user_id,
                message=message_text
            )

            # Send response
            if not self._shutdown.is_set():
                await update.message.reply_text(response)
        except Exception as e:
            logger.error(f"Error processing Telegram message: {e}")
            if not self._shutdown.is_set():
                await update.message.reply_text("Sorry, I encountered an error processing your message.")

    async def _start_command(self, update: Update, context: CallbackContext):
        """Handle /start command."""
        if self._shutdown.is_set():
            return
        welcome_message = """
Hello! I'm RAVANA's conversational AI. I can chat with you and help you delegate tasks to RAVANA.

Just send me a message and I'll respond. You can also use these commands:
/task <description> - Delegate a task to RAVANA
/mood - Check my current mood
/help - Show this help message
        """
        if not self._shutdown.is_set():
            await update.message.reply_text(welcome_message)

    async def _help_command(self, update: Update, context: CallbackContext):
        """Handle /help command."""
        if self._shutdown.is_set():
            return
        help_message = """
**Conversational AI Commands:**
/task <description> - Delegate a task to RAVANA
/mood - Check my current mood
/help - Show this help message

Just send me a message and I'll respond!
        """
        if not self._shutdown.is_set():
            await update.message.reply_text(help_message)

    async def _task_command(self, update: Update, context: CallbackContext):
        """Handle /task command."""
        if self._shutdown.is_set():
            return
        try:
            user_id = str(update.effective_user.id)
            task_description = " ".join(context.args)

            if not task_description:
                if not self._shutdown.is_set():
                    await update.message.reply_text("Please provide a task description. Usage: /task <description>")
                return

            self.conversational_ai.handle_task_from_user(
                user_id, task_description)
            if not self._shutdown.is_set():
                await update.message.reply_text("I've sent your task to RAVANA. I'll let you know when there's an update!")
        except Exception as e:
            logger.error(f"Error handling Telegram task command: {e}")
            if not self._shutdown.is_set():
                await update.message.reply_text("Sorry, I encountered an error processing your task.")

    async def _mood_command(self, update: Update, context: CallbackContext):
        """Handle /mood command."""
        if self._shutdown.is_set():
            return
        if not self._shutdown.is_set():
            await update.message.reply_text("I'm doing well, thank you for asking!")

    async def start(self):
        """Start the Telegram bot."""
        if self._shutdown.is_set():
            logger.warning("Telegram bot shutdown event is set, cannot start")
            return

        # Check if this specific instance is already started
        if self._started:
            logger.warning("Telegram bot already started, skipping...")
            return

        try:
            logger.info("Initializing Telegram bot application...")
            await self.application.initialize()
            logger.info("Starting Telegram bot application...")
            await self.application.start()
            logger.info("Starting Telegram bot updater...")
            await self.application.updater.start_polling()
            self._started = True  # Mark as started
            self.connected = True
            self._running = True
            logger.info("Telegram bot started and connected successfully")

            # Keep the bot running until shutdown
            while not self._shutdown.is_set() and self._running:
                await asyncio.sleep(1)

        except Exception as e:
            self._started = False  # Reset on error
            self.connected = False
            self._running = False
            if not self._shutdown.is_set():
                logger.error(f"Error starting Telegram bot: {e}")
                logger.exception("Full traceback:")
                # Remove from active instances on error
                await TelegramBot.remove_instance(self.token)
                raise  # Re-raise the exception so it's properly handled

    async def stop(self):
        """Stop the Telegram bot."""
        if self._shutdown.is_set():
            logger.info("Telegram bot already shut down")
            return

        logger.info("Stopping Telegram bot...")
        self._shutdown.set()
        self._started = False  # Reset started flag
        self.connected = False
        self._running = False

        try:
            # Only stop if the bot was actually started
            if hasattr(self.application, 'updater') and self.application.updater:
                logger.info("Stopping Telegram bot updater...")
                await self.application.updater.stop()
            if hasattr(self.application, 'stop'):
                logger.info("Stopping Telegram bot application...")
                await self.application.stop()
            if hasattr(self.application, 'shutdown'):
                logger.info("Shutting down Telegram bot application...")
                await self.application.shutdown()
            logger.info("Telegram bot stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")
            logger.exception("Full traceback:")
        finally:
            # Remove from active instances
            await TelegramBot.remove_instance(self.token)

    async def send_message(self, user_id: str, message: str):
        """
        Send a message to a user.

        Args:
            user_id: Telegram user ID
            message: Message to send
        """
        if self._shutdown.is_set():
            return
        try:
            # Split long messages (Telegram limit is 4096 characters)
            if len(message) > 4000:
                chunks = [message[i:i+3990]
                          for i in range(0, len(message), 3990)]
                for chunk in chunks:
                    if not self._shutdown.is_set():
                        await self.application.bot.send_message(chat_id=int(user_id), text=chunk)
            else:
                if not self._shutdown.is_set():
                    await self.application.bot.send_message(chat_id=int(user_id), text=message)
        except Exception as e:
            logger.error(
                f"Error sending Telegram message to user {user_id}: {e}")
