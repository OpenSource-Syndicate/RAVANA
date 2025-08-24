import asyncio
import logging
from typing import Optional
from telegram import Update, Bot
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackContext, filters

logger = logging.getLogger(__name__)

class TelegramBot:
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
        
        # Initialize Telegram application
        self.application = Application.builder().token(token).build()
        
        # For graceful shutdown
        self._shutdown = asyncio.Event()
        # Track if the bot has been started
        self._started = False
        
        # Register handlers
        self._register_handlers()
        
    def _register_handlers(self):
        """Register Telegram bot handlers."""
        # Message handler for regular messages
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )
        
        # Command handlers
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("task", self._task_command))
        self.application.add_handler(CommandHandler("mood", self._mood_command))
        
    async def _handle_message(self, update: Update, context: CallbackContext):
        """Handle incoming messages."""
        if self._shutdown.is_set():
            return
        try:
            user_id = str(update.effective_user.id)
            username = update.effective_user.username or f"user_{user_id}"
            message_text = update.message.text
            
            # Update user profile with username
            self.conversational_ai.user_profile_manager.update_username(user_id, username)
            
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
            logger.error(f"Error handling Telegram message: {e}")
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
                
            self.conversational_ai.handle_task_from_user(user_id, task_description)
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
            
        if self._started:
            logger.warning("Telegram bot already started, skipping...")
            return
            
        try:
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()
            self._started = True  # Mark as started
            logger.info("Telegram bot started")
        except Exception as e:
            if not self._shutdown.is_set():
                logger.error(f"Error starting Telegram bot: {e}")
            
    async def stop(self):
        """Stop the Telegram bot."""
        if self._shutdown.is_set():
            logger.info("Telegram bot already shut down")
            return
            
        self._shutdown.set()
        self._started = False  # Reset started flag
        try:
            # Only stop if the bot was actually started
            if hasattr(self.application, 'updater') and self.application.updater:
                await self.application.updater.stop()
            if hasattr(self.application, 'stop'):
                await self.application.stop()
            if hasattr(self.application, 'shutdown'):
                await self.application.shutdown()
            logger.info("Telegram bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")
        
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
                chunks = [message[i:i+3990] for i in range(0, len(message), 3990)]
                for chunk in chunks:
                    if not self._shutdown.is_set():
                        await self.application.bot.send_message(chat_id=int(user_id), text=chunk)
            else:
                if not self._shutdown.is_set():
                    await self.application.bot.send_message(chat_id=int(user_id), text=message)
        except Exception as e:
            logger.error(f"Error sending Telegram message to user {user_id}: {e}")