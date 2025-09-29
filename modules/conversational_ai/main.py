import logging
import traceback
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any

# Import required modules
from .emotional_intelligence.conversational_ei import ConversationalEmotionalIntelligence
from .memory.memory_interface import SharedMemoryInterface
from .communication.ravana_bridge import RAVANACommunicator
from .profiles.user_profile_manager import UserProfileManager
from .communication.data_models import UserPlatformProfile

logger = logging.getLogger(__name__)


class ConversationalAI:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConversationalAI, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initialize the Conversational AI module with all required components.
        """
        # Prevent multiple initializations
        if ConversationalAI._initialized:
            logger.warning(
                "ConversationalAI instance already initialized, skipping...")
            return

        # Initialize core components
        self.emotional_intelligence = ConversationalEmotionalIntelligence()
        self.memory_interface = SharedMemoryInterface()
        self.ravana_communicator = RAVANACommunicator(
            "conversational_ai_bridge", self)
        self.user_profile_manager = UserProfileManager()

        # Initialize shutdown event
        self._shutdown = asyncio.Event()

        # Load configuration
        self.config = self._load_config()

        # Initialize bots (will be set up in start method)
        self.discord_bot = None
        self.telegram_bot = None
        self._bot_tasks = []

        # Mark as initialized
        ConversationalAI._initialized = True

        logger.info("Conversational AI module initialized successfully")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json file."""
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            # Return default configuration
            return {
                "discord_token": "",
                "telegram_token": "",
                "platforms": {
                    "discord": {"enabled": False, "command_prefix": "!"},
                    "telegram": {"enabled": False, "command_prefix": "/"}
                }
            }

    async def _run_discord_bot(self):
        """Run the Discord bot in a separate task."""
        try:
            if self.discord_bot:
                logger.info("Starting Discord bot...")
                # For Discord bot, we need to handle the blocking start method differently
                # We'll run it in a task and handle shutdown properly

                async def discord_bot_runner():
                    try:
                        await self.discord_bot.start()
                    except Exception as e:
                        if not self._shutdown.is_set():
                            logger.error(f"Error in Discord bot task: {e}")
                            logger.error(
                                f"Full traceback: {traceback.format_exc()}")

                # Create and store the task
                discord_task = asyncio.create_task(discord_bot_runner())
                # Store reference to the task so it's not garbage collected
                if not hasattr(self, '_discord_bot_task'):
                    self._discord_bot_task = discord_task

                logger.info("Discord bot start task created and running")
        except Exception as e:
            logger.error(f"Error starting Discord bot task: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")

    async def _run_telegram_bot(self):
        """Run the Telegram bot in a separate task."""
        try:
            if self.telegram_bot:
                logger.info("Starting Telegram bot...")
                # For Telegram bot, we need to handle the blocking start method differently
                # We'll run it in a task and handle shutdown properly

                async def telegram_bot_runner():
                    try:
                        await self.telegram_bot.start()
                    except Exception as e:
                        if not self._shutdown.is_set():
                            logger.error(f"Error in Telegram bot task: {e}")
                            logger.error(
                                f"Full traceback: {traceback.format_exc()}")

                # Create and store the task
                telegram_task = asyncio.create_task(telegram_bot_runner())
                # Store reference to the task so it's not garbage collected
                if not hasattr(self, '_telegram_bot_task'):
                    self._telegram_bot_task = telegram_task

                logger.info("Telegram bot start task created and running")
        except Exception as e:
            logger.error(f"Error starting Telegram bot task: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")

    async def start(self, standalone: bool = True):
        """
        Start the Conversational AI module.

        Args:
            standalone: Whether to run in standalone mode or integrated with RAVANA
        """
        try:
            logger.info(
                f"Starting Conversational AI module in {'standalone' if standalone else 'integrated'} mode")

            # Initialize bots - now also for integrated mode
            # Initialize Discord bot if configured and enabled
            if (self.config.get("platforms", {}).get("discord", {}).get("enabled", False) and
                    self.config.get("discord_token")):
                try:
                    from .bots.discord_bot import DiscordBot
                    discord_config = self.config["platforms"]["discord"]
                    self.discord_bot = DiscordBot.get_instance(
                        token=self.config["discord_token"],
                        command_prefix=discord_config["command_prefix"],
                        conversational_ai=self
                    )
                    logger.info("Discord bot initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Discord bot: {e}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")

            # Initialize Telegram bot if configured and enabled
            if (self.config.get("platforms", {}).get("telegram", {}).get("enabled", False) and
                    self.config.get("telegram_token")):
                try:
                    from .bots.telegram_bot import TelegramBot
                    telegram_config = self.config["platforms"]["telegram"]
                    self.telegram_bot = await TelegramBot.get_instance(
                        token=self.config["telegram_token"],
                        command_prefix=telegram_config["command_prefix"],
                        conversational_ai=self
                    )
                    logger.info("Telegram bot initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize Telegram bot: {e}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")

            # Start RAVANA communicator
            await self.ravana_communicator.start()

            # Start bots
            bot_tasks = []

            # Start Discord bot if available
            if self.discord_bot:
                try:
                    logger.info("Attempting to start Discord bot...")
                    # Create a task for the Discord bot to run independently
                    discord_task = asyncio.create_task(self._run_discord_bot())
                    bot_tasks.append(discord_task)
                    logger.info("Discord bot start task created")
                except Exception as e:
                    logger.error(f"Failed to start Discord bot task: {e}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")

            # Start Telegram bot if available
            if self.telegram_bot:
                try:
                    logger.info("Attempting to start Telegram bot...")
                    # Create a task for the Telegram bot to run independently
                    telegram_task = asyncio.create_task(
                        self._run_telegram_bot())
                    bot_tasks.append(telegram_task)
                    logger.info("Telegram bot start task created")
                except Exception as e:
                    logger.error(f"Failed to start Telegram bot task: {e}")
                    logger.error(f"Full traceback: {traceback.format_exc()}")

            # Store bot tasks to prevent them from being garbage collected
            self._bot_tasks = bot_tasks

            # If in standalone mode, wait indefinitely for bot tasks or shutdown event
            if standalone and bot_tasks:
                logger.info(
                    "Bots started in standalone mode, running indefinitely. Press Ctrl+C to stop.")
                logger.info(f"Number of bot tasks: {len(bot_tasks)}")
                logger.info(
                    f"Shutdown event is set: {self._shutdown.is_set()}")
                try:
                    # Wait for shutdown event while keeping bots running
                    await self._shutdown.wait()
                except asyncio.CancelledError:
                    logger.info("Main task cancelled")
                except Exception as e:
                    logger.error(f"Error while waiting for shutdown: {e}")
                    logger.exception("Full traceback:")
                finally:
                    # Stop bots
                    await self.stop()
            elif standalone:
                # No bots but in standalone mode, run a simple loop
                logger.info(
                    "No bots available, running in standalone mode. Press Ctrl+C to stop.")
                try:
                    while not self._shutdown.is_set():
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    logger.info("Main task cancelled")
                except Exception as e:
                    logger.error(f"Error in standalone loop: {e}")
                    logger.exception("Full traceback:")
                finally:
                    await self.stop()
            else:
                # In integrated mode, keep the bots running in background tasks
                # The tasks are already running, so we just need to ensure they stay alive
                logger.info("Bots started in integrated mode")
                logger.info(f"Number of bot tasks running: {len(bot_tasks)}")
                # In integrated mode, we don't wait here as the main system will manage the lifecycle
                # But we do log that the bots are running
                if bot_tasks:
                    logger.info("Bot tasks are running in the background")

            logger.info("Conversational AI module started successfully")
        except Exception as e:
            logger.error(f"Error starting Conversational AI module: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    async def stop(self):
        """Stop the Conversational AI module."""
        logger.info("Stopping Conversational AI module...")
        self._shutdown.set()

        # Cancel any running bot tasks
        if hasattr(self, '_discord_bot_task') and self._discord_bot_task:
            self._discord_bot_task.cancel()
            try:
                await self._discord_bot_task
            except asyncio.CancelledError:
                pass

        if hasattr(self, '_telegram_bot_task') and self._telegram_bot_task:
            self._telegram_bot_task.cancel()
            try:
                await self._telegram_bot_task
            except asyncio.CancelledError:
                pass

        # Stop bots if they're running
        if self.discord_bot:
            try:
                await self.discord_bot.stop()
                logger.info("Discord bot stopped")
            except Exception as e:
                logger.error(f"Error stopping Discord bot: {e}")

        if self.telegram_bot:
            try:
                await self.telegram_bot.stop()
                logger.info("Telegram bot stopped")
            except Exception as e:
                logger.error(f"Error stopping Telegram bot: {e}")

    def process_user_message(self, platform: str, user_id: str, message: str) -> str:
        """
        Process an incoming user message and generate a response.

        Args:
            platform: The platform the message came from (discord/telegram)
            user_id: The unique identifier of the user
            message: The user's message

        Returns:
            The AI's response to the message
        """
        try:
            # Track user platform preference
            self._track_user_platform(user_id, platform)

            # Get context from shared memory
            context = self.memory_interface.get_context(user_id)

            # Process message with emotional intelligence
            self.emotional_intelligence.set_persona("Balanced")
            emotional_context = self.emotional_intelligence.process_user_message(
                message, context)

            # Generate response using emotional intelligence
            response = self.emotional_intelligence.generate_response(
                message, emotional_context)

            # Store conversation in memory
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_message": message,
                "ai_response": response,
                "emotional_context": emotional_context
            }

            self.memory_interface.store_conversation(
                user_id, conversation_entry)

            # Extract thoughts from the conversation
            thoughts = self.emotional_intelligence.extract_thoughts_from_conversation(
                message, response, emotional_context)

            # Send thoughts to RAVANA if any were extracted
            if thoughts:
                for thought in thoughts:
                    # Add metadata to the thought
                    thought_with_metadata = {
                        "thought_type": thought.get("thought_type", "insight"),
                        "payload": thought.get("content", ""),
                        "priority": thought.get("priority", "medium"),
                        "emotional_context": thought.get("emotional_context", {}),
                        "metadata": {
                            **thought.get("metadata", {}),
                            "user_id": user_id,
                            "platform": platform,
                            "conversation_id": f"{user_id}_{datetime.now().isoformat()}"
                        }
                    }

                    # Send thought to RAVANA
                    self.ravana_communicator.send_thought_to_ravana(
                        thought_with_metadata)

            # Synchronize emotional context with RAVANA
            self._synchronize_emotional_context(user_id, emotional_context)

            return response
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            return "I'm sorry, I encountered an error processing your message."

    async def process_user_message_async(self, platform: str, user_id: str, message: str) -> str:
        """
        Process an incoming user message and generate a response asynchronously.

        Args:
            platform: The platform the message came from (discord/telegram)
            user_id: The unique identifier of the user
            message: The user's message

        Returns:
            The AI's response to the message
        """
        # This is the async version of process_user_message
        return self.process_user_message(platform, user_id, message)

    def handle_task_from_user(self, user_id: str, task_description: str):
        """
        Handle a task delegation from a user.

        Args:
            user_id: The unique identifier of the user
            task_description: Description of the task to delegate
        """
        try:
            logger.info(
                f"Handling task from user {user_id}: {task_description}")

            # Send the task to RAVANA via the communicator
            task_data = {
                "type": "user_task",
                "user_id": user_id,
                "task_description": task_description,
                "timestamp": datetime.now().isoformat()
            }

            self.ravana_communicator.send_task_to_ravana(task_data)
            logger.info(
                f"Task from user {user_id} sent to RAVANA for processing")

        except Exception as e:
            logger.error(f"Error handling task from user {user_id}: {e}")

    def _synchronize_emotional_context(self, user_id: str, emotional_context: Dict[str, Any]):
        """
        Synchronize emotional context with RAVANA.

        Args:
            user_id: The unique identifier of the user
            emotional_context: The emotional context to synchronize
        """
        try:
            # Send emotional context to RAVANA
            emotional_data = {
                "type": "emotional_context_update",
                "user_id": user_id,
                "emotional_context": emotional_context,
                "timestamp": datetime.now().isoformat()
            }

            self.ravana_communicator.send_emotional_context_to_ravana(
                emotional_data)
        except Exception as e:
            logger.error(
                f"Error synchronizing emotional context for user {user_id}: {e}")

    def _track_user_platform(self, user_id: str, platform: str):
        """
        Track the user's platform preference.

        Args:
            user_id: The unique identifier of the user
            platform: The platform the user is using (discord/telegram)
        """
        try:
            # Create or update user platform profile
            profile = UserPlatformProfile(
                user_id=user_id,
                last_platform=platform,
                # In a real implementation, this would be the platform-specific user ID
                platform_user_id=user_id,
                preferences={},
                last_interaction=datetime.now()
            )

            # Store in user profile manager
            self.user_profile_manager.set_user_platform_profile(
                user_id, profile)

            logger.debug(f"Tracked platform {platform} for user {user_id}")
        except Exception as e:
            logger.error(
                f"Error tracking user platform for user {user_id}: {e}")

    async def send_message_to_user(self, user_id: str, message: str, platform: str = None):
        """
        Send a message to a user through the appropriate platform.

        Args:
            user_id: The unique identifier of the user
            message: The message to send
            platform: The platform to use (discord/telegram), if None will use last known platform
        """
        try:
            # Determine the appropriate platform to use
            if not platform:
                # Try to get the user's last used platform from their profile
                profile = self.user_profile_manager.get_user_platform_profile(
                    user_id)
                if profile:
                    platform = profile.last_platform
                    logger.debug(
                        f"Using last known platform {platform} for user {user_id}")
                else:
                    # If no profile exists, we'll try both platforms
                    logger.debug(
                        f"No platform profile found for user {user_id}, will try both platforms")

            success = False

            # Try to send message through the specified platform first
            if platform == "discord" and self.discord_bot:
                try:
                    await self.discord_bot.send_message(user_id, message)
                    success = True
                except Exception as e:
                    logger.warning(
                        f"Failed to send message via Discord to user {user_id}: {e}")

            elif platform == "telegram" and self.telegram_bot:
                try:
                    await self.telegram_bot.send_message(user_id, message)
                    success = True
                except Exception as e:
                    logger.warning(
                        f"Failed to send message via Telegram to user {user_id}: {e}")

            # If the specified platform failed or no platform was specified, try both platforms
            if not success:
                # Try Discord first if available
                if self.discord_bot:
                    try:
                        await self.discord_bot.send_message(user_id, message)
                        success = True
                    except Exception as e:
                        logger.warning(
                            f"Failed to send message via Discord to user {user_id}: {e}")

                # Try Telegram if Discord failed or isn't available
                if not success and self.telegram_bot:
                    try:
                        await self.telegram_bot.send_message(user_id, message)
                        success = True
                    except Exception as e:
                        logger.warning(
                            f"Failed to send message via Telegram to user {user_id}: {e}")

            if not success:
                logger.warning(
                    f"Failed to send message to user {user_id} via any platform")

        except Exception as e:
            logger.error(f"Error sending message to user {user_id}: {e}")
