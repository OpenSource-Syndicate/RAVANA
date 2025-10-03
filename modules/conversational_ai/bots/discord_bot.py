import asyncio
import logging
import discord
from discord.ext import commands

from .base_bot import BaseBot

logger = logging.getLogger(__name__)


class DiscordBot(BaseBot):
    # Class variables to track instances
    _instance_started = False
    _active_instance = None

    def __init__(self, token: str, command_prefix: str, conversational_ai):
        """
        Initialize the Discord bot.

        Args:
            token: Discord bot token
            command_prefix: Command prefix for the bot
            conversational_ai: Reference to the main ConversationalAI instance
        """
        super().__init__(token, command_prefix, conversational_ai)

        # Initialize Discord client
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.dm_messages = True

        self.bot = commands.Bot(
            command_prefix=command_prefix,
            intents=intents,
            help_command=None
        )

        # For graceful shutdown
        self._shutdown = asyncio.Event()
        # Track if the bot has been started
        self._started = False

        # Register event handlers
        self._register_events()

    @classmethod
    def get_instance(cls, token: str, command_prefix: str, conversational_ai):
        """
        Get a singleton instance of the DiscordBot.

        Args:
            token: Discord bot token
            command_prefix: Command prefix
            conversational_ai: Reference to the main ConversationalAI instance

        Returns:
            DiscordBot instance
        """
        if cls._active_instance is None:
            cls._active_instance = cls(
                token, command_prefix, conversational_ai)
            logger.info("Created new DiscordBot instance")
        else:
            logger.warning(
                "DiscordBot instance already exists, returning existing instance")
        return cls._active_instance

    def _register_events(self):
        """Register Discord bot event handlers."""
        @self.bot.event
        async def on_ready():
            if not self._shutdown.is_set():
                logger.info(f'Discord bot logged in as {self.bot.user}')
                await self.bot.change_presence(
                    status=discord.Status.online,
                    activity=discord.Activity(
                        type=discord.ActivityType.listening, name="your thoughts")
                )
                self.connected = True
                logger.info("Discord bot is now connected and ready")

        @self.bot.event
        async def on_disconnect():
            if not self._shutdown.is_set():
                logger.warning("Discord bot disconnected")
                self.connected = False

        @self.bot.event
        async def on_resumed():
            if not self._shutdown.is_set():
                logger.info("Discord bot connection resumed")
                self.connected = True

        @self.bot.event
        async def on_message(message):
            # Ignore messages from the bot itself
            if message.author == self.bot.user:
                return

            # Ignore messages if shutdown is requested
            if self._shutdown.is_set():
                return

            # Process commands
            if message.content.startswith(self.command_prefix):
                await self.bot.process_commands(message)
                return

            # Process regular messages
            # Respond if it's a DM, or if the bot is mentioned, or if it's a reply to the bot
            should_respond = (
                isinstance(message.channel, discord.DMChannel) or
                (message.guild and self.bot.user in message.mentions) or
                (message.reference and message.reference.resolved and message.reference.resolved.author == self.bot.user)
            )

            # Also respond if the bot is pinged anywhere in the message
            if message.guild and f"<@{self.bot.user.id}>" in message.content:
                should_respond = True

            if should_respond:
                # Get user ID and process message
                user_id = str(message.author.id)
                # Update user profile with username
                self.conversational_ai.user_profile_manager.update_username(
                    user_id, message.author.name)

                # Create a task to process the message asynchronously to prevent blocking the event loop
                asyncio.create_task(
                    self._process_discord_message(message, user_id))

        @self.bot.command(name='task')
        async def task_command(ctx, *, task_description: str):
            """Command to delegate a task to RAVANA."""
            if self._shutdown.is_set():
                return
            user_id = str(ctx.author.id)
            self.handle_task_from_user(
                user_id, task_description)
            if not self._shutdown.is_set():
                await ctx.send("I've sent your task to RAVANA. I'll let you know when there's an update!")

        @self.bot.command(name='mood')
        async def mood_command(ctx):
            """Command to check the AI's current mood."""
            if self._shutdown.is_set():
                return
            # This would need to be implemented based on how mood is tracked per user
            if not self._shutdown.is_set():
                await ctx.send("I'm doing well, thank you for asking!")

        @self.bot.command(name='help')
        async def help_command(ctx):
            """Display help information."""
            if self._shutdown.is_set():
                return
            help_text = f"""
**Conversational AI Commands:**
`{self.command_prefix}task <description>` - Delegate a task to RAVANA
`{self.command_prefix}mood` - Check the AI's current mood
`{self.command_prefix}help` - Display this help message

You can also just message me directly or mention me in a server!
            """
            if not self._shutdown.is_set():
                await ctx.send(help_text)

    async def _process_discord_message(self, message, user_id):
        """Process a Discord message asynchronously without blocking the event loop."""
        try:
            # Offload the LLM processing to a background task to prevent blocking the event loop
            logger.debug("Creating background task for LLM processing")
            llm_task = asyncio.create_task(
                self.conversational_ai.process_user_message_async(
                    platform="discord",
                    user_id=user_id,
                    message=message.content
                )
            )

            # Add the task to a set to prevent it from being garbage collected
            if not hasattr(self, '_background_tasks'):
                self._background_tasks = set()
            self._background_tasks.add(llm_task)
            llm_task.add_done_callback(self._background_tasks.discard)

            # Wait for the LLM response
            try:
                # 60 second timeout
                response = await asyncio.wait_for(llm_task, timeout=60.0)
            except asyncio.TimeoutError:
                logger.error("LLM processing timed out after 60 seconds")
                response = "Sorry, I'm taking too long to respond. Please try again."

            # Split long messages to comply with Discord's limits
            if len(response) > 2000:
                chunks = [response[i:i+1990]
                          for i in range(0, len(response), 1990)]
                for chunk in chunks:
                    if not self._shutdown.is_set():
                        await message.channel.send(chunk)
            else:
                if not self._shutdown.is_set():
                    await message.channel.send(response)

        except Exception as e:
            logger.error(
                f"Error processing Discord message: {e}", exc_info=True)
            if not self._shutdown.is_set():
                await message.channel.send("Sorry, I encountered an error processing your message.")

    async def start(self):
        """Start the Discord bot."""
        if self._shutdown.is_set():
            logger.warning("Discord bot shutdown event is set, cannot start")
            return

        # Check if any instance is already started
        if DiscordBot._instance_started:
            logger.warning(
                "Discord bot already started in another instance, skipping...")
            return

        if self._started:
            logger.warning("Discord bot already started, skipping...")
            return

        try:
            self._started = True  # Mark as started
            DiscordBot._instance_started = True  # Mark class as started
            logger.info("Attempting to connect Discord bot...")
            # Run the bot without blocking indefinitely
            await self.bot.start(self.token)
        except Exception as e:
            self._started = False  # Reset on error
            DiscordBot._instance_started = False  # Reset class on error
            self.connected = False
            if not self._shutdown.is_set():
                logger.error(f"Error starting Discord bot: {e}")
                raise  # Re-raise the exception so it's properly handled

    async def stop(self):
        """Stop the Discord bot."""
        if self._shutdown.is_set():
            logger.info("Discord bot already shut down")
            return

        self._shutdown.set()
        self._started = False  # Reset started flag
        DiscordBot._instance_started = False  # Reset class started flag
        self.connected = False
        DiscordBot._active_instance = None  # Clear active instance
        try:
            await self.bot.close()
            logger.info("Discord bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Discord bot: {e}")

    async def send_message(self, user_id: str, message: str):
        """
        Send a message to a user.

        Args:
            user_id: Discord user ID
            message: Message to send
        """
        if self._shutdown.is_set():
            return
        try:
            user = await self.bot.fetch_user(int(user_id))
            if user:
                # Split long messages
                if len(message) > 2000:
                    chunks = [message[i:i+1990]
                              for i in range(0, len(message), 1990)]
                    for chunk in chunks:
                        if not self._shutdown.is_set():
                            await user.send(chunk)
                else:
                    if not self._shutdown.is_set():
                        await user.send(message)
        except discord.NotFound:
            logger.warning(f"Could not find Discord user with ID {user_id}")
        except Exception as e:
            logger.error(
                f"Error sending Discord message to user {user_id}: {e}")
