import asyncio
import logging
import signal
import sys
import json
import os
from typing import Dict, Any
from datetime import datetime

# Add the parent directory to the path to import modules
sys.path.append("..")

# Import bot interfaces
from modules.conversational_ai.bots.discord_bot import DiscordBot
from modules.conversational_ai.bots.telegram_bot import TelegramBot

# Import emotional intelligence
from modules.conversational_ai.emotional_intelligence.conversational_ei import ConversationalEmotionalIntelligence

# Import memory interface
from modules.conversational_ai.memory.memory_interface import SharedMemoryInterface

# Import user profile manager
from modules.conversational_ai.profiles.user_profile_manager import UserProfileManager

# Import RAVANA communication bridge
from modules.conversational_ai.communication.ravana_bridge import RAVANACommunicator

logger = logging.getLogger(__name__)

class ConversationalAI:
    """Main conversational AI module that integrates all components."""
    
    def __init__(self, channel: str = "memory_service"):
        """Initialize the conversational AI module."""
        logger.info("Initializing Conversational AI module...")
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize shutdown event
        self._shutdown = asyncio.Event()
        
        # Initialize components
        self.memory_interface = SharedMemoryInterface()
        self.user_profile_manager = UserProfileManager()
        self.emotional_intelligence = ConversationalEmotionalIntelligence()
        self.ravana_communicator = RAVANACommunicator(channel, self)
        
        # Initialize bot interfaces (will be configured from config)
        self.bots = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json file."""
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info("Configuration loaded successfully")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using default configuration")
            return {
                "discord_token": "YOUR_DISCORD_TOKEN_HERE",
                "telegram_token": "YOUR_TELEGRAM_TOKEN_HERE",
                "platforms": {
                    "discord": {"enabled": True, "command_prefix": "!"},
                    "telegram": {"enabled": True, "command_prefix": "/"}
                }
            }
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing config file: {e}, using default configuration")
            return {
                "discord_token": "YOUR_DISCORD_TOKEN_HERE",
                "telegram_token": "YOUR_TELEGRAM_TOKEN_HERE",
                "platforms": {
                    "discord": {"enabled": True, "command_prefix": "!"},
                    "telegram": {"enabled": True, "command_prefix": "/"}
                }
            }
    
    async def start(self, standalone: bool = True):
        """Start the conversational AI module.
        
        Args:
            standalone: Whether this is running as a standalone module (True) or 
                       as part of the main RAVANA system (False)
        """
        logger.info("Starting Conversational AI module...")
        
        # Set up signal handlers for graceful shutdown (only in standalone mode)
        if standalone:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize bot interfaces
        logger.info("Initializing bot interfaces...")
        try:
            # Discord bot
            discord_config = self.config.get("platforms", {}).get("discord", {})
            if discord_config.get("enabled", True):
                discord_token = self.config.get("discord_token", "YOUR_DISCORD_TOKEN_HERE")
                discord_prefix = discord_config.get("command_prefix", "!")
                self.bots["discord"] = DiscordBot(discord_token, discord_prefix, self)
                logger.info("Discord bot configured")
            else:
                logger.info("Discord bot disabled in configuration")
            
            # Telegram bot
            telegram_config = self.config.get("platforms", {}).get("telegram", {})
            if telegram_config.get("enabled", True):
                telegram_token = self.config.get("telegram_token", "YOUR_TELEGRAM_TOKEN_HERE")
                telegram_prefix = telegram_config.get("command_prefix", "/")
                self.bots["telegram"] = TelegramBot(telegram_token, telegram_prefix, self)
                logger.info("Telegram bot configured")
            else:
                logger.info("Telegram bot disabled in configuration")
        except Exception as e:
            logger.error(f"Error initializing bot interfaces: {e}")
            return
        
        # Start bot interfaces
        logger.info("Starting bot interfaces...")
        bot_tasks = []
        for platform, bot in self.bots.items():
            logger.info(f"Starting {platform} bot...")
            bot_tasks.append(asyncio.create_task(bot.start()))
        
        # Start RAVANA communication bridge
        logger.info("Starting RAVANA communication bridge...")
        ravana_task = asyncio.create_task(self.ravana_communicator.start())
        
        # Wait for all tasks
        try:
            await asyncio.gather(*bot_tasks, ravana_task)
        except asyncio.CancelledError:
            logger.info("Conversational AI module received cancel signal")
            await self.shutdown()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            await self.shutdown()

    async def shutdown(self):
        """Gracefully shutdown the conversational AI module."""
        if self._shutdown.is_set():
            return
            
        logger.info("Shutting down Conversational AI module...")
        self._shutdown.set()
        
        # Stop all bots
        for platform, bot in self.bots.items():
            logger.info(f"Stopping {platform} bot...")
            await bot.stop()
        
        # Stop RAVANA communication bridge
        logger.info("Stopping RAVANA communication bridge...")
        await self.ravana_communicator.stop()
        
        logger.info("Conversational AI module shutdown complete")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown())

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
            # Get or create user profile
            user_profile = self.user_profile_manager.get_user_profile(user_id, platform)
            
            # Get context from shared memory
            context = self.memory_interface.get_context(user_id)
            
            # Process message with emotional intelligence
            self.emotional_intelligence.set_persona(user_profile.get("personality", {}).get("persona", "Balanced"))
            emotional_context = self.emotional_intelligence.process_user_message(message, context)
            
            # Update user profile with emotional state
            user_profile["emotional_state"] = emotional_context
            self.user_profile_manager.update_user_profile(user_id, user_profile)
            
            # Generate response using emotional intelligence
            response = self.emotional_intelligence.generate_response(message, emotional_context)
            
            # Store conversation in memory
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_message": message,
                "ai_response": response,
                "emotional_context": emotional_context
            }
            
            self.memory_interface.store_conversation(user_id, conversation_entry)
            
            # Store in user profile manager for full history
            self.user_profile_manager.store_chat_message(user_id, {
                "timestamp": datetime.now().isoformat(),
                "sender": "user",
                "content": message,
                "emotional_context": emotional_context
            })
            
            self.user_profile_manager.store_chat_message(user_id, {
                "timestamp": datetime.now().isoformat(),
                "sender": "ai",
                "content": response,
                "emotional_context": emotional_context
            })
            
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
                    self.ravana_communicator.send_thought_to_ravana(thought_with_metadata)
            
            # Synchronize emotional context with RAVANA
            self._synchronize_emotional_context(user_id, emotional_context)
            
            return response
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            return "I'm sorry, I encountered an error processing your message."

    def _synchronize_emotional_context(self, user_id: str, emotional_context: Dict[str, Any]):
        """
        Synchronize emotional context with RAVANA core system.
        
        Args:
            user_id: The unique identifier of the user
            emotional_context: Current emotional context
        """
        try:
            # Create an emotional synchronization message
            emotional_sync_message = {
                "thought_type": "emotional_sync",
                "payload": {
                    "user_id": user_id,
                    "emotional_state": emotional_context
                },
                "priority": "low",
                "emotional_context": emotional_context,
                "metadata": {
                    "user_id": user_id,
                    "sync_type": "emotional_context"
                }
            }
            
            # Send to RAVANA
            self.ravana_communicator.send_thought_to_ravana(emotional_sync_message)
            
        except Exception as e:
            logger.error(f"Error synchronizing emotional context: {e}")

    async def send_message_to_user(self, user_id: str, message: str, platform: str = None):
        """
        Send a message to a user from RAVANA.
        
        Args:
            user_id: The unique identifier of the user
            message: The message to send
            platform: The platform to send the message on (if None, will use user's last platform)
        """
        try:
            # Get user profile to determine platform if not specified
            if platform is None:
                user_profile = self.user_profile_manager.get_user_profile(user_id)
                platform = user_profile.get("platform", "discord")
            
            # Send message through appropriate bot
            if platform in self.bots:
                await self.bots[platform].send_message(user_id, message)
                logger.info(f"Sent message to user {user_id} on {platform}")
            else:
                logger.warning(f"Cannot send message to user {user_id}, platform {platform} not available")
        except Exception as e:
            logger.error(f"Error sending message to user {user_id}: {e}")

    def handle_task_from_user(self, user_id: str, task_description: str):
        """
        Handle a task delegation from a user to RAVANA.
        
        Args:
            user_id: The unique identifier of the user
            task_description: Description of the task to delegate
        """
        try:
            # Send task to RAVANA through communication bridge
            task = {
                "user_id": user_id,
                "task_description": task_description,
                "timestamp": datetime.now().isoformat()
            }
            
            self.ravana_communicator.send_task_to_ravana(task)
            logger.info(f"Task from user {user_id} sent to RAVANA: {task_description}")
        except Exception as e:
            logger.error(f"Error handling task from user {user_id}: {e}")

if __name__ == "__main__":
    # For testing purposes
    conversational_ai = ConversationalAI()
    asyncio.run(conversational_ai.start())