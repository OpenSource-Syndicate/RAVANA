import logging
import traceback
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Any

# Import required modules
from .emotional_intelligence.conversational_ei import ConversationalEmotionalIntelligence
from .memory.memory_interface import SharedMemoryInterface
from .communication.ravana_bridge import RAVANACommunicator
from .profiles.user_profile_manager import UserProfileManager

logger = logging.getLogger(__name__)


class ConversationalAI:
    def __init__(self):
        """
        Initialize the Conversational AI module with all required components.
        """
        # Initialize core components
        self.emotional_intelligence = ConversationalEmotionalIntelligence()
        self.memory_interface = SharedMemoryInterface()
        self.ravana_communicator = RAVANACommunicator("conversational_ai_bridge", self)
        self.user_profile_manager = UserProfileManager()
        
        # Initialize shutdown event
        self._shutdown = asyncio.Event()
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize bots (will be set up in start method)
        self.discord_bot = None
        self.telegram_bot = None
        
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

    async def start(self, standalone: bool = True):
        """
        Start the Conversational AI module.
        
        Args:
            standalone: Whether to run in standalone mode or integrated with RAVANA
        """
        try:
            logger.info(f"Starting Conversational AI module in {'standalone' if standalone else 'integrated'} mode")
            
            # Initialize bots if in standalone mode
            if standalone:
                # Initialize Discord bot if configured and enabled
                if (self.config.get("platforms", {}).get("discord", {}).get("enabled", False) and 
                    self.config.get("discord_token")):
                    try:
                        from .bots.discord_bot import DiscordBot
                        discord_config = self.config["platforms"]["discord"]
                        self.discord_bot = DiscordBot(
                            token=self.config["discord_token"],
                            command_prefix=discord_config["command_prefix"],
                            conversational_ai=self
                        )
                        logger.info("Discord bot initialized")
                    except Exception as e:
                        logger.error(f"Failed to initialize Discord bot: {e}")
                
                # Initialize Telegram bot if configured and enabled
                if (self.config.get("platforms", {}).get("telegram", {}).get("enabled", False) and 
                    self.config.get("telegram_token")):
                    try:
                        from .bots.telegram_bot import TelegramBot
                        telegram_config = self.config["platforms"]["telegram"]
                        self.telegram_bot = TelegramBot(
                            token=self.config["telegram_token"],
                            command_prefix=telegram_config["command_prefix"],
                            conversational_ai=self
                        )
                        logger.info("Telegram bot initialized")
                    except Exception as e:
                        logger.error(f"Failed to initialize Telegram bot: {e}")
            
            # Start RAVANA communicator
            await self.ravana_communicator.start()
            
            # Start bots if in standalone mode
            if standalone:
                # Start Discord bot if available
                if self.discord_bot:
                    try:
                        await self.discord_bot.start()
                    except Exception as e:
                        logger.error(f"Failed to start Discord bot: {e}")
                
                # Start Telegram bot if available
                if self.telegram_bot:
                    try:
                        await self.telegram_bot.start()
                    except Exception as e:
                        logger.error(f"Failed to start Telegram bot: {e}")
            
            # Main loop for integrated mode
            if not standalone:
                while not self._shutdown.is_set():
                    # In integrated mode, the module is controlled by the main RAVANA system
                    await asyncio.sleep(1)
            
            logger.info("Conversational AI module started successfully")
        except Exception as e:
            logger.error(f"Error starting Conversational AI module: {e}")
            raise

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
            # Get context from shared memory
            context = self.memory_interface.get_context(user_id)
            
            # Process message with emotional intelligence
            self.emotional_intelligence.set_persona("Balanced")
            emotional_context = self.emotional_intelligence.process_user_message(message, context)
            
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
        try:
            # Get context from shared memory
            context = self.memory_interface.get_context(user_id)
            
            # Process message with emotional intelligence
            self.emotional_intelligence.set_persona("Balanced")
            emotional_context = self.emotional_intelligence.process_user_message(message, context)
            
            # Generate response using emotional intelligence (async version)
            response = await self._generate_response_async(message, emotional_context)
            
            # Store conversation in memory
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_message": message,
                "ai_response": response,
                "emotional_context": emotional_context
            }
            
            self.memory_interface.store_conversation(user_id, conversation_entry)
            
            # Extract thoughts from the conversation (async version)
            thoughts = await self._extract_thoughts_from_conversation_async(
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

    async def _generate_response_async(self, prompt: str, emotional_state: Dict[str, Any]) -> str:
        """
        Generate an emotionally-aware response using async LLM calls.
        
        Args:
            prompt: The user's message
            emotional_state: Current emotional state
            
        Returns:
            Generated response
        """
        try:
            # Use the async version from emotional intelligence module
            response = await self.emotional_intelligence.generate_response_async(prompt, emotional_state)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self.emotional_intelligence._generate_fallback_response(prompt, emotional_state)

    async def _extract_thoughts_from_conversation_async(self, user_message: str, ai_response: str, 
                                                       emotional_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract meaningful thoughts and insights from the conversation asynchronously.
        
        Args:
            user_message: The user's message
            ai_response: The AI's response
            emotional_context: Current emotional context
            
        Returns:
            List of extracted thoughts as structured dictionaries
        """
        try:
            # Use the async version from emotional intelligence module
            thoughts = await self.emotional_intelligence.extract_thoughts_from_conversation_async(
                user_message, ai_response, emotional_context)
            return thoughts
        except Exception as e:
            logger.error(f"Error extracting thoughts from conversation: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def _synchronize_emotional_context(self, user_id: str, emotional_context: Dict[str, Any]):
        """
        Synchronize the user's emotional context with the RAVANA system.
        
        Args:
            user_id: The unique identifier of the user
            emotional_context: The current emotional context to synchronize
        """
        try:
            # Create emotional state payload for RAVANA
            emotional_payload = {
                "user_id": user_id,
                "emotional_state": emotional_context,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to RAVANA for synchronization
            self.ravana_communicator.send_thought_to_ravana(emotional_payload)
        except Exception as e:
            logger.error(f"Error synchronizing emotional context with RAVANA: {e}")
            
    def send_message_to_user(self, user_id: str, message: str):
        """
        Send a message to a user (placeholder for actual implementation).
        
        Args:
            user_id: The unique identifier of the user
            message: The message to send
        """
        # This would need to be implemented to actually send messages through the appropriate platform
        logger.info(f"Sending message to user {user_id}: {message}")
