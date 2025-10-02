import logging
import traceback
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import tempfile
import requests
from pathlib import Path

# Import required modules
from .emotional_intelligence.conversational_ei import ConversationalEmotionalIntelligence
from .memory.memory_interface import SharedMemoryInterface
from .communication.ravana_bridge import RAVANACommunicator
from .profiles.user_profile_manager import UserProfileManager
from .communication.data_models import UserPlatformProfile

# Import multi-modal service
from services.multi_modal_service import MultiModalService

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
        
        # Initialize enhanced multi-modal service
        self.multi_modal_service = MultiModalService()

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

    def process_user_message(self, platform: str, user_id: str, message: str, media_urls: Optional[List[str]] = None) -> str:
        """
        Process an incoming user message and generate a response.

        Args:
            platform: The platform the message came from (discord/telegram)
            user_id: The unique identifier of the user
            message: The user's message
            media_urls: Optional list of media URLs (images, audio, video) attached to the message

        Returns:
            The AI's response to the message
        """
        try:
            # Track user platform preference
            self._track_user_platform(user_id, platform)

            # Process multi-modal input if media is provided
            multi_modal_context = {}
            if media_urls:
                multi_modal_context = self._process_multi_modal_input(media_urls, message)

            # Get context from shared memory
            context = self.memory_interface.get_context(user_id)

            # Combine text and multi-modal context
            combined_context = {**context, **multi_modal_context}

            # Process message with emotional intelligence
            self.emotional_intelligence.set_persona("Balanced")
            emotional_context = self.emotional_intelligence.process_user_message(
                message, combined_context)

            # Generate response using emotional intelligence
            response = self.emotional_intelligence.generate_response(
                message, emotional_context)

            # Store conversation in memory
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_message": message,
                "ai_response": response,
                "emotional_context": emotional_context,
                "multi_modal_context": multi_modal_context  # Store multi-modal context
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
                        "multi_modal_context": multi_modal_context,
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

            # Enhance learning from user interaction
            self._enhance_learning_from_interaction(user_id, message, response, emotional_context)

            return response
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            return "I'm sorry, I encountered an error processing your message."

    def _process_multi_modal_input(self, media_urls: List[str], text_content: str) -> Dict[str, Any]:
        """
        Process multi-modal inputs (images, audio, video) and extract meaningful information.
        Enhanced to use the MultiModalService for better processing capabilities.
        
        Args:
            media_urls: List of URLs to media content
            text_content: Accompanying text content
            
        Returns:
            Dictionary containing processed multi-modal context
        """
        # Use the enhanced MultiModalService for processing
        return self._process_multi_modal_input_with_service(media_urls, text_content, "unknown_user")

    def _get_media_type_from_url(self, url: str) -> str:
        """
        Determine the media type from the URL.
        
        Args:
            url: URL to media content
            
        Returns:
            Media type string (e.g., 'image/jpeg', 'audio/mpeg', 'video/mp4')
        """
        # Simple extension-based detection
        lower_url = url.lower()
        
        if any(ext in lower_url for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
            return 'image'
        elif any(ext in lower_url for ext in ['.mp3', '.wav', '.ogg', '.m4a', '.flac']):
            return 'audio'
        elif any(ext in lower_url for ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']):
            return 'video'
        else:
            # Try to get content type from the URL headers
            try:
                response = requests.head(url, timeout=5)
                content_type = response.headers.get('content-type', 'unknown').lower()
                return content_type
            except:
                return 'unknown'

    def _download_media_to_temp(self, url: str) -> Optional[Path]:
        """
        Download media from URL to a temporary file.
        
        Args:
            url: URL to download from
            
        Returns:
            Path to temporary file, or None if download failed
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Create temporary file
            suffix = Path(url).suffix or '.tmp'
            temp_file = Path(tempfile.mktemp(suffix=suffix))
            
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            return temp_file
        except Exception as e:
            logger.error(f"Error downloading media from {url}: {e}")
            return None

    def _analyze_image(self, image_path: Path, text_content: str) -> Dict[str, Any]:
        """
        Analyze an image and extract meaningful information.
        
        Args:
            image_path: Path to the image file
            text_content: Accompanying text content
            
        Returns:
            Dictionary containing image analysis
        """
        try:
            # Try to use PIL/Pillow for basic image analysis
            from PIL import Image
            import os
            
            # Get basic image information
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
                format = img.format
                
                # Calculate aspect ratio
                aspect_ratio = width / height if height != 0 else 0
                
                # Basic analysis
                analysis = {
                    "description": f"Image attached to message: {text_content[:50]}{'...' if len(text_content) > 50 else ''}",
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(aspect_ratio, 2),
                    "mode": mode,
                    "format": format,
                    "file_size": os.path.getsize(image_path),
                    "elements_identified": ["dimensions", "format", "size"],  # Basic elements identified
                    "relevance_to_text": "medium"  # Would calculate actual relevance in a full implementation
                }
                
                # If the image is RGB or RGBA, we could do more analysis
                if mode in ['RGB', 'RGBA']:
                    # Calculate average color (simplified)
                    try:
                        # For performance, just sample a few pixels
                        pixels = list(img.getdata())
                        if len(pixels) > 0:
                            if mode == 'RGB':
                                avg_r = sum(p[0] for p in pixels[::max(1, len(pixels)//100)]) // min(100, len(pixels))  # Sample 100 pixels max
                                avg_g = sum(p[1] for p in pixels[::max(1, len(pixels)//100)]) // min(100, len(pixels))
                                avg_b = sum(p[2] for p in pixels[::max(1, len(pixels)//100)]) // min(100, len(pixels))
                                analysis["dominant_color"] = f"RGB({avg_r}, {avg_g}, {avg_b})"
                            elif mode == 'RGBA':
                                avg_r = sum(p[0] for p in pixels[::max(1, len(pixels)//100)]) // min(100, len(pixels))
                                avg_g = sum(p[1] for p in pixels[::max(1, len(pixels)//100)]) // min(100, len(pixels))
                                avg_b = sum(p[2] for p in pixels[::max(1, len(pixels)//100)]) // min(100, len(pixels))
                                avg_a = sum(p[3] for p in pixels[::max(1, len(pixels)//100)]) // min(100, len(pixels))
                                analysis["dominant_color"] = f"RGBA({avg_r}, {avg_g}, {avg_b}, {avg_a})"
                    except Exception as e:
                        logger.warning(f"Could not compute dominant color: {e}")
                
                return analysis
        except ImportError:
            logger.warning("PIL/Pillow not available, using fallback image analysis")
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
        
        # Fallback implementation
        import os
        analysis = {
            "description": f"Image attached to message: {text_content[:50]}{'...' if len(text_content) > 50 else ''}",
            "elements_identified": ["visual content"],  # Would contain actual identified elements
            "relevance_to_text": "medium",  # Would calculate actual relevance
            "file_info": {
                "exists": image_path.exists(),
                "size": os.path.getsize(image_path) if image_path.exists() else 0,
                "extension": image_path.suffix
            }
        }
        return analysis

    def _analyze_audio(self, audio_path: Path) -> Dict[str, Any]:
        """
        Analyze an audio file and extract meaningful information.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing audio analysis
        """
        try:
            # Try to use scipy for basic audio analysis
            from scipy.io import wavfile
            import numpy as np
            import os
            
            # Check file extension first
            if audio_path.suffix.lower() not in ['.wav', '.mp3', '.flac', '.ogg']:
                # For non-WAV files, we might need pydub
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(audio_path)
                    duration_seconds = len(audio) / 1000.0  # pydub returns duration in milliseconds
                    sample_rate = audio.frame_rate
                    channels = audio.channels
                    bit_depth = audio.sample_width * 8
                except ImportError:
                    logger.warning("pydub not available, using basic audio analysis")
                    # Basic file info only
                    return {
                        "transcript": "Audio content detected (transcription would appear here in a full implementation)",
                        "language": "unknown",
                        "duration": "unknown",
                        "confidence": 0.0,
                        "file_info": {
                            "exists": audio_path.exists(),
                            "size": os.path.getsize(audio_path) if audio_path.exists() else 0,
                            "extension": audio_path.suffix
                        }
                    }
            else:
                # For WAV files, use scipy
                sample_rate, data = wavfile.read(str(audio_path))
                duration_seconds = len(data) / sample_rate
                channels = 1 if len(data.shape) == 1 else data.shape[1]
                
                # Calculate basic audio properties
                max_amplitude = np.max(np.abs(data))
                avg_amplitude = np.mean(np.abs(data))
                
            analysis = {
                "transcript": "Audio content detected (transcription would appear here in a full implementation)",
                "language": "unknown",
                "duration": round(duration_seconds, 2),
                "confidence": 0.0,
                "sample_rate": sample_rate,
                "channels": channels,
                "bit_depth": bit_depth if 'bit_depth' in locals() else (data.dtype.itemsize * 8 if 'data' in locals() else "unknown"),
                "max_amplitude": int(max_amplitude) if 'max_amplitude' in locals() else "unknown",
                "avg_amplitude": round(float(avg_amplitude), 2) if 'avg_amplitude' in locals() else "unknown",
                "file_size": os.path.getsize(audio_path)
            }
            
            return analysis
        except ImportError as e:
            logger.warning(f"Audio analysis libraries not available ({e}), using fallback")
        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
        
        # Fallback implementation
        import os
        analysis = {
            "transcript": "Audio content detected (transcription would appear here in a full implementation)",
            "language": "unknown",
            "duration": "unknown",
            "confidence": 0.0,
            "file_info": {
                "exists": audio_path.exists(),
                "size": os.path.getsize(audio_path) if audio_path.exists() else 0,
                "extension": audio_path.suffix
            }
        }
        return analysis

    def _analyze_video(self, video_path: Path) -> Dict[str, Any]:
        """
        Analyze a video file and extract meaningful information.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video analysis
        """
        try:
            # Try to use opencv for basic video analysis
            import cv2
            import os
            
            # Open the video file
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate some basic metrics
            bitrate = (os.path.getsize(video_path) * 8) / duration if duration > 0 else 0
            
            # Sample a few frames to analyze
            key_frames = []
            if frame_count > 0:
                # Sample 5 frames evenly distributed throughout the video
                sample_indices = [int(i * frame_count / 5) for i in range(5)]
                
                for idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        # Calculate basic metrics for the frame
                        avg_brightness = cv2.mean(frame)[:3]  # Average brightness for RGB channels
                        key_frames.append({
                            "frame_number": idx,
                            "timestamp": idx / fps,
                            "avg_brightness": [float(b) for b in avg_brightness]
                        })
            
            cap.release()
            
            analysis = {
                "summary": "Video content detected (analysis would appear here in a full implementation)",
                "duration": round(duration, 2),
                "frame_count": frame_count,
                "fps": fps,
                "width": width,
                "height": height,
                "bitrate": round(bitrate, 2),
                "key_frames_analyzed": len(key_frames),
                "sampled_frames": key_frames,
                "detected_scenes": [],
                "file_size": os.path.getsize(video_path)
            }
            
            return analysis
        except ImportError:
            logger.warning("OpenCV not available, using fallback video analysis")
        except Exception as e:
            logger.error(f"Error in video analysis: {e}")
        
        # Fallback implementation
        import os
        analysis = {
            "summary": "Video content detected (analysis would appear here in a full implementation)",
            "duration": "unknown",
            "key_frames_analyzed": 0,
            "detected_scenes": [],
            "file_info": {
                "exists": video_path.exists(),
                "size": os.path.getsize(video_path) if video_path.exists() else 0,
                "extension": video_path.suffix
            }
        }
        return analysis

    async def process_user_message_async(self, platform: str, user_id: str, message: str, media_urls: Optional[List[str]] = None) -> str:
        """
        Process an incoming user message and generate a response asynchronously.

        Args:
            platform: The platform the message came from (discord/telegram)
            user_id: The unique identifier of the user
            message: The user's message
            media_urls: Optional list of media URLs (images, audio, video) attached to the message

        Returns:
            The AI's response to the message
        """
        # This is the async version of process_user_message
        return self.process_user_message(platform, user_id, message, media_urls)

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

    def _enhance_learning_from_interaction(self, user_id: str, user_message: str, ai_response: str, emotional_context: Dict[str, Any]):
        """
        Enhance learning and self-reflection based on user interactions.
        
        Args:
            user_id: The unique identifier of the user
            user_message: The user's message
            ai_response: The AI's response
            emotional_context: The emotional context of the interaction
        """
        try:
            # Create a learning experience from the interaction
            learning_experience = {
                "user_id": user_id,
                "interaction_type": "conversation",
                "user_input": user_message,
                "ai_output": ai_response,
                "emotional_context": emotional_context,
                "timestamp": datetime.now().isoformat(),
                "experience_id": f"learning_exp_{user_id}_{datetime.now().timestamp()}"
            }

            # Send learning experience to RAVANA for processing
            self.ravana_communicator.send_learning_experience_to_ravana(learning_experience)

            # Store in memory for future reference
            self.memory_interface.store_learning_experience(user_id, learning_experience)

            # Update user profile with interaction data
            self._update_user_profile_from_interaction(user_id, user_message, ai_response, emotional_context)

            # Extract potential improvement opportunities from the interaction
            self._extract_improvement_opportunities(user_message, ai_response, emotional_context)

            logger.info(f"Enhanced learning from interaction with user {user_id}")

        except Exception as e:
            logger.error(f"Error enhancing learning from interaction: {e}")

    def _update_user_profile_from_interaction(self, user_id: str, user_message: str, ai_response: str, emotional_context: Dict[str, Any]):
        """
        Update user profile based on interaction data.
        
        Args:
            user_id: The unique identifier of the user
            user_message: The user's message
            ai_response: The AI's response
            emotional_context: The emotional context of the interaction
        """
        try:
            # Get current user profile
            profile = self.user_profile_manager.get_user_platform_profile(user_id)
            if not profile:
                from .communication.data_models import UserPlatformProfile
                profile = UserPlatformProfile(
                    user_id=user_id,
                    last_platform="unknown",
                    platform_user_id=user_id,
                    preferences={},
                    last_interaction=datetime.now()
                )

            # Update interaction statistics
            if 'interaction_stats' not in profile.preferences:
                profile.preferences['interaction_stats'] = {
                    'total_interactions': 0,
                    'positive_interactions': 0,
                    'negative_interactions': 0,
                    'topics_discussed': {},
                    'preferred_communication_styles': {}
                }

            stats = profile.preferences['interaction_stats']
            stats['total_interactions'] += 1

            # Analyze emotional context to determine interaction sentiment
            if emotional_context.get('overall_sentiment', 'neutral') in ['positive', 'happy', 'excited', 'confident']:
                stats['positive_interactions'] += 1
            elif emotional_context.get('overall_sentiment', 'neutral') in ['negative', 'sad', 'frustrated', 'angry']:
                stats['negative_interactions'] += 1

            # Extract topics from user message
            topics = self._extract_topics_from_message(user_message)
            for topic in topics:
                if topic not in stats['topics_discussed']:
                    stats['topics_discussed'][topic] = 0
                stats['topics_discussed'][topic] += 1

            # Update preferred communication styles
            communication_style = emotional_context.get('communication_style', 'balanced')
            if communication_style not in stats['preferred_communication_styles']:
                stats['preferred_communication_styles'][communication_style] = 0
            stats['preferred_communication_styles'][communication_style] += 1

            # Update last interaction timestamp
            profile.last_interaction = datetime.now()

            # Save updated profile
            self.user_profile_manager.set_user_platform_profile(user_id, profile)

        except Exception as e:
            logger.error(f"Error updating user profile from interaction: {e}")

    def _extract_topics_from_message(self, message: str) -> List[str]:
        """
        Extract topics from a user message.
        
        Args:
            message: The user's message
            
        Returns:
            List of extracted topics
        """
        try:
            # Try using scikit-learn for more sophisticated topic extraction
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
            import numpy as np
            import re
            
            # Clean the message text
            cleaned_message = re.sub(r'[^\w\s]', ' ', message.lower())
            
            # Define a list of potential topics with seed words
            topic_definitions = [
                {"name": "technology", "keywords": ["computer", "software", "programming", "code", "algorithm", "ai", "artificial intelligence", "tech", "app", "application", "data", "database", "machine learning", "neural network", "python", "javascript", "java", "c++", "developer", "coding", "framework", "api", "cloud", "web", "mobile", "robot", "system", "platform", "interface", "software", "hardware", "digital", "electronic", "internet", "network", "server", "client", "database"]},
                {"name": "science", "keywords": ["science", "physics", "chemistry", "biology", "research", "experiment", "discovery", "lab", "laboratory", "scientist", "hypothesis", "theory", "evidence", "research", "study", "analysis", "laboratory", "molecule", "atom", "cell", "gene", "protein", "energy", "force", "matter", "universe", "space", "planet", "star", "galaxy", "quantum", "relativity", "matter", "energy"]},
                {"name": "philosophy", "keywords": ["philosophy", "think", "thought", "meaning", "purpose", "ethics", "morality", "existence", "truth", "knowledge", "reality", "consciousness", "mind", "idea", "concept", "belief", "value", "wisdom", "reason", "logic", "argument", "principle", "truth", "justice", "good", "right", "wrong", "free will", "determinism", "skepticism", "nihilism", "existentialism"]},
                {"name": "creativity", "keywords": ["creative", "invent", "design", "art", "music", "literature", "imagination", "paint", "draw", "sculpt", "compose", "write", "create", "innovate", "original", "inspire", "express", "artistic", "aesthetic", "beauty", "form", "style", "color", "melody", "harmony", "rhythm", "poetry", "novel", "story", "narrative", "character", "plot", "theme", "metaphor", "symbol", "imagination"]},
                {"name": "problem solving", "keywords": ["problem", "solution", "solve", "fix", "debug", "troubleshoot", "issue", "challenge", "obstacle", "difficulty", "struggle", "overcome", "resolve", "address", "tackle", "approach", "method", "strategy", "technique", "algorithm", "process", "procedure", "technique", "approach", "method", "tool", "fix", "improve", "optimize", "enhance", "refine"]},
                {"name": "learning", "keywords": ["learn", "study", "education", "knowledge", "understand", "explain", "teach", "instruct", "guide", "mentor", "student", "teacher", "school", "university", "course", "lesson", "subject", "topic", "material", "curriculum", "academic", "intellectual", "cognitive", "mind", "brain", "memory", "comprehend", "grasp", "apprehend", "absorb", "acquire", "master", "proficient", "skilled"]},
                {"name": "personal development", "keywords": ["improve", "better", "growth", "development", "skill", "ability", "talent", "competency", "capability", "strength", "weakness", "progress", "advance", "evolve", "mature", "enhance", "refine", "perfect", "excel", "succeed", "achieve", "accomplish", "goal", "objective", "target", "aim", "purpose", "mission", "vision", "plan", "strategy", "self", "personal", "growth", "improvement"]},
                {"name": "social", "keywords": ["friend", "family", "people", "social", "relationship", "community", "society", "group", "team", "collaborate", "cooperate", "communicate", "interact", "connect", "bond", "trust", "love", "care", "support", "help", "assist", "aid", "cooperation", "collaboration", "partnership", "alliance", "friendship", "colleague", "associate", "acquaintance", "contact", "network", "connection"]},
            ]
            
            # Check for matches with seed words first
            detected_topics = set()
            message_lower = message.lower()
            
            for topic_def in topic_definitions:
                topic_name = topic_def["name"]
                keywords = topic_def["keywords"]
                if any(keyword in message_lower for keyword in keywords):
                    detected_topics.add(topic_name)
            
            # If we detected topics with simple matching, return those
            if detected_topics:
                return list(detected_topics)
            
            # Fallback to more sophisticated NLP approach if needed
            # For now, just return general if no topics were detected
            return ['general']
            
        except ImportError:
            logger.warning("Scikit-learn not available, using fallback topic extraction")
        except Exception as e:
            logger.error(f"Error in topic extraction: {e}")
        
        # Fallback: use the original simple implementation
        topics = []
        message_lower = message.lower()
        
        # Common topic keywords
        topic_keywords = {
            'technology': ['tech', 'computer', 'software', 'programming', 'code', 'algorithm', 'ai', 'artificial intelligence'],
            'science': ['science', 'physics', 'chemistry', 'biology', 'research', 'experiment', 'discovery'],
            'philosophy': ['philosophy', 'think', 'thought', 'meaning', 'purpose', 'ethics', 'morality'],
            'creativity': ['creative', 'invent', 'design', 'art', 'music', 'literature', 'imagination'],
            'problem solving': ['problem', 'solution', 'solve', 'fix', 'debug', 'troubleshoot'],
            'learning': ['learn', 'study', 'education', 'knowledge', 'understand', 'explain'],
            'personal development': ['improve', 'better', 'growth', 'development', 'skill'],
            'social': ['friend', 'family', 'people', 'social', 'relationship', 'community']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ['general']

    def _extract_improvement_opportunities(self, user_message: str, ai_response: str, emotional_context: Dict[str, Any]):
        """
        Extract potential improvement opportunities from user interactions.
        
        Args:
            user_message: The user's message
            ai_response: The AI's response
            emotional_context: The emotional context of the interaction
        """
        try:
            # Analyze the interaction for improvement opportunities
            opportunities = []
            
            # Check for confusion or unclear responses
            if 'confused' in ai_response.lower() or 'unclear' in ai_response.lower() or \
               'not sure' in ai_response.lower() or 'unsure' in ai_response.lower():
                opportunities.append({
                    'type': 'response_clarity',
                    'description': 'Response contained uncertainty indicators',
                    'priority': 'medium',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Check for repetitive responses
            if hasattr(self, '_last_responses') and len(self._last_responses) > 2:
                if ai_response in self._last_responses[-2:]:
                    opportunities.append({
                        'type': 'response_repetition',
                        'description': 'Response was repetitive compared to recent responses',
                        'priority': 'low',
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Check for emotional alignment issues
            if emotional_context.get('emotional_alignment', 'good') == 'poor':
                opportunities.append({
                    'type': 'emotional_alignment',
                    'description': 'Poor emotional alignment with user input',
                    'priority': 'high',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Store recent responses for repetition checking
            if not hasattr(self, '_last_responses'):
                self._last_responses = []
            self._last_responses.append(ai_response)
            if len(self._last_responses) > 10:
                self._last_responses.pop(0)
            
            # Send improvement opportunities to RAVANA if any were found
            if opportunities:
                improvement_data = {
                    "type": "interaction_improvement_opportunities",
                    "opportunities": opportunities,
                    "user_message": user_message,
                    "ai_response": ai_response,
                    "emotional_context": emotional_context,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.ravana_communicator.send_improvement_opportunities_to_ravana(improvement_data)

        except Exception as e:
            logger.error(f"Error extracting improvement opportunities: {e}")
