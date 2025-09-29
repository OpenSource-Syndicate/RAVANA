import logging
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime
from ..communication.data_models import UserPlatformProfile

logger = logging.getLogger(__name__)


class UserProfileManager:
    def __init__(self, profiles_path: str = "profiles"):
        """
        Initialize the user profile manager.

        Args:
            profiles_path: Path to the user profiles storage
        """
        self.profiles_path = profiles_path
        self._ensure_profiles_directory()

    def _ensure_profiles_directory(self):
        """Ensure the profiles directory exists."""
        if not os.path.exists(self.profiles_path):
            os.makedirs(self.profiles_path)
            logger.info(f"Created profiles directory: {self.profiles_path}")

    def _get_user_profile_path(self, user_id: str) -> str:
        """Get the file path for a user's profile."""
        return os.path.join(self.profiles_path, f"user_{user_id}_profile.json")

    def get_user_profile(self, user_id: str, platform: str = "discord") -> Dict[str, Any]:
        """
        Get or create a user profile.

        Args:
            user_id: Unique identifier for the user
            platform: Platform the user is on

        Returns:
            Dictionary containing user profile data
        """
        try:
            profile_path = self._get_user_profile_path(user_id)

            # Load existing profile if it exists
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, 'r') as f:
                        profile = json.load(f)
                    return profile
                except Exception as e:
                    logger.error(
                        f"Error loading profile for user {user_id}: {e}")

            # Create new profile
            profile = self._create_user_profile(user_id, platform)

            # Save profile
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)

            return profile
        except Exception as e:
            logger.error(f"Error getting profile for user {user_id}: {e}")
            # Return a basic profile as fallback
            return self._create_user_profile(user_id, platform)

    def _create_user_profile(self, user_id: str, platform: str) -> Dict[str, Any]:
        """Create a new user profile."""
        return {
            "user_id": user_id,
            "platform": platform,
            "username": f"user_{user_id}",
            "created_at": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "personality": {
                "persona": "Balanced",
                "preferences": {}
            },
            "chat_history": [],
            "emotional_state": {},
            "interests": [],
            "interaction_count": 0
        }

    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """
        Update a user's profile.

        Args:
            user_id: Unique identifier for the user
            profile_data: Updated profile data
        """
        try:
            profile_path = self._get_user_profile_path(user_id)

            # Update last seen timestamp
            profile_data["last_seen"] = datetime.now().isoformat()

            # Save profile
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)

            logger.debug(f"Updated profile for user {user_id}")
        except Exception as e:
            logger.error(f"Error updating profile for user {user_id}: {e}")

    def update_username(self, user_id: str, username: str):
        """
        Update a user's username.

        Args:
            user_id: Unique identifier for the user
            username: New username
        """
        try:
            profile = self.get_user_profile(user_id)
            profile["username"] = username
            self.update_user_profile(user_id, profile)
            logger.debug(f"Updated username for user {user_id} to {username}")
        except Exception as e:
            logger.error(f"Error updating username for user {user_id}: {e}")

    def store_chat_message(self, user_id: str, message: Dict[str, Any]):
        """
        Store a chat message in the user's profile.

        Args:
            user_id: Unique identifier for the user
            message: Message data to store
        """
        try:
            profile = self.get_user_profile(user_id)

            # Add message to chat history
            profile["chat_history"].append(message)

            # Keep only recent messages (last 100)
            if len(profile["chat_history"]) > 100:
                profile["chat_history"] = profile["chat_history"][-100:]

            # Update interaction count
            profile["interaction_count"] = profile.get(
                "interaction_count", 0) + 1

            # Save updated profile
            self.update_user_profile(user_id, profile)

            logger.debug(f"Stored chat message for user {user_id}")
        except Exception as e:
            logger.error(f"Error storing chat message for user {user_id}: {e}")

    def _get_user_platform_profile_path(self, user_id: str) -> str:
        """Get the file path for a user's platform profile."""
        return os.path.join(self.profiles_path, f"user_{user_id}_platform_profile.json")

    def get_user_platform_profile(self, user_id: str) -> Optional[UserPlatformProfile]:
        """
        Get a user's platform profile.

        Args:
            user_id: Unique identifier for the user

        Returns:
            UserPlatformProfile object or None if not found
        """
        try:
            profile_path = self._get_user_platform_profile_path(user_id)

            # Load existing platform profile if it exists
            if os.path.exists(profile_path):
                try:
                    with open(profile_path, 'r') as f:
                        profile_data = json.load(f)
                    return UserPlatformProfile.from_dict(profile_data)
                except Exception as e:
                    logger.error(
                        f"Error loading platform profile for user {user_id}: {e}")

            return None
        except Exception as e:
            logger.error(
                f"Error getting platform profile for user {user_id}: {e}")
            return None

    def set_user_platform_profile(self, user_id: str, profile: UserPlatformProfile):
        """
        Set a user's platform profile.

        Args:
            user_id: Unique identifier for the user
            profile: UserPlatformProfile object to store
        """
        try:
            profile_path = self._get_user_platform_profile_path(user_id)

            # Save platform profile
            with open(profile_path, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)

            logger.debug(f"Set platform profile for user {user_id}")
        except Exception as e:
            logger.error(
                f"Error setting platform profile for user {user_id}: {e}")
