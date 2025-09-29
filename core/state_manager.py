"""
State Manager for RAVANA AGI System

This module provides enhanced state persistence and recovery capabilities
beyond the basic functionality in ShutdownCoordinator.
"""

import json
import logging
import gzip
import shutil
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

from core.config import Config

logger = logging.getLogger(__name__)


class StateManager:
    """
    Enhanced state management for the RAVANA AGI system.

    Provides state persistence, recovery, and validation capabilities
    with support for multiple state formats and recovery strategies.
    """

    def __init__(self, base_dir: str = "."):
        """
        Initialize the state manager.

        Args:
            base_dir: Base directory for state files
        """
        self.base_dir = Path(base_dir)
        self.state_file = self.base_dir / Config.SHUTDOWN_STATE_FILE
        self.backup_dir = self.base_dir / "state_backups"
        self.backup_dir.mkdir(exist_ok=True)

        # State validation rules
        self.required_fields = ["timestamp", "version", "shutdown_info"]
        self.supported_versions = ["1.0", "1.1"]

        logger.info(
            f"StateManager initialized - State file: {self.state_file}")

    async def save_state(self, state_data: Dict[str, Any]) -> bool:
        """
        Save system state with validation and backup.

        Args:
            state_data: State data to save

        Returns:
            True if save was successful
        """
        try:
            # Add metadata
            state_data.update({
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.1",  # Updated version
                "saved_by": "StateManager"
            })

            # Validate state data
            if not self._validate_state_data(state_data):
                logger.error("State data validation failed")
                return False

            # Create backup if existing state file exists
            if self.state_file.exists():
                await self._create_backup()

            # Save state data with compression if enabled
            if getattr(Config, 'SHUTDOWN_COMPRESSION_ENABLED', True):
                await self._save_compressed_state(state_data)
            else:
                # Save state data
                with open(self.state_file, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f, indent=2, default=str)

            logger.info(f"State saved successfully to {self.state_file}")

            # Clean up old backups
            await self._cleanup_old_backups()

            return True

        except Exception as e:
            logger.error(f"Error saving state: {e}", exc_info=True)
            return False

    async def _save_compressed_state(self, state_data: Dict[str, Any]) -> None:
        """
        Save state data with compression.

        Args:
            state_data: State data to save with compression
        """
        try:
            # Create temporary file for uncompressed data
            temp_file = self.state_file.with_suffix('.tmp')

            # Save uncompressed data to temporary file
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, default=str)

            # Compress to final file
            with open(temp_file, 'rb') as f_in:
                with gzip.open(self.state_file.with_suffix('.json.gz'), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove temporary file
            temp_file.unlink()

            logger.info(
                f"Compressed state saved to {self.state_file.with_suffix('.json.gz')}")

        except Exception as e:
            logger.error(f"Error saving compressed state: {e}")
            # Fall back to uncompressed save
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, default=str)

    async def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Load system state with validation and recovery options.

        Returns:
            Loaded state data or None if not available
        """
        try:
            # Check for compressed state file first
            compressed_file = self.state_file.with_suffix('.json.gz')
            if compressed_file.exists():
                return await self._load_compressed_state(compressed_file)

            # Fall back to uncompressed file
            if not self.state_file.exists():
                logger.info("No state file found")
                return None

            # Load state data
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)

            # Validate loaded state
            if not self._validate_state_data(state_data):
                logger.warning(
                    "State file validation failed, attempting backup recovery")
                return await self._recover_from_backup()

            # Check if state is too old
            if self._is_state_too_old(state_data):
                logger.warning("State file is too old, skipping recovery")
                return None

            logger.info("State loaded successfully")
            return state_data

        except Exception as e:
            logger.error(f"Error loading state: {e}")
            logger.info("Attempting backup recovery...")
            return await self._recover_from_backup()

    async def _load_compressed_state(self, compressed_file: Path) -> Optional[Dict[str, Any]]:
        """
        Load compressed state data.

        Args:
            compressed_file: Path to compressed state file

        Returns:
            Loaded state data or None if not available
        """
        try:
            # Decompress file
            with gzip.open(compressed_file, 'rb') as f_in:
                state_data = json.loads(f_in.read().decode('utf-8'))

            # Validate loaded state
            if not self._validate_state_data(state_data):
                logger.warning("Compressed state file validation failed")
                return None

            # Check if state is too old
            if self._is_state_too_old(state_data):
                logger.warning(
                    "Compressed state file is too old, skipping recovery")
                return None

            logger.info("Compressed state loaded successfully")
            return state_data

        except Exception as e:
            logger.error(f"Error loading compressed state: {e}")
            return None

    def _validate_state_data(self, state_data: Dict[str, Any]) -> bool:
        """Validate state data structure."""
        try:
            # Check required fields
            for field in self.required_fields:
                if field not in state_data:
                    logger.error(f"Missing required field: {field}")
                    return False

            # Check version compatibility
            version = state_data.get("version")
            if version not in self.supported_versions:
                logger.error(f"Unsupported state version: {version}")
                return False

            # Validate timestamp
            timestamp_str = state_data.get("timestamp")
            if timestamp_str:
                datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

            return True

        except Exception as e:
            logger.error(f"State validation error: {e}")
            return False

    def _is_state_too_old(self, state_data: Dict[str, Any]) -> bool:
        """Check if state is too old to be useful."""
        try:
            timestamp_str = state_data.get("timestamp")
            if not timestamp_str:
                return True

            state_time = datetime.fromisoformat(
                timestamp_str.replace('Z', '+00:00'))
            age = datetime.utcnow() - state_time.replace(tzinfo=None)

            # Consider state too old if older than 24 hours
            max_age = timedelta(hours=24)
            return age > max_age

        except Exception as e:
            logger.warning(f"Error checking state age: {e}")
            return True

    async def _create_backup(self) -> None:
        """Create backup of current state file."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = self.backup_dir / f"state_backup_{timestamp}.json"

            # Check for compressed state file
            compressed_file = self.state_file.with_suffix('.json.gz')
            source_file = compressed_file if compressed_file.exists() else self.state_file

            # Copy current state file to backup
            if source_file.exists():
                shutil.copy2(source_file, backup_file)
                logger.info(f"State backup created: {backup_file}")

        except Exception as e:
            logger.warning(f"Error creating state backup: {e}")

    async def _recover_from_backup(self) -> Optional[Dict[str, Any]]:
        """Attempt to recover state from backup files."""
        try:
            backup_files = list(self.backup_dir.glob("state_backup_*.json*"))

            if not backup_files:
                logger.info("No backup files found")
                return None

            # Sort by modification time (newest first)
            backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

            for backup_file in backup_files:
                try:
                    # Handle compressed backup files
                    if backup_file.suffix == '.gz':
                        with gzip.open(backup_file, 'rb') as f:
                            state_data = json.loads(f.read().decode('utf-8'))
                    else:
                        with open(backup_file, 'r', encoding='utf-8') as f:
                            state_data = json.load(f)

                    if self._validate_state_data(state_data):
                        logger.info(
                            f"Recovered state from backup: {backup_file}")
                        return state_data

                except Exception as e:
                    logger.warning(f"Failed to load backup {backup_file}: {e}")
                    continue

            logger.warning("All backup recovery attempts failed")
            return None

        except Exception as e:
            logger.error(f"Error during backup recovery: {e}")
            return None

    async def _cleanup_old_backups(self, max_backups: int = 5) -> None:
        """Clean up old backup files."""
        try:
            backup_files = list(self.backup_dir.glob("state_backup_*.json*"))

            if len(backup_files) <= max_backups:
                return

            # Sort by modification time (oldest first)
            backup_files.sort(key=lambda f: f.stat().st_mtime)

            # Remove oldest backups
            files_to_remove = backup_files[:-max_backups]
            for backup_file in files_to_remove:
                backup_file.unlink()
                logger.debug(f"Removed old backup: {backup_file}")

            logger.info(f"Cleaned up {len(files_to_remove)} old backup files")

        except Exception as e:
            logger.warning(f"Error cleaning up old backups: {e}")

    def cleanup_state_files(self) -> None:
        """Clean up state files after successful recovery."""
        try:
            files_to_clean = [
                self.state_file,
                self.base_dir / "action_cache.pkl",
                self.base_dir / "shutdown_log.json"
            ]

            # Clean up compressed state files
            compressed_file = self.state_file.with_suffix('.json.gz')
            if compressed_file.exists():
                files_to_clean.append(compressed_file)

            # Clean up component state files
            for state_file in self.base_dir.glob("*_state.json"):
                files_to_clean.append(state_file)

            for state_file in files_to_clean:
                if state_file.exists():
                    state_file.unlink()
                    logger.debug(f"Cleaned up state file: {state_file}")

            # Clean up binary state files
            for pkl_file in self.base_dir.glob("*.pkl"):
                pkl_file.unlink()
                logger.debug(f"Cleaned up binary file: {pkl_file}")

            logger.info("State files cleaned up after recovery")

        except Exception as e:
            logger.warning(f"Error cleaning up state files: {e}")


# Utility functions for backward compatibility
async def save_system_state(state_data: Dict[str, Any]) -> bool:
    """Save system state using the default state manager."""
    state_manager = StateManager()
    return await state_manager.save_state(state_data)


async def load_system_state() -> Optional[Dict[str, Any]]:
    """Load system state using the default state manager."""
    state_manager = StateManager()
    return await state_manager.load_state()


def cleanup_all_state_files() -> None:
    """Clean up all state files."""
    state_manager = StateManager()
    state_manager.cleanup_state_files()
