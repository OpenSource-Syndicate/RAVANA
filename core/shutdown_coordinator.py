"""
Graceful Shutdown Coordinator for RAVANA AGI System

This module provides centralized shutdown management with timeout handling,
resource cleanup coordination, and state persistence.
"""

import asyncio
import logging
import json
import os
import pickle
import tempfile
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum

from core.config import Config

logger = logging.getLogger(__name__)


class ShutdownPhase(Enum):
    """Enumeration of shutdown phases."""
    PRE_SHUTDOWN_VALIDATION = "pre_shutdown_validation"
    SIGNAL_RECEIVED = "signal_received"
    COMPONENT_NOTIFICATION = "component_notification"
    TASKS_STOPPING = "tasks_stopping"
    RESOURCE_CLEANUP = "resource_cleanup"
    SERVICE_SHUTDOWN = "service_shutdown"
    STATE_PERSISTENCE = "state_persistence"
    FINAL_VALIDATION = "final_validation"
    SHUTDOWN_COMPLETE = "shutdown_complete"


class ShutdownPriority(Enum):
    """Enumeration of shutdown priorities."""
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class Shutdownable:
    """
    Interface for components that support graceful shutdown.
    """
    
    async def prepare_shutdown(self) -> bool:
        """
        Prepare component for shutdown.
        
        Returns:
            bool: True if preparation was successful, False otherwise
        """
        return True
    
    async def shutdown(self, timeout: float = 30.0) -> bool:
        """
        Shutdown component with timeout.
        
        Args:
            timeout: Maximum time to wait for shutdown
            
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        return True
    
    def get_shutdown_metrics(self) -> Dict[str, Any]:
        """
        Get shutdown-related metrics for this component.
        
        Returns:
            Dict containing shutdown metrics
        """
        return {}


class ComponentRegistration:
    """Represents a registered component for shutdown."""
    
    def __init__(self, component: Any, priority: ShutdownPriority, is_async: bool = True):
        self.component = component
        self.priority = priority
        self.is_async = is_async


class ShutdownCoordinator:
    """
    Centralized coordinator for graceful shutdown of the RAVANA AGI system.
    
    Manages the shutdown process across multiple phases with timeout handling,
    resource cleanup, and state persistence.
    """
    
    def __init__(self, agi_system=None):
        """
        Initialize the shutdown coordinator.
        
        Args:
            agi_system: Reference to the main AGI system
        """
        self.agi_system = agi_system
        self.shutdown_in_progress = False
        self.shutdown_start_time = None
        self.current_phase = None
        
        # Registered cleanup handlers
        self.cleanup_handlers: List[Callable] = []
        self.async_cleanup_handlers: List[Callable] = []
        
        # Registered components
        self.registered_components: List[ComponentRegistration] = []
        
        # State tracking
        self.shutdown_state = {
            "phase": None,
            "start_time": None,
            "completed_phases": [],
            "errors": [],
            "component_metrics": {}
        }
        
        logger.info("ShutdownCoordinator initialized")
    
    def register_component(self, component: Any, priority: ShutdownPriority, is_async: bool = True):
        """
        Register a component for shutdown management.
        
        Args:
            component: Component to register
            priority: Priority level for shutdown
            is_async: Whether the component shutdown is async
        """
        registration = ComponentRegistration(component, priority, is_async)
        self.registered_components.append(registration)
        
        # Sort components by priority (highest first)
        self.registered_components.sort(key=lambda x: x.priority.value)
        
        logger.debug(f"Registered component with priority {priority.name}: {type(component).__name__}")
    
    def register_cleanup_handler(self, handler: Callable, is_async: bool = False):
        """
        Register a cleanup handler to be called during shutdown.
        
        Args:
            handler: Cleanup function to call
            is_async: Whether the handler is async
        """
        if is_async:
            self.async_cleanup_handlers.append(handler)
        else:
            self.cleanup_handlers.append(handler)
        
        logger.debug(f"Registered {'async' if is_async else 'sync'} cleanup handler: {handler.__name__}")
    
    async def initiate_shutdown(self, reason: str = "manual"):
        """
        Initiate the graceful shutdown process.
        
        Args:
            reason: Reason for shutdown (signal, manual, error, etc.)
        """
        if self.shutdown_in_progress:
            logger.warning("Shutdown already in progress, ignoring duplicate request")
            return
        
        self.shutdown_in_progress = True
        self.shutdown_start_time = datetime.utcnow()
        
        logger.info(f"🛑 Initiating graceful shutdown - Reason: {reason}")
        logger.info(f"📋 Shutdown timeout: {Config.SHUTDOWN_TIMEOUT}s, Force timeout: {Config.FORCE_SHUTDOWN_AFTER}s")
        
        self.shutdown_state.update({
            "reason": reason,
            "start_time": self.shutdown_start_time.isoformat(),
            "timeout_config": {
                "graceful_timeout": Config.SHUTDOWN_TIMEOUT,
                "force_timeout": Config.FORCE_SHUTDOWN_AFTER
            }
        })
        
        try:
            # Start shutdown with timeout
            await asyncio.wait_for(
                self._execute_shutdown_phases(),
                timeout=Config.FORCE_SHUTDOWN_AFTER
            )
            logger.info("✅ Graceful shutdown completed successfully")
            
        except asyncio.TimeoutError:
            logger.error(f"⚠️  Graceful shutdown exceeded {Config.FORCE_SHUTDOWN_AFTER}s timeout, forcing shutdown")
            await self._force_shutdown()
            
        except Exception as e:
            logger.error(f"❌ Error during graceful shutdown: {e}", exc_info=True)
            self.shutdown_state["errors"].append(str(e))
            await self._force_shutdown()
        
        finally:
            self._log_shutdown_summary()
    
    async def _execute_shutdown_phases(self):
        """Execute all shutdown phases in sequence."""
        phases = [
            (ShutdownPhase.PRE_SHUTDOWN_VALIDATION, self._phase_pre_shutdown_validation),
            (ShutdownPhase.SIGNAL_RECEIVED, self._phase_signal_received),
            (ShutdownPhase.COMPONENT_NOTIFICATION, self._phase_component_notification),
            (ShutdownPhase.TASKS_STOPPING, self._phase_stop_background_tasks),
            (ShutdownPhase.RESOURCE_CLEANUP, self._phase_resource_cleanup),
            (ShutdownPhase.SERVICE_SHUTDOWN, self._phase_service_shutdown),
            (ShutdownPhase.STATE_PERSISTENCE, self._phase_state_persistence),
            (ShutdownPhase.FINAL_VALIDATION, self._phase_final_validation),
        ]
        
        for phase_name, phase_handler in phases:
            await self._execute_phase(phase_name, phase_handler)
        
        self.current_phase = ShutdownPhase.SHUTDOWN_COMPLETE
        self.shutdown_state["phase"] = ShutdownPhase.SHUTDOWN_COMPLETE
        self.shutdown_state["completed_phases"].append(ShutdownPhase.SHUTDOWN_COMPLETE)
    
    async def _execute_phase(self, phase_name: ShutdownPhase, phase_handler: Callable):
        """
        Execute a single shutdown phase with error handling.
        
        Args:
            phase_name: Name of the shutdown phase
            phase_handler: Async function to execute the phase
        """
        self.current_phase = phase_name
        self.shutdown_state["phase"] = phase_name.value
        
        logger.info(f"🔄 Executing shutdown phase: {phase_name.value}")
        
        try:
            phase_start = datetime.utcnow()
            await phase_handler()
            
            phase_duration = (datetime.utcnow() - phase_start).total_seconds()
            logger.info(f"✅ Phase '{phase_name.value}' completed in {phase_duration:.2f}s")
            
            self.shutdown_state["completed_phases"].append(phase_name.value)
            
        except Exception as e:
            logger.error(f"❌ Error in shutdown phase '{phase_name.value}': {e}", exc_info=True)
            self.shutdown_state["errors"].append(f"{phase_name.value}: {str(e)}")
            # Continue with next phase despite errors
    
    async def _phase_pre_shutdown_validation(self):
        """Phase 0: Pre-shutdown validation and health checks."""
        if not getattr(Config, 'SHUTDOWN_HEALTH_CHECK_ENABLED', True):
            logger.info("Pre-shutdown health checks disabled")
            return
        
        logger.info("Performing pre-shutdown validation...")
        
        # Component health checks
        for registration in self.registered_components:
            try:
                component = registration.component
                if hasattr(component, 'get_health_status'):
                    health = component.get_health_status()
                    logger.info(f"Component {type(component).__name__} health: {health}")
            except Exception as e:
                logger.warning(f"Health check failed for {type(component).__name__}: {e}")
    
    async def _phase_signal_received(self):
        """Phase 1: Signal reception and initial setup."""
        if self.agi_system:
            # Set the shutdown event to signal other components
            self.agi_system._shutdown.set()
            logger.info("Shutdown signal sent to AGI system")
    
    async def _phase_component_notification(self):
        """Phase 2: Notify all registered components of shutdown."""
        logger.info(f"Notifying {len(self.registered_components)} registered components of shutdown...")
        
        # Prepare all components for shutdown
        for registration in self.registered_components:
            try:
                component = registration.component
                if hasattr(component, 'prepare_shutdown'):
                    if registration.is_async:
                        await asyncio.wait_for(
                            component.prepare_shutdown(),
                            timeout=getattr(Config, 'COMPONENT_PREPARE_TIMEOUT', 10.0)
                        )
                    else:
                        component.prepare_shutdown()
                    logger.debug(f"Component {type(component).__name__} prepared for shutdown")
                else:
                    logger.debug(f"Component {type(component).__name__} has no prepare_shutdown method")
            except asyncio.TimeoutError:
                logger.warning(f"Component {type(component).__name__} prepare_shutdown timed out")
            except Exception as e:
                logger.error(f"Error preparing component {type(component).__name__} for shutdown: {e}")
    
    async def _phase_stop_background_tasks(self):
        """Phase 3: Stop all background tasks."""
        if not self.agi_system or not self.agi_system.background_tasks:
            logger.info("No background tasks to stop")
            return
        
        logger.info(f"Cancelling {len(self.agi_system.background_tasks)} background tasks...")
        
        # Cancel all background tasks
        tasks_to_cancel = []
        for task in self.agi_system.background_tasks:
            if not task.done():
                task.cancel()
                tasks_to_cancel.append(task)
        
        # Wait for tasks to complete with timeout
        if tasks_to_cancel:
            try:
                # Use asyncio.gather with return_exceptions=True to avoid loop issues
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                    timeout=Config.SHUTDOWN_TIMEOUT // 2
                )
                logger.info("Background tasks cancelled successfully")
            except asyncio.TimeoutError:
                logger.warning("Timeout while waiting for background tasks to cancel")
                # Force close any remaining tasks
                for task in tasks_to_cancel:
                    if not task.done():
                        logger.warning(f"Force closing task: {task}")
                        task.cancel()
    
    async def _phase_resource_cleanup(self):
        """Phase 4: Clean up system resources."""
        logger.info("Cleaning up system resources...")
        
        # Execute registered cleanup handlers
        for handler in self.cleanup_handlers:
            try:
                handler()
                logger.debug(f"Executed sync cleanup handler: {handler.__name__}")
            except Exception as e:
                logger.error(f"Error in sync cleanup handler {handler.__name__}: {e}")
        
        for handler in self.async_cleanup_handlers:
            try:
                await asyncio.wait_for(handler(), timeout=Config.RESOURCE_CLEANUP_TIMEOUT)
                logger.debug(f"Executed async cleanup handler: {handler.__name__}")
            except asyncio.TimeoutError:
                logger.warning(f"Async cleanup handler {handler.__name__} exceeded timeout")
            except Exception as e:
                logger.error(f"Error in async cleanup handler {handler.__name__}: {e}")
        
        # Clean up AGI system database session
        if self.agi_system and hasattr(self.agi_system, 'session'):
            try:
                self.agi_system.session.close()
                logger.info("Database session closed")
            except Exception as e:
                logger.error(f"Error closing database session: {e}")
        
        # Clean up temporary files if enabled
        if Config.TEMP_FILE_CLEANUP_ENABLED:
            await self._cleanup_temp_files()
    
    async def _phase_service_shutdown(self):
        """Phase 5: Shutdown external services."""
        logger.info("Shutting down external services...")
        
        # Shutdown all registered components
        component_shutdown_timeout = getattr(Config, 'COMPONENT_SHUTDOWN_TIMEOUT', 15.0)
        
        for registration in self.registered_components:
            try:
                component = registration.component
                if hasattr(component, 'shutdown'):
                    logger.info(f"Shutting down component: {type(component).__name__}")
                    if registration.is_async:
                        success = await asyncio.wait_for(
                            component.shutdown(),
                            timeout=component_shutdown_timeout
                        )
                    else:
                        success = component.shutdown()
                    
                    if success:
                        logger.info(f"Component {type(component).__name__} shutdown successful")
                    else:
                        logger.warning(f"Component {type(component).__name__} shutdown reported failure")
                        
                    # Collect metrics
                    if hasattr(component, 'get_shutdown_metrics'):
                        metrics = component.get_shutdown_metrics()
                        self.shutdown_state["component_metrics"][type(component).__name__] = metrics
                else:
                    logger.debug(f"Component {type(component).__name__} has no shutdown method")
            except asyncio.TimeoutError:
                logger.warning(f"Component {type(component).__name__} shutdown timed out")
            except Exception as e:
                logger.error(f"Error shutting down component {type(component).__name__}: {e}")
    
    async def _phase_state_persistence(self):
        """Phase 6: Persist system state for recovery."""
        if not Config.STATE_PERSISTENCE_ENABLED:
            logger.info("State persistence disabled")
            return
        
        logger.info("Persisting system state...")
        
        try:
            state_data = await self._collect_system_state()
            
            # Add enhanced state validation
            if getattr(Config, 'SHUTDOWN_STATE_VALIDATION_ENABLED', True):
                is_valid = self._validate_state_data(state_data)
                if not is_valid:
                    logger.warning("State validation failed, saving anyway for recovery")
            
            state_file = Path(Config.SHUTDOWN_STATE_FILE)
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            logger.info(f"System state saved to {state_file}")
            
            # Create backup if enabled
            if getattr(Config, 'SHUTDOWN_BACKUP_ENABLED', True):
                await self._create_state_backup(state_data)
            
            # Also save action cache if enabled
            if Config.ACTION_CACHE_PERSIST and hasattr(self.agi_system, 'action_manager'):
                await self._save_action_cache()
            
        except Exception as e:
            logger.error(f"Error persisting system state: {e}")
    
    async def _phase_final_validation(self):
        """Phase 7: Final validation and cleanup."""
        logger.info("Performing final validation...")
        
        # Validate state file integrity if enabled
        if getattr(Config, 'SHUTDOWN_VALIDATION_ENABLED', True):
            try:
                state_file = Path(Config.SHUTDOWN_STATE_FILE)
                if state_file.exists():
                    with open(state_file, 'r', encoding='utf-8') as f:
                        json.load(f)  # Try to parse JSON
                    logger.info("State file integrity validation passed")
            except Exception as e:
                logger.warning(f"State file integrity validation failed: {e}")
        
        # Ensure ChromaDB persistence if enabled
        if Config.CHROMADB_PERSIST_ON_SHUTDOWN:
            try:
                await self._persist_chromadb()
            except Exception as e:
                logger.error(f"Error persisting ChromaDB: {e}")
        
        # Log shutdown statistics
        elapsed = (datetime.utcnow() - self.shutdown_start_time).total_seconds()
        logger.info(f"Total shutdown time: {elapsed:.2f}s")
    
    async def _persist_chromadb(self):
        """Persist ChromaDB data during shutdown."""
        logger.info("Persisting ChromaDB data...")
        
        try:
            # Try to access ChromaDB client through memory service
            if (self.agi_system and 
                hasattr(self.agi_system, 'memory_service') and 
                hasattr(self.agi_system.memory_service, 'client')):
                
                client = self.agi_system.memory_service.client
                if hasattr(client, 'persist'):
                    # Call ChromaDB persist method
                    client.persist()
                    logger.info("ChromaDB persistence completed successfully")
                else:
                    logger.info("ChromaDB client has no persist method")
            else:
                logger.info("ChromaDB client not available for persistence")
                
        except Exception as e:
            logger.error(f"Error during ChromaDB persistence: {e}")

    def _validate_state_data(self, state_data: Dict[str, Any]) -> bool:
        """
        Validate state data before persistence.
        
        Args:
            state_data: State data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = ['shutdown_info', 'timestamp', 'version']
            for field in required_fields:
                if field not in state_data:
                    logger.warning(f"Missing required field in state data: {field}")
                    return False
            
            # Validate timestamp
            timestamp = state_data.get('timestamp')
            if not timestamp or not isinstance(timestamp, str):
                logger.warning("Invalid timestamp in state data")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating state data: {e}")
            return False
    
    async def _create_state_backup(self, state_data: Dict[str, Any]):
        """
        Create a backup of the state data.
        
        Args:
            state_data: State data to backup
        """
        try:
            backup_dir = Path("backups")
            backup_dir.mkdir(exist_ok=True)
            
            # Create timestamped backup file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"shutdown_state_{timestamp}.json"
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            logger.info(f"State backup created: {backup_file}")
            
            # Clean up old backups
            await self._cleanup_old_backups(backup_dir)
            
        except Exception as e:
            logger.error(f"Error creating state backup: {e}")
    
    async def _cleanup_old_backups(self, backup_dir: Path):
        """
        Clean up old backup files.
        
        Args:
            backup_dir: Directory containing backup files
        """
        try:
            backup_count = getattr(Config, 'SHUTDOWN_BACKUP_COUNT', 5)
            backup_files = list(backup_dir.glob("shutdown_state_*.json"))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old backups
            for old_backup in backup_files[backup_count:]:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")

    async def _collect_system_state(self) -> Dict[str, Any]:
        """Collect system state for persistence."""
        state_data = {
            "shutdown_info": self.shutdown_state,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.1"  # Updated version
        }
        
        if not self.agi_system:
            return state_data
        
        try:
            # Collect AGI system state
            agi_state = {}
            
            # Current mood
            if hasattr(self.agi_system, 'emotional_intelligence'):
                agi_state["mood"] = self.agi_system.emotional_intelligence.get_mood_vector()
            
            # Current plans
            if hasattr(self.agi_system, 'current_plan'):
                agi_state["current_plan"] = self.agi_system.current_plan
                agi_state["current_task_prompt"] = getattr(self.agi_system, 'current_task_prompt', None)
            
            # Shared state
            if hasattr(self.agi_system, 'shared_state'):
                shared_state = self.agi_system.shared_state
                agi_state["shared_state"] = {
                    "mood": getattr(shared_state, 'mood', {}),
                    "current_situation_id": getattr(shared_state, 'current_situation_id', None),
                    "current_task": getattr(shared_state, 'current_task', None)
                }
            
            # Research progress
            if hasattr(self.agi_system, 'research_in_progress'):
                agi_state["research_in_progress"] = list(self.agi_system.research_in_progress.keys())
            
            # Invention history
            if hasattr(self.agi_system, 'invention_history'):
                agi_state["invention_history"] = self.agi_system.invention_history[-10:]  # Last 10
            
            # Snake Agent state
            if hasattr(self.agi_system, 'snake_agent') and self.agi_system.snake_agent:
                try:
                    snake_state = self.agi_system.snake_agent.state.to_dict()
                    agi_state["snake_agent"] = {
                        "state": snake_state,
                        "running": self.agi_system.snake_agent.running,
                        "analysis_count": getattr(self.agi_system.snake_agent, 'analysis_count', 0),
                        "experiment_count": getattr(self.agi_system.snake_agent, 'experiment_count', 0),
                        "communication_count": getattr(self.agi_system.snake_agent, 'communication_count', 0)
                    }
                    logger.info("Snake Agent state collected for persistence")
                except Exception as e:
                    logger.error(f"Error collecting Snake Agent state: {e}")
                    agi_state["snake_agent"] = {"error": str(e)}
            
            state_data["agi_system"] = agi_state
            
        except Exception as e:
            logger.error(f"Error collecting AGI system state: {e}")
            state_data["agi_system"] = {"error": str(e)}
        
        return state_data
    
    async def _save_action_cache(self):
        """Save action manager cache."""
        try:
            action_manager = self.agi_system.action_manager
            if hasattr(action_manager, 'get_cache_data'):
                cache_data = action_manager.get_cache_data()
                
                cache_file = Path("action_cache.pkl")
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                
                logger.info(f"Action cache saved to {cache_file}")
                
        except Exception as e:
            logger.error(f"Error saving action cache: {e}")
    
    async def _cleanup_temp_files(self):
        """Clean up temporary files used by the system."""
        try:
            temp_dirs = [
                Path(tempfile.gettempdir()) / "ravana_audio",
                Path(tempfile.gettempdir()) / "ravana_images",
                Path(tempfile.gettempdir()) / "ravana_temp"
            ]
            
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    for file_path in temp_dir.iterdir():
                        try:
                            if file_path.is_file():
                                file_path.unlink()
                            elif file_path.is_dir():
                                # Remove empty directories
                                try:
                                    file_path.rmdir()
                                except OSError:
                                    pass  # Directory not empty
                        except Exception as e:
                            logger.warning(f"Could not remove temp file {file_path}: {e}")
                    
                    # Try to remove the directory itself if empty
                    try:
                        temp_dir.rmdir()
                        logger.info(f"Cleaned up temp directory: {temp_dir}")
                    except OSError:
                        pass  # Directory not empty or doesn't exist
            
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
    
    async def _force_shutdown(self):
        """Force immediate shutdown when graceful shutdown fails."""
        logger.warning("🚨 Forcing immediate shutdown...")
        
        # Cancel any remaining tasks forcefully
        if self.agi_system and self.agi_system.background_tasks:
            for task in self.agi_system.background_tasks:
                if not task.done():
                    task.cancel()
        
        # Close database connection forcefully
        if self.agi_system and hasattr(self.agi_system, 'session'):
            try:
                self.agi_system.session.close()
            except Exception:
                pass  # Ignore errors during force shutdown
        
        logger.warning("Force shutdown completed")
    
    def _log_shutdown_summary(self):
        """Log a summary of the shutdown process."""
        elapsed = (datetime.utcnow() - self.shutdown_start_time).total_seconds()
        
        logger.info("=" * 60)
        logger.info("🛑 SHUTDOWN SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Duration: {elapsed:.2f}s")
        logger.info(f"Completed Phases: {len(self.shutdown_state['completed_phases'])}")
        logger.info(f"Errors: {len(self.shutdown_state['errors'])}")
        
        if self.shutdown_state['errors']:
            logger.warning("Errors during shutdown:")
            for error in self.shutdown_state['errors']:
                logger.warning(f"  - {error}")
        
        logger.info("=" * 60)
        
        # Save shutdown log
        try:
            log_data = {
                "shutdown_summary": self.shutdown_state,
                "duration_seconds": elapsed,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            log_file = Path("shutdown_log.json")
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Could not save shutdown log: {e}")


async def load_previous_state() -> Optional[Dict[str, Any]]:
    """
    Load previous system state if available.
    
    Returns:
        Previous state data or None if not available
    """
    try:
        state_file = Path(Config.SHUTDOWN_STATE_FILE)
        if not state_file.exists():
            return None
        
        with open(state_file, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        logger.info(f"Loaded previous system state from {state_file}")
        return state_data
        
    except Exception as e:
        logger.error(f"Error loading previous state: {e}")
        return None


def cleanup_state_file():
    """Clean up the state file after successful recovery."""
    try:
        state_file = Path(Config.SHUTDOWN_STATE_FILE)
        if state_file.exists():
            state_file.unlink()
            logger.info("Previous state file cleaned up")
    except Exception as e:
        logger.warning(f"Could not clean up state file: {e}")