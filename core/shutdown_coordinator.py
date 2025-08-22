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

from core.config import Config

logger = logging.getLogger(__name__)


class ShutdownPhase:
    """Enumeration of shutdown phases."""
    SIGNAL_RECEIVED = "signal_received"
    TASKS_STOPPING = "tasks_stopping"
    MEMORY_SERVICE_CLEANUP = "memory_service_cleanup"
    RESOURCE_CLEANUP = "resource_cleanup"
    STATE_PERSISTENCE = "state_persistence"
    FINAL_CLEANUP = "final_cleanup"
    SHUTDOWN_COMPLETE = "shutdown_complete"


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
        
        # State tracking
        self.shutdown_state = {
            "phase": None,
            "start_time": None,
            "completed_phases": [],
            "errors": []
        }
        
        logger.info("ShutdownCoordinator initialized")
    
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
            (ShutdownPhase.SIGNAL_RECEIVED, self._phase_signal_received),
            (ShutdownPhase.TASKS_STOPPING, self._phase_stop_background_tasks),
            (ShutdownPhase.MEMORY_SERVICE_CLEANUP, self._phase_memory_service_cleanup),
            (ShutdownPhase.RESOURCE_CLEANUP, self._phase_resource_cleanup),
            (ShutdownPhase.STATE_PERSISTENCE, self._phase_state_persistence),
            (ShutdownPhase.FINAL_CLEANUP, self._phase_final_cleanup),
        ]
        
        for phase_name, phase_handler in phases:
            await self._execute_phase(phase_name, phase_handler)
        
        self.current_phase = ShutdownPhase.SHUTDOWN_COMPLETE
        self.shutdown_state["phase"] = ShutdownPhase.SHUTDOWN_COMPLETE
        self.shutdown_state["completed_phases"].append(ShutdownPhase.SHUTDOWN_COMPLETE)
    
    async def _execute_phase(self, phase_name: str, phase_handler: Callable):
        """
        Execute a single shutdown phase with error handling.
        
        Args:
            phase_name: Name of the shutdown phase
            phase_handler: Async function to execute the phase
        """
        self.current_phase = phase_name
        self.shutdown_state["phase"] = phase_name
        
        logger.info(f"🔄 Executing shutdown phase: {phase_name}")
        
        try:
            phase_start = datetime.utcnow()
            await phase_handler()
            
            phase_duration = (datetime.utcnow() - phase_start).total_seconds()
            logger.info(f"✅ Phase '{phase_name}' completed in {phase_duration:.2f}s")
            
            self.shutdown_state["completed_phases"].append(phase_name)
            
        except Exception as e:
            logger.error(f"❌ Error in shutdown phase '{phase_name}': {e}", exc_info=True)
            self.shutdown_state["errors"].append(f"{phase_name}: {str(e)}")
            # Continue with next phase despite errors
    
    async def _phase_signal_received(self):
        """Phase 1: Signal reception and initial setup."""
        if self.agi_system:
            # Set the shutdown event to signal other components
            self.agi_system._shutdown.set()
            logger.info("Shutdown signal sent to AGI system")
    
    async def _phase_stop_background_tasks(self):
        """Phase 2: Stop all background tasks."""
        if not self.agi_system or not self.agi_system.background_tasks:
            logger.info("No background tasks to stop")
            return
        
        logger.info(f"Cancelling {len(self.agi_system.background_tasks)} background tasks...")
        
        # Cancel all background tasks
        for task in self.agi_system.background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self.agi_system.background_tasks, return_exceptions=True),
                timeout=Config.SHUTDOWN_TIMEOUT // 2
            )
            logger.info("All background tasks stopped successfully")
            
        except asyncio.TimeoutError:
            logger.warning("Some background tasks did not stop within timeout")
    
    async def _phase_memory_service_cleanup(self):
        """Phase 3: Clean up episodic memory service."""
        if not self.agi_system or not hasattr(self.agi_system, 'memory_service'):
            logger.info("No memory service to clean up")
            return
        
        logger.info("Cleaning up episodic memory service...")
        
        try:
            # Call memory service cleanup if it has one
            memory_service = self.agi_system.memory_service
            if hasattr(memory_service, 'cleanup'):
                await asyncio.wait_for(
                    memory_service.cleanup(),
                    timeout=Config.MEMORY_SERVICE_SHUTDOWN_TIMEOUT
                )
                logger.info("Memory service cleanup completed")
            else:
                logger.info("Memory service has no cleanup method")
                
        except asyncio.TimeoutError:
            logger.warning("Memory service cleanup exceeded timeout")
        except Exception as e:
            logger.error(f"Error during memory service cleanup: {e}")
    
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
    
    async def _phase_state_persistence(self):
        """Phase 5: Persist system state for recovery."""
        if not Config.STATE_PERSISTENCE_ENABLED:
            logger.info("State persistence disabled")
            return
        
        logger.info("Persisting system state...")
        
        try:
            state_data = await self._collect_system_state()
            
            state_file = Path(Config.SHUTDOWN_STATE_FILE)
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            logger.info(f"System state saved to {state_file}")
            
            # Also save action cache if enabled
            if Config.ACTION_CACHE_PERSIST and hasattr(self.agi_system, 'action_manager'):
                await self._save_action_cache()
            
        except Exception as e:
            logger.error(f"Error persisting system state: {e}")
    
    async def _phase_final_cleanup(self):
        """Phase 6: Final cleanup operations."""
        logger.info("Performing final cleanup...")
        
        # Ensure ChromaDB persistence if enabled
        if Config.CHROMADB_PERSIST_ON_SHUTDOWN:
            try:
                # This would need to be implemented based on ChromaDB client access
                logger.info("ChromaDB persistence requested (placeholder)")
            except Exception as e:
                logger.error(f"Error persisting ChromaDB: {e}")
        
        # Log shutdown statistics
        elapsed = (datetime.utcnow() - self.shutdown_start_time).total_seconds()
        logger.info(f"Total shutdown time: {elapsed:.2f}s")
    
    async def _collect_system_state(self) -> Dict[str, Any]:
        """Collect system state for persistence."""
        state_data = {
            "shutdown_info": self.shutdown_state,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0"
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