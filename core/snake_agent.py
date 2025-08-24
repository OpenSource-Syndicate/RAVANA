"""
Snake Agent Core Module

This module implements the core Snake Agent that autonomously monitors,
analyzes, and experiments with the RAVANA codebase in the background.
"""

import asyncio
import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib

from core.config import Config
from core.snake_llm import create_snake_coding_llm, create_snake_reasoning_llm, SnakeConfigValidator

logger = logging.getLogger(__name__)


@dataclass
class SnakeAgentState:
    """State management for Snake Agent"""
    last_analysis_time: datetime = None
    analyzed_files: Set[str] = None
    pending_experiments: List[Dict[str, Any]] = None
    communication_queue: List[Dict[str, Any]] = None
    learning_history: List[Dict[str, Any]] = None
    current_task: Optional[str] = None
    mood: str = "curious"
    experiment_success_rate: float = 0.0
    
    def __post_init__(self):
        if self.analyzed_files is None:
            self.analyzed_files = set()
        if self.pending_experiments is None:
            self.pending_experiments = []
        if self.communication_queue is None:
            self.communication_queue = []
        if self.learning_history is None:
            self.learning_history = []
        if self.last_analysis_time is None:
            self.last_analysis_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for persistence"""
        return {
            "last_analysis_time": self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            "analyzed_files": list(self.analyzed_files),
            "pending_experiments": self.pending_experiments,
            "communication_queue": self.communication_queue,
            "learning_history": self.learning_history,
            "current_task": self.current_task,
            "mood": self.mood,
            "experiment_success_rate": self.experiment_success_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SnakeAgentState':
        """Create state from dictionary"""
        state = cls()
        if data.get("last_analysis_time"):
            state.last_analysis_time = datetime.fromisoformat(data["last_analysis_time"])
        state.analyzed_files = set(data.get("analyzed_files", []))
        state.pending_experiments = data.get("pending_experiments", [])
        state.communication_queue = data.get("communication_queue", [])
        state.learning_history = data.get("learning_history", [])
        state.current_task = data.get("current_task")
        state.mood = data.get("mood", "curious")
        state.experiment_success_rate = data.get("experiment_success_rate", 0.0)
        return state


class FileSystemMonitor:
    """Monitors RAVANA codebase for changes"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.monitored_extensions = {'.py', '.json', '.md', '.txt', '.yml', '.yaml'}
        self.excluded_dirs = {'__pycache__', '.git', '.venv', 'node_modules', '.qoder'}
        self.file_hashes: Dict[str, str] = {}
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Cannot hash file {file_path}: {e}")
            return ""
    
    def scan_for_changes(self) -> List[Dict[str, Any]]:
        """Scan for file changes since last check"""
        changes = []
        
        for file_path in self._get_monitored_files():
            try:
                current_hash = self.get_file_hash(file_path)
                file_key = str(file_path.relative_to(self.root_path))
                
                if file_key not in self.file_hashes:
                    # New file
                    changes.append({
                        "type": "new",
                        "path": file_key,
                        "absolute_path": str(file_path),
                        "hash": current_hash,
                        "timestamp": datetime.now()
                    })
                elif self.file_hashes[file_key] != current_hash:
                    # Modified file
                    changes.append({
                        "type": "modified",
                        "path": file_key,
                        "absolute_path": str(file_path),
                        "old_hash": self.file_hashes[file_key],
                        "new_hash": current_hash,
                        "timestamp": datetime.now()
                    })
                
                self.file_hashes[file_key] = current_hash
                
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
        
        return changes
    
    def _get_monitored_files(self) -> List[Path]:
        """Get list of files to monitor"""
        files = []
        
        for file_path in self.root_path.rglob("*"):
            if (file_path.is_file() and 
                file_path.suffix in self.monitored_extensions and
                not any(excluded in file_path.parts for excluded in self.excluded_dirs)):
                files.append(file_path)
        
        return files


class SnakeAgent:
    """Main Snake Agent class for autonomous code analysis and improvement"""
    
    def __init__(self, agi_system):
        self.agi_system = agi_system
        self.config = Config()
        self.state = SnakeAgentState()
        
        # Initialize components (will be set during startup)
        self.coding_llm = None
        self.reasoning_llm = None
        self.file_monitor = None
        self.code_analyzer = None
        self.safe_experimenter = None
        self.ravana_communicator = None
        
        # Control flags
        self.running = False
        self._shutdown_event = asyncio.Event()
        self._task_lock = asyncio.Lock()
        
        # Performance tracking
        self.analysis_count = 0
        self.experiment_count = 0
        self.communication_count = 0
        
        # State persistence
        self.state_file = Path("snake_agent_state.json")
    
    async def initialize(self) -> bool:
        """Initialize Snake Agent components"""
        try:
            logger.info("Initializing Snake Agent...")
            
            # Validate configuration
            startup_report = SnakeConfigValidator.get_startup_report()
            if not startup_report["config_valid"]:
                logger.error(f"Snake Agent configuration invalid: {startup_report}")
                return False
            
            logger.info(f"Ollama connection: {startup_report['ollama_connected']}")
            logger.info(f"Available models: {startup_report['available_models']}")
            
            # Initialize LLM interfaces
            self.coding_llm = await create_snake_coding_llm()
            self.reasoning_llm = await create_snake_reasoning_llm()
            
            # Initialize file system monitor
            workspace_path = getattr(self.agi_system, 'workspace_path', os.getcwd())
            self.file_monitor = FileSystemMonitor(workspace_path)
            
            # Load previous state
            await self._load_state()
            
            logger.info("Snake Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Snake Agent: {e}")
            return False
    
    async def start_autonomous_operation(self):
        """Start the main autonomous operation loop"""
        if not await self.initialize():
            logger.error("Cannot start Snake Agent - initialization failed")
            return
        
        self.running = True
        logger.info("Starting Snake Agent autonomous operation")
        
        try:
            while self.running and not self._shutdown_event.is_set():
                async with self._task_lock:
                    await self._execute_analysis_cycle()
                
                # Wait for next cycle
                await asyncio.sleep(self.config.SNAKE_AGENT_INTERVAL)
                
        except asyncio.CancelledError:
            logger.info("Snake Agent operation cancelled")
        except Exception as e:
            logger.error(f"Error in Snake Agent autonomous operation: {e}")
        finally:
            await self._cleanup()
    
    async def _execute_analysis_cycle(self):
        """Execute one complete analysis cycle"""
        try:
            cycle_start = time.time()
            
            # Validate state integrity before proceeding
            if not self._validate_state():
                logger.warning("State validation failed, reinitializing state")
                self._reinitialize_state()
            
            # Update mood based on recent performance (with error handling)
            try:
                self._update_mood()
            except Exception as e:
                logger.error(f"Error updating mood: {e}")
                # Ensure mood is set to a default value if update fails
                if not hasattr(self.state, 'mood') or not self.state.mood:
                    self.state.mood = "curious"
            
            # 1. Monitor for file system changes
            try:
                changes = self.file_monitor.scan_for_changes()
                if changes:
                    logger.info(f"Detected {len(changes)} file changes")
                    await self._process_file_changes(changes)
            except Exception as e:
                logger.error(f"Error monitoring file changes: {e}")
            
            # 2. Periodic codebase analysis (even without changes)
            try:
                if self._should_perform_periodic_analysis():
                    await self._perform_periodic_analysis()
            except Exception as e:
                logger.error(f"Error in periodic analysis: {e}")
            
            # 3. Process pending experiments
            try:
                if self.state.pending_experiments:
                    await self._process_pending_experiments()
            except Exception as e:
                logger.error(f"Error processing experiments: {e}")
            
            # 4. Handle communication queue
            try:
                if self.state.communication_queue:
                    await self._process_communication_queue()
            except Exception as e:
                logger.error(f"Error processing communications: {e}")
            
            # 5. Save state
            try:
                await self._save_state()
            except Exception as e:
                logger.error(f"Error saving state: {e}")
            
            cycle_time = time.time() - cycle_start
            logger.debug(f"Analysis cycle completed in {cycle_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in analysis cycle: {e}")
            # Attempt to save state even if cycle failed
            try:
                await self._save_state()
            except Exception as save_error:
                logger.error(f"Failed to save state after cycle error: {save_error}")
    
    async def _process_file_changes(self, changes: List[Dict[str, Any]]):
        """Process detected file changes"""
        for change in changes:
            try:
                if change["type"] in ["new", "modified"] and change["path"].endswith('.py'):
                    # Analyze Python files immediately
                    await self._analyze_file(change["absolute_path"], change["type"])
                    
                self.state.analyzed_files.add(change["path"])
                
            except Exception as e:
                logger.error(f"Error processing file change {change['path']}: {e}")
    
    async def _analyze_file(self, file_path: str, change_type: str):
        """Analyze a specific file for improvement opportunities"""
        try:
            # Import analyzer here to avoid circular imports
            from core.snake_code_analyzer import SnakeCodeAnalyzer
            
            if not self.code_analyzer:
                self.code_analyzer = SnakeCodeAnalyzer(self.coding_llm)
            
            logger.info(f"Analyzing file: {file_path} (change: {change_type})")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # Perform analysis
            analysis_result = await self.code_analyzer.analyze_code(
                code_content, file_path, change_type
            )
            
            # Check if improvements are suggested
            if analysis_result.get("improvements_suggested", False):
                # Create experiment proposal
                experiment = {
                    "id": f"exp_{int(time.time())}_{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                    "file_path": file_path,
                    "analysis": analysis_result,
                    "status": "pending",
                    "created_at": datetime.now().isoformat(),
                    "priority": analysis_result.get("priority", "medium")
                }
                
                self.state.pending_experiments.append(experiment)
                logger.info(f"Created experiment proposal: {experiment['id']}")
            
            self.analysis_count += 1
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
    
    async def _process_pending_experiments(self):
        """Process experiments in the queue"""
        if not self.state.pending_experiments:
            return
            
        # Sort experiments by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        self.state.pending_experiments.sort(
            key=lambda x: priority_order.get(x.get("priority", "low"), 2)
        )
        
        # Process highest priority experiment
        experiment = self.state.pending_experiments[0]
        try:
            logger.info(f"Processing experiment: {experiment['id']}")
            
            # Import experimenter here to avoid circular imports
            if not self.safe_experimenter:
                from core.snake_safe_experimenter import SnakeSafeExperimenter
                self.safe_experimenter = SnakeSafeExperimenter(self.coding_llm, self.reasoning_llm)
            
            # Execute experiment
            result = await self.safe_experimenter.execute_experiment(experiment)
            
            # Update success rate
            success = result.get("success", False)
            self._update_experiment_success_rate(success)
            
            # Update experiment status
            experiment["status"] = "completed" if success else "failed"
            experiment["completed_at"] = datetime.now().isoformat()
            experiment["result"] = result
            
            # Add to learning history
            self.state.learning_history.append({
                "experiment_id": experiment["id"],
                "file_path": experiment["file_path"],
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "result_summary": result.get("summary", "No summary")
            })
            
            # Communicate significant results
            if success and result.get("impact_score", 0) > 0.7:
                await self._communicate_result(result)
            
            self.experiment_count += 1
            logger.info(f"Experiment {experiment['id']} completed with success: {success}")
            
        except Exception as e:
            logger.error(f"Error processing experiment {experiment['id']}: {e}")
            experiment["status"] = "error"
            experiment["error"] = str(e)
        finally:
            # Remove processed experiment from queue
            self.state.pending_experiments.pop(0)
    
    async def _process_communication_queue(self):
        """Process communication queue"""
        if not self.state.communication_queue:
            return
            
        # Process oldest communication first
        communication = self.state.communication_queue.pop(0)
        try:
            # Import communicator here to avoid circular imports
            if not self.ravana_communicator:
                from core.snake_ravana_communicator import SnakeRavanaCommunicator
                self.ravana_communicator = SnakeRavanaCommunicator()
            
            await self.ravana_communicator.send_message(
                communication["message"],
                communication["priority"]
            )
            
            self.communication_count += 1
            logger.info(f"Sent communication message with priority {communication['priority']}")
            
        except Exception as e:
            logger.error(f"Error sending communication: {e}")
            # Re-queue failed communications at lower priority
            communication["priority"] = "low"
            communication["retry_count"] = communication.get("retry_count", 0) + 1
            if communication["retry_count"] < 3:  # Max 3 retries
                self.state.communication_queue.append(communication)
    
    async def _communicate_result(self, result: Dict[str, Any]):
        """Communicate significant results to the main system"""
        try:
            priority = self._calculate_communication_priority(result)
            message = {
                "type": "experiment_result",
                "content": result,
                "timestamp": datetime.now().isoformat()
            }
            
            self.state.communication_queue.append({
                "message": message,
                "priority": priority,
                "created_at": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error queuing communication: {e}")
    
    def _should_perform_periodic_analysis(self) -> bool:
        """Determine if periodic analysis should be performed"""
        if not self.state.last_analysis_time:
            return True
        
        time_since_analysis = datetime.now() - self.state.last_analysis_time
        # Perform periodic analysis every hour
        return time_since_analysis > timedelta(hours=1)
    
    async def _perform_periodic_analysis(self):
        """Perform periodic analysis of the codebase"""
        try:
            logger.info("Performing periodic codebase analysis")
            
            # Select files for analysis based on importance and last analysis time
            files_to_analyze = self._select_files_for_periodic_analysis()
            
            for file_path in files_to_analyze[:3]:  # Limit to 3 files per cycle
                await self._analyze_file(file_path, "periodic")
            
            self.state.last_analysis_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in periodic analysis: {e}")
    
    def _select_files_for_periodic_analysis(self) -> List[str]:
        """Select important files for periodic analysis"""
        important_files = []
        
        # Core system files
        core_patterns = ['core/system.py', 'core/llm.py', 'core/action_manager.py']
        
        # Module files
        module_patterns = ['modules/*/main.py', 'modules/*/*.py']
        
        workspace_path = Path(getattr(self.agi_system, 'workspace_path', os.getcwd()))
        
        for pattern in core_patterns:
            file_path = workspace_path / pattern
            if file_path.exists():
                important_files.append(str(file_path))
        
        return important_files
    
    def _update_mood(self):
        """Update agent mood based on recent performance"""
        # Ensure state exists and has experiment_success_rate attribute
        if not hasattr(self.state, 'experiment_success_rate'):
            logger.warning("State missing experiment_success_rate, initializing to 0.0")
            self.state.experiment_success_rate = 0.0
        
        success_rate = self.state.experiment_success_rate
        
        if success_rate > 0.8:
            self.state.mood = "confident"
        elif success_rate > 0.5:
            self.state.mood = "curious"
        elif success_rate > 0.2:
            self.state.mood = "cautious"
        else:
            self.state.mood = "frustrated"
        
        logger.debug(f"Mood updated to '{self.state.mood}' based on success rate: {success_rate:.3f}")
    
    def _update_experiment_success_rate(self, success: bool):
        """Update experiment success rate with exponential moving average"""
        alpha = 0.1  # Learning rate
        current_success = 1.0 if success else 0.0
        self.state.experiment_success_rate = (
            alpha * current_success + (1 - alpha) * self.state.experiment_success_rate
        )
    
    def _validate_state(self) -> bool:
        """Validate that the agent state has all required attributes"""
        try:
            # Check if state exists
            if not hasattr(self, 'state') or self.state is None:
                logger.error("Snake Agent state is None or missing")
                return False
            
            # Check required attributes
            required_attrs = [
                'experiment_success_rate', 'mood', 'last_analysis_time',
                'analyzed_files', 'pending_experiments', 'communication_queue',
                'learning_history'
            ]
            
            for attr in required_attrs:
                if not hasattr(self.state, attr):
                    logger.warning(f"State missing required attribute: {attr}")
                    return False
            
            # Validate data types
            if not isinstance(self.state.experiment_success_rate, (int, float)):
                logger.warning(f"Invalid experiment_success_rate type: {type(self.state.experiment_success_rate)}")
                return False
            
            if not isinstance(self.state.mood, str):
                logger.warning(f"Invalid mood type: {type(self.state.mood)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating state: {e}")
            return False
    
    def _reinitialize_state(self):
        """Reinitialize the agent state with safe defaults"""
        try:
            logger.info("Reinitializing Snake Agent state with safe defaults")
            
            # Create new state instance
            self.state = SnakeAgentState()
            
            # Ensure all required attributes are properly initialized
            if not hasattr(self.state, 'experiment_success_rate') or self.state.experiment_success_rate is None:
                self.state.experiment_success_rate = 0.0
            
            if not hasattr(self.state, 'mood') or not self.state.mood:
                self.state.mood = "curious"
            
            # Initialize collections if they don't exist
            if not hasattr(self.state, 'analyzed_files') or self.state.analyzed_files is None:
                self.state.analyzed_files = set()
            
            if not hasattr(self.state, 'pending_experiments') or self.state.pending_experiments is None:
                self.state.pending_experiments = []
            
            if not hasattr(self.state, 'communication_queue') or self.state.communication_queue is None:
                self.state.communication_queue = []
            
            if not hasattr(self.state, 'learning_history') or self.state.learning_history is None:
                self.state.learning_history = []
            
            logger.info("State reinitialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error reinitializing state: {e}")
            # Last resort - create minimal state
            self.state = SnakeAgentState()
    
    def _calculate_communication_priority(self, result: Dict[str, Any]) -> str:
        """Calculate communication priority based on experiment result"""
        impact_score = result.get("impact_score", 0.5)
        safety_score = result.get("safety_score", 0.5)
        
        combined_score = (impact_score + safety_score) / 2
        
        if combined_score > 0.8:
            return "high"
        elif combined_score > 0.6:
            return "medium"
        else:
            return "low"
    
    async def _save_state(self):
        """Save agent state to disk"""
        try:
            state_data = self.state.to_dict()
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    async def _load_state(self):
        """Load agent state from disk"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                self.state = SnakeAgentState.from_dict(state_data)
                logger.info("Loaded previous Snake Agent state")
        except Exception as e:
            logger.warning(f"Could not load previous state: {e}")
    
    async def stop(self):
        """Stop the Snake Agent gracefully"""
        logger.info("Stopping Snake Agent...")
        self.running = False
        self._shutdown_event.set()
        
        # Save final state
        await self._save_state()
    
    async def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Snake Agent resources...")
        await self._save_state()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "running": self.running,
            "mood": self.state.mood,
            "analysis_count": self.analysis_count,
            "experiment_count": self.experiment_count,
            "communication_count": self.communication_count,
            "pending_experiments": len(self.state.pending_experiments),
            "communication_queue": len(self.state.communication_queue),
            "success_rate": self.state.experiment_success_rate,
            "last_analysis": self.state.last_analysis_time.isoformat() if self.state.last_analysis_time else None
        }
    
    # Shutdownable interface implementation
    async def prepare_shutdown(self) -> bool:
        """
        Prepare Snake Agent for shutdown.
        
        Returns:
            bool: True if preparation was successful
        """
        logger.info("Preparing Snake Agent for shutdown...")
        try:
            # Save current state immediately
            await self._save_state()
            
            # Set shutdown flag to stop autonomous operation
            self.running = False
            self._shutdown_event.set()
            
            logger.info("Snake Agent prepared for shutdown")
            return True
        except Exception as e:
            logger.error(f"Error preparing Snake Agent for shutdown: {e}")
            return False
    
    async def shutdown(self, timeout: float = 30.0) -> bool:
        """
        Shutdown Snake Agent with timeout.
        
        Args:
            timeout: Maximum time to wait for shutdown
            
        Returns:
            bool: True if shutdown was successful
        """
        logger.info(f"Shutting down Snake Agent with timeout {timeout}s...")
        try:
            # Create a task for the shutdown process
            shutdown_task = asyncio.create_task(self._shutdown_process())
            
            # Wait for shutdown with timeout
            await asyncio.wait_for(shutdown_task, timeout=timeout)
            
            logger.info("Snake Agent shutdown completed successfully")
            return True
        except asyncio.TimeoutError:
            logger.warning("Snake Agent shutdown timed out")
            return False
        except Exception as e:
            logger.error(f"Error during Snake Agent shutdown: {e}")
            return False
    
    async def _shutdown_process(self):
        """Internal shutdown process."""
        try:
            # Stop autonomous operation
            self.running = False
            self._shutdown_event.set()
            
            # Save final state
            await self._save_state()
            
            # Cleanup resources
            await self._cleanup()
            
        except Exception as e:
            logger.error(f"Error in Snake Agent shutdown process: {e}")
    
    def get_shutdown_metrics(self) -> Dict[str, Any]:
        """
        Get shutdown-related metrics for the Snake Agent.
        
        Returns:
            Dict containing shutdown metrics
        """
        return {
            "analysis_count": self.analysis_count,
            "experiment_count": self.experiment_count,
            "communication_count": self.communication_count,
            "pending_experiments": len(self.state.pending_experiments),
            "communication_queue": len(self.state.communication_queue),
            "success_rate": self.state.experiment_success_rate
        }