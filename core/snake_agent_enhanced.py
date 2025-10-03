"""
Enhanced Snake Agent

This module implements the enhanced Snake Agent that uses threading and multiprocessing
to continuously improve RAVANA through concurrent analysis, experimentation, and improvement.
"""

import asyncio
import ast
import logging
import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

from core.config import Config
from core.snake_llm import create_snake_coding_llm, create_snake_reasoning_llm
from core.snake_data_models import (
    SnakeAgentConfiguration, FileChangeEvent, AnalysisTask, ExperimentTask,
    ImprovementProposal, CommunicationMessage, TaskPriority
)
from core.snake_log_manager import SnakeLogManager
from core.snake_threading_manager import SnakeThreadingManager
from core.snake_process_manager import SnakeProcessManager
from core.snake_file_monitor import ContinuousFileMonitor
from modules.self_improvement.self_goal_manager import GoalPriority

# VLTM imports
from core.vltm_store import VeryLongTermMemoryStore
from core.vltm_memory_integration_manager import MemoryIntegrationManager
from core.vltm_advanced_retrieval import AdvancedRetrievalEngine
from core.vltm_consolidation_engine import MemoryConsolidationEngine
from core.vltm_lifecycle_manager import MemoryLifecycleManager
from core.vltm_storage_backend import StorageBackend
from core.vltm_scheduler import ConsolidationScheduler
from core.vltm_data_models import (
    DEFAULT_VLTM_CONFIG, MemoryType, MemoryRecord, ConsolidationType
)
from services.memory_service import MemoryService
from services.knowledge_service import KnowledgeService

logger = logging.getLogger(__name__)


class EnhancedSnakeAgent:
    """Enhanced Snake Agent with threading and multiprocessing capabilities"""

    def __init__(self, agi_system):
        self.agi_system = agi_system
        self.config = Config()

        # Get the current event loop for threading operations
        self.loop = asyncio.get_event_loop()

        # Enhanced configuration
        self.snake_config = SnakeAgentConfiguration(
            max_threads=int(os.getenv('SNAKE_MAX_THREADS', '8')),
            max_processes=int(os.getenv('SNAKE_MAX_PROCESSES', '4')),
            analysis_threads=int(os.getenv('SNAKE_ANALYSIS_THREADS', '3')),
            file_monitor_interval=float(
                os.getenv('SNAKE_MONITOR_INTERVAL', '2.0')),
            enable_performance_monitoring=os.getenv(
                'SNAKE_PERF_MONITORING', 'true').lower() == 'true'
        )

        # Core components
        self.log_manager: Optional[SnakeLogManager] = None
        self.threading_manager: Optional[SnakeThreadingManager] = None
        self.process_manager: Optional[SnakeProcessManager] = None
        self.file_monitor: Optional[ContinuousFileMonitor] = None

        # LLM interfaces
        self.coding_llm = None
        self.reasoning_llm = None

        # Control and coordination
        self.running = False
        self.initialized = False
        self._shutdown_event = asyncio.Event()
        self._coordination_lock = asyncio.Lock()

        # Performance metrics
        self.start_time: Optional[datetime] = None
        self.improvements_applied = 0
        self.experiments_completed = 0
        self.files_analyzed = 0
        self.communications_sent = 0

        # State persistence
        self.state_file = Path("enhanced_snake_state.json")

        # Very Long-Term Memory components
        self.vltm_store: Optional[VeryLongTermMemoryStore] = None
        self.memory_integration_manager: Optional[MemoryIntegrationManager] = None
        self.consolidation_engine: Optional[MemoryConsolidationEngine] = None
        self.consolidation_scheduler: Optional[ConsolidationScheduler] = None
        self.lifecycle_manager: Optional[MemoryLifecycleManager] = None
        self.storage_backend: Optional[StorageBackend] = None

        # External memory services
        self.memory_service: Optional[MemoryService] = None
        self.knowledge_service: Optional[KnowledgeService] = None

        # VLTM state
        self.vltm_enabled = os.getenv(
            'SNAKE_VLTM_ENABLED', 'true').lower() == 'true'
        self.vltm_storage_dir = Path(
            os.getenv('SNAKE_VLTM_STORAGE_DIR', 'snake_vltm_storage'))
        self.session_id = str(uuid.uuid4())  # Unique session identifier

    @property
    def state(self):
        """Property to access the snake agent's state for compatibility"""
        # Return a state object that has a to_dict method for compatibility
        class SnakeAgentState:
            def __init__(self, agent):
                self.agent = agent
                
            def to_dict(self):
                # Return a dictionary representation of the agent's state
                return {
                    "start_time": self.agent.start_time.isoformat() if self.agent.start_time else None,
                    "improvements_applied": self.agent.improvements_applied,
                    "experiments_completed": self.agent.experiments_completed,
                    "files_analyzed": self.agent.files_analyzed,
                    "communications_sent": self.agent.communications_sent,
                    "running": self.agent.running,
                    "initialized": self.agent.initialized,
                    "vltm_enabled": self.agent.vltm_enabled,
                    "session_id": self.agent.session_id
                }
        
        return SnakeAgentState(self)

    @state.setter
    def state(self, value):
        """Setter for the state property to update agent state from a SnakeAgentState object"""
        # Check if value is our own state object (the one we return in the getter)
        if hasattr(value, 'agent') and hasattr(value.agent, 'start_time'):
            # It's our own state object, copy the values
            self.start_time = value.agent.start_time
            self.improvements_applied = value.agent.improvements_applied
            self.experiments_completed = value.agent.experiments_completed
            self.files_analyzed = value.agent.files_analyzed
            self.communications_sent = value.agent.communications_sent
            self.running = value.agent.running
            self.initialized = value.agent.initialized
            self.vltm_enabled = value.agent.vltm_enabled
            self.session_id = value.agent.session_id
        elif hasattr(value, 'to_dict'):
            # It's another state object with a to_dict method
            state_dict = value.to_dict()
            
            # Check if it's a classic SnakeAgentState (which has different fields than our enhanced state) 
            if all(key in state_dict for key in ['last_analysis_time', 'analyzed_files', 'pending_experiments']):
                # It's a classic SnakeAgentState, convert it to enhanced state format
                self._update_from_classic_snake_agent_state(state_dict)
            else:
                # It's a state dictionary with similar structure to our enhanced state
                self._update_from_state_dict(state_dict)
        else:
            # Assume it's a dictionary or dict-like object
            self._update_from_state_dict(value)

    def _update_from_classic_snake_agent_state(self, state_dict: Dict[str, Any]):
        """Update enhanced agent state from a classic SnakeAgentState dictionary"""
        # Convert from classic SnakeAgentState format to enhanced format
        if "last_analysis_time" in state_dict and state_dict["last_analysis_time"] is not None:
            try:
                # Map last_analysis_time to start_time
                self.start_time = datetime.fromisoformat(state_dict["last_analysis_time"])
            except ValueError:
                logger.warning(f"Could not parse last_analysis_time: {state_dict['last_analysis_time']}")
                self.start_time = None
        else:
            self.start_time = None

        # Map analyzed_files count to files_analyzed
        self.files_analyzed = len(state_dict.get("analyzed_files", []))
        
        # Map pending_experiments count to a proxy value for experiment tracking
        pending_experiments = state_dict.get("pending_experiments", [])
        # We can't map this directly, but we could count completed experiments differently
        # For now, we'll just log that we have pending experiments
        if pending_experiments:
            logger.info(f"Restoring with {len(pending_experiments)} pending experiments")
        
        # For other values not in classic state, keep current values or use defaults
        # We can't map all values directly, so set reasonable defaults
        # Note: actual values will be reloaded from the enhanced state file in _load_state
        self.improvements_applied = 0
        self.experiments_completed = 0
        self.running = False
        self.initialized = False
        self.communications_sent = 0  # Will be counted over time

    def _update_from_state_dict(self, state_dict: Dict[str, Any]):
        """Update agent state from a state dictionary"""
        if "start_time" in state_dict and state_dict["start_time"] is not None:
            try:
                self.start_time = datetime.fromisoformat(state_dict["start_time"])
            except ValueError:
                logger.warning(f"Could not parse start_time: {state_dict['start_time']}")
                self.start_time = None
        else:
            self.start_time = None

        self.improvements_applied = state_dict.get("improvements_applied", 0)
        self.experiments_completed = state_dict.get("experiments_completed", 0)
        self.files_analyzed = state_dict.get("files_analyzed", 0)
        self.communications_sent = state_dict.get("communications_sent", 0)
        self.running = state_dict.get("running", False)
        self.initialized = state_dict.get("initialized", False)
        self.vltm_enabled = state_dict.get("vltm_enabled", True)
        
        # Only update session_id if provided and not empty
        if state_dict.get("session_id"):
            self.session_id = state_dict["session_id"]

    async def initialize(self) -> bool:
        """Initialize all Enhanced Snake Agent components"""
        try:
            logger.info("Initializing Enhanced Snake Agent...")

            # Validate configuration
            config_issues = self.snake_config.validate()
            if config_issues:
                logger.error(f"Configuration issues: {config_issues}")
                return False

            # Initialize log manager first
            self.log_manager = SnakeLogManager("snake_logs")
            self.log_manager.start_log_processor()

            await self.log_manager.log_system_event(
                "enhanced_snake_init_start",
                {"config": self.snake_config.to_dict()},
                worker_id="enhanced_snake"
            )

            # Initialize LLM interfaces with better error handling and fallback
            try:
                self.coding_llm = await create_snake_coding_llm(self.log_manager)
                self.reasoning_llm = await create_snake_reasoning_llm(self.log_manager)
            except Exception as e:
                logger.error(f"Failed to initialize LLM interfaces: {e}")
                await self.log_manager.log_system_event(
                    "enhanced_snake_llm_init_failed",
                    {"error": str(e)},
                    level="error",
                    worker_id="enhanced_snake"
                )
                # Continue initialization without LLMs - they'll be retried later
                self.coding_llm = None
                self.reasoning_llm = None

            # Initialize Very Long-Term Memory if enabled
            if self.vltm_enabled:
                if not await self._initialize_vltm():
                    logger.warning(
                        "Failed to initialize VLTM - continuing without it")
                    self.vltm_enabled = False
                    await self.log_manager.log_system_event(
                        "enhanced_snake_vltm_init_failed",
                        {"warning": "VLTM initialization failed"},
                        level="warning",
                        worker_id="enhanced_snake"
                    )

            # Initialize threading manager with retry logic
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                try:
                    self.threading_manager = SnakeThreadingManager(
                        self.snake_config, self.log_manager)
                    if await self.threading_manager.initialize():
                        break
                    else:
                        raise Exception(
                            "Threading manager initialization returned False")
                except Exception as e:
                    retry_count += 1
                    logger.warning(
                        f"Failed to initialize threading manager (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        # Exponential backoff
                        await asyncio.sleep(2 ** retry_count)
                    else:
                        raise Exception(
                            "Failed to initialize threading manager after retries")

            # Initialize process manager with retry logic
            retry_count = 0
            while retry_count < max_retries:
                try:
                    self.process_manager = SnakeProcessManager(
                        self.snake_config, self.log_manager)
                    if await self.process_manager.initialize():
                        break
                    else:
                        raise Exception(
                            "Process manager initialization returned False")
                except Exception as e:
                    retry_count += 1
                    logger.warning(
                        f"Failed to initialize process manager (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        # Exponential backoff
                        await asyncio.sleep(2 ** retry_count)
                    else:
                        raise Exception(
                            "Failed to initialize process manager after retries")

            # Initialize file monitor with retry logic
            retry_count = 0
            while retry_count < max_retries:
                try:
                    self.file_monitor = ContinuousFileMonitor(
                        self, self.snake_config, self.log_manager)
                    if await self.file_monitor.initialize():
                        break
                    else:
                        raise Exception(
                            "File monitor initialization returned False")
                except Exception as e:
                    retry_count += 1
                    logger.warning(
                        f"Failed to initialize file monitor (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        # Exponential backoff
                        await asyncio.sleep(2 ** retry_count)
                    else:
                        raise Exception(
                            "Failed to initialize file monitor after retries")

            # Set up component callbacks
            await self._setup_component_callbacks()

            # Load previous state
            await self._load_state()

            self.initialized = True

            await self.log_manager.log_system_event(
                "enhanced_snake_init_complete",
                {"initialized": True},
                worker_id="enhanced_snake"
            )

            logger.info("Enhanced Snake Agent initialized successfully")
            return True

        except Exception as e:
            if self.log_manager:
                # Log with full traceback to snake_errors.log
                self.log_manager.log_error_with_traceback(e, "Enhanced Snake Agent initialization failed", 
                    {"component": "enhanced_snake_agent", "phase": "initialization"})
                # Also log to system events
                await self.log_manager.log_system_event(
                    "enhanced_snake_init_failed",
                    {"error": str(e)},
                    level="error",
                    worker_id="enhanced_snake"
                )
            else:
                # Fallback logging if log_manager is not set up
                logger.error(f"Failed to initialize Enhanced Snake Agent: {e}", exc_info=True)
            self.initialized = False
            return False

    async def _setup_component_callbacks(self):
        """Set up callbacks between components"""
        # File monitor callbacks
        self.file_monitor.set_change_callback(self._handle_file_change)

        # Threading manager callbacks
        self.threading_manager.set_callbacks(
            file_change_callback=self._process_file_change,
            analysis_callback=self._process_analysis_task,
            communication_callback=self._process_communication
        )

        # Process manager callbacks
        self.process_manager.set_callbacks(
            experiment_callback=self._handle_experiment_result,
            analysis_callback=self._handle_analysis_result,
            improvement_callback=self._handle_improvement_result
        )

    async def _initialize_vltm(self) -> bool:
        """Initialize Very Long-Term Memory components"""
        try:
            logger.info("Initializing Very Long-Term Memory system...")

            # Create VLTM storage directory
            self.vltm_storage_dir.mkdir(parents=True, exist_ok=True)

            # Initialize external memory services
            self.memory_service = MemoryService()
            self.knowledge_service = KnowledgeService(self.agi_system.engine)

            # Initialize VLTM store
            logger.info("Initializing VLTM store...")
            self.vltm_store = VeryLongTermMemoryStore(
                config=DEFAULT_VLTM_CONFIG,
                base_storage_dir=str(self.vltm_storage_dir)
            )

            if not await self.vltm_store.initialize():
                logger.error("Failed to initialize VLTM store")
                return False

            # Initialize storage backend
            logger.info("Initializing VLTM storage backend...")
            self.storage_backend = StorageBackend(
                config=DEFAULT_VLTM_CONFIG,
                base_storage_dir=str(self.vltm_storage_dir)
            )

            if not await self.storage_backend.initialize():
                logger.error("Failed to initialize VLTM storage backend")
                return False

            # Initialize consolidation engine
            self.consolidation_engine = MemoryConsolidationEngine(
                config=DEFAULT_VLTM_CONFIG,
                storage_backend=self.storage_backend
            )

            # Initialize lifecycle manager
            self.lifecycle_manager = MemoryLifecycleManager(
                config=DEFAULT_VLTM_CONFIG,
                storage_backend=self.storage_backend
            )

            # Initialize retrieval engine
            self.retrieval_engine = AdvancedRetrievalEngine(
                storage_backend=self.storage_backend,
                config=DEFAULT_VLTM_CONFIG
            )

            # Initialize consolidation scheduler
            self.consolidation_scheduler = ConsolidationScheduler(
                config=DEFAULT_VLTM_CONFIG
            )

            # Initialize memory integration manager
            logger.info("Initializing memory integration manager...")
            self.memory_integration_manager = MemoryIntegrationManager(
                vltm_store=self.vltm_store,
                consolidation_engine=self.consolidation_engine,
                retrieval_engine=self.retrieval_engine,
                memory_service=self.memory_service,
                knowledge_service=self.knowledge_service,
                config=DEFAULT_VLTM_CONFIG
            )

            # Set up consolidation components
            self.memory_integration_manager.set_consolidation_components(
                consolidation_engine=self.consolidation_engine,
                lifecycle_manager=self.lifecycle_manager,
                consolidation_scheduler=self.consolidation_scheduler
            )

            # Start memory integration
            logger.info("Starting memory integration...")
            if not await self.memory_integration_manager.initialize():
                logger.error("Failed to start memory integration")
                return False

            logger.info(
                "Very Long-Term Memory system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing VLTM: {e}", exc_info=True)
            return False

    async def start_autonomous_operation(self):
        """Start the enhanced autonomous operation with threading and multiprocessing"""
        if not self.initialized or self.log_manager is None:
            if not await self.initialize():
                logger.error(
                    "Cannot start Enhanced Snake Agent - initialization failed")
                return

        self.running = True
        self.start_time = datetime.now()

        try:
            logger.info("Starting Enhanced Snake Agent autonomous operation")

            await self.log_manager.log_system_event(
                "autonomous_operation_start",
                {"start_time": self.start_time.isoformat()},
                worker_id="enhanced_snake"
            )

            # Start all threading components
            if not await self.threading_manager.start_all_threads():
                raise Exception("Failed to start threading components")

            # Start all process components
            if not await self.process_manager.start_all_processes():
                raise Exception("Failed to start process components")

            # Start file monitoring
            if not await self.file_monitor.start_monitoring():
                raise Exception("Failed to start file monitoring")

            # Start coordination loop
            await self._coordination_loop()

        except asyncio.CancelledError:
            logger.info("Enhanced Snake Agent operation cancelled")
        except Exception as e:
            if self.log_manager:
                # Log with full traceback to snake_errors.log
                self.log_manager.log_error_with_traceback(e, "Error in Enhanced Snake Agent operation", 
                    {"component": "enhanced_snake_agent", "phase": "operation"})
                # Also log to system events
                await self.log_manager.log_system_event(
                    "autonomous_operation_error",
                    {"error": str(e)},
                    level="error",
                    worker_id="enhanced_snake"
                )
            else:
                # Fallback logging if log_manager is not set up
                logger.error(f"Error in Enhanced Snake Agent operation: {e}", exc_info=True)
        finally:
            await self._cleanup()

    async def _coordination_loop(self):
        """Main coordination loop for the enhanced agent"""
        coordination_interval = 10.0  # 10 seconds
        last_health_check = datetime.now()
        last_metrics_log = datetime.now()
        last_self_evaluation = datetime.now()
        last_goal_setting = datetime.now()

        while self.running and not self._shutdown_event.is_set():
            try:
                async with self._coordination_lock:
                    current_time = datetime.now()

                    # Periodic health checks
                    if current_time - last_health_check >= timedelta(minutes=5):
                        await self._perform_health_check()
                        last_health_check = current_time

                    # Periodic metrics logging
                    if current_time - last_metrics_log >= timedelta(minutes=10):
                        await self._log_performance_metrics()
                        last_metrics_log = current_time

                    # Periodic self-evaluation
                    if current_time - last_self_evaluation >= timedelta(hours=1):
                        await self._perform_self_evaluation()
                        last_self_evaluation = current_time

                    # Periodic goal setting and adjustment
                    if current_time - last_goal_setting >= timedelta(hours=2):
                        await self._set_improvement_goals()
                        last_goal_setting = current_time

                    # State persistence
                    await self._save_state()

                # Wait for next coordination cycle
                await asyncio.sleep(coordination_interval)

            except Exception as e:
                if self.log_manager:
                    # Log with full traceback to snake_errors.log
                    self.log_manager.log_error_with_traceback(e, "Error in coordination loop", 
                        {"component": "enhanced_snake_agent", "phase": "coordination_loop"})
                    # Also log to system events
                    await self.log_manager.log_system_event(
                        "coordination_loop_error",
                        {"error": str(e)},
                        level="error",
                        worker_id="enhanced_snake"
                    )
                else:
                    # Fallback logging if log_manager is not set up
                    logger.error(f"Error in coordination loop: {e}", exc_info=True)
                await asyncio.sleep(coordination_interval)

    async def _perform_self_evaluation(self):
        """Perform self-evaluation to assess performance and identify improvement areas"""
        try:
            logger.info("Performing self-evaluation...")
            
            # Calculate performance metrics
            metrics = await self._calculate_self_evaluation_metrics()
            
            # Store self-evaluation in VLTM
            if self.vltm_enabled and self.vltm_store:
                await self._store_self_evaluation(metrics)
            
            # Log the self-evaluation
            await self.log_manager.log_system_event(
                "self_evaluation_performed",
                {
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat()
                },
                worker_id="enhanced_snake"
            )
            
            logger.info(f"Self-evaluation completed. Performance score: {metrics.get('performance_score', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Error performing self-evaluation: {e}")
            await self.log_manager.log_system_event(
                "self_evaluation_error",
                {"error": str(e)},
                level="error",
                worker_id="enhanced_snake"
            )

    async def _calculate_self_evaluation_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for self-evaluation"""
        if not self.start_time:
            return {}
            
        uptime = datetime.now() - self.start_time
        
        # Calculate various metrics
        metrics = {
            "uptime_seconds": uptime.total_seconds(),
            "total_improvements_applied": self.improvements_applied,
            "total_experiments_completed": self.experiments_completed,
            "total_files_analyzed": self.files_analyzed,
            "improvements_per_hour": self.improvements_applied / max(uptime.total_seconds() / 3600, 1),
            "experiments_per_hour": self.experiments_completed / max(uptime.total_seconds() / 3600, 1),
            "files_analyzed_per_hour": self.files_analyzed / max(uptime.total_seconds() / 3600, 1),
            "improvement_success_rate": self.improvements_applied / max(self.experiments_completed, 1),
            "recent_activities": []  # Will be populated with recent activity patterns
        }
        
        # Calculate performance score (0-1 scale)
        # Weighted combination of various factors
        improvement_factor = min(1.0, metrics["improvements_per_hour"] * 2)  # Max 0.5 improvements per hour = full score
        experiment_factor = min(1.0, metrics["experiments_per_hour"] * 1)   # Max 1 experiment per hour = full score
        analysis_factor = min(1.0, metrics["files_analyzed_per_hour"] * 0.1)  # Max 10 files per hour = full score
        success_factor = metrics["improvement_success_rate"]
        
        # Weighted performance score
        metrics["performance_score"] = (
            improvement_factor * 0.4 + 
            experiment_factor * 0.3 + 
            analysis_factor * 0.2 + 
            success_factor * 0.1
        )
        
        # Identify patterns and trends
        metrics["trends"] = await self._analyze_trends(metrics)
        
        return metrics

    async def _analyze_trends(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in performance over time"""
        try:
            # For now, we'll just return a basic analysis
            # In a full implementation, this would compare to historical data
            trends = {
                "improvement_trend": "increasing" if current_metrics.get("improvements_per_hour", 0) > 0.1 else "stable",
                "experiment_trend": "increasing" if current_metrics.get("experiments_per_hour", 0) > 0.5 else "stable",
                "analysis_trend": "increasing" if current_metrics.get("files_analyzed_per_hour", 0) > 1.0 else "stable",
                "suggestions": []
            }
            
            # Add suggestions based on trends
            if current_metrics.get("improvements_per_hour", 0) < 0.1:
                trends["suggestions"].append("Focus on generating more practical improvements")
            if current_metrics.get("experiments_per_hour", 0) < 0.5:
                trends["suggestions"].append("Increase experimentation to validate more hypotheses")
            if current_metrics.get("files_analyzed_per_hour", 0) < 1.0:
                trends["suggestions"].append("Improve code analysis efficiency")
            
            return trends
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {"error": str(e)}

    async def _store_self_evaluation(self, metrics: Dict[str, Any]):
        """Store self-evaluation in Very Long-Term Memory"""
        try:
            if not self.vltm_store:
                return

            memory_content = {
                "event_type": "self_evaluation",
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "agent_context": {
                    "total_experiments": self.experiments_completed,
                    "total_improvements": self.improvements_applied,
                    "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0
                }
            }

            metadata = {
                "source": "enhanced_snake_agent",
                "category": "self_assessment",
                "performance_score": metrics.get("performance_score"),
                "evaluation_type": "comprehensive"
            }

            memory_id = await self.vltm_store.store_memory(
                content=memory_content,
                memory_type="strategic_knowledge",
                metadata=metadata,
                source_session=self.session_id
            )

            if memory_id:
                logger.debug(f"Stored self-evaluation memory: {memory_id}")

        except Exception as e:
            logger.error(f"Error storing self-evaluation memory: {e}")

    async def _set_improvement_goals(self):
        """Set or adjust improvement goals based on self-evaluation"""
        try:
            logger.info("Setting improvement goals...")
            
            # Calculate current state metrics
            current_metrics = await self._calculate_self_evaluation_metrics()
            
            # Define goals based on current state and trends
            goals = await self._derive_improvement_goals(current_metrics)
            
            # Store goals in state for tracking
            self.current_goals = goals
            
            # Sync goals with the main system goal manager
            await self._sync_goals_with_system(goals)
            
            # Log the goals
            await self.log_manager.log_system_event(
                "improvement_goals_set",
                {
                    "goals": goals,
                    "timestamp": datetime.now().isoformat()
                },
                worker_id="enhanced_snake"
            )
            
            logger.info(f"Set {len(goals)} improvement goals")
            
        except Exception as e:
            logger.error(f"Error setting improvement goals: {e}")
            await self.log_manager.log_system_event(
                "improvement_goals_error",
                {"error": str(e)},
                level="error",
                worker_id="enhanced_snake"
            )

    async def _derive_improvement_goals(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Derive improvement goals based on current metrics and trends"""
        goals = []
        
        # Goal: Increase improvement rate if it's low
        if current_metrics.get("improvements_per_hour", 0) < 0.2:
            goals.append({
                "id": f"goal_{uuid.uuid4().hex[:8]}",
                "type": "improvement_rate",
                "description": "Increase number of improvements applied per hour",
                "target": 0.3,  # Target 0.3 improvements per hour
                "current": current_metrics.get("improvements_per_hour", 0),
                "timeframe_days": 7,
                "priority": "high",
                "strategies": [
                    "Focus on higher-impact improvements",
                    "Streamline improvement validation process",
                    "Improve heuristics for identifying improvement opportunities"
                ]
            })
        
        # Goal: Increase experiment rate if it's low
        if current_metrics.get("experiments_per_hour", 0) < 0.7:
            goals.append({
                "id": f"goal_{uuid.uuid4().hex[:8]}",
                "type": "experiment_rate",
                "description": "Increase number of experiments conducted per hour",
                "target": 1.0,  # Target 1 experiment per hour
                "current": current_metrics.get("experiments_per_hour", 0),
                "timeframe_days": 5,
                "priority": "medium",
                "strategies": [
                    "Automate more experimental setup procedures",
                    "Improve hypothesis generation algorithms",
                    "Parallelize experiment execution where possible"
                ]
            })
        
        # Goal: Improve success rate if it's low
        if current_metrics.get("improvement_success_rate", 0) < 0.5:
            goals.append({
                "id": f"goal_{uuid.uuid4().hex[:8]}",
                "type": "success_rate",
                "description": "Improve success rate of proposed improvements",
                "target": 0.7,  # Target 70% success rate
                "current": current_metrics.get("improvement_success_rate", 0),
                "timeframe_days": 10,
                "priority": "high",
                "strategies": [
                    "Improve pre-implementation validation checks",
                    "Implement better risk assessment for proposed changes",
                    "Learn from past failed improvements"
                ]
            })
        
        # Goal: Expand analysis capabilities
        if current_metrics.get("files_analyzed_per_hour", 0) < 2.0:
            goals.append({
                "id": f"goal_{uuid.uuid4().hex[:8]}",
                "type": "analysis_rate",
                "description": "Increase number of files analyzed per hour",
                "target": 3.0,  # Target 3 files per hour
                "current": current_metrics.get("files_analyzed_per_hour", 0),
                "timeframe_days": 7,
                "priority": "medium",
                "strategies": [
                    "Optimize file analysis algorithms",
                    "Implement more efficient change detection",
                    "Focus analysis on higher-value files"
                ]
            })
        
        # Add any strategic goals based on system needs
        strategic_goals = await self._identify_strategic_goals()
        goals.extend(strategic_goals)
        
        return goals

    async def _identify_strategic_goals(self) -> List[Dict[str, Any]]:
        """Identify strategic improvement goals based on system needs"""
        try:
            # This would involve deeper analysis of the system to identify strategic needs
            # For now, we'll return some common strategic goals
            
            strategic_goals = []
            
            # Goal: Enhance specific capabilities based on system usage
            strategic_goals.append({
                "id": f"strategic_goal_{uuid.uuid4().hex[:8]}",
                "type": "capability_enhancement",
                "description": "Enhance code analysis capabilities for specific improvement areas",
                "target": "implemented",
                "current": "planned",
                "timeframe_days": 14,
                "priority": "medium",
                "strategies": [
                    "Improve detection of performance bottlenecks",
                    "Enhance refactoring suggestion algorithms",
                    "Add domain-specific analysis for RAVANA components"
                ]
            })
            
            # Goal: Improve learning from experiments
            strategic_goals.append({
                "id": f"strategic_goal_{uuid.uuid4().hex[:8]}",
                "type": "learning_enhancement",
                "description": "Improve learning from experiment results to enhance future suggestions",
                "target": "implemented",
                "current": "planned",
                "timeframe_days": 21,
                "priority": "high",
                "strategies": [
                    "Implement better result analysis algorithms",
                    "Create feedback loops from improvement outcomes",
                    "Build knowledge graphs of improvement relationships"
                ]
            })
            
            return strategic_goals
        except Exception as e:
            logger.error(f"Error identifying strategic goals: {e}")
            return []

    def _handle_file_change(self, file_event: FileChangeEvent):
        """Handle file change events from file monitor"""
        try:
            # Convert to analysis task if it's a Python file
            if file_event.file_path.endswith('.py'):
                # Determine if this should be an indexing task (for many threads) or study task (for single thread)
                # This will be processed by indexing threads (many)
                analysis_type = "file_change_indexing"

                analysis_task = AnalysisTask(
                    task_id=f"analysis_{uuid.uuid4().hex[:8]}",
                    file_path=file_event.file_path,
                    analysis_type=analysis_type,
                    priority=TaskPriority.MEDIUM,
                    created_at=datetime.now(),
                    change_context={
                        "event_type": file_event.event_type,
                        "old_hash": file_event.old_hash,
                        "new_hash": file_event.file_hash
                    }
                )

                # Queue for threaded analysis
                self.threading_manager.queue_analysis_task(analysis_task)

            # Log file change
            self.log_manager.log_system_event_sync(
                "file_change_handled",
                {
                    "event_type": file_event.event_type,
                    "file_path": file_event.file_path,
                    "queued_for_analysis": file_event.file_path.endswith('.py')
                },
                worker_id="enhanced_snake"
            )

        except Exception as e:
            self.log_manager.log_system_event_sync(
                "file_change_error",
                {"error": str(e), "event": file_event.to_dict()},
                level="error",
                worker_id="enhanced_snake"
            )

    def _process_file_change(self, file_event: FileChangeEvent):
        """Process file change in threading context"""
        try:
            # Update file analysis count
            self.files_analyzed += 1

            # Log processing
            self.log_manager.log_system_event_sync(
                "file_change_processed",
                {"file_path": file_event.file_path},
                worker_id="file_processor"
            )

        except Exception as e:
            self.log_manager.log_system_event_sync(
                "file_processing_error",
                {"error": str(e)},
                level="error",
                worker_id="file_processor"
            )

    def _process_analysis_task(self, analysis_task: AnalysisTask):
        """Process analysis task in threading context with enhanced self-analysis"""
        try:
            # Perform deeper analysis of the code or file
            improvement_opportunities = self._identify_improvement_opportunities(analysis_task)
            
            # For significant findings, create experiment task
            if analysis_task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL] or improvement_opportunities:
                experiment_task = {
                    "type": "experiment",
                    "task_id": f"exp_{uuid.uuid4().hex[:8]}",
                    "data": {
                        "file_path": analysis_task.file_path,
                        "analysis_type": analysis_task.analysis_type,
                        "priority": analysis_task.priority.value,
                        "improvement_opportunities": improvement_opportunities
                    }
                }

                # Distribute to process manager
                self.process_manager.distribute_task(experiment_task)

            # Log analysis processing
            self.log_manager.log_system_event_sync(
                "analysis_task_processed",
                {
                    "task_id": analysis_task.task_id,
                    "file_path": analysis_task.file_path,
                    "experiment_created": analysis_task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL] or bool(improvement_opportunities),
                    "improvement_opportunities_found": len(improvement_opportunities)
                },
                worker_id="analysis_processor"
            )

        except Exception as e:
            self.log_manager.log_system_event_sync(
                "analysis_processing_error",
                {"error": str(e), "task": analysis_task.to_dict()},
                level="error",
                worker_id="analysis_processor"
            )

    def _identify_improvement_opportunities(self, analysis_task: AnalysisTask) -> List[Dict[str, Any]]:
        """
        Identify potential improvement opportunities in the analyzed code.
        
        Args:
            analysis_task: The analysis task containing file information
            
        Returns:
            List of improvement opportunities found
        """
        opportunities = []
        
        try:
            # Read the file to analyze
            if os.path.exists(analysis_task.file_path):
                with open(analysis_task.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Analyze for common improvement opportunities
                opportunities.extend(self._analyze_code_for_improvements(content, analysis_task.file_path))
            
        except Exception as e:
            logger.warning(f"Could not analyze file for improvements: {e}")
        
        return opportunities

    def _analyze_code_for_improvements(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyze code content for improvement opportunities.
        
        Args:
            content: The content of the file to analyze
            file_path: Path to the file being analyzed
            
        Returns:
            List of identified improvement opportunities
        """
        opportunities = []
        
        # Check for various improvement opportunities
        lines = content.split('\n')
        
        # Look for functions that are too long (potential refactoring opportunity)
        long_functions = self._find_long_functions(lines)
        for func_info in long_functions:
            opportunities.append({
                "type": "refactoring",
                "category": "function_length",
                "description": f"Function '{func_info['name']}' is too long ({func_info['lines']} lines) and should be split",
                "location": f"{file_path}:{func_info['line_num']}",
                "severity": "medium",
                "suggestion": "Break down the function into smaller, more manageable functions"
            })
        
        # Look for duplicated code blocks
        duplicate_blocks = self._find_duplicate_blocks(lines)
        for block_info in duplicate_blocks:
            opportunities.append({
                "type": "refactoring",
                "category": "code_duplication",
                "description": f"Code block duplicated {block_info['count']} times",
                "location": f"{file_path}:{block_info['line_num']}",
                "severity": "high",
                "suggestion": "Extract duplicated code into a reusable function"
            })
        
        # Look for complex functions that could benefit from optimization
        complex_functions = self._find_complex_functions(lines)
        for func_info in complex_functions:
            opportunities.append({
                "type": "optimization",
                "category": "complexity",
                "description": f"Function '{func_info['name']}' has high cyclomatic complexity ({func_info['complexity']})",
                "location": f"{file_path}:{func_info['line_num']}",
                "severity": "medium",
                "suggestion": "Simplify the function logic or split into smaller functions"
            })
        
        # Look for performance improvements
        performance_issues = self._find_performance_issues(lines)
        for issue_info in performance_issues:
            opportunities.append({
                "type": "optimization",
                "category": "performance",
                "description": issue_info['description'],
                "location": f"{file_path}:{issue_info['line_num']}",
                "severity": issue_info['severity'],
                "suggestion": issue_info['suggestion']
            })
        
        # Look for potential bugs
        potential_bugs = self._find_potential_bugs(lines)
        for bug_info in potential_bugs:
            opportunities.append({
                "type": "bug_fix",
                "category": "potential_bug",
                "description": bug_info['description'],
                "location": f"{file_path}:{bug_info['line_num']}",
                "severity": bug_info['severity'],
                "suggestion": bug_info['suggestion']
            })
        
        return opportunities

    def _find_long_functions(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Find functions that are too long."""
        long_functions = []
        current_function = None
        function_start_line = 0
        line_count = 0
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Check if this line starts a function definition
            if stripped_line.startswith('def ') and not stripped_line.startswith('class '):
                # If we were tracking a previous function, check if it was too long
                if current_function and line_count > 50:  # Threshold: 50 lines
                    long_functions.append({
                        "name": current_function,
                        "line_num": function_start_line + 1,
                        "lines": line_count
                    })
                
                # Start tracking the new function
                current_function = stripped_line[4:stripped_line.find('(')]  # Extract function name
                function_start_line = i
                line_count = 1
            elif current_function is not None:
                # Count lines in the current function (ignoring empty lines and comments)
                if stripped_line and not stripped_line.startswith('#'):
                    line_count += 1
                # Check if we've exited the function (when indentation drops to 0 or function definition level)
                elif line_count > 0:
                    # Check if this line has less indentation than the function definition
                    if len(line) - len(line.lstrip()) <= len(lines[function_start_line]) - len(lines[function_start_line].lstrip()):
                        # Check if we're at class or module level again
                        if (stripped_line.startswith('def ') or 
                            stripped_line.startswith('class ') or 
                            stripped_line.startswith('@') or 
                            not stripped_line.startswith(' ')):
                            # This is the end of the function
                            if line_count > 50:
                                long_functions.append({
                                    "name": current_function,
                                    "line_num": function_start_line + 1,
                                    "lines": line_count
                                })
                            current_function = None
        
        # Check the last function
        if current_function and line_count > 50:
            long_functions.append({
                "name": current_function,
                "line_num": function_start_line + 1,
                "lines": line_count
            })
        
        return long_functions

    def _find_duplicate_blocks(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Find duplicated code blocks."""
        import hashlib
        from difflib import SequenceMatcher
        
        blocks = {}
        duplicate_blocks = []
        
        # First, try to find blocks using AST for more accurate detection
        try:
            # Join lines to form the code string
            code_text = '\n'.join(lines)
            
            # Parse the code to get AST
            try:
                tree = ast.parse(code_text)
            except SyntaxError:
                # If parsing fails, fall back to the simpler line-based approach
                raise ValueError("Syntax error in code, using fallback method")
            
            # For line-based duplicate detection with fuzzy matching
            # Group similar lines into potential blocks with similarity check
            for i in range(len(lines) - 5):  # Minimum block size of 5 lines
                # Create a normalized version of the block (remove whitespace, comments, etc.)
                block_text = '\n'.join(lines[i:i+5]).strip()
                # Create a hash of the normalized block as the signature
                block_key = hashlib.md5(block_text.encode()).hexdigest()
                
                if block_key not in blocks:
                    blocks[block_key] = []
                blocks[block_key].append({
                    'start_line': i,
                    'block': lines[i:i+5]
                })
            
            # Find blocks that appear more than once with similarity
            for block_key, occurrences in blocks.items():
                if len(occurrences) > 1:
                    # Calculate similarity between occurrences
                    reference_block = occurrences[0]['block']
                    similar_occurrences = [occurrences[0]]
                    
                    for occurrence in occurrences[1:]:
                        similarity = SequenceMatcher(None, reference_block, occurrence['block']).ratio()
                        if similarity > 0.8:  # 80% similarity threshold
                            similar_occurrences.append(occurrence)
                    
                    if len(similar_occurrences) > 1:
                        duplicate_blocks.append({
                            "count": len(similar_occurrences),
                            "line_num": similar_occurrences[0]['start_line'] + 1,
                            "size": len(similar_occurrences[0]['block']),
                            "similarity": f"{similarity*100:.1f}%"
                        })
        except Exception as e:
            # Fallback to the original implementation if AST approach fails
            logger.warning(f"Using fallback duplicate detection due to: {e}")
            blocks = {}
            duplicate_blocks = []
            
            # Group similar lines into potential blocks
            for i in range(len(lines) - 10):  # Minimum block size of 10 lines
                block_key = tuple(lines[i:i+10])  # Use 10 lines as a signature
                if block_key not in blocks:
                    blocks[block_key] = []
                blocks[block_key].append(i)
            
            # Find blocks that appear more than once
            for block_key, positions in blocks.items():
                if len(positions) > 1:
                    duplicate_blocks.append({
                        "count": len(positions),
                        "line_num": positions[0] + 1,
                        "size": len(block_key)
                    })
        
        return duplicate_blocks

    def _find_complex_functions(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Find functions with high cyclomatic complexity."""
        
        complex_functions = []
        
        try:
            # Try to parse the code using AST for accurate complexity calculation
            code_text = '\n'.join(lines)
            tree = ast.parse(code_text)
            
            # Walk through the AST to find function definitions and calculate complexity
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    complexity = self._calculate_cyclomatic_complexity(node)
                    
                    if complexity > 10:  # Threshold: complexity of 10
                        complex_functions.append({
                            "name": node.name,
                            "line_num": node.lineno,
                            "complexity": complexity
                        })
        except Exception as e:
            # Fallback to the original implementation if AST approach fails
            logger.warning(f"Using fallback complexity detection due to: {e}")
            
            # Original implementation
            current_function = None
            function_start_line = 0
            complexity = 4  # Base complexity (function entry, exit, etc.)
            
            for i, line in enumerate(lines):
                stripped_line = line.strip()
                
                if stripped_line.startswith('def '):
                    # If we were tracking a previous function, check if it was too complex
                    if current_function and complexity > 10:  # Threshold: complexity of 10
                        complex_functions.append({
                            "name": current_function,
                            "line_num": function_start_line + 1,
                            "complexity": complexity
                        })
                    
                    # Start tracking the new function
                    current_function = stripped_line[4:stripped_line.find('(')]
                    function_start_line = i
                    complexity = 4  # Reset complexity for new function
                elif current_function is not None:
                    # Add complexity for control flow statements
                    if any(keyword in line for keyword in ['if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except ', 'finally:']):
                        complexity += 1
                    # Add complexity for logical operators
                    complexity += line.count(' and ') + line.count(' or ')
            
            # Check the last function
            if current_function and complexity > 10:
                complex_functions.append({
                    "name": current_function,
                    "line_num": function_start_line + 1,
                    "complexity": complexity
                })
        
        return complex_functions

    def _calculate_cyclomatic_complexity(self, func_node) -> int:
        """Calculate the cyclomatic complexity of a function node."""
        complexity = 1  # Base complexity: main path
        
        for child in ast.walk(func_node):
            # Add complexity for control flow statements
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1  # Each except block adds complexity
            elif isinstance(child, ast.BoolOp):
                # For boolean operations like 'a and b or c', add complexity for each operand after the first
                complexity += len(child.values) - 1
        
        return complexity

    def _find_performance_issues(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Find potential performance issues."""
        issues = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Look for inefficient operations
            if 'for ' in line and ' in ' in line and 'range(0, len(' in line:
                issues.append({
                    "description": "Inefficient iteration using range(len()) instead of direct iteration",
                    "line_num": i + 1,
                    "severity": "medium",
                    "suggestion": "Use 'for item in list' instead of 'for i in range(len(list))'"
                })
            
            if stripped_line.startswith('import ') and 'from ' not in line and len(stripped_line.split('.')) > 3:
                issues.append({
                    "description": "Deep import that could affect startup performance",
                    "line_num": i + 1,
                    "severity": "low",
                    "suggestion": "Consider optimizing import structure or lazy loading"
                })
        
        return issues

    def _find_potential_bugs(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Find potential bugs."""
        issues = []
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Look for potential bugs
            if stripped_line.startswith('if ') and stripped_line.endswith(':') and '==' not in line and '!=' not in line:
                # Check for potential assignment instead of comparison (though in Python this would cause an error)
                if '=' in stripped_line[3:-1] and not 'is ' in stripped_line:  # Basic check
                    issues.append({
                        "description": "Potential logic issue in if statement",
                        "line_num": i + 1,
                        "severity": "high",
                        "suggestion": "Review the condition for correct logical operators"
                    })
            
            if 'except:' in stripped_line or stripped_line.startswith('except :'):
                issues.append({
                    "description": "Broad exception handling that may hide errors",
                    "line_num": i + 1,
                    "severity": "medium",
                    "suggestion": "Use specific exception types instead of bare 'except:'"
                })
        
        return issues

    def _process_communication(self, comm_message: CommunicationMessage):
        """Process communication message in threading context"""
        try:
            self.communications_sent += 1

            # Log communication
            self.log_manager.log_system_event_sync(
                "communication_processed",
                {
                    "message_id": comm_message.message_id,
                    "message_type": comm_message.message_type,
                    "priority": comm_message.priority.value
                },
                worker_id="communication_processor"
            )

        except Exception as e:
            self.log_manager.log_system_event_sync(
                "communication_error",
                {"error": str(e)},
                level="error",
                worker_id="communication_processor"
            )

    async def _handle_experiment_result(self, result: Dict[str, Any]):
        """Handle experiment results from process manager with enhanced improvement proposal system"""
        try:
            self.experiments_completed += 1

            # Store experiment result in VLTM
            if self.vltm_enabled and self.vltm_store:
                await self._store_experiment_memory(result)

            # If experiment was successful, create improvement proposal
            improvement_opportunities = []
            if result.get("success", False):
                # Analyze the experiment result for improvement opportunities
                improvement_opportunities = await self._analyze_improvement_opportunities_from_result(result)
                
                if improvement_opportunities or result.get("data", {}).get("improvement_opportunities"):
                    # Combine opportunities from result and analysis
                    all_opportunities = improvement_opportunities + result.get("data", {}).get("improvement_opportunities", [])
                    
                    improvement_task = {
                        "type": "improvement",
                        "task_id": f"imp_{uuid.uuid4().hex[:8]}",
                        "data": {
                            "experiment_result": result,
                            "improvement_opportunities": all_opportunities,
                            "priority": TaskPriority.MEDIUM.value
                        }
                    }

                    # Queue improvement processing
                    self.process_manager.distribute_task(improvement_task)

            await self.log_manager.log_system_event(
                "experiment_result_handled",
                {
                    "task_id": result.get("task_id"),
                    "success": result.get("success", False),
                    "improvement_queued": result.get("success", False),
                    "improvement_opportunities_found": len(improvement_opportunities)
                },
                worker_id="enhanced_snake"
            )

        except Exception as e:
            await self.log_manager.log_system_event(
                "experiment_result_error",
                {"error": str(e)},
                level="error",
                worker_id="enhanced_snake"
            )

    async def _analyze_improvement_opportunities_from_result(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze experiment results for improvement opportunities.
        
        Args:
            result: The experiment result to analyze
            
        Returns:
            List of improvement opportunities identified from the result
        """
        opportunities = []
        
        try:
            # Use LLM to analyze the results and suggest improvements
            experiment_data = result.get("data", {})
            experiment_result = result.get("results", {})
            
            # Prepare an analysis prompt
            analysis_prompt = f"""
            Analyze the following experiment result and identify potential improvement opportunities:
            
            Experiment Hypothesis: {result.get('hypothesis', 'N/A')}
            Experiment Findings: {result.get('findings', 'N/A')}
            Experiment Success: {result.get('success', 'N/A')}
            Experiment Confidence: {result.get('confidence', 'N/A')}
            Results Data: {experiment_result}
            
            Based on this experiment, identify specific improvement opportunities.
            Focus on:
            1. Code improvements
            2. Process improvements
            3. Architectural improvements
            
            Return a list of improvement opportunities with descriptions.
            """
            
            if self.reasoning_llm:
                try:
                    # This is a placeholder - in a real implementation, we'd call the reasoning LLM
                    pass
                except:
                    pass
            
            # For now, we'll use a simple heuristic-based approach
            # In a full implementation, this would use the LLM to generate insights
            if result.get('success', False) and result.get('confidence', 0.0) > 0.7:
                # If experiment was successful and high confidence, suggest implementing the findings
                opportunities.append({
                    "type": "implementation",
                    "category": "successful_experiment",
                    "description": f"Successfully validated hypothesis: {result.get('hypothesis', 'N/A')}. Implementation recommended.",
                    "location": "system-wide",
                    "severity": "medium",
                    "suggestion": "Implement the successful approach in the main codebase"
                })
            elif not result.get('success', False):
                # If experiment failed, suggest alternative approaches
                opportunities.append({
                    "type": "alternative_approach",
                    "category": "failed_experiment",
                    "description": f"Experiment hypothesis not validated: {result.get('hypothesis', 'N/A')}. Alternative approach needed.",
                    "location": "system-wide",
                    "severity": "medium",
                    "suggestion": "Design a new experiment with an alternative approach"
                })
        
        except Exception as e:
            logger.warning(f"Could not analyze experiment result for improvements: {e}")
        
        return opportunities

    async def _handle_analysis_result(self, result: Dict[str, Any]):
        """Handle analysis results from process manager"""
        try:
            await self.log_manager.log_system_event(
                "analysis_result_handled",
                {"task_id": result.get("task_id")},
                worker_id="enhanced_snake"
            )

        except Exception as e:
            await self.log_manager.log_system_event(
                "analysis_result_error",
                {"error": str(e)},
                level="error",
                worker_id="enhanced_snake"
            )

    async def _handle_improvement_result(self, result: Dict[str, Any]):
        """Handle improvement results from process manager"""
        try:
            success = result.get("success", False)
            if success:
                self.improvements_applied += 1

            # Record improvement metrics for performance tracking
            if hasattr(self.agi_system, 'performance_tracker'):
                # Calculate impact score based on various factors
                impact_score = self._calculate_improvement_impact(result)
                confidence = result.get("confidence", 0.5)  # Default confidence
                
                # Get implementation time if available
                start_time_str = result.get("start_time")
                if start_time_str:
                    try:
                        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                        implementation_time = (datetime.now() - start_time).total_seconds()
                    except:
                        implementation_time = 0
                else:
                    implementation_time = 0
                
                # Record the improvement with the performance tracker
                self.agi_system.performance_tracker.record_improvement(
                    improvement_id=result.get("task_id", "unknown"),
                    improvement_type=result.get("type", "general"),
                    impact_score=impact_score,
                    confidence=confidence,
                    implementation_time=implementation_time,
                    success=success
                )
                
                # Record additional metrics
                await self.agi_system.performance_tracker.record_metric(
                    name="improvement_impact",
                    value=impact_score,
                    unit="score",
                    source="snake_agent",
                    tags=["improvement", "impact"]
                )
                
                # Increment improvement counter in tracker
                self.agi_system.performance_tracker.increment_improvement_count()

            await self.log_manager.log_system_event(
                "improvement_result_handled",
                {
                    "task_id": result.get("task_id"),
                    "success": success
                },
                worker_id="enhanced_snake"
            )

        except Exception as e:
            await self.log_manager.log_system_event(
                "improvement_result_error",
                {"error": str(e)},
                level="error",
                worker_id="enhanced_snake"
            )

    def _calculate_improvement_impact(self, result: Dict[str, Any]) -> float:
        """
        Calculate the impact score of an improvement based on various factors.
        
        Args:
            result: The improvement result data
            
        Returns:
            Impact score between 0 and 1
        """
        # Default impact calculation - in a real system, this would be more sophisticated
        base_impact = 0.5  # Base impact
        
        # Increase impact if the improvement was successful
        if result.get("success", False):
            base_impact += 0.3
        
        # Consider the confidence in the improvement
        confidence = result.get("confidence", 0.5)
        base_impact += (confidence - 0.5) * 0.4  # Adjust by up to 0.2 based on confidence
        
        # Consider the type of improvement
        imp_type = result.get("type", "")
        if "performance" in imp_type.lower():
            base_impact += 0.1
        elif "efficiency" in imp_type.lower():
            base_impact += 0.1
        elif "bug" in imp_type.lower():
            base_impact += 0.05  # Bug fixes have moderate impact
        
        # Ensure impact is between 0 and 1
        return max(0.0, min(1.0, base_impact))

    async def _perform_health_check(self):
        """Perform system health check"""
        try:
            health_status = {
                "threading_manager": json.dumps({
                    "active": bool(self.threading_manager),
                    "threads": self.threading_manager.get_thread_status() if self.threading_manager else {},
                    "queues": self.threading_manager.get_queue_status() if self.threading_manager else {}
                }),
                "process_manager": json.dumps({
                    "active": bool(self.process_manager),
                    "processes": self.process_manager.get_process_status() if self.process_manager else {},
                    "queues": self.process_manager.get_queue_status() if self.process_manager else {}
                }),
                "file_monitor": json.dumps({
                    "active": bool(self.file_monitor),
                    "status": self.file_monitor.get_monitoring_status() if self.file_monitor else {}
                })
            }

            await self.log_manager.log_system_event(
                "health_check",
                health_status,
                worker_id="enhanced_snake"
            )

        except Exception as e:
            await self.log_manager.log_system_event(
                "health_check_error",
                {"error": str(e)},
                level="error",
                worker_id="enhanced_snake"
            )

    async def _log_performance_metrics(self):
        """Log performance metrics"""
        try:
            if self.start_time:
                uptime = datetime.now() - self.start_time

                metrics = {
                    "uptime_seconds": uptime.total_seconds(),
                    "improvements_applied": self.improvements_applied,
                    "experiments_completed": self.experiments_completed,
                    "files_analyzed": self.files_analyzed,
                    "communications_sent": self.communications_sent,
                    "improvements_per_hour": self.improvements_applied / max(uptime.total_seconds() / 3600, 1),
                    "experiments_per_hour": self.experiments_completed / max(uptime.total_seconds() / 3600, 1)
                }

                await self.log_manager.log_system_event(
                    "performance_metrics",
                    metrics,
                    worker_id="enhanced_snake"
                )

        except Exception as e:
            await self.log_manager.log_system_event(
                "metrics_logging_error",
                {"error": str(e)},
                level="error",
                worker_id="enhanced_snake"
            )

    async def _save_state(self):
        """Save enhanced agent state"""
        try:
            state_data = {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "improvements_applied": self.improvements_applied,
                "experiments_completed": self.experiments_completed,
                "files_analyzed": self.files_analyzed,
                "communications_sent": self.communications_sent,
                "config": self.snake_config.to_dict(),
                "last_saved": datetime.now().isoformat()
            }

            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving enhanced agent state: {e}")

    async def _load_state(self):
        """Load enhanced agent state"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)

                self.improvements_applied = state_data.get(
                    "improvements_applied", 0)
                self.experiments_completed = state_data.get(
                    "experiments_completed", 0)
                self.files_analyzed = state_data.get("files_analyzed", 0)
                self.communications_sent = state_data.get(
                    "communications_sent", 0)

                if state_data.get("start_time"):
                    self.start_time = datetime.fromisoformat(
                        state_data["start_time"])

                logger.info("Loaded enhanced agent state")

        except Exception as e:
            logger.warning(f"Could not load enhanced agent state: {e}")

    async def stop(self):
        """Stop the Enhanced Snake Agent gracefully"""
        logger.info("Stopping Enhanced Snake Agent...")
        self.running = False
        self._shutdown_event.set()

        try:
            if self.log_manager:
                await self.log_manager.log_system_event(
                    "enhanced_snake_shutdown_start",
                    {"uptime": (datetime.now() - self.start_time).total_seconds()
                     if self.start_time else 0},
                    worker_id="enhanced_snake"
                )

            # Stop file monitoring
            if self.file_monitor:
                await self.file_monitor.stop_monitoring()

            # Stop threading manager
            if self.threading_manager:
                await self.threading_manager.shutdown()

            # Stop process manager
            if self.process_manager:
                await self.process_manager.shutdown()

            # Save final state
            await self._save_state()

            # Stop log manager last
            if self.log_manager:
                self.log_manager.stop_log_processor()

            logger.info("Enhanced Snake Agent stopped successfully")

        except Exception as e:
            if self.log_manager:
                # Log with full traceback to snake_errors.log
                self.log_manager.log_error_with_traceback(e, "Error during Enhanced Snake Agent shutdown", 
                    {"component": "enhanced_snake_agent", "phase": "shutdown"})
            else:
                # Fallback logging if log_manager is not set up
                logger.error(f"Error during Enhanced Snake Agent shutdown: {e}", exc_info=True)

    async def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Enhanced Snake Agent resources...")

        # Stop VLTM integration if enabled
        if self.vltm_enabled and self.memory_integration_manager:
            try:
                await self.memory_integration_manager.stop_integration()
                logger.info("VLTM integration stopped")
            except Exception as e:
                logger.error(f"Error stopping VLTM integration: {e}")

        await self._save_state()

    async def _store_file_change_memory(self, file_event: FileChangeEvent):
        """Store file change event as memory in VLTM"""
        try:
            if not self.vltm_store:
                return

            memory_content = {
                "event_type": "file_change",
                "file_path": file_event.file_path,
                "change_type": file_event.event_type,
                "file_hash": file_event.file_hash,
                "old_hash": file_event.old_hash,
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "agent_context": {
                    "files_analyzed": self.files_analyzed,
                    "experiments_completed": self.experiments_completed
                }
            }

            metadata = {
                "source": "enhanced_snake_agent",
                "category": "code_change",
                "file_extension": Path(file_event.file_path).suffix,
                "change_significance": "high" if file_event.file_path.endswith('.py') else "medium"
            }

            memory_id = await self.vltm_store.store_memory(
                content=memory_content,
                memory_type=MemoryType.CODE_PATTERN,
                metadata=metadata,
                source_session=self.session_id
            )

            if memory_id:
                logger.debug(f"Stored file change memory: {memory_id}")

        except Exception as e:
            logger.error(f"Error storing file change memory: {e}")

    async def _store_experiment_memory(self, result: Dict[str, Any]):
        """Store experiment result as memory in VLTM"""
        try:
            if not self.vltm_store:
                return

            memory_content = {
                "event_type": "experiment_result",
                "experiment_id": result.get("task_id"),
                "success": result.get("success", False),
                "experiment_data": result.get("data", {}),
                "results": result.get("results", {}),
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "agent_context": {
                    "total_experiments": self.experiments_completed,
                    "total_improvements": self.improvements_applied,
                    "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0
                }
            }

            # Determine memory type based on result
            if result.get("success", False):
                memory_type = MemoryType.SUCCESSFUL_IMPROVEMENT
            else:
                memory_type = MemoryType.FAILED_EXPERIMENT

            metadata = {
                "source": "enhanced_snake_agent",
                "category": "experiment",
                "outcome": "success" if result.get("success", False) else "failure",
                "experiment_type": result.get("type", "unknown")
            }

            memory_id = await self.vltm_store.store_memory(
                content=memory_content,
                memory_type=memory_type,
                metadata=metadata,
                source_session=self.session_id
            )

            if memory_id:
                logger.debug(f"Stored experiment memory: {memory_id}")

        except Exception as e:
            logger.error(f"Error storing experiment memory: {e}")

    async def get_vltm_insights(self, query: str) -> List[Dict[str, Any]]:
        """Get insights from very long-term memory"""
        try:
            if not self.vltm_enabled or not self.vltm_store:
                return []

            # Search memories for insights
            memories = await self.vltm_store.search_memories(
                query=query,
                memory_types=[MemoryType.STRATEGIC_KNOWLEDGE,
                              MemoryType.SUCCESSFUL_IMPROVEMENT],
                limit=10
            )

            return memories

        except Exception as e:
            logger.error(f"Error getting VLTM insights: {e}")
            return []

    async def trigger_memory_consolidation(self, consolidation_type: ConsolidationType = ConsolidationType.DAILY):
        """Manually trigger memory consolidation"""
        try:
            if not self.vltm_enabled or not self.consolidation_engine:
                logger.warning(
                    "VLTM not enabled or consolidation engine not available")
                return

            from core.vltm_data_models import ConsolidationRequest

            request = ConsolidationRequest(
                consolidation_type=consolidation_type,
                force_consolidation=True
            )

            result = await self.consolidation_engine.consolidate_memories(request)

            if result.success:
                logger.info(f"Memory consolidation completed: {result.memories_processed} memories processed, "
                            f"{result.patterns_extracted} patterns extracted")

                await self.log_manager.log_system_event(
                    "memory_consolidation_completed",
                    {
                        "consolidation_id": result.consolidation_id,
                        "memories_processed": result.memories_processed,
                        "patterns_extracted": result.patterns_extracted,
                        "processing_time": result.processing_time_seconds
                    },
                    worker_id="enhanced_snake"
                )
            else:
                logger.error(
                    f"Memory consolidation failed: {result.error_message}")

        except Exception as e:
            logger.error(f"Error triggering memory consolidation: {e}")

    async def _sync_goals_with_system(self, snake_goals: List[Dict[str, Any]]):
        """Sync snake agent goals with the main system goal manager"""
        try:
            # Check if main system has goal manager available
            if not hasattr(self, 'agi_system') or not self.agi_system or not hasattr(self.agi_system, 'self_goal_manager'):
                logger.warning("AGI system or goal manager not available for sync")
                return

            # Convert snake agent goals to main system goals
            for goal in snake_goals:
                goal_type = goal.get('type', 'general')
                title = goal.get('description', f"Improve {goal_type.replace('_', ' ').title()}")
                target_date = datetime.now(timezone.utc) + timedelta(days=goal.get('timeframe_days', 7))
                
                # Map priority from snake agent format to main system format
                priority_map = {
                    'low': GoalPriority.LOW,
                    'medium': GoalPriority.MEDIUM,
                    'high': GoalPriority.HIGH,
                    'critical': GoalPriority.CRITICAL
                }
                priority = priority_map.get(goal.get('priority', 'medium'), GoalPriority.MEDIUM)
                
                # Check if goal already exists in the system to avoid duplicates
                existing_goal = None
                for sys_goal in self.agi_system.self_goal_manager.goals.values():
                    if sys_goal.title == title or goal.get('id') in sys_goal.id:
                        existing_goal = sys_goal
                        break
                
                if existing_goal is None:
                    # Create goal in the main system
                    self.agi_system.self_goal_manager.create_goal(
                        title=title,
                        description=f"Snake Agent goal: {goal.get('description', 'General improvement goal')}. "
                                  f"Target: {goal.get('target', 'unknown')}, "
                                  f"Current: {goal.get('current', 'unknown')}. "
                                  f"Strategies: {', '.join(goal.get('strategies', []))}",
                        category="snake_agent_improvement",
                        priority=priority,
                        target_date=target_date,
                        metrics={goal_type: goal.get('target', 0.5)},
                        dependencies=[]
                    )
                    logger.info(f"Added snake agent goal to main system: {title}")
            
            # Save the updated goals to persistent storage
            await self.agi_system.self_goal_manager.save_goals()
            
        except Exception as e:
            logger.error(f"Error syncing goals with system: {e}")
            import traceback
            traceback.print_exc()

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the enhanced agent"""
        status = {
            "running": self.running,
            "initialized": self.initialized,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "metrics": {
                "improvements_applied": self.improvements_applied,
                "experiments_completed": self.experiments_completed,
                "files_analyzed": self.files_analyzed,
                "communications_sent": self.communications_sent
            },
            "components": {
                "threading_manager": bool(self.threading_manager),
                "process_manager": bool(self.process_manager),
                "file_monitor": bool(self.file_monitor),
                "log_manager": bool(self.log_manager),
                "vltm_enabled": self.vltm_enabled,
                "vltm_store": bool(self.vltm_store),
                "memory_integration_manager": bool(self.memory_integration_manager),
                "consolidation_engine": bool(self.consolidation_engine)
            }
        }

        # Add component-specific status if available
        if self.threading_manager:
            status["threading_status"] = self.threading_manager.get_thread_status()
            status["thread_queues"] = self.threading_manager.get_queue_status()

        if self.process_manager:
            status["process_status"] = self.process_manager.get_process_status()
            status["process_queues"] = self.process_manager.get_queue_status()

        if self.file_monitor:
            status["monitoring_status"] = self.file_monitor.get_monitoring_status()

        # Add VLTM status if enabled
        if self.vltm_enabled:
            vltm_status = {
                "enabled": True,
                "session_id": self.session_id,
                "storage_dir": str(self.vltm_storage_dir),
                "integration_active": bool(self.memory_integration_manager and
                                           getattr(self.memory_integration_manager, 'integration_active', False))
            }

            # Add store statistics if available
            if self.vltm_store:
                try:
                    vltm_stats = await self.vltm_store.get_memory_statistics()
                    vltm_status["statistics"] = vltm_stats
                except Exception as e:
                    logger.warning(f"Could not get VLTM statistics: {e}")

            status["vltm_status"] = vltm_status
        else:
            status["vltm_status"] = {"enabled": False}

        return status
