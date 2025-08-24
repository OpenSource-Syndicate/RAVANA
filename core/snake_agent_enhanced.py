"""
Enhanced Snake Agent

This module implements the enhanced Snake Agent that uses threading and multiprocessing
to continuously improve RAVANA through concurrent analysis, experimentation, and improvement.
"""

import asyncio
import logging
import os
import json
import time
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

# VLTM imports
from core.vltm_store import VeryLongTermMemoryStore
from core.vltm_memory_integration_manager import MemoryIntegrationManager
from core.vltm_consolidation_engine import MemoryConsolidationEngine
from core.vltm_consolidation_scheduler import ConsolidationScheduler
from core.vltm_lifecycle_manager import MemoryLifecycleManager
from core.vltm_storage_backend import StorageBackend
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
        
        # Enhanced configuration
        self.snake_config = SnakeAgentConfiguration(
            max_threads=int(os.getenv('SNAKE_MAX_THREADS', '8')),
            max_processes=int(os.getenv('SNAKE_MAX_PROCESSES', '4')),
            analysis_threads=int(os.getenv('SNAKE_ANALYSIS_THREADS', '3')),
            file_monitor_interval=float(os.getenv('SNAKE_MONITOR_INTERVAL', '2.0')),
            enable_performance_monitoring=os.getenv('SNAKE_PERF_MONITORING', 'true').lower() == 'true'
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
        self.vltm_enabled = os.getenv('SNAKE_VLTM_ENABLED', 'true').lower() == 'true'
        self.vltm_storage_dir = Path(os.getenv('SNAKE_VLTM_STORAGE_DIR', 'snake_vltm_storage'))
        self.session_id = str(uuid.uuid4())  # Unique session identifier
    
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
            
            # Initialize LLM interfaces with better error handling
            try:
                self.coding_llm = await create_snake_coding_llm()
                self.reasoning_llm = await create_snake_reasoning_llm()
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
                    logger.warning("Failed to initialize VLTM - continuing without it")
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
                    self.threading_manager = SnakeThreadingManager(self.snake_config, self.log_manager)
                    if await self.threading_manager.initialize():
                        break
                    else:
                        raise Exception("Threading manager initialization returned False")
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Failed to initialize threading manager (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    else:
                        raise Exception("Failed to initialize threading manager after retries")
            
            # Initialize process manager with retry logic
            retry_count = 0
            while retry_count < max_retries:
                try:
                    self.process_manager = SnakeProcessManager(self.snake_config, self.log_manager)
                    if await self.process_manager.initialize():
                        break
                    else:
                        raise Exception("Process manager initialization returned False")
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Failed to initialize process manager (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    else:
                        raise Exception("Failed to initialize process manager after retries")
            
            # Initialize file monitor with retry logic
            retry_count = 0
            while retry_count < max_retries:
                try:
                    self.file_monitor = ContinuousFileMonitor(self, self.snake_config, self.log_manager)
                    if await self.file_monitor.initialize():
                        break
                    else:
                        raise Exception("File monitor initialization returned False")
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Failed to initialize file monitor (attempt {retry_count}/{max_retries}): {e}")
                    if retry_count < max_retries:
                        await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                    else:
                        raise Exception("Failed to initialize file monitor after retries")
            
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
            logger.error(f"Failed to initialize Enhanced Snake Agent: {e}", exc_info=True)
            if self.log_manager:
                await self.log_manager.log_system_event(
                    "enhanced_snake_init_failed",
                    {"error": str(e)},
                    level="error",
                    worker_id="enhanced_snake"
                )
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
            self.knowledge_service = KnowledgeService()
            
            # Initialize VLTM store
            self.vltm_store = VeryLongTermMemoryStore(
                config=DEFAULT_VLTM_CONFIG,
                base_storage_dir=str(self.vltm_storage_dir)
            )
            
            if not await self.vltm_store.initialize():
                logger.error("Failed to initialize VLTM store")
                return False
            
            # Initialize storage backend
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
            
            # Initialize consolidation scheduler
            self.consolidation_scheduler = ConsolidationScheduler(
                config=DEFAULT_VLTM_CONFIG
            )
            
            # Initialize memory integration manager
            self.memory_integration_manager = MemoryIntegrationManager(
                existing_memory_service=self.memory_service,
                knowledge_service=self.knowledge_service,
                vltm_store=self.vltm_store,
                config=DEFAULT_VLTM_CONFIG
            )
            
            # Set up consolidation components
            self.memory_integration_manager.set_consolidation_components(
                consolidation_engine=self.consolidation_engine,
                lifecycle_manager=self.lifecycle_manager,
                consolidation_scheduler=self.consolidation_scheduler
            )
            
            # Start memory integration
            if not await self.memory_integration_manager.start_integration():
                logger.error("Failed to start memory integration")
                return False
            
            logger.info("Very Long-Term Memory system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing VLTM: {e}", exc_info=True)
            return False
    
    async def start_autonomous_operation(self):
        """Start the enhanced autonomous operation with threading and multiprocessing"""
        if not self.initialized:
            if not await self.initialize():
                logger.error("Cannot start Enhanced Snake Agent - initialization failed")
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
            logger.error(f"Error in Enhanced Snake Agent operation: {e}")
            await self.log_manager.log_system_event(
                "autonomous_operation_error",
                {"error": str(e)},
                level="error",
                worker_id="enhanced_snake"
            )
        finally:
            await self._cleanup()
    
    async def _coordination_loop(self):
        """Main coordination loop for the enhanced agent"""
        coordination_interval = 10.0  # 10 seconds
        last_health_check = datetime.now()
        last_metrics_log = datetime.now()
        
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
                    
                    # State persistence
                    await self._save_state()
                
                # Wait for next coordination cycle
                await asyncio.sleep(coordination_interval)
                
            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                await self.log_manager.log_system_event(
                    "coordination_loop_error",
                    {"error": str(e)},
                    level="error",
                    worker_id="enhanced_snake"
                )
                await asyncio.sleep(coordination_interval)
    
    def _handle_file_change(self, file_event: FileChangeEvent):
        """Handle file change events from file monitor"""
        try:
            # Convert to analysis task if it's a Python file
            if file_event.file_path.endswith('.py'):
                analysis_task = AnalysisTask(
                    task_id=f"analysis_{uuid.uuid4().hex[:8]}",
                    file_path=file_event.file_path,
                    analysis_type="file_change",
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
            asyncio.create_task(self.log_manager.log_system_event(
                "file_change_handled",
                {
                    "event_type": file_event.event_type,
                    "file_path": file_event.file_path,
                    "queued_for_analysis": file_event.file_path.endswith('.py')
                },
                worker_id="enhanced_snake"
            ))
            
        except Exception as e:
            asyncio.create_task(self.log_manager.log_system_event(
                "file_change_error",
                {"error": str(e), "event": file_event.to_dict()},
                level="error",
                worker_id="enhanced_snake"
            ))
    
    def _process_file_change(self, file_event: FileChangeEvent):
        """Process file change in threading context"""
        try:
            # Update file analysis count
            self.files_analyzed += 1
            
            # Store in VLTM if enabled
            if self.vltm_enabled and self.vltm_store:
                asyncio.create_task(self._store_file_change_memory(file_event))
            
            # Log processing
            asyncio.create_task(self.log_manager.log_system_event(
                "file_change_processed",
                {"file_path": file_event.file_path},
                worker_id="file_processor"
            ))
            
        except Exception as e:
            asyncio.create_task(self.log_manager.log_system_event(
                "file_processing_error",
                {"error": str(e)},
                level="error",
                worker_id="file_processor"
            ))
    
    def _process_analysis_task(self, analysis_task: AnalysisTask):
        """Process analysis task in threading context"""
        try:
            # For significant findings, create experiment task
            if analysis_task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]:
                experiment_task = {
                    "type": "experiment",
                    "task_id": f"exp_{uuid.uuid4().hex[:8]}",
                    "data": {
                        "file_path": analysis_task.file_path,
                        "analysis_type": analysis_task.analysis_type,
                        "priority": analysis_task.priority.value
                    }
                }
                
                # Distribute to process manager
                self.process_manager.distribute_task(experiment_task)
            
            # Log analysis processing
            asyncio.create_task(self.log_manager.log_system_event(
                "analysis_task_processed",
                {
                    "task_id": analysis_task.task_id,
                    "file_path": analysis_task.file_path,
                    "experiment_created": analysis_task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]
                },
                worker_id="analysis_processor"
            ))
            
        except Exception as e:
            asyncio.create_task(self.log_manager.log_system_event(
                "analysis_processing_error",
                {"error": str(e), "task": analysis_task.to_dict()},
                level="error",
                worker_id="analysis_processor"
            ))
    
    def _process_communication(self, comm_message: CommunicationMessage):
        """Process communication message in threading context"""
        try:
            self.communications_sent += 1
            
            # Log communication
            asyncio.create_task(self.log_manager.log_system_event(
                "communication_processed",
                {
                    "message_id": comm_message.message_id,
                    "message_type": comm_message.message_type,
                    "priority": comm_message.priority.value
                },
                worker_id="communication_processor"
            ))
            
        except Exception as e:
            asyncio.create_task(self.log_manager.log_system_event(
                "communication_error",
                {"error": str(e)},
                level="error",
                worker_id="communication_processor"
            ))
    
    async def _handle_experiment_result(self, result: Dict[str, Any]):
        """Handle experiment results from process manager"""
        try:
            self.experiments_completed += 1
            
            # Store experiment result in VLTM
            if self.vltm_enabled and self.vltm_store:
                await self._store_experiment_memory(result)
            
            # If experiment was successful, create improvement proposal
            if result.get("success", False):
                improvement_task = {
                    "type": "improvement",
                    "task_id": f"imp_{uuid.uuid4().hex[:8]}",
                    "data": {
                        "experiment_result": result,
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
                    "improvement_queued": result.get("success", False)
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
            if result.get("success", False):
                self.improvements_applied += 1
            
            await self.log_manager.log_system_event(
                "improvement_result_handled",
                {
                    "task_id": result.get("task_id"),
                    "success": result.get("success", False)
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
    
    async def _perform_health_check(self):
        """Perform system health check"""
        try:
            health_status = {
                "threading_manager": {
                    "active": bool(self.threading_manager),
                    "threads": self.threading_manager.get_thread_status() if self.threading_manager else {},
                    "queues": self.threading_manager.get_queue_status() if self.threading_manager else {}
                },
                "process_manager": {
                    "active": bool(self.process_manager),
                    "processes": self.process_manager.get_process_status() if self.process_manager else {},
                    "queues": self.process_manager.get_queue_status() if self.process_manager else {}
                },
                "file_monitor": {
                    "active": bool(self.file_monitor),
                    "status": self.file_monitor.get_monitoring_status() if self.file_monitor else {}
                }
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
                
                self.improvements_applied = state_data.get("improvements_applied", 0)
                self.experiments_completed = state_data.get("experiments_completed", 0)
                self.files_analyzed = state_data.get("files_analyzed", 0)
                self.communications_sent = state_data.get("communications_sent", 0)
                
                if state_data.get("start_time"):
                    self.start_time = datetime.fromisoformat(state_data["start_time"])
                
                logger.info("Loaded enhanced agent state")
                
        except Exception as e:
            logger.warning(f"Could not load enhanced agent state: {e}")
    
    async def stop(self):
        """Stop the Enhanced Snake Agent gracefully"""
        logger.info("Stopping Enhanced Snake Agent...")
        self.running = False
        self._shutdown_event.set()
        
        try:
            await self.log_manager.log_system_event(
                "enhanced_snake_shutdown_start",
                {"uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0},
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
            logger.error(f"Error during Enhanced Snake Agent shutdown: {e}")
    
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
                memory_types=[MemoryType.STRATEGIC_KNOWLEDGE, MemoryType.SUCCESSFUL_IMPROVEMENT],
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
                logger.warning("VLTM not enabled or consolidation engine not available")
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
                logger.error(f"Memory consolidation failed: {result.error_message}")
            
        except Exception as e:
            logger.error(f"Error triggering memory consolidation: {e}")
    
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