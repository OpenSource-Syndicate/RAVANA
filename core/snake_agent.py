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
from dataclasses import dataclass
import hashlib

from core.config import Config
from core.snake_llm import create_snake_coding_llm, create_snake_reasoning_llm, SnakeConfigValidator
from core.snake_data_models import (
    SnakeAgentConfiguration, FileChangeEvent, AnalysisTask,
    TaskPriority, CommunicationMessage
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

# Allow tests to patch the analyzer class via the core.snake_agent module
SnakeCodeAnalyzer = None

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
            state.last_analysis_time = datetime.fromisoformat(
                data["last_analysis_time"])
        state.analyzed_files = set(data.get("analyzed_files", []))
        state.pending_experiments = data.get("pending_experiments", [])
        state.communication_queue = data.get("communication_queue", [])
        state.learning_history = data.get("learning_history", [])
        state.current_task = data.get("current_task")
        state.mood = data.get("mood", "curious")
        state.experiment_success_rate = data.get(
            "experiment_success_rate", 0.0)
        return state


class FileSystemMonitor:
    """Monitors RAVANA codebase for changes"""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.monitored_extensions = {
            '.py', '.json', '.md', '.txt', '.yml', '.yaml'}
        self.excluded_dirs = {'__pycache__', '.git',
                              '.venv', 'node_modules', '.qoder'}
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

        # Initialize components (will be set during startup)
        self.coding_llm = None
        self.reasoning_llm = None
        self.file_monitor = None  # Will be replaced with ContinuousFileMonitor if enhanced
        self.code_analyzer = None
        self.safe_experimenter = None
        self.ravana_communicator = None
        
        # Enhanced components
        self.log_manager: Optional[SnakeLogManager] = None
        self.threading_manager: Optional[SnakeThreadingManager] = None
        self.process_manager: Optional[SnakeProcessManager] = None
        self.continuous_file_monitor: Optional[ContinuousFileMonitor] = None

        # Control flags
        self.running = False
        self.initialized = False
        self._shutdown_event = asyncio.Event()
        self._task_lock = asyncio.Lock()
        self._coordination_lock = asyncio.Lock()

        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.analysis_count = 0
        self.experiment_count = 0
        self.improvements_applied = 0
        self.files_analyzed = 0
        self.communications_sent = 0
        self.communication_count = 0

        # State persistence
        self.state_file = Path("snake_agent_state.json")

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

    async def initialize(self) -> bool:
        """Initialize Snake Agent components"""
        try:
            logger.info("Initializing Snake Agent...")

            # Validate configuration
            startup_report = SnakeConfigValidator.get_startup_report()
            if not startup_report["config_valid"]:
                logger.error(
                    f"Snake Agent configuration invalid: {startup_report}")
                return False

            logger.info(
                f"Ollama connection: {startup_report['ollama_connected']}")
            logger.info(
                f"Available models: {startup_report['available_models']}")

            # Initialize LLM interfaces
            self.coding_llm = await create_snake_coding_llm()
            self.reasoning_llm = await create_snake_reasoning_llm()

            # Initialize file system monitor (base functionality)
            workspace_path = getattr(
                self.agi_system, 'workspace_path', os.getcwd())
            self.file_monitor = FileSystemMonitor(workspace_path)
            
            # Enhanced initialization
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
                    self.continuous_file_monitor = ContinuousFileMonitor(
                        self, self.snake_config, self.log_manager)
                    if await self.continuous_file_monitor.initialize():
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

            logger.info("Snake Agent initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Snake Agent: {e}", exc_info=True)
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
        if self.continuous_file_monitor:
            self.continuous_file_monitor.set_change_callback(self._handle_file_change)

        # Threading manager callbacks
        if self.threading_manager:
            self.threading_manager.set_callbacks(
                file_change_callback=self._process_file_change,
                analysis_callback=self._process_analysis_task,
                communication_callback=self._process_communication
            )

        # Process manager callbacks
        if self.process_manager:
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

            logger.info(
                "Very Long-Term Memory system initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing VLTM: {e}", exc_info=True)
            return False

    def _handle_file_change(self, file_event: FileChangeEvent):
        """Handle file change events from file monitor"""
        try:
            # Convert to analysis task if it's a Python file
            if file_event.file_path.endswith('.py'):
                # Determine if this should be an indexing task (for many threads) or study task (for single thread)
                # This will be processed by indexing threads (many)
                analysis_type = "file_change_indexing"

                analysis_task = AnalysisTask(
                    task_id=f"analysis_{hashlib.md5(file_event.file_path.encode()).hexdigest()[:8]}",
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
                if self.threading_manager:
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
                    "task_id": f"exp_{hashlib.md5(analysis_task.task_id.encode()).hexdigest()[:8]}",
                    "data": {
                        "file_path": analysis_task.file_path,
                        "analysis_type": analysis_task.analysis_type,
                        "priority": analysis_task.priority.value
                    }
                }

                # Distribute to process manager
                if self.process_manager:
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
                    "task_id": f"imp_{hashlib.md5(str(result.get('task_id', '')).encode()).hexdigest()[:8]}",
                    "data": {
                        "experiment_result": result,
                        "priority": TaskPriority.MEDIUM.value
                    }
                }

                # Queue improvement processing
                if self.process_manager:
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
                metadata=metadata
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
                metadata=metadata
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

    async def start_autonomous_operation(self):
        """Start the enhanced autonomous operation with threading and multiprocessing"""
        if not self.initialized:
            if not await self.initialize():
                logger.error(
                    "Cannot start Snake Agent - initialization failed")
                return

        self.running = True
        self.start_time = datetime.now()

        try:
            logger.info("Starting Snake Agent autonomous operation")

            await self.log_manager.log_system_event(
                "autonomous_operation_start",
                {"start_time": self.start_time.isoformat()},
                worker_id="enhanced_snake"
            )

            # Start all threading components
            if self.threading_manager and not await self.threading_manager.start_all_threads():
                raise Exception("Failed to start threading components")

            # Start all process components
            if self.process_manager and not await self.process_manager.start_all_processes():
                raise Exception("Failed to start process components")

            # Start file monitoring
            if self.continuous_file_monitor and not await self.continuous_file_monitor.start_monitoring():
                raise Exception("Failed to start file monitoring")

            # Start coordination loop
            await self._coordination_loop()

        except asyncio.CancelledError:
            logger.info("Snake Agent operation cancelled")
        except Exception as e:
            logger.error(f"Error in Snake Agent operation: {e}")
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
                    "active": bool(self.continuous_file_monitor),
                    "status": self.continuous_file_monitor.get_monitoring_status() if self.continuous_file_monitor else {}
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
                    # If configured, use a lightweight peek-based prioritizer to choose few files
                    if getattr(self.config, 'SNAKE_USE_PEEK_PRIORITIZER', False):
                        try:
                            # Lazy import to avoid startup costs
                            from core.snake_indexer import load_index, build_index
                            from pathlib import Path as _P

                            workspace_path = _P(
                                getattr(self.agi_system, 'workspace_path', os.getcwd()))
                            index_file = workspace_path / '.snake_index.json'

                            summaries = load_index(
                                index_file) if index_file.exists() else None
                            if summaries is None:
                                # Build one-shot index (persist it to speed future runs)
                                summaries = build_index(
                                    workspace_path, index_file=index_file, max_files=500)

                            top_n = 10
                            for s in (summaries or [])[:top_n]:
                                fp = s.get('path')
                                if fp and fp.endswith('.py'):
                                    await self._analyze_file(fp, 'periodic_peek')
                        except Exception as inner_e:
                            logger.warning(
                                f"Peek prioritizer failed, falling back to full analysis: {inner_e}")
                            await self._perform_periodic_analysis()
                    else:
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
                logger.error(
                    f"Failed to save state after cycle error: {save_error}")

    async def _process_file_changes(self, changes: List[Dict[str, Any]]):
        """Process detected file changes"""
        for change in changes:
            try:
                if change["type"] in ["new", "modified"] and change["path"].endswith('.py'):
                    # Analyze Python files immediately
                    await self._analyze_file(change["absolute_path"], change["type"])

                self.state.analyzed_files.add(change["path"])

            except Exception as e:
                logger.error(
                    f"Error processing file change {change['path']}: {e}")

    async def _analyze_file(self, file_path: str, change_type: str):
        """Analyze a specific file for improvement opportunities"""
        try:
            # Use module-level SnakeCodeAnalyzer if tests have patched it, otherwise import
            analyzer_cls = SnakeCodeAnalyzer
            if analyzer_cls is None:
                from core.snake_code_analyzer import SnakeCodeAnalyzer as _Analyzer
                analyzer_cls = _Analyzer

            if not self.code_analyzer:
                self.code_analyzer = analyzer_cls(self.coding_llm)

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
                self.safe_experimenter = SnakeSafeExperimenter(
                    self.coding_llm, self.reasoning_llm)

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
            logger.info(
                f"Experiment {experiment['id']} completed with success: {success}")

        except Exception as e:
            logger.error(
                f"Error processing experiment {experiment['id']}: {e}")
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
            logger.info(
                f"Sent communication message with priority {communication['priority']}")

        except Exception as e:
            logger.error(f"Error sending communication: {e}")
            # Re-queue failed communications at lower priority
            communication["priority"] = "low"
            communication["retry_count"] = communication.get(
                "retry_count", 0) + 1
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

            # Limit to 3 files per cycle
            for file_path in files_to_analyze[:3]:
                await self._analyze_file(file_path, "periodic")

            self.state.last_analysis_time = datetime.now()

        except Exception as e:
            logger.error(f"Error in periodic analysis: {e}")

    def _select_files_for_periodic_analysis(self) -> List[str]:
        """Select important files for periodic analysis"""
        important_files = []

        # Core system files
        core_patterns = ['core/system.py',
                         'core/llm.py', 'core/action_manager.py']

        # Module files
        module_patterns = ['modules/*/main.py', 'modules/*/*.py']

        workspace_path = Path(
            getattr(self.agi_system, 'workspace_path', os.getcwd()))

        for pattern in core_patterns:
            file_path = workspace_path / pattern
            if file_path.exists():
                important_files.append(str(file_path))

        return important_files

    def _update_mood(self):
        """Update agent mood based on recent performance"""
        # Ensure state exists and has experiment_success_rate attribute
        if not hasattr(self.state, 'experiment_success_rate'):
            logger.warning(
                "State missing experiment_success_rate, initializing to 0.0")
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

        logger.debug(
            f"Mood updated to '{self.state.mood}' based on success rate: {success_rate:.3f}")

    def _update_experiment_success_rate(self, success: bool):
        """Update experiment success rate with exponential moving average"""
        alpha = 0.1  # Learning rate
        current_success = 1.0 if success else 0.0
        self.state.experiment_success_rate = (
            alpha * current_success + (1 - alpha) *
            self.state.experiment_success_rate
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
                # Use __dict__ membership so that deleted attributes are detected
                if not (hasattr(self.state, attr) and attr in getattr(self.state, '__dict__', {})):
                    logger.warning(f"State missing required attribute: {attr}")
                    return False

            # Validate data types
            if not isinstance(self.state.experiment_success_rate, (int, float)):
                logger.warning(
                    f"Invalid experiment_success_rate type: {type(self.state.experiment_success_rate)}")
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

        try:
            await self.log_manager.log_system_event(
                "enhanced_snake_shutdown_start",
                {"uptime": (datetime.now() - self.start_time).total_seconds()
                 if self.start_time else 0},
                worker_id="enhanced_snake"
            )

            # Stop file monitoring
            if self.continuous_file_monitor:
                await self.continuous_file_monitor.stop_monitoring()

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

            logger.info("Snake Agent stopped successfully")

        except Exception as e:
            logger.error(f"Error during Snake Agent shutdown: {e}")

    async def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Snake Agent resources...")

        # Stop VLTM integration if enabled
        if self.vltm_enabled and self.memory_integration_manager:
            try:
                await self.memory_integration_manager.stop_integration()
                logger.info("VLTM integration stopped")
            except Exception as e:
                logger.error(f"Error stopping VLTM integration: {e}")

        # Stop log manager
        if self.log_manager:
            self.log_manager.stop_log_processor()

        await self._save_state()

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the enhanced agent"""
        status = {
            "running": self.running,
            "initialized": self.initialized,
            "mood": self.state.mood,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "metrics": {
                "improvements_applied": self.improvements_applied,
                "analysis_count": self.analysis_count,
                "experiment_count": self.experiment_count,
                "experiments_completed": self.experiments_completed,
                "files_analyzed": self.files_analyzed,
                "communication_count": self.communication_count,
                "communications_sent": self.communications_sent,
                "pending_experiments": len(self.state.pending_experiments),
                "communication_queue": len(self.state.communication_queue),
                "success_rate": self.state.experiment_success_rate
            },
            "components": {
                "threading_manager": bool(self.threading_manager),
                "process_manager": bool(self.process_manager),
                "file_monitor": bool(self.continuous_file_monitor),
                "log_manager": bool(self.log_manager),
                "vltm_enabled": self.vltm_enabled,
                "vltm_store": bool(self.vltm_store),
                "memory_integration_manager": bool(self.memory_integration_manager),
                "consolidation_engine": bool(self.consolidation_engine)
            },
            "last_analysis": self.state.last_analysis_time.isoformat() if self.state.last_analysis_time else None
        }

        # Add component-specific status if available
        if self.threading_manager:
            status["threading_status"] = self.threading_manager.get_thread_status()
            status["thread_queues"] = self.threading_manager.get_queue_status()

        if self.process_manager:
            status["process_status"] = self.process_manager.get_process_status()
            status["process_queues"] = self.process_manager.get_queue_status()

        if self.continuous_file_monitor:
            status["monitoring_status"] = self.continuous_file_monitor.get_monitoring_status()

        # Add VLTM status if enabled
        if self.vltm_enabled:
            vltm_status = {
                "enabled": True,
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
