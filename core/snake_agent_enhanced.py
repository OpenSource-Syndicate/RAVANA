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
            
            # Initialize LLM interfaces
            self.coding_llm = await create_snake_coding_llm()
            self.reasoning_llm = await create_snake_reasoning_llm()
            
            # Initialize threading manager
            self.threading_manager = SnakeThreadingManager(self.snake_config, self.log_manager)
            if not await self.threading_manager.initialize():
                raise Exception("Failed to initialize threading manager")
            
            # Initialize process manager
            self.process_manager = SnakeProcessManager(self.snake_config, self.log_manager)
            if not await self.process_manager.initialize():
                raise Exception("Failed to initialize process manager")
            
            # Initialize file monitor
            self.file_monitor = ContinuousFileMonitor(self, self.snake_config, self.log_manager)
            if not await self.file_monitor.initialize():
                raise Exception("Failed to initialize file monitor")
            
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
            logger.error(f"Failed to initialize Enhanced Snake Agent: {e}")
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
        await self._save_state()
    
    def get_status(self) -> Dict[str, Any]:
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
                "log_manager": bool(self.log_manager)
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
        
        return status