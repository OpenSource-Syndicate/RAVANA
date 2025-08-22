"""
Snake Threading Manager

This module manages concurrent threads for Snake Agent operations including
file monitoring, code analysis, and RAVANA communication.
"""

import asyncio
import threading
import queue
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

from core.snake_data_models import (
    ThreadState, ThreadStatus, TaskInfo, TaskPriority, TaskStatus,
    FileChangeEvent, AnalysisTask, CommunicationMessage, WorkerMetrics,
    SnakeAgentConfiguration
)
from core.snake_log_manager import SnakeLogManager


class SnakeThreadingManager:
    """Manages concurrent threads for Snake Agent operations"""
    
    def __init__(self, config: SnakeAgentConfiguration, log_manager: SnakeLogManager):
        self.config = config
        self.log_manager = log_manager
        
        # Thread management
        self.active_threads: Dict[str, ThreadState] = {}
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.max_threads,
            thread_name_prefix="Snake"
        )
        
        # Task queues
        self.file_change_queue = queue.Queue(maxsize=config.max_queue_size)
        self.analysis_queue = queue.Queue(maxsize=config.max_queue_size)
        self.communication_queue = queue.Queue(maxsize=config.max_queue_size)
        
        # Coordination
        self.shutdown_event = threading.Event()
        self.coordination_lock = threading.Lock()
        
        # Worker metrics
        self.worker_metrics: Dict[str, WorkerMetrics] = {}
        
        # Callbacks for external integration
        self.file_change_callback: Optional[Callable] = None
        self.analysis_callback: Optional[Callable] = None
        self.communication_callback: Optional[Callable] = None
        
        # Performance tracking
        self.started_at = datetime.now()
        self.threads_created = 0
        self.tasks_processed = 0
        
    async def initialize(self) -> bool:
        """Initialize the threading manager"""
        try:
            await self.log_manager.log_system_event(
                "threading_manager_init",
                {"config": self.config.to_dict()},
                worker_id="threading_manager"
            )
            return True
        except Exception as e:
            await self.log_manager.log_system_event(
                "threading_manager_init_failed",
                {"error": str(e)},
                level="error",
                worker_id="threading_manager"
            )
            return False
    
    async def start_all_threads(self) -> bool:
        """Start all worker threads"""
        try:
            success = True
            
            # Start file monitor thread
            if not await self.start_file_monitor_thread():
                success = False
            
            # Start analysis threads
            if not await self.start_analysis_threads(self.config.analysis_threads):
                success = False
            
            # Start communication thread
            if not await self.start_communication_thread():
                success = False
            
            # Start performance monitoring thread if enabled
            if self.config.enable_performance_monitoring:
                if not await self.start_performance_monitor_thread():
                    success = False
            
            await self.log_manager.log_system_event(
                "all_threads_started",
                {"success": success, "active_threads": len(self.active_threads)},
                worker_id="threading_manager"
            )
            
            return success
            
        except Exception as e:
            await self.log_manager.log_system_event(
                "start_all_threads_failed",
                {"error": str(e)},
                level="error",
                worker_id="threading_manager"
            )
            return False
    
    async def start_file_monitor_thread(self) -> bool:
        """Start dedicated file monitoring thread"""
        try:
            thread_id = f"file_monitor_{uuid.uuid4().hex[:8]}"
            
            thread_state = ThreadState(
                thread_id=thread_id,
                name="FileMonitor",
                status=ThreadStatus.STARTING,
                start_time=datetime.now(),
                last_activity=datetime.now()
            )
            
            # Create and start thread
            thread = threading.Thread(
                target=self._file_monitor_worker,
                args=(thread_id,),
                name=f"Snake-FileMonitor-{thread_id}",
                daemon=True
            )
            
            thread_state.thread_object = thread
            thread_state.status = ThreadStatus.RUNNING
            
            with self.coordination_lock:
                self.active_threads[thread_id] = thread_state
            
            thread.start()
            self.threads_created += 1
            
            # Initialize worker metrics
            self.worker_metrics[thread_id] = WorkerMetrics(
                worker_id=thread_id,
                worker_type="thread",
                start_time=datetime.now()
            )
            
            await self.log_manager.log_system_event(
                "file_monitor_thread_started",
                {"thread_id": thread_id},
                worker_id="threading_manager"
            )
            
            return True
            
        except Exception as e:
            await self.log_manager.log_system_event(
                "file_monitor_thread_failed",
                {"error": str(e)},
                level="error",
                worker_id="threading_manager"
            )
            return False
    
    async def start_analysis_threads(self, count: int = 3) -> bool:
        """Start multiple code analysis threads"""
        try:
            success_count = 0
            
            for i in range(count):
                thread_id = f"analyzer_{i}_{uuid.uuid4().hex[:8]}"
                
                thread_state = ThreadState(
                    thread_id=thread_id,
                    name=f"CodeAnalyzer-{i}",
                    status=ThreadStatus.STARTING,
                    start_time=datetime.now(),
                    last_activity=datetime.now()
                )
                
                # Create and start thread
                thread = threading.Thread(
                    target=self._analysis_worker,
                    args=(thread_id,),
                    name=f"Snake-Analyzer-{thread_id}",
                    daemon=True
                )
                
                thread_state.thread_object = thread
                thread_state.status = ThreadStatus.RUNNING
                
                with self.coordination_lock:
                    self.active_threads[thread_id] = thread_state
                
                thread.start()
                self.threads_created += 1
                success_count += 1
                
                # Initialize worker metrics
                self.worker_metrics[thread_id] = WorkerMetrics(
                    worker_id=thread_id,
                    worker_type="thread",
                    start_time=datetime.now()
                )
            
            await self.log_manager.log_system_event(
                "analysis_threads_started",
                {"requested": count, "started": success_count},
                worker_id="threading_manager"
            )
            
            return success_count == count
            
        except Exception as e:
            await self.log_manager.log_system_event(
                "analysis_threads_failed",
                {"error": str(e)},
                level="error",
                worker_id="threading_manager"
            )
            return False
    
    async def start_communication_thread(self) -> bool:
        """Start RAVANA communication thread"""
        try:
            thread_id = f"communicator_{uuid.uuid4().hex[:8]}"
            
            thread_state = ThreadState(
                thread_id=thread_id,
                name="RavanaCommunicator",
                status=ThreadStatus.STARTING,
                start_time=datetime.now(),
                last_activity=datetime.now()
            )
            
            # Create and start thread
            thread = threading.Thread(
                target=self._communication_worker,
                args=(thread_id,),
                name=f"Snake-Communicator-{thread_id}",
                daemon=True
            )
            
            thread_state.thread_object = thread
            thread_state.status = ThreadStatus.RUNNING
            
            with self.coordination_lock:
                self.active_threads[thread_id] = thread_state
            
            thread.start()
            self.threads_created += 1
            
            # Initialize worker metrics
            self.worker_metrics[thread_id] = WorkerMetrics(
                worker_id=thread_id,
                worker_type="thread",
                start_time=datetime.now()
            )
            
            await self.log_manager.log_system_event(
                "communication_thread_started",
                {"thread_id": thread_id},
                worker_id="threading_manager"
            )
            
            return True
            
        except Exception as e:
            await self.log_manager.log_system_event(
                "communication_thread_failed",
                {"error": str(e)},
                level="error",
                worker_id="threading_manager"
            )
            return False
    
    async def start_performance_monitor_thread(self) -> bool:
        """Start performance monitoring thread"""
        try:
            thread_id = f"perf_monitor_{uuid.uuid4().hex[:8]}"
            
            thread_state = ThreadState(
                thread_id=thread_id,
                name="PerformanceMonitor",
                status=ThreadStatus.STARTING,
                start_time=datetime.now(),
                last_activity=datetime.now()
            )
            
            # Create and start thread
            thread = threading.Thread(
                target=self._performance_monitor_worker,
                args=(thread_id,),
                name=f"Snake-PerfMonitor-{thread_id}",
                daemon=True
            )
            
            thread_state.thread_object = thread
            thread_state.status = ThreadStatus.RUNNING
            
            with self.coordination_lock:
                self.active_threads[thread_id] = thread_state
            
            thread.start()
            self.threads_created += 1
            
            await self.log_manager.log_system_event(
                "performance_monitor_started",
                {"thread_id": thread_id},
                worker_id="threading_manager"
            )
            
            return True
            
        except Exception as e:
            await self.log_manager.log_system_event(
                "performance_monitor_failed",
                {"error": str(e)},
                level="error",
                worker_id="threading_manager"
            )
            return False
    
    def _file_monitor_worker(self, worker_id: str):
        """Worker thread for file monitoring"""
        while not self.shutdown_event.is_set():
            try:
                # Update thread state
                with self.coordination_lock:
                    if worker_id in self.active_threads:
                        self.active_threads[worker_id].update_activity("monitoring_files")
                
                # Get file change event from queue with timeout
                try:
                    file_event = self.file_change_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process file change
                start_time = time.time()
                
                if self.file_change_callback:
                    try:
                        self.file_change_callback(file_event)
                        
                        # Update metrics
                        processing_time = time.time() - start_time
                        if worker_id in self.worker_metrics:
                            self.worker_metrics[worker_id].record_task_completion(processing_time)
                        
                        # Update thread state
                        with self.coordination_lock:
                            if worker_id in self.active_threads:
                                self.active_threads[worker_id].increment_processed()
                        
                        self.tasks_processed += 1
                        
                    except Exception as e:
                        # Log callback error
                        asyncio.create_task(self.log_manager.log_system_event(
                            "file_change_callback_error",
                            {"error": str(e), "file_event": file_event.to_dict()},
                            level="error",
                            worker_id=worker_id
                        ))
                        
                        # Update error metrics
                        if worker_id in self.worker_metrics:
                            self.worker_metrics[worker_id].record_task_failure()
                        
                        with self.coordination_lock:
                            if worker_id in self.active_threads:
                                self.active_threads[worker_id].increment_error()
                
                self.file_change_queue.task_done()
                
            except Exception as e:
                # Log worker error
                asyncio.create_task(self.log_manager.log_system_event(
                    "file_monitor_worker_error",
                    {"error": str(e)},
                    level="error",
                    worker_id=worker_id
                ))
                
                with self.coordination_lock:
                    if worker_id in self.active_threads:
                        self.active_threads[worker_id].increment_error()
                        self.active_threads[worker_id].status = ThreadStatus.ERROR
                
                # Wait before retrying
                time.sleep(1.0)
        
        # Thread shutdown
        with self.coordination_lock:
            if worker_id in self.active_threads:
                self.active_threads[worker_id].status = ThreadStatus.STOPPED
    
    def _analysis_worker(self, worker_id: str):
        """Worker thread for code analysis"""
        while not self.shutdown_event.is_set():
            try:
                # Update thread state
                with self.coordination_lock:
                    if worker_id in self.active_threads:
                        self.active_threads[worker_id].update_activity("analyzing_code")
                
                # Get analysis task from queue with timeout
                try:
                    analysis_task = self.analysis_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process analysis task
                start_time = time.time()
                
                if self.analysis_callback:
                    try:
                        self.analysis_callback(analysis_task)
                        
                        # Update metrics
                        processing_time = time.time() - start_time
                        if worker_id in self.worker_metrics:
                            self.worker_metrics[worker_id].record_task_completion(processing_time)
                        
                        # Update thread state
                        with self.coordination_lock:
                            if worker_id in self.active_threads:
                                self.active_threads[worker_id].increment_processed()
                        
                        self.tasks_processed += 1
                        
                    except Exception as e:
                        # Log callback error
                        asyncio.create_task(self.log_manager.log_system_event(
                            "analysis_callback_error",
                            {"error": str(e), "task": analysis_task.to_dict()},
                            level="error",
                            worker_id=worker_id
                        ))
                        
                        # Update error metrics
                        if worker_id in self.worker_metrics:
                            self.worker_metrics[worker_id].record_task_failure()
                        
                        with self.coordination_lock:
                            if worker_id in self.active_threads:
                                self.active_threads[worker_id].increment_error()
                
                self.analysis_queue.task_done()
                
            except Exception as e:
                # Log worker error
                asyncio.create_task(self.log_manager.log_system_event(
                    "analysis_worker_error",
                    {"error": str(e)},
                    level="error",
                    worker_id=worker_id
                ))
                
                with self.coordination_lock:
                    if worker_id in self.active_threads:
                        self.active_threads[worker_id].increment_error()
                        self.active_threads[worker_id].status = ThreadStatus.ERROR
                
                # Wait before retrying
                time.sleep(1.0)
        
        # Thread shutdown
        with self.coordination_lock:
            if worker_id in self.active_threads:
                self.active_threads[worker_id].status = ThreadStatus.STOPPED
    
    def _communication_worker(self, worker_id: str):
        """Worker thread for RAVANA communication"""
        while not self.shutdown_event.is_set():
            try:
                # Update thread state
                with self.coordination_lock:
                    if worker_id in self.active_threads:
                        self.active_threads[worker_id].update_activity("communicating")
                
                # Get communication message from queue with timeout
                try:
                    comm_message = self.communication_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process communication message
                start_time = time.time()
                
                if self.communication_callback:
                    try:
                        self.communication_callback(comm_message)
                        
                        # Update metrics
                        processing_time = time.time() - start_time
                        if worker_id in self.worker_metrics:
                            self.worker_metrics[worker_id].record_task_completion(processing_time)
                        
                        # Update thread state
                        with self.coordination_lock:
                            if worker_id in self.active_threads:
                                self.active_threads[worker_id].increment_processed()
                        
                        self.tasks_processed += 1
                        
                    except Exception as e:
                        # Log callback error
                        asyncio.create_task(self.log_manager.log_system_event(
                            "communication_callback_error",
                            {"error": str(e), "message": comm_message.to_dict()},
                            level="error",
                            worker_id=worker_id
                        ))
                        
                        # Update error metrics
                        if worker_id in self.worker_metrics:
                            self.worker_metrics[worker_id].record_task_failure()
                        
                        with self.coordination_lock:
                            if worker_id in self.active_threads:
                                self.active_threads[worker_id].increment_error()
                
                self.communication_queue.task_done()
                
            except Exception as e:
                # Log worker error
                asyncio.create_task(self.log_manager.log_system_event(
                    "communication_worker_error",
                    {"error": str(e)},
                    level="error",
                    worker_id=worker_id
                ))
                
                with self.coordination_lock:
                    if worker_id in self.active_threads:
                        self.active_threads[worker_id].increment_error()
                        self.active_threads[worker_id].status = ThreadStatus.ERROR
                
                # Wait before retrying
                time.sleep(1.0)
        
        # Thread shutdown
        with self.coordination_lock:
            if worker_id in self.active_threads:
                self.active_threads[worker_id].status = ThreadStatus.STOPPED
    
    def _performance_monitor_worker(self, worker_id: str):
        """Worker thread for performance monitoring"""
        import psutil
        
        while not self.shutdown_event.is_set():
            try:
                # Update thread state
                with self.coordination_lock:
                    if worker_id in self.active_threads:
                        self.active_threads[worker_id].update_activity("monitoring_performance")
                
                # Collect system metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                memory_usage_mb = memory_info.used / (1024 * 1024)
                
                # Update worker metrics
                for metrics in self.worker_metrics.values():
                    metrics.add_resource_sample(cpu_usage, memory_usage_mb)
                
                # Log performance data periodically
                current_time = datetime.now()
                if (current_time.minute % 5 == 0 and current_time.second < 5):  # Every 5 minutes
                    asyncio.create_task(self.log_manager.log_system_event(
                        "performance_metrics",
                        {
                            "cpu_usage": cpu_usage,
                            "memory_usage_mb": memory_usage_mb,
                            "active_threads": len(self.active_threads),
                            "tasks_processed": self.tasks_processed,
                            "queue_sizes": {
                                "file_changes": self.file_change_queue.qsize(),
                                "analysis": self.analysis_queue.qsize(),
                                "communication": self.communication_queue.qsize()
                            }
                        },
                        worker_id=worker_id
                    ))
                
                # Sleep for monitoring interval
                time.sleep(10.0)  # Monitor every 10 seconds
                
            except Exception as e:
                # Log monitoring error
                asyncio.create_task(self.log_manager.log_system_event(
                    "performance_monitor_error",
                    {"error": str(e)},
                    level="error",
                    worker_id=worker_id
                ))
                
                time.sleep(10.0)
        
        # Thread shutdown
        with self.coordination_lock:
            if worker_id in self.active_threads:
                self.active_threads[worker_id].status = ThreadStatus.STOPPED
    
    def set_callbacks(self, 
                     file_change_callback: Optional[Callable] = None,
                     analysis_callback: Optional[Callable] = None,
                     communication_callback: Optional[Callable] = None):
        """Set callbacks for processing tasks"""
        self.file_change_callback = file_change_callback
        self.analysis_callback = analysis_callback
        self.communication_callback = communication_callback
    
    def queue_file_change(self, file_event: FileChangeEvent) -> bool:
        """Queue a file change event for processing"""
        try:
            self.file_change_queue.put_nowait(file_event)
            return True
        except queue.Full:
            asyncio.create_task(self.log_manager.log_system_event(
                "file_change_queue_full",
                {"event": file_event.to_dict()},
                level="warning",
                worker_id="threading_manager"
            ))
            return False
    
    def queue_analysis_task(self, analysis_task: AnalysisTask) -> bool:
        """Queue an analysis task for processing"""
        try:
            self.analysis_queue.put_nowait(analysis_task)
            return True
        except queue.Full:
            asyncio.create_task(self.log_manager.log_system_event(
                "analysis_queue_full",
                {"task": analysis_task.to_dict()},
                level="warning",
                worker_id="threading_manager"
            ))
            return False
    
    def queue_communication_message(self, comm_message: CommunicationMessage) -> bool:
        """Queue a communication message for processing"""
        try:
            self.communication_queue.put_nowait(comm_message)
            return True
        except queue.Full:
            asyncio.create_task(self.log_manager.log_system_event(
                "communication_queue_full",
                {"message": comm_message.to_dict()},
                level="warning",
                worker_id="threading_manager"
            ))
            return False
    
    def get_thread_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active threads"""
        with self.coordination_lock:
            return {
                thread_id: thread_state.to_dict()
                for thread_id, thread_state in self.active_threads.items()
            }
    
    def get_queue_status(self) -> Dict[str, int]:
        """Get status of all queues"""
        return {
            "file_changes": self.file_change_queue.qsize(),
            "analysis": self.analysis_queue.qsize(),
            "communication": self.communication_queue.qsize()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all workers"""
        return {
            worker_id: metrics.to_dict()
            for worker_id, metrics in self.worker_metrics.items()
        }
    
    async def shutdown(self, timeout: float = 30.0) -> bool:
        """Shutdown all threads gracefully"""
        try:
            await self.log_manager.log_system_event(
                "threading_manager_shutdown_start",
                {"active_threads": len(self.active_threads)},
                worker_id="threading_manager"
            )
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Wait for threads to finish
            shutdown_start = time.time()
            
            while time.time() - shutdown_start < timeout:
                with self.coordination_lock:
                    running_threads = [
                        t for t in self.active_threads.values()
                        if t.status in [ThreadStatus.RUNNING, ThreadStatus.STARTING]
                    ]
                
                if not running_threads:
                    break
                
                time.sleep(0.5)
            
            # Force shutdown thread pool
            self.thread_pool.shutdown(wait=True, timeout=10.0)
            
            await self.log_manager.log_system_event(
                "threading_manager_shutdown_complete",
                {"shutdown_time": time.time() - shutdown_start},
                worker_id="threading_manager"
            )
            
            return True
            
        except Exception as e:
            await self.log_manager.log_system_event(
                "threading_manager_shutdown_error",
                {"error": str(e)},
                level="error",
                worker_id="threading_manager"
            )
            return False