"""
Snake Process Manager

This module manages worker processes for CPU-intensive tasks including
code experiments, deep analysis, and improvement processing.
"""

import asyncio
import multiprocessing
import queue
import time
import uuid
import os
from datetime import datetime
from typing import Dict, Optional, Callable, Any
from concurrent.futures import ProcessPoolExecutor

from core.snake_data_models import (
    ProcessState, ProcessStatus, SnakeAgentConfiguration
)
from core.snake_log_manager import SnakeLogManager


class SnakeProcessManager:
    """Manages worker processes for CPU-intensive Snake Agent tasks"""
    
    def __init__(self, config: SnakeAgentConfiguration, log_manager: SnakeLogManager):
        self.config = config
        self.log_manager = log_manager
        
        # Process management
        self.active_processes: Dict[int, ProcessState] = {}
        self.process_pool = ProcessPoolExecutor(max_workers=config.max_processes)
        
        # Inter-process communication
        self.task_queue = multiprocessing.Queue(maxsize=config.max_queue_size)
        self.result_queue = multiprocessing.Queue()
        self.shutdown_event = multiprocessing.Event()
        
        # Callbacks
        self.experiment_callback: Optional[Callable] = None
        self.analysis_callback: Optional[Callable] = None
        self.improvement_callback: Optional[Callable] = None
        
        # Metrics
        self.tasks_distributed = 0
        self.results_collected = 0
    
    async def initialize(self) -> bool:
        """Initialize the process manager"""
        try:
            await self.log_manager.log_system_event(
                "process_manager_init",
                {"max_processes": self.config.max_processes},
                worker_id="process_manager"
            )
            await self.start_result_collector()
            return True
        except Exception as e:
            await self.log_manager.log_system_event(
                "process_manager_init_failed",
                {"error": str(e)},
                level="error",
                worker_id="process_manager"
            )
            return False
    
    async def start_all_processes(self) -> bool:
        """Start all worker processes"""
        try:
            success = (
                await self.start_experiment_processes(2) and
                await self.start_analysis_processes(1) and
                await self.start_improvement_process()
            )
            
            await self.log_manager.log_system_event(
                "all_processes_started",
                {"success": success, "active_processes": len(self.active_processes)},
                worker_id="process_manager"
            )
            return success
        except Exception as e:
            await self.log_manager.log_system_event(
                "start_processes_failed",
                {"error": str(e)},
                level="error",
                worker_id="process_manager"
            )
            return False
    
    async def start_experiment_processes(self, count: int = 2) -> bool:
        """Start experiment worker processes"""
        success_count = 0
        for i in range(count):
            if self._start_worker_process(f"experimenter_{i}", self._experiment_worker):
                success_count += 1
        return success_count == count
    
    async def start_analysis_processes(self, count: int = 1) -> bool:
        """Start analysis worker processes"""
        success_count = 0
        for i in range(count):
            if self._start_worker_process(f"analyzer_{i}", self._analysis_worker):
                success_count += 1
        return success_count == count
    
    async def start_improvement_process(self) -> bool:
        """Start improvement worker process"""
        return self._start_worker_process("improvement", self._improvement_worker)
    
    def _start_worker_process(self, name: str, target_func: Callable) -> bool:
        """Start a worker process"""
        try:
            process = multiprocessing.Process(
                target=target_func,
                args=(name, self.task_queue, self.result_queue, self.shutdown_event),
                name=f"Snake-{name}",
                daemon=False
            )
            
            process.start()
            
            process_state = ProcessState(
                process_id=process.pid,
                name=name,
                status=ProcessStatus.ACTIVE,
                start_time=datetime.now(),
                last_heartbeat=datetime.now(),
                process_object=process
            )
            
            self.active_processes[process.pid] = process_state
            return True
        except Exception:
            return False
    
    async def start_result_collector(self):
        """Start background task to collect results"""
        asyncio.create_task(self._result_collector_loop())
    
    async def _result_collector_loop(self):
        """Background loop to collect results from processes"""
        while not self.shutdown_event.is_set():
            try:
                result = self.result_queue.get(timeout=1.0)
                await self._process_result(result)
                self.results_collected += 1
            except:
                pass
            await asyncio.sleep(0.1)
    
    async def _process_result(self, result: Dict[str, Any]):
        """Process result from worker process"""
        try:
            result_type = result.get("type")
            
            if result_type == "experiment" and self.experiment_callback:
                await self.experiment_callback(result)
            elif result_type == "analysis" and self.analysis_callback:
                await self.analysis_callback(result)
            elif result_type == "improvement" and self.improvement_callback:
                await self.improvement_callback(result)
                
        except Exception as e:
            await self.log_manager.log_system_event(
                "result_processing_error",
                {"error": str(e)},
                level="error",
                worker_id="process_manager"
            )
    
    @staticmethod
    def _experiment_worker(name: str, task_queue, result_queue, shutdown_event):
        """Experiment worker process main function"""
        import time
        process_id = os.getpid()
        
        while not shutdown_event.is_set():
            try:
                task = task_queue.get(timeout=5.0)
                if task.get("type") == "experiment":
                    start_time = time.time()
                    # Simulate experiment processing
                    time.sleep(1.0)  # Placeholder for actual experiment
                    
                    result_queue.put({
                        "type": "experiment",
                        "process_id": process_id,
                        "task_id": task.get("task_id"),
                        "result": {"status": "completed"},
                        "success": True,
                        "processing_time": time.time() - start_time,
                        "timestamp": datetime.now().isoformat()
                    })
            except:
                continue
    
    @staticmethod
    def _analysis_worker(name: str, task_queue, result_queue, shutdown_event):
        """Analysis worker process main function"""
        import time
        process_id = os.getpid()
        
        while not shutdown_event.is_set():
            try:
                task = task_queue.get(timeout=5.0)
                if task.get("type") == "analysis":
                    start_time = time.time()
                    # Simulate analysis processing
                    time.sleep(2.0)  # Placeholder for actual analysis
                    
                    result_queue.put({
                        "type": "analysis",
                        "process_id": process_id,
                        "task_id": task.get("task_id"),
                        "result": {"status": "analyzed"},
                        "success": True,
                        "processing_time": time.time() - start_time,
                        "timestamp": datetime.now().isoformat()
                    })
            except:
                continue
    
    @staticmethod
    def _improvement_worker(name: str, task_queue, result_queue, shutdown_event):
        """Improvement worker process main function"""
        import time
        process_id = os.getpid()
        
        while not shutdown_event.is_set():
            try:
                task = task_queue.get(timeout=5.0)
                if task.get("type") == "improvement":
                    start_time = time.time()
                    # Simulate improvement processing
                    time.sleep(1.5)  # Placeholder for actual improvement
                    
                    result_queue.put({
                        "type": "improvement",
                        "process_id": process_id,
                        "task_id": task.get("task_id"),
                        "result": {"status": "improved"},
                        "success": True,
                        "processing_time": time.time() - start_time,
                        "timestamp": datetime.now().isoformat()
                    })
            except:
                continue
    
    def set_callbacks(self, experiment_callback=None, analysis_callback=None, improvement_callback=None):
        """Set callbacks for processing results"""
        self.experiment_callback = experiment_callback
        self.analysis_callback = analysis_callback
        self.improvement_callback = improvement_callback
    
    def distribute_task(self, task: Dict[str, Any]) -> bool:
        """Distribute task to worker processes"""
        try:
            self.task_queue.put_nowait(task)
            self.tasks_distributed += 1
            return True
        except:
            return False
    
    def get_process_status(self) -> Dict[int, Dict[str, Any]]:
        """Get status of all processes"""
        return {
            pid: state.to_dict()
            for pid, state in self.active_processes.items()
        }
    
    def get_queue_status(self) -> Dict[str, int]:
        """Get queue status"""
        return {
            "task_queue": self.task_queue.qsize(),
            "result_queue": self.result_queue.qsize()
        }
    
    async def shutdown(self, timeout: float = 30.0) -> bool:
        """Shutdown all processes gracefully"""
        try:
            await self.log_manager.log_system_event(
                "process_manager_shutdown",
                {"active_processes": len(self.active_processes)},
                worker_id="process_manager"
            )
            
            self.shutdown_event.set()
            
            # Wait for processes to finish
            for process_state in self.active_processes.values():
                if process_state.process_object:
                    process_state.process_object.join(timeout=timeout/len(self.active_processes))
                    if process_state.process_object.is_alive():
                        process_state.process_object.terminate()
            
            self.process_pool.shutdown(wait=True, timeout=10.0)
            return True
            
        except Exception as e:
            await self.log_manager.log_system_event(
                "shutdown_error",
                {"error": str(e)},
                level="error",
                worker_id="process_manager"
            )
            return False