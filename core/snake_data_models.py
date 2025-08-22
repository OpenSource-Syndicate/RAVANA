"""
Snake Agent Data Models

This module defines data models for threading, multiprocessing, and state management
in the enhanced Snake Agent system.
"""

import asyncio
import threading
import multiprocessing
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field, asdict
from enum import Enum


class ThreadStatus(Enum):
    """Thread status enumeration"""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ProcessStatus(Enum):
    """Process status enumeration"""
    STARTING = "starting"
    ACTIVE = "active"
    IDLE = "idle"
    STOPPING = "stopping"
    TERMINATED = "terminated"
    ERROR = "error"


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    BACKGROUND = 1


class TaskStatus(Enum):
    """Task execution status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ThreadState:
    """State tracking for individual threads"""
    thread_id: str
    name: str
    status: ThreadStatus
    start_time: datetime
    last_activity: datetime
    processed_items: int = 0
    error_count: int = 0
    current_task: Optional[str] = None
    thread_object: Optional[threading.Thread] = field(default=None, repr=False)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['start_time'] = self.start_time.isoformat()
        data['last_activity'] = self.last_activity.isoformat()
        # Remove non-serializable thread object
        data.pop('thread_object', None)
        return data
    
    def update_activity(self, task: Optional[str] = None):
        """Update last activity time and current task"""
        self.last_activity = datetime.now()
        if task:
            self.current_task = task
    
    def increment_processed(self):
        """Increment processed items counter"""
        self.processed_items += 1
        self.update_activity()
    
    def increment_error(self):
        """Increment error counter"""
        self.error_count += 1
        self.update_activity()


@dataclass
class ProcessState:
    """State tracking for worker processes"""
    process_id: int
    name: str
    status: ProcessStatus
    start_time: datetime
    last_heartbeat: datetime
    tasks_completed: int = 0
    tasks_failed: int = 0
    queue_size: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    process_object: Optional[multiprocessing.Process] = field(default=None, repr=False)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['status'] = self.status.value
        data['start_time'] = self.start_time.isoformat()
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        # Remove non-serializable process object
        data.pop('process_object', None)
        return data
    
    def update_heartbeat(self):
        """Update last heartbeat time"""
        self.last_heartbeat = datetime.now()
    
    def increment_completed(self):
        """Increment completed tasks counter"""
        self.tasks_completed += 1
        self.update_heartbeat()
    
    def increment_failed(self):
        """Increment failed tasks counter"""
        self.tasks_failed += 1
        self.update_heartbeat()


@dataclass
class TaskInfo:
    """Information about a task in the system"""
    task_id: str
    task_type: str
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    data: Dict[str, Any] = field(default_factory=dict)
    worker_id: Optional[str] = None
    error_message: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data
    
    def mark_started(self, worker_id: str):
        """Mark task as started"""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
        self.worker_id = worker_id
    
    def mark_completed(self):
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def mark_failed(self, error_message: str):
        """Mark task as failed"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message
        self.retries += 1
    
    def can_retry(self) -> bool:
        """Check if task can be retried"""
        return self.retries < self.max_retries and self.status == TaskStatus.FAILED


@dataclass
class FileChangeEvent:
    """Represents a file system change event"""
    event_id: str
    event_type: str  # created, modified, deleted, moved
    file_path: str
    absolute_path: str
    timestamp: datetime
    file_hash: Optional[str] = None
    old_hash: Optional[str] = None
    processed: bool = False
    worker_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class AnalysisTask:
    """Represents a code analysis task"""
    task_id: str
    file_path: str
    analysis_type: str  # file_change, periodic, deep_analysis, security_scan
    priority: TaskPriority
    created_at: datetime
    file_content: Optional[str] = None
    change_context: Optional[Dict[str, Any]] = None
    requirements: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class ExperimentTask:
    """Represents a code experiment task"""
    task_id: str
    experiment_type: str  # improvement, optimization, refactoring, security_fix
    file_path: str
    hypothesis: str
    proposed_changes: Dict[str, Any]
    safety_requirements: Dict[str, Any]
    priority: TaskPriority
    created_at: datetime
    estimated_duration: float = 300.0  # 5 minutes default
    requires_approval: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class ImprovementProposal:
    """Represents a code improvement proposal"""
    proposal_id: str
    title: str
    description: str
    file_path: str
    improvement_type: str  # performance, security, maintainability, bug_fix
    original_code: str
    proposed_code: str
    justification: str
    impact_assessment: Dict[str, Any]
    safety_score: float
    confidence_score: float
    priority: TaskPriority
    created_at: datetime
    status: str = "proposed"  # proposed, approved, rejected, applied
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class CommunicationMessage:
    """Represents a communication message with RAVANA"""
    message_id: str
    direction: str  # outbound, inbound
    message_type: str  # improvement_proposal, experiment_result, status_update, query
    content: Dict[str, Any]
    priority: TaskPriority
    created_at: datetime
    sent_at: Optional[datetime] = None
    response_received_at: Optional[datetime] = None
    status: str = "pending"  # pending, sent, acknowledged, failed
    retries: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        data['sent_at'] = self.sent_at.isoformat() if self.sent_at else None
        data['response_received_at'] = self.response_received_at.isoformat() if self.response_received_at else None
        return data


@dataclass
class WorkerMetrics:
    """Performance metrics for workers"""
    worker_id: str
    worker_type: str  # thread, process
    start_time: datetime
    tasks_processed: int = 0
    tasks_failed: int = 0
    average_processing_time: float = 0.0
    last_activity: Optional[datetime] = None
    cpu_usage_samples: List[float] = field(default_factory=list)
    memory_usage_samples: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        data['last_activity'] = self.last_activity.isoformat() if self.last_activity else None
        return data
    
    def record_task_completion(self, processing_time: float):
        """Record completion of a task"""
        self.tasks_processed += 1
        self.last_activity = datetime.now()
        
        # Update average processing time
        if self.average_processing_time == 0.0:
            self.average_processing_time = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_processing_time = (
                alpha * processing_time + 
                (1 - alpha) * self.average_processing_time
            )
    
    def record_task_failure(self):
        """Record failure of a task"""
        self.tasks_failed += 1
        self.last_activity = datetime.now()
    
    def add_resource_sample(self, cpu_usage: float, memory_usage: float):
        """Add resource usage sample"""
        # Keep only last 100 samples
        if len(self.cpu_usage_samples) >= 100:
            self.cpu_usage_samples.pop(0)
        if len(self.memory_usage_samples) >= 100:
            self.memory_usage_samples.pop(0)
        
        self.cpu_usage_samples.append(cpu_usage)
        self.memory_usage_samples.append(memory_usage)


@dataclass
class SystemHealthMetrics:
    """Overall system health metrics"""
    timestamp: datetime
    active_threads: int
    active_processes: int
    queued_tasks: int
    completed_tasks: int
    failed_tasks: int
    average_queue_wait_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    file_changes_per_hour: float
    experiments_per_hour: float
    improvements_applied: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SnakeAgentConfiguration:
    """Configuration for Snake Agent threading and multiprocessing"""
    max_threads: int = 8
    max_processes: int = 4
    analysis_threads: int = 3
    file_monitor_interval: float = 2.0  # seconds
    process_heartbeat_interval: float = 10.0  # seconds
    max_queue_size: int = 1000
    task_timeout: float = 300.0  # 5 minutes
    cleanup_interval: float = 3600.0  # 1 hour
    log_level: str = "INFO"
    enable_performance_monitoring: bool = True
    auto_recovery: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def validate(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        if self.max_threads < 1:
            issues.append("max_threads must be at least 1")
        
        if self.max_processes < 1:
            issues.append("max_processes must be at least 1")
        
        if self.analysis_threads > self.max_threads:
            issues.append("analysis_threads cannot exceed max_threads")
        
        if self.file_monitor_interval < 0.1:
            issues.append("file_monitor_interval too low (minimum 0.1 seconds)")
        
        if self.task_timeout < 10.0:
            issues.append("task_timeout too low (minimum 10 seconds)")
        
        return issues