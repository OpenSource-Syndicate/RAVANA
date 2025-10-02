"""
Snake Log Manager

This module manages separate log files for different Snake Agent activities
including improvements, experiments, analysis, communication, and system events.
"""

import logging
import json
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from logging.handlers import RotatingFileHandler


@dataclass
class ImprovementRecord:
    """Data model for improvement records"""
    id: str
    type: str  # code_fix, performance, architecture, security
    description: str
    file_path: str
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    impact_score: float
    safety_score: float
    timestamp: datetime
    worker_id: str
    status: str  # proposed, tested, applied, rejected

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ExperimentRecord:
    """Data model for experiment records"""
    id: str
    file_path: str
    experiment_type: str
    description: str
    hypothesis: str
    methodology: str
    result: Dict[str, Any]
    success: bool
    safety_score: float
    duration: float
    timestamp: datetime
    worker_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class AnalysisRecord:
    """Data model for analysis records"""
    id: str
    file_path: str
    analysis_type: str  # file_change, periodic, deep_analysis
    findings: Dict[str, Any]
    suggestions: List[Dict[str, Any]]
    priority: str  # high, medium, low
    confidence: float
    processing_time: float
    timestamp: datetime
    worker_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class CommunicationRecord:
    """Data model for communication records"""
    id: str
    direction: str  # outbound, inbound
    message_type: str
    content: Dict[str, Any]
    priority: str
    status: str  # sent, received, failed, pending
    response_time: Optional[float]
    timestamp: datetime
    worker_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class SnakeLogManager:
    """Manages separate log files for different Snake Agent activities"""

    def __init__(self, log_directory: str = "snake_logs"):
        self.log_dir = Path(log_directory)
        self.log_dir.mkdir(exist_ok=True)

        # Configure logging formatters
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        self.json_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # Create separate loggers for different activities
        self.improvement_logger = self._create_logger("improvement")
        self.experiment_logger = self._create_logger("experiments")
        self.analysis_logger = self._create_logger("analysis")
        self.communication_logger = self._create_logger("communication")
        self.system_logger = self._create_logger("system")
        self.error_logger = self._create_error_logger("snake_errors")  # New error logger
        self.interactions_logger = self._create_json_logger("interactions")  # New interactions logger

        # Thread-safe logging queue and worker
        self.log_queue = queue.Queue()
        self.log_worker_thread = None
        self.worker_running = False
        self.shutdown_event = threading.Event()

        # Performance metrics
        self.logs_processed = 0
        self.queue_high_water_mark = 0

    def _create_logger(self, name: str) -> logging.Logger:
        """Create specialized logger with separate file and rotation"""
        logger = logging.getLogger(f"snake.{name}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Clear any existing handlers
        logger.handlers.clear()

        # Create rotating file handler (10MB max, 5 backups)
        log_file = self.log_dir / f"{name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(self.formatter)
        logger.addHandler(file_handler)

        # Create JSON file handler for structured data
        json_log_file = self.log_dir / f"{name}_structured.json"
        json_handler = RotatingFileHandler(
            json_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        json_handler.setFormatter(self.json_formatter)
        logger.addHandler(json_handler)

        return logger

    def _create_error_logger(self, name: str) -> logging.Logger:
        """Create specialized error logger with traceback support"""
        logger = logging.getLogger(f"snake.{name}")
        logger.setLevel(logging.ERROR)
        logger.propagate = False

        # Clear any existing handlers
        logger.handlers.clear()

        from core.config import Config
        # Use configured values or defaults
        max_bytes = getattr(Config, 'SNAKE_LOG_MAX_FILE_SIZE', 10 * 1024 * 1024)
        backup_count = getattr(Config, 'SNAKE_LOG_BACKUP_COUNT', 5)
        
        # Create rotating file handler for errors with traceback
        log_file = self.log_dir / f"{name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # Use a formatter that includes traceback information
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(exc_info)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(error_formatter)
        logger.addHandler(file_handler)

        return logger

    def _create_json_logger(self, name: str) -> logging.Logger:
        """Create logger specifically for JSON structured logging of interactions"""
        logger = logging.getLogger(f"snake.{name}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Clear any existing handlers
        logger.handlers.clear()

        from core.config import Config
        # Use configured values or defaults
        max_bytes = getattr(Config, 'SNAKE_LOG_MAX_FILE_SIZE', 10 * 1024 * 1024)
        backup_count = getattr(Config, 'SNAKE_LOG_BACKUP_COUNT', 5)
        
        # Create JSON file handler for structured data
        json_log_file = self.log_dir / f"{name}.json"
        json_handler = RotatingFileHandler(
            json_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # Create a JSON formatter
        json_handler.setFormatter(self.json_formatter)
        logger.addHandler(json_handler)

        return logger

    def start_log_processor(self):
        """Start background thread for processing logs"""
        if self.log_worker_thread and self.log_worker_thread.is_alive():
            return

        self.worker_running = True
        self.shutdown_event.clear()
        self.log_worker_thread = threading.Thread(
            target=self._log_processor_worker,
            daemon=True,
            name="SnakeLogProcessor"
        )
        self.log_worker_thread.start()
        logging.getLogger(__name__).info("Snake Log Manager started")

    def stop_log_processor(self, timeout: float = 5.0):
        """Stop background log processor"""
        if not self.worker_running:
            return

        self.worker_running = False
        self.shutdown_event.set()

        # Add sentinel value to queue to wake up worker
        self.log_queue.put(None)

        if self.log_worker_thread and self.log_worker_thread.is_alive():
            self.log_worker_thread.join(timeout=timeout)

        # Close all file handlers to release file locks
        self._close_all_handlers()

        logging.getLogger(__name__).info("Snake Log Manager stopped")

    def _close_all_handlers(self):
        """Close all file handlers to prevent file locking issues"""
        try:
            for logger in [self.improvement_logger, self.experiment_logger,
                           self.analysis_logger, self.communication_logger, self.system_logger]:
                if logger:
                    # Close and remove all handlers
                    for handler in logger.handlers[:]:
                        handler.close()
                        logger.removeHandler(handler)
        except Exception as e:
            print(f"Error closing handlers: {e}")

    def _log_processor_worker(self):
        """Background worker thread for processing log entries"""
        while self.worker_running:
            try:
                # Get log entry from queue with timeout
                try:
                    log_entry = self.log_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Check for shutdown sentinel
                if log_entry is None:
                    break

                # Process log entry
                self._process_log_entry(log_entry)
                self.logs_processed += 1

                # Update queue metrics
                current_size = self.log_queue.qsize()
                if current_size > self.queue_high_water_mark:
                    self.queue_high_water_mark = current_size

            except Exception as e:
                # Use basic logging to avoid recursion
                print(f"Error in log processor: {e}")

    def _process_log_entry(self, log_entry: Dict[str, Any]):
        """Process a single log entry"""
        try:
            log_type = log_entry.get("type")
            logger = self._get_logger_for_type(log_type)

            if logger:
                # Log structured data as JSON
                json_data = json.dumps(
                    log_entry, default=str, ensure_ascii=False)
                logger.info(json_data)
                self.system_logger.info(f"Processing log entry: {json_data}")

        except Exception as e:
            # Fallback to system logger
            self.system_logger.error(f"Failed to process log entry: {e}")

    def _get_logger_for_type(self, log_type: str) -> Optional[logging.Logger]:
        """Get appropriate logger for log type"""
        logger_map = {
            "improvement": self.improvement_logger,
            "experiment": self.experiment_logger,
            "analysis": self.analysis_logger,
            "communication": self.communication_logger,
            "system": self.system_logger
        }
        return logger_map.get(log_type)

    async def log_improvement(self, record: ImprovementRecord):
        """Log improvement activity to dedicated file"""
        log_entry = {
            "type": "improvement",
            "record": record.to_dict(),
            "logged_at": datetime.now().isoformat()
        }
        self.log_queue.put(log_entry)

    async def log_experiment(self, record: ExperimentRecord):
        """Log experiment details to dedicated file"""
        log_entry = {
            "type": "experiment",
            "record": record.to_dict(),
            "logged_at": datetime.now().isoformat()
        }
        self.log_queue.put(log_entry)

    async def log_analysis(self, record: AnalysisRecord):
        """Log code analysis results to dedicated file"""
        log_entry = {
            "type": "analysis",
            "record": record.to_dict(),
            "logged_at": datetime.now().isoformat()
        }
        self.log_queue.put(log_entry)

    async def log_communication(self, record: CommunicationRecord):
        """Log communication activity to dedicated file"""
        log_entry = {
            "type": "communication",
            "record": record.to_dict(),
            "logged_at": datetime.now().isoformat()
        }
        self.log_queue.put(log_entry)

    async def log_system_event(self, event_type: str, data: Dict[str, Any],
                               level: str = "info", worker_id: str = "system"):
        """Log system events to dedicated file"""
        log_entry = {
            "type": "system",
            "event_type": event_type,
            "data": data,
            "level": level,
            "worker_id": worker_id,
            "logged_at": datetime.now().isoformat()
        }
        self.log_queue.put(log_entry)

    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            "logs_processed": self.logs_processed,
            "queue_size": self.log_queue.qsize(),
            "queue_high_water_mark": self.queue_high_water_mark,
            "worker_running": self.worker_running,
            "log_files": [
                str(f) for f in self.log_dir.glob("*.log")
            ]
        }

    def get_recent_logs(self, log_type: str, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent logs of specified type"""
        try:
            log_file = self.log_dir / f"{log_type}_structured.json"
            if not log_file.exists():
                return []

            recent_logs = []
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Get last 'count' lines
                for line in lines[-count:]:
                    try:
                        # Extract JSON from log line
                        # Format: timestamp - level - json_data
                        parts = line.strip().split(' - ', 2)
                        if len(parts) >= 3:
                            json_data = json.loads(parts[2])
                            recent_logs.append(json_data)
                    except json.JSONDecodeError:
                        continue

            return recent_logs

        except Exception as e:
            self.system_logger.error(f"Error getting recent logs: {e}")
            return []

    def log_error_with_traceback(self, error: Exception, context: str = "", extra_data: Dict[str, Any] = None):
        """Log error with full traceback information"""
        try:
            import traceback
            error_info = {
                "timestamp": datetime.now().isoformat(),
                "context": context,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exception(type(error), error, error.__traceback__)
            }
            
            if extra_data:
                error_info["extra_data"] = extra_data

            # Log to error file with traceback
            self.error_logger.error(
                f"{context} - {str(error)}", 
                exc_info=True  # This will include the full traceback
            )

            # Also add to the JSON interactions log for comprehensive tracking
            json_error_info = {
                "type": "error",
                "data": error_info
            }
            self.interactions_logger.info(json.dumps(json_error_info, default=str))
        except Exception as e:
            print(f"Error in error logging system: {e}")

    def log_interaction(self, interaction_type: str, prompt: str, response: str = None, 
                       metadata: Dict[str, Any] = None):
        """Log LLM interactions (prompts and responses)"""
        try:
            interaction_data = {
                "timestamp": datetime.now().isoformat(),
                "interaction_type": interaction_type,
                "prompt": prompt,
                "response": response,
                "metadata": metadata or {}
            }

            json_interaction = {
                "type": "interaction",
                "data": interaction_data
            }
            self.interactions_logger.info(json.dumps(json_interaction, default=str))
        except Exception as e:
            self.system_logger.error(f"Error logging interaction: {e}")

    def log_detailed_event(self, event_type: str, data: Dict[str, Any], 
                          metadata: Dict[str, Any] = None):
        """Log detailed system events with full context"""
        try:
            event_data = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "data": data,
                "metadata": metadata or {}
            }

            json_event = {
                "type": "detailed_event",
                "data": event_data
            }
            self.interactions_logger.info(json.dumps(json_event, default=str))
        except Exception as e:
            self.system_logger.error(f"Error logging detailed event: {e}")

    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Cleanup old log files"""
        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)

            for log_file in self.log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    self.system_logger.info(
                        f"Cleaned up old log file: {log_file}")

        except Exception as e:
            self.system_logger.error(f"Error cleaning up logs: {e}")
