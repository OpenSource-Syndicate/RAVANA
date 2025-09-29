"""
Continuous File Monitor

This module implements continuous file monitoring for the RAVANA codebase
using threading to watch for file changes in real-time.
"""

import threading
import time
import hashlib
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from core.snake_data_models import FileChangeEvent, SnakeAgentConfiguration
from core.snake_log_manager import SnakeLogManager


@dataclass
class FileMetadata:
    """Metadata for tracked files"""
    path: str
    hash: str
    size: int
    modified_time: float
    last_analyzed: Optional[datetime] = None
    analysis_count: int = 0


class SnakeFileEventHandler(FileSystemEventHandler):
    """Custom file system event handler for Snake Agent"""

    def __init__(self, file_monitor):
        super().__init__()
        self.file_monitor = file_monitor

    def on_modified(self, event):
        if not event.is_directory:
            self.file_monitor._handle_file_event("modified", event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self.file_monitor._handle_file_event("created", event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self.file_monitor._handle_file_event("deleted", event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self.file_monitor._handle_file_event(
                "moved", event.dest_path, event.src_path)


class ContinuousFileMonitor:
    """Continuously monitors RAVANA files for changes using threading"""

    def __init__(self, snake_agent, config: SnakeAgentConfiguration, log_manager: SnakeLogManager):
        self.snake_agent = snake_agent
        self.config = config
        self.log_manager = log_manager

        # File monitoring configuration
        self.root_path = Path(os.getcwd())
        self.monitored_extensions = {'.py', '.json',
                                     '.md', '.txt', '.yml', '.yaml', '.toml'}
        self.excluded_dirs = {'__pycache__', '.git',
                              '.venv', 'node_modules', '.qoder', 'snake_logs'}
        self.excluded_files = {'snake_agent_state.json'}

        # File tracking
        self.tracked_files: Dict[str, FileMetadata] = {}
        self.file_lock = threading.Lock()

        # Threading components
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.watchdog_observer: Optional[Observer] = None
        self.event_handler: Optional[SnakeFileEventHandler] = None

        # Change processing
        self.change_queue: List[FileChangeEvent] = []
        self.queue_lock = threading.Lock()
        self.processing_thread: Optional[threading.Thread] = None

        # Callbacks
        self.change_callback: Optional[Callable] = None

        # Performance metrics
        self.files_scanned = 0
        self.changes_detected = 0
        self.events_processed = 0
        self.scan_duration_total = 0.0

        # Control flags
        self.shutdown_event = threading.Event()
        self.loop = snake_agent.loop

    async def initialize(self) -> bool:
        """Initialize the file monitor"""
        try:
            await self.log_manager.log_system_event(
                "file_monitor_init",
                {
                    "root_path": str(self.root_path),
                    "monitored_extensions": list(self.monitored_extensions),
                    "excluded_dirs": list(self.excluded_dirs)
                },
                worker_id="file_monitor"
            )

            # Perform initial scan
            await self._initial_file_scan()

            # Setup watchdog observer
            self.event_handler = SnakeFileEventHandler(self)
            self.watchdog_observer = Observer()
            self.watchdog_observer.schedule(
                self.event_handler,
                str(self.root_path),
                recursive=True
            )

            return True

        except Exception as e:
            await self.log_manager.log_system_event(
                "file_monitor_init_failed",
                {"error": str(e)},
                level="error",
                worker_id="file_monitor"
            )
            return False

    async def start_monitoring(self) -> bool:
        """Start continuous file monitoring"""
        try:
            if self.monitoring_active:
                return True

            self.monitoring_active = True
            self.shutdown_event.clear()

            # Start watchdog observer
            self.watchdog_observer.start()

            # Start monitor thread for periodic scans
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="Snake-FileMonitor",
                daemon=True
            )
            self.monitor_thread.start()

            # Start event processing thread
            self.processing_thread = threading.Thread(
                target=self._event_processing_loop,
                name="Snake-EventProcessor",
                daemon=True
            )
            self.processing_thread.start()

            await self.log_manager.log_system_event(
                "file_monitor_started",
                {"monitoring_active": True},
                worker_id="file_monitor"
            )

            return True

        except Exception as e:
            await self.log_manager.log_system_event(
                "file_monitor_start_failed",
                {"error": str(e)},
                level="error",
                worker_id="file_monitor"
            )
            return False

    async def _initial_file_scan(self):
        """Perform initial scan of all monitored files"""
        try:
            scan_start = time.time()
            files_found = 0

            for file_path in self._get_monitored_files():
                try:
                    file_hash = self._calculate_file_hash(file_path)
                    file_stats = file_path.stat()

                    relative_path = str(file_path.relative_to(self.root_path))

                    metadata = FileMetadata(
                        path=relative_path,
                        hash=file_hash,
                        size=file_stats.st_size,
                        modified_time=file_stats.st_mtime
                    )

                    with self.file_lock:
                        self.tracked_files[relative_path] = metadata

                    files_found += 1

                except Exception as e:
                    await self.log_manager.log_system_event(
                        "file_scan_error",
                        {"file": str(file_path), "error": str(e)},
                        level="warning",
                        worker_id="file_monitor"
                    )

            scan_duration = time.time() - scan_start
            self.scan_duration_total += scan_duration
            self.files_scanned += files_found

            await self.log_manager.log_system_event(
                "initial_scan_complete",
                {
                    "files_scanned": files_found,
                    "scan_duration": scan_duration,
                    "files_per_second": files_found / scan_duration if scan_duration > 0 else 0
                },
                worker_id="file_monitor"
            )

        except Exception as e:
            await self.log_manager.log_system_event(
                "initial_scan_failed",
                {"error": str(e)},
                level="error",
                worker_id="file_monitor"
            )

    def _monitor_loop(self):
        """Main monitoring loop running in separate thread"""
        while self.monitoring_active and not self.shutdown_event.is_set():
            try:
                # Perform periodic scan for changes not caught by watchdog
                self._periodic_scan()

                # Sleep for the configured interval
                time.sleep(self.config.file_monitor_interval)

            except Exception as e:
                # Use thread-safe logging
                self._log_error("monitor_loop_error", {"error": str(e)})
                time.sleep(self.config.file_monitor_interval)

    def _periodic_scan(self):
        """Perform periodic scan for file changes"""
        try:
            scan_start = time.time()
            changes_found = 0

            # Get current file list
            current_files = {
                str(f.relative_to(self.root_path)): f
                for f in self._get_monitored_files()
            }

            with self.file_lock:
                tracked_paths = set(self.tracked_files.keys())

            # Check for new files
            new_files = set(current_files.keys()) - tracked_paths
            for relative_path in new_files:
                file_path = current_files[relative_path]
                self._handle_file_event("created", str(file_path))
                changes_found += 1

            # Check for deleted files
            deleted_files = tracked_paths - set(current_files.keys())
            for relative_path in deleted_files:
                self._handle_file_event("deleted", str(
                    self.root_path / relative_path))
                changes_found += 1

            # Check for modified files
            for relative_path, file_path in current_files.items():
                if relative_path in tracked_paths:
                    if self._check_file_modification(relative_path, file_path):
                        self._handle_file_event("modified", str(file_path))
                        changes_found += 1

            scan_duration = time.time() - scan_start
            self.scan_duration_total += scan_duration

            if changes_found > 0:
                self._log_info("periodic_scan_changes", {
                    "changes_found": changes_found,
                    "scan_duration": scan_duration
                })

        except Exception as e:
            self._log_error("periodic_scan_error", {"error": str(e)})

    def _check_file_modification(self, relative_path: str, file_path: Path) -> bool:
        """Check if a file has been modified"""
        try:
            file_stats = file_path.stat()

            with self.file_lock:
                metadata = self.tracked_files.get(relative_path)
                if not metadata:
                    return True  # New file

                # Check modification time first (quick check)
                if file_stats.st_mtime > metadata.modified_time:
                    # Verify with hash (accurate check)
                    current_hash = self._calculate_file_hash(file_path)
                    if current_hash != metadata.hash:
                        return True

            return False

        except Exception:
            return True  # Assume modified if we can't check

    def _handle_file_event(self, event_type: str, file_path: str, old_path: str = None):
        """Handle a file system event"""
        try:
            # Filter out non-monitored files
            path_obj = Path(file_path)

            if not self._should_monitor_file(path_obj):
                return

            # Create file change event
            event_id = f"{event_type}_{uuid.uuid4().hex[:8]}"
            relative_path = str(path_obj.relative_to(self.root_path))

            # Calculate hash for existing files
            file_hash = None
            old_hash = None

            if event_type in ["created", "modified"] and path_obj.exists():
                file_hash = self._calculate_file_hash(path_obj)

                with self.file_lock:
                    if relative_path in self.tracked_files:
                        old_hash = self.tracked_files[relative_path].hash

            change_event = FileChangeEvent(
                event_id=event_id,
                event_type=event_type,
                file_path=relative_path,
                absolute_path=file_path,
                timestamp=datetime.now(),
                file_hash=file_hash,
                old_hash=old_hash
            )

            # Update tracked files
            self._update_tracked_file(relative_path, path_obj, event_type)

            # Queue event for processing
            with self.queue_lock:
                self.change_queue.append(change_event)

            self.changes_detected += 1

        except Exception as e:
            self._log_error("file_event_error", {
                "event_type": event_type,
                "file_path": file_path,
                "error": str(e)
            })

    def _update_tracked_file(self, relative_path: str, file_path: Path, event_type: str):
        """Update tracked file metadata"""
        try:
            with self.file_lock:
                if event_type == "deleted":
                    self.tracked_files.pop(relative_path, None)
                elif event_type in ["created", "modified"] and file_path.exists():
                    file_stats = file_path.stat()
                    file_hash = self._calculate_file_hash(file_path)

                    self.tracked_files[relative_path] = FileMetadata(
                        path=relative_path,
                        hash=file_hash,
                        size=file_stats.st_size,
                        modified_time=file_stats.st_mtime
                    )

        except Exception as e:
            self._log_error("tracked_file_update_error", {
                "relative_path": relative_path,
                "event_type": event_type,
                "error": str(e)
            })

    def _event_processing_loop(self):
        """Process queued file change events"""
        while self.monitoring_active and not self.shutdown_event.is_set():
            try:
                # Get events from queue
                events_to_process = []

                with self.queue_lock:
                    if self.change_queue:
                        events_to_process = self.change_queue.copy()
                        self.change_queue.clear()

                # Process events
                for event in events_to_process:
                    try:
                        if self.change_callback:
                            asyncio.run_coroutine_threadsafe(
                                self.change_callback(event), self.loop)

                        event.processed = True
                        self.events_processed += 1

                    except Exception as e:
                        self._log_error("event_callback_error", {
                            "event_id": event.event_id,
                            "error": str(e)
                        })

                # Log processing summary if events were processed
                if events_to_process:
                    self._log_info("events_processed", {
                        "count": len(events_to_process),
                        "total_processed": self.events_processed
                    })

                # Sleep briefly
                time.sleep(0.5)

            except Exception as e:
                self._log_error("event_processing_error", {"error": str(e)})
                time.sleep(1.0)

    def _get_monitored_files(self) -> List[Path]:
        """Get list of files to monitor"""
        files = []

        for file_path in self.root_path.rglob("*"):
            if self._should_monitor_file(file_path):
                files.append(file_path)

        return files

    def _should_monitor_file(self, file_path: Path) -> bool:
        """Check if a file should be monitored"""
        try:
            # Must be a file
            if not file_path.is_file():
                return False

            # Check extension
            if file_path.suffix not in self.monitored_extensions:
                return False

            # Check excluded directories
            if any(excluded in file_path.parts for excluded in self.excluded_dirs):
                return False

            # Check excluded files
            if file_path.name in self.excluded_files:
                return False

            return True

        except Exception:
            return False

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    def _log_info(self, event_type: str, data: Dict):
        """Thread-safe info logging"""
        try:
            # Use asyncio to schedule logging
            import asyncio
            asyncio.run_coroutine_threadsafe(
                self.log_manager.log_system_event(
                    event_type, data, worker_id="file_monitor"),
                self.loop
            )
        except Exception:
            pass  # Ignore logging errors in worker thread

    def _log_error(self, event_type: str, data: Dict):
        """Thread-safe error logging"""
        try:
            import asyncio
            asyncio.run_coroutine_threadsafe(
                self.log_manager.log_system_event(
                    event_type, data, level="error", worker_id="file_monitor"),
                self.loop
            )
        except Exception:
            pass  # Ignore logging errors in worker thread

    def set_change_callback(self, callback: Callable[[FileChangeEvent], None]):
        """Set callback for file change events"""
        self.change_callback = callback

    def get_monitoring_status(self) -> Dict[str, any]:
        """Get current monitoring status"""
        return {
            "monitoring_active": self.monitoring_active,
            "tracked_files": len(self.tracked_files),
            "files_scanned": self.files_scanned,
            "changes_detected": self.changes_detected,
            "events_processed": self.events_processed,
            "queue_size": len(self.change_queue),
            "average_scan_time": self.scan_duration_total / max(1, self.files_scanned)
        }

    def get_tracked_files(self) -> Dict[str, Dict]:
        """Get information about tracked files"""
        with self.file_lock:
            return {
                path: {
                    "hash": metadata.hash,
                    "size": metadata.size,
                    "modified_time": metadata.modified_time,
                    "analysis_count": metadata.analysis_count
                }
                for path, metadata in self.tracked_files.items()
            }

    async def stop_monitoring(self, timeout: float = 10.0) -> bool:
        """Stop file monitoring gracefully"""
        try:
            await self.log_manager.log_system_event(
                "file_monitor_stopping",
                {"tracked_files": len(self.tracked_files)},
                worker_id="file_monitor"
            )

            self.monitoring_active = False
            self.shutdown_event.set()

            # Stop watchdog observer
            if self.watchdog_observer:
                self.watchdog_observer.stop()
                self.watchdog_observer.join(timeout=timeout/2)

            # Wait for threads to finish
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=timeout/4)

            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=timeout/4)

            await self.log_manager.log_system_event(
                "file_monitor_stopped",
                {
                    "final_stats": self.get_monitoring_status()
                },
                worker_id="file_monitor"
            )

            return True

        except Exception as e:
            await self.log_manager.log_system_event(
                "file_monitor_stop_error",
                {"error": str(e)},
                level="error",
                worker_id="file_monitor"
            )
            return False
