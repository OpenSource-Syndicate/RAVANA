"""
Snake Inter-Process Communication Manager

This module implements an IPC manager for coordinating between threads and processes
in the enhanced Snake Agent system.
"""

import asyncio
import multiprocessing
import threading
import queue
import time
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from core.snake_data_models import SnakeAgentConfiguration
from core.snake_log_manager import SnakeLogManager


class MessageType(Enum):
    """Types of IPC messages"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"
    BROADCAST = "broadcast"
    COORDINATION = "coordination"
    ERROR = "error"


class MessagePriority(Enum):
    """Message priorities"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class IPCMessage:
    """Inter-process communication message"""
    message_id: str
    message_type: MessageType
    priority: MessagePriority
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None  # For request-response correlation
    ttl_seconds: float = 300.0  # Time to live

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "ttl_seconds": self.ttl_seconds
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IPCMessage':
        """Create message from dictionary"""
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            priority=MessagePriority(data["priority"]),
            sender_id=data["sender_id"],
            recipient_id=data.get("recipient_id"),
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id"),
            ttl_seconds=data.get("ttl_seconds", 300.0)
        )

    def is_expired(self) -> bool:
        """Check if message has expired"""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl_seconds


class MessageChannel:
    """Communication channel for messages"""

    def __init__(self, channel_name: str, max_size: int = 1000):
        self.channel_name = channel_name
        self.max_size = max_size

        # Multiprocessing queues for cross-process communication
        self.process_queue = multiprocessing.Queue(maxsize=max_size)

        # Threading queues for same-process communication
        self.thread_queue = queue.Queue(maxsize=max_size)

        # Message handlers
        self.message_handlers: Dict[MessageType, List[Callable]] = {}

        # Statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.messages_dropped = 0

    def send_message(self, message: IPCMessage, use_process_queue: bool = True) -> bool:
        """Send message through channel"""
        try:
            serialized = json.dumps(message.to_dict())

            if use_process_queue:
                self.process_queue.put_nowait(serialized)
            else:
                self.thread_queue.put_nowait(message)

            self.messages_sent += 1
            return True

        except (queue.Full, Exception):
            self.messages_dropped += 1
            return False

    def receive_message(self, timeout: float = 1.0, use_process_queue: bool = True) -> Optional[IPCMessage]:
        """Receive message from channel"""
        try:
            if use_process_queue:
                serialized = self.process_queue.get(timeout=timeout)
                message_dict = json.loads(serialized)
                message = IPCMessage.from_dict(message_dict)
            else:
                message = self.thread_queue.get(timeout=timeout)

            # Check if message has expired
            if message.is_expired():
                self.messages_dropped += 1
                return None

            self.messages_received += 1
            return message

        except (queue.Empty, json.JSONDecodeError):
            return None

    def add_handler(self, message_type: MessageType, handler: Callable[[IPCMessage], None]):
        """Add message handler for specific type"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)

    def handle_message(self, message: IPCMessage):
        """Handle message with registered handlers"""
        handlers = self.message_handlers.get(message.message_type, [])
        for handler in handlers:
            try:
                handler(message)
            except Exception as e:
                # Log handler error but continue with other handlers
                pass

    def get_status(self) -> Dict[str, Any]:
        """Get channel status"""
        return {
            "channel_name": self.channel_name,
            "process_queue_size": self.process_queue.qsize(),
            "thread_queue_size": self.thread_queue.qsize(),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_dropped": self.messages_dropped,
            "handler_count": sum(len(handlers) for handlers in self.message_handlers.values())
        }


class ComponentRegistry:
    """Registry of components for IPC"""

    def __init__(self):
        self.components: Dict[str, Dict[str, Any]] = {}
        self.component_lock = threading.Lock()

    def register_component(self, component_id: str, component_type: str,
                           metadata: Optional[Dict[str, Any]] = None):
        """Register a component"""
        with self.component_lock:
            self.components[component_id] = {
                "component_type": component_type,
                "registered_at": datetime.now(),
                "last_heartbeat": datetime.now(),
                "metadata": metadata or {},
                "status": "active"
            }

    def unregister_component(self, component_id: str):
        """Unregister a component"""
        with self.component_lock:
            if component_id in self.components:
                self.components[component_id]["status"] = "inactive"

    def update_heartbeat(self, component_id: str):
        """Update component heartbeat"""
        with self.component_lock:
            if component_id in self.components:
                self.components[component_id]["last_heartbeat"] = datetime.now()

    def get_components_by_type(self, component_type: str) -> List[str]:
        """Get component IDs by type"""
        with self.component_lock:
            return [
                comp_id for comp_id, info in self.components.items()
                if info["component_type"] == component_type and info["status"] == "active"
            ]

    def get_component_info(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get component information"""
        with self.component_lock:
            return self.components.get(component_id)

    def cleanup_stale_components(self, max_age_seconds: float = 300.0):
        """Clean up components that haven't sent heartbeats"""
        current_time = datetime.now()
        stale_components = []

        with self.component_lock:
            for comp_id, info in self.components.items():
                age = (current_time - info["last_heartbeat"]).total_seconds()
                if age > max_age_seconds:
                    stale_components.append(comp_id)

            for comp_id in stale_components:
                self.components[comp_id]["status"] = "stale"


class MessageRouter:
    """Routes messages between components"""

    def __init__(self, component_registry: ComponentRegistry):
        self.component_registry = component_registry
        self.routing_rules: Dict[str, List[str]] = {}
        self.message_stats: Dict[str, int] = {}

    def add_routing_rule(self, sender_pattern: str, recipient_ids: List[str]):
        """Add routing rule for message forwarding"""
        self.routing_rules[sender_pattern] = recipient_ids

    def route_message(self, message: IPCMessage) -> List[str]:
        """Route message and return list of recipient IDs"""
        recipients = []

        # Direct recipient
        if message.recipient_id:
            recipients.append(message.recipient_id)
        else:
            # Broadcast - find all active components of relevant types
            if message.message_type == MessageType.BROADCAST:
                recipients.extend(self.component_registry.components.keys())
            elif message.message_type == MessageType.SHUTDOWN:
                recipients.extend(self.component_registry.components.keys())

        # Apply routing rules
        for pattern, rule_recipients in self.routing_rules.items():
            if pattern in message.sender_id:
                recipients.extend(rule_recipients)

        # Update stats
        route_key = f"{message.sender_id}->{len(recipients)}"
        self.message_stats[route_key] = self.message_stats.get(
            route_key, 0) + 1

        return list(set(recipients))  # Remove duplicates


class SnakeIPCManager:
    """Main IPC manager for Snake Agent system"""

    def __init__(self, config: SnakeAgentConfiguration, log_manager: SnakeLogManager):
        self.config = config
        self.log_manager = log_manager

        # Core components
        self.component_registry = ComponentRegistry()
        self.message_router = MessageRouter(self.component_registry)

        # Communication channels
        self.channels: Dict[str, MessageChannel] = {}
        self.default_channels = [
            "coordination", "task_distribution", "status_updates",
            "heartbeats", "emergency"
        ]

        # Message processing
        self.message_processor_thread: Optional[threading.Thread] = None
        self.running = False
        self.shutdown_event = threading.Event()

        # Request-response tracking
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.request_timeout = 30.0  # seconds

        # Statistics
        self.total_messages_processed = 0
        self.message_processing_time = 0.0

        # Component ID for this manager
        self.manager_id = f"ipc_manager_{uuid.uuid4().hex[:8]}"

    async def initialize(self) -> bool:
        """Initialize the IPC manager"""
        try:
            await self.log_manager.log_system_event(
                "ipc_manager_init",
                {"manager_id": self.manager_id},
                worker_id="ipc_manager"
            )

            # Create default channels
            for channel_name in self.default_channels:
                self.channels[channel_name] = MessageChannel(
                    channel_name,
                    self.config.max_queue_size
                )

            # Register self as component
            self.component_registry.register_component(
                self.manager_id,
                "ipc_manager",
                {"version": "1.0", "channels": list(self.channels.keys())}
            )

            return True

        except Exception as e:
            await self.log_manager.log_system_event(
                "ipc_manager_init_failed",
                {"error": str(e)},
                level="error",
                worker_id="ipc_manager"
            )
            return False

    async def start(self) -> bool:
        """Start IPC processing"""
        try:
            if self.running:
                return True

            self.running = True
            self.shutdown_event.clear()

            # Start message processor
            self.message_processor_thread = threading.Thread(
                target=self._message_processor_loop,
                name="Snake-IPCProcessor",
                daemon=True
            )
            self.message_processor_thread.start()

            # Start heartbeat sender
            asyncio.create_task(self._heartbeat_loop())

            # Start cleanup task
            asyncio.create_task(self._cleanup_loop())

            await self.log_manager.log_system_event(
                "ipc_manager_started",
                {"channels": len(self.channels)},
                worker_id="ipc_manager"
            )

            return True

        except Exception as e:
            await self.log_manager.log_system_event(
                "ipc_manager_start_failed",
                {"error": str(e)},
                level="error",
                worker_id="ipc_manager"
            )
            return False

    def _message_processor_loop(self):
        """Main message processing loop"""
        while self.running and not self.shutdown_event.is_set():
            try:
                messages_processed = 0

                # Process messages from all channels
                for channel in self.channels.values():
                    # Process from both queues
                    for use_process_queue in [True, False]:
                        try:
                            message = channel.receive_message(
                                timeout=0.1,
                                use_process_queue=use_process_queue
                            )

                            if message:
                                start_time = time.time()
                                self._process_message(message, channel)
                                processing_time = time.time() - start_time

                                self.total_messages_processed += 1
                                self.message_processing_time += processing_time
                                messages_processed += 1

                        except Exception as e:
                            asyncio.create_task(self.log_manager.log_system_event(
                                "message_processing_error",
                                {"error": str(e)},
                                level="error",
                                worker_id="ipc_manager"
                            ))

                # Sleep briefly if no messages processed
                if messages_processed == 0:
                    time.sleep(0.1)

            except Exception as e:
                asyncio.create_task(self.log_manager.log_system_event(
                    "message_processor_loop_error",
                    {"error": str(e)},
                    level="error",
                    worker_id="ipc_manager"
                ))
                time.sleep(1.0)

    def _process_message(self, message: IPCMessage, channel: MessageChannel):
        """Process a single message"""
        try:
            # Handle system messages
            if message.message_type == MessageType.HEARTBEAT:
                self.component_registry.update_heartbeat(message.sender_id)

            elif message.message_type == MessageType.TASK_RESPONSE:
                # Handle response to pending request
                if message.correlation_id and message.correlation_id in self.pending_requests:
                    future = self.pending_requests.pop(message.correlation_id)
                    if not future.done():
                        future.set_result(message.payload)

            # Route message to appropriate recipients
            recipients = self.message_router.route_message(message)

            # Forward message to recipients through their channels
            for recipient_id in recipients:
                if recipient_id != message.sender_id:  # Don't send back to sender
                    self._forward_message(message, recipient_id)

            # Call channel handlers
            channel.handle_message(message)

        except Exception as e:
            asyncio.create_task(self.log_manager.log_system_event(
                "process_message_error",
                {"error": str(e), "message_id": message.message_id},
                level="error",
                worker_id="ipc_manager"
            ))

    def _forward_message(self, message: IPCMessage, recipient_id: str):
        """Forward message to specific recipient"""
        # For now, we'll use the coordination channel for forwarding
        # In a more complex system, each component would have its own channel
        coordination_channel = self.channels.get("coordination")
        if coordination_channel:
            # Create forwarded message
            forwarded_message = IPCMessage(
                message_id=f"fwd_{uuid.uuid4().hex[:8]}",
                message_type=message.message_type,
                priority=message.priority,
                sender_id=self.manager_id,
                recipient_id=recipient_id,
                payload={
                    "original_message": message.to_dict(),
                    "forwarded_by": self.manager_id,
                    "forwarded_at": datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                correlation_id=message.correlation_id
            )

            coordination_channel.send_message(forwarded_message)

    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running and not self.shutdown_event.is_set():
            try:
                await self.send_heartbeat()
                await asyncio.sleep(10.0)  # Heartbeat every 10 seconds
            except Exception as e:
                await self.log_manager.log_system_event(
                    "heartbeat_loop_error",
                    {"error": str(e)},
                    level="error",
                    worker_id="ipc_manager"
                )
                await asyncio.sleep(10.0)

    async def _cleanup_loop(self):
        """Periodic cleanup of expired messages and stale components"""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Cleanup stale components
                self.component_registry.cleanup_stale_components()

                # Cleanup expired pending requests
                current_time = time.time()
                expired_requests = [
                    req_id for req_id, future in self.pending_requests.items()
                    if current_time - future._start_time > self.request_timeout
                ]

                for req_id in expired_requests:
                    future = self.pending_requests.pop(req_id)
                    if not future.done():
                        future.set_exception(TimeoutError("Request timeout"))

                await asyncio.sleep(60.0)  # Cleanup every minute

            except Exception as e:
                await self.log_manager.log_system_event(
                    "cleanup_loop_error",
                    {"error": str(e)},
                    level="error",
                    worker_id="ipc_manager"
                )
                await asyncio.sleep(60.0)

    def register_component(self, component_id: str, component_type: str,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Register a component with the IPC system"""
        try:
            self.component_registry.register_component(
                component_id, component_type, metadata)

            asyncio.create_task(self.log_manager.log_system_event(
                "component_registered",
                {
                    "component_id": component_id,
                    "component_type": component_type,
                    "metadata": metadata
                },
                worker_id="ipc_manager"
            ))

            return True

        except Exception as e:
            asyncio.create_task(self.log_manager.log_system_event(
                "component_registration_error",
                {"error": str(e), "component_id": component_id},
                level="error",
                worker_id="ipc_manager"
            ))
            return False

    def send_message(self, channel_name: str, message: IPCMessage,
                     use_process_queue: bool = True) -> bool:
        """Send message through specified channel"""
        try:
            channel = self.channels.get(channel_name)
            if not channel:
                return False

            return channel.send_message(message, use_process_queue)

        except Exception as e:
            asyncio.create_task(self.log_manager.log_system_event(
                "send_message_error",
                {"error": str(e), "channel": channel_name},
                level="error",
                worker_id="ipc_manager"
            ))
            return False

    async def send_request(self, channel_name: str, recipient_id: str,
                           payload: Dict[str, Any], timeout: float = 30.0) -> Any:
        """Send request and wait for response"""
        correlation_id = f"req_{uuid.uuid4().hex[:8]}"

        # Create future for response
        future = asyncio.Future()
        future._start_time = time.time()
        self.pending_requests[correlation_id] = future

        # Send request message
        request_message = IPCMessage(
            message_id=f"msg_{uuid.uuid4().hex[:8]}",
            message_type=MessageType.TASK_REQUEST,
            priority=MessagePriority.MEDIUM,
            sender_id=self.manager_id,
            recipient_id=recipient_id,
            payload=payload,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            ttl_seconds=timeout
        )

        success = self.send_message(channel_name, request_message)
        if not success:
            self.pending_requests.pop(correlation_id, None)
            raise RuntimeError("Failed to send request")

        try:
            # Wait for response
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self.pending_requests.pop(correlation_id, None)
            raise TimeoutError("Request timeout")

    async def send_heartbeat(self):
        """Send heartbeat message"""
        heartbeat_message = IPCMessage(
            message_id=f"hb_{uuid.uuid4().hex[:8]}",
            message_type=MessageType.HEARTBEAT,
            priority=MessagePriority.LOW,
            sender_id=self.manager_id,
            recipient_id=None,  # Broadcast
            payload={
                "timestamp": datetime.now().isoformat(),
                "status": "active",
                "processed_messages": self.total_messages_processed
            },
            timestamp=datetime.now()
        )

        self.send_message("heartbeats", heartbeat_message)

    def broadcast_message(self, message_type: MessageType, payload: Dict[str, Any],
                          priority: MessagePriority = MessagePriority.MEDIUM):
        """Broadcast message to all components"""
        broadcast_message = IPCMessage(
            message_id=f"bc_{uuid.uuid4().hex[:8]}",
            message_type=message_type,
            priority=priority,
            sender_id=self.manager_id,
            recipient_id=None,  # Broadcast
            payload=payload,
            timestamp=datetime.now()
        )

        self.send_message("coordination", broadcast_message)

    def add_message_handler(self, channel_name: str, message_type: MessageType,
                            handler: Callable[[IPCMessage], None]):
        """Add message handler to channel"""
        channel = self.channels.get(channel_name)
        if channel:
            channel.add_handler(message_type, handler)

    async def get_status(self) -> Dict[str, Any]:
        """Get IPC manager status"""
        channel_statuses = {}
        for name, channel in self.channels.items():
            # Handle both sync and async get_status methods
            if hasattr(channel.get_status, "__call__"):
                if asyncio.iscoroutinefunction(channel.get_status):
                    channel_statuses[name] = await channel.get_status()
                else:
                    channel_statuses[name] = channel.get_status()
            else:
                channel_statuses[name] = str(channel.get_status)

        return {
            "manager_id": self.manager_id,
            "running": self.running,
            "total_messages_processed": self.total_messages_processed,
            "average_processing_time": (
                self.message_processing_time /
                max(1, self.total_messages_processed)
            ),
            "pending_requests": len(self.pending_requests),
            "registered_components": len(self.component_registry.components),
            "channels": channel_statuses,
            "component_registry": {
                comp_id: {
                    "type": info["component_type"],
                    "status": info["status"],
                    "last_heartbeat": info["last_heartbeat"].isoformat()
                }
                for comp_id, info in self.component_registry.components.items()
            }
        }

    async def shutdown(self, timeout: float = 30.0) -> bool:
        """Shutdown the IPC manager"""
        try:
            await self.log_manager.log_system_event(
                "ipc_manager_shutdown",
                {"processed_messages": self.total_messages_processed},
                worker_id="ipc_manager"
            )

            # Send shutdown broadcast
            self.broadcast_message(
                MessageType.SHUTDOWN,
                {"reason": "system_shutdown", "timestamp": datetime.now().isoformat()},
                MessagePriority.CRITICAL
            )

            self.running = False
            self.shutdown_event.set()

            # Wait for message processor to finish
            if self.message_processor_thread and self.message_processor_thread.is_alive():
                self.message_processor_thread.join(timeout=timeout)

            return True

        except Exception as e:
            await self.log_manager.log_system_event(
                "ipc_manager_shutdown_error",
                {"error": str(e)},
                level="error",
                worker_id="ipc_manager"
            )
            return False
