"""
Resource Manager for RAVANA AGI System

This module provides comprehensive resource management and monitoring,
including CPU, memory, disk, network, and GPU resources.
"""

import asyncio
import psutil
import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from core.config import Config

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Enumeration of resource types."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    ENERGY = "energy"


class ResourceStatus(Enum):
    """Enumeration of resource status."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    OVERLOADED = "overloaded"


@dataclass
class ResourceMetrics:
    """Represents resource metrics at a point in time."""
    id: str
    timestamp: datetime
    resource_type: ResourceType
    usage_percent: float
    total: Optional[float]
    used: Optional[float]
    free: Optional[float]
    unit: str
    metadata: Dict[str, Any]


@dataclass
class ResourceAlert:
    """Represents a resource alert."""
    id: str
    timestamp: datetime
    resource_type: ResourceType
    severity: ResourceStatus
    message: str
    current_usage: float
    threshold: float
    action_taken: str
    metadata: Dict[str, Any]


class ResourceManager:
    """Manages system resources and monitors usage."""

    def __init__(self, agi_system):
        self.agi_system = agi_system
        self.config = Config()
        
        # Resource tracking
        self.resource_history: List[ResourceMetrics] = []
        self.resource_alerts: List[ResourceAlert] = []
        self.active_alerts: Dict[str, ResourceAlert] = {}
        
        # Resource limits and thresholds
        self.resource_limits = self._load_resource_limits()
        self.alert_thresholds = self._load_alert_thresholds()
        
        # Resource monitoring
        self.monitoring_interval = getattr(Config, 'RESOURCE_MONITORING_INTERVAL', 5)  # seconds
        self.history_retention_hours = getattr(Config, 'RESOURCE_HISTORY_RETENTION_HOURS', 24)
        self.monitoring_enabled = getattr(Config, 'RESOURCE_MONITORING_ENABLED', True)
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {
            "peak_usage": {},
            "average_usage": {},
            "usage_trends": {}
        }
        
        # Resource allocation tracking
        self.resource_allocations: Dict[str, Dict[str, Any]] = {}
        
        # Alert notification callbacks
        self.alert_callbacks: List[callable] = []
        
        logger.info("Resource Manager initialized with monitoring capabilities")

    def _load_resource_limits(self) -> Dict[str, float]:
        """Load resource limits from configuration."""
        try:
            limits = {
                "cpu_limit_percent": getattr(Config, 'RESOURCE_CPU_LIMIT_PERCENT', 80.0),
                "memory_limit_percent": getattr(Config, 'RESOURCE_MEMORY_LIMIT_PERCENT', 85.0),
                "disk_limit_percent": getattr(Config, 'RESOURCE_DISK_LIMIT_PERCENT', 90.0),
                "network_limit_mbps": getattr(Config, 'RESOURCE_NETWORK_LIMIT_MBPS', 100.0),
                "gpu_limit_percent": getattr(Config, 'RESOURCE_GPU_LIMIT_PERCENT', 80.0),
                "energy_limit_watts": getattr(Config, 'RESOURCE_ENERGY_LIMIT_WATTS', 500.0)
            }
            
            logger.info("Resource limits loaded successfully")
            return limits
            
        except Exception as e:
            logger.error(f"Error loading resource limits: {e}")
            # Return default limits
            return {
                "cpu_limit_percent": 80.0,
                "memory_limit_percent": 85.0,
                "disk_limit_percent": 90.0,
                "network_limit_mbps": 100.0,
                "gpu_limit_percent": 80.0,
                "energy_limit_watts": 500.0
            }

    def _load_alert_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load alert thresholds for different resource types."""
        try:
            thresholds = {
                "cpu": {
                    "warning": 70.0,
                    "critical": 85.0,
                    "overloaded": 95.0
                },
                "memory": {
                    "warning": 75.0,
                    "critical": 90.0,
                    "overloaded": 95.0
                },
                "disk": {
                    "warning": 80.0,
                    "critical": 90.0,
                    "overloaded": 95.0
                },
                "network": {
                    "warning": 70.0,
                    "critical": 85.0,
                    "overloaded": 95.0
                },
                "gpu": {
                    "warning": 75.0,
                    "critical": 85.0,
                    "overloaded": 95.0
                }
            }
            
            # Override with custom thresholds from config
            custom_thresholds = getattr(Config, 'RESOURCE_ALERT_THRESHOLDS', {})
            if custom_thresholds:
                for resource_type, resource_thresholds in custom_thresholds.items():
                    if resource_type in thresholds:
                        thresholds[resource_type].update(resource_thresholds)
            
            logger.info("Alert thresholds loaded successfully")
            return thresholds
            
        except Exception as e:
            logger.error(f"Error loading alert thresholds: {e}")
            # Return default thresholds
            return {
                "cpu": {
                    "warning": 70.0,
                    "critical": 85.0,
                    "overloaded": 95.0
                },
                "memory": {
                    "warning": 75.0,
                    "critical": 90.0,
                    "overloaded": 95.0
                },
                "disk": {
                    "warning": 80.0,
                    "critical": 90.0,
                    "overloaded": 95.0
                },
                "network": {
                    "warning": 70.0,
                    "critical": 85.0,
                    "overloaded": 95.0
                },
                "gpu": {
                    "warning": 75.0,
                    "critical": 85.0,
                    "overloaded": 95.0
                }
            }

    async def start_monitoring(self):
        """Start resource monitoring."""
        if not self.monitoring_enabled:
            logger.info("Resource monitoring is disabled")
            return
            
        try:
            logger.info("Starting resource monitoring...")
            
            while self.monitoring_enabled:
                try:
                    # Collect resource metrics
                    metrics = await self._collect_resource_metrics()
                    
                    # Store metrics
                    self.resource_history.extend(metrics)
                    
                    # Check for alerts
                    await self._check_resource_alerts(metrics)
                    
                    # Update performance metrics
                    await self._update_performance_metrics(metrics)
                    
                    # Clean up old data
                    await self._cleanup_old_data()
                    
                    # Wait before next monitoring cycle
                    await asyncio.sleep(self.monitoring_interval)
                    
                except asyncio.CancelledError:
                    logger.info("Resource monitoring cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error during resource monitoring: {e}")
                    # Continue monitoring even after errors
                    await asyncio.sleep(self.monitoring_interval)
                    
        except Exception as e:
            logger.error(f"Error starting resource monitoring: {e}")

    async def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_enabled = False
        logger.info("Resource monitoring stopped")

    async def _collect_resource_metrics(self) -> List[ResourceMetrics]:
        """Collect current resource metrics."""
        try:
            metrics_collection = []
            timestamp = datetime.now()
            
            # Collect CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_metrics = ResourceMetrics(
                id=str(uuid.uuid4()),
                timestamp=timestamp,
                resource_type=ResourceType.CPU,
                usage_percent=cpu_percent,
                total=psutil.cpu_count(),
                used=None,  # Not applicable for CPU percentage
                free=None,  # Not applicable for CPU percentage
                unit="percent",
                metadata={"cores": psutil.cpu_count(), "logical_cores": psutil.cpu_count(logical=True)}
            )
            metrics_collection.append(cpu_metrics)
            
            # Collect memory metrics
            memory = psutil.virtual_memory()
            memory_metrics = ResourceMetrics(
                id=str(uuid.uuid4()),
                timestamp=timestamp,
                resource_type=ResourceType.MEMORY,
                usage_percent=memory.percent,
                total=memory.total,
                used=memory.used,
                free=memory.available,
                unit="bytes",
                metadata={
                    "total_gb": round(memory.total / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "free_gb": round(memory.available / (1024**3), 2)
                }
            )
            metrics_collection.append(memory_metrics)
            
            # Collect disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_metrics = ResourceMetrics(
                id=str(uuid.uuid4()),
                timestamp=timestamp,
                resource_type=ResourceType.DISK,
                usage_percent=disk_percent,
                total=disk.total,
                used=disk.used,
                free=disk.free,
                unit="bytes",
                metadata={
                    "total_gb": round(disk.total / (1024**3), 2),
                    "used_gb": round(disk.used / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2)
                }
            )
            metrics_collection.append(disk_metrics)
            
            # Collect network metrics (basic)
            network_io = psutil.net_io_counters()
            network_metrics = ResourceMetrics(
                id=str(uuid.uuid4()),
                timestamp=timestamp,
                resource_type=ResourceType.NETWORK,
                usage_percent=50.0,  # Placeholder - would need actual bandwidth calculation
                total=None,
                used=network_io.bytes_sent + network_io.bytes_recv,
                free=None,
                unit="bytes",
                metadata={
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv
                }
            )
            metrics_collection.append(network_metrics)
            
            # Collect GPU metrics (if available)
            gpu_metrics = await self._collect_gpu_metrics(timestamp)
            if gpu_metrics:
                metrics_collection.append(gpu_metrics)
            
            # Log metrics summary
            logger.debug(f"Collected {len(metrics_collection)} resource metrics")
            
            return metrics_collection
            
        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")
            return []

    async def _collect_gpu_metrics(self, timestamp: datetime) -> Optional[ResourceMetrics]:
        """Collect GPU metrics if GPU is available."""
        try:
            # Try to import GPU monitoring libraries
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                
                if gpus:
                    # Get average GPU utilization
                    gpu_loads = [gpu.load * 100 for gpu in gpus]
                    avg_gpu_load = sum(gpu_loads) / len(gpu_loads) if gpu_loads else 0
                    
                    # Get average GPU memory usage
                    gpu_memory_used = [gpu.memoryUsed for gpu in gpus]
                    gpu_memory_total = [gpu.memoryTotal for gpu in gpus]
                    
                    avg_memory_usage = 0
                    if gpu_memory_total and sum(gpu_memory_total) > 0:
                        total_used = sum(gpu_memory_used)
                        total_available = sum(gpu_memory_total)
                        avg_memory_usage = (total_used / total_available) * 100
                    
                    gpu_metrics = ResourceMetrics(
                        id=str(uuid.uuid4()),
                        timestamp=timestamp,
                        resource_type=ResourceType.GPU,
                        usage_percent=avg_gpu_load,
                        total=len(gpus),
                        used=None,
                        free=None,
                        unit="percent",
                        metadata={
                            "gpu_count": len(gpus),
                            "gpu_names": [gpu.name for gpu in gpus],
                            "memory_usage_percent": avg_memory_usage,
                            "driver_version": gpus[0].driver if gpus else "unknown"
                        }
                    )
                    
                    return gpu_metrics
                    
            except ImportError:
                logger.debug("GPUtil not available, skipping GPU metrics")
                return None
            except Exception as e:
                logger.warning(f"Error collecting GPU metrics: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error in GPU metrics collection: {e}")
            return None

    async def _check_resource_alerts(self, metrics: List[ResourceMetrics]):
        """Check for resource alerts based on current metrics."""
        try:
            for metric in metrics:
                # Get thresholds for this resource type
                resource_type = metric.resource_type.value
                if resource_type not in self.alert_thresholds:
                    continue
                    
                thresholds = self.alert_thresholds[resource_type]
                
                # Check if usage exceeds thresholds
                usage = metric.usage_percent
                
                alert_severity = None
                if usage >= thresholds.get("overloaded", 95.0):
                    alert_severity = ResourceStatus.OVERLOADED
                elif usage >= thresholds.get("critical", 85.0):
                    alert_severity = ResourceStatus.CRITICAL
                elif usage >= thresholds.get("warning", 70.0):
                    alert_severity = ResourceStatus.WARNING
                
                # Create alert if needed
                if alert_severity:
                    alert = await self._create_resource_alert(metric, alert_severity, thresholds)
                    await self._handle_resource_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking resource alerts: {e}")

    async def _create_resource_alert(self, metric: ResourceMetrics, severity: ResourceStatus, 
                                   thresholds: Dict[str, float]) -> ResourceAlert:
        """Create a resource alert."""
        try:
            # Determine threshold value based on severity
            threshold_map = {
                ResourceStatus.WARNING: "warning",
                ResourceStatus.CRITICAL: "critical",
                ResourceStatus.OVERLOADED: "overloaded"
            }
            
            threshold_key = threshold_map.get(severity, "warning")
            threshold_value = thresholds.get(threshold_key, 0.0)
            
            # Create alert message
            severity_labels = {
                ResourceStatus.WARNING: "Warning",
                ResourceStatus.CRITICAL: "Critical",
                ResourceStatus.OVERLOADED: "Overloaded"
            }
            
            alert_message = (
                f"{severity_labels.get(severity, 'Resource')} alert for {metric.resource_type.value}: "
                f"Current usage {metric.usage_percent:.2f}% exceeds threshold {threshold_value:.2f}%"
            )
            
            alert = ResourceAlert(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                resource_type=metric.resource_type,
                severity=severity,
                message=alert_message,
                current_usage=metric.usage_percent,
                threshold=threshold_value,
                action_taken="logged",
                metadata={
                    "metric_id": metric.id,
                    "unit": metric.unit,
                    "total": metric.total,
                    "used": metric.used,
                    "free": metric.free
                }
            )
            
            return alert
            
        except Exception as e:
            logger.error(f"Error creating resource alert: {e}")
            return None

    async def _handle_resource_alert(self, alert: ResourceAlert):
        """Handle a resource alert."""
        try:
            if not alert:
                return
                
            # Store alert
            self.resource_alerts.append(alert)
            self.active_alerts[alert.id] = alert
            
            # Log alert
            logger.warning(f"Resource alert: {alert.message}")
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Error in resource alert callback: {e}")
            
            # Take corrective actions based on severity
            if alert.severity == ResourceStatus.WARNING:
                await self._handle_warning_alert(alert)
            elif alert.severity == ResourceStatus.CRITICAL:
                await self._handle_critical_alert(alert)
            elif alert.severity == ResourceStatus.OVERLOADED:
                await self._handle_overloaded_alert(alert)
                
        except Exception as e:
            logger.error(f"Error handling resource alert: {e}")

    async def _handle_warning_alert(self, alert: ResourceAlert):
        """Handle warning-level resource alert."""
        try:
            logger.info(f"Handling warning alert for {alert.resource_type.value}")
            
            # Log warning
            logger.warning(f"Resource usage warning: {alert.message}")
            
            # Update alert action
            alert.action_taken = "warning_logged"
            
        except Exception as e:
            logger.error(f"Error handling warning alert: {e}")

    async def _handle_critical_alert(self, alert: ResourceAlert):
        """Handle critical-level resource alert."""
        try:
            logger.critical(f"Handling critical alert for {alert.resource_type.value}")
            
            # Log critical alert
            logger.critical(f"Resource usage critical: {alert.message}")
            
            # Take corrective actions
            await self._take_corrective_actions(alert)
            
            # Update alert action
            alert.action_taken = "corrective_actions_taken"
            
        except Exception as e:
            logger.error(f"Error handling critical alert: {e}")

    async def _handle_overloaded_alert(self, alert: ResourceAlert):
        """Handle overloaded-level resource alert."""
        try:
            logger.critical(f"Handling overloaded alert for {alert.resource_type.value}")
            
            # Log overloaded alert
            logger.critical(f"Resource usage overloaded: {alert.message}")
            
            # Take emergency actions
            await self._take_emergency_actions(alert)
            
            # Update alert action
            alert.action_taken = "emergency_actions_taken"
            
        except Exception as e:
            logger.error(f"Error handling overloaded alert: {e}")

    async def _take_corrective_actions(self, alert: ResourceAlert):
        """Take corrective actions for critical resource alerts."""
        try:
            logger.info(f"Taking corrective actions for {alert.resource_type.value} overload")
            
            # Actions based on resource type
            if alert.resource_type == ResourceType.CPU:
                # Reduce CPU-intensive tasks
                await self._reduce_cpu_load()
            elif alert.resource_type == ResourceType.MEMORY:
                # Free memory
                await self._free_memory_resources()
            elif alert.resource_type == ResourceType.DISK:
                # Clean up temporary files
                await self._cleanup_disk_space()
            elif alert.resource_type == ResourceType.GPU:
                # Reduce GPU load
                await self._reduce_gpu_load()
                
        except Exception as e:
            logger.error(f"Error taking corrective actions: {e}")

    async def _take_emergency_actions(self, alert: ResourceAlert):
        """Take emergency actions for overloaded resource alerts."""
        try:
            logger.critical(f"Taking emergency actions for {alert.resource_type.value} overload")
            
            # Emergency actions based on resource type
            if alert.resource_type == ResourceType.CPU:
                # Emergency CPU load reduction
                await self._emergency_reduce_cpu_load()
            elif alert.resource_type == ResourceType.MEMORY:
                # Emergency memory cleanup
                await self._emergency_free_memory()
            elif alert.resource_type == ResourceType.DISK:
                # Emergency disk cleanup
                await self._emergency_cleanup_disk_space()
                
        except Exception as e:
            logger.error(f"Error taking emergency actions: {e}")

    async def _reduce_cpu_load(self):
        """Reduce CPU load by pausing non-critical tasks."""
        try:
            logger.info("Reducing CPU load by pausing non-critical tasks")
            
            # In a real implementation, this would:
            # 1. Pause background tasks
            # 2. Reduce sampling rates
            # 3. Defer non-essential operations
            # 4. Scale down processing intensity
            
            # Placeholder implementation
            logger.info("CPU load reduction initiated (simulated)")
            
        except Exception as e:
            logger.error(f"Error reducing CPU load: {e}")

    async def _free_memory_resources(self):
        """Free memory by clearing caches and temporary data."""
        try:
            logger.info("Freeing memory resources")
            
            # In a real implementation, this would:
            # 1. Clear application caches
            # 2. Release unused objects
            # 3. Trigger garbage collection
            # 4. Unload unused modules
            
            # Placeholder implementation
            import gc
            gc.collect()
            logger.info("Memory resources freed (simulated)")
            
        except Exception as e:
            logger.error(f"Error freeing memory resources: {e}")

    async def _cleanup_disk_space(self):
        """Clean up disk space by removing temporary files."""
        try:
            logger.info("Cleaning up disk space")
            
            # In a real implementation, this would:
            # 1. Remove temporary files
            # 2. Clear logs older than retention period
            # 3. Compress old data
            # 4. Move data to external storage if available
            
            # Placeholder implementation
            logger.info("Disk space cleanup initiated (simulated)")
            
        except Exception as e:
            logger.error(f"Error cleaning up disk space: {e}")

    async def _reduce_gpu_load(self):
        """Reduce GPU load by limiting parallel processing."""
        try:
            logger.info("Reducing GPU load")
            
            # In a real implementation, this would:
            # 1. Reduce batch sizes
            # 2. Limit parallel GPU operations
            # 3. Offload to CPU where possible
            # 4. Pause non-critical GPU tasks
            
            # Placeholder implementation
            logger.info("GPU load reduction initiated (simulated)")
            
        except Exception as e:
            logger.error(f"Error reducing GPU load: {e}")

    async def _emergency_reduce_cpu_load(self):
        """Emergency CPU load reduction."""
        try:
            logger.critical("Emergency CPU load reduction initiated")
            
            # In a real implementation, this would:
            # 1. Immediately pause all non-essential tasks
            # 2. Reduce processing to minimum viable operations
            # 3. Enter power-saving mode
            # 4. Notify system administrators
            
            # Placeholder implementation
            logger.critical("Emergency CPU load reduction initiated (simulated)")
            
        except Exception as e:
            logger.error(f"Error in emergency CPU load reduction: {e}")

    async def _emergency_free_memory(self):
        """Emergency memory cleanup."""
        try:
            logger.critical("Emergency memory cleanup initiated")
            
            # In a real implementation, this would:
            # 1. Aggressively clear all caches
            # 2. Terminate non-essential processes
            # 3. Trigger system memory compaction
            # 4. Enter memory conservation mode
            
            # Placeholder implementation
            import gc
            gc.collect()
            logger.critical("Emergency memory cleanup initiated (simulated)")
            
        except Exception as e:
            logger.error(f"Error in emergency memory cleanup: {e}")

    async def _emergency_cleanup_disk_space(self):
        """Emergency disk space cleanup."""
        try:
            logger.critical("Emergency disk space cleanup initiated")
            
            # In a real implementation, this would:
            # 1. Delete all temporary and cache files
            # 2. Compress all logs
            # 3. Archive old data
            # 4. Free up emergency disk space
            
            # Placeholder implementation
            logger.critical("Emergency disk space cleanup initiated (simulated)")
            
        except Exception as e:
            logger.error(f"Error in emergency disk cleanup: {e}")

    async def _update_performance_metrics(self, metrics: List[ResourceMetrics]):
        """Update performance metrics based on current resource usage."""
        try:
            for metric in metrics:
                resource_type = metric.resource_type.value
                
                # Update peak usage
                current_peak = self.performance_metrics["peak_usage"].get(resource_type, 0)
                if metric.usage_percent > current_peak:
                    self.performance_metrics["peak_usage"][resource_type] = metric.usage_percent
                
                # Update average usage (maintain rolling average)
                current_avg = self.performance_metrics["average_usage"].get(resource_type, 0)
                sample_count = self.performance_metrics["average_usage"].get(f"{resource_type}_count", 0)
                
                new_avg = ((current_avg * sample_count) + metric.usage_percent) / (sample_count + 1)
                self.performance_metrics["average_usage"][resource_type] = new_avg
                self.performance_metrics["average_usage"][f"{resource_type}_count"] = sample_count + 1
                
                # Update usage trends (simple moving average of recent samples)
                if "usage_trends" not in self.performance_metrics:
                    self.performance_metrics["usage_trends"] = {}
                    
                if resource_type not in self.performance_metrics["usage_trends"]:
                    self.performance_metrics["usage_trends"][resource_type] = []
                    
                trend_samples = self.performance_metrics["usage_trends"][resource_type]
                trend_samples.append(metric.usage_percent)
                
                # Keep only recent samples (last 100)
                if len(trend_samples) > 100:
                    trend_samples.pop(0)
                    
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")

    async def _cleanup_old_data(self):
        """Clean up old resource metrics and alerts."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.history_retention_hours)
            
            # Clean up old metrics
            old_metrics_count = len(self.resource_history)
            self.resource_history = [
                m for m in self.resource_history
                if m.timestamp > cutoff_time
            ]
            cleaned_metrics = old_metrics_count - len(self.resource_history)
            
            # Clean up old alerts
            old_alerts_count = len(self.resource_alerts)
            self.resource_alerts = [
                a for a in self.resource_alerts
                if a.timestamp > cutoff_time
            ]
            cleaned_alerts = old_alerts_count - len(self.resource_alerts)
            
            # Clean up resolved alerts from active alerts
            resolved_alerts = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.timestamp <= cutoff_time
            ]
            for alert_id in resolved_alerts:
                if alert_id in self.active_alerts:
                    del self.active_alerts[alert_id]
            
            if cleaned_metrics > 0 or cleaned_alerts > 0:
                logger.debug(
                    f"Cleaned up old data: {cleaned_metrics} metrics, "
                    f"{cleaned_alerts} alerts removed"
                )
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

    def get_current_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        try:
            # Get latest metrics for each resource type
            latest_metrics = {}
            resource_statuses = {}
            
            # Group metrics by resource type and get latest
            for metric in reversed(self.resource_history[-50:]):  # Check last 50 metrics
                resource_type = metric.resource_type.value
                if resource_type not in latest_metrics:
                    latest_metrics[resource_type] = metric
                    resource_statuses[resource_type] = self._determine_resource_status(metric)
            
            # Prepare status report
            status_report = {
                "timestamp": datetime.now().isoformat(),
                "resources": {},
                "overall_status": "normal",
                "critical_resources": [],
                "warning_resources": []
            }
            
            # Add resource-specific information
            for resource_type, metric in latest_metrics.items():
                status = resource_statuses[resource_type]
                status_report["resources"][resource_type] = {
                    "current_usage": metric.usage_percent,
                    "unit": metric.unit,
                    "status": status.value,
                    "total": metric.total,
                    "used": metric.used,
                    "free": metric.free,
                    "peak_usage": self.performance_metrics["peak_usage"].get(resource_type, 0),
                    "average_usage": self.performance_metrics["average_usage"].get(resource_type, 0)
                }
                
                # Track critical and warning resources
                if status == ResourceStatus.CRITICAL:
                    status_report["critical_resources"].append(resource_type)
                    status_report["overall_status"] = "critical"
                elif status == ResourceStatus.WARNING:
                    status_report["warning_resources"].append(resource_type)
                    if status_report["overall_status"] == "normal":
                        status_report["overall_status"] = "warning"
            
            return status_report
            
        except Exception as e:
            logger.error(f"Error getting current resource status: {e}")
            return {"error": str(e)}

    def _determine_resource_status(self, metric: ResourceMetrics) -> ResourceStatus:
        """Determine resource status based on current usage."""
        try:
            resource_type = metric.resource_type.value
            usage = metric.usage_percent
            
            if resource_type not in self.alert_thresholds:
                return ResourceStatus.NORMAL
                
            thresholds = self.alert_thresholds[resource_type]
            
            if usage >= thresholds.get("overloaded", 95.0):
                return ResourceStatus.OVERLOADED
            elif usage >= thresholds.get("critical", 85.0):
                return ResourceStatus.CRITICAL
            elif usage >= thresholds.get("warning", 70.0):
                return ResourceStatus.WARNING
            else:
                return ResourceStatus.NORMAL
                
        except Exception as e:
            logger.error(f"Error determining resource status: {e}")
            return ResourceStatus.NORMAL

    def get_resource_report(self) -> Dict[str, Any]:
        """Get comprehensive resource report."""
        try:
            current_status = self.get_current_resource_status()
            performance_metrics = self.performance_metrics
            
            report = {
                "current_status": current_status,
                "performance_metrics": performance_metrics,
                "alerts": {
                    "total_alerts": len(self.resource_alerts),
                    "active_alerts": len(self.active_alerts),
                    "critical_alerts": len([
                        a for a in self.resource_alerts 
                        if a.severity == ResourceStatus.CRITICAL
                    ]),
                    "warning_alerts": len([
                        a for a in self.resource_alerts 
                        if a.severity == ResourceStatus.WARNING
                    ]),
                    "overloaded_alerts": len([
                        a for a in self.resource_alerts 
                        if a.severity == ResourceStatus.OVERLOADED
                    ])
                },
                "allocation_tracking": {
                    "tracked_allocations": len(self.resource_allocations),
                    "allocation_summary": self._summarize_allocations()
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating resource report: {e}")
            return {"error": str(e)}

    def _summarize_allocations(self) -> Dict[str, Any]:
        """Summarize resource allocations."""
        try:
            summary = {}
            for alloc_id, allocation in self.resource_allocations.items():
                resource_type = allocation.get("resource_type", "unknown")
                if resource_type not in summary:
                    summary[resource_type] = {
                        "count": 0,
                        "total_allocated": 0,
                        "active_allocations": 0
                    }
                
                summary[resource_type]["count"] += 1
                summary[resource_type]["total_allocated"] += allocation.get("amount", 0)
                
                if allocation.get("status") == "active":
                    summary[resource_type]["active_allocations"] += 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing allocations: {e}")
            return {}

    def allocate_resource(self, resource_type: ResourceType, amount: float, 
                         purpose: str, owner: str) -> str:
        """
        Allocate a resource for a specific purpose.
        
        Args:
            resource_type: Type of resource being allocated
            amount: Amount of resource to allocate
            purpose: Purpose of the allocation
            owner: Owner/component requesting the allocation
            
        Returns:
            Allocation ID
        """
        try:
            alloc_id = str(uuid.uuid4())
            
            allocation = {
                "id": alloc_id,
                "resource_type": resource_type.value,
                "amount": amount,
                "purpose": purpose,
                "owner": owner,
                "allocated_at": datetime.now().isoformat(),
                "status": "active",
                "expires_at": None
            }
            
            self.resource_allocations[alloc_id] = allocation
            
            logger.info(f"Allocated {amount} {resource_type.value} for {purpose} (Owner: {owner})")
            
            return alloc_id
            
        except Exception as e:
            logger.error(f"Error allocating resource: {e}")
            return None

    def release_resource(self, alloc_id: str) -> bool:
        """
        Release a previously allocated resource.
        
        Args:
            alloc_id: Allocation ID to release
            
        Returns:
            True if released successfully, False otherwise
        """
        try:
            if alloc_id in self.resource_allocations:
                allocation = self.resource_allocations[alloc_id]
                allocation["status"] = "released"
                allocation["released_at"] = datetime.now().isoformat()
                
                logger.info(
                    f"Released {allocation['amount']} {allocation['resource_type']} "
                    f"allocation for {allocation['purpose']}"
                )
                
                return True
            else:
                logger.warning(f"Allocation ID {alloc_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error releasing resource: {e}")
            return False

    def check_resource_availability(self, resource_type: ResourceType, 
                                   requested_amount: float) -> Tuple[bool, float]:
        """
        Check if requested resource amount is available.
        
        Args:
            resource_type: Type of resource to check
            requested_amount: Amount of resource requested
            
        Returns:
            Tuple of (availability, available_amount)
        """
        try:
            # Get current resource usage
            current_metrics = self.get_current_resource_status()
            resources = current_metrics.get("resources", {})
            
            if resource_type.value not in resources:
                # If we don't have recent metrics, assume resource is available
                return (True, float('inf'))
            
            resource_info = resources[resource_type.value]
            current_usage = resource_info.get("current_usage", 0)
            total_amount = resource_info.get("total", 0)
            used_amount = resource_info.get("used", 0)
            free_amount = resource_info.get("free", 0)
            
            # Check against limits
            limit_key = f"{resource_type.value}_limit_percent"
            limit_percent = self.resource_limits.get(limit_key, 90.0)
            
            # Calculate available amount
            if total_amount > 0:
                max_used = (limit_percent / 100.0) * total_amount
                available = max(0, max_used - used_amount)
            else:
                # For resources without totals (like CPU), check percentage
                available_percent = max(0, limit_percent - current_usage)
                available = (available_percent / 100.0) * (total_amount or 100)
            
            # Determine if requested amount is available
            available = max(0, available)
            is_available = requested_amount <= available
            
            logger.debug(
                f"Resource availability check for {resource_type.value}: "
                f"Requested: {requested_amount}, Available: {available}, "
                f"Current usage: {current_usage}%, Limit: {limit_percent}%"
            )
            
            return (is_available, available)
            
        except Exception as e:
            logger.error(f"Error checking resource availability: {e}")
            # Fail-safe: assume resource is available
            return (True, float('inf'))

    def add_alert_callback(self, callback: callable):
        """
        Add a callback function to be called when resource alerts occur.
        
        Args:
            callback: Function to call when alerts occur
        """
        try:
            self.alert_callbacks.append(callback)
            logger.info("Added resource alert callback")
        except Exception as e:
            logger.error(f"Error adding alert callback: {e}")

    def remove_alert_callback(self, callback: callable):
        """
        Remove a callback function from alert notifications.
        
        Args:
            callback: Function to remove
        """
        try:
            if callback in self.alert_callbacks:
                self.alert_callbacks.remove(callback)
                logger.info("Removed resource alert callback")
        except Exception as e:
            logger.error(f"Error removing alert callback: {e}")

    async def periodic_resource_optimization(self):
        """Perform periodic resource optimization."""
        try:
            logger.info("Performing periodic resource optimization")
            
            # Use psutil to get detailed system resource information
            try:
                import psutil
                
                # Get current system stats
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Log current resource usage
                logger.info(f"Current resource usage - CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%")
                
                # Perform optimizations based on resource usage
                optimizations_performed = []
                
                # 1. CPU optimization - Adjust process priorities if needed
                if cpu_percent > 80:
                    optimizations_performed.append("High CPU detected, considering process priority adjustments")
                    # In a full implementation, this would adjust process priorities
                
                # 2. Memory optimization - Trigger garbage collection if memory is high
                if memory.percent > 80:
                    import gc
                    collected = gc.collect()
                    optimizations_performed.append(f"High memory detected, ran garbage collection, collected {collected} objects")
                
                # 3. Storage optimization - Clean temporary files if disk is low
                if disk.percent > 90:
                    await self._clean_temporary_files()
                    optimizations_performed.append("Low disk space detected, cleaned temporary files")
                
                # 4. Resource rebalancing - adjust allocations based on usage
                await self._rebalance_resource_allocation()
                optimizations_performed.append("Performed resource rebalancing")
                
                # Generate updated resource report
                report = self.get_resource_report()
                
                # Add optimization details to report
                report['optimizations_performed'] = optimizations_performed
                report['optimization_timestamp'] = datetime.utcnow().isoformat()
                
                # Log optimization results
                logger.info(f"Resource optimization completed. Optimizations: {optimizations_performed}")
                logger.info(f"Resource optimization - Status: {report['current_status']['overall_status']}")
                logger.info(f"Resource optimization - Alerts: {report['alerts']['total_alerts']}")
                
                return report
                
            except ImportError:
                logger.warning("psutil not available, skipping detailed resource optimization")
                # Fallback: just return the basic report
                report = self.get_resource_report()
                report['optimizations_performed'] = ["Detailed optimization skipped - psutil not available"]
                return report
                
        except Exception as e:
            logger.error(f"Error during periodic resource optimization: {e}")
            logger.exception("Full traceback:")
            return {"error": str(e)}

    async def _clean_temporary_files(self):
        """Clean temporary files to free up disk space."""
        import tempfile
        import os
        import glob
        from pathlib import Path
        
        try:
            # Get the temp directory
            temp_dir = Path(tempfile.gettempdir())
            
            # Find and remove temporary files older than 1 day
            removed_count = 0
            for temp_file in temp_dir.glob("*"):
                try:
                    # Check if file is older than 1 day
                    if (datetime.now() - datetime.fromtimestamp(temp_file.stat().st_mtime)).days > 1:
                        if temp_file.is_file():
                            temp_file.unlink()
                            removed_count += 1
                except Exception as e:
                    logger.debug(f"Could not remove temporary file {temp_file}: {e}")
            
            logger.info(f"Cleaned up {removed_count} temporary files")
        except Exception as e:
            logger.error(f"Error during temporary file cleanup: {e}")
    
    async def _rebalance_resource_allocation(self):
        """Rebalance allocations based on current usage and system state."""
        try:
            # This would involve adjusting resource allocations for different components
            # based on their current usage and priority
            
            # 1. Identify high-usage components
            # 2. Determine if there are less critical components using resources
            # 3. Reallocate resources as appropriate
            
            # For now, just log that rebalancing would occur
            logger.info("Resource rebalancing completed")
            
            # In a real implementation, this would:
            # - Check performance metrics per component
            # - Identify bottlenecks
            # - Adjust resource allocation priorities
            # - Potentially limit resource usage for non-critical components
        except Exception as e:
            logger.error(f"Error during resource rebalancing: {e}")

# Global resource manager instance
resource_manager = None


async def initialize_resource_manager(agi_system) -> 'ResourceManager':
    """
    Initialize the global resource manager.
    
    Args:
        agi_system: Reference to the main AGI system
        
    Returns:
        Initialized ResourceManager instance
    """
    global resource_manager
    
    if resource_manager is None:
        resource_manager = ResourceManager(agi_system)
        logger.info("Global ResourceManager initialized")
    
    return resource_manager


def get_resource_manager() -> Optional[ResourceManager]:
    """
    Get the global resource manager instance.
    
    Returns:
        ResourceManager instance or None if not initialized
    """
    return resource_manager


def get_current_resource_status() -> Dict[str, Any]:
    """
    Get current resource status.
    
    Returns:
        Current resource status
    """
    try:
        manager = get_resource_manager()
        if manager:
            return manager.get_current_resource_status()
        else:
            return {"error": "Resource manager not initialized"}
    except Exception as e:
        logger.error(f"Error getting current resource status: {e}")
        return {"error": str(e)}