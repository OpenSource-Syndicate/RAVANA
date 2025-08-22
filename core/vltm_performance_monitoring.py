"""
Very Long-Term Memory Performance Monitoring System

This module implements comprehensive performance monitoring for VLTM operations,
including memory operations, consolidation efficiency, and retrieval performance.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of performance metrics"""
    OPERATION_TIME = "operation_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    CONSOLIDATION_EFFICIENCY = "consolidation_efficiency"
    INDEX_PERFORMANCE = "index_performance"


class OperationType(str, Enum):
    """Types of VLTM operations to monitor"""
    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_SEARCH = "memory_search"
    CONSOLIDATION = "consolidation"
    PATTERN_EXTRACTION = "pattern_extraction"
    COMPRESSION = "compression"
    INDEX_REBUILD = "index_rebuild"
    MIGRATION = "migration"


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    metric_id: str
    metric_type: MetricType
    operation_type: OperationType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationProfile:
    """Performance profile for an operation"""
    operation_type: OperationType
    total_executions: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    error_count: int = 0
    success_rate: float = 100.0
    last_execution: Optional[datetime] = None


@dataclass
class PerformanceAlert:
    """Performance alert for threshold violations"""
    alert_id: str
    alert_type: str
    operation_type: OperationType
    threshold_value: float
    actual_value: float
    timestamp: datetime
    severity: str
    message: str


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for VLTM operations.
    
    Monitors and tracks:
    - Operation execution times
    - Throughput and latency metrics
    - Error rates and success rates
    - Memory and resource usage
    - Cache performance
    - Consolidation efficiency
    - Index performance
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.operation_profiles: Dict[OperationType, OperationProfile] = {}
        
        # Real-time monitoring
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        
        # Performance alerts
        self.alerts: List[PerformanceAlert] = []
        self.alert_thresholds: Dict[str, float] = {
            "max_operation_time_ms": 5000,
            "min_success_rate": 95.0,
            "max_error_rate": 5.0,
            "max_memory_usage_mb": 1000,
            "min_cache_hit_rate": 80.0
        }
        
        # Statistics
        self.monitoring_start_time = datetime.utcnow()
        self.is_monitoring = False
        
        logger.info("Performance Monitor initialized")
    
    async def start_monitoring(self) -> bool:
        """Start performance monitoring"""
        try:
            self.is_monitoring = True
            self.monitoring_start_time = datetime.utcnow()
            
            # Initialize operation profiles
            for op_type in OperationType:
                self.operation_profiles[op_type] = OperationProfile(operation_type=op_type)
            
            logger.info("Performance monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start performance monitoring: {e}")
            return False
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        try:
            self.is_monitoring = False
            
            # Generate final performance report
            final_report = await self.generate_performance_report()
            
            logger.info("Performance monitoring stopped")
            return final_report
            
        except Exception as e:
            logger.error(f"Error stopping performance monitoring: {e}")
            return None
    
    def start_operation(self, operation_type: OperationType, 
                       operation_id: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking an operation"""
        if not self.is_monitoring:
            return ""
        
        op_id = operation_id or str(uuid.uuid4())
        
        self.active_operations[op_id] = {
            "operation_type": operation_type,
            "start_time": time.time(),
            "start_timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }
        
        return op_id
    
    def end_operation(self, operation_id: str, 
                     success: bool = True,
                     error_message: Optional[str] = None,
                     result_metadata: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """End tracking an operation and record metrics"""
        if not self.is_monitoring or operation_id not in self.active_operations:
            return None
        
        try:
            operation_data = self.active_operations.pop(operation_id)
            operation_type = operation_data["operation_type"]
            
            # Calculate execution time
            execution_time_ms = (time.time() - operation_data["start_time"]) * 1000
            
            # Update operation profile
            profile = self.operation_profiles[operation_type]
            profile.total_executions += 1
            profile.total_time_ms += execution_time_ms
            profile.avg_time_ms = profile.total_time_ms / profile.total_executions
            profile.min_time_ms = min(profile.min_time_ms, execution_time_ms)
            profile.max_time_ms = max(profile.max_time_ms, execution_time_ms)
            profile.last_execution = datetime.utcnow()
            
            if not success:
                profile.error_count += 1
            
            profile.success_rate = ((profile.total_executions - profile.error_count) / 
                                  profile.total_executions) * 100
            
            # Record performance metric
            metric = PerformanceMetric(
                metric_id=str(uuid.uuid4()),
                metric_type=MetricType.OPERATION_TIME,
                operation_type=operation_type,
                value=execution_time_ms,
                timestamp=datetime.utcnow(),
                metadata={
                    "success": success,
                    "error_message": error_message,
                    "result_metadata": result_metadata or {},
                    **operation_data["metadata"]
                }
            )
            
            self.metrics[f"{operation_type.value}_time"].append(metric)
            
            # Check for performance alerts
            await self._check_performance_alerts(operation_type, execution_time_ms, success)
            
            return execution_time_ms
            
        except Exception as e:
            logger.error(f"Error ending operation {operation_id}: {e}")
            return None
    
    async def record_throughput_metric(self, operation_type: OperationType, 
                                     operations_per_second: float,
                                     metadata: Optional[Dict[str, Any]] = None):
        """Record throughput metric"""
        if not self.is_monitoring:
            return
        
        metric = PerformanceMetric(
            metric_id=str(uuid.uuid4()),
            metric_type=MetricType.THROUGHPUT,
            operation_type=operation_type,
            value=operations_per_second,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.metrics[f"{operation_type.value}_throughput"].append(metric)
    
    async def record_cache_performance(self, cache_name: str, 
                                     hit_rate: float,
                                     total_requests: int,
                                     metadata: Optional[Dict[str, Any]] = None):
        """Record cache performance metrics"""
        if not self.is_monitoring:
            return
        
        metric = PerformanceMetric(
            metric_id=str(uuid.uuid4()),
            metric_type=MetricType.CACHE_HIT_RATE,
            operation_type=OperationType.MEMORY_RETRIEVE,  # Default operation type
            value=hit_rate,
            timestamp=datetime.utcnow(),
            metadata={
                "cache_name": cache_name,
                "total_requests": total_requests,
                **(metadata or {})
            }
        )
        
        self.metrics[f"{cache_name}_cache_performance"].append(metric)
        
        # Check cache performance alert
        if hit_rate < self.alert_thresholds.get("min_cache_hit_rate", 80.0):
            await self._create_alert(
                alert_type="cache_performance",
                operation_type=OperationType.MEMORY_RETRIEVE,
                threshold_value=self.alert_thresholds["min_cache_hit_rate"],
                actual_value=hit_rate,
                severity="warning",
                message=f"Cache hit rate for {cache_name} below threshold: {hit_rate}%"
            )
    
    async def record_consolidation_efficiency(self, 
                                            memories_processed: int,
                                            patterns_extracted: int,
                                            processing_time_ms: float,
                                            metadata: Optional[Dict[str, Any]] = None):
        """Record consolidation efficiency metrics"""
        if not self.is_monitoring:
            return
        
        efficiency_score = (patterns_extracted / memories_processed) * 100 if memories_processed > 0 else 0
        
        metric = PerformanceMetric(
            metric_id=str(uuid.uuid4()),
            metric_type=MetricType.CONSOLIDATION_EFFICIENCY,
            operation_type=OperationType.CONSOLIDATION,
            value=efficiency_score,
            timestamp=datetime.utcnow(),
            metadata={
                "memories_processed": memories_processed,
                "patterns_extracted": patterns_extracted,
                "processing_time_ms": processing_time_ms,
                **(metadata or {})
            }
        )
        
        self.metrics["consolidation_efficiency"].append(metric)
    
    async def record_memory_usage(self, component_name: str, 
                                memory_usage_mb: float,
                                metadata: Optional[Dict[str, Any]] = None):
        """Record memory usage metrics"""
        if not self.is_monitoring:
            return
        
        metric = PerformanceMetric(
            metric_id=str(uuid.uuid4()),
            metric_type=MetricType.MEMORY_USAGE,
            operation_type=OperationType.MEMORY_STORE,  # Default operation type
            value=memory_usage_mb,
            timestamp=datetime.utcnow(),
            metadata={
                "component_name": component_name,
                **(metadata or {})
            }
        )
        
        self.metrics[f"{component_name}_memory_usage"].append(metric)
        
        # Check memory usage alert
        if memory_usage_mb > self.alert_thresholds.get("max_memory_usage_mb", 1000):
            await self._create_alert(
                alert_type="memory_usage",
                operation_type=OperationType.MEMORY_STORE,
                threshold_value=self.alert_thresholds["max_memory_usage_mb"],
                actual_value=memory_usage_mb,
                severity="critical",
                message=f"Memory usage for {component_name} exceeds threshold: {memory_usage_mb}MB"
            )
    
    async def _check_performance_alerts(self, operation_type: OperationType, 
                                      execution_time_ms: float, success: bool):
        """Check for performance threshold violations"""
        
        # Check operation time threshold
        max_time_threshold = self.alert_thresholds.get("max_operation_time_ms", 5000)
        if execution_time_ms > max_time_threshold:
            await self._create_alert(
                alert_type="slow_operation",
                operation_type=operation_type,
                threshold_value=max_time_threshold,
                actual_value=execution_time_ms,
                severity="warning",
                message=f"Operation {operation_type.value} exceeded time threshold: {execution_time_ms:.2f}ms"
            )
        
        # Check success rate
        profile = self.operation_profiles[operation_type]
        if profile.total_executions >= 10:  # Only check after sufficient operations
            min_success_rate = self.alert_thresholds.get("min_success_rate", 95.0)
            if profile.success_rate < min_success_rate:
                await self._create_alert(
                    alert_type="low_success_rate",
                    operation_type=operation_type,
                    threshold_value=min_success_rate,
                    actual_value=profile.success_rate,
                    severity="critical",
                    message=f"Success rate for {operation_type.value} below threshold: {profile.success_rate:.2f}%"
                )
    
    async def _create_alert(self, alert_type: str, operation_type: OperationType,
                          threshold_value: float, actual_value: float,
                          severity: str, message: str):
        """Create a performance alert"""
        
        alert = PerformanceAlert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            operation_type=operation_type,
            threshold_value=threshold_value,
            actual_value=actual_value,
            timestamp=datetime.utcnow(),
            severity=severity,
            message=message
        )
        
        self.alerts.append(alert)
        
        # Keep only recent alerts (last 1000)
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]
        
        logger.warning(f"Performance Alert [{severity}]: {message}")
    
    async def get_operation_statistics(self, operation_type: OperationType) -> Dict[str, Any]:
        """Get detailed statistics for an operation type"""
        
        if operation_type not in self.operation_profiles:
            return {}
        
        profile = self.operation_profiles[operation_type]
        
        # Get recent metrics for detailed analysis
        time_metrics = [
            m for m in self.metrics.get(f"{operation_type.value}_time", [])
            if m.timestamp > datetime.utcnow() - timedelta(hours=24)
        ]
        
        if time_metrics:
            recent_times = [m.value for m in time_metrics]
            percentiles = {
                "p50": statistics.median(recent_times),
                "p95": statistics.quantiles(recent_times, n=20)[18] if len(recent_times) >= 20 else max(recent_times),
                "p99": statistics.quantiles(recent_times, n=100)[98] if len(recent_times) >= 100 else max(recent_times)
            }
        else:
            percentiles = {"p50": 0, "p95": 0, "p99": 0}
        
        return {
            "operation_type": operation_type.value,
            "total_executions": profile.total_executions,
            "average_time_ms": profile.avg_time_ms,
            "min_time_ms": profile.min_time_ms if profile.min_time_ms != float('inf') else 0,
            "max_time_ms": profile.max_time_ms,
            "success_rate": profile.success_rate,
            "error_count": profile.error_count,
            "last_execution": profile.last_execution.isoformat() if profile.last_execution else None,
            "percentiles": percentiles,
            "recent_metrics_count": len(time_metrics)
        }
    
    async def get_recent_alerts(self, hours: int = 24, 
                              severity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent performance alerts"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alerts
            if alert.timestamp > cutoff_time
        ]
        
        if severity_filter:
            recent_alerts = [
                alert for alert in recent_alerts
                if alert.severity == severity_filter
            ]
        
        return [
            {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "operation_type": alert.operation_type.value,
                "threshold_value": alert.threshold_value,
                "actual_value": alert.actual_value,
                "timestamp": alert.timestamp.isoformat(),
                "severity": alert.severity,
                "message": alert.message
            }
            for alert in sorted(recent_alerts, key=lambda a: a.timestamp, reverse=True)
        ]
    
    async def generate_performance_report(self, 
                                        include_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=include_hours)
            
            # Overall statistics
            total_operations = sum(profile.total_executions for profile in self.operation_profiles.values())
            total_errors = sum(profile.error_count for profile in self.operation_profiles.values())
            overall_success_rate = ((total_operations - total_errors) / total_operations * 100) if total_operations > 0 else 100
            
            # Operation-specific statistics
            operation_stats = {}
            for op_type in OperationType:
                operation_stats[op_type.value] = await self.get_operation_statistics(op_type)
            
            # Recent alerts summary
            recent_alerts = await self.get_recent_alerts(hours=include_hours)
            alert_summary = {
                "total_alerts": len(recent_alerts),
                "critical_alerts": len([a for a in recent_alerts if a["severity"] == "critical"]),
                "warning_alerts": len([a for a in recent_alerts if a["severity"] == "warning"]),
                "recent_alerts": recent_alerts[:10]  # Top 10 recent alerts
            }
            
            # Performance trends
            trends = await self._calculate_performance_trends(cutoff_time)
            
            # System health score
            health_score = await self._calculate_system_health_score()
            
            return {
                "report_generated_at": datetime.utcnow().isoformat(),
                "monitoring_period_hours": include_hours,
                "monitoring_duration_hours": (datetime.utcnow() - self.monitoring_start_time).total_seconds() / 3600,
                "overall_statistics": {
                    "total_operations": total_operations,
                    "total_errors": total_errors,
                    "overall_success_rate": overall_success_rate,
                    "system_health_score": health_score
                },
                "operation_statistics": operation_stats,
                "alert_summary": alert_summary,
                "performance_trends": trends,
                "active_operations_count": len(self.active_operations),
                "monitoring_status": "active" if self.is_monitoring else "stopped"
            }
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": str(e)}
    
    async def _calculate_performance_trends(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        
        trends = {}
        
        for op_type in OperationType:
            time_metrics = [
                m for m in self.metrics.get(f"{op_type.value}_time", [])
                if m.timestamp > cutoff_time
            ]
            
            if len(time_metrics) >= 2:
                # Simple trend calculation (first half vs second half)
                sorted_metrics = sorted(time_metrics, key=lambda m: m.timestamp)
                mid_point = len(sorted_metrics) // 2
                
                first_half_avg = statistics.mean([m.value for m in sorted_metrics[:mid_point]])
                second_half_avg = statistics.mean([m.value for m in sorted_metrics[mid_point:]])
                
                trend_direction = "improving" if second_half_avg < first_half_avg else "degrading"
                trend_percentage = abs((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0
                
                trends[op_type.value] = {
                    "trend_direction": trend_direction,
                    "trend_percentage": trend_percentage,
                    "first_half_avg_ms": first_half_avg,
                    "second_half_avg_ms": second_half_avg,
                    "sample_size": len(time_metrics)
                }
        
        return trends
    
    async def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        
        try:
            scores = []
            
            # Success rate score
            total_ops = sum(p.total_executions for p in self.operation_profiles.values())
            total_errors = sum(p.error_count for p in self.operation_profiles.values())
            success_rate = ((total_ops - total_errors) / total_ops * 100) if total_ops > 0 else 100
            scores.append(min(success_rate, 100))
            
            # Performance score (based on recent operation times)
            performance_scores = []
            for op_type in OperationType:
                profile = self.operation_profiles[op_type]
                if profile.avg_time_ms > 0:
                    # Score based on how close to minimum time (lower is better)
                    if profile.min_time_ms > 0:
                        perf_score = max(0, 100 - ((profile.avg_time_ms - profile.min_time_ms) / profile.min_time_ms * 50))
                        performance_scores.append(perf_score)
            
            if performance_scores:
                scores.append(statistics.mean(performance_scores))
            
            # Alert score (fewer alerts is better)
            recent_alerts = await self.get_recent_alerts(hours=1)
            alert_penalty = min(len(recent_alerts) * 5, 50)  # Max 50 point penalty
            alert_score = max(0, 100 - alert_penalty)
            scores.append(alert_score)
            
            # Overall health score
            return statistics.mean(scores) if scores else 100.0
            
        except Exception as e:
            logger.error(f"Error calculating system health score: {e}")
            return 50.0  # Default neutral score
    
    def update_alert_thresholds(self, thresholds: Dict[str, float]) -> bool:
        """Update performance alert thresholds"""
        try:
            self.alert_thresholds.update(thresholds)
            logger.info(f"Updated alert thresholds: {thresholds}")
            return True
        except Exception as e:
            logger.error(f"Error updating alert thresholds: {e}")
            return False
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "is_monitoring": self.is_monitoring,
            "monitoring_start_time": self.monitoring_start_time.isoformat(),
            "monitoring_duration_hours": (datetime.utcnow() - self.monitoring_start_time).total_seconds() / 3600,
            "active_operations_count": len(self.active_operations),
            "total_metrics_stored": sum(len(metrics) for metrics in self.metrics.values()),
            "total_alerts": len(self.alerts),
            "alert_thresholds": self.alert_thresholds
        }