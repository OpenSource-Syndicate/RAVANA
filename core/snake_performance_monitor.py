"""
Snake Agent Performance Monitoring and Metrics Collection

This module provides tools for monitoring Snake Agent performance,
collecting metrics, and analyzing system behavior in real-time.
"""

import asyncio
import time
import psutil
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque
import threading
import statistics

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.snake_agent_enhanced import EnhancedSnakeAgent


class SnakeAgentPerformanceMonitor:
    """Performance monitor for the Snake Agent"""
    
    def __init__(self, agent: EnhancedSnakeAgent):
        self.agent = agent
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 metrics
        self.start_time = time.time()
        
        # Initialize process for system metrics
        self.process = psutil.Process()
    
    def collect_metrics(self):
        """Collect current performance metrics"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Collect system metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        current_memory_mb = memory_info.used / 1024 / 1024  # Convert to MB
        
        # Calculate rates based on uptime
        improvements_per_hour = (self.agent.improvements_applied / max(uptime / 3600, 1)) if hasattr(self.agent, 'improvements_applied') else 0
        experiments_per_hour = (self.agent.experiments_completed / max(uptime / 3600, 1)) if hasattr(self.agent, 'experiments_completed') else 0
        
        # Create performance metrics object
        metrics = {
            "timestamp": time.time(),
            "uptime_seconds": uptime,
            "current_time": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "current_memory_mb": current_memory_mb,
            "total_improvements": getattr(self.agent, 'improvements_applied', 0),
            "total_experiments": getattr(self.agent, 'experiments_completed', 0),
            "improvements_per_hour": improvements_per_hour,
            "experiments_per_hour": experiments_per_hour,
            "files_analyzed": getattr(self.agent, 'files_analyzed', 0),
            "communications_sent": getattr(self.agent, 'communications_sent', 0),
            "processing_rate_per_minute": experiments_per_hour / 60 if experiments_per_hour > 0 else 0,
            "vltm_enabled": getattr(self.agent, 'vltm_enabled', False),
            "agent_running": getattr(self.agent, 'running', False)
        }
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.metrics_history:
            return {}
        
        # Extract values
        cpu_values = [m["cpu_percent"] for m in self.metrics_history]
        memory_values = [m["current_memory_mb"] for m in self.metrics_history]
        improvements_values = [m["improvements_per_hour"] for m in self.metrics_history]
        experiments_values = [m["experiments_per_hour"] for m in self.metrics_history]
        
        return {
            "total_uptime_minutes": self.metrics_history[-1]["uptime_seconds"] / 60 if self.metrics_history else 0,
            "total_tasks_processed": sum(m["total_experiments"] for m in self.metrics_history),
            "total_improvements": sum(m["total_improvements"] for m in self.metrics_history),
            "peak_memory_mb": max(memory_values) if memory_values else 0,
            "avg_cpu_percent": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            "avg_memory_mb": sum(memory_values) / len(memory_values) if memory_values else 0,
            "avg_improvements_per_hour": sum(improvements_values) / len(improvements_values) if improvements_values else 0,
            "avg_experiments_per_hour": sum(experiments_values) / len(experiments_values) if experiments_values else 0,
            "latest_metrics": self.metrics_history[-1] if self.metrics_history else None
        }
    
    def export_metrics(self, filename: str):
        """Export collected metrics to a file"""
        metrics_data = {
            "export_time": datetime.now().isoformat(),
            "agent_info": {
                "type": "EnhancedSnakeAgent",
                "initialization_time": getattr(self.agent, 'start_time', 'N/A').isoformat() if hasattr(self.agent, 'start_time') and self.agent.start_time else 'N/A',
                "running": getattr(self.agent, 'running', False)
            },
            "metrics_history": list(self.metrics_history)
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        print(f"Metrics exported to {filename}")


@dataclass
class SystemMetrics:
    """System-level metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int


@dataclass
class AgentMetrics:
    """Snake Agent-specific metrics"""
    timestamp: datetime
    agent_uptime: float  # seconds
    improvements_applied: int
    experiments_completed: int
    files_analyzed: int
    communications_sent: int
    improvements_per_hour: float
    experiments_per_hour: float
    files_analyzed_per_hour: float
    vltm_enabled: bool
    vltm_memory_count: Optional[int] = None
    threading_active: bool = False
    thread_count: Optional[int] = None
    process_active: bool = False
    process_count: Optional[int] = None
    active_tasks: Optional[int] = None
    queue_sizes: Optional[Dict[str, int]] = None
    error_count: Optional[int] = None


@dataclass
class PerformanceMetrics:
    """Performance-focused metrics"""
    timestamp: datetime
    response_time: float  # seconds
    processing_rate: float  # operations per second
    throughput: float  # items processed per second
    resource_utilization: Dict[str, float]  # cpu, memory, etc.
    latency_percentiles: Dict[str, float]  # p50, p95, p99, etc.
    error_rate: float
    success_rate: float


class MetricsCollector:
    """Collects metrics from various sources"""
    
    def __init__(self, agent: EnhancedSnakeAgent, collection_interval: float = 5.0):
        self.agent = agent
        self.collection_interval = collection_interval
        self.running = False
        self.collection_task: Optional[asyncio.Task] = None
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 metrics
        self.system_history: deque = deque(maxlen=1000)
        self.agent_history: deque = deque(maxlen=1000)
        
        # Initialize process for system metrics
        self.process = psutil.Process()
    
    async def start_collection(self):
        """Start the metrics collection loop"""
        if self.running:
            return
        
        self.running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        print("Metrics collector started")
    
    async def stop_collection(self):
        """Stop the metrics collection loop"""
        if not self.running:
            return
        
        self.running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling
        
        print("Metrics collector stopped")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Collect metrics
                system_metrics = self._collect_system_metrics()
                agent_metrics = await self._collect_agent_metrics()
                
                # Store metrics
                self.system_history.append(system_metrics)
                self.agent_history.append(agent_metrics)
                
                # Sleep for the interval
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                print(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-level metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / 1024 / 1024  # Convert to MB
        
        # Disk usage
        disk_usage = psutil.disk_usage('/').percent if os.name == 'posix' else psutil.disk_usage('C:\\').percent
        
        # Network I/O
        net_io = psutil.net_io_counters()
        network_io = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv
        }
        
        # Process count
        process_count = len(psutil.pids())
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            disk_usage_percent=disk_usage,
            network_io=network_io,
            process_count=process_count
        )
    
    async def _collect_agent_metrics(self) -> AgentMetrics:
        """Collect agent-specific metrics"""
        # Get agent status
        status = await self.agent.get_status() if self.agent else {}
        
        # Extract metrics from status
        uptime = status.get("uptime", 0)
        metrics = status.get("metrics", {})
        
        # Calculate derived metrics
        improvements_applied = metrics.get("improvements_applied", 0)
        experiments_completed = metrics.get("experiments_completed", 0)
        files_analyzed = metrics.get("files_analyzed", 0)
        communications_sent = metrics.get("communications_sent", 0)
        
        # Calculate rates (if uptime is available)
        improvements_per_hour = (improvements_applied / max(uptime / 3600, 1)) if uptime > 0 else 0
        experiments_per_hour = (experiments_completed / max(uptime / 3600, 1)) if uptime > 0 else 0
        files_analyzed_per_hour = (files_analyzed / max(uptime / 3600, 1)) if uptime > 0 else 0
        
        # Component status
        components = status.get("components", {})
        threading_active = components.get("threading_manager", False)
        process_active = components.get("process_manager", False)

        # Get queue sizes
        queue_sizes = {}
        if "thread_queues" in status:
            queue_sizes.update(status["thread_queues"])
        if "process_queues" in status:
            queue_sizes.update(status["process_queues"])

        return AgentMetrics(
            timestamp=datetime.now(),
            agent_uptime=uptime,
            improvements_applied=improvements_applied,
            experiments_completed=experiments_completed,
            files_analyzed=files_analyzed,
            communications_sent=communications_sent,
            improvements_per_hour=improvements_per_hour,
            experiments_per_hour=experiments_per_hour,
            files_analyzed_per_hour=files_analyzed_per_hour,
            vltm_enabled=status.get("vltm_status", {}).get("enabled", False),
            threading_active=threading_active,
            process_active=process_active,
            queue_sizes=queue_sizes,
        )


class PerformanceAnalyzer:
    """Analyzes collected metrics to identify performance trends and issues"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.collector = metrics_collector
        self.performance_thresholds = {
            "cpu_percent": 80.0,  # Alert if CPU usage exceeds 80%
            "memory_mb": 2048.0,  # Alert if memory usage exceeds 2GB
            "response_time": 5.0,  # Alert if response time exceeds 5s
            "error_rate": 0.05,    # Alert if error rate exceeds 5%
            "processing_rate": 0.1 # Alert if processing rate drops below 0.1 ops/sec
        }
    
    def get_system_trends(self, minutes: int = 10) -> Dict[str, Any]:
        """Get system performance trends for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_system_metrics = [
            metric for metric in self.collector.system_history
            if metric.timestamp >= cutoff_time
        ]
        
        if not recent_system_metrics:
            return {}
        
        # Calculate statistics
        cpu_percentages = [m.cpu_percent for m in recent_system_metrics]
        memory_mb_values = [m.memory_mb for m in recent_system_metrics]
        
        trends = {
            "cpu": {
                "avg": statistics.mean(cpu_percentages),
                "max": max(cpu_percentages),
                "min": min(cpu_percentages),
                "current": recent_system_metrics[-1].cpu_percent,
                "trend": self._calculate_trend(cpu_percentages)
            },
            "memory": {
                "avg": statistics.mean(memory_mb_values),
                "max": max(memory_mb_values),
                "min": min(memory_mb_values),
                "current": recent_system_metrics[-1].memory_mb,
                "trend": self._calculate_trend(memory_mb_values)
            },
            "period": f"last {minutes} minutes",
            "sample_count": len(recent_system_metrics)
        }
        
        return trends
    
    def get_agent_trends(self, minutes: int = 10) -> Dict[str, Any]:
        """Get agent performance trends for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_agent_metrics = [
            metric for metric in self.collector.agent_history
            if metric.timestamp >= cutoff_time
        ]
        
        if not recent_agent_metrics:
            return {}
        
        # Calculate statistics for performance metrics
        improvements_per_hour = [m.improvements_per_hour for m in recent_agent_metrics]
        experiments_per_hour = [m.experiments_per_hour for m in recent_agent_metrics]
        files_analyzed_per_hour = [m.files_analyzed_per_hour for m in recent_agent_metrics]
        
        trends = {
            "improvements": {
                "avg": statistics.mean(improvements_per_hour),
                "max": max(improvements_per_hour),
                "min": min(improvements_per_hour),
                "current": recent_agent_metrics[-1].improvements_per_hour,
                "trend": self._calculate_trend(improvements_per_hour)
            },
            "experiments": {
                "avg": statistics.mean(experiments_per_hour),
                "max": max(experiments_per_hour),
                "min": min(experiments_per_hour),
                "current": recent_agent_metrics[-1].experiments_per_hour,
                "trend": self._calculate_trend(experiments_per_hour)
            },
            "analysis": {
                "avg": statistics.mean(files_analyzed_per_hour),
                "max": max(files_analyzed_per_hour),
                "min": min(files_analyzed_per_hour),
                "current": recent_agent_metrics[-1].files_analyzed_per_hour,
                "trend": self._calculate_trend(files_analyzed_per_hour)
            },
            "period": f"last {minutes} minutes",
            "sample_count": len(recent_agent_metrics)
        }
        
        return trends
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get any performance alerts based on thresholds"""
        alerts = []
        
        if not self.collector.system_history or not self.collector.agent_history:
            return alerts
        
        # Get the latest metrics
        latest_system = self.collector.system_history[-1]
        latest_agent = self.collector.agent_history[-1]
        
        # Check system metrics against thresholds
        if latest_system.cpu_percent > self.performance_thresholds["cpu_percent"]:
            alerts.append({
                "type": "high_cpu",
                "severity": "warning",
                "message": f"High CPU usage: {latest_system.cpu_percent:.1f}%",
                "value": latest_system.cpu_percent,
                "threshold": self.performance_thresholds["cpu_percent"],
                "timestamp": latest_system.timestamp
            })
        
        if latest_system.memory_mb > self.performance_thresholds["memory_mb"]:
            alerts.append({
                "type": "high_memory",
                "severity": "warning",
                "message": f"High memory usage: {latest_system.memory_mb:.1f} MB",
                "value": latest_system.memory_mb,
                "threshold": self.performance_thresholds["memory_mb"],
                "timestamp": latest_system.timestamp
            })
        
        # Check agent metrics against thresholds
        if latest_agent.experiments_per_hour < self.performance_thresholds["processing_rate"]:
            alerts.append({
                "type": "low_throughput",
                "severity": "warning",
                "message": f"Low processing rate: {latest_agent.experiments_per_hour:.2f} exp/hour",
                "value": latest_agent.experiments_per_hour,
                "threshold": self.performance_thresholds["processing_rate"],
                "timestamp": latest_agent.timestamp
            })
        
        return alerts
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction based on last 5 values"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Use the last few values to determine trend
        recent_values = values[-5:] if len(values) >= 5 else values
        
        if len(recent_values) < 2:
            return "stable"
        
        # Simple linear regression to determine trend
        n = len(recent_values)
        x = list(range(n))
        y = recent_values
        
        # Calculate slope (m) of linear regression line
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x2 = sum(i**2 for i in x)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2) if (n * sum_x2 - sum_x**2) != 0 else 0
        
        if slope > 0.1:  # Positive trend
            return "increasing"
        elif slope < -0.1:  # Negative trend
            return "decreasing"
        else:  # Stable
            return "stable"
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "agent_uptime_hours": self.collector.agent.start_time.total_seconds() / 3600 if self.collector.agent.start_time else 0,
            "system_trends_10min": self.get_system_trends(10),
            "agent_trends_10min": self.get_agent_trends(10),
            "system_trends_60min": self.get_system_trends(60),
            "agent_trends_60min": self.get_agent_trends(60),
            "current_alerts": self.get_alerts(),
            "total_samples": {
                "system": len(self.collector.system_history),
                "agent": len(self.collector.agent_history)
            }
        }
        
        return report


class MetricsDashboard:
    """Real-time dashboard for monitoring Snake Agent performance"""
    
    def __init__(self, analyzer: PerformanceAnalyzer):
        self.analyzer = analyzer
        self.refresh_interval = 2.0  # seconds
        self.running = False
        self.display_task: Optional[asyncio.Task] = None
    
    async def start_dashboard(self):
        """Start the dashboard display"""
        if self.running:
            return
        
        self.running = True
        self.display_task = asyncio.create_task(self._display_loop())
        print("Performance dashboard started")
    
    async def stop_dashboard(self):
        """Stop the dashboard display"""
        if not self.running:
            return
        
        self.running = False
        if self.display_task:
            self.display_task.cancel()
            try:
                await self.display_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling
    
    async def _display_loop(self):
        """Main display loop"""
        while self.running:
            try:
                # Clear screen (works on Unix and Windows)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                # Print dashboard header
                print("=" * 80)
                print("SNAKE AGENT PERFORMANCE DASHBOARD")
                print("=" * 80)
                
                # Print system metrics
                await self._print_system_metrics()
                print()
                
                # Print agent metrics
                await self._print_agent_metrics()
                print()

                # Print queue status
                await self._print_queue_status()
                print()
                
                # Print alerts
                self._print_alerts()
                print()
                
                # Print trends
                await self._print_trends()
                
                # Print footer
                print("=" * 80)
                print("Press Ctrl+C to exit")
                
                # Sleep for refresh interval
                await asyncio.sleep(self.refresh_interval)
                
            except Exception as e:
                print(f"Error in dashboard display: {e}")
                await asyncio.sleep(self.refresh_interval)
    
    async def _print_system_metrics(self):
        """Print current system metrics"""
        if not self.analyzer.collector.system_history:
            print("No system metrics available yet...")
            return
        
        latest = self.analyzer.collector.system_history[-1]
        
        print(f"SYSTEM METRICS (as of {latest.timestamp.strftime('%H:%M:%S')}):")
        print(f"  CPU Usage:    {latest.cpu_percent:6.2f}%")
        print(f"  Memory:       {latest.memory_mb:8.2f} MB ({latest.memory_percent:5.2f}%)")
        print(f"  Disk Usage:   {latest.disk_usage_percent:6.2f}%")
        print(f"  Process Count: {latest.process_count}")
    
    async def _print_agent_metrics(self):
        """Print current agent metrics"""
        if not self.analyzer.collector.agent_history:
            print("No agent metrics available yet...")
            return
        
        latest = self.analyzer.collector.agent_history[-1]
        
        print(f"AGENT METRICS (Uptime: {latest.agent_uptime / 3600:.2f}h):")
        print(f"  Improvements: {latest.improvements_applied:6d} (Rate: {latest.improvements_per_hour:5.2f}/h)")
        print(f"  Experiments:  {latest.experiments_completed:6d} (Rate: {latest.experiments_per_hour:5.2f}/h)")
        print(f"  Files Analyzed: {latest.files_analyzed:4d} (Rate: {latest.files_analyzed_per_hour:5.2f}/h)")
        print(f"  Communications: {latest.communications_sent:4d}")
        print(f"  VLTM Enabled: {'Yes' if latest.vltm_enabled else 'No'}")
        print(f"  Threading Active: {'Yes' if latest.threading_active else 'No'}")
        print(f"  Process Active: {'Yes' if latest.process_active else 'No'}")

    async def _print_queue_status(self):
        """Print current queue status"""
        if not self.analyzer.collector.agent_history:
            print("No agent metrics available yet...")
            return

        latest = self.analyzer.collector.agent_history[-1]
        queue_sizes = latest.queue_sizes

        print("QUEUE STATUS:")
        if not queue_sizes:
            print("  No queue information available.")
            return

        for queue_name, size in queue_sizes.items():
            print(f"  {queue_name:<25}: {size}")
    
    def _print_alerts(self):
        """Print current performance alerts"""
        alerts = self.analyzer.get_alerts()
        
        if not alerts:
            print("No performance alerts")
        else:
            print(f"PERFORMANCE ALERTS ({len(alerts)}):")
            for alert in alerts:
                severity = alert['severity'].upper()
                print(f"  [{severity}] {alert['message']}")
    
    async def _print_trends(self):
        """Print performance trends"""
        # Get 10-minute trends
        agent_trends = self.analyzer.get_agent_trends(10)
        system_trends = self.analyzer.get_system_trends(10)
        
        print(f"TRENDS (last 10 minutes):")
        if agent_trends:
            print(f"  Improvements: {agent_trends['improvements']['trend']} (avg: {agent_trends['improvements']['avg']:.2f}/h)")
            print(f"  Experiments:  {agent_trends['experiments']['trend']} (avg: {agent_trends['experiments']['avg']:.2f}/h)")
            print(f"  Analysis:     {agent_trends['analysis']['trend']} (avg: {agent_trends['analysis']['avg']:.2f}/h)")
        if system_trends:
            print(f"  CPU:          {system_trends['cpu']['trend']} (avg: {system_trends['cpu']['avg']:.1f}%)")
            print(f"  Memory:       {system_trends['memory']['trend']} (avg: {system_trends['memory']['avg']:.1f} MB)")


class MetricsExporter:
    """Exports metrics to files in various formats"""
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.export_dir = Path("metrics_exports")
        self.export_dir.mkdir(exist_ok=True)
    
    def export_to_json(self, filename: str = None) -> str:
        """Export metrics to JSON format"""
        if filename is None:
            filename = f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_path = self.export_dir / filename
        
        # Prepare data for export
        export_data = {
            "export_time": datetime.now().isoformat(),
            "system_metrics": [asdict(m) for m in self.collector.system_history],
            "agent_metrics": [asdict(m) for m in self.collector.agent_history]
        }
        
        # Convert datetime objects to ISO strings
        for metrics_list in [export_data["system_metrics"], export_data["agent_metrics"]]:
            for metric in metrics_list:
                if "timestamp" in metric:
                    metric["timestamp"] = metric["timestamp"].isoformat()
        
        # Write to file
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Metrics exported to {export_path}")
        return str(export_path)
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of the collected metrics"""
        if not self.collector.system_history or not self.collector.agent_history:
            return "No metrics available for summary."
        
        latest_system = self.collector.system_history[-1]
        latest_agent = self.collector.agent_history[-1]
        
        # Calculate averages
        cpu_values = [m.cpu_percent for m in self.collector.system_history]
        memory_values = [m.memory_mb for m in self.collector.system_history]
        
        avg_cpu = statistics.mean(cpu_values)
        avg_memory = statistics.mean(memory_values)
        max_cpu = max(cpu_values)
        max_memory = max(memory_values)
        
        # Generate report
        report = f"""
SNAKE AGENT PERFORMANCE SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
===============================================

SYSTEM PERFORMANCE:
  Current CPU: {latest_system.cpu_percent:.2f}%
  Current Memory: {latest_system.memory_mb:.2f} MB
  Average CPU: {avg_cpu:.2f}%
  Average Memory: {avg_memory:.2f} MB
  Peak CPU: {max_cpu:.2f}%
  Peak Memory: {max_memory:.2f} MB

AGENT PERFORMANCE:
  Uptime: {latest_agent.agent_uptime / 3600:.2f} hours
  Total Improvements: {latest_agent.improvements_applied}
  Total Experiments: {latest_agent.experiments_completed}
  Total Files Analyzed: {latest_agent.files_analyzed}
  Improvements Rate: {latest_agent.improvements_per_hour:.2f} per hour
  Experiments Rate: {latest_agent.experiments_per_hour:.2f} per hour
  Analysis Rate: {latest_agent.files_analyzed_per_hour:.2f} per hour

SUMMARY:
  - Collected {len(self.collector.system_history)} system metric samples
  - Collected {len(self.collector.agent_history)} agent metric samples
  - Monitoring duration: {latest_agent.agent_uptime:.2f} seconds
        """
        
        # Write report to file
        report_filename = f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = self.export_dir / report_filename
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Summary report generated: {report_path}")
        return str(report_path)


async def run_monitoring_demo(agent: EnhancedSnakeAgent):
    """Run a demo of the monitoring system"""
    print("Starting Snake Agent monitoring demo...")
    
    # Create metrics collector
    collector = MetricsCollector(agent, collection_interval=2.0)
    await collector.start_collection()
    
    # Create performance analyzer
    analyzer = PerformanceAnalyzer(collector)
    
    # Create metrics exporter
    exporter = MetricsExporter(collector)
    
    try:
        # Let it run for a bit
        print("Collecting metrics for 15 seconds...")
        await asyncio.sleep(15)
        
        # Generate report
        print("\nGenerating performance report...")
        report_path = exporter.generate_summary_report()
        
        # Export metrics
        print("Exporting metrics to JSON...")
        json_path = exporter.export_to_json()
        
        # Print trends
        print("\nRecent trends:")
        agent_trends = analyzer.get_agent_trends(5)
        if agent_trends:
            for category, data in agent_trends.items():
                if 'trend' in data:
                    print(f"  {category}: {data['trend']} (current: {data['current']:.2f})")
        
        print(f"\nReport saved to: {report_path}")
        print(f"JSON data saved to: {json_path}")
        
    finally:
        await collector.stop_collection()


if __name__ == "__main__":
    # Import required modules for the test
    from unittest.mock import Mock
    
    # Create a mock agent for demonstration
    async def demo():
        # Mock AGI system
        mock_agi = Mock()
        mock_agi.workspace_path = os.getcwd()
        mock_agi.engine = Mock()
        
        # Create EnhancedSnakeAgent
        from core.snake_agent_enhanced import EnhancedSnakeAgent
        from datetime import timedelta
        
        # Patch the LLM creation since we're just demoing
        with patch('core.snake_llm.create_snake_coding_llm') as mock_coding_llm, \
             patch('core.snake_llm.create_snake_reasoning_llm') as mock_reasoning_llm:
            
            mock_coding_llm.return_value = Mock()
            mock_reasoning_llm.return_value = Mock()
            
            agent = EnhancedSnakeAgent(mock_agi)
            agent.start_time = datetime.now() - timedelta(minutes=5)  # Simulate 5 min uptime
            
            # Initialize basic required properties
            agent.running = True
            agent.start_time = datetime.now() - timedelta(seconds=300)  # 5 minutes ago
            agent.improvements_applied = 15
            agent.experiments_completed = 42
            agent.files_analyzed = 28
            agent.communications_sent = 8
            
            # Test the monitoring system
            await run_monitoring_demo(agent)
    
    from unittest.mock import patch
    asyncio.run(demo())