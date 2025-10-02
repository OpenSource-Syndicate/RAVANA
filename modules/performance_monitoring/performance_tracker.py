"""
Performance Monitoring and Tracking Module for RAVANA AGI System

This module provides comprehensive performance monitoring, metrics collection, 
and improvement tracking for the AGI system.
"""
import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Represents a single performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    source: str
    tags: List[str] = None
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class ImprovementMetric:
    """Tracks improvement-specific metrics"""
    improvement_id: str
    type: str
    impact_score: float
    confidence: float
    implementation_time: float
    success: bool
    timestamp: datetime
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class PerformanceTracker:
    """Main class for tracking AGI performance and improvements"""
    
    def __init__(self, storage_path: str = "performance_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Performance metrics storage
        self.metrics: List[PerformanceMetric] = []
        self.improvement_metrics: List[ImprovementMetric] = []
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.iteration_count = 0
        self.experiment_count = 0
        self.improvement_count = 0
        
        # Performance trends
        self.performance_trends = {}
        
        logger.info("Performance Tracker initialized")
    
    def record_metric(self, name: str, value: float, unit: str, source: str, tags: List[str] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            source=source,
            tags=tags or []
        )
        self.metrics.append(metric)
        logger.debug(f"Recorded metric: {name} = {value} {unit}")
    
    def record_improvement(self, improvement_id: str, improvement_type: str, impact_score: float, 
                          confidence: float, implementation_time: float, success: bool):
        """Record an improvement metric"""
        improvement_metric = ImprovementMetric(
            improvement_id=improvement_id,
            type=improvement_type,
            impact_score=impact_score,
            confidence=confidence,
            implementation_time=implementation_time,
            success=success,
            timestamp=datetime.utcnow()
        )
        self.improvement_metrics.append(improvement_metric)
        logger.info(f"Recorded improvement: {improvement_id}, impact: {impact_score}, success: {success}")
    
    def get_system_uptime(self) -> float:
        """Get system uptime in seconds"""
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    def increment_iteration_count(self):
        """Increment iteration counter"""
        self.iteration_count += 1
    
    def increment_experiment_count(self):
        """Increment experiment counter"""
        self.experiment_count += 1
    
    def increment_improvement_count(self):
        """Increment improvement counter"""
        self.improvement_count += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics"""
        now = datetime.utcnow()
        total_uptime = self.get_system_uptime()
        
        # Calculate rates
        hours_active = max(1, total_uptime / 3600)
        iterations_per_hour = self.iteration_count / hours_active
        experiments_per_hour = self.experiment_count / hours_active
        improvements_per_hour = self.improvement_count / hours_active
        
        # Calculate improvement success rate
        if self.improvement_metrics:
            successful_improvements = [imp for imp in self.improvement_metrics if imp.success]
            success_rate = len(successful_improvements) / len(self.improvement_metrics)
            avg_impact_score = statistics.mean([imp.impact_score for imp in self.improvement_metrics]) if self.improvement_metrics else 0
        else:
            success_rate = 0
            avg_impact_score = 0
        
        summary = {
            "uptime_hours": total_uptime / 3600,
            "total_iterations": self.iteration_count,
            "total_experiments": self.experiment_count,
            "total_improvements": self.improvement_count,
            "iterations_per_hour": round(iterations_per_hour, 2),
            "experiments_per_hour": round(experiments_per_hour, 2),
            "improvements_per_hour": round(improvements_per_hour, 2),
            "improvement_success_rate": round(success_rate, 2),
            "average_impact_score": round(avg_impact_score, 2),
            "timestamp": now.isoformat()
        }
        
        return summary
    
    async def calculate_advanced_metrics(self) -> Dict[str, Any]:
        """Calculate advanced performance metrics"""
        # Calculate these metrics asynchronously to avoid blocking
        
        # Performance trend analysis
        trend_metrics = await self._analyze_performance_trends()
        
        # Efficiency metrics
        efficiency_metrics = await self._calculate_efficiency_metrics()
        
        # Learning rate metrics
        learning_metrics = await self._calculate_learning_metrics()
        
        return {
            "trend_analysis": trend_metrics,
            "efficiency_metrics": efficiency_metrics,
            "learning_metrics": learning_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        # Group metrics by time windows
        hourly_windows = {}
        for metric in self.metrics:
            hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_windows:
                hourly_windows[hour_key] = []
            hourly_windows[hour_key].append(metric)
        
        # Calculate trends
        trends = {}
        for hour, hour_metrics in hourly_windows.items():
            for metric in hour_metrics:
                if metric.name not in trends:
                    trends[metric.name] = []
                trends[metric.name].append({
                    'timestamp': hour.isoformat(),
                    'value': metric.value
                })
        
        # Calculate trend direction for key metrics
        key_trends = {}
        for name, values in trends.items():
            if len(values) > 1:
                # Simple trend: compare first and last values
                first_value = values[0]['value']
                last_value = values[-1]['value']
                if first_value != 0:
                    change = (last_value - first_value) / first_value
                    key_trends[name] = {
                        'change_percentage': round(change * 100, 2),
                        'direction': 'increasing' if change > 0.05 else 'decreasing' if change < -0.05 else 'stable'
                    }
                else:
                    key_trends[name] = {
                        'change_percentage': 0,
                        'direction': 'stable'
                    }
        
        return key_trends
    
    async def _calculate_efficiency_metrics(self) -> Dict[str, Any]:
        """Calculate efficiency metrics"""
        if not self.improvement_metrics:
            return {
                "improvement_efficiency": 0,
                "average_implementation_time": 0,
                "efficiency_score": 0
            }
        
        # Calculate improvement efficiency
        successful_improvements = [imp for imp in self.improvement_metrics if imp.success]
        
        if successful_improvements:
            avg_impact = statistics.mean([imp.impact_score for imp in successful_improvements])
            avg_time = statistics.mean([imp.implementation_time for imp in successful_improvements])
            
            # Efficiency score based on impact/time ratio
            efficiency_score = avg_impact / max(avg_time, 0.001)  # Avoid division by zero
            
            return {
                "improvement_efficiency": avg_impact,
                "average_implementation_time": round(avg_time, 2),
                "efficiency_score": round(efficiency_score, 2)
            }
        else:
            return {
                "improvement_efficiency": 0,
                "average_implementation_time": 0,
                "efficiency_score": 0
            }
    
    async def _calculate_learning_metrics(self) -> Dict[str, Any]:
        """Calculate learning-related metrics"""
        if len(self.improvement_metrics) < 2:
            return {
                "learning_rate": 0,
                "adaptation_score": 0,
                "knowledge_integration_rate": 0
            }
        
        # Calculate learning rate based on improvement success over time
        # Divide improvements into early and recent periods
        sorted_improvements = sorted(self.improvement_metrics, key=lambda x: x.timestamp)
        
        mid_point = len(sorted_improvements) // 2
        early_improvements = sorted_improvements[:mid_point]
        recent_improvements = sorted_improvements[mid_point:]
        
        if early_improvements and recent_improvements:
            early_success_rate = sum(1 for imp in early_improvements if imp.success) / len(early_improvements)
            recent_success_rate = sum(1 for imp in recent_improvements if imp.success) / len(recent_improvements)
            
            learning_rate = recent_success_rate - early_success_rate
            adaptation_score = recent_success_rate  # How well we're adapting now
            
            return {
                "learning_rate": round(learning_rate, 3),
                "adaptation_score": round(adaptation_score, 3),
                "knowledge_integration_rate": round(recent_success_rate, 3)
            }
        else:
            return {
                "learning_rate": 0,
                "adaptation_score": 0,
                "knowledge_integration_rate": 0
            }
    
    async def save_performance_data(self):
        """Save performance data to storage"""
        try:
            # Save metrics
            metrics_file = self.storage_path / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump([m.to_dict() for m in self.metrics], f, indent=2)
            
            # Save improvement metrics
            improvements_file = self.storage_path / "improvements.json"
            with open(improvements_file, 'w') as f:
                json.dump([im.to_dict() for im in self.improvement_metrics], f, indent=2)
            
            # Save summary
            summary = self.get_performance_summary()
            summary_file = self.storage_path / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Performance data saved to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    async def load_performance_data(self):
        """Load performance data from storage"""
        try:
            # Load metrics
            metrics_file = self.storage_path / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    self.metrics = [
                        PerformanceMetric(
                            name=m['name'],
                            value=m['value'],
                            unit=m['unit'],
                            timestamp=datetime.fromisoformat(m['timestamp']),
                            source=m['source'],
                            tags=m.get('tags', [])
                        ) for m in metrics_data
                    ]
            
            # Load improvement metrics
            improvements_file = self.storage_path / "improvements.json"
            if improvements_file.exists():
                with open(improvements_file, 'r') as f:
                    improvements_data = json.load(f)
                    self.improvement_metrics = [
                        ImprovementMetric(
                            improvement_id=im['improvement_id'],
                            type=im['type'],
                            impact_score=im['impact_score'],
                            confidence=im['confidence'],
                            implementation_time=im['implementation_time'],
                            success=im['success'],
                            timestamp=datetime.fromisoformat(im['timestamp'])
                        ) for im in improvements_data
                    ]
            
            logger.info(f"Performance data loaded from {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            # Initialize with empty lists if loading fails
            self.metrics = []
            self.improvement_metrics = []
    
    async def get_detailed_report(self) -> Dict[str, Any]:
        """Get a detailed performance report"""
        summary = self.get_performance_summary()
        advanced_metrics = await self.calculate_advanced_metrics()
        
        # Add historical data
        historical_data = await self._get_historical_performance()
        
        report = {
            "summary": summary,
            "advanced_metrics": advanced_metrics,
            "historical_data": historical_data,
            "recommendations": await self._generate_recommendations(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return report
    
    async def _get_historical_performance(self) -> Dict[str, Any]:
        """Get historical performance data"""
        # Calculate performance over different time windows
        now = datetime.utcnow()
        
        # Performance in last hour
        last_hour = [m for m in self.metrics if now - m.timestamp <= timedelta(hours=1)]
        last_hour_improvements = [im for im in self.improvement_metrics if now - im.timestamp <= timedelta(hours=1)]
        
        # Performance in last day
        last_day = [m for m in self.metrics if now - m.timestamp <= timedelta(days=1)]
        last_day_improvements = [im for im in self.improvement_metrics if now - im.timestamp <= timedelta(days=1)]
        
        return {
            "last_hour": {
                "metric_count": len(last_hour),
                "improvement_count": len(last_hour_improvements),
                "improvement_success_rate": len([im for im in last_hour_improvements if im.success]) / len(last_hour_improvements) if last_hour_improvements else 0
            },
            "last_day": {
                "metric_count": len(last_day),
                "improvement_count": len(last_day_improvements),
                "improvement_success_rate": len([im for im in last_day_improvements if im.success]) / len(last_day_improvements) if last_day_improvements else 0
            }
        }
    
    async def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Based on calculated metrics, generate recommendations
        summary = self.get_performance_summary()
        
        if summary['improvements_per_hour'] < 0.2:
            recommendations.append({
                "category": "improvement_rate",
                "recommendation": "Increase improvement generation rate",
                "priority": "high",
                "suggestion": "Review and optimize the improvement identification algorithms"
            })
        
        if summary['improvement_success_rate'] < 0.5:
            recommendations.append({
                "category": "success_rate",
                "recommendation": "Improve success rate of implemented improvements",
                "priority": "high",
                "suggestion": "Implement more thorough validation before applying improvements"
            })
        
        if summary['average_impact_score'] < 0.3:
            recommendations.append({
                "category": "impact",
                "recommendation": "Focus on higher-impact improvements",
                "priority": "medium",
                "suggestion": "Develop better impact prediction models for proposed improvements"
            })
        
        return recommendations

# Global performance tracker instance
performance_tracker = PerformanceTracker()