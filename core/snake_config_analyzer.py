"""
Configuration Analysis Tool for Snake Agent

This module analyzes current configuration parameters and identifies
optimization opportunities for peak performance.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from core.config import Config


@dataclass
class ConfigParameter:
    """Represents a configuration parameter with its properties"""
    name: str
    current_value: Any
    default_value: Any
    environment_source: Optional[str]
    description: str
    min_optimal: Any = None
    max_optimal: Any = None
    recommended_value: Any = None
    performance_impact: str = "unknown"  # low, medium, high


class ConfigAnalyzer:
    """Analyzes Snake Agent configuration parameters for optimization opportunities"""
    
    def __init__(self):
        self.config = Config()
        self.parameters: List[ConfigParameter] = []
        self.analysis_results = {}
    
    def analyze_all_parameters(self) -> Dict[str, Any]:
        """Analyze all Snake Agent configuration parameters"""
        print("Analyzing Snake Agent configuration parameters for optimization...")
        
        # Collect all relevant parameters
        self._collect_parameters()
        
        # Analyze each parameter
        optimization_opportunities = []
        recommendations = []
        
        for param in self.parameters:
            analysis = self._analyze_parameter(param)
            if analysis:
                if analysis.get("optimizable"):
                    optimization_opportunities.append({
                        "parameter": param.name,
                        "current": param.current_value,
                        "recommended": analysis.get("recommended_value"),
                        "impact": analysis.get("performance_impact", "unknown"),
                        "reason": analysis.get("reason", "")
                    })
                if analysis.get("recommendation"):
                    recommendations.append(analysis["recommendation"])
        
        self.analysis_results = {
            "total_parameters": len(self.parameters),
            "optimizable_parameters": len(optimization_opportunities),
            "optimization_opportunities": optimization_opportunities,
            "recommendations": recommendations,
            "parameter_details": [
                {
                    "name": p.name,
                    "current_value": p.current_value,
                    "default_value": p.default_value,
                    "source": p.environment_source,
                    "description": p.description,
                    "recommended_value": p.recommended_value,
                    "performance_impact": p.performance_impact
                } for p in self.parameters
            ]
        }
        
        return self.analysis_results
    
    def _collect_parameters(self):
        """Collect all relevant Snake Agent parameters"""
        # Threading and multiprocessing parameters
        self.parameters.extend([
            ConfigParameter(
                name="SNAKE_MAX_THREADS",
                current_value=self.config.SNAKE_MAX_THREADS,
                default_value=12,
                environment_source=os.environ.get("SNAKE_MAX_THREADS"),
                description="Maximum number of threads for analysis tasks",
                min_optimal=4,
                max_optimal=32,
                recommended_value=16 if self._has_sufficient_cpu() else 12,
                performance_impact="high"
            ),
            ConfigParameter(
                name="SNAKE_MAX_PROCESSES",
                current_value=self.config.SNAKE_MAX_PROCESSES,
                default_value=6,
                environment_source=os.environ.get("SNAKE_MAX_PROCESSES"),
                description="Maximum number of processes for experimentation",
                min_optimal=2,
                max_optimal=16,
                recommended_value=8 if self._has_sufficient_cpu() else 6,
                performance_impact="high"
            ),
            ConfigParameter(
                name="SNAKE_ANALYSIS_THREADS",
                current_value=self.config.SNAKE_ANALYSIS_THREADS,
                default_value=4,
                environment_source=os.environ.get("SNAKE_ANALYSIS_THREADS"),
                description="Number of threads dedicated to code analysis",
                min_optimal=2,
                max_optimal=8,
                recommended_value=6 if self._has_sufficient_cpu() else 4,
                performance_impact="high"
            ),
            ConfigParameter(
                name="SNAKE_MONITOR_INTERVAL",
                current_value=self.config.SNAKE_MONITOR_INTERVAL,
                default_value=1.0,
                environment_source=os.environ.get("SNAKE_MONITOR_INTERVAL"),
                description="Interval in seconds for file monitoring checks",
                min_optimal=0.5,
                max_optimal=5.0,
                recommended_value=0.8,  # Slightly faster than default for better responsiveness
                performance_impact="medium"
            ),
            ConfigParameter(
                name="SNAKE_PERF_MONITORING",
                current_value=self.config.SNAKE_PERF_MONITORING,
                default_value=True,
                environment_source=os.environ.get("SNAKE_PERF_MONITORING"),
                description="Enable performance monitoring and metrics logging",
                recommended_value=True,  # Keep enabled for optimization
                performance_impact="low"
            ),
            ConfigParameter(
                name="SNAKE_AUTO_RECOVERY",
                current_value=self.config.SNAKE_AUTO_RECOVERY,
                default_value=True,
                environment_source=os.environ.get("SNAKE_AUTO_RECOVERY"),
                description="Enable automatic recovery from failures",
                recommended_value=True,  # Keep enabled for stability
                performance_impact="medium"
            ),
            ConfigParameter(
                name="SNAKE_TASK_TIMEOUT",
                current_value=self.config.SNAKE_TASK_TIMEOUT,
                default_value=600.0,  # 10 minutes
                environment_source=os.environ.get("SNAKE_TASK_TIMEOUT"),
                description="Maximum time in seconds for task execution",
                min_optimal=300.0,  # 5 minutes
                max_optimal=1800.0,  # 30 minutes
                recommended_value=900.0,  # 15 minutes
                performance_impact="medium"
            ),
            ConfigParameter(
                name="SNAKE_HEARTBEAT_INTERVAL",
                current_value=getattr(self.config, 'SNAKE_HEARTBEAT_INTERVAL', 5.0),
                default_value=5.0,
                environment_source=os.environ.get("SNAKE_HEARTBEAT_INTERVAL"),
                description="Interval for heartbeat checks in seconds",
                min_optimal=1.0,
                max_optimal=10.0,
                recommended_value=3.0,  # More frequent for better responsiveness
                performance_impact="medium"
            ),
            ConfigParameter(
                name="SNAKE_MAX_QUEUE_SIZE",
                current_value=self.config.SNAKE_MAX_QUEUE_SIZE,
                default_value=2000,
                environment_source=os.environ.get("SNAKE_MAX_QUEUE_SIZE"),
                description="Maximum size of task queues",
                min_optimal=1000,
                max_optimal=10000,
                recommended_value=3000,  # Higher for better buffering
                performance_impact="medium"
            ),
            ConfigParameter(
                name="SNAKE_CLEANUP_INTERVAL",
                current_value=getattr(self.config, 'SNAKE_CLEANUP_INTERVAL', 1800.0),
                default_value=1800.0,  # 30 minutes
                environment_source=os.environ.get("SNAKE_CLEANUP_INTERVAL"),
                description="Interval for resource cleanup in seconds",
                min_optimal=900.0,  # 15 minutes
                max_optimal=3600.0,  # 60 minutes
                recommended_value=1200.0,  # 20 minutes
                performance_impact="low"
            ),
            ConfigParameter(
                name="SNAKE_LOG_RETENTION_DAYS",
                current_value=self.config.SNAKE_LOG_RETENTION_DAYS,
                default_value=60,
                environment_source=os.environ.get("SNAKE_LOG_RETENTION_DAYS"),
                description="Number of days to retain logs",
                min_optimal=30,
                max_optimal=90,
                recommended_value=45,
                performance_impact="low"
            ),
            ConfigParameter(
                name="SNAKE_USE_PEEK_PRIORITIZER",
                current_value=getattr(self.config, 'SNAKE_USE_PEEK_PRIORITIZER', True),
                default_value=True,
                environment_source=os.environ.get("SNAKE_USE_PEEK_PRIORITIZER"),
                description="Enable lightweight peek-based file prioritization",
                recommended_value=True,  # Keep enabled for efficiency
                performance_impact="high"
            ),
        ])
    
    def _analyze_parameter(self, param: ConfigParameter) -> Optional[Dict[str, Any]]:
        """Analyze a single parameter for optimization opportunities"""
        analysis = {
            "parameter": param.name,
            "current_value": param.current_value,
            "default_value": param.default_value,
            "recommended_value": param.recommended_value,
            "optimizable": False,
            "performance_impact": param.performance_impact,
        }
        
        # Check if parameter is optimizable based on recommendations
        if param.recommended_value is not None and param.current_value != param.recommended_value:
            analysis["optimizable"] = True
            analysis["recommended_value"] = param.recommended_value
            analysis["reason"] = self._get_recommendation_reason(param)
        
        # Special checks for specific parameters
        if param.name == "SNAKE_MAX_THREADS":
            if param.current_value > self._get_cpu_count():
                analysis["optimizable"] = True
                analysis["recommended_value"] = self._get_cpu_count()
                analysis["reason"] = f"Current value ({param.current_value}) exceeds CPU cores ({self._get_cpu_count()})"
        
        elif param.name == "SNAKE_MAX_PROCESSES":
            if param.current_value > self._get_cpu_count():
                analysis["optimizable"] = True
                analysis["recommended_value"] = self._get_cpu_count()
                analysis["reason"] = f"Current value ({param.current_value}) exceeds CPU cores ({self._get_cpu_count()})"
        
        elif param.name == "SNAKE_ANALYSIS_THREADS":
            if param.current_value > param.current_value:  # If analysis threads exceed max threads
                # This shouldn't happen based on our setup, but just in case
                pass
        
        return analysis if analysis.get("optimizable") or param.recommended_value else analysis
    
    def _get_recommendation_reason(self, param: ConfigParameter) -> str:
        """Get the reason for a recommendation"""
        if param.name == "SNAKE_MAX_THREADS":
            return f"Recommended based on CPU capacity (current: {param.current_value}, recommended: {param.recommended_value})"
        elif param.name == "SNAKE_MAX_PROCESSES":
            return f"Recommended based on CPU capacity (current: {param.current_value}, recommended: {param.recommended_value})"
        elif param.name == "SNAKE_MONITOR_INTERVAL":
            return f"Faster monitoring for improved responsiveness (current: {param.current_value}s, recommended: {param.recommended_value}s)"
        elif param.name == "SNAKE_TASK_TIMEOUT":
            return f"Extended timeout for complex tasks (current: {param.current_value}s, recommended: {param.recommended_value}s)"
        elif param.name == "SNAKE_HEARTBEAT_INTERVAL":
            return f"More frequent heartbeats for better monitoring (current: {param.current_value}s, recommended: {param.recommended_value}s)"
        elif param.name == "SNAKE_MAX_QUEUE_SIZE":
            return f"Larger queue size for better buffering (current: {param.current_value}, recommended: {param.recommended_value})"
        elif param.name == "SNAKE_CLEANUP_INTERVAL":
            return f"More frequent cleanup for better resource management (current: {param.current_value}s, recommended: {param.recommended_value}s)"
        else:
            return f"Recommended optimization for performance (current: {param.current_value}, recommended: {param.recommended_value})"
    
    def _has_sufficient_cpu(self) -> bool:
        """Check if system has sufficient CPU for higher thread/process counts"""
        try:
            import psutil
            cpu_count = psutil.cpu_count(logical=True)
            return cpu_count >= 8  # Consider systems with 8+ logical CPUs as high-capacity
        except ImportError:
            # Fallback if psutil is not available
            cpu_count = os.cpu_count() or 4
            return cpu_count >= 8
    
    def _get_cpu_count(self) -> int:
        """Get the number of CPU cores"""
        try:
            import psutil
            return psutil.cpu_count(logical=True) or 4
        except ImportError:
            return os.cpu_count() or 4
    
    def generate_config_recommendations(self) -> List[str]:
        """Generate shell commands for optimal configuration"""
        recommendations = []
        
        # Analyze parameters first
        self.analyze_all_parameters()
        
        # Generate export commands for parameters that need optimization
        for opportunity in self.analysis_results["optimization_opportunities"]:
            param_name = opportunity["parameter"]
            recommended_value = opportunity["recommended"]
            
            # Create export command
            if isinstance(recommended_value, bool):
                value_str = str(recommended_value).lower()
            elif isinstance(recommended_value, (int, float)):
                value_str = str(recommended_value)
            else:
                value_str = str(recommended_value)
            
            recommendations.append(f"export {param_name}={value_str}")
        
        return recommendations
    
    def print_analysis_report(self):
        """Print a detailed analysis report"""
        results = self.analyze_all_parameters()
        
        print("\n" + "="*80)
        print("SNAKE AGENT CONFIGURATION ANALYSIS REPORT")
        print("="*80)
        print(f"Total Parameters Analyzed: {results['total_parameters']}")
        print(f"Parameters with Optimization Opportunities: {results['optimizable_parameters']}")
        print()
        
        if results['optimization_opportunities']:
            print("OPTIMIZATION OPPORTUNITIES:")
            print("-" * 50)
            for opportunity in results['optimization_opportunities']:
                print(f"Parameter: {opportunity['parameter']}")
                print(f"  Current: {opportunity['current']}")
                print(f"  Recommended: {opportunity['recommended']}")
                print(f"  Impact: {opportunity['impact']}")
                print(f"  Reason: {opportunity['reason']}")
                print()
        
        print("CONFIGURATION RECOMMENDATIONS:")
        print("-" * 50)
        recommendations = self.generate_config_recommendations()
        for rec in recommendations:
            print(f"  {rec}")
        
        print()
        print("RECOMMENDED ENVIRONMENT SETUP:")
        print("-" * 50)
        print("# Optimal Snake Agent Configuration")
        print("# Add these to your environment or shell profile")
        for rec in recommendations:
            print(rec)
        print()
    
    def export_analysis_to_file(self, filename: str = "config_analysis.json") -> str:
        """Export the analysis to a JSON file"""
        results = self.analyze_all_parameters()
        
        filepath = Path(filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Configuration analysis exported to {filepath}")
        return str(filepath)


def run_config_analysis():
    """Run the configuration analysis"""
    analyzer = ConfigAnalyzer()
    analyzer.print_analysis_report()
    return analyzer


if __name__ == "__main__":
    analyzer = run_config_analysis()
    # Also export to file
    analyzer.export_analysis_to_file()