"""
Snake Parallel Code Analyzer

This module implements parallel code analysis using multiple worker threads
to analyze code changes, detect improvement opportunities, and generate suggestions.
"""

import asyncio
import threading
import queue
import time
import uuid
import ast
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from core.snake_data_models import (
    AnalysisTask, TaskPriority, AnalysisRecord, SnakeAgentConfiguration
)
from core.snake_log_manager import SnakeLogManager


@dataclass
class CodeMetrics:
    """Code quality metrics"""
    lines_of_code: int
    complexity_score: float
    maintainability_index: float
    test_coverage: float
    security_issues: List[str]
    performance_issues: List[str]
    style_violations: List[str]
    potential_bugs: List[str]


@dataclass
class AnalysisResult:
    """Result of code analysis"""
    file_path: str
    analysis_type: str
    metrics: CodeMetrics
    suggestions: List[Dict[str, Any]]
    priority: TaskPriority
    confidence: float
    processing_time: float
    analyzer_id: str
    timestamp: datetime


class CodeQualityAnalyzer:
    """Individual code quality analyzer"""
    
    def __init__(self, analyzer_id: str):
        self.analyzer_id = analyzer_id
        self.analysis_count = 0
        
    def analyze_python_code(self, code_content: str, file_path: str) -> CodeMetrics:
        """Analyze Python code for quality metrics"""
        try:
            # Parse AST
            tree = ast.parse(code_content)
            
            # Calculate metrics
            lines_of_code = len(code_content.splitlines())
            complexity_score = self._calculate_complexity(tree)
            maintainability_index = self._calculate_maintainability(code_content, complexity_score)
            
            # Detect issues
            security_issues = self._detect_security_issues(code_content)
            performance_issues = self._detect_performance_issues(code_content, tree)
            style_violations = self._detect_style_violations(code_content)
            potential_bugs = self._detect_potential_bugs(code_content, tree)
            
            return CodeMetrics(
                lines_of_code=lines_of_code,
                complexity_score=complexity_score,
                maintainability_index=maintainability_index,
                test_coverage=0.0,  # Would need external test runner
                security_issues=security_issues,
                performance_issues=performance_issues,
                style_violations=style_violations,
                potential_bugs=potential_bugs
            )
            
        except SyntaxError as e:
            # Handle syntax errors gracefully
            return CodeMetrics(
                lines_of_code=len(code_content.splitlines()),
                complexity_score=0.0,
                maintainability_index=0.0,
                test_coverage=0.0,
                security_issues=[f"Syntax error: {str(e)}"],
                performance_issues=[],
                style_violations=[],
                potential_bugs=[]
            )
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity += 1
        
        return float(complexity)
    
    def _calculate_maintainability(self, code_content: str, complexity: float) -> float:
        """Calculate maintainability index (0-100)"""
        lines = len(code_content.splitlines())
        if lines == 0:
            return 0.0
        
        # Simplified maintainability calculation
        volume = lines * 4.342  # Halstead volume approximation
        maintainability = max(0, (171 - 5.2 * complexity - 0.23 * volume) * 100 / 171)
        return min(100.0, maintainability)
    
    def _detect_security_issues(self, code_content: str) -> List[str]:
        """Detect potential security issues"""
        issues = []
        
        # Check for common security anti-patterns
        security_patterns = [
            (r'eval\s*\(', "Use of eval() function is dangerous"),
            (r'exec\s*\(', "Use of exec() function is dangerous"),
            (r'subprocess\.call\s*\(.*shell\s*=\s*True', "Shell injection vulnerability"),
            (r'pickle\.loads?\s*\(', "Pickle deserialization can be unsafe"),
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret detected"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
        ]
        
        for pattern, message in security_patterns:
            if re.search(pattern, code_content, re.IGNORECASE):
                issues.append(message)
        
        return issues
    
    def _detect_performance_issues(self, code_content: str, tree: ast.AST) -> List[str]:
        """Detect potential performance issues"""
        issues = []
        
        # Check for performance anti-patterns
        perf_patterns = [
            (r'\.append\s*\(.*\)\s*in.*for.*in', "List comprehension may be faster than append in loop"),
            (r'range\s*\(\s*len\s*\(', "Consider using enumerate() instead of range(len())"),
            (r'\.keys\s*\(\s*\).*in', "Direct dictionary iteration is faster than .keys()"),
        ]
        
        for pattern, message in perf_patterns:
            if re.search(pattern, code_content):
                issues.append(message)
        
        # AST-based checks
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check for nested loops
                for child in ast.walk(node):
                    if isinstance(child, ast.For) and child != node:
                        issues.append("Nested loops detected - consider optimization")
                        break
        
        return issues
    
    def _detect_style_violations(self, code_content: str) -> List[str]:
        """Detect style violations"""
        violations = []
        lines = code_content.splitlines()
        
        for i, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 88:
                violations.append(f"Line {i}: Line too long ({len(line)} > 88)")
            
            # Check trailing whitespace
            if line.endswith(' ') or line.endswith('\t'):
                violations.append(f"Line {i}: Trailing whitespace")
            
            # Check indentation consistency
            if line.startswith(' ') and line.startswith('\t'):
                violations.append(f"Line {i}: Mixed tabs and spaces")
        
        # Check for missing docstrings
        if 'def ' in code_content and '"""' not in code_content and "'''" not in code_content:
            violations.append("Missing docstrings for functions")
        
        return violations
    
    def _detect_potential_bugs(self, code_content: str, tree: ast.AST) -> List[str]:
        """Detect potential bugs"""
        bugs = []
        
        # Check for common bug patterns
        bug_patterns = [
            (r'==\s*None', "Use 'is None' instead of '== None'"),
            (r'!=\s*None', "Use 'is not None' instead of '!= None'"),
            (r'except\s*:', "Bare except clause can hide bugs"),
            (r'time\.sleep\s*\(.*\).*async', "Blocking sleep in async function"),
        ]
        
        for pattern, message in bug_patterns:
            if re.search(pattern, code_content):
                bugs.append(message)
        
        # AST-based checks
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # Check for assignment in comparison
                if isinstance(node.left, ast.Name) and len(node.ops) == 1:
                    if isinstance(node.ops[0], ast.Eq):
                        bugs.append("Possible assignment instead of comparison")
        
        return bugs
    
    def generate_suggestions(self, metrics: CodeMetrics, file_path: str) -> List[Dict[str, Any]]:
        """Generate improvement suggestions based on metrics"""
        suggestions = []
        
        # Complexity suggestions
        if metrics.complexity_score > 10:
            suggestions.append({
                "type": "refactoring",
                "priority": "high",
                "title": "Reduce complexity",
                "description": f"Code complexity is {metrics.complexity_score:.1f}, consider breaking into smaller functions",
                "impact": "maintainability",
                "effort": "medium"
            })
        
        # Maintainability suggestions
        if metrics.maintainability_index < 50:
            suggestions.append({
                "type": "refactoring",
                "priority": "medium",
                "title": "Improve maintainability",
                "description": f"Maintainability index is {metrics.maintainability_index:.1f}, consider refactoring",
                "impact": "maintainability",
                "effort": "high"
            })
        
        # Security suggestions
        for issue in metrics.security_issues:
            suggestions.append({
                "type": "security",
                "priority": "critical",
                "title": "Security vulnerability",
                "description": issue,
                "impact": "security",
                "effort": "medium"
            })
        
        # Performance suggestions
        for issue in metrics.performance_issues:
            suggestions.append({
                "type": "performance",
                "priority": "medium",
                "title": "Performance optimization",
                "description": issue,
                "impact": "performance",
                "effort": "low"
            })
        
        # Style suggestions
        if len(metrics.style_violations) > 5:
            suggestions.append({
                "type": "style",
                "priority": "low",
                "title": "Style improvements",
                "description": f"Multiple style violations detected ({len(metrics.style_violations)} issues)",
                "impact": "readability",
                "effort": "low"
            })
        
        # Bug suggestions
        for bug in metrics.potential_bugs:
            suggestions.append({
                "type": "bug_fix",
                "priority": "high",
                "title": "Potential bug",
                "description": bug,
                "impact": "reliability",
                "effort": "low"
            })
        
        return suggestions


class ParallelCodeAnalyzer:
    """Parallel code analyzer with multiple worker threads"""
    
    def __init__(self, config: SnakeAgentConfiguration, log_manager: SnakeLogManager):
        self.config = config
        self.log_manager = log_manager
        
        # Worker management
        self.num_workers = config.analysis_threads
        self.workers: Dict[str, CodeQualityAnalyzer] = {}
        self.worker_threads: List[threading.Thread] = []
        
        # Task processing
        self.task_queue = queue.Queue(maxsize=config.max_queue_size)
        self.result_queue = queue.Queue()
        
        # Control
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Callbacks
        self.analysis_complete_callback: Optional[callable] = None
        
        # Metrics
        self.tasks_processed = 0
        self.total_processing_time = 0.0
        self.analysis_cache: Dict[str, AnalysisResult] = {}
        
        # Performance tracking
        self.worker_stats: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> bool:
        """Initialize the parallel analyzer"""
        try:
            await self.log_manager.log_system_event(
                "parallel_analyzer_init",
                {"num_workers": self.num_workers},
                worker_id="parallel_analyzer"
            )
            
            # Create worker analyzers
            for i in range(self.num_workers):
                worker_id = f"analyzer_{i}_{uuid.uuid4().hex[:8]}"
                self.workers[worker_id] = CodeQualityAnalyzer(worker_id)
                self.worker_stats[worker_id] = {
                    "tasks_processed": 0,
                    "total_time": 0.0,
                    "average_time": 0.0,
                    "last_activity": None
                }
            
            return True
            
        except Exception as e:
            await self.log_manager.log_system_event(
                "parallel_analyzer_init_failed",
                {"error": str(e)},
                level="error",
                worker_id="parallel_analyzer"
            )
            return False
    
    async def start_workers(self) -> bool:
        """Start all worker threads"""
        try:
            if self.running:
                return True
            
            self.running = True
            self.shutdown_event.clear()
            
            # Start worker threads
            for worker_id in self.workers.keys():
                thread = threading.Thread(
                    target=self._worker_loop,
                    args=(worker_id,),
                    name=f"Snake-Analyzer-{worker_id}",
                    daemon=True
                )
                thread.start()
                self.worker_threads.append(thread)
            
            # Start result processor
            result_thread = threading.Thread(
                target=self._result_processor_loop,
                name="Snake-AnalysisResultProcessor",
                daemon=True
            )
            result_thread.start()
            self.worker_threads.append(result_thread)
            
            await self.log_manager.log_system_event(
                "parallel_analyzer_started",
                {"workers_started": len(self.workers)},
                worker_id="parallel_analyzer"
            )
            
            return True
            
        except Exception as e:
            await self.log_manager.log_system_event(
                "parallel_analyzer_start_failed",
                {"error": str(e)},
                level="error",
                worker_id="parallel_analyzer"
            )
            return False
    
    def _worker_loop(self, worker_id: str):
        """Main loop for worker thread"""
        analyzer = self.workers[worker_id]
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Get task from queue
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process task
                start_time = time.time()
                result = self._process_analysis_task(task, analyzer)
                processing_time = time.time() - start_time
                
                # Update worker stats
                self.worker_stats[worker_id]["tasks_processed"] += 1
                self.worker_stats[worker_id]["total_time"] += processing_time
                self.worker_stats[worker_id]["average_time"] = (
                    self.worker_stats[worker_id]["total_time"] / 
                    self.worker_stats[worker_id]["tasks_processed"]
                )
                self.worker_stats[worker_id]["last_activity"] = datetime.now()
                
                # Queue result
                if result:
                    self.result_queue.put(result)
                
                self.task_queue.task_done()
                
            except Exception as e:
                # Log worker error
                asyncio.create_task(self.log_manager.log_system_event(
                    "analyzer_worker_error",
                    {"worker_id": worker_id, "error": str(e)},
                    level="error",
                    worker_id="parallel_analyzer"
                ))
                time.sleep(1.0)
    
    def _process_analysis_task(self, task: AnalysisTask, analyzer: CodeQualityAnalyzer) -> Optional[AnalysisResult]:
        """Process a single analysis task"""
        try:
            # Check cache first
            file_hash = self._calculate_file_hash(task.file_path)
            cache_key = f"{task.file_path}_{file_hash}_{task.analysis_type}"
            
            if cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key]
                # Update timestamp but keep analysis
                cached_result.timestamp = datetime.now()
                cached_result.analyzer_id = analyzer.analyzer_id
                return cached_result
            
            # Read file content
            try:
                with open(task.file_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()
            except Exception as e:
                return AnalysisResult(
                    file_path=task.file_path,
                    analysis_type=task.analysis_type,
                    metrics=CodeMetrics(0, 0.0, 0.0, 0.0, [f"File read error: {e}"], [], [], []),
                    suggestions=[],
                    priority=TaskPriority.LOW,
                    confidence=0.0,
                    processing_time=0.0,
                    analyzer_id=analyzer.analyzer_id,
                    timestamp=datetime.now()
                )
            
            start_time = time.time()
            
            # Analyze code
            if task.file_path.endswith('.py'):
                metrics = analyzer.analyze_python_code(code_content, task.file_path)
            else:
                # Basic analysis for non-Python files
                metrics = CodeMetrics(
                    lines_of_code=len(code_content.splitlines()),
                    complexity_score=0.0,
                    maintainability_index=50.0,
                    test_coverage=0.0,
                    security_issues=[],
                    performance_issues=[],
                    style_violations=[],
                    potential_bugs=[]
                )
            
            # Generate suggestions
            suggestions = analyzer.generate_suggestions(metrics, task.file_path)
            
            # Determine priority based on issues found
            priority = self._calculate_priority(metrics, suggestions)
            
            # Calculate confidence based on analysis completeness
            confidence = self._calculate_confidence(metrics, task.analysis_type)
            
            processing_time = time.time() - start_time
            
            result = AnalysisResult(
                file_path=task.file_path,
                analysis_type=task.analysis_type,
                metrics=metrics,
                suggestions=suggestions,
                priority=priority,
                confidence=confidence,
                processing_time=processing_time,
                analyzer_id=analyzer.analyzer_id,
                timestamp=datetime.now()
            )
            
            # Cache result
            self.analysis_cache[cache_key] = result
            
            # Limit cache size
            if len(self.analysis_cache) > 1000:
                # Remove oldest entries
                oldest_keys = sorted(self.analysis_cache.keys())[:100]
                for key in oldest_keys:
                    del self.analysis_cache[key]
            
            analyzer.analysis_count += 1
            return result
            
        except Exception as e:
            # Return error result
            return AnalysisResult(
                file_path=task.file_path,
                analysis_type=task.analysis_type,
                metrics=CodeMetrics(0, 0.0, 0.0, 0.0, [f"Analysis error: {e}"], [], [], []),
                suggestions=[],
                priority=TaskPriority.LOW,
                confidence=0.0,
                processing_time=0.0,
                analyzer_id=analyzer.analyzer_id,
                timestamp=datetime.now()
            )
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of file for caching"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.md5(content).hexdigest()
        except Exception:
            return ""
    
    def _calculate_priority(self, metrics: CodeMetrics, suggestions: List[Dict[str, Any]]) -> TaskPriority:
        """Calculate task priority based on metrics and suggestions"""
        # Critical if security issues
        if metrics.security_issues:
            return TaskPriority.CRITICAL
        
        # High if potential bugs or very low maintainability
        if metrics.potential_bugs or metrics.maintainability_index < 30:
            return TaskPriority.HIGH
        
        # Medium if performance issues or moderate complexity
        if metrics.performance_issues or metrics.complexity_score > 15:
            return TaskPriority.MEDIUM
        
        # Low for style issues only
        if metrics.style_violations:
            return TaskPriority.LOW
        
        return TaskPriority.BACKGROUND
    
    def _calculate_confidence(self, metrics: CodeMetrics, analysis_type: str) -> float:
        """Calculate confidence score for analysis"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for syntax-valid code
        if not any("Syntax error" in issue for issue in metrics.security_issues):
            confidence += 0.3
        
        # Higher confidence for comprehensive analysis
        if analysis_type == "deep_analysis":
            confidence += 0.2
        
        # Adjust based on code size
        if metrics.lines_of_code > 10:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _result_processor_loop(self):
        """Process analysis results"""
        while self.running and not self.shutdown_event.is_set():
            try:
                # Get result from queue
                try:
                    result = self.result_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process result
                self.tasks_processed += 1
                self.total_processing_time += result.processing_time
                
                # Log analysis completion
                asyncio.create_task(self.log_manager.log_analysis(
                    AnalysisRecord(
                        id=f"analysis_{uuid.uuid4().hex[:8]}",
                        file_path=result.file_path,
                        analysis_type=result.analysis_type,
                        findings={
                            "metrics": {
                                "lines_of_code": result.metrics.lines_of_code,
                                "complexity": result.metrics.complexity_score,
                                "maintainability": result.metrics.maintainability_index
                            },
                            "issues": {
                                "security": len(result.metrics.security_issues),
                                "performance": len(result.metrics.performance_issues),
                                "style": len(result.metrics.style_violations),
                                "bugs": len(result.metrics.potential_bugs)
                            }
                        },
                        suggestions=result.suggestions,
                        priority=result.priority.name.lower(),
                        confidence=result.confidence,
                        processing_time=result.processing_time,
                        timestamp=result.timestamp,
                        worker_id=result.analyzer_id
                    )
                ))
                
                # Call completion callback
                if self.analysis_complete_callback:
                    try:
                        self.analysis_complete_callback(result)
                    except Exception as e:
                        asyncio.create_task(self.log_manager.log_system_event(
                            "analysis_callback_error",
                            {"error": str(e)},
                            level="error",
                            worker_id="parallel_analyzer"
                        ))
                
            except Exception as e:
                asyncio.create_task(self.log_manager.log_system_event(
                    "result_processor_error",
                    {"error": str(e)},
                    level="error",
                    worker_id="parallel_analyzer"
                ))
                time.sleep(1.0)
    
    def queue_analysis_task(self, task: AnalysisTask) -> bool:
        """Queue a task for analysis"""
        try:
            self.task_queue.put_nowait(task)
            return True
        except queue.Full:
            asyncio.create_task(self.log_manager.log_system_event(
                "analysis_queue_full",
                {"task": task.to_dict()},
                level="warning",
                worker_id="parallel_analyzer"
            ))
            return False
    
    def set_analysis_complete_callback(self, callback: callable):
        """Set callback for analysis completion"""
        self.analysis_complete_callback = callback
    
    def get_status(self) -> Dict[str, Any]:
        """Get analyzer status"""
        return {
            "running": self.running,
            "num_workers": len(self.workers),
            "queue_size": self.task_queue.qsize(),
            "results_pending": self.result_queue.qsize(),
            "tasks_processed": self.tasks_processed,
            "average_processing_time": (
                self.total_processing_time / max(1, self.tasks_processed)
            ),
            "cache_size": len(self.analysis_cache),
            "worker_stats": self.worker_stats
        }
    
    async def shutdown(self, timeout: float = 30.0) -> bool:
        """Shutdown the parallel analyzer"""
        try:
            await self.log_manager.log_system_event(
                "parallel_analyzer_shutdown",
                {"tasks_processed": self.tasks_processed},
                worker_id="parallel_analyzer"
            )
            
            self.running = False
            self.shutdown_event.set()
            
            # Wait for workers to finish
            for thread in self.worker_threads:
                if thread.is_alive():
                    thread.join(timeout=timeout/len(self.worker_threads))
            
            return True
            
        except Exception as e:
            await self.log_manager.log_system_event(
                "parallel_analyzer_shutdown_error",
                {"error": str(e)},
                level="error",
                worker_id="parallel_analyzer"
            )
            return False