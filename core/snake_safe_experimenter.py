"""
Snake Safe Experimenter

This module provides a safe experimentation environment for testing code modifications
before proposing them to the main RAVANA system.
"""

import asyncio
import logging
import os
import tempfile
import shutil
import sys
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import ast

from core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result of a code experiment"""
    success: bool
    safe: bool
    test_passed: bool
    performance_impact: Optional[float] = None
    error_message: Optional[str] = None
    safety_score: float = 0.0
    impact_score: float = 0.0
    execution_time: float = 0.0
    memory_usage: Optional[float] = None
    output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "safe": self.safe,
            "test_passed": self.test_passed,
            "performance_impact": self.performance_impact,
            "error_message": self.error_message,
            "safety_score": self.safety_score,
            "impact_score": self.impact_score,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "output": self.output
        }


class SandboxEnvironment:
    """Isolated environment for running code experiments"""

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.sandbox_dir: Optional[Path] = None
        self.original_cwd = os.getcwd()
        self.is_active = False

    async def __aenter__(self):
        """Enter sandbox environment"""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit and cleanup sandbox environment"""
        await self.cleanup()

    async def setup(self):
        """Setup isolated sandbox environment"""
        try:
            # Create temporary directory for sandbox
            self.sandbox_dir = Path(tempfile.mkdtemp(
                prefix=f"snake_sandbox_{self.experiment_id}_"))
            logger.info(f"Created sandbox directory: {self.sandbox_dir}")

            # Copy RAVANA core files (read-only)
            await self._copy_ravana_files()

            # Setup virtual environment (optional)
            # await self._setup_virtual_env()

            self.is_active = True

        except Exception as e:
            logger.error(f"Failed to setup sandbox: {e}")
            await self.cleanup()
            raise

    async def cleanup(self):
        """Cleanup sandbox environment"""
        if self.sandbox_dir and self.sandbox_dir.exists():
            try:
                shutil.rmtree(self.sandbox_dir)
                logger.info(
                    f"Cleaned up sandbox directory: {self.sandbox_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up sandbox: {e}")

        self.is_active = False

    async def _copy_ravana_files(self):
        """Copy necessary RAVANA files to sandbox"""
        # Copy core modules (read-only)
        source_dirs = ['core', 'modules', 'services', 'database']

        for dir_name in source_dirs:
            source_path = Path(self.original_cwd) / dir_name
            if source_path.exists():
                dest_path = self.sandbox_dir / dir_name
                shutil.copytree(source_path, dest_path)

        # Copy configuration files
        config_files = ['pyproject.toml', 'requirements.txt']
        for config_file in config_files:
            source_file = Path(self.original_cwd) / config_file
            if source_file.exists():
                shutil.copy2(source_file, self.sandbox_dir / config_file)

    async def execute_code(self, code: str, file_path: str, timeout: int = None) -> Tuple[bool, str, float]:
        """Execute code in sandbox environment"""
        if not self.is_active:
            raise RuntimeError("Sandbox not active")

        timeout = timeout or Config().SNAKE_SANDBOX_TIMEOUT

        try:
            # Write code to file in sandbox
            target_file = self.sandbox_dir / file_path
            target_file.parent.mkdir(parents=True, exist_ok=True)

            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(code)

            # Execute code with timeout
            start_time = time.time()

            # Run the target file directly to avoid quoting/backslash issues on Windows
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(target_file),
                cwd=self.sandbox_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                execution_time = time.time() - start_time

                if process.returncode == 0:
                    return True, stdout.decode('utf-8'), execution_time
                else:
                    error_output = stderr.decode('utf-8')
                    return False, error_output, execution_time

            except asyncio.TimeoutError:
                try:
                    process.kill()
                except Exception:
                    pass
                await process.wait()
                return False, f"Execution timed out after {timeout} seconds", timeout

        except Exception as e:
            return False, f"Execution error: {str(e)}", 0.0

    async def run_tests(self, test_file: str = None) -> Tuple[bool, str]:
        """Run tests in sandbox environment"""
        try:
            if test_file:
                # Run specific test file
                cmd = [sys.executable, '-m', 'pytest', test_file, '-v']
            else:
                # Run all tests
                test_dir = self.sandbox_dir / 'tests'
                if test_dir.exists():
                    cmd = [sys.executable, '-m', 'pytest', str(test_dir), '-v']
                else:
                    return True, "No tests found"

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.sandbox_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return True, stdout.decode('utf-8')
            else:
                return False, stderr.decode('utf-8')

        except Exception as e:
            return False, f"Test execution error: {str(e)}"


class CodeSafetyAnalyzer:
    """Analyzes code for safety issues before execution"""

    def __init__(self):
        self.dangerous_imports = {
            'os', 'subprocess', 'sys', 'shutil', 'glob', 'socket', 'urllib',
            'requests', 'http', 'ftplib', 'telnetlib', 'smtplib'
        }

        self.dangerous_functions = {
            'exec', 'eval', 'compile', '__import__', 'open', 'file',
            'input', 'raw_input', 'exit', 'quit'
        }

        self.dangerous_attributes = {
            '__globals__', '__locals__', '__code__', '__closure__'
        }

    def analyze_safety(self, code: str) -> Tuple[bool, List[str], float]:
        """Analyze code safety and return (is_safe, warnings, safety_score)"""
        warnings = []
        safety_score = 1.0

        try:
            # Parse AST
            tree = ast.parse(code)

            # Check for dangerous patterns
            for node in ast.walk(tree):
                # Check imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module_names = []
                    if isinstance(node, ast.Import):
                        module_names = [alias.name for alias in node.names]
                    else:
                        module_names = [node.module] if node.module else []

                    for module in module_names:
                        if module and any(dangerous in module for dangerous in self.dangerous_imports):
                            warnings.append(
                                f"Dangerous import detected: {module}")
                            safety_score -= 0.2

                # Check function calls
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.dangerous_functions:
                            warnings.append(
                                f"Dangerous function call: {node.func.id}")
                            safety_score -= 0.3

                # Check attribute access
                elif isinstance(node, ast.Attribute):
                    if node.attr in self.dangerous_attributes:
                        warnings.append(
                            f"Dangerous attribute access: {node.attr}")
                        safety_score -= 0.2

                # Check for network operations
                elif isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Attribute) and
                            isinstance(node.func.value, ast.Name)):
                        func_name = f"{node.func.value.id}.{node.func.attr}"
                        if any(net_op in func_name for net_op in ['socket.', 'urllib.', 'requests.']):
                            warnings.append(
                                f"Network operation detected: {func_name}")
                            safety_score -= 0.1

        except SyntaxError as e:
            warnings.append(f"Syntax error: {e}")
            safety_score = 0.0

        # Check file size
        if len(code) > Config().SNAKE_MAX_FILE_SIZE:
            warnings.append(
                f"Code size ({len(code)}) exceeds maximum ({Config().SNAKE_MAX_FILE_SIZE})")
            safety_score -= 0.1

        # Check for blacklisted paths in code
        for blacklisted_path in Config().SNAKE_BLACKLIST_PATHS:
            if blacklisted_path and blacklisted_path in code:
                warnings.append(
                    f"Blacklisted path detected: {blacklisted_path}")
                safety_score -= 0.3

        safety_score = max(0.0, safety_score)
        is_safe = safety_score > 0.7 and len(
            [w for w in warnings if 'Dangerous' in w]) == 0

        return is_safe, warnings, safety_score


class PerformanceBenchmark:
    """Benchmarks code performance"""

    def __init__(self):
        self.baseline_metrics: Dict[str, float] = {}

    async def benchmark_code(self, code: str, sandbox: SandboxEnvironment,
                             iterations: int = 3) -> Dict[str, float]:
        """Benchmark code performance"""
        metrics = {
            "avg_execution_time": 0.0,
            "min_execution_time": float('inf'),
            "max_execution_time": 0.0,
            "memory_usage": 0.0
        }

        execution_times = []

        for i in range(iterations):
            success, output, execution_time = await sandbox.execute_code(
                code, f"benchmark_{i}.py"
            )

            if success:
                execution_times.append(execution_time)
                metrics["min_execution_time"] = min(
                    metrics["min_execution_time"], execution_time)
                metrics["max_execution_time"] = max(
                    metrics["max_execution_time"], execution_time)
            else:
                logger.warning(f"Benchmark iteration {i} failed: {output}")

        if execution_times:
            metrics["avg_execution_time"] = sum(
                execution_times) / len(execution_times)
        else:
            metrics["min_execution_time"] = 0.0

        return metrics

    def calculate_performance_impact(self, new_metrics: Dict[str, float],
                                     baseline_metrics: Dict[str, float]) -> float:
        """Calculate performance impact compared to baseline"""
        if not baseline_metrics:
            return 0.0

        # Compare execution times
        baseline_time = baseline_metrics.get("avg_execution_time", 1.0)
        new_time = new_metrics.get("avg_execution_time", 1.0)

        if baseline_time == 0:
            return 0.0

        # Positive impact = improvement, negative = degradation
        impact = (baseline_time - new_time) / baseline_time
        return impact


class SnakeSafeExperimenter:
    """Main safe experimenter for testing code modifications"""

    def __init__(self, coding_llm, reasoning_llm):
        self.coding_llm = coding_llm
        self.reasoning_llm = reasoning_llm
        self.safety_analyzer = CodeSafetyAnalyzer()
        self.performance_benchmark = PerformanceBenchmark()

    async def run_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Run a complete code experiment"""
        experiment_id = experiment["id"]
        file_path = experiment["file_path"]
        analysis = experiment["analysis"]

        logger.info(f"Running experiment {experiment_id} on {file_path}")

        try:
            # Generate improved code
            improved_code = await self._generate_improved_code(experiment)
            if not improved_code:
                return {"success": False, "error": "Failed to generate improved code"}

            # Safety analysis
            is_safe, safety_warnings, safety_score = self.safety_analyzer.analyze_safety(
                improved_code)
            if not is_safe:
                logger.warning(
                    f"Experiment {experiment_id} failed safety check: {safety_warnings}")
                return {
                    "success": False,
                    "safe": False,
                    "safety_score": safety_score,
                    "safety_warnings": safety_warnings,
                    "error": "Code failed safety analysis"
                }

            # Run experiment in sandbox
            result = await self._run_sandbox_experiment(
                experiment_id, file_path, improved_code
            )

            # Add safety information
            result.safety_score = safety_score
            result.safe = is_safe

            # Evaluate with reasoning LLM
            evaluation = await self._evaluate_experiment_result(experiment, result)

            final_result = result.to_dict()
            final_result.update(evaluation)

            logger.info(
                f"Experiment {experiment_id} completed: success={result.success}, safe={result.safe}")

            return final_result

        except Exception as e:
            logger.error(f"Error running experiment {experiment_id}: {e}")
            return {
                "success": False,
                "safe": False,
                "error": str(e),
                "safety_score": 0.0,
                "impact_score": 0.0
            }

    async def _generate_improved_code(self, experiment: Dict[str, Any]) -> str:
        """Generate improved code based on analysis"""
        try:
            file_path = experiment["file_path"]
            analysis = experiment["analysis"]

            # Read original code
            with open(file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()

            # Get improvement suggestions from coding LLM
            improved_code = await self.coding_llm.generate_improvement(
                json.dumps(analysis, indent=2), original_code
            )

            return improved_code

        except Exception as e:
            logger.error(f"Error generating improved code: {e}")
            return ""

    async def _run_sandbox_experiment(self, experiment_id: str, file_path: str,
                                      improved_code: str) -> ExperimentResult:
        """Run experiment in isolated sandbox"""
        async with SandboxEnvironment(experiment_id) as sandbox:
            result = ExperimentResult(
                success=False, safe=True, test_passed=False)

            try:
                # Execute improved code
                success, output, execution_time = await sandbox.execute_code(
                    improved_code, file_path
                )

                result.success = success
                result.output = output
                result.execution_time = execution_time

                if not success:
                    result.error_message = output
                    return result

                # Run tests
                test_passed, test_output = await sandbox.run_tests()
                result.test_passed = test_passed

                if not test_passed:
                    result.output += f"\nTest output: {test_output}"

                # Benchmark performance
                performance_metrics = await self.performance_benchmark.benchmark_code(
                    improved_code, sandbox
                )

                result.performance_impact = performance_metrics.get(
                    "avg_execution_time", 0.0)
                result.memory_usage = performance_metrics.get(
                    "memory_usage", 0.0)

                # Calculate impact score
                result.impact_score = self._calculate_impact_score(
                    result, performance_metrics)

                logger.info(
                    f"Sandbox experiment completed: success={result.success}, tests={result.test_passed}")

            except Exception as e:
                result.success = False
                result.safe = False
                result.error_message = str(e)
                logger.error(f"Sandbox experiment error: {e}")

            return result

    def _calculate_impact_score(self, result: ExperimentResult,
                                performance_metrics: Dict[str, float]) -> float:
        """Calculate overall impact score of the experiment"""
        score = 0.0

        # Success contributes to impact
        if result.success:
            score += 0.3

        # Test passing contributes to impact
        if result.test_passed:
            score += 0.3

        # Performance improvement contributes
        if result.execution_time < 1.0:  # Fast execution
            score += 0.2

        # Safety contributes
        score += result.safety_score * 0.2

        return min(1.0, score)

    async def _evaluate_experiment_result(self, experiment: Dict[str, Any],
                                          result: ExperimentResult) -> Dict[str, Any]:
        """Evaluate experiment result using reasoning LLM"""
        try:
            evaluation_data = {
                "experiment": experiment,
                "result": result.to_dict()
            }

            evaluation = await self.reasoning_llm.analyze_system_impact(evaluation_data)

            return {
                "llm_evaluation": evaluation,
                "recommendation": evaluation.get("recommendation", "review_required"),
                "risk_assessment": evaluation.get("risk_level", "medium")
            }

        except Exception as e:
            logger.error(f"Error evaluating experiment result: {e}")
            return {
                "llm_evaluation": {"error": str(e)},
                "recommendation": "manual_review",
                "risk_assessment": "high"
            }
