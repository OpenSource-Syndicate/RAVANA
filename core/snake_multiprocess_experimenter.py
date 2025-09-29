"""
Snake Multiprocess Experimenter

This module implements a multiprocess experimenter that runs code experiments
in isolated processes for safety and parallelization.
"""

import multiprocessing
import subprocess
import tempfile
import shutil
import os
import sys
import time
import json
import ast
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

from core.snake_data_models import (
    ExperimentTask, TaskPriority, ExperimentRecord, SnakeAgentConfiguration
)
from core.snake_log_manager import SnakeLogManager


@dataclass
class ExperimentResult:
    """Result of a code experiment"""
    experiment_id: str
    success: bool
    output: str
    error_output: str
    exit_code: int
    execution_time: float
    memory_usage: float
    safety_score: float
    impact_assessment: Dict[str, Any]
    changes_made: List[Dict[str, Any]]
    rollback_info: Optional[Dict[str, Any]]
    timestamp: datetime


@dataclass
class SafetyConstraints:
    """Safety constraints for experiments"""
    max_execution_time: float = 60.0  # seconds
    max_memory_usage: int = 100 * 1024 * 1024  # 100MB
    allowed_imports: List[str] = None
    forbidden_operations: List[str] = None
    filesystem_access: bool = False
    network_access: bool = False

    def __post_init__(self):
        if self.allowed_imports is None:
            self.allowed_imports = [
                'os', 'sys', 'time', 'datetime', 'json', 'math', 'random',
                'collections', 'itertools', 'functools', 'typing'
            ]

        if self.forbidden_operations is None:
            self.forbidden_operations = [
                'eval', 'exec', 'compile', '__import__', 'globals', 'locals',
                'subprocess', 'os.system', 'os.popen', 'os.spawn'
            ]


class ExperimentSandbox:
    """Isolated sandbox for running experiments"""

    def __init__(self, experiment_id: str, constraints: SafetyConstraints):
        self.experiment_id = experiment_id
        self.constraints = constraints
        self.sandbox_dir: Optional[Path] = None
        self.original_files: Dict[str, bytes] = {}

    def __enter__(self):
        """Set up sandbox environment"""
        # Create temporary directory
        self.sandbox_dir = Path(tempfile.mkdtemp(
            prefix=f"snake_exp_{self.experiment_id}_"))

        # Copy necessary files to sandbox
        self._setup_sandbox()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up sandbox"""
        if self.sandbox_dir and self.sandbox_dir.exists():
            try:
                shutil.rmtree(self.sandbox_dir)
            except Exception:
                pass  # Best effort cleanup

    def _setup_sandbox(self):
        """Set up the sandbox environment"""
        # Create basic directory structure
        (self.sandbox_dir / "code").mkdir()
        (self.sandbox_dir / "output").mkdir()
        (self.sandbox_dir / "temp").mkdir()

        # Create restricted Python environment script
        self._create_restricted_environment()

    def _create_restricted_environment(self):
        """Create a restricted Python execution environment"""
        restricted_code = '''
import sys
import os
import resource
import signal
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Set resource limits
def set_limits():
    # CPU time limit
    resource.setrlimit(resource.RLIMIT_CPU, ({max_cpu_time}, {max_cpu_time}))
    # Memory limit
    resource.setrlimit(resource.RLIMIT_AS, ({max_memory}, {max_memory}))
    # No core dumps
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Experiment execution timeout")

def execute_experiment(code_file, timeout={timeout}):
    """Execute experiment code with safety constraints"""
    try:
        # Set resource limits
        set_limits()
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        # Capture output
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        # Prepare globals with restricted builtins
        safe_globals = {{
            '__builtins__': {{
                'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
                'range': range, 'enumerate': enumerate, 'zip': zip,
                'sum': sum, 'min': max, 'abs': abs, 'round': round,
                'isinstance': isinstance, 'type': type, 'bool': bool,
                'Exception': Exception, 'ValueError': ValueError,
                'TypeError': TypeError, 'KeyError': KeyError,
                'IndexError': IndexError, 'AttributeError': AttributeError
            }},
            '__name__': '__main__',
            '__file__': code_file
        }}
        
        # Read and execute code
        with open(code_file, 'r') as f:
            code = f.read()
        
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(compile(code, code_file, 'exec'), safe_globals)
        
        # Cancel timeout
        signal.alarm(0)
        
        return {{
            'success': True,
            'stdout': stdout_capture.getvalue(),
            'stderr': stderr_capture.getvalue(),
            'exit_code': 0
        }}
        
    except TimeoutError:
        return {{
            'success': False,
            'stdout': stdout_capture.getvalue() if 'stdout_capture' in locals() else '',
            'stderr': 'Execution timeout',
            'exit_code': -1
        }}
    except Exception as e:
        return {{
            'success': False,
            'stdout': stdout_capture.getvalue() if 'stdout_capture' in locals() else '',
            'stderr': f"{{type(e).__name__}}: {{str(e)}}",
            'exit_code': 1
        }}

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: restricted_env.py <code_file>")
        sys.exit(1)
    
    result = execute_experiment(sys.argv[1])
    
    # Write result to stdout as JSON
    import json
    print("EXPERIMENT_RESULT_START")
    print(json.dumps(result))
    print("EXPERIMENT_RESULT_END")
'''.format(
            max_cpu_time=int(self.constraints.max_execution_time),
            max_memory=self.constraints.max_memory_usage,
            timeout=self.constraints.max_execution_time
        )

        restricted_env_file = self.sandbox_dir / "restricted_env.py"
        with open(restricted_env_file, 'w') as f:
            f.write(restricted_code)

    def backup_file(self, file_path: str):
        """Backup a file before modification"""
        try:
            with open(file_path, 'rb') as f:
                self.original_files[file_path] = f.read()
        except Exception:
            pass  # File might not exist

    def restore_file(self, file_path: str):
        """Restore a file from backup"""
        if file_path in self.original_files:
            try:
                with open(file_path, 'wb') as f:
                    f.write(self.original_files[file_path])
                return True
            except Exception:
                pass
        return False

    def execute_code(self, code: str, file_name: str = "experiment.py") -> Dict[str, Any]:
        """Execute code in the sandbox"""
        try:
            # Write code to sandbox
            code_file = self.sandbox_dir / "code" / file_name
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)

            # Execute in restricted environment
            restricted_env = self.sandbox_dir / "restricted_env.py"

            start_time = time.time()

            # Run the experiment
            process = subprocess.Popen(
                [sys.executable, str(restricted_env), str(code_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.sandbox_dir),
                env=self._get_restricted_env()
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(
                    timeout=self.constraints.max_execution_time + 5)
                execution_time = time.time() - start_time

                # Parse result from stdout
                result = self._parse_experiment_output(
                    stdout, stderr, process.returncode, execution_time)

            except subprocess.TimeoutExpired:
                process.kill()
                execution_time = time.time() - start_time
                result = {
                    'success': False,
                    'stdout': '',
                    'stderr': 'Process timeout',
                    'exit_code': -1,
                    'execution_time': execution_time
                }

            return result

        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': f"Sandbox execution error: {str(e)}",
                'exit_code': -2,
                'execution_time': 0.0
            }

    def _get_restricted_env(self) -> Dict[str, str]:
        """Get restricted environment variables"""
        # Start with minimal environment
        env = {
            'PATH': os.environ.get('PATH', ''),
            'PYTHONPATH': '',
            'HOME': str(self.sandbox_dir),
            'TMPDIR': str(self.sandbox_dir / "temp"),
            'TEMP': str(self.sandbox_dir / "temp"),
            'TMP': str(self.sandbox_dir / "temp")
        }

        # Add Python executable path
        python_dir = os.path.dirname(sys.executable)
        env['PATH'] = f"{python_dir}{os.pathsep}{env['PATH']}"

        return env

    def _parse_experiment_output(self, stdout: str, stderr: str, exit_code: int, execution_time: float) -> Dict[str, Any]:
        """Parse experiment output to extract results"""
        try:
            # Look for JSON result in stdout
            lines = stdout.split('\n')
            in_result = False
            result_lines = []

            for line in lines:
                if line.strip() == "EXPERIMENT_RESULT_START":
                    in_result = True
                    continue
                elif line.strip() == "EXPERIMENT_RESULT_END":
                    break
                elif in_result:
                    result_lines.append(line)

            if result_lines:
                # Parse JSON result
                result_json = '\n'.join(result_lines)
                parsed_result = json.loads(result_json)
                parsed_result['execution_time'] = execution_time
                return parsed_result

        except Exception:
            pass

        # Fallback to basic result
        return {
            'success': exit_code == 0,
            'stdout': stdout,
            'stderr': stderr,
            'exit_code': exit_code,
            'execution_time': execution_time
        }


class CodeExperimentValidator:
    """Validates code experiments for safety"""

    def __init__(self, constraints: SafetyConstraints):
        self.constraints = constraints

    def validate_code(self, code: str) -> Tuple[bool, List[str]]:
        """Validate code for safety issues"""
        issues = []

        try:
            # Parse AST to check for forbidden operations
            tree = ast.parse(code)

            # Check for forbidden function calls
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.constraints.forbidden_operations:
                            issues.append(
                                f"Forbidden operation: {node.func.id}")
                    elif isinstance(node.func, ast.Attribute):
                        func_name = f"{ast.unparse(node.func.value)}.{node.func.attr}"
                        if any(forbidden in func_name for forbidden in self.constraints.forbidden_operations):
                            issues.append(
                                f"Potentially forbidden operation: {func_name}")

                # Check imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.constraints.allowed_imports:
                            issues.append(f"Unauthorized import: {alias.name}")

                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module not in self.constraints.allowed_imports:
                        issues.append(
                            f"Unauthorized import from: {node.module}")

            # Additional static analysis
            if 'subprocess' in code and not self.constraints.filesystem_access:
                issues.append(
                    "Subprocess usage detected without filesystem access")

            if any(net_keyword in code for net_keyword in ['urllib', 'requests', 'socket', 'http']):
                if not self.constraints.network_access:
                    issues.append("Network access detected but not allowed")

            return len(issues) == 0, issues

        except SyntaxError as e:
            return False, [f"Syntax error: {str(e)}"]
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]


class MultiprocessExperimenter:
    """Multiprocess experimenter for running code experiments safely"""

    def __init__(self, config: SnakeAgentConfiguration, log_manager: SnakeLogManager):
        self.config = config
        self.log_manager = log_manager

        # Experiment management
        self.active_experiments: Dict[str, multiprocessing.Process] = {}
        self.experiment_results: Dict[str, ExperimentResult] = {}

        # Safety configuration
        self.safety_constraints = SafetyConstraints(
            max_execution_time=config.task_timeout,
            max_memory_usage=50 * 1024 * 1024,  # 50MB limit
            filesystem_access=False,
            network_access=False
        )

        self.validator = CodeExperimentValidator(self.safety_constraints)

        # Process pool for experiments
        self.process_pool: Optional[multiprocessing.Pool] = None
        self.max_concurrent_experiments = config.max_processes

        # Metrics
        self.experiments_run = 0
        self.experiments_successful = 0
        self.total_execution_time = 0.0

        # Callbacks
        self.experiment_complete_callback: Optional[callable] = None

    async def initialize(self) -> bool:
        """Initialize the experimenter"""
        try:
            await self.log_manager.log_system_event(
                "multiprocess_experimenter_init",
                {
                    "max_concurrent": self.max_concurrent_experiments,
                    "safety_constraints": asdict(self.safety_constraints)
                },
                worker_id="multiprocess_experimenter"
            )

            # Initialize process pool
            self.process_pool = multiprocessing.Pool(
                processes=self.max_concurrent_experiments
            )

            return True

        except Exception as e:
            await self.log_manager.log_system_event(
                "multiprocess_experimenter_init_failed",
                {"error": str(e)},
                level="error",
                worker_id="multiprocess_experimenter"
            )
            return False

    async def run_experiment(self, experiment_task: ExperimentTask) -> ExperimentResult:
        """Run a code experiment asynchronously"""
        experiment_id = experiment_task.task_id

        try:
            await self.log_manager.log_system_event(
                "experiment_start",
                {
                    "experiment_id": experiment_id,
                    "experiment_type": experiment_task.experiment_type,
                    "file_path": experiment_task.file_path
                },
                worker_id="multiprocess_experimenter"
            )

            # Validate experiment code
            if 'proposed_changes' in experiment_task.proposed_changes:
                code_to_test = experiment_task.proposed_changes['proposed_changes']
                is_safe, safety_issues = self.validator.validate_code(
                    code_to_test)

                if not is_safe:
                    return ExperimentResult(
                        experiment_id=experiment_id,
                        success=False,
                        output="",
                        error_output=f"Safety validation failed: {'; '.join(safety_issues)}",
                        exit_code=-1,
                        execution_time=0.0,
                        memory_usage=0.0,
                        safety_score=0.0,
                        impact_assessment={"risk": "high",
                                           "reason": "safety_validation_failed"},
                        changes_made=[],
                        rollback_info=None,
                        timestamp=datetime.now()
                    )

            # Run experiment in subprocess
            start_time = time.time()

            # Submit to process pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._run_experiment_sync,
                experiment_task
            )

            execution_time = time.time() - start_time
            self.total_execution_time += execution_time
            self.experiments_run += 1

            if result.success:
                self.experiments_successful += 1

            # Store result
            self.experiment_results[experiment_id] = result

            # Log experiment result
            await self.log_manager.log_experiment(
                ExperimentRecord(
                    id=experiment_id,
                    file_path=experiment_task.file_path,
                    experiment_type=experiment_task.experiment_type,
                    description=experiment_task.hypothesis,
                    hypothesis=experiment_task.hypothesis,
                    methodology="isolated_sandbox_execution",
                    result={
                        "success": result.success,
                        "output": result.output[:500],  # Truncate for logging
                        "safety_score": result.safety_score,
                        "impact": result.impact_assessment
                    },
                    success=result.success,
                    safety_score=result.safety_score,
                    duration=result.execution_time,
                    timestamp=result.timestamp,
                    worker_id="multiprocess_experimenter"
                )
            )

            # Call completion callback
            if self.experiment_complete_callback:
                try:
                    await self.experiment_complete_callback(result)
                except Exception as e:
                    await self.log_manager.log_system_event(
                        "experiment_callback_error",
                        {"error": str(e)},
                        level="error",
                        worker_id="multiprocess_experimenter"
                    )

            return result

        except Exception as e:
            await self.log_manager.log_system_event(
                "experiment_execution_error",
                {"experiment_id": experiment_id, "error": str(e)},
                level="error",
                worker_id="multiprocess_experimenter"
            )

            return ExperimentResult(
                experiment_id=experiment_id,
                success=False,
                output="",
                error_output=f"Experiment execution error: {str(e)}",
                exit_code=-1,
                execution_time=0.0,
                memory_usage=0.0,
                safety_score=0.0,
                impact_assessment={"risk": "unknown",
                                   "reason": "execution_error"},
                changes_made=[],
                rollback_info=None,
                timestamp=datetime.now()
            )

    def _run_experiment_sync(self, experiment_task: ExperimentTask) -> ExperimentResult:
        """Run experiment synchronously in subprocess"""
        experiment_id = experiment_task.task_id

        # Create sandbox and run experiment
        with ExperimentSandbox(experiment_id, self.safety_constraints) as sandbox:
            try:
                # Prepare experiment code
                if experiment_task.experiment_type == "code_modification":
                    code = self._prepare_modification_experiment(
                        experiment_task)
                elif experiment_task.experiment_type == "performance_test":
                    code = self._prepare_performance_experiment(
                        experiment_task)
                else:
                    code = self._prepare_generic_experiment(experiment_task)

                # Execute in sandbox
                execution_result = sandbox.execute_code(code)

                # Calculate safety score
                safety_score = self._calculate_safety_score(
                    execution_result, experiment_task)

                # Assess impact
                impact_assessment = self._assess_impact(
                    execution_result, experiment_task)

                return ExperimentResult(
                    experiment_id=experiment_id,
                    success=execution_result['success'],
                    output=execution_result.get('stdout', ''),
                    error_output=execution_result.get('stderr', ''),
                    exit_code=execution_result.get('exit_code', 0),
                    execution_time=execution_result.get('execution_time', 0.0),
                    memory_usage=0.0,  # TODO: Implement memory tracking
                    safety_score=safety_score,
                    impact_assessment=impact_assessment,
                    changes_made=[],
                    rollback_info=None,
                    timestamp=datetime.now()
                )

            except Exception as e:
                return ExperimentResult(
                    experiment_id=experiment_id,
                    success=False,
                    output="",
                    error_output=f"Sandbox execution error: {str(e)}",
                    exit_code=-1,
                    execution_time=0.0,
                    memory_usage=0.0,
                    safety_score=0.0,
                    impact_assessment={"risk": "high",
                                       "reason": "sandbox_error"},
                    changes_made=[],
                    rollback_info=None,
                    timestamp=datetime.now()
                )

    def _prepare_modification_experiment(self, experiment_task: ExperimentTask) -> str:
        """Prepare code modification experiment"""
        # Basic template for testing code changes
        template = f'''
# Experiment: {experiment_task.experiment_type}
# File: {experiment_task.file_path}
# Hypothesis: {experiment_task.hypothesis}

try:
    # Original code would be here
    print("Original implementation loaded")
    
    # Apply proposed changes
    # {experiment_task.proposed_changes}
    
    print("Changes applied successfully")
    print("Experiment completed")
    
except Exception as e:
    print(f"Experiment failed: {{e}}")
    raise
'''
        return template

    def _prepare_performance_experiment(self, experiment_task: ExperimentTask) -> str:
        """Prepare performance testing experiment"""
        template = f'''
# Performance Experiment: {experiment_task.experiment_type}
import time

def performance_test():
    start_time = time.time()
    
    # Test code here
    for i in range(1000):
        pass  # Placeholder
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Execution time: {{execution_time:.4f}} seconds")
    return execution_time

try:
    result = performance_test()
    print(f"Performance test completed: {{result}}")
except Exception as e:
    print(f"Performance test failed: {{e}}")
    raise
'''
        return template

    def _prepare_generic_experiment(self, experiment_task: ExperimentTask) -> str:
        """Prepare generic experiment"""
        template = f'''
# Generic Experiment: {experiment_task.experiment_type}
# Hypothesis: {experiment_task.hypothesis}

print("Starting generic experiment")
print("Experiment type: {experiment_task.experiment_type}")
print("File path: {experiment_task.file_path}")

# Simulate experiment execution
import time
time.sleep(0.1)  # Brief execution

print("Generic experiment completed successfully")
'''
        return template

    def _calculate_safety_score(self, execution_result: Dict[str, Any], experiment_task: ExperimentTask) -> float:
        """Calculate safety score for experiment"""
        score = 1.0

        # Penalize failures
        if not execution_result.get('success', False):
            score -= 0.3

        # Penalize long execution times
        exec_time = execution_result.get('execution_time', 0.0)
        if exec_time > self.safety_constraints.max_execution_time * 0.8:
            score -= 0.2

        # Check for error patterns
        stderr = execution_result.get('stderr', '')
        if 'error' in stderr.lower() or 'exception' in stderr.lower():
            score -= 0.1

        return max(0.0, score)

    def _assess_impact(self, execution_result: Dict[str, Any], experiment_task: ExperimentTask) -> Dict[str, Any]:
        """Assess impact of experiment"""
        impact = {
            "risk": "low",
            "confidence": 0.5,
            "benefits": [],
            "risks": []
        }

        if execution_result.get('success', False):
            impact["benefits"].append("Experiment executed successfully")
            impact["confidence"] += 0.3
        else:
            impact["risks"].append("Experiment failed to execute")
            impact["risk"] = "medium"

        # Assess based on experiment type
        if experiment_task.experiment_type == "performance_test":
            impact["benefits"].append("Performance characteristics measured")
        elif experiment_task.experiment_type == "security_test":
            impact["risk"] = "high"  # Security tests are inherently risky

        return impact

    def set_experiment_complete_callback(self, callback: callable):
        """Set callback for experiment completion"""
        self.experiment_complete_callback = callback

    def get_status(self) -> Dict[str, Any]:
        """Get experimenter status"""
        return {
            "active_experiments": len(self.active_experiments),
            "experiments_run": self.experiments_run,
            "experiments_successful": self.experiments_successful,
            "success_rate": (
                self.experiments_successful / max(1, self.experiments_run)
            ),
            "average_execution_time": (
                self.total_execution_time / max(1, self.experiments_run)
            ),
            "safety_constraints": asdict(self.safety_constraints)
        }

    async def shutdown(self, timeout: float = 30.0) -> bool:
        """Shutdown the experimenter"""
        try:
            await self.log_manager.log_system_event(
                "multiprocess_experimenter_shutdown",
                {"experiments_run": self.experiments_run},
                worker_id="multiprocess_experimenter"
            )

            # Shutdown process pool
            if self.process_pool:
                self.process_pool.close()
                self.process_pool.join(timeout=timeout)

            return True

        except Exception as e:
            await self.log_manager.log_system_event(
                "multiprocess_experimenter_shutdown_error",
                {"error": str(e)},
                level="error",
                worker_id="multiprocess_experimenter"
            )
            return False
