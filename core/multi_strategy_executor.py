"""
Multi-Strategy Execution Engine

Sophisticated execution system supporting multiple operational modes including
parallel execution, sequential learning, hybrid execution, and adaptive switching.
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Strategy execution modes"""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class ExecutionStatus(Enum):
    """Status of strategy execution"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PARTIAL_SUCCESS = "partial_success"


class AdaptiveTrigger(Enum):
    """Triggers for adaptive strategy switching"""
    FAILURE_RATE = "failure_rate"
    PROGRESS_STALL = "progress_stall"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    BETTER_ALTERNATIVE = "better_alternative"
    TIME_CONSTRAINT = "time_constraint"


@dataclass
class ExecutionResult:
    """Result of strategy execution"""
    strategy_id: str
    status: ExecutionStatus
    success: bool
    artifact: Any
    execution_time: float
    intermediate_results: List[Any] = field(default_factory=list)
    error_message: str = ""
    resource_usage: Dict[str, float] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)
    success_factors: List[str] = field(default_factory=list)
    failure_points: List[str] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class ExecutionContext:
    """Context for strategy execution"""
    build_id: str
    description: str
    constraints: Dict[str, Any]
    shared_resources: Dict[str, Any]
    progress_callback: Optional[Callable] = None
    cancellation_token: Optional[asyncio.Event] = None


@dataclass
class ParallelExecution:
    """Configuration for parallel execution"""
    max_concurrent: int = 3
    resource_sharing: bool = True
    result_aggregation: str = "best_result"  # "best_result", "majority_vote", "ensemble"
    timeout_per_strategy: float = 3600.0
    failure_tolerance: float = 0.5  # Continue if this fraction succeeds


@dataclass
class SequentialExecution:
    """Configuration for sequential execution"""
    learning_enabled: bool = True
    early_termination: bool = True
    success_threshold: float = 0.8
    adaptation_frequency: int = 1  # Adapt after every N strategies
    carry_forward_results: bool = True


@dataclass
class HybridExecution:
    """Configuration for hybrid execution"""
    parallel_phases: List[str] = field(default_factory=lambda: ["exploration", "refinement"])
    sequential_phases: List[str] = field(default_factory=lambda: ["analysis", "validation"])
    phase_transitions: Dict[str, str] = field(default_factory=dict)
    dynamic_switching: bool = True


class MultiStrategyExecutor:
    """
    Sophisticated execution system for multiple building strategies
    """
    
    def __init__(self, agi_system, action_manager=None):
        self.agi_system = agi_system
        self.action_manager = action_manager or agi_system.action_manager
        
        # Execution tracking
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[ExecutionResult] = []
        
        # Resource management
        self.resource_pool = {
            "computational_threads": ThreadPoolExecutor(max_workers=4),
            "memory_budget": 1024 * 1024 * 1024,  # 1GB
            "time_budget": 3600.0,  # 1 hour
            "api_calls_remaining": 1000
        }
        
        # Learning from execution
        self.strategy_performance_map: Dict[str, List[float]] = {}
        self.execution_patterns: Dict[str, Any] = {}
        
        # Configuration
        self.default_parallel_config = ParallelExecution()
        self.default_sequential_config = SequentialExecution()
        self.default_hybrid_config = HybridExecution()
        
        # Adaptive switching
        self.adaptive_triggers: Dict[AdaptiveTrigger, float] = {
            AdaptiveTrigger.FAILURE_RATE: 0.7,  # Switch if >70% fail
            AdaptiveTrigger.PROGRESS_STALL: 300.0,  # Switch after 5min stall
            AdaptiveTrigger.RESOURCE_EXHAUSTION: 0.9,  # Switch at 90% resource usage
            AdaptiveTrigger.BETTER_ALTERNATIVE: 0.2,  # Switch if alternative is 20% better
            AdaptiveTrigger.TIME_CONSTRAINT: 0.8  # Switch at 80% time elapsed
        }
        
        logger.info("Multi-Strategy Executor initialized")
    
    async def execute_strategies(
        self,
        build_id: str,
        strategies: List[Dict[str, Any]],
        mode: Union[ExecutionMode, str] = ExecutionMode.ADAPTIVE,
        parallel: bool = True,
        max_iterations: int = 3,
        failure_tolerance: float = 0.5,
        adaptive_switching: bool = True,
        timeout_seconds: float = 3600.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute multiple strategies with sophisticated coordination
        """
        # Convert string mode to enum if needed
        if isinstance(mode, str):
            try:
                mode = ExecutionMode(mode.lower())
            except ValueError:
                mode = ExecutionMode.ADAPTIVE
        
        logger.info(f"Executing {len(strategies)} strategies for build {build_id} in {mode.value} mode")
        
        # Initialize execution context
        execution_context = ExecutionContext(
            build_id=build_id,
            description=kwargs.get('description', 'Unknown build challenge'),
            constraints=kwargs.get('constraints', {}),
            shared_resources=self._initialize_shared_resources(),
            cancellation_token=asyncio.Event()
        )
        
        # Track execution
        execution_id = f"{build_id}_{int(time.time())}"
        self.active_executions[execution_id] = {
            'build_id': build_id,
            'mode': mode,
            'strategies': strategies,
            'start_time': time.time(),
            'status': 'running',
            'results': [],
            'current_iteration': 0,
            'max_iterations': max_iterations
        }
        
        try:
            # Execute based on mode
            if mode == ExecutionMode.PARALLEL:
                result = await self._execute_parallel(
                    execution_context, strategies, 
                    ParallelExecution(
                        max_concurrent=min(len(strategies), self.default_parallel_config.max_concurrent),
                        failure_tolerance=failure_tolerance,
                        timeout_per_strategy=timeout_seconds / len(strategies)
                    )
                )
            elif mode == ExecutionMode.SEQUENTIAL:
                result = await self._execute_sequential(
                    execution_context, strategies,
                    SequentialExecution(
                        early_termination=adaptive_switching,
                        success_threshold=1.0 - failure_tolerance
                    )
                )
            elif mode == ExecutionMode.HYBRID:
                result = await self._execute_hybrid(
                    execution_context, strategies,
                    HybridExecution(dynamic_switching=adaptive_switching)
                )
            else:  # ADAPTIVE mode
                result = await self._execute_adaptive(
                    execution_context, strategies, max_iterations, 
                    failure_tolerance, timeout_seconds, **kwargs
                )
            
            # Update execution tracking
            self.active_executions[execution_id]['status'] = 'completed'
            self.active_executions[execution_id]['results'] = result.get('execution_results', [])
            
            return result
            
        except Exception as e:
            logger.error(f"Strategy execution failed for build {build_id}: {e}", exc_info=True)
            self.active_executions[execution_id]['status'] = 'failed'
            return {
                'success': False,
                'message': f"Execution failed: {str(e)}",
                'execution_results': [],
                'lessons': [f"Execution system error: {str(e)}"]
            }
        
        finally:
            # Clean up execution
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    async def cancel_execution(self, build_id: str) -> bool:
        """Cancel active execution for a build"""
        cancelled = False
        
        for execution_id, execution in list(self.active_executions.items()):
            if execution['build_id'] == build_id:
                execution['status'] = 'cancelled'
                if 'cancellation_token' in execution:
                    execution['cancellation_token'].set()
                cancelled = True
                
                logger.info(f"Cancelled execution {execution_id} for build {build_id}")
        
        return cancelled
    
    async def get_execution_progress(self, build_id: str) -> Optional[Dict[str, Any]]:
        """Get progress information for active execution"""
        for execution_id, execution in self.active_executions.items():
            if execution['build_id'] == build_id:
                elapsed_time = time.time() - execution['start_time']
                
                return {
                    'execution_id': execution_id,
                    'mode': execution['mode'].value,
                    'elapsed_time': elapsed_time,
                    'current_iteration': execution['current_iteration'],
                    'max_iterations': execution['max_iterations'],
                    'strategies_total': len(execution['strategies']),
                    'results_count': len(execution['results']),
                    'status': execution['status']
                }
        
        return None
    
    # Core execution methods
    
    async def _execute_parallel(
        self,
        context: ExecutionContext,
        strategies: List[Dict[str, Any]],
        config: ParallelExecution
    ) -> Dict[str, Any]:
        """Execute strategies in parallel with resource coordination"""
        logger.info(f"Starting parallel execution of {len(strategies)} strategies")
        
        # Limit concurrent executions
        semaphore = asyncio.Semaphore(config.max_concurrent)
        
        async def execute_single_strategy(strategy):
            async with semaphore:
                return await self._execute_single_strategy_async(context, strategy, config.timeout_per_strategy)
        
        # Launch all strategies
        tasks = [execute_single_strategy(strategy) for strategy in strategies]
        
        # Collect results as they complete
        results = []
        successful_results = []
        
        try:
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    results.append(result)
                    
                    if result.success:
                        successful_results.append(result)
                        
                        # Check if we can terminate early
                        if len(successful_results) >= len(strategies) * (1 - config.failure_tolerance):
                            logger.info(f"Early termination: {len(successful_results)} successes achieved")
                            break
                            
                except Exception as e:
                    logger.warning(f"Strategy execution failed: {e}")
                    results.append(ExecutionResult(
                        strategy_id="unknown",
                        status=ExecutionStatus.FAILED,
                        success=False,
                        artifact=None,
                        execution_time=0.0,
                        error_message=str(e)
                    ))
        
        except asyncio.CancelledError:
            logger.info("Parallel execution cancelled")
            return {
                'success': False,
                'message': 'Execution cancelled',
                'execution_results': results
            }
        
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Aggregate results
        return self._aggregate_parallel_results(results, config)
    
    async def _execute_sequential(
        self,
        context: ExecutionContext,
        strategies: List[Dict[str, Any]],
        config: SequentialExecution
    ) -> Dict[str, Any]:
        """Execute strategies sequentially with learning between attempts"""
        logger.info(f"Starting sequential execution of {len(strategies)} strategies")
        
        results = []
        accumulated_knowledge = {}
        
        for i, strategy in enumerate(strategies):
            logger.info(f"Executing strategy {i+1}/{len(strategies)}: {strategy.get('name', 'Unknown')}")
            
            # Apply accumulated knowledge to strategy
            if config.learning_enabled and accumulated_knowledge:
                strategy = self._adapt_strategy_with_knowledge(strategy, accumulated_knowledge)
            
            # Execute strategy
            result = await self._execute_single_strategy_async(context, strategy)
            results.append(result)
            
            # Extract knowledge for future strategies
            if config.learning_enabled:
                knowledge = self._extract_execution_knowledge(result, strategy)
                accumulated_knowledge.update(knowledge)
            
            # Check for early termination
            if config.early_termination and result.success:
                if result.confidence_score >= config.success_threshold:
                    logger.info(f"Early termination: success threshold {config.success_threshold} achieved")
                    break
            
            # Adaptive strategy modification
            if config.adaptation_frequency > 0 and (i + 1) % config.adaptation_frequency == 0:
                remaining_strategies = strategies[i+1:]
                if remaining_strategies:
                    adapted_strategies = await self._adapt_remaining_strategies(
                        remaining_strategies, results, accumulated_knowledge
                    )
                    strategies[i+1:] = adapted_strategies
        
        return self._aggregate_sequential_results(results, config)
    
    async def _execute_hybrid(
        self,
        context: ExecutionContext,
        strategies: List[Dict[str, Any]],
        config: HybridExecution
    ) -> Dict[str, Any]:
        """Execute strategies using hybrid parallel/sequential approach"""
        logger.info(f"Starting hybrid execution of {len(strategies)} strategies")
        
        all_results = []
        
        # Phase 1: Parallel exploration of different approaches
        exploration_strategies = self._select_strategies_for_phase(strategies, "exploration")
        if exploration_strategies:
            parallel_config = ParallelExecution(max_concurrent=min(3, len(exploration_strategies)))
            exploration_results = await self._execute_parallel(context, exploration_strategies, parallel_config)
            all_results.extend(exploration_results.get('execution_results', []))
            
            # Identify promising approaches
            promising_strategies = self._identify_promising_strategies(
                exploration_results.get('execution_results', []), strategies
            )
        else:
            promising_strategies = strategies[:2]  # Fallback to first 2
        
        # Phase 2: Sequential refinement of promising approaches
        if promising_strategies:
            sequential_config = SequentialExecution(learning_enabled=True, early_termination=True)
            refinement_results = await self._execute_sequential(context, promising_strategies, sequential_config)
            all_results.extend(refinement_results.get('execution_results', []))
        
        return self._aggregate_hybrid_results(all_results, config)
    
    async def _execute_adaptive(
        self,
        context: ExecutionContext,
        strategies: List[Dict[str, Any]],
        max_iterations: int,
        failure_tolerance: float,
        timeout_seconds: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute strategies with adaptive mode switching"""
        logger.info(f"Starting adaptive execution of {len(strategies)} strategies")
        
        start_time = time.time()
        all_results = []
        current_mode = ExecutionMode.PARALLEL  # Start with parallel
        iteration = 0
        
        remaining_strategies = strategies.copy()
        
        while iteration < max_iterations and remaining_strategies and not context.cancellation_token.is_set():
            iteration += 1
            elapsed_time = time.time() - start_time
            
            logger.info(f"Adaptive iteration {iteration}/{max_iterations}, mode: {current_mode.value}")
            
            # Check if we need to switch modes
            switch_trigger = self._check_adaptive_triggers(
                all_results, elapsed_time, timeout_seconds, remaining_strategies
            )
            
            if switch_trigger:
                new_mode = self._select_adaptive_mode(current_mode, switch_trigger, all_results)
                if new_mode != current_mode:
                    logger.info(f"Adaptive switch: {current_mode.value} -> {new_mode.value} (trigger: {switch_trigger.value})")
                    current_mode = new_mode
            
            # Execute with current mode
            if current_mode == ExecutionMode.PARALLEL:
                batch_size = min(3, len(remaining_strategies))
                batch_strategies = remaining_strategies[:batch_size]
                config = ParallelExecution(
                    max_concurrent=batch_size,
                    failure_tolerance=failure_tolerance,
                    timeout_per_strategy=(timeout_seconds - elapsed_time) / batch_size
                )
                batch_results = await self._execute_parallel(context, batch_strategies, config)
                
            else:  # Sequential mode
                batch_size = min(2, len(remaining_strategies))
                batch_strategies = remaining_strategies[:batch_size]
                config = SequentialExecution(
                    learning_enabled=True,
                    early_termination=True,
                    success_threshold=1.0 - failure_tolerance
                )
                batch_results = await self._execute_sequential(context, batch_strategies, config)
            
            # Collect results
            batch_execution_results = batch_results.get('execution_results', [])
            all_results.extend(batch_execution_results)
            
            # Remove executed strategies
            executed_count = len(batch_execution_results)
            remaining_strategies = remaining_strategies[executed_count:]
            
            # Check termination conditions
            successful_results = [r for r in all_results if r.success]
            if len(successful_results) >= len(strategies) * (1 - failure_tolerance):
                logger.info(f"Adaptive execution terminating: sufficient successes achieved")
                break
            
            # Check timeout
            if elapsed_time >= timeout_seconds * 0.9:
                logger.info(f"Adaptive execution terminating: timeout approaching")
                break
        
        return self._aggregate_adaptive_results(all_results, strategies, max_iterations)
    
    # Strategy execution core
    
    async def _execute_single_strategy_async(
        self,
        context: ExecutionContext,
        strategy: Dict[str, Any],
        timeout: float = 3600.0
    ) -> ExecutionResult:
        """Execute a single strategy asynchronously"""
        
        strategy_id = strategy.get('id', f"strategy_{int(time.time())}")
        strategy_name = strategy.get('name', 'Unknown Strategy')
        
        logger.info(f"Executing strategy: {strategy_name}")
        
        start_time = time.time()
        
        try:
            # Create timeout context
            async with asyncio.timeout(timeout):
                # Execute strategy steps
                execution_artifact = await self._execute_strategy_steps(context, strategy)
                
                execution_time = time.time() - start_time
                
                # Evaluate success
                success, confidence = self._evaluate_strategy_success(execution_artifact, strategy)
                
                # Extract lessons learned
                lessons = self._extract_lessons_from_execution(strategy, execution_artifact, success)
                
                result = ExecutionResult(
                    strategy_id=strategy_id,
                    status=ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED,
                    success=success,
                    artifact=execution_artifact,
                    execution_time=execution_time,
                    confidence_score=confidence,
                    lessons_learned=lessons,
                    success_factors=strategy.get('success_factors', []) if success else [],
                    failure_points=strategy.get('failure_modes', []) if not success else []
                )
                
                # Record performance for learning
                self._record_strategy_performance(strategy_id, result)
                
                return result
                
        except asyncio.TimeoutError:
            logger.warning(f"Strategy {strategy_name} timed out after {timeout} seconds")
            return ExecutionResult(
                strategy_id=strategy_id,
                status=ExecutionStatus.TIMEOUT,
                success=False,
                artifact=None,
                execution_time=timeout,
                error_message=f"Execution timed out after {timeout} seconds"
            )
            
        except Exception as e:
            logger.error(f"Strategy {strategy_name} failed with exception: {e}", exc_info=True)
            return ExecutionResult(
                strategy_id=strategy_id,
                status=ExecutionStatus.FAILED,
                success=False,
                artifact=None,
                execution_time=time.time() - start_time,
                error_message=str(e),
                lessons_learned=[f"Exception during execution: {str(e)}"]
            )
    
    async def _execute_strategy_steps(
        self,
        context: ExecutionContext,
        strategy: Dict[str, Any]
    ) -> Any:
        """Execute the individual steps of a strategy"""
        
        strategy_type = strategy.get('strategy_type', 'experimental')
        approach_steps = strategy.get('approach_steps', [])
        
        if not approach_steps:
            # Generate default steps based on strategy type
            approach_steps = self._generate_default_steps(strategy_type)
        
        intermediate_results = []
        current_artifact = None
        
        for i, step in enumerate(approach_steps):
            logger.info(f"Executing step {i+1}/{len(approach_steps)}: {step[:50]}...")
            
            try:
                # Execute step through action manager
                step_result = await self._execute_strategy_step(context, step, current_artifact)
                intermediate_results.append(step_result)
                current_artifact = step_result
                
                # Check for early termination signals
                if context.cancellation_token.is_set():
                    logger.info("Strategy execution cancelled")
                    break
                    
            except Exception as e:
                logger.warning(f"Step {i+1} failed: {e}")
                intermediate_results.append({'error': str(e), 'step': step})
                # Continue with remaining steps unless it's a critical failure
                if 'critical' in step.lower():
                    break
        
        # Combine intermediate results into final artifact
        final_artifact = self._combine_intermediate_results(intermediate_results, strategy)
        
        return final_artifact
    
    async def _execute_strategy_step(
        self,
        context: ExecutionContext,
        step: str,
        previous_result: Any
    ) -> Any:
        """Execute a single strategy step"""
        
        # Map step description to action
        action_mapping = self._map_step_to_action(step)
        
        if not action_mapping:
            # Fallback to generic problem-solving action
            action_mapping = {
                'action': 'solve_problem',
                'problem_description': step,
                'context': context.description,
                'previous_result': str(previous_result) if previous_result else None
            }
        
        # Execute through action manager
        try:
            result = await self.action_manager.execute_action_enhanced(action_mapping)
            return result
        except Exception as e:
            logger.warning(f"Action execution failed for step '{step}': {e}")
            return {'error': str(e), 'step': step, 'attempted_action': action_mapping}
    
    # Helper methods for result aggregation and analysis
    
    def _aggregate_parallel_results(
        self,
        results: List[ExecutionResult],
        config: ParallelExecution
    ) -> Dict[str, Any]:
        """Aggregate results from parallel execution"""
        
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {
                'success': False,
                'message': f'All {len(results)} parallel strategies failed',
                'execution_results': results,
                'lessons': self._extract_aggregate_lessons(results)
            }
        
        # Select best result based on aggregation method
        if config.result_aggregation == "best_result":
            best_result = max(successful_results, key=lambda r: r.confidence_score)
            return {
                'success': True,
                'message': f'{len(successful_results)}/{len(results)} strategies succeeded',
                'artifact': best_result.artifact,
                'execution_results': results,
                'best_strategy': best_result.strategy_id,
                'lessons': self._extract_aggregate_lessons(results)
            }
        
        # Add other aggregation methods as needed
        return {
            'success': True,
            'message': f'{len(successful_results)}/{len(results)} strategies succeeded',
            'execution_results': results,
            'lessons': self._extract_aggregate_lessons(results)
        }
    
    def _aggregate_sequential_results(
        self,
        results: List[ExecutionResult],
        config: SequentialExecution
    ) -> Dict[str, Any]:
        """Aggregate results from sequential execution"""
        
        successful_results = [r for r in results if r.success]
        
        if successful_results:
            # Use the last successful result (most refined)
            final_result = successful_results[-1]
            return {
                'success': True,
                'message': f'Sequential execution succeeded after {len(results)} attempts',
                'artifact': final_result.artifact,
                'execution_results': results,
                'final_strategy': final_result.strategy_id,
                'lessons': self._extract_aggregate_lessons(results)
            }
        else:
            return {
                'success': False,
                'message': f'All {len(results)} sequential strategies failed',
                'execution_results': results,
                'lessons': self._extract_aggregate_lessons(results)
            }
    
    # Utility methods
    
    def _initialize_shared_resources(self) -> Dict[str, Any]:
        """Initialize shared resources for strategy execution"""
        return {
            'computation_pool': self.resource_pool['computational_threads'],
            'memory_tracker': {'used': 0, 'limit': self.resource_pool['memory_budget']},
            'api_tracker': {'used': 0, 'limit': self.resource_pool['api_calls_remaining']},
            'intermediate_data': {},
            'execution_cache': {}
        }
    
    def _record_strategy_performance(self, strategy_id: str, result: ExecutionResult):
        """Record strategy performance for learning"""
        if strategy_id not in self.strategy_performance_map:
            self.strategy_performance_map[strategy_id] = []
        
        performance_score = result.confidence_score if result.success else 0.0
        self.strategy_performance_map[strategy_id].append(performance_score)
        
        # Keep only recent performances
        if len(self.strategy_performance_map[strategy_id]) > 10:
            self.strategy_performance_map[strategy_id] = self.strategy_performance_map[strategy_id][-10:]
    
    def _extract_aggregate_lessons(self, results: List[ExecutionResult]) -> List[str]:
        """Extract aggregate lessons from multiple execution results"""
        all_lessons = []
        
        for result in results:
            all_lessons.extend(result.lessons_learned)
        
        # Remove duplicates while preserving order
        unique_lessons = []
        for lesson in all_lessons:
            if lesson not in unique_lessons:
                unique_lessons.append(lesson)
        
        return unique_lessons[:10]  # Limit to top 10 lessons
    
    def _generate_default_steps(self, strategy_type: str) -> List[str]:
        """Generate default steps based on strategy type"""
        
        step_templates = {
            'physics_based': [
                "Analyze physical constraints and governing equations",
                "Create mathematical model of the system",
                "Simulate system behavior and validate assumptions",
                "Design experimental validation approach"
            ],
            'computational': [
                "Define computational requirements and constraints",
                "Design algorithm or computational approach",
                "Implement solution with error handling",
                "Test and optimize performance"
            ],
            'experimental': [
                "Design experimental approach and methodology",
                "Identify required resources and tools",
                "Execute controlled experiments",
                "Analyze results and draw conclusions"
            ],
            'heuristic': [
                "Identify relevant patterns and rules",
                "Apply heuristic methods to problem",
                "Validate results against known cases",
                "Refine approach based on outcomes"
            ]
        }
        
        return step_templates.get(strategy_type, [
            "Analyze the problem systematically",
            "Develop solution approach",
            "Implement solution",
            "Validate and refine results"
        ])
    
    def _map_step_to_action(self, step: str) -> Optional[Dict[str, Any]]:
        """Map a strategy step to an action"""
        step_lower = step.lower()
        
        if any(word in step_lower for word in ['code', 'program', 'implement', 'algorithm']):
            return {
                'action': 'write_python_code',
                'task_description': step,
                'include_tests': True
            }
        elif any(word in step_lower for word in ['experiment', 'test', 'validate']):
            return {
                'action': 'propose_and_test_invention',
                'invention_description': step
            }
        elif any(word in step_lower for word in ['analyze', 'study', 'research']):
            return {
                'action': 'log_message',
                'message': f"Analysis step: {step}",
                'level': 'info'
            }
        
        return None  # Let the system handle it generically