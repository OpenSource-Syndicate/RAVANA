"""
Intelligent Builder Controller

Central orchestration system for the Intelligent Adaptive Builder enhancement.
Manages complex building challenges, coordinates multiple strategies, and integrates
with all builder components.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from modules.personality.enhanced_personality import (
    EnhancedPersonality, BuildDifficulty, RiskTolerance
)

logger = logging.getLogger(__name__)


class BuildPhase(Enum):
    """Phases of the building process"""
    ANALYSIS = "analysis"
    STRATEGY_GENERATION = "strategy_generation"
    RESEARCH = "research"
    EXECUTION = "execution"
    EVALUATION = "evaluation"
    ITERATION = "iteration"
    COMPLETION = "completion"
    FAILURE_ANALYSIS = "failure_analysis"


class BuildStatus(Enum):
    """Status of build attempts"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REQUIRES_RESEARCH = "requires_research"
    STRATEGY_EXHAUSTED = "strategy_exhausted"


class PersistenceMode(Enum):
    """Persistence levels for build attempts"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class ResourceConstraints:
    """Resource constraints for building attempts"""
    max_time_seconds: int = 3600
    max_strategies: int = 5
    max_parallel_strategies: int = 2
    max_research_depth: int = 3
    computational_budget: float = 1.0
    memory_limit_mb: int = 1024


@dataclass
class BuildContext:
    """Context information for a building challenge"""
    domain: str
    complexity_factors: List[str]
    prerequisites: List[str]
    constraints: List[str]
    success_criteria: List[str]
    similar_challenges: List[str] = field(default_factory=list)
    historical_attempts: List[str] = field(default_factory=list)


@dataclass
class BuildProgress:
    """Progress tracking for build attempts"""
    current_phase: BuildPhase
    phase_progress: float  # 0.0 to 1.0
    overall_progress: float  # 0.0 to 1.0
    strategies_attempted: int
    strategies_failed: int
    strategies_succeeded: int
    current_strategy: Optional[str]
    intermediate_results: List[Any] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)


@dataclass
class BuildResult:
    """Result of a building attempt"""
    build_id: str
    success: bool
    final_artifact: Any
    strategies_used: List[str]
    total_time: float
    iterations: int
    failure_points: List[str]
    success_factors: List[str]
    lessons_learned: List[str]
    confidence_score: float
    novelty_score: float
    build_log: List[str] = field(default_factory=list)


class IntelligentBuilderController:
    """
    Central orchestration system for intelligent building operations
    """

    def __init__(self, agi_system, enhanced_personality: Optional[EnhancedPersonality] = None):
        self.agi_system = agi_system
        self.enhanced_personality = enhanced_personality or EnhancedPersonality()

        # Component references (will be injected)
        self.strategy_reasoning_manager = None
        self.multi_strategy_executor = None
        self.failure_analysis_engine = None
        self.online_research_agent = None

        # Active builds
        self.active_builds: Dict[str, Dict[str, Any]] = {}
        self.build_history: List[BuildResult] = []

        # Configuration
        self.default_constraints = ResourceConstraints()
        self.global_learning_enabled = True

        # Metrics
        self.total_builds_attempted = 0
        self.total_builds_succeeded = 0
        self.impossible_challenges_completed = 0

        logger.info("Intelligent Builder Controller initialized")

    def inject_components(self, strategy_manager, executor, failure_engine, research_agent):
        """Inject dependent components after initialization"""
        self.strategy_reasoning_manager = strategy_manager
        self.multi_strategy_executor = executor
        self.failure_analysis_engine = failure_engine
        self.online_research_agent = research_agent

        logger.info("Builder components injected successfully")

    async def attempt_build(
        self,
        description: str,
        difficulty_level: Union[BuildDifficulty,
                                str] = BuildDifficulty.IMPOSSIBLE,
        max_strategies: int = 5,
        parallel_execution: bool = True,
        enable_online_research: bool = True,
        persistence_mode: PersistenceMode = PersistenceMode.HIGH,
        constraints: Optional[ResourceConstraints] = None,
        context: Optional[BuildContext] = None
    ) -> BuildResult:
        """
        Attempt to build a complex artifact with intelligent strategy management
        """
        # Convert string difficulty to enum if needed
        if isinstance(difficulty_level, str):
            try:
                difficulty_level = BuildDifficulty(difficulty_level.lower())
            except ValueError:
                difficulty_level = BuildDifficulty.IMPOSSIBLE

        # Generate unique build ID
        build_id = str(uuid.uuid4())
        start_time = time.time()

        # Initialize build tracking
        build_state = {
            'id': build_id,
            'description': description,
            'difficulty': difficulty_level,
            'status': BuildStatus.PENDING,
            'phase': BuildPhase.ANALYSIS,
            'constraints': constraints or self.default_constraints,
            'context': context or BuildContext(domain='general', complexity_factors=[],
                                               prerequisites=[], constraints=[], success_criteria=[]),
            'start_time': start_time,
            'strategies': [],
            'results': [],
            'logs': [],
            'persistence_mode': persistence_mode
        }

        self.active_builds[build_id] = build_state
        self.total_builds_attempted += 1

        logger.info(
            f"Starting build attempt: '{description[:50]}...' (ID: {build_id})")

        try:
            # Phase 1: Challenge Analysis
            build_state['phase'] = BuildPhase.ANALYSIS
            await self._log_build_event(build_id, "Starting challenge analysis")

            challenge_assessment = await self.enhanced_personality.assess_building_challenge(
                description, difficulty_level.value
            )

            build_state['assessment'] = challenge_assessment
            await self._log_build_event(build_id, f"Challenge assessment complete. Attraction: {challenge_assessment['assessment']['challenge_attraction']:.2f}")

            # Check if challenge is worth attempting based on personality
            if challenge_assessment['assessment']['challenge_attraction'] < 0.3:
                await self._log_build_event(build_id, "Challenge rejected due to low attraction score")
                return await self._complete_build(build_id, False, "Challenge not engaging enough")

            # Phase 2: Strategy Generation
            build_state['phase'] = BuildPhase.STRATEGY_GENERATION
            await self._log_build_event(build_id, "Generating build strategies")

            strategies = await self._generate_build_strategies(build_id, max_strategies)
            build_state['strategies'] = strategies

            if not strategies:
                await self._log_build_event(build_id, "No viable strategies generated")
                return await self._complete_build(build_id, False, "No viable strategies found")

            await self._log_build_event(build_id, f"Generated {len(strategies)} strategies")

            # Phase 3: Research (if enabled)
            if enable_online_research and self.online_research_agent:
                build_state['phase'] = BuildPhase.RESEARCH
                await self._log_build_event(build_id, "Conducting online research")

                research_results = await self._conduct_research(build_id, description, strategies)
                build_state['research'] = research_results

                # Update strategies based on research
                strategies = await self._refine_strategies_with_research(strategies, research_results)
                build_state['strategies'] = strategies

            # Phase 4: Execution
            build_state['phase'] = BuildPhase.EXECUTION
            build_state['status'] = BuildStatus.IN_PROGRESS

            execution_result = await self._execute_build_strategies(
                build_id, strategies, parallel_execution, persistence_mode
            )

            # Phase 5: Evaluation and Completion
            build_state['phase'] = BuildPhase.EVALUATION

            success = execution_result.get('success', False)
            final_result = await self._complete_build(
                build_id,
                success,
                execution_result.get('message', 'Build completed'),
                execution_result.get('artifact')
            )

            # Record outcome with personality system
            self.enhanced_personality.record_building_outcome(build_id, {
                'success': success,
                'difficulty': difficulty_level,
                'description': description,
                'lessons': execution_result.get('lessons', [])
            })

            if success:
                self.total_builds_succeeded += 1
                if difficulty_level in [BuildDifficulty.IMPOSSIBLE, BuildDifficulty.TRANSCENDENT]:
                    self.impossible_challenges_completed += 1

            return final_result

        except Exception as e:
            logger.error(
                f"Build attempt {build_id} failed with exception: {e}", exc_info=True)
            await self._log_build_event(build_id, f"Build failed with exception: {str(e)}")
            return await self._complete_build(build_id, False, f"Exception occurred: {str(e)}")

        finally:
            # Clean up active build
            if build_id in self.active_builds:
                del self.active_builds[build_id]

    async def analyze_build_failure(
        self,
        build_id: str,
        failure_details: Dict[str, Any],
        research_similar_failures: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of build failures for learning extraction
        """
        if not self.failure_analysis_engine:
            logger.warning("Failure analysis engine not available")
            return {"analysis": "Failure analysis engine not initialized"}

        logger.info(f"Analyzing failure for build {build_id}")

        # Get build history for context
        build_context = None
        for result in self.build_history:
            if result.build_id == build_id:
                build_context = result
                break

        if not build_context:
            logger.warning(f"Build context not found for {build_id}")
            return {"analysis": "Build context not available"}

        # Perform failure analysis
        analysis = await self.failure_analysis_engine.analyze_failure(
            failure_details,
            build_context,
            research_similar_failures
        )

        # Extract lessons for personality system
        lessons = analysis.get('lessons_learned', [])
        if lessons:
            self.enhanced_personality.record_building_outcome(build_id, {
                'success': False,
                'lessons': lessons,
                'failure_analysis': analysis
            })

        logger.info(f"Failure analysis complete for build {build_id}")
        return analysis

    async def generate_alternative_strategies(
        self,
        original_strategy: Dict[str, Any],
        failure_reason: str,
        research_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate alternative approaches based on failure analysis
        """
        if not self.strategy_reasoning_manager:
            logger.warning("Strategy reasoning manager not available")
            return []

        logger.info("Generating alternative strategies based on failure")

        # Use personality insights for alternative generation
        creative_insights = self.enhanced_personality.creative_amplifier.map_cross_domain_patterns(
            f"Alternative to failed strategy: {failure_reason}",
            num_domains=4
        )

        # Generate alternatives using strategy manager
        alternatives = await self.strategy_reasoning_manager.generate_alternatives(
            original_strategy,
            failure_reason,
            creative_insights,
            research_context
        )

        logger.info(f"Generated {len(alternatives)} alternative strategies")
        return alternatives

    async def get_build_progress(self, build_id: str) -> Optional[BuildProgress]:
        """
        Monitor ongoing build attempts and intermediate results
        """
        if build_id not in self.active_builds:
            return None

        build_state = self.active_builds[build_id]

        # Calculate progress metrics
        phase_progress = self._calculate_phase_progress(build_state)
        overall_progress = self._calculate_overall_progress(build_state)

        strategies_attempted = len([s for s in build_state.get(
            'strategies', []) if s.get('attempted', False)])
        strategies_failed = len([s for s in build_state.get(
            'strategies', []) if s.get('failed', False)])
        strategies_succeeded = len([s for s in build_state.get(
            'strategies', []) if s.get('succeeded', False)])

        current_strategy = None
        if build_state.get('current_strategy'):
            current_strategy = build_state['current_strategy'].get(
                'name', 'Unknown')

        return BuildProgress(
            current_phase=build_state['phase'],
            phase_progress=phase_progress,
            overall_progress=overall_progress,
            strategies_attempted=strategies_attempted,
            strategies_failed=strategies_failed,
            strategies_succeeded=strategies_succeeded,
            current_strategy=current_strategy,
            intermediate_results=build_state.get('intermediate_results', []),
            lessons_learned=build_state.get('lessons_learned', []),
            next_actions=build_state.get('next_actions', [])
        )

    async def cancel_build(self, build_id: str, reason: str = "User requested cancellation"):
        """Cancel an active build attempt"""
        if build_id not in self.active_builds:
            logger.warning(f"Cannot cancel build {build_id}: not found")
            return False

        build_state = self.active_builds[build_id]
        build_state['status'] = BuildStatus.CANCELLED

        await self._log_build_event(build_id, f"Build cancelled: {reason}")

        # Clean up any running strategies
        if self.multi_strategy_executor:
            await self.multi_strategy_executor.cancel_execution(build_id)

        # Complete the build as cancelled
        await self._complete_build(build_id, False, f"Cancelled: {reason}")

        logger.info(f"Build {build_id} cancelled: {reason}")
        return True

    def get_build_statistics(self) -> Dict[str, Any]:
        """Get overall build statistics"""
        success_rate = (self.total_builds_succeeded / self.total_builds_attempted
                        if self.total_builds_attempted > 0 else 0.0)

        return {
            "total_builds_attempted": self.total_builds_attempted,
            "total_builds_succeeded": self.total_builds_succeeded,
            "success_rate": success_rate,
            "impossible_challenges_completed": self.impossible_challenges_completed,
            "active_builds": len(self.active_builds),
            "personality_stats": {
                "current_risk_tolerance": self.enhanced_personality.risk_controller.current_risk_tolerance.value,
                "creativity_level": self.enhanced_personality.creativity,
                "learning_momentum": self.enhanced_personality.learning_momentum
            }
        }

    # Private helper methods

    async def _generate_build_strategies(self, build_id: str, max_strategies: int) -> List[Dict[str, Any]]:
        """Generate initial build strategies"""
        if not self.strategy_reasoning_manager:
            logger.warning("Strategy reasoning manager not available")
            return []

        build_state = self.active_builds[build_id]
        description = build_state['description']
        context = build_state['context']
        assessment = build_state.get('assessment', {})

        strategies = await self.strategy_reasoning_manager.generate_strategies(
            description,
            context,
            assessment,
            max_strategies
        )

        return strategies

    async def _conduct_research(self, build_id: str, description: str, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Conduct online research for the build challenge"""
        if not self.online_research_agent:
            return {}

        research_results = await self.online_research_agent.research_build_challenge(
            description,
            [s.get('name', '') for s in strategies],
            depth=self.active_builds[build_id]['constraints'].max_research_depth
        )

        return research_results

    async def _refine_strategies_with_research(self, strategies: List[Dict[str, Any]], research: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Refine strategies based on research results"""
        if not research or not self.strategy_reasoning_manager:
            return strategies

        refined_strategies = await self.strategy_reasoning_manager.refine_with_research(
            strategies, research
        )

        return refined_strategies

    async def _execute_build_strategies(
        self,
        build_id: str,
        strategies: List[Dict[str, Any]],
        parallel: bool,
        persistence_mode: PersistenceMode
    ) -> Dict[str, Any]:
        """Execute build strategies with the multi-strategy executor"""
        if not self.multi_strategy_executor:
            logger.error("Multi-strategy executor not available")
            return {"success": False, "message": "Executor not initialized"}

        build_state = self.active_builds[build_id]
        constraints = build_state['constraints']

        # Configure execution parameters based on persistence mode
        execution_params = self._get_execution_params(
            persistence_mode, constraints)

        result = await self.multi_strategy_executor.execute_strategies(
            build_id,
            strategies,
            parallel=parallel,
            **execution_params
        )

        return result

    def _get_execution_params(self, persistence_mode: PersistenceMode, constraints: ResourceConstraints) -> Dict[str, Any]:
        """Get execution parameters based on persistence mode"""
        base_params = {
            "max_iterations": 3,
            "failure_tolerance": 0.5,
            "adaptive_switching": True
        }

        if persistence_mode == PersistenceMode.LOW:
            base_params.update({
                "max_iterations": 1,
                "failure_tolerance": 0.2,
                "timeout_seconds": constraints.max_time_seconds // 4
            })
        elif persistence_mode == PersistenceMode.MODERATE:
            base_params.update({
                "max_iterations": 2,
                "failure_tolerance": 0.4,
                "timeout_seconds": constraints.max_time_seconds // 2
            })
        elif persistence_mode == PersistenceMode.HIGH:
            base_params.update({
                "max_iterations": 5,
                "failure_tolerance": 0.7,
                "timeout_seconds": constraints.max_time_seconds
            })
        elif persistence_mode == PersistenceMode.EXTREME:
            base_params.update({
                "max_iterations": 10,
                "failure_tolerance": 0.9,
                "timeout_seconds": constraints.max_time_seconds * 2,
                "enable_creative_mutations": True
            })

        return base_params

    async def _complete_build(self, build_id: str, success: bool, message: str, artifact: Any = None) -> BuildResult:
        """Complete a build attempt and record results"""
        build_state = self.active_builds.get(build_id, {})

        end_time = time.time()
        total_time = end_time - build_state.get('start_time', end_time)

        # Gather execution details
        strategies_used = [s.get('name', 'Unknown') for s in build_state.get(
            'strategies', []) if s.get('attempted', False)]
        failure_points = []
        success_factors = []
        lessons_learned = build_state.get('lessons_learned', [])

        # Extract failure points and success factors from execution results
        for result in build_state.get('results', []):
            if result.get('success', False):
                success_factors.extend(result.get('success_factors', []))
            else:
                failure_points.extend(result.get('failure_points', []))

            lessons_learned.extend(result.get('lessons', []))

        # Calculate confidence and novelty scores
        confidence_score = build_state.get('assessment', {}).get(
            'assessment', {}).get('confidence', 0.5)
        novelty_score = self._calculate_novelty_score(build_state)

        # Create build result
        result = BuildResult(
            build_id=build_id,
            success=success,
            final_artifact=artifact,
            strategies_used=strategies_used,
            total_time=total_time,
            iterations=len(build_state.get('results', [])),
            failure_points=list(set(failure_points)),  # Remove duplicates
            success_factors=list(set(success_factors)),
            lessons_learned=list(set(lessons_learned)),
            confidence_score=confidence_score,
            novelty_score=novelty_score,
            build_log=build_state.get('logs', [])
        )

        # Add to history
        self.build_history.append(result)

        # Keep history bounded
        if len(self.build_history) > 1000:
            self.build_history = self.build_history[-1000:]

        logger.info(
            f"Build {build_id} completed. Success: {success}. Message: {message}")
        return result

    async def _log_build_event(self, build_id: str, message: str):
        """Log an event for a build attempt"""
        if build_id in self.active_builds:
            timestamp = datetime.now().isoformat()
            log_entry = f"[{timestamp}] {message}"
            self.active_builds[build_id].setdefault(
                'logs', []).append(log_entry)

        logger.info(f"Build {build_id}: {message}")

    def _calculate_phase_progress(self, build_state: Dict[str, Any]) -> float:
        """Calculate progress within the current phase"""
        phase = build_state.get('phase', BuildPhase.ANALYSIS)

        if phase == BuildPhase.ANALYSIS:
            return 1.0 if 'assessment' in build_state else 0.5
        elif phase == BuildPhase.STRATEGY_GENERATION:
            return 1.0 if build_state.get('strategies') else 0.3
        elif phase == BuildPhase.RESEARCH:
            return 1.0 if build_state.get('research') else 0.2
        elif phase == BuildPhase.EXECUTION:
            strategies = build_state.get('strategies', [])
            if not strategies:
                return 0.0
            attempted = len(
                [s for s in strategies if s.get('attempted', False)])
            return attempted / len(strategies)
        else:
            return 1.0

    def _calculate_overall_progress(self, build_state: Dict[str, Any]) -> float:
        """Calculate overall progress across all phases"""
        phase_weights = {
            BuildPhase.ANALYSIS: 0.15,
            BuildPhase.STRATEGY_GENERATION: 0.20,
            BuildPhase.RESEARCH: 0.15,
            BuildPhase.EXECUTION: 0.40,
            BuildPhase.EVALUATION: 0.10
        }

        current_phase = build_state.get('phase', BuildPhase.ANALYSIS)
        completed_weight = 0.0

        # Add weight for completed phases
        for phase, weight in phase_weights.items():
            if phase.value < current_phase.value:  # Assuming enum ordering
                completed_weight += weight
            elif phase == current_phase:
                phase_progress = self._calculate_phase_progress(build_state)
                completed_weight += weight * phase_progress
                break

        return min(1.0, completed_weight)

    def _calculate_novelty_score(self, build_state: Dict[str, Any]) -> float:
        """Calculate novelty score based on approach and insights used"""
        base_novelty = 0.5

        # Increase novelty for cross-domain insights
        assessment = build_state.get('assessment', {})
        insights = assessment.get('approach', {}).get(
            'cross_domain_insights', [])
        novelty_bonus = len(insights) * 0.1

        # Increase novelty for impossible challenges
        difficulty = build_state.get('difficulty', BuildDifficulty.MODERATE)
        if difficulty == BuildDifficulty.IMPOSSIBLE:
            novelty_bonus += 0.3
        elif difficulty == BuildDifficulty.TRANSCENDENT:
            novelty_bonus += 0.5

        # Factor in creative synthesis
        synthesis = assessment.get('approach', {}).get('novel_synthesis', {})
        if synthesis and synthesis.get('confidence', 0) > 0.7:
            novelty_bonus += 0.2

        return min(1.0, base_novelty + novelty_bonus)
