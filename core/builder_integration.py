"""
Builder System Integration

Integration layer that connects the Intelligent Adaptive Builder components
with the existing RAVANA AGI system architecture.
"""

import logging
from typing import Dict, List, Any
from sqlmodel import Session

from core.system import AGISystem
from core.intelligent_builder_controller import IntelligentBuilderController
from core.strategy_reasoning_manager import StrategyReasoningManager
from core.multi_strategy_executor import MultiStrategyExecutor
from core.failure_analysis_engine import FailureAnalysisEngine
from core.builder_models import (
    create_builder_tables, BuildAttempt, BuildStrategy,
    StrategyExecution, FailureAnalysis, PersonalityState
)
from modules.personality.enhanced_personality import EnhancedPersonality

logger = logging.getLogger(__name__)


class IntelligentBuilderSystem:
    """
    Integrated intelligent builder system that extends RAVANA's capabilities
    """

    def __init__(self, agi_system: AGISystem):
        self.agi_system = agi_system
        self.engine = agi_system.engine
        self.session = Session(self.engine)

        # Initialize enhanced personality system
        self.enhanced_personality = EnhancedPersonality(
            name=agi_system.personality.name,
            origin=agi_system.personality.origin,
            traits=agi_system.personality.traits.copy(),
            creativity=agi_system.personality.creativity
        )

        # Initialize core builder components
        self.strategy_reasoning_manager = StrategyReasoningManager(
            agi_system, self.enhanced_personality
        )

        self.multi_strategy_executor = MultiStrategyExecutor(
            agi_system, agi_system.action_manager
        )

        self.failure_analysis_engine = FailureAnalysisEngine(
            agi_system, agi_system.knowledge_service
        )

        # Initialize central controller
        self.builder_controller = IntelligentBuilderController(
            agi_system, self.enhanced_personality
        )

        # Inject component dependencies
        self.builder_controller.inject_components(
            self.strategy_reasoning_manager,
            self.multi_strategy_executor,
            self.failure_analysis_engine,
            None  # Online research agent placeholder
        )

        # Integration state
        self.integration_active = False
        self.builder_session = None

        logger.info("Intelligent Builder System initialized")

    async def initialize_builder_system(self):
        """Initialize the builder system and integrate with AGI"""
        try:
            # Create database tables
            create_builder_tables(self.engine)
            logger.info("Builder database tables created")

            # Initialize personality state
            await self._initialize_personality_state()

            # Integrate with AGI system decision-making
            await self._integrate_with_decision_engine()

            # Integrate with AGI action system
            await self._integrate_with_action_system()

            # Integrate with AGI memory and learning
            await self._integrate_with_memory_system()

            self.integration_active = True
            logger.info("Builder system integration complete")

        except Exception as e:
            logger.error(
                f"Failed to initialize builder system: {e}", exc_info=True)
            raise

    async def attempt_impossible_build(
        self,
        description: str,
        difficulty: str = "impossible",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main entry point for attempting impossible builds
        """
        if not self.integration_active:
            await self.initialize_builder_system()

        logger.info(f"Attempting impossible build: '{description[:50]}...'")

        try:
            # Use the builder controller for the attempt
            result = await self.builder_controller.attempt_build(
                description=description,
                difficulty_level=difficulty,
                **kwargs
            )

            # Update AGI system state with results
            await self._update_agi_system_state(result)

            return {
                'success': result.success,
                'build_id': result.build_id,
                'artifact': result.final_artifact,
                'strategies_used': result.strategies_used,
                'lessons_learned': result.lessons_learned,
                'confidence_score': result.confidence_score,
                'novelty_score': result.novelty_score,
                'execution_time': result.total_time
            }

        except Exception as e:
            logger.error(
                f"Impossible build attempt failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'lessons_learned': [f"System error during build attempt: {str(e)}"]
            }

    async def enhance_agi_decision_making(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance AGI decision-making with builder capabilities
        """
        enhanced_context = decision_context.copy()

        # Check if this is a building-related decision
        is_building_task = await self._is_building_related_task(decision_context)

        if is_building_task:
            # Add builder-specific insights
            builder_insights = await self._generate_builder_insights(decision_context)
            enhanced_context['builder_insights'] = builder_insights

            # Add personality influence
            personality_influence = self.enhanced_personality.influence_builder_decision(
                decision_context)
            enhanced_context['builder_personality_influence'] = personality_influence

            # Add failure prevention measures
            prevention_measures = await self._suggest_failure_prevention(decision_context)
            enhanced_context['failure_prevention'] = prevention_measures

        return enhanced_context

    async def process_agi_curiosity_for_building(self, curiosity_topics: List[str]) -> List[str]:
        """
        Process AGI curiosity topics for potential building challenges
        """
        building_challenges = []

        for topic in curiosity_topics:
            # Use creative reasoning to generate building challenges
            creative_insights = self.enhanced_personality.creative_amplifier.map_cross_domain_patterns(
                f"Build something related to: {topic}", num_domains=2
            )

            for insight in creative_insights:
                challenge = f"Build a {insight.bridge_concept} inspired by {insight.metaphor}"
                building_challenges.append(challenge)

        # Filter for feasible yet ambitious challenges
        filtered_challenges = []
        for challenge in building_challenges:
            assessment = await self.enhanced_personality.assess_building_challenge(challenge)
            if assessment['assessment']['challenge_attraction'] > 0.6:
                filtered_challenges.append(challenge)

        return filtered_challenges[:3]  # Return top 3 challenges

    async def integrate_with_reflection_system(self, reflection_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate builder insights with AGI reflection system
        """
        enhanced_reflection = reflection_data.copy()

        # Add builder performance reflection
        builder_performance = await self._analyze_builder_performance()
        enhanced_reflection['builder_performance'] = builder_performance

        # Add lessons from recent failures
        recent_lessons = await self._get_recent_failure_lessons()
        enhanced_reflection['builder_lessons'] = recent_lessons

        # Add personality evolution insights
        personality_evolution = await self._analyze_personality_evolution()
        enhanced_reflection['personality_evolution'] = personality_evolution

        return enhanced_reflection

    # Integration helper methods

    async def _initialize_personality_state(self):
        """Initialize personality state in database"""
        try:
            # Check if personality state exists
            existing_state = self.session.query(PersonalityState).first()

            if not existing_state:
                # Create new personality state
                personality_state = PersonalityState(
                    name=self.enhanced_personality.name,
                    creativity_level=self.enhanced_personality.creativity,
                    risk_tolerance=self.enhanced_personality.risk_controller.current_risk_tolerance.value,
                    base_confidence=self.enhanced_personality.risk_controller.confidence_profile.base_confidence,
                    learning_modifier=self.enhanced_personality.risk_controller.confidence_profile.learning_modifier,
                    failure_resilience=self.enhanced_personality.risk_controller.confidence_profile.failure_resilience,
                    success_amplification=self.enhanced_personality.risk_controller.confidence_profile.success_amplification,
                    impossible_attraction=self.enhanced_personality.risk_controller.confidence_profile.impossible_attraction,
                    learning_momentum=self.enhanced_personality.learning_momentum
                )

                self.session.add(personality_state)
                self.session.commit()
                logger.info("Personality state initialized in database")
            else:
                # Load existing state into enhanced personality
                self._load_personality_state(existing_state)
                logger.info("Loaded existing personality state from database")

        except Exception as e:
            logger.error(
                f"Failed to initialize personality state: {e}", exc_info=True)

    async def _integrate_with_decision_engine(self):
        """Integrate builder capabilities with AGI decision engine"""

        # Extend decision engine with builder-aware decision making
        original_make_decision = self.agi_system._make_decision

        async def enhanced_make_decision(situation):
            # Get original decision
            decision = await original_make_decision(situation)

            # Enhance with builder insights if applicable
            enhanced_context = await self.enhance_agi_decision_making({
                'situation': situation,
                'decision': decision
            })

            if 'builder_insights' in enhanced_context:
                decision['builder_insights'] = enhanced_context['builder_insights']
                decision['enhanced_by_builder'] = True

            return decision

        # Replace the decision method
        self.agi_system._make_decision = enhanced_make_decision
        logger.info("Decision engine integration complete")

    async def _integrate_with_action_system(self):
        """Integrate builder actions with AGI action system"""

        # Add builder-specific actions to the action registry
        builder_actions = {
            'attempt_impossible_build': {
                'name': 'attempt_impossible_build',
                'description': 'Attempt to build something impossible or highly complex',
                'parameters': [
                    {'name': 'description', 'type': 'string', 'required': True},
                    {'name': 'difficulty', 'type': 'string',
                        'required': False, 'default': 'impossible'},
                    {'name': 'max_strategies', 'type': 'integer',
                        'required': False, 'default': 5}
                ]
            },
            'analyze_build_failure': {
                'name': 'analyze_build_failure',
                'description': 'Analyze a build failure for learning extraction',
                'parameters': [
                    {'name': 'build_id', 'type': 'string', 'required': True},
                    {'name': 'failure_details', 'type': 'object', 'required': True}
                ]
            },
            'generate_creative_solution': {
                'name': 'generate_creative_solution',
                'description': 'Generate creative solutions using cross-domain insights',
                'parameters': [
                    {'name': 'problem', 'type': 'string', 'required': True},
                    {'name': 'domains', 'type': 'array', 'required': False}
                ]
            }
        }

        # Register actions with the action manager
        for action_name, action_def in builder_actions.items():
            self.agi_system.action_manager.action_registry.register_action(
                action_name, action_def)

        logger.info("Action system integration complete")

    async def _integrate_with_memory_system(self):
        """Integrate builder learning with AGI memory system"""

        # Extend memory system to include builder-specific memories
        original_memorize = self.agi_system._memorize_interaction

        async def enhanced_memorize_interaction(situation_prompt, decision, action_output):
            # Perform original memorization
            await original_memorize(situation_prompt, decision, action_output)

            # Add builder-specific memories if this was a building action
            if self._is_building_action(decision):
                await self._memorize_building_interaction(situation_prompt, decision, action_output)

        # Replace memorization method
        self.agi_system._memorize_interaction = enhanced_memorize_interaction
        logger.info("Memory system integration complete")

    async def _is_building_related_task(self, context: Dict[str, Any]) -> bool:
        """Check if a task is building-related"""

        situation = context.get('situation', {})
        if isinstance(situation, dict):
            prompt = situation.get('prompt', '')
        else:
            prompt = str(situation)

        building_keywords = [
            'build', 'construct', 'create', 'design', 'develop', 'implement',
            'make', 'fabricate', 'assemble', 'engineer', 'invent', 'prototype'
        ]

        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in building_keywords)

    async def _generate_builder_insights(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate builder-specific insights for decision context"""

        situation = context.get('situation', {})
        if isinstance(situation, dict):
            description = situation.get('prompt', '')
        else:
            description = str(situation)

        # Generate creative insights
        creative_insights = self.enhanced_personality.creative_amplifier.map_cross_domain_patterns(
            description, num_domains=3
        )

        # Assess challenge difficulty
        assessment = await self.enhanced_personality.assess_building_challenge(description)

        return {
            'creative_insights': [
                {
                    'domain': insight.domain_source,
                    'metaphor': insight.metaphor,
                    'application': insight.application,
                    'confidence': insight.confidence
                }
                for insight in creative_insights[:2]  # Top 2 insights
            ],
            'challenge_assessment': assessment['assessment'],
            'personality_response': assessment['personality_response'],
            'recommended_approach': assessment['approach']['novel_synthesis']
        }

    async def _suggest_failure_prevention(self, context: Dict[str, Any]) -> List[str]:
        """Suggest failure prevention measures"""

        # Get common failure patterns
        common_failures = await self._get_common_failure_patterns()

        prevention_measures = [
            "Conduct thorough requirements analysis before implementation",
            "Design comprehensive test cases including edge cases",
            "Implement robust error handling and recovery mechanisms",
            "Monitor resource usage throughout execution",
            "Plan for scalability and performance requirements"
        ]

        # Add domain-specific prevention measures
        domain_measures = await self._get_domain_specific_prevention(context)
        prevention_measures.extend(domain_measures)

        return prevention_measures[:5]  # Return top 5 measures

    async def _update_agi_system_state(self, build_result):
        """Update AGI system state with build results"""

        # Update shared state with build outcomes
        if hasattr(self.agi_system, 'shared_state'):
            if not hasattr(self.agi_system.shared_state, 'builder_results'):
                self.agi_system.shared_state.builder_results = []

            self.agi_system.shared_state.builder_results.append({
                'build_id': build_result.build_id,
                'success': build_result.success,
                'confidence': build_result.confidence_score,
                'lessons': build_result.lessons_learned[-3:] if build_result.lessons_learned else []
            })

            # Keep only recent results
            if len(self.agi_system.shared_state.builder_results) > 10:
                self.agi_system.shared_state.builder_results = (
                    self.agi_system.shared_state.builder_results[-10:]
                )

        # Update personality system with outcomes
        self.enhanced_personality.record_building_outcome(build_result.build_id, {
            'success': build_result.success,
            'difficulty': 'impossible',  # Most builds are impossible challenges
            'lessons': build_result.lessons_learned
        })

    def _load_personality_state(self, state: PersonalityState):
        """Load personality state from database"""

        self.enhanced_personality.creativity = state.creativity_level
        self.enhanced_personality.learning_momentum = state.learning_momentum

        # Update risk controller
        risk_controller = self.enhanced_personality.risk_controller
        risk_controller.confidence_profile.base_confidence = state.base_confidence
        risk_controller.confidence_profile.learning_modifier = state.learning_modifier
        risk_controller.confidence_profile.failure_resilience = state.failure_resilience
        risk_controller.confidence_profile.success_amplification = state.success_amplification
        risk_controller.confidence_profile.impossible_attraction = state.impossible_attraction

        # Load history
        if state.success_history:
            # Keep recent 50
            risk_controller.success_history = state.success_history[-50:]
        if state.failure_history:
            risk_controller.failure_history = state.failure_history[-50:]

    def _is_building_action(self, decision: Dict[str, Any]) -> bool:
        """Check if a decision represents a building action"""

        action = decision.get('action', '')
        building_actions = [
            'attempt_impossible_build',
            'write_python_code',
            'propose_and_test_invention',
            'generate_creative_solution'
        ]

        return action in building_actions

    async def _memorize_building_interaction(self, situation_prompt: str, decision: Dict[str, Any], action_output: Any):
        """Memorize building-specific interactions"""

        # Create builder-specific memory summary
        builder_summary = f"""
        Building Task: {situation_prompt}
        
        Builder Decision: {decision.get('action', 'unknown')}
        Builder Confidence: {decision.get('confidence', 0.0):.2f}
        
        Outcome: {action_output}
        
        Builder Insights Applied: {decision.get('builder_insights', {}).get('creative_insights', [])}
        """

        # Store in knowledge base as builder experience
        try:
            await self.agi_system.knowledge_service.add_knowledge(
                content=builder_summary,
                source="intelligent_builder",
                category="building_experience"
            )
        except Exception as e:
            logger.warning(f"Failed to store building memory: {e}")

    async def _analyze_builder_performance(self) -> Dict[str, Any]:
        """Analyze recent builder performance"""

        # Query recent build attempts
        recent_builds = self.session.query(BuildAttempt).order_by(
            BuildAttempt.created_at.desc()
        ).limit(20).all()

        if not recent_builds:
            return {"message": "No recent builds to analyze"}

        successful_builds = [b for b in recent_builds if b.success]
        success_rate = len(successful_builds) / len(recent_builds)

        avg_confidence = sum(b.confidence_score for b in successful_builds) / \
            len(successful_builds) if successful_builds else 0.0
        avg_novelty = sum(b.novelty_score for b in successful_builds) / \
            len(successful_builds) if successful_builds else 0.0

        return {
            "recent_builds": len(recent_builds),
            "success_rate": success_rate,
            "average_confidence": avg_confidence,
            "average_novelty": avg_novelty,
            "impossible_challenges_attempted": len([b for b in recent_builds if b.difficulty == "impossible"]),
            "lessons_learned_count": sum(len(b.lessons_learned) for b in recent_builds)
        }

    async def _get_recent_failure_lessons(self) -> List[str]:
        """Get lessons from recent failures"""

        recent_failures = self.session.query(FailureAnalysis).order_by(
            FailureAnalysis.analysis_timestamp.desc()
        ).limit(5).all()

        all_lessons = []
        for failure in recent_failures:
            all_lessons.extend(failure.lessons_learned)

        # Remove duplicates while preserving order
        unique_lessons = []
        for lesson in all_lessons:
            if lesson not in unique_lessons:
                unique_lessons.append(lesson)

        return unique_lessons[:5]  # Return top 5 unique lessons

    async def _analyze_personality_evolution(self) -> Dict[str, Any]:
        """Analyze how the personality system has evolved"""

        current_state = self.session.query(PersonalityState).order_by(
            PersonalityState.updated_at.desc()
        ).first()

        if not current_state:
            return {"message": "No personality evolution data available"}

        return {
            "current_creativity": current_state.creativity_level,
            "current_risk_tolerance": current_state.risk_tolerance,
            "learning_momentum": current_state.learning_momentum,
            "builds_attempted": current_state.total_builds_attempted,
            "builds_succeeded": current_state.total_builds_succeeded,
            "impossible_challenges_completed": current_state.impossible_challenges_completed,
            "recent_success_rate": current_state.recent_success_rate
        }

    async def _get_common_failure_patterns(self) -> List[str]:
        """Get common failure patterns from history"""

        failures = self.session.query(FailureAnalysis).limit(50).all()

        all_root_causes = []
        for failure in failures:
            all_root_causes.extend(failure.root_causes)

        # Count frequency
        from collections import Counter
        cause_counts = Counter(all_root_causes)

        return [cause for cause, count in cause_counts.most_common(5)]

    async def _get_domain_specific_prevention(self, context: Dict[str, Any]) -> List[str]:
        """Get domain-specific prevention measures"""

        # This is a simplified implementation
        # In practice, this would analyze the context and provide domain-specific advice

        return [
            "Validate assumptions through small-scale experiments",
            "Use version control for all design iterations",
            "Document decision rationale for future reference"
        ]
