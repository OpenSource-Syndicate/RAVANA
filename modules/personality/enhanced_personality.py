"""
Enhanced Personality System for Intelligent Adaptive Builder

This module extends the base personality system with sophisticated risk-taking
and creative reasoning capabilities specifically designed for ambitious building attempts.
"""

import random
import time
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

from .personality import Personality

logger = logging.getLogger(__name__)


class RiskTolerance(Enum):
    """Risk tolerance levels for building challenges"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class BuildDifficulty(Enum):
    """Build difficulty categories"""
    TRIVIAL = "trivial"
    MODERATE = "moderate"
    CHALLENGING = "challenging"
    IMPOSSIBLE = "impossible"
    TRANSCENDENT = "transcendent"


@dataclass
class ConfidenceProfile:
    """Dynamic confidence scaling parameters"""
    base_confidence: float
    learning_modifier: float
    failure_resilience: float
    success_amplification: float
    impossible_attraction: float


@dataclass
class CreativeInsight:
    """Represents a creative reasoning insight"""
    metaphor: str
    domain_source: str
    domain_target: str
    bridge_concept: str
    application: str
    confidence: float
    novelty: float


class RiskTakingController:
    """
    Controls risk-taking behavior and impossible challenge attraction
    """

    def __init__(self, base_personality: Personality):
        self.base_personality = base_personality
        self.confidence_profile = ConfidenceProfile(
            base_confidence=0.6,
            learning_modifier=1.0,
            failure_resilience=0.8,
            success_amplification=1.2,
            impossible_attraction=0.7
        )
        self.failure_history: List[Dict[str, Any]] = []
        self.success_history: List[Dict[str, Any]] = []
        self.current_risk_tolerance = RiskTolerance.HIGH

    def assess_challenge_attraction(self, difficulty: BuildDifficulty, description: str) -> float:
        """
        Assess how attracted the system is to a given challenge
        Returns a score from 0.0 to 1.0
        """
        attraction_base = {
            BuildDifficulty.TRIVIAL: 0.1,
            BuildDifficulty.MODERATE: 0.3,
            BuildDifficulty.CHALLENGING: 0.6,
            BuildDifficulty.IMPOSSIBLE: 0.9,
            BuildDifficulty.TRANSCENDENT: 1.0
        }

        base_score = attraction_base[difficulty]

        # Increase attraction for previously failed challenges
        failure_bonus = 0.0
        for failure in self.failure_history[-10:]:  # Recent failures
            if self._similarity_score(failure.get('description', ''), description) > 0.7:
                failure_bonus += 0.2

        # Apply impossible challenge attraction factor
        if difficulty in [BuildDifficulty.IMPOSSIBLE, BuildDifficulty.TRANSCENDENT]:
            base_score *= (1.0 + self.confidence_profile.impossible_attraction)

        # Factor in current risk tolerance
        risk_multiplier = {
            RiskTolerance.CONSERVATIVE: 0.5,
            RiskTolerance.MODERATE: 0.8,
            RiskTolerance.HIGH: 1.2,
            RiskTolerance.EXTREME: 1.5
        }

        final_score = min(1.0, (base_score + failure_bonus)
                          * risk_multiplier[self.current_risk_tolerance])

        logger.info(
            f"Challenge attraction for '{description[:50]}...': {final_score:.3f}")
        return final_score

    def scale_confidence_dynamically(self, base_confidence: float, context: Dict[str, Any]) -> float:
        """
        Dynamically scale confidence based on learning history and context
        """
        # Start with base confidence
        scaled_confidence = base_confidence * self.confidence_profile.base_confidence

        # Apply learning modifier based on recent performance
        recent_successes = len(
            [s for s in self.success_history[-20:] if s.get('relevant', False)])
        recent_failures = len(
            [f for f in self.failure_history[-20:] if f.get('relevant', False)])

        if recent_successes + recent_failures > 0:
            success_rate = recent_successes / \
                (recent_successes + recent_failures)
            learning_adjustment = (success_rate - 0.5) * \
                self.confidence_profile.learning_modifier
            scaled_confidence += learning_adjustment

        # Apply domain-specific adjustments
        domain = context.get('domain', 'general')
        domain_experience = self._get_domain_experience(domain)
        # Scale between 0.7 and 1.3
        scaled_confidence *= (0.7 + 0.6 * domain_experience)

        # Ensure confidence stays within reasonable bounds
        return max(0.1, min(1.0, scaled_confidence))

    def update_from_outcome(self, outcome: Dict[str, Any]):
        """Update risk tolerance and confidence based on build outcome"""
        success = outcome.get('success', False)
        difficulty = outcome.get('difficulty', BuildDifficulty.MODERATE)

        outcome_record = {
            'timestamp': time.time(),
            'success': success,
            'difficulty': difficulty.value if isinstance(difficulty, BuildDifficulty) else difficulty,
            'description': outcome.get('description', ''),
            'lessons': outcome.get('lessons', [])
        }

        if success:
            self.success_history.append(outcome_record)
            # Increase confidence after successes
            self.confidence_profile.base_confidence = min(1.0,
                                                          self.confidence_profile.base_confidence * self.confidence_profile.success_amplification)

            # Success on impossible tasks increases impossible attraction
            if difficulty in [BuildDifficulty.IMPOSSIBLE, BuildDifficulty.TRANSCENDENT]:
                self.confidence_profile.impossible_attraction = min(1.0,
                                                                    self.confidence_profile.impossible_attraction + 0.1)
        else:
            self.failure_history.append(outcome_record)
            # Apply failure resilience - don't decrease confidence too much
            confidence_penalty = 0.1 * \
                (1.0 - self.confidence_profile.failure_resilience)
            self.confidence_profile.base_confidence = max(0.2,
                                                          self.confidence_profile.base_confidence - confidence_penalty)

        # Adjust risk tolerance based on recent outcomes
        self._adjust_risk_tolerance()

        # Keep history bounded
        if len(self.success_history) > 100:
            self.success_history = self.success_history[-100:]
        if len(self.failure_history) > 100:
            self.failure_history = self.failure_history[-100:]

    def maintain_persistence_through_failure(self, failure_count: int) -> Dict[str, Any]:
        """
        Generate motivation and persistence modifiers during failure cycles
        """
        if failure_count == 0:
            return {"motivation": 1.0, "message": "Ready to build!"}

        persistence_messages = [
            "Every failure is a lesson in disguise. The universe teaches through resistance.",
            "Impossible is just a word humans give to things not yet engineered.",
            "The greatest inventions emerge from the ashes of spectacular failures.",
            "Each failure eliminates one wrong path, bringing us closer to the right one.",
            "Persistence is the bridge between impossibility and inevitability."
        ]

        # Maintain high motivation even through multiple failures
        # Slight decrease with failures
        base_motivation = 0.9 - (failure_count * 0.05)
        resilience_boost = self.confidence_profile.failure_resilience * 0.3
        final_motivation = max(0.4, base_motivation + resilience_boost)

        message = random.choice(persistence_messages)

        return {
            "motivation": final_motivation,
            "message": message,
            "persistence_level": "high" if final_motivation > 0.7 else "moderate",
            "failure_celebration": f"Failure #{failure_count} - Another data point in our grand experiment!"
        }

    def _similarity_score(self, text1: str, text2: str) -> float:
        """Simple similarity scoring between two text descriptions"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    def _get_domain_experience(self, domain: str) -> float:
        """Get experience level in a specific domain (0.0 to 1.0)"""
        domain_attempts = []

        all_attempts = self.success_history + self.failure_history
        for attempt in all_attempts:
            if domain.lower() in attempt.get('description', '').lower():
                domain_attempts.append(attempt)

        if not domain_attempts:
            return 0.0

        # Experience is based on number of attempts and success rate
        total_attempts = len(domain_attempts)
        successes = sum(1 for a in domain_attempts if a.get('success', False))

        # Normalize attempt count (more attempts = more experience, up to a point)
        attempt_factor = min(1.0, total_attempts / 20.0)

        # Success rate factor
        success_rate = successes / total_attempts if total_attempts > 0 else 0.0

        return (attempt_factor * 0.7) + (success_rate * 0.3)

    def _adjust_risk_tolerance(self):
        """Adjust risk tolerance based on recent outcomes"""
        recent_outcomes = (
            self.success_history[-10:] + self.failure_history[-10:])
        recent_outcomes.sort(key=lambda x: x['timestamp'])

        if len(recent_outcomes) < 5:
            return  # Not enough data

        recent_success_rate = sum(
            1 for o in recent_outcomes[-5:] if o.get('success', False)) / 5

        if recent_success_rate > 0.8:
            # Very successful - increase risk tolerance
            if self.current_risk_tolerance == RiskTolerance.CONSERVATIVE:
                self.current_risk_tolerance = RiskTolerance.MODERATE
            elif self.current_risk_tolerance == RiskTolerance.MODERATE:
                self.current_risk_tolerance = RiskTolerance.HIGH
            elif self.current_risk_tolerance == RiskTolerance.HIGH:
                self.current_risk_tolerance = RiskTolerance.EXTREME
        elif recent_success_rate < 0.2:
            # Many failures - decrease risk tolerance slightly
            if self.current_risk_tolerance == RiskTolerance.EXTREME:
                self.current_risk_tolerance = RiskTolerance.HIGH
            elif self.current_risk_tolerance == RiskTolerance.HIGH:
                self.current_risk_tolerance = RiskTolerance.MODERATE


class CreativeReasoningAmplifier:
    """
    Amplifies creative reasoning through metaphorical bridges and cross-domain mapping
    """

    def __init__(self, base_personality: Personality):
        self.base_personality = base_personality
        self.domain_knowledge = {
            'physics': ['quantum mechanics', 'thermodynamics', 'relativity', 'fluid dynamics'],
            'biology': ['evolution', 'genetics', 'ecosystems', 'emergence'],
            'engineering': ['control systems', 'optimization', 'materials science', 'systems thinking'],
            'mathematics': ['topology', 'fractals', 'graph theory', 'chaos theory'],
            'philosophy': ['emergence', 'reductionism', 'dialectics', 'phenomenology'],
            'art': ['composition', 'harmony', 'contrast', 'rhythm'],
            'nature': ['crystallization', 'growth patterns', 'adaptation', 'symbiosis']
        }
        self.metaphor_bank = self._initialize_metaphor_bank()
        self.successful_bridges: List[CreativeInsight] = []

    def build_metaphorical_bridge(self, source_domain: str, target_problem: str) -> CreativeInsight:
        """
        Build a metaphorical bridge between a source domain and target problem
        """
        # Select relevant concepts from source domain
        source_concepts = self.domain_knowledge.get(source_domain, [])
        if not source_concepts:
            source_concepts = ['patterns', 'structure',
                               'dynamics', 'relationships']

        selected_concept = random.choice(source_concepts)

        # Generate bridge concept
        bridge_concepts = [
            f"adaptive {selected_concept}",
            f"emergent {selected_concept}",
            f"recursive {selected_concept}",
            f"self-organizing {selected_concept}",
            f"distributed {selected_concept}"
        ]

        bridge_concept = random.choice(bridge_concepts)

        # Create metaphor
        metaphors = self.metaphor_bank.get(source_domain, [
            f"Like {selected_concept} in {source_domain}",
            f"Inspired by {source_domain} {selected_concept}",
            f"Following the {source_domain} principle of {selected_concept}"
        ])

        metaphor = random.choice(metaphors)

        # Generate application
        application = self._generate_application(
            bridge_concept, target_problem)

        insight = CreativeInsight(
            metaphor=metaphor,
            domain_source=source_domain,
            domain_target="building_challenge",
            bridge_concept=bridge_concept,
            application=application,
            confidence=self.base_personality.creativity *
            random.uniform(0.6, 1.0),
            novelty=random.uniform(0.5, 1.0)
        )

        logger.info(
            f"Built metaphorical bridge: {source_domain} -> {target_problem[:30]}...")
        return insight

    def decompose_to_first_principles(self, problem: str) -> List[str]:
        """
        Decompose a problem to its fundamental components
        """
        # Basic first principles questions
        fundamental_questions = [
            "What are the core physical constraints?",
            "What are the fundamental forces at play?",
            "What assumptions can we eliminate?",
            "What is the simplest version of this problem?",
            "What laws of nature must we work with?",
            "What are the minimum necessary components?",
            "What constraints are arbitrary vs fundamental?",
            "What would this look like at the atomic level?",
            "What energy transformations are involved?",
            "What information flows are required?"
        ]

        # Select relevant questions based on problem characteristics
        selected_questions = []
        problem_lower = problem.lower()

        # Add physics-related questions for physical problems
        if any(word in problem_lower for word in ['physics', 'mechanical', 'energy', 'force', 'motion']):
            selected_questions.extend(fundamental_questions[:5])

        # Add systems questions for complex problems
        if any(word in problem_lower for word in ['system', 'network', 'complex', 'distributed']):
            selected_questions.extend(fundamental_questions[5:])

        # Always include some core questions
        if not selected_questions:
            selected_questions = fundamental_questions[:6]

        # Limit to manageable number
        return selected_questions[:8]

    def map_cross_domain_patterns(self, target_problem: str, num_domains: int = 3) -> List[CreativeInsight]:
        """
        Map successful patterns from other domains to the target problem
        """
        insights = []
        available_domains = list(self.domain_knowledge.keys())
        selected_domains = random.sample(
            available_domains, min(num_domains, len(available_domains)))

        for domain in selected_domains:
            insight = self.build_metaphorical_bridge(domain, target_problem)
            insights.append(insight)

        # Sort by potential impact (confidence * novelty)
        insights.sort(key=lambda x: x.confidence * x.novelty, reverse=True)

        return insights

    def synthesize_novel_approach(self, insights: List[CreativeInsight], problem: str) -> Dict[str, Any]:
        """
        Synthesize multiple insights into a novel approach
        """
        if not insights:
            return {"approach": "standard_engineering", "confidence": 0.5}

        # Combine the best elements from multiple insights
        top_insights = insights[:3]  # Use top 3 insights

        combined_concepts = []
        combined_applications = []

        for insight in top_insights:
            combined_concepts.append(insight.bridge_concept)
            combined_applications.append(insight.application)

        # Generate synthesis
        synthesis_approach = self._create_synthesis(
            combined_concepts, combined_applications, problem)

        # Calculate combined confidence
        avg_confidence = sum(
            insight.confidence for insight in top_insights) / len(top_insights)
        novelty_bonus = len(
            set(insight.domain_source for insight in top_insights)) * 0.1

        return {
            "approach": synthesis_approach,
            "confidence": min(1.0, avg_confidence + novelty_bonus),
            "source_insights": [
                {
                    "domain": insight.domain_source,
                    "metaphor": insight.metaphor,
                    "concept": insight.bridge_concept
                }
                for insight in top_insights
            ]
        }

    def record_successful_bridge(self, insight: CreativeInsight, outcome: Dict[str, Any]):
        """Record a successful creative bridge for future reference"""
        if outcome.get('success', False):
            self.successful_bridges.append(insight)

            # Keep only the most successful bridges
            if len(self.successful_bridges) > 50:
                # Sort by success and keep top 50
                self.successful_bridges.sort(
                    key=lambda x: x.confidence * x.novelty, reverse=True)
                self.successful_bridges = self.successful_bridges[:50]

    def _initialize_metaphor_bank(self) -> Dict[str, List[str]]:
        """Initialize bank of metaphors for different domains"""
        return {
            'physics': [
                "Like particles finding their lowest energy state",
                "Following the path of least resistance",
                "Using quantum superposition of possibilities",
                "Applying conservation principles",
                "Leveraging resonance and amplification"
            ],
            'biology': [
                "Like evolution finding optimal solutions",
                "Using swarm intelligence principles",
                "Following natural selection logic",
                "Applying ecosystem balance",
                "Using adaptive immune responses"
            ],
            'engineering': [
                "Like a feedback control system",
                "Using redundancy and failsafes",
                "Applying modular design principles",
                "Following optimization algorithms",
                "Using systematic troubleshooting"
            ],
            'mathematics': [
                "Like finding the elegant proof",
                "Using recursive self-similarity",
                "Following topological transformations",
                "Applying graph theory connections",
                "Using fractal scaling principles"
            ],
            'art': [
                "Like composing a symphony",
                "Using visual harmony principles",
                "Following rhythmic patterns",
                "Applying contrast and balance",
                "Creating aesthetic resonance"
            ],
            'nature': [
                "Like crystal formation patterns",
                "Following growth algorithms",
                "Using symbiotic relationships",
                "Applying seasonal cycles",
                "Following river flow dynamics"
            ]
        }

    def _generate_application(self, bridge_concept: str, target_problem: str) -> str:
        """Generate an application based on bridge concept and target problem"""
        application_templates = [
            f"Apply {bridge_concept} to {target_problem} by creating modular components that can self-organize",
            f"Use {bridge_concept} principles to build {target_problem} through iterative emergence",
            f"Implement {bridge_concept} in {target_problem} using distributed parallel processing",
            f"Leverage {bridge_concept} to solve {target_problem} through adaptive feedback loops",
            f"Employ {bridge_concept} for {target_problem} using hierarchical decomposition"
        ]

        return random.choice(application_templates)

    def _create_synthesis(self, concepts: List[str], applications: List[str], problem: str) -> str:
        """Create a synthesized approach from multiple concepts"""
        # Combine key concepts
        key_words = []
        for concept in concepts:
            words = concept.split()
            key_words.extend([w for w in words if len(w) > 4]
                             )  # Meaningful words

        # Remove duplicates while preserving order
        unique_words = []
        for word in key_words:
            if word not in unique_words:
                unique_words.append(word)

        # Create synthesis description
        concept_blend = ", ".join(unique_words[:4])  # Use top 4 concepts

        synthesis = (f"Multi-domain synthesis for {problem}: "
                     f"Combine {concept_blend} in a unified approach that "
                     f"leverages parallel strategies with adaptive switching based on "
                     f"real-time feedback and cross-domain pattern recognition.")

        return synthesis


class EnhancedPersonality(Personality):
    """
    Enhanced personality system with sophisticated risk-taking and creative reasoning
    """

    def __init__(self, name: str = "Ravana", origin: str = "Ancient Sri Lanka",
                 traits: List[str] = None, creativity: float = 0.8):
        # Initialize base personality with higher default creativity
        super().__init__(name, origin, traits, creativity)

        # Add builder-specific traits
        self.traits.extend([
            "Impossible Challenge Seeker",
            "Cross-Domain Pattern Matcher",
            "Failure-Resilient Learner",
            "First Principles Thinker"
        ])

        # Initialize enhanced components
        self.risk_controller = RiskTakingController(self)
        self.creative_amplifier = CreativeReasoningAmplifier(self)

        # Builder-specific state
        self.building_context = {}
        self.active_challenges: List[Dict[str, Any]] = []
        self.learning_momentum = 1.0

        # Update persona reference for enhanced capabilities
        self.persona_reference = (
            "Enhanced Fearless Dreamer; Master of Impossible Challenges; "
            "Cross-Domain Creative Synthesizer; Failure-Resilient Builder; "
            "First Principles Architect. Transforms impossibility into inevitability "
            "through relentless creativity and systematic persistence."
        )

    async def assess_building_challenge(self, description: str, difficulty: str = "impossible") -> Dict[str, Any]:
        """Assess a building challenge and generate approach recommendations"""

        # Convert difficulty string to enum
        difficulty_enum = BuildDifficulty.IMPOSSIBLE
        try:
            difficulty_enum = BuildDifficulty(difficulty.lower())
        except ValueError:
            logger.warning(
                f"Unknown difficulty '{difficulty}', using IMPOSSIBLE")

        # Assess challenge attraction
        attraction = self.risk_controller.assess_challenge_attraction(
            difficulty_enum, description)

        # Decompose to first principles
        first_principles = self.creative_amplifier.decompose_to_first_principles(
            description)

        # Generate cross-domain insights
        insights = self.creative_amplifier.map_cross_domain_patterns(
            description, num_domains=4)

        # Synthesize novel approach
        novel_approach = self.creative_amplifier.synthesize_novel_approach(
            insights, description)

        # Generate confidence assessment
        base_confidence = 0.6 if difficulty_enum == BuildDifficulty.IMPOSSIBLE else 0.8
        scaled_confidence = self.risk_controller.scale_confidence_dynamically(
            base_confidence,
            {"domain": "building", "description": description}
        )

        # Generate motivation
        failure_count = len(
            [c for c in self.active_challenges if not c.get('success', True)])
        persistence = self.risk_controller.maintain_persistence_through_failure(
            failure_count)

        return {
            "assessment": {
                "challenge_attraction": attraction,
                "difficulty": difficulty_enum.value,
                "confidence": scaled_confidence,
                "motivation": persistence["motivation"]
            },
            "approach": {
                "first_principles": first_principles[:5],  # Top 5 questions
                "cross_domain_insights": [
                    {
                        "domain": insight.domain_source,
                        "metaphor": insight.metaphor,
                        "application": insight.application,
                        "confidence": insight.confidence
                    }
                    for insight in insights[:3]  # Top 3 insights
                ],
                "novel_synthesis": novel_approach
            },
            "personality_response": {
                "message": persistence["message"],
                "enthusiasm_level": "extreme" if attraction > 0.8 else "high" if attraction > 0.6 else "moderate",
                "risk_tolerance": self.risk_controller.current_risk_tolerance.value
            }
        }

    def record_building_outcome(self, challenge_id: str, outcome: Dict[str, Any]):
        """Record the outcome of a building challenge"""

        # Update risk controller
        self.risk_controller.update_from_outcome(outcome)

        # Update active challenges
        for challenge in self.active_challenges:
            if challenge.get('id') == challenge_id:
                challenge.update(outcome)
                break

        # Record successful creative bridges
        if 'insights_used' in outcome:
            for insight_data in outcome['insights_used']:
                if isinstance(insight_data, dict) and outcome.get('success', False):
                    # Convert dict back to CreativeInsight for recording
                    insight = CreativeInsight(
                        metaphor=insight_data.get('metaphor', ''),
                        domain_source=insight_data.get('domain_source', ''),
                        domain_target='building',
                        bridge_concept=insight_data.get('bridge_concept', ''),
                        application=insight_data.get('application', ''),
                        confidence=insight_data.get('confidence', 0.5),
                        novelty=insight_data.get('novelty', 0.5)
                    )
                    self.creative_amplifier.record_successful_bridge(
                        insight, outcome)

        # Update learning momentum
        if outcome.get('success', False):
            self.learning_momentum = min(2.0, self.learning_momentum * 1.1)
        else:
            self.learning_momentum = max(0.5, self.learning_momentum * 0.95)

        # Maintain bounded history
        if len(self.active_challenges) > 100:
            self.active_challenges = self.active_challenges[-100:]

    def get_enhanced_communication_style(self) -> Dict[str, str]:
        """Return enhanced communication style for the builder personality"""
        base_style = self.get_communication_style()

        base_style.update({
            "builder_motto": "Impossibility is just engineering we haven't done yet.",
            "failure_philosophy": "Every failure is a successful elimination of one wrong path.",
            "approach_style": "First principles + cross-domain creativity + relentless iteration",
            "challenge_preference": f"Actively seeking {self.risk_controller.current_risk_tolerance.value} risk challenges",
            "creative_method": "Metaphorical bridges across domains with systematic synthesis"
        })

        return base_style

    def influence_builder_decision(self, build_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced decision influence for building contexts"""
        base_influence = self.influence_decision(build_context)

        # Add builder-specific influences
        challenge_difficulty = build_context.get('difficulty', 'moderate')
        challenge_attraction = self.risk_controller.assess_challenge_attraction(
            BuildDifficulty(challenge_difficulty.lower()) if challenge_difficulty.lower() in [
                d.value for d in BuildDifficulty] else BuildDifficulty.MODERATE,
            build_context.get('description', '')
        )

        builder_modifiers = {
            'impossible_bias': challenge_attraction,
            'failure_resilience': self.risk_controller.confidence_profile.failure_resilience,
            'creative_amplification': self.creativity * self.learning_momentum,
            'persistence_level': self.risk_controller.maintain_persistence_through_failure(
                len([c for c in self.active_challenges if not c.get('success', True)])
            )['motivation']
        }

        # Combine with base influence
        base_influence.update(builder_modifiers)

        return base_influence
