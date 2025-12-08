"""
Enhanced emotional intelligence system with dynamic parameters that reflect human nature,
including emotional regulation, complex emotional processing, and adaptive responses.
"""
import logging
import json
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from core.config import Config
from .mood_processor import MoodProcessor

logger = logging.getLogger(__name__)


class EmotionalRegulationStrategy(Enum):
    """
    Types of emotional regulation strategies
    """
    COGNITIVE_REAPPRAISAL = "cognitive_reappraisal"
    ATTENTION_DEPLOYMENT = "attention_deployment"
    SITUATION_SELECTION = "situation_selection"
    SITUATION_MODIFICATION = "situation_modification"
    RESPONSE_MODULATION = "response_modulation"
    ACCEPTANCE = "acceptance"


@dataclass
class EmotionalState:
    """
    Represents the current emotional state with dynamic parameters
    """
    # Core emotional dimensions
    valence: float = 0.0  # Pleasantness-unpleasantness (-1 to 1)
    arousal: float = 0.5  # Activation level (0 to 1)
    dominance: float = 0.5  # Sense of control (0 to 1)
    
    # Emotional regulation parameters
    regulation_capacity: float = 0.6  # Available capacity for regulation (0-1)
    regulation_speed: float = 0.5  # Speed of regulation (0-1)
    recovery_rate: float = 0.4  # Rate of return to baseline (0-1)
    
    # Emotional complexity measures
    emotional_complexity: int = 1  # Number of emotions experienced simultaneously
    emotional_differentiation: float = 0.5  # Ability to distinguish emotions (0-1)
    
    # Stress and resilience factors
    stress_level: float = 0.3  # Current stress level (0-1)
    resilience: float = 0.6  # Ability to bounce back (0-1)
    emotional_flexibility: float = 0.7  # Ability to adapt emotional responses (0-1)


class EmotionalEvent:
    """Structure for logging emotional events with enhanced details"""

    def __init__(self, timestamp: datetime, mood_changes: Dict[str, float],
                 triggers: List[str], context: str, intensity: float, 
                 regulation_used: Optional[EmotionalRegulationStrategy] = None,
                 outcome_valence: float = 0.0):
        self.timestamp = timestamp
        self.mood_changes = mood_changes
        self.triggers = triggers
        self.context = context
        self.intensity = intensity
        self.regulation_used = regulation_used
        self.outcome_valence = outcome_valence  # Result of emotional response (-1 to 1)


class EnhancedEmotionalIntelligence:
    """
    Enhanced emotional intelligence system with dynamic parameters
    """
    def __init__(self, config_path='modules/emotional_intellegence/config.json',
                 persona_path='modules/emotional_intellegence/persona.json'):
        # Initialize expanded mood dimensions
        self._initialize_mood_dimensions()

        # Initialize mood vector with all emotions
        self.mood_vector: Dict[str, float] = {
            mood: 0.0 for mood in self.ALL_MOODS}

        # Initialize emotional momentum tracking
        self.mood_momentum: Dict[str, float] = {
            mood: 0.0 for mood in self.ALL_MOODS}

        # Initialize emotional event history
        self.emotional_events: List[EmotionalEvent] = []

        self.last_action_result: Optional[dict] = None
        self._load_config(config_path)
        self._load_personas(persona_path)
        self.set_persona(self.personas.get("default_persona", "Balanced"))
        self.mood_processor = MoodProcessor(self)

        # Initialize enhanced mood dynamics parameters
        self.momentum_factor = self.config.get(
            "mood_dynamics", {}).get("momentum_factor", 0.7)
        self.damping_factor = self.config.get(
            "mood_dynamics", {}).get("damping_factor", 0.9)
        
        # Enhanced emotional state tracking
        self.emotional_state = EmotionalState()
        self.emotional_inertia = 0.1  # Resistance to emotional change
        self.baseline_stability = 0.7  # How stable baseline emotions are
        
        # Emotional regulation system
        self.regulation_strategies = [
            EmotionalRegulationStrategy.COGNITIVE_REAPPRAISAL,
            EmotionalRegulationStrategy.ATTENTION_DEPLOYMENT,
            EmotionalRegulationStrategy.SITUATION_SELECTION,
            EmotionalRegulationStrategy.SITUATION_MODIFICATION,
            EmotionalRegulationStrategy.RESPONSE_MODULATION,
            EmotionalRegulationStrategy.ACCEPTANCE
        ]
        
        # Emotional memory and learning
        self.emotional_memory_strength = 0.8
        self.mood_congruent_memory = True
        self.emotional_learning_rate = 0.1
        
        # Circadian and temporal factors
        self.circadian_influence = self._get_circadian_influence()
        self.seasonal_affect = 0.0  # Seasonal emotional variations
        
        # Social and cultural factors
        self.social_influence_sensitivity = 0.5
        self.cultural_emotion_norms = {}
        
        # Individual differences
        self.neuroticism_level = 0.3
        self.extroversion_level = 0.5
        self.emotional_stability_baseline = 0.7

    def _initialize_mood_dimensions(self):
        """Initialize expanded mood dimensions with primary, secondary, and complex emotions"""
        # Primary emotions from config
        config = Config()
        primary_emotions = config.POSITIVE_MOODS + config.NEGATIVE_MOODS

        # Extended primary emotions from enhanced config
        emotion_config = {
            "joy_based": ["Confident", "Excited", "Inspired", "Satisfied", "Joyful", "Elated"],
            "interest_based": ["Curious", "Reflective", "Intrigued", "Engaged", "Interested", "Fascinated"],
            "sadness_based": ["Disappointed", "Bored", "Low Energy", "Melancholic", "Saddened", "Gloomy"],
            "anger_based": ["Frustrated", "Irritated", "Stuck", "Resentful", "Angry", "Enraged"],
            "fear_based": ["Anxious", "Apprehensive", "Cautious", "Suspicious", "Fearful", "Terrified"],
            "surprise_based": ["Astonished", "Bewildered", "Amazed", "Shocked", "Surprised", "Awe"],
            "love_based": ["Loving", "Affectionate", "Tender", "Attached", "Caring", "Compassionate"],
            "disgust_based": ["Disgusted", "Revolted", "Loathing", "Repulsed", "Horrified", "Appalled"]
        }

        # Flatten all primary emotions
        extended_primary = []
        for emotion_category in emotion_config.values():
            extended_primary.extend(emotion_category)

        # Secondary emotions
        secondary_emotions = [
            "Hopeful", "Grateful", "Proud", "Guilty", "Lonely", "Nostalgic",
            "Embarrassed", "Jealous", "Relieved", "Surprised", "Envious", "Peaceful",
            "Compassionate", "Confused", "Optimistic", "Pessimistic", "Content", 
            "Apathetic", "Vigilant", "Determined", "Playful", "Serene", "Bewildered"
        ]

        # Complex emotional blends
        complex_emotions = [
            "Bittersweet", "Nostalgic contentment", "Melancholic curiosity",
            "Anxious anticipation", "Contemplative serenity", "Frustrated determination",
            "Cautious optimism", "Melancholic joy", "Tender sadness",
            "Reflective excitement", "Pragmatic hope", "Wistful gratitude",
            "Wonderful concern", "Peaceful vigilance", "Determined patience",
            "Adaptive resilience", "Thoughtful caution", "Curious wonder",
            "Balanced contemplation", "Thoughtful determination", "Mindful serenity",
            "Cautious wonder", "Contemplative focus", "Thoughtful analysis",
            "Balanced mindfulness", "Serene awareness", "Peaceful alertness",
            "Gentle determination", "Mindful curiosity", "Thoughtful vigilance"
        ]

        # Combine all emotions
        self.ALL_MOODS = list(
            set(primary_emotions + extended_primary + secondary_emotions + complex_emotions))
        self.PRIMARY_MOODS = extended_primary
        self.SECONDARY_MOODS = secondary_emotions
        self.COMPLEX_EMOTIONS = complex_emotions
        
        # Initialize emotion categories mapping
        self.emotion_categories = {}
        for category, emotions in emotion_config.items():
            for emotion in emotions:
                if emotion in self.ALL_MOODS:
                    self.emotion_categories[emotion] = category

    def _get_circadian_influence(self) -> float:
        """Get current circadian rhythm influence on emotions"""
        current_hour = datetime.now().hour
        
        # Simplified circadian model: lowest around 3-4 AM, highest around noon
        # This creates a sinusoidal pattern
        hour_angle = (current_hour / 24.0) * 2 * 3.14159
        circadian_value = (math.sin(hour_angle - 3.14159/2) + 1) / 2  # Normalize to 0-1
        
        return circadian_value

    def update_emotional_state(self, context: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Update emotional state based on context and internal dynamics
        """
        context = context or {}
        
        # Update circadian influence
        self.circadian_influence = self._get_circadian_influence()
        
        # Update stress level based on context
        if 'stress_indicators' in context:
            stress_level = sum(context['stress_indicators']) / len(context['stress_indicators'])
            self.emotional_state.stress_level = max(0.0, min(1.0, stress_level))
        
        # Update emotional regulation capacity based on stress and fatigue
        fatigue_factor = context.get('cognitive_fatigue', 0.0)
        self.emotional_state.regulation_capacity = max(0.1, 0.7 - self.emotional_state.stress_level - fatigue_factor)
        
        # Update recovery rate based on resilience and current state
        base_recovery = 0.3
        resilience_boost = self.emotional_state.resilience * 0.2
        stress_reduction = self.emotional_state.stress_level * -0.3
        self.emotional_state.recovery_rate = max(0.05, base_recovery + resilience_boost + stress_reduction)
        
        # Update emotional complexity based on cognitive load
        cognitive_load = context.get('cognitive_load', 0.3)
        self.emotional_state.emotional_complexity = max(1, min(5, int(2 + cognitive_load * 3)))
        
        # Apply temporal decay to emotions
        self._apply_emotional_decay()
        
        return {
            'valence': self.emotional_state.valence,
            'arousal': self.emotional_state.arousal,
            'dominance': self.emotional_state.dominance,
            'stress_level': self.emotional_state.stress_level,
            'regulation_capacity': self.emotional_state.regulation_capacity,
            'recovery_rate': self.emotional_state.recovery_rate
        }

    def _apply_emotional_decay(self):
        """Apply temporal decay to emotional states"""
        decay_rate = 0.01  # Base decay rate
        
        # Adjust for circadian influences
        decay_rate *= (1 + (self.circadian_influence - 0.5) * 0.2)  # Affect up to 20%
        
        # Adjust for stress (high stress reduces decay rate - emotions linger)
        stress_adjustment = max(0.5, 1.0 - self.emotional_state.stress_level * 0.3)
        effective_decay = decay_rate * stress_adjustment
        
        # Apply decay to mood vector
        for mood, value in self.mood_vector.items():
            # Apply decay toward baseline, but don't let positive emotions decay as much
            if value > 0.1:
                new_value = max(0.1, value - effective_decay)
            else:
                new_value = max(0.0, value + effective_decay)  # Allow for recovery
            self.mood_vector[mood] = new_value

    def process_action_result_enhanced(self, action_result: Dict[str, Any], 
                                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced emotional processing of action results with dynamic parameters
        """
        # Store previous mood state for learning
        previous_mood = dict(self.mood_vector)
        
        # Determine emotional impact based on action result
        emotional_impact = self._assess_emotional_impact(action_result)
        
        # Apply emotional impact with regulation
        regulated_impact = self._apply_emotional_regulation(emotional_impact, context or {})
        
        # Update mood vector with regulated impact
        for mood, impact in regulated_impact.items():
            if mood in self.mood_vector:
                # Apply impact with momentum and damping
                momentum_effect = self.mood_momentum.get(mood, 0.0) * self.momentum_factor
                new_value = max(0.0, min(1.0, 
                    self.mood_vector[mood] + impact + momentum_effect))
                
                # Apply damping to prevent oscillations
                self.mood_vector[mood] = new_value * self.damping_factor
                
                # Update momentum
                self.mood_momentum[mood] = self.mood_momentum[mood] * 0.8 + impact * 0.2
        
        # Update emotional state
        self.update_emotional_state(context)
        
        # Log emotional event
        mood_changes = {
            mood: self.mood_vector[mood] - previous_mood[mood]
            for mood in self.mood_vector
            if abs(self.mood_vector[mood] - previous_mood[mood]) > 0.01
        }
        
        if mood_changes:
            triggers = self._identify_triggers(action_result)
            context_str = json.dumps(action_result)[:200]  # Truncate for brevity
            
            event = EmotionalEvent(
                timestamp=datetime.now(),
                mood_changes=mood_changes,
                triggers=triggers,
                context=context_str,
                intensity=self._calculate_emotional_intensity(mood_changes),
                regulation_used=self._select_appropriate_regulation(context or {}),
                outcome_valence=emotional_impact.get('valence', 0.0)
            )
            
            self.emotional_events.append(event)
            
            # Keep only recent events (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.emotional_events = [
                event for event in self.emotional_events
                if event.timestamp > cutoff_time
            ]
        
        # Learn from emotional outcomes
        self._learn_from_emotional_outcomes(action_result, previous_mood)
        
        return {
            'emotional_impact': emotional_impact,
            'regulated_impact': regulated_impact,
            'mood_changes': mood_changes,
            'emotional_state': self.emotional_state.__dict__.copy(),
            'regulation_used': self._select_appropriate_regulation(context or {})
        }

    def _assess_emotional_impact(self, action_result: Dict[str, Any]) -> Dict[str, float]:
        """Assess the emotional impact of an action result"""
        impact = {}
        
        # Determine outcome valence (positive/negative)
        outcome_valence = self._assess_outcome_valence(action_result)
        
        # Determine outcome magnitude
        outcome_magnitude = self._assess_outcome_magnitude(action_result)
        
        # Map general emotional response based on valence and magnitude
        base_emotion = "Satisfied" if outcome_valence > 0.1 else "Disappointed" if outcome_valence < -0.1 else "Neutral"
        
        # Apply impact based on valence and magnitude
        if abs(outcome_valence) > 0.3:  # Significant outcome
            impact[base_emotion] = outcome_magnitude * outcome_valence
            
            # Add related emotions based on the primary emotion
            related_emotions = self._get_related_emotions(base_emotion)
            for related_emotion in related_emotions:
                if related_emotion in self.ALL_MOODS:
                    impact[related_emotion] = impact[base_emotion] * 0.6  # Reduced intensity
        
        # Consider surprise factor
        if action_result.get('unexpected', False):
            surprise_emotions = ["Surprised", "Astonished", "Bewildered"]
            for emotion in surprise_emotions:
                if emotion in self.ALL_MOODS:
                    impact[emotion] = min(0.5, outcome_magnitude * 0.5)
        
        return impact

    def _assess_outcome_valence(self, action_result: Dict[str, Any]) -> float:
        """Assess whether the outcome is positive, negative, or neutral (-1 to 1)"""
        # Check for explicit success/failure indicators
        if 'success' in action_result:
            return 1.0 if action_result['success'] else -1.0
        
        # Check for outcome quality indicators
        if 'outcome_quality' in action_result:
            # Assume outcome_quality is on a scale, normalize to -1 to 1
            quality = action_result['outcome_quality']
            if isinstance(quality, (int, float)):
                return max(-1.0, min(1.0, quality / 5.0))  # Assuming scale of 0-10 becomes -1 to 1
        
        # Check for performance indicators
        if 'performance' in action_result:
            perf = action_result['performance']
            if isinstance(perf, (int, float)):
                # Normalize performance metric (assuming 0-1 scale)
                return max(-1.0, min(1.0, (perf - 0.5) * 2))  # 0.5 becomes 0, 0 becomes -1, 1 becomes 1
        
        # Default: neutral outcome
        return 0.0

    def _assess_outcome_magnitude(self, action_result: Dict[str, Any]) -> float:
        """Assess the magnitude of the outcome (0 to 1)"""
        # Check for explicit importance indicators
        if 'importance' in action_result:
            importance = action_result['importance']
            if isinstance(importance, (int, float)):
                return max(0.0, min(1.0, importance))
        
        # Check for impact indicators
        if 'impact_level' in action_result:
            impact = action_result['impact_level']
            if isinstance(impact, (int, float)):
                return max(0.0, min(1.0, impact))
        
        # Default: medium magnitude
        return 0.5

    def _apply_emotional_regulation(self, emotional_impact: Dict[str, float], 
                                  context: Dict[str, Any]) -> Dict[str, float]:
        """Apply emotional regulation to emotional impact"""
        regulated_impact = {}
        
        # Determine which regulation strategy to use
        strategy = self._select_appropriate_regulation(context)
        
        # Apply regulation based on selected strategy
        for mood, impact in emotional_impact.items():
            if strategy == EmotionalRegulationStrategy.COGNITIVE_REAPPRAISAL:
                # Cognitive reappraisal reduces the emotional impact
                regulated_impact[mood] = impact * (1 - self.emotional_state.regulation_capacity * 0.3)
            elif strategy == EmotionalRegulationStrategy.ACCEPTANCE:
                # Acceptance preserves the emotional impact but reduces reactivity
                regulated_impact[mood] = impact
                # Add a sense of peace or acceptance
                if abs(impact) > 0.3 and 'Accepting' in self.ALL_MOODS:
                    regulated_impact['Accepting'] = min(0.3, self.emotional_state.regulation_capacity * 0.4)
            elif strategy == EmotionalRegulationStrategy.ATTENTION_DEPLOYMENT:
                # Attention deployment changes focus but doesn't diminish impact
                attention_factor = 1 - self.emotional_state.regulation_capacity * 0.2
                regulated_impact[mood] = impact * attention_factor
            else:
                # Other strategies have different effects
                regulated_impact[mood] = impact * (1 - self.emotional_state.regulation_capacity * 0.1)
        
        # Apply regulation speed factor
        time_factor = min(1.0, self.emotional_state.regulation_speed * 2)
        for mood in regulated_impact:
            regulated_impact[mood] *= time_factor
        
        return regulated_impact

    def _select_appropriate_regulation(self, context: Dict[str, Any]) -> EmotionalRegulationStrategy:
        """Select the most appropriate regulation strategy based on context"""
        # Factors that influence strategy selection
        stress_level = self.emotional_state.stress_level
        cognitive_load = context.get('cognitive_load', 0.3)
        emotional_intensity = context.get('emotional_intensity', 0.5)
        
        # High stress and high load favor acceptance
        if stress_level > 0.7 and cognitive_load > 0.7:
            return EmotionalRegulationStrategy.ACCEPTANCE
        
        # High emotional intensity with moderate load favors cognitive reappraisal
        if emotional_intensity > 0.6 and cognitive_load < 0.6:
            return EmotionalRegulationStrategy.COGNITIVE_REAPPRAISAL
        
        # Moderate load and moderate stress favor attention deployment
        if 0.3 < cognitive_load < 0.7 and stress_level < 0.5:
            return EmotionalRegulationStrategy.ATTENTION_DEPLOYMENT
        
        # Default to cognitive reappraisal when regulation capacity is high
        if self.emotional_state.regulation_capacity > 0.6:
            return EmotionalRegulationStrategy.COGNITIVE_REAPPRAISAL
        
        # Otherwise, use acceptance
        return EmotionalRegulationStrategy.ACCEPTANCE

    def _get_related_emotions(self, primary_emotion: str) -> List[str]:
        """Get emotions related to a primary emotion"""
        emotion_relationships = {
            "Confident": ["Secure", "Assured", "Optimistic", "Determined"],
            "Excited": ["Energetic", "Enthusiastic", "Playful", "Joyful"],
            "Inspired": ["Creative", "Motivated", "Engaged", "Inquisitive"],
            "Satisfied": ["Content", "Pleased", "Grateful", "Peaceful"],
            "Disappointed": ["Sad", "Regretful", "Dissatisfied", "Dejected"],
            "Frustrated": ["Irritated", "Annoyed", "Agitated", "Tense"],
            "Anxious": ["Worried", "Nervous", "Apprehensive", "Tense"],
            "Curious": ["Inquisitive", "Interested", "Engaged", "Exploratory"],
            "Reflective": ["Contemplative", "Thoughtful", "Pensive", "Mindful"]
        }
        
        return emotion_relationships.get(primary_emotion, [])

    def _identify_triggers(self, action_result: Dict[str, Any]) -> List[str]:
        """Identify emotional triggers from action result"""
        triggers = []
        
        # Look for specific trigger words in the result
        result_str = json.dumps(action_result).lower()
        
        trigger_categories = {
            'achievement': ['success', 'achieve', 'complete', 'finish', 'win', 'accomplish'],
            'failure': ['fail', 'error', 'problem', 'issue', 'wrong', 'incorrect'],
            'social': ['people', 'team', 'colleague', 'user', 'human', 'social'],
            'learning': ['learn', 'discover', 'new', 'insight', 'understand', 'knowledge'],
            'challenge': ['difficult', 'hard', 'complex', 'challenge', 'struggle', 'effort']
        }
        
        for category, keywords in trigger_categories.items():
            if any(keyword in result_str for keyword in keywords):
                triggers.append(category)
        
        return triggers

    def _calculate_emotional_intensity(self, mood_changes: Dict[str, float]) -> float:
        """Calculate overall emotional intensity from mood changes"""
        if not mood_changes:
            return 0.0
        
        # Calculate mean absolute change
        abs_changes = [abs(change) for change in mood_changes.values()]
        return sum(abs_changes) / len(abs_changes)

    def _learn_from_emotional_outcomes(self, action_result: Dict[str, Any], 
                                     previous_mood: Dict[str, float]):
        """Learn from emotional outcomes to improve emotional regulation"""
        # This method would implement learning algorithms to improve emotional responses
        # For now, we'll update some parameters based on outcomes
        
        # Assess whether the emotional response was appropriate to the outcome
        outcome_valence = self._assess_outcome_valence(action_result)
        avg_mood_change = sum(
            self.mood_vector[mood] - previous_mood[mood] 
            for mood in self.mood_vector
        ) / len(self.mood_vector) if self.mood_vector else 0
        
        # If emotional response was disproportionate to outcome, adjust sensitivity
        if abs(avg_mood_change) > 0.5 and abs(outcome_valence) < 0.3:
            # Response was too intense for the outcome, reduce sensitivity slightly
            self.emotional_inertia = min(0.3, self.emotional_inertia + 0.01)
        elif abs(avg_mood_change) < 0.2 and abs(outcome_valence) > 0.5:
            # Response was too muted for the outcome, increase sensitivity slightly
            self.emotional_inertia = max(0.05, self.emotional_inertia - 0.01)
        
        # Update emotional learning based on outcome
        self.emotional_learning_rate = min(0.2, self.emotional_learning_rate + 0.001)

    def get_emotional_regulation_suggestions(self, context: Dict[str, Any] = None) -> List[str]:
        """Provide suggestions for emotional regulation based on current state"""
        suggestions = []
        
        current_stress = self.emotional_state.stress_level
        regulation_capacity = self.emotional_state.regulation_capacity
        
        if current_stress > 0.7:
            suggestions.append("High stress detected. Consider using acceptance or attention deployment strategies.")
        
        if regulation_capacity < 0.4:
            suggestions.append("Emotional regulation capacity is low. Consider taking a break or practicing mindfulness.")
        
        if self.emotional_state.recovery_rate < 0.3:
            suggestions.append("Slow emotional recovery detected. Practice stress-reduction techniques.")
        
        # Add context-specific suggestions
        if context and context.get('cognitive_load', 0) > 0.8:
            suggestions.append("High cognitive load detected. Simplify tasks to reduce emotional interference.")
        
        return suggestions

    def get_enhanced_emotional_context(self) -> Dict[str, Any]:
        """Get enhanced emotional context for decision making and memory retrieval"""
        return {
            'valence': self.emotional_state.valence,
            'arousal': self.emotional_state.arousal,
            'dominance': self.emotional_state.dominance,
            'stress_level': self.emotional_state.stress_level,
            'regulation_capacity': self.emotional_state.regulation_capacity,
            'current_mood_profile': dict(self.mood_vector),
            'emotional_state': self.emotional_state.__dict__.copy(),
            'circadian_influence': self.circadian_influence,
            'recent_events_count': len(self.emotional_events),
            'emotional_complexity': self.emotional_state.emotional_complexity,
            'emotional_differentiation': self.emotional_state.emotional_differentiation
        }

    def regulate_emotion(self, target_emotion: str, target_intensity: float, 
                        strategy: EmotionalRegulationStrategy = None) -> Dict[str, float]:
        """Actively regulate a specific emotion toward a target intensity"""
        if target_emotion not in self.mood_vector:
            return {}
        
        current_intensity = self.mood_vector[target_emotion]
        target_change = target_intensity - current_intensity
        
        if abs(target_change) < 0.05:  # Already close to target
            return {target_emotion: current_intensity}
        
        # Use specified strategy or select appropriate one
        if strategy is None:
            strategy = self._select_appropriate_regulation({})
        
        # Adjust the emotion using the selected strategy
        adjustment_factor = self.emotional_state.regulation_capacity * self.emotional_state.regulation_speed
        
        if strategy == EmotionalRegulationStrategy.COGNITIVE_REAPPRAISAL:
            adjustment = target_change * adjustment_factor
        elif strategy == EmotionalRegulationStrategy.ACCEPTANCE:
            # Acceptance doesn't force change, just reduces resistance
            adjustment = target_change * adjustment_factor * 0.3
        elif strategy == EmotionalRegulationStrategy.ATTENTION_DEPLOYMENT:
            adjustment = target_change * adjustment_factor * 0.7
        else:
            adjustment = target_change * adjustment_factor * 0.5
        
        # Apply the adjustment
        new_intensity = max(0.0, min(1.0, current_intensity + adjustment))
        self.mood_vector[target_emotion] = new_intensity
        
        # Update emotional state
        self._update_emotional_state_from_mood_change(target_emotion, adjustment)
        
        return {target_emotion: new_intensity}

    def _update_emotional_state_from_mood_change(self, emotion: str, change: float):
        """Update emotional state based on a specific mood change"""
        # Update valence based on the change
        emotion_category = self.emotion_categories.get(emotion, "neutral")
        positive_categories = ["joy_based", "love_based", "interest_based"]
        negative_categories = ["sadness_based", "anger_based", "fear_based"]
        
        if emotion_category in positive_categories:
            self.emotional_state.valence = min(1.0, self.emotional_state.valence + (change * 0.3))
        elif emotion_category in negative_categories:
            self.emotional_state.valence = max(-1.0, self.emotional_state.valence + (change * 0.3))
        
        # Update arousal based on activation level of the emotion
        activation_emotions = ["Excited", "Angry", "Anxious", "Energetic", "Frustrated"]
        calming_emotions = ["Calm", "Peaceful", "Serene", "Content"]
        
        if emotion in activation_emotions:
            self.emotional_state.arousal = min(1.0, self.emotional_state.arousal + (abs(change) * 0.2))
        elif emotion in calming_emotions:
            self.emotional_state.arousal = max(0.0, self.emotional_state.arousal - (abs(change) * 0.2))

    def get_emotional_resilience_metrics(self) -> Dict[str, float]:
        """Get metrics related to emotional resilience"""
        # Calculate emotional resilience based on various factors
        stability_score = 1.0 - self.emotional_inertia  # Lower inertia = more stable
        recovery_score = self.emotional_state.recovery_rate
        regulation_score = self.emotional_state.regulation_capacity
        
        resilience_index = (stability_score + recovery_score + regulation_score) / 3
        emotional_variability = self._calculate_emotional_variability()
        
        return {
            'resilience_index': resilience_index,
            'stability_score': stability_score,
            'recovery_rate': recovery_score,
            'regulation_capacity': regulation_score,
            'emotional_variability': emotional_variability,
            'emotional_flexibility': self.emotional_state.emotional_flexibility
        }

    def _calculate_emotional_variability(self) -> float:
        """Calculate how much emotions fluctuate over time"""
        if len(self.emotional_events) < 5:
            return 0.5  # Not enough data, return medium variability
        
        # Calculate variance in emotional intensity
        intensities = [event.intensity for event in self.emotional_events[-10:]]  # Last 10 events
        if not intensities:
            return 0.5
        
        mean_intensity = sum(intensities) / len(intensities)
        variance = sum((i - mean_intensity) ** 2 for i in intensities) / len(intensities)
        
        # Normalize to 0-1 scale where higher is more variable
        return min(1.0, variance * 2)  # Scale factor to keep within reasonable bounds