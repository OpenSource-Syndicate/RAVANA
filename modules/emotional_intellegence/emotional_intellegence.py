import logging
from typing import Dict, Optional, List
import json
from datetime import datetime, timedelta
from core.config import Config
from .mood_processor import MoodProcessor

logger = logging.getLogger(__name__)


class EmotionalEvent:
    """Structure for logging emotional events"""

    def __init__(self, timestamp: datetime, mood_changes: Dict[str, float],
                 triggers: List[str], context: str, intensity: float):
        self.timestamp = timestamp
        self.mood_changes = mood_changes
        self.triggers = triggers
        self.context = context
        self.intensity = intensity


class EmotionalIntelligence:
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

        # Initialize mood dynamics parameters
        self.momentum_factor = self.config.get(
            "mood_dynamics", {}).get("momentum_factor", 0.7)
        self.damping_factor = self.config.get(
            "mood_dynamics", {}).get("damping_factor", 0.9)

    def _initialize_mood_dimensions(self):
        """Initialize expanded mood dimensions with primary, secondary, and complex emotions"""
        # Primary emotions from config
        config = Config()
        primary_emotions = config.POSITIVE_MOODS + config.NEGATIVE_MOODS

        # Extended primary emotions from enhanced config
        emotion_config = {
            "joy_based": ["Confident", "Excited", "Inspired", "Satisfied"],
            "interest_based": ["Curious", "Reflective", "Intrigued", "Engaged"],
            "sadness_based": ["Disappointed", "Bored", "Low Energy", "Melancholic"],
            "anger_based": ["Frustrated", "Irritated", "Stuck", "Resentful"],
            "fear_based": ["Anxious", "Apprehensive", "Cautious", "Suspicious"],
            "surprise_based": ["Astonished", "Bewildered", "Amazed", "Shocked"]
        }

        # Flatten all primary emotions
        extended_primary = []
        for emotion_category in emotion_config.values():
            extended_primary.extend(emotion_category)

        # Secondary emotions
        secondary_emotions = [
            "Hopeful", "Grateful", "Proud", "Guilty", "Lonely", "Nostalgic",
            "Embarrassed", "Jealous", "Relieved", "Surprised", "Envious", "Peaceful",
            "Compassionate", "Confused", "Optimistic", "Pessimistic"
        ]

        # Complex emotional blends
        complex_emotions = [
            "Bittersweet", "Nostalgic contentment", "Melancholic curiosity", 
            "Anxious anticipation", "Contemplative serenity", "Frustrated determination",
            "Cautious optimism", "Melancholic joy", "Tender sadness", 
            "Reflective excitement", "Pragmatic hope", "Wistful gratitude",
            "Wonderful concern", "Peaceful vigilance", "Determined patience",
            "Adaptive resilience", "Thoughtful caution", "Curious wonder",
            "Balanced contemplation", "Thoughtful determination", "Mindful serenity"
        ]

        # Combine all emotions
        self.ALL_MOODS = list(
            set(primary_emotions + extended_primary + secondary_emotions + complex_emotions))
        self.PRIMARY_MOODS = extended_primary
        self.SECONDARY_MOODS = secondary_emotions
        self.COMPLEX_EMOTIONS = complex_emotions

    def _load_config(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f).get(
                    "emotional_intelligence_config", {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load or parse EI config: {e}")
            self.config = {"triggers": {},
                           "behavior_influences": {}, "mood_dynamics": {}}

    def _load_personas(self, persona_path: str):
        try:
            with open(persona_path, 'r') as f:
                self.personas = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load or parse personas config: {e}")
            self.personas = {"personas": {}, "default_persona": "Balanced"}

    def set_persona(self, persona_name: str):
        self.persona = self.personas.get("personas", {}).get(persona_name, {})
        if not self.persona:
            logger.warning(
                f"Persona '{persona_name}' not found. Using default multipliers.")
            self.persona = {"mood_multipliers": {}, "adaptation_rate": 0.1}
        logger.info(f"Emotional persona set to: {persona_name}")

    def update_mood(self, mood: str, delta: float):
        """Enhanced mood update with momentum and persona adaptation"""
        logger.debug(f"Updating mood '{mood}' by {delta}")
        if mood in self.mood_vector:
            # Apply persona multiplier
            multiplier = self.persona.get(
                "mood_multipliers", {}).get(mood, 1.0)

            # Apply momentum factor to prevent rapid mood swings
            momentum_effect = self.mood_momentum.get(
                mood, 0.0) * self.momentum_factor
            adjusted_delta = (delta * multiplier) + momentum_effect

            # Update mood with damping to prevent oscillations
            new_value = max(0.0, self.mood_vector[mood] + adjusted_delta)
            self.mood_vector[mood] = new_value * self.damping_factor

            # Update momentum
            self.mood_momentum[mood] = self.mood_momentum[mood] * \
                0.8 + adjusted_delta * 0.2

    def blend_moods(self):
        """Blend related moods for more nuanced emotional states"""
        blending_config = self.config.get("mood_dynamics", {})
        threshold = blending_config.get("blending_threshold", 0.6)

        # Enhanced blending rules with complex emotions
        blend_rules = {
            ("Confident", "Curious"): "Inspired",
            ("Frustrated", "Stuck"): "Resentful",
            ("Anxious", "Cautious"): "Apprehensive",
            ("Excited", "Satisfied"): "Proud",
            ("Bored", "Low Energy"): "Melancholic",
            ("Curious", "Reflective"): "Intrigued",
            ("Frustrated", "Irritated"): "Stuck",
            ("Excited", "Inspired"): "Energized",
            # Complex emotional blends
            ("Hopeful", "Anxious"): "Anxious anticipation",
            ("Satisfied", "Melancholic"): "Bittersweet",
            ("Curious", "Peaceful"): "Contemplative serenity",
            ("Frustrated", "Determined"): "Frustrated determination",
            ("Cautious", "Optimistic"): "Cautious optimism",
            ("Melancholic", "Joyful"): "Melancholic joy",
            ("Grateful", "Nostalgic"): "Wistful gratitude",
            ("Concerned", "Wonder"): "Wonderful concern",
            ("Calm", "Alert"): "Peaceful vigilance",
            ("Patient", "Determined"): "Determined patience",
            ("Adaptive", "Strong"): "Adaptive resilience",
            ("Thoughtful", "Cautious"): "Thoughtful caution",
            ("Curious", "Awe"): "Curious wonder",
            ("Balanced", "Thoughtful"): "Balanced contemplation",
            ("Thoughtful", "Determined"): "Thoughtful determination",
            ("Mindful", "Calm"): "Mindful serenity"
        }

        # Dynamic blending based on current mood intensities
        for (mood1, mood2), blended_mood in blend_rules.items():
            if (mood1 in self.mood_vector and mood2 in self.mood_vector and
                    blended_mood in self.mood_vector):
                # If both source moods are above threshold, boost the blended mood
                intensity1 = self.mood_vector[mood1]
                intensity2 = self.mood_vector[mood2]

                # Only blend if both moods are present
                if intensity1 > 0.1 and intensity2 > 0.1:
                    # Calculate blend strength based on the geometric mean of intensities
                    blend_strength = (intensity1 * intensity2) ** 0.5
                    # Apply blending with dynamic scaling
                    scaling_factor = blending_config.get("blending_scaling", 0.15)
                    self.update_mood(blended_mood, blend_strength * scaling_factor)

        # Advanced blending: detect emotion patterns and create complex states
        self._detect_complex_emotional_patterns()

    def _detect_complex_emotional_patterns(self):
        """Detect complex emotional patterns and set appropriate complex emotions."""
        # Define complex emotional patterns
        patterns = {
            "contemplative_state": {
                "moods": ["Reflective", "Calm", "Curious"],
                "complex_emotion": "Balanced contemplation",
                "threshold": 0.4
            },
            "anxious_curious": {
                "moods": ["Anxious", "Curious", "Alert"],
                "complex_emotion": "Anxious anticipation",
                "threshold": 0.5
            },
            "bittersweet_memories": {
                "moods": ["Melancholic", "Satisfied", "Nostalgic"],
                "complex_emotion": "Bittersweet",
                "threshold": 0.5
            },
            "thoughtful_determination": {
                "moods": ["Thoughtful", "Determined", "Focussed"],
                "complex_emotion": "Thoughtful determination",
                "threshold": 0.6
            },
            "mindful_resilience": {
                "moods": ["Mindful", "Calm", "Adaptive"],
                "complex_emotion": "Adaptive resilience",
                "threshold": 0.5
            },
            "curious_wonder": {
                "moods": ["Curious", "Amazed", "Thoughtful"],
                "complex_emotion": "Curious wonder",
                "threshold": 0.5
            }
        }

        for pattern_name, pattern_data in patterns.items():
            moods = pattern_data["moods"]
            complex_emotion = pattern_data["complex_emotion"]
            threshold = pattern_data["threshold"]

            # Check if all required moods are above threshold
            all_moods_present = all(
                self.mood_vector.get(mood, 0) >= threshold for mood in moods
            )

            if all_moods_present and complex_emotion in self.mood_vector:
                # Calculate the average intensity of contributing moods
                avg_intensity = sum(self.mood_vector[mood] for mood in moods) / len(moods)
                # Set the complex emotion with a moderate boost
                self.update_mood(complex_emotion, avg_intensity * 0.7)

    def decay_moods(self, decay: float = 0.05):
        """Enhanced mood decay with stability controls"""
        logger.debug(f"Decaying all moods by {decay}")
        stability_threshold = self.config.get(
            "mood_dynamics", {}).get("stability_threshold", 0.3)

        for mood in self.mood_vector:
            # Apply stronger decay to moods above stability threshold
            current_value = self.mood_vector[mood]
            if current_value > stability_threshold:
                effective_decay = decay * 1.5  # Stronger decay for high-intensity moods
            else:
                effective_decay = decay

            self.mood_vector[mood] = max(0.0, current_value - effective_decay)

            # Reduce momentum as well
            self.mood_momentum[mood] = self.mood_momentum[mood] * 0.9

    def log_emotional_event(self, mood_changes: Dict[str, float],
                            triggers: List[str], context: str):
        """Log emotional events with timestamps and intensity"""
        intensity = sum(abs(change) for change in mood_changes.values())
        event = EmotionalEvent(
            timestamp=datetime.now(),
            mood_changes=mood_changes,
            triggers=triggers,
            context=context,
            intensity=intensity
        )
        self.emotional_events.append(event)

        # Keep only recent events (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.emotional_events = [
            event for event in self.emotional_events
            if event.timestamp > cutoff_time
        ]

    def get_emotional_context(self) -> Dict[str, any]:
        """Get emotional context for memory retrieval and decision making"""
        return {
            "dominant_mood": self.get_dominant_mood(),
            "mood_vector": self.get_mood_vector(),
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "triggers": event.triggers,
                    "intensity": event.intensity
                }
                for event in self.emotional_events[-5:]  # Last 5 events
            ]
        }

    def process_action_result(self, action_result: dict):
        # Store previous mood state for learning
        previous_mood = dict(self.mood_vector)

        self.mood_processor.process_action_result(action_result)

        # Apply mood blending after processing
        self.blend_moods()

        # Log emotional event
        mood_changes = {
            mood: self.mood_vector[mood] - previous_mood[mood]
            for mood in self.mood_vector
            if abs(self.mood_vector[mood] - previous_mood[mood]) > 0.01
        }

        if mood_changes:
            triggers = [k for k, v in action_result.items() if v]
            context = json.dumps(action_result)[:200]  # Truncate for brevity
            self.log_emotional_event(mood_changes, triggers, context)
        
        # Enhanced: Learn from emotional outcomes to improve emotional regulation
        # Add mood information to action_result for learning
        action_result_with_moods = action_result.copy()
        action_result_with_moods['mood_before'] = previous_mood
        action_result_with_moods['mood_after'] = dict(self.mood_vector)
        
        # Apply emotional learning
        self.learn_from_emotional_outcomes(action_result_with_moods)

    def process_action_natural(self, action_output: str):
        # Store previous mood state for event logging
        previous_mood = dict(self.mood_vector)

        self.mood_processor.process_action_natural(action_output)

        # Apply mood blending after processing
        self.blend_moods()

        # Log emotional event
        mood_changes = {
            mood: self.mood_vector[mood] - previous_mood[mood]
            for mood in self.mood_vector
            if abs(self.mood_vector[mood] - previous_mood[mood]) > 0.01
        }

        if mood_changes:
            # Extract likely triggers from action output
            triggers = []
            for trigger in self.config.get("triggers", {}):
                if trigger.lower() in action_output.lower():
                    triggers.append(trigger)

            self.log_emotional_event(
                mood_changes, triggers, action_output[:200])

    def get_dominant_mood(self) -> str:
        """Get the most dominant mood, with tie-breaking logic"""
        if not self.mood_vector:
            return "Neutral"

        # Get all moods sorted by intensity
        sorted_moods = sorted(self.mood_vector.items(),
                              key=lambda x: x[1], reverse=True)

        # If top mood is significantly stronger than second, return it
        if len(sorted_moods) >= 2 and sorted_moods[0][1] > sorted_moods[1][1] + 0.2:
            return sorted_moods[0][0]

        # For close ties, consider mood groups
        primary_mood = sorted_moods[0][0]
        secondary_mood = sorted_moods[1][0] if len(sorted_moods) > 1 else None

        # Special combinations that create meaningful dominant moods
        mood_combinations = {
            ("Confident", "Curious"): "Inspired",
            ("Frustrated", "Stuck"): "Resentful",
            ("Excited", "Satisfied"): "Proud",
            ("Anxious", "Cautious"): "Apprehensive"
        }

        # Check if we have a meaningful combination
        combination_key = (
            primary_mood, secondary_mood) if secondary_mood else None
        if combination_key in mood_combinations:
            return mood_combinations[combination_key]
        elif (secondary_mood, primary_mood) in mood_combinations:  # Reverse order
            return mood_combinations[(secondary_mood, primary_mood)]

        # Default to primary mood
        return primary_mood

    def get_mood_vector(self) -> Dict[str, float]:
        return dict(self.mood_vector)

    def get_mood_intensity_level(self, mood: str) -> str:
        """Get the intensity level of a specific mood (low, medium, high)"""
        if mood not in self.mood_vector:
            return "unknown"

        value = self.mood_vector[mood]
        intensity_levels = self.config.get("emotion_intensity_levels", {})

        low_range = intensity_levels.get("low", [0.0, 0.33])
        med_range = intensity_levels.get("medium", [0.34, 0.66])
        high_range = intensity_levels.get("high", [0.67, 1.0])

        if low_range[0] <= value <= low_range[1]:
            return "low"
        elif med_range[0] <= value <= med_range[1]:
            return "medium"
        elif high_range[0] <= value <= high_range[1]:
            return "high"
        else:
            return "unknown"

    def set_mood_vector(self, mood_vector: Dict[str, float]):
        """Set the mood vector for state recovery."""
        try:
            for mood, value in mood_vector.items():
                if mood in self.mood_vector:
                    self.mood_vector[mood] = max(0.0, float(value))
            logger.info("Mood vector restored from previous state")
        except Exception as e:
            logger.error(f"Error setting mood vector: {e}")

    def influence_behavior(self) -> dict:
        """
        Generate behavior modifications based on current emotional state.
        This function now returns more comprehensive emotional influences on behavior.
        """
        dominant_mood = self.get_dominant_mood()
        mood_vector = self.get_mood_vector()
        
        # Define comprehensive behavior influences for different moods
        behavior_influences = self.config.get("behavior_influences", {})
        
        # Default behavior modifications
        behavior_modifications = {
            "exploration_bias": 0.0,  # How much to favor exploration vs exploitation
            "risk_tolerance": 0.5,    # How much risk to accept (0.0 to 1.0)
            "learning_rate": 0.1,     # How quickly to adapt behavior
            "action_selection_bias": 0.0,  # Bias toward certain types of actions
            "attention_span": 1.0,    # How long to focus on a task
            "social_engagement": 0.5, # Level of social interaction (for conversational AI)
            "creativity_bias": 0.5,   # Tendency toward creative solutions
            "conservation_factor": 0.0,  # How much to preserve current state
            "decision_speed": 0.5,    # Speed vs thoroughness in decision making
            "memory_recall_bias": 0.0  # Bias toward positive or negative memories
        }
        
        # Apply mood-specific modifications
        if dominant_mood in behavior_influences:
            mood_influences = behavior_influences[dominant_mood]
            for key, value in mood_influences.items():
                if key in behavior_modifications:
                    behavior_modifications[key] = value
        else:
            # Apply default mood-based modifications if specific mood not found
            if dominant_mood in ["Curious", "Excited", "Intrigued", "Inspired"]:
                behavior_modifications.update({
                    "exploration_bias": 0.8,
                    "risk_tolerance": 0.7,
                    "creativity_bias": 0.8,
                    "learning_rate": 0.3,
                    "action_selection_bias": 0.6
                })
            elif dominant_mood in ["Frustrated", "Stuck", "Irritated", "Resentful"]:
                behavior_modifications.update({
                    "exploration_bias": 0.6,
                    "risk_tolerance": 0.4,
                    "attention_span": 0.3,
                    "decision_speed": 0.8,  # Haste due to frustration
                    "conservation_factor": 0.7
                })
            elif dominant_mood in ["Anxious", "Apprehensive", "Worried"]:
                behavior_modifications.update({
                    "exploration_bias": 0.2,
                    "risk_tolerance": 0.2,
                    "decision_speed": 0.2,  # Hesitant decision making
                    "conservation_factor": 0.9,
                    "memory_recall_bias": -0.5  # Tendency toward negative memories
                })
            elif dominant_mood in ["Confident", "Satisfied", "Proud"]:
                behavior_modifications.update({
                    "exploration_bias": 0.5,
                    "risk_tolerance": 0.6,
                    "decision_speed": 0.7,  # Confident decisions
                    "action_selection_bias": 0.7
                })
            elif dominant_mood in ["Calm", "Peaceful", "Content"]:
                behavior_modifications.update({
                    "exploration_bias": 0.3,
                    "risk_tolerance": 0.3,
                    "decision_speed": 0.3,  # Thoughtful, deliberate decisions
                    "learning_rate": 0.1,  # Cautious learning
                    "attention_span": 0.9  # Sustained attention
                })
            elif dominant_mood in ["Bored", "Low Energy"]:
                behavior_modifications.update({
                    "exploration_bias": 0.9,  # Seeking stimulation
                    "creativity_bias": 0.7,
                    "attention_span": 0.2,
                    "action_selection_bias": 0.8  # Seeking change
                })
        
        # Apply intensity-based scaling
        dominant_intensity = mood_vector.get(dominant_mood, 0.0)
        if dominant_intensity > 0.7:  # High intensity mood
            # Amplify all behavior modifications
            for key in behavior_modifications:
                if key != "exploration_bias" and key != "risk_tolerance":
                    behavior_modifications[key] = self._scale_behavior_mod(
                        behavior_modifications[key], 
                        dominant_intensity, 
                        amplify=True
                    )
        
        # Add emotional memory bias based on current mood
        behavior_modifications["emotional_memory_bias"] = self._calculate_emotional_memory_bias(mood_vector)
        
        # Add emotional decision weight
        behavior_modifications["emotional_decision_weight"] = self._calculate_emotional_decision_weight(mood_vector)
        
        # Log the behavior influence for debugging
        logger.debug(f"Emotional influence on behavior: {dominant_mood} ({dominant_intensity:.2f}) -> {behavior_modifications}")
        
        return behavior_modifications

    def _scale_behavior_mod(self, value: float, intensity: float, amplify: bool = True) -> float:
        """Scale behavior modification based on emotion intensity."""
        if amplify:
            # Amplify the behavior modification based on intensity
            return min(1.0, value + (intensity * 0.3))
        else:
            # Reduce the behavior modification
            return max(0.0, value - (intensity * 0.3))
    
    def _calculate_emotional_memory_bias(self, mood_vector: Dict[str, float]) -> float:
        """Calculate bias toward emotionally congruent memories."""
        # Positive emotions increase positive memory recall
        positive_moods = [mood for mood in mood_vector 
                         if mood in ["Confident", "Excited", "Inspired", "Satisfied", 
                                   "Hopeful", "Grateful", "Proud", "Peaceful", "Joyful"]]
        negative_moods = [mood for mood in mood_vector 
                         if mood in ["Frustrated", "Stuck", "Anxious", "Apprehensive", 
                                   "Disappointed", "Bored", "Melancholic", "Resentful"]]
        
        positive_score = sum(mood_vector.get(mood, 0.0) for mood in positive_moods)
        negative_score = sum(mood_vector.get(mood, 0.0) for mood in negative_moods)
        
        # Return a bias value between -1 (negative bias) and 1 (positive bias)
        total_emotion = positive_score + negative_score
        if total_emotion == 0:
            return 0.0
        
        return (positive_score - negative_score) / total_emotion

    def _calculate_emotional_decision_weight(self, mood_vector: Dict[str, float]) -> float:
        """Calculate how much emotions should influence decisions."""
        # Calculate overall emotional intensity
        total_intensity = sum(mood_vector.values()) / len(mood_vector) if mood_vector else 0.0
        
        # Determine if emotions are highly active
        highly_emotional = any(val > 0.7 for val in mood_vector.values())
        
        # Emotions should have more weight when highly active, but not override rational thought completely
        weight = min(0.5, total_intensity * 0.8)  # Cap emotional influence
        if highly_emotional:
            weight = min(0.7, weight + 0.2)  # Boost if highly emotional
        
        return weight

    def adapt_persona(self, experience: str, outcome: str):
        """Adapt persona based on significant emotional experiences"""
        adaptation_rate = self.persona.get("adaptation_rate", 0.1)

        # This is a simplified adaptation mechanism
        # In a full implementation, this would use more sophisticated learning
        if "success" in outcome.lower() or "positive" in outcome.lower():
            # Strengthen positive mood multipliers
            for mood in ["Confident", "Curious", "Inspired"]:
                if mood in self.persona.get("mood_multipliers", {}):
                    current_multiplier = self.persona["mood_multipliers"][mood]
                    self.persona["mood_multipliers"][mood] = current_multiplier + \
                        (adaptation_rate * 0.1)
        elif "failure" in outcome.lower() or "negative" in outcome.lower():
            # Strengthen caution-related mood multipliers
            for mood in ["Cautious", "Anxious", "Apprehensive"]:
                if mood in self.persona.get("mood_multipliers", {}):
                    current_multiplier = self.persona["mood_multipliers"][mood]
                    self.persona["mood_multipliers"][mood] = current_multiplier + \
                        (adaptation_rate * 0.1)

    def learn_from_emotional_outcomes(self, action_result: dict):
        """Learn from emotional outcomes to improve emotional regulation."""
        # Extract outcome information
        outcome = action_result.get("result", "")
        mood_before = action_result.get("mood_before", {})
        mood_after = action_result.get("mood_after", {})
        
        # Calculate mood change
        mood_change = {}
        for mood in self.ALL_MOODS:
            before_val = mood_before.get(mood, 0.0)
            after_val = mood_after.get(mood, 0.0)
            mood_change[mood] = after_val - before_val
        
        # Evaluate emotional outcome quality
        positive_outcomes = ["success", "achievement", "progress", "learning", "discovery", "improvement"]
        negative_outcomes = ["failure", "error", "setback", "frustration", "confusion", "regression"]
        
        is_positive = any(outcome_phrase in outcome.lower() for outcome_phrase in positive_outcomes)
        is_negative = any(outcome_phrase in outcome.lower() for outcome_phrase in negative_outcomes)
        
        # Learn emotional regulation strategies based on outcome
        if is_positive and any(val > 0.3 for val in mood_change.values()):
            # Learn to repeat emotional patterns that led to positive outcomes
            self._learn_positive_emotional_pattern(mood_before, mood_after)
        elif is_negative and any(val < -0.3 for val in mood_change.values()):
            # Learn emotional regulation strategies for negative outcomes
            self._learn_negative_emotional_regulation(mood_before, mood_after)
        
    def _learn_positive_emotional_pattern(self, mood_before: dict, mood_after: dict):
        """Learn emotional patterns that led to positive outcomes."""
        # Identify which moods were beneficial for positive outcomes
        beneficial_moods = []
        for mood, after_val in mood_after.items():
            before_val = mood_before.get(mood, 0.0)
            # If mood increased and was associated with positive outcome, it's potentially beneficial
            if after_val > before_val and after_val > 0.5:
                beneficial_moods.append((mood, after_val - before_val))
        
        # Update learning-based modifiers based on beneficial moods
        for mood, strength in beneficial_moods:
            if mood not in self.mood_vector:
                continue
                
            # Apply a small boost to beneficial moods for future similar situations
            current_modifier = self.persona.get("mood_multipliers", {}).get(mood, 1.0)
            # Only boost slightly to avoid overfitting to a single experience
            new_modifier = min(2.0, current_modifier + (strength * 0.1))
            
            # Ensure persona has mood_multipliers dict
            if "mood_multipliers" not in self.persona:
                self.persona["mood_multipliers"] = {}
            self.persona["mood_multipliers"][mood] = new_modifier
    
    def _learn_negative_emotional_regulation(self, mood_before: dict, mood_after: dict):
        """Learn emotional regulation strategies for negative outcomes."""
        # Identify which moods led to negative outcomes and should be regulated
        problematic_moods = []
        for mood, after_val in mood_after.items():
            before_val = mood_before.get(mood, 0.0)
            # If mood increased significantly and was associated with negative outcome, it's problematic
            if after_val > before_val and after_val > 0.6:
                problematic_moods.append((mood, after_val - before_val))
        
        # Apply regulation strategies for problematic moods
        for mood, intensity in problematic_moods:
            if mood not in self.mood_vector:
                continue
                
            # Apply emotional regulation techniques
            self._apply_emotional_regulation(mood, intensity)
    
    def _apply_emotional_regulation(self, mood: str, intensity: float):
        """Apply emotional regulation techniques to problematic moods."""
        # Counteract the problematic mood with balancing emotions
        balancing_moods = {
            "Anxious": ["Calm", "Mindful", "Thoughtful"],
            "Frustrated": ["Patient", "Adaptive", "Determined"],
            "Overwhelmed": ["Calm", "Mindful", "Organized"],
            "Stuck": ["Curious", "Adaptive", "Determined"],
            "Bored": ["Curious", "Excited", "Engaged"],
            "Confused": ["Thoughtful", "Reflective", "Patient"],
            "Angry": ["Calm", "Empathetic", "Principled"],
            "Sad": ["Grateful", "Hopeful", "Connected"]
        }
        
        # Apply balancing emotions if available
        if mood in balancing_moods:
            for balancing_mood in balancing_moods[mood]:
                if balancing_mood in self.mood_vector:
                    # Apply a proportional boost to balancing emotions
                    regulation_strength = min(0.3, intensity * 0.5)  # Cap regulation strength
                    self.update_mood(balancing_mood, regulation_strength)
        
        # Apply general regulation techniques
        self._apply_general_regulation_techniques()
    
    def _apply_general_regulation_techniques(self):
        """Apply general emotional regulation techniques."""
        # Technique 1: Temporarily reduce intensity of all emotions
        for mood in self.mood_vector:
            current_val = self.mood_vector[mood]
            # Apply soft damping to reduce emotional intensity
            self.mood_vector[mood] = current_val * 0.9
        
        # Technique 2: Boost regulatory emotions
        regulatory_moods = ["Mindful", "Calm", "Balanced", "Thoughtful", "Patient"]
        for mood in regulatory_moods:
            if mood in self.mood_vector:
                # Apply small boost to regulatory emotions
                self.update_mood(mood, 0.1)
    
    def get_emotional_regulation_advice(self, current_mood_state: dict) -> str:
        """Provide emotional regulation advice based on learned patterns."""
        # Determine the dominant problematic mood
        sorted_moods = sorted(current_mood_state.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_moods:
            return "Emotional state is balanced."
        
        dominant_mood, intensity = sorted_moods[0]
        
        if intensity < 0.4:
            return "Emotional state is well-regulated."
        
        # Provide targeted regulation advice
        regulation_advice = {
            "Anxious": "Practice mindfulness and focus on immediate, controllable tasks.",
            "Frustrated": "Take a step back and consider alternative approaches to the challenge.",
            "Overwhelmed": "Break down tasks into smaller, manageable components.",
            "Stuck": "Approach the problem from a different angle or take a brief break.",
            "Bored": "Seek new challenges or explore related topics that spark curiosity.",
            "Confused": "Review fundamental concepts and approach the problem systematically.",
            "Angry": "Take a pause and consider the situation from other perspectives.",
            "Sad": "Connect with positive memories or engage in activities that bring contentment."
        }
        
        if dominant_mood in regulation_advice:
            return f"{dominant_mood} regulation: {regulation_advice[dominant_mood]}"
        else:
            return f"Regulation advice for {dominant_mood}: Consider mindfulness techniques to achieve emotional balance."


if __name__ == "__main__":
    ei = EmotionalIntelligence()
    action_outputs = [
        "The agent discovered a new topic about quantum computing.",
        "Task completed successfully.",
        "An error occurred while processing the data.",
        "The agent repeated the same step multiple times.",
        "No activity detected for a long period.",
        "Major project milestone achieved!",
    ]
    for i, output in enumerate(action_outputs):
        ei.process_action_natural(output)
        print(f"After action {i+1}: {output}")
        print("Mood vector:", ei.get_mood_vector())
        print("Dominant mood:", ei.get_dominant_mood())
        print("Behavior suggestion:", ei.influence_behavior())
        print("-")

    ei.set_persona("Pessimistic")
    print("\nSwitching persona to Pessimistic...\n")
    for i, output in enumerate(action_outputs):
        ei.process_action_natural(output)
        print(f"After action {i+1}: {output}")
        print("Mood vector:", ei.get_mood_vector())
        print("Dominant mood:", ei.get_dominant_mood())
        print("Behavior suggestion:", ei.influence_behavior())
        print("-")
