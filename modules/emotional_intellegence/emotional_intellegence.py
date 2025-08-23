import logging
from typing import Dict, Optional, List, Tuple
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
        self.mood_vector: Dict[str, float] = {mood: 0.0 for mood in self.ALL_MOODS}
        
        # Initialize emotional momentum tracking
        self.mood_momentum: Dict[str, float] = {mood: 0.0 for mood in self.ALL_MOODS}
        
        # Initialize emotional event history
        self.emotional_events: List[EmotionalEvent] = []
        
        self.last_action_result: Optional[dict] = None
        self._load_config(config_path)
        self._load_personas(persona_path)
        self.set_persona(self.personas.get("default_persona", "Balanced"))
        self.mood_processor = MoodProcessor(self)
        
        # Initialize mood dynamics parameters
        self.momentum_factor = self.config.get("mood_dynamics", {}).get("momentum_factor", 0.7)
        self.damping_factor = self.config.get("mood_dynamics", {}).get("damping_factor", 0.9)

    def _initialize_mood_dimensions(self):
        """Initialize expanded mood dimensions with primary, secondary, and complex emotions"""
        # Primary emotions from config
        primary_emotions = Config.POSITIVE_MOODS + Config.NEGATIVE_MOODS
        
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
        
        # Combine all emotions
        self.ALL_MOODS = list(set(primary_emotions + extended_primary + secondary_emotions))
        self.PRIMARY_MOODS = extended_primary
        self.SECONDARY_MOODS = secondary_emotions

    def _load_config(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f).get("emotional_intelligence_config", {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load or parse EI config: {e}")
            self.config = {"triggers": {}, "behavior_influences": {}, "mood_dynamics": {}}

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
            logger.warning(f"Persona '{persona_name}' not found. Using default multipliers.")
            self.persona = {"mood_multipliers": {}, "adaptation_rate": 0.1}
        logger.info(f"Emotional persona set to: {persona_name}")

    def update_mood(self, mood: str, delta: float):
        """Enhanced mood update with momentum and persona adaptation"""
        logger.debug(f"Updating mood '{mood}' by {delta}")
        if mood in self.mood_vector:
            # Apply persona multiplier
            multiplier = self.persona.get("mood_multipliers", {}).get(mood, 1.0)
            
            # Apply momentum factor to prevent rapid mood swings
            momentum_effect = self.mood_momentum.get(mood, 0.0) * self.momentum_factor
            adjusted_delta = (delta * multiplier) + momentum_effect
            
            # Update mood with damping to prevent oscillations
            new_value = max(0.0, self.mood_vector[mood] + adjusted_delta)
            self.mood_vector[mood] = new_value * self.damping_factor
            
            # Update momentum
            self.mood_momentum[mood] = self.mood_momentum[mood] * 0.8 + adjusted_delta * 0.2

    def blend_moods(self):
        """Blend related moods for more nuanced emotional states"""
        blending_config = self.config.get("mood_dynamics", {})
        threshold = blending_config.get("blending_threshold", 0.6)
        
        # Example blending rules
        blend_rules = {
            ("Confident", "Curious"): "Inspired",
            ("Frustrated", "Stuck"): "Resentful",
            ("Anxious", "Cautious"): "Apprehensive",
            ("Excited", "Satisfied"): "Proud"
        }
        
        for (mood1, mood2), blended_mood in blend_rules.items():
            if (mood1 in self.mood_vector and mood2 in self.mood_vector and 
                blended_mood in self.mood_vector):
                # If both source moods are above threshold, boost the blended mood
                if (self.mood_vector[mood1] > threshold and 
                    self.mood_vector[mood2] > threshold):
                    blend_strength = (self.mood_vector[mood1] + self.mood_vector[mood2]) / 2
                    self.update_mood(blended_mood, blend_strength * 0.1)

    def decay_moods(self, decay: float = 0.05):
        """Enhanced mood decay with stability controls"""
        logger.debug(f"Decaying all moods by {decay}")
        stability_threshold = self.config.get("mood_dynamics", {}).get("stability_threshold", 0.3)
        
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
        # Store previous mood state for event logging
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
            
            self.log_emotional_event(mood_changes, triggers, action_output[:200])

    def get_dominant_mood(self) -> str:
        return max(self.mood_vector, key=lambda m: self.mood_vector[m])

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
        mood = self.get_dominant_mood()
        return self.config.get("behavior_influences", {}).get(mood, {})

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
                    self.persona["mood_multipliers"][mood] = current_multiplier + (adaptation_rate * 0.1)
        elif "failure" in outcome.lower() or "negative" in outcome.lower():
            # Strengthen caution-related mood multipliers
            for mood in ["Cautious", "Anxious", "Apprehensive"]:
                if mood in self.persona.get("mood_multipliers", {}):
                    current_multiplier = self.persona["mood_multipliers"][mood]
                    self.persona["mood_multipliers"][mood] = current_multiplier + (adaptation_rate * 0.1)

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