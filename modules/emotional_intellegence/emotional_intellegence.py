import logging
from typing import Dict, Optional
import json
from core.config import Config
from .mood_processor import MoodProcessor

logger = logging.getLogger(__name__)

class EmotionalIntelligence:
    def __init__(self, config_path='modules/emotional_intellegence/config.json', persona_path='modules/emotional_intellegence/persona.json'):
        self.BASIC_MOODS = Config.POSITIVE_MOODS + Config.NEGATIVE_MOODS
        self.mood_vector: Dict[str, float] = {mood: 0.0 for mood in self.BASIC_MOODS}
        self.last_action_result: Optional[dict] = None
        self._load_config(config_path)
        self._load_personas(persona_path)
        self.set_persona(self.personas.get("default_persona", "Optimistic"))
        self.mood_processor = MoodProcessor(self)

    def _load_config(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f).get("emotional_intelligence_config", {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load or parse EI config: {e}")
            self.config = {"triggers": {}, "behavior_influences": {}}

    def _load_personas(self, persona_path: str):
        try:
            with open(persona_path, 'r') as f:
                self.personas = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load or parse personas config: {e}")
            self.personas = {"personas": {}, "default_persona": "Optimistic"}

    def set_persona(self, persona_name: str):
        self.persona = self.personas.get("personas", {}).get(persona_name, {})
        if not self.persona:
            logger.warning(f"Persona '{persona_name}' not found. Using default multipliers.")
            self.persona = {"mood_multipliers": {}}
        logger.info(f"Emotional persona set to: {persona_name}")

    def update_mood(self, mood: str, delta: float):
        logger.debug(f"Updating mood '{mood}' by {delta}")
        if mood in self.mood_vector:
            multiplier = self.persona.get("mood_multipliers", {}).get(mood, 1.0)
            self.mood_vector[mood] = max(0.0, self.mood_vector[mood] + delta * multiplier)

    def decay_moods(self, decay: float = 0.05):
        logger.debug(f"Decaying all moods by {decay}")
        for mood in self.mood_vector:
            self.mood_vector[mood] = max(0.0, self.mood_vector[mood] - decay)

    def process_action_result(self, action_result: dict):
        self.mood_processor.process_action_result(action_result)

    def process_action_natural(self, action_output: str):
        self.mood_processor.process_action_natural(action_output)

    def get_dominant_mood(self) -> str:
        return max(self.mood_vector, key=lambda m: self.mood_vector[m])

    def get_mood_vector(self) -> Dict[str, float]:
        return dict(self.mood_vector)

    def influence_behavior(self) -> dict:
        mood = self.get_dominant_mood()
        return self.config.get("behavior_influences", {}).get(mood, {})

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