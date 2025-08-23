import logging
from typing import Dict
import json
import re
from core.llm import call_llm

logger = logging.getLogger(__name__)

class MoodProcessor:
    def __init__(self, emotional_intelligence_instance):
        self.ei = emotional_intelligence_instance

    def process_action_result(self, action_result: dict):
        logger.debug(f"Processing action result: {action_result}")
        logger.debug(f"Mood vector before update: {self.ei.mood_vector}")
        self.ei.decay_moods()

        mood_updates = self.ei.config.get("mood_updates", {})

        for trigger, is_present in action_result.items():
            if is_present and trigger in mood_updates:
                update = mood_updates[trigger]
                if "prompt" in update:
                    # Use LLM for nuanced update
                    llm_based_update = self._get_llm_mood_update(update["prompt"], self.ei.get_mood_vector(), action_result)
                    for mood, delta in llm_based_update.items():
                        self.ei.update_mood(mood, delta)
                else:
                    # Use direct deltas
                    for mood, delta in update.items():
                        self.ei.update_mood(mood, delta)
        
        logger.debug(f"Mood vector after update: {self.ei.mood_vector}")
        self.ei.last_action_result = action_result

    def _get_llm_mood_update(self, prompt_template: str, current_mood: Dict[str, float], action_result: dict) -> Dict[str, float]:
        # Enhanced prompt with emotional context
        prompt = f"""
You are an AI's emotional core. Your task is to update the AI's mood based on its recent action.
Analyze the action result and the AI's current emotional state to determine a nuanced mood update.

**Current Mood:**
{json.dumps(current_mood, indent=2)}

**Recent Emotional Events:**
{json.dumps([{
    "timestamp": event.timestamp.isoformat(),
    "triggers": event.triggers,
    "intensity": event.intensity
} for event in self.ei.emotional_events[-3:]], indent=2)}

**Action Result:**
{json.dumps(action_result, indent=2)}

**All Possible Moods:**
{json.dumps(self.ei.ALL_MOODS, indent=2)}

**Instructions:**
{prompt_template}

**Your JSON Response (only a JSON object with mood deltas, e.g., {{"Confident": 0.1, "Frustrated": -0.05}}):**
"""
        llm_response = call_llm(prompt)
        try:
            return json.loads(llm_response)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM mood update response: {e}")
            return {}

    def process_action_natural(self, action_output: str):
        logger.debug(f"Processing natural action output: {action_output}")

        definitions = self.ei.config["triggers"]
        
        # Enhanced prompt with emotional context
        prompt = f"""
You are an AI analysis system. Your task is to classify an AI agent's action output based on predefined triggers.
Analyze the action output below and respond with only a valid JSON object mapping each trigger to a boolean value.
Be nuanced: an action can trigger multiple categories. For example, discovering a new fact while making progress on a task should trigger both.

**Context:**
- Dominant Mood: {self.ei.get_dominant_mood()}
- Mood Intensity Level: {self.ei.get_mood_intensity_level(self.ei.get_dominant_mood())}
- Persona: {self.ei.persona.get('name', 'default')}
- Recent Emotional Events Count: {len(self.ei.emotional_events)}

**Trigger Definitions:**
{json.dumps(definitions, indent=2)}

**Action Output:**
"{action_output}"

**Your JSON Response (only the JSON object):**
"""
        
        llm_response = call_llm(prompt)
        logger.debug(f"LLM response: {llm_response}")

        triggers = {}
        try:
            # The LLM should return only a valid JSON object, so we can parse it directly.
            triggers = json.loads(llm_response)
            logger.debug(f"Parsed triggers: {triggers}")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response JSON: {e}")
            # Fallback to regex if direct parsing fails, just in case.
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    triggers = json.loads(json_str)
                    logger.debug(f"Parsed triggers with regex fallback: {triggers}")
                except json.JSONDecodeError as fallback_e:
                    logger.error(f"Error parsing LLM response JSON with regex fallback: {fallback_e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during parsing: {e}")

        self.process_action_result(triggers)