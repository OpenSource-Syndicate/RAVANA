import logging
from typing import Dict
import json
import re
from .llm import call_llm

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
                for mood, delta in mood_updates[trigger].items():
                    self.ei.update_mood(mood, delta)
        
        logger.debug(f"Mood vector after update: {self.ei.mood_vector}")
        self.ei.last_action_result = action_result

    def process_action_natural(self, action_output: str):
        logger.debug(f"Processing natural action output: {action_output}")

        definitions = self.ei.config["triggers"]
        
        prompt = f"""
You are an AI analysis system. Your task is to classify an AI agent's action output based on predefined triggers.
Analyze the action output below and respond with only a valid JSON object mapping each trigger to a boolean value.
Be nuanced: an action can trigger multiple categories. For example, discovering a new fact while making progress on a task should trigger both.

**Context:**
- Dominant Mood: {self.ei.get_dominant_mood()}
- Persona: {self.ei.persona.get('name', 'default')}

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