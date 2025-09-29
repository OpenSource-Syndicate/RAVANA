import logging
from typing import Dict
import json
import re
from core.llm import call_llm, safe_call_llm

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
                    llm_based_update = self._get_llm_mood_update(
                        update["prompt"], self.ei.get_mood_vector(), action_result)
                    for mood, delta in llm_based_update.items():
                        self.ei.update_mood(mood, delta)
                else:
                    # Use direct deltas
                    for mood, delta in update.items():
                        self.ei.update_mood(mood, delta)

        logger.debug(f"Mood vector after update: {self.ei.mood_vector}")
        self.ei.last_action_result = action_result

    def _extract_json_from_response(self, response: str) -> Dict:
        """
        Extract JSON from LLM response with multiple fallback strategies and enhanced error handling.

        Args:
            response: Raw LLM response string

        Returns:
            Dictionary parsed from JSON, or empty dict if parsing fails
        """
        # Handle empty or None responses
        if not response or not str(response).strip():
            logger.warning("Empty or None response received from LLM")
            return {}

        response_text = str(response).strip()

        # Check for common error indicators
        error_indicators = ["error", "exception", "[error", "failure"]
        if any(indicator in response_text.lower() for indicator in error_indicators):
            logger.warning(
                f"LLM response indicates error: {response_text[:100]}...")
            return {}

        # Strategy 1: Try to parse the entire response as JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed: {e}")

        # Strategy 2: Look for JSON in markdown code blocks
        json_match = re.search(
            r'```(?:json)?\s*({.*?})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from code block: {e}")

        # Strategy 3: Look for any JSON-like structure
        json_match = re.search(r'({.*})', response_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                # Fix common JSON issues
                # Add quotes to keys
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                # Remove trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                # Remove trailing commas
                json_str = re.sub(r',\s*\]', ']', json_str)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse extracted JSON structure: {e}")

        # Strategy 4: Handle common LLM response patterns
        # Remove common prefixes/suffixes
        # Remove everything before first {
        cleaned_response = re.sub(r'^[^{]*', '', response_text)
        # Remove everything after last }
        cleaned_response = re.sub(r'[^}]*$', '', cleaned_response)

        if cleaned_response and cleaned_response.startswith('{') and cleaned_response.endswith('}'):
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse cleaned response: {e}")

        logger.error(
            f"Could not extract valid JSON from LLM response (length: {len(response_text)}): {response_text[:200]}...")
        return {}

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
        # Use safe_call_llm instead of call_llm for better error handling
        llm_response = safe_call_llm(prompt, timeout=30, retries=3)

        # Extract JSON from the response
        return self._extract_json_from_response(llm_response)

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

        # Use safe_call_llm instead of call_llm for better error handling
        llm_response = safe_call_llm(prompt, timeout=30, retries=3)
        logger.debug(f"LLM response: {llm_response}")

        # Extract JSON from the response
        triggers = self._extract_json_from_response(llm_response)
        self.process_action_result(triggers)
