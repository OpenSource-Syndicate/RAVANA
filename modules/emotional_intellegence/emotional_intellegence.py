from typing import Dict, Optional
import random
from llm import call_llm

class EmotionalIntelligence:
    BASIC_MOODS = [
        "Curious", "Frustrated", "Confident", "Stuck", "Low Energy", "Reflective"
    ]

    def __init__(self):
        # Initialize all moods to 0.0
        self.mood_vector: Dict[str, float] = {mood: 0.0 for mood in self.BASIC_MOODS}
        self.last_action_result: Optional[dict] = None

    def update_mood(self, mood: str, delta: float):
        print(f"[DEBUG] Updating mood '{mood}' by {delta}")
        if mood in self.mood_vector:
            self.mood_vector[mood] = max(0.0, self.mood_vector[mood] + delta)

    def decay_moods(self, decay: float = 0.05):
        print(f"[DEBUG] Decaying all moods by {decay}")
        for mood in self.mood_vector:
            self.mood_vector[mood] = max(0.0, self.mood_vector[mood] - decay)

    def process_action_result(self, action_result: dict):
        print(f"[DEBUG] Processing action result: {action_result}")
        print(f"[DEBUG] Mood vector before update: {self.mood_vector}")
        self.decay_moods()
        if action_result.get('new_topic') or action_result.get('fact_discovered'):
            self.update_mood('Curious', 0.3)
        if action_result.get('success'):
            self.update_mood('Confident', 0.2)
            self.update_mood('Frustrated', -0.1)
        if action_result.get('error'):
            self.update_mood('Frustrated', 0.3)
            self.update_mood('Confident', -0.1)
        if action_result.get('loop') or action_result.get('low_output_variance'):
            self.update_mood('Stuck', 0.3)
        if action_result.get('inactivity') or action_result.get('failure_streak'):
            self.update_mood('Low Energy', 0.3)
        if action_result.get('major_completion'):
            self.update_mood('Reflective', 0.5)
        print(f"[DEBUG] Mood vector after update: {self.mood_vector}")
        self.last_action_result = action_result

    def process_action_natural(self, action_output: str):
        print(f"[DEBUG] Processing natural action output: {action_output}")
        prompt = (
            "Given the following output from an AI action, classify which of these mood triggers are present: "
            "'success', 'error', 'new_topic', 'fact_discovered', 'major_completion', 'inactivity', 'loop', 'low_output_variance', 'failure_streak'.\n"
            "Return a JSON dictionary with these keys set to true or false.\n"
            f"Output: {action_output}"
        )
        llm_response = call_llm(prompt)
        print(f"[DEBUG] LLM response: {llm_response}")
        import json
        # Remove markdown code block if present
        resp = llm_response.strip()
        if resp.startswith("```"):
            resp = resp.split('\n', 1)[-1]
            if resp.startswith("json"):
                resp = resp[4:]
            resp = resp.strip("` \n")
        try:
            triggers = json.loads(resp)
            print(f"[DEBUG] Parsed triggers: {triggers}")
        except Exception as e:
            print(f"[DEBUG] Error parsing LLM response: {e}")
            triggers = {}
        self.process_action_result(triggers)

    def get_dominant_mood(self) -> str:
        # Return the mood with the highest intensity
        return max(self.mood_vector, key=lambda m: self.mood_vector[m])

    def get_mood_vector(self) -> Dict[str, float]:
        return dict(self.mood_vector)

    def influence_behavior(self) -> dict:
        """Return a dict of suggested behavior modifications based on dominant mood."""
        mood = self.get_dominant_mood()
        if mood == "Curious":
            return {"curiosity_trigger": True, "explore_more": True}
        elif mood == "Frustrated":
            return {"suggest_break": True, "try_simpler_task": True}
        elif mood == "Confident":
            return {"take_on_harder_challenges": True}
        elif mood == "Stuck":
            return {"force_lateral_thinking": True, "random_task": True}
        elif mood == "Low Energy":
            return {"pick_low_effort_task": True, "fun_task": True}
        elif mood == "Reflective":
            return {"activate_self_reflection": True}
        else:
            return {}

if __name__ == "__main__":
    ei = EmotionalIntelligence()
    # Simulate a sequence of natural language action outputs
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