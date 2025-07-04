import logging
from core.state import SharedState
from core.config import Config

logger = logging.getLogger(__name__)

class ReflectionModule:
    """
    A module for self-reflection, allowing the AGI to analyze its own performance
    and generate hypotheses for improvement.
    """
    def generate_hypothesis(self, shared_state: SharedState) -> str:
        """
        Analyzes the agent's recent performance and formulates a testable hypothesis.
        """
        # Example: Analyze mood and decision quality
        recent_moods = shared_state.mood_history
        if len(recent_moods) < 10:
            return None  # Not enough data

        # A simple heuristic: if mood has been consistently negative, hypothesize that it affects performance.
        negative_mood_count = 0
        for mood_vector in recent_moods:
            if not mood_vector:  # Skip if empty
                continue
            # Find the dominant mood in the vector
            dominant_mood = max(mood_vector, key=mood_vector.get)
            if dominant_mood in Config.NEGATIVE_MOODS:
                negative_mood_count += 1
        
        if negative_mood_count > 5:
            return "I hypothesize that my plans are less effective when I am in a negative mood."

        return None 