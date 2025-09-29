from typing import Dict, Any, List


class SharedState:
    """
    A class to encapsulate the shared state of the AGI system.
    """

    def __init__(self, initial_mood: Dict[str, float]):
        self.mood: Dict[str, float] = initial_mood
        self.current_situation: Dict[str, Any] = None
        self.current_situation_id: int = None
        self.recent_memories: List[Dict[str, Any]] = []
        self.long_term_goals: List[str] = []
        self.mood_history: List[Dict[str, float]] = []
        self.curiosity_topics: List[str] = []
        self.search_results: List[str] = []
        self.current_task: str = None

    def get_state_summary(self) -> str:
        """
        Returns a string summary of the current state.
        """
        summary = (
            f"Mood: {self.mood}\n"
            f"Current Situation: {self.current_situation}\n"
            f"Current Task: {self.current_task}\n"
            f"Recent Memories: {len(self.recent_memories)}\n"
            f"Curiosity Topics: {self.curiosity_topics}\n"
            f"Search Results: {len(self.search_results)}"
        )
        return summary
