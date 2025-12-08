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

        # Consciousness-related attributes
        self.attention_focus: List[str] = []  # Currently attended items
        self.consciousness_metrics: Dict[str, Any] = {
            'awareness_level': 0,
            'self_knowledge_update_count': 0,
            'experience_coherence': 0
        }
        self.introspection_history: List[Dict[str, Any]] = []
        self.phenomenal_character: str = ""  # Subjective experience quality

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
            f"Search Results: {len(self.search_results)}\n"
            f"Attention Focus: {self.attention_focus}\n"
            f"Consciousness Metrics: {self.consciousness_metrics}\n"
            f"Phenomenal Character: {self.phenomenal_character}\n"
            f"Introspection History: {len(self.introspection_history)} items"
        )
        return summary
