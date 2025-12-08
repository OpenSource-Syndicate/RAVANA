"""
Dynamic attention mechanisms that adapt based on cognitive load, emotional state,
task relevance, and fatigue levels.
"""
import datetime
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class AttentionState:
    """
    Represents the current state of attention system with dynamic parameters
    """
    focus_strength: float = 0.5  # Current ability to maintain focus (0-1)
    focus_duration: float = 120  # How long focus can be maintained (seconds)
    distraction_resistance: float = 0.7  # Resistance to irrelevant stimuli (0-1)
    attention_span: int = 7  # Average number of items that can be attended
    cognitive_load: float = 0.0  # Current cognitive burden (0-1)
    fatigue_level: float = 0.0  # Mental fatigue accumulation (0-1)
    emotional_interference: float = 0.0  # How emotions affect attention (0-1)
    attention_decay_rate: float = 0.01  # Rate at which attention degrades over time
    recovery_rate: float = 0.05  # Rate at which attention recovers after rest


class AttentionDynamics:
    """
    Implements dynamic attention mechanisms that adapt based on:
    - Cognitive load
    - Emotional state
    - Task relevance
    - Fatigue levels
    """
    def __init__(self):
        self.state = AttentionState()
        self.focus_history: List[Dict[str, Any]] = []
        self.last_update_time = datetime.datetime.now()
        
        # Attention decay and recovery parameters
        self.max_fatigue = 1.0
        self.fatigue_recovery_rate = 0.02  # per second when idle
        self.fatigue_accumulation_rate = 0.05  # per second when actively focused
        
        # Emotional state effects
        self.emotional_state_influence = {
            'high_arousal': {'focus_strength': -0.1, 'distraction_resistance': -0.2},
            'low_arousal': {'focus_strength': -0.2, 'attention_span': -1},
            'positive_valence': {'focus_strength': 0.1, 'distraction_resistance': 0.1},
            'negative_valence': {'focus_strength': -0.15, 'attention_span': -1}
        }

    def update_attention_dynamics(self, context: Dict[str, Any]):
        """
        Update attention parameters based on current context and experience
        """
        current_time = datetime.datetime.now()
        time_delta = (current_time - self.last_update_time).total_seconds()
        self.last_update_time = current_time

        # Update fatigue based on time and activity
        if context.get('is_actively_focusing', False):
            self.state.fatigue_level = min(self.state.fatigue_level + 
                                          self.fatigue_accumulation_rate * time_delta, 
                                          self.max_fatigue)
        else:
            self.state.fatigue_level = max(self.state.fatigue_level - 
                                          self.fatigue_recovery_rate * time_delta, 
                                          0.0)

        # Update cognitive load from context
        if 'cognitive_load' in context:
            self.state.cognitive_load = context['cognitive_load']

        # Apply emotional state influences
        if 'emotional_state' in context:
            emotional_state = context['emotional_state']
            self._apply_emotional_influences(emotional_state)

        # Adjust attention parameters based on fatigue and load
        self._adjust_parameters_for_fatigue()
        self._adjust_parameters_for_cognitive_load()

        # Update focus strength based on all factors
        self._update_focus_strength()

        # Record state change for historical analysis
        self.focus_history.append({
            'timestamp': current_time,
            'state_snapshot': self.state.__dict__.copy(),
            'context': context
        })

        # Keep only recent history
        if len(self.focus_history) > 100:
            self.focus_history = self.focus_history[-100:]

    def _apply_emotional_influences(self, emotional_state: Dict[str, Any]):
        """
        Apply emotional state influences to attention parameters
        """
        valence = emotional_state.get('valence', 0.0)  # -1 to 1
        arousal = emotional_state.get('arousal', 0.5)  # 0 to 1
        
        # Determine emotional categories
        emotional_categories = []
        
        if arousal > 0.7:
            emotional_categories.append('high_arousal')
        elif arousal < 0.3:
            emotional_categories.append('low_arousal')
            
        if valence > 0.3:
            emotional_categories.append('positive_valence')
        elif valence < -0.3:
            emotional_categories.append('negative_valence')

        # Apply influences from each category
        for category in emotional_categories:
            if category in self.emotional_state_influence:
                influences = self.emotional_state_influence[category]
                for param, effect in influences.items():
                    if hasattr(self.state, param):
                        current_value = getattr(self.state, param)
                        new_value = current_value + effect
                        # Ensure bounds
                        if param in ['focus_strength', 'distraction_resistance']:
                            new_value = max(0.0, min(1.0, new_value))
                        elif param == 'attention_span':
                            new_value = max(1, min(15, new_value))  # Reasonable bounds
                        setattr(self.state, param, new_value)

    def _adjust_parameters_for_fatigue(self):
        """
        Adjust attention parameters based on fatigue level
        """
        fatigue_factor = self.state.fatigue_level
        
        # Higher fatigue reduces focus strength
        self.state.focus_strength = max(0.1, self.state.focus_strength * (1 - fatigue_factor * 0.7))
        
        # Higher fatigue reduces distraction resistance
        self.state.distraction_resistance = max(0.1, self.state.distraction_resistance * (1 - fatigue_factor * 0.5))
        
        # Higher fatigue reduces attention span
        self.state.attention_span = max(2, int(self.state.attention_span * (1 - fatigue_factor * 0.3)))
        
        # Higher fatigue reduces focus duration
        self.state.focus_duration = max(30, self.state.focus_duration * (1 - fatigue_factor * 0.6))

    def _adjust_parameters_for_cognitive_load(self):
        """
        Adjust attention parameters based on cognitive load
        """
        load_factor = self.state.cognitive_load
        
        # Higher cognitive load reduces focus strength
        self.state.focus_strength = max(0.1, self.state.focus_strength * (1 - load_factor * 0.5))
        
        # Higher cognitive load reduces distraction resistance
        self.state.distraction_resistance = max(0.1, self.state.distraction_resistance * (1 - load_factor * 0.4))
        
        # Higher cognitive load reduces attention span
        self.state.attention_span = max(3, int(self.state.attention_span * (1 - load_factor * 0.2)))

    def _update_focus_strength(self):
        """
        Update focus strength based on all contributing factors
        """
        # Base focus strength is modified by fatigue and cognitive load
        fatigue_effect = self.state.fatigue_level * -0.5
        load_effect = self.state.cognitive_load * -0.3
        emotional_effect = self.state.emotional_interference * -0.2
        
        # Calculate composite adjustment
        adjustment = fatigue_effect + load_effect + emotional_effect
        self.state.focus_strength = max(0.05, min(1.0, self.state.focus_strength + adjustment))

    def allocate_attention(self, items: List[Any], context: Dict[str, Any] = None) -> List[Any]:
        """
        Allocate attention dynamically to relevant items based on current state
        """
        if not items:
            return []
            
        # Update dynamics with context
        if context:
            self.update_attention_dynamics(context)
        
        # Calculate attention capacity
        available_capacity = max(1, int(self.state.attention_span * (1 - self.state.cognitive_load * 0.3)))
        
        # Score items by relevance (this would use more sophisticated logic in real implementation)
        scored_items = []
        for item in items:
            relevance_score = self._calculate_item_relevance(item, context)
            scored_items.append((item, relevance_score))
        
        # Sort by relevance and return top items based on capacity
        scored_items.sort(key=lambda x: x[1], reverse=True)
        attended_items = [item for item, score in scored_items[:available_capacity]]
        
        return attended_items

    def _calculate_item_relevance(self, item: Any, context: Dict[str, Any]) -> float:
        """
        Calculate how relevant an item is to current context
        """
        relevance = 0.5  # Base relevance
        
        # Add relevance based on context
        if context:
            # Check if item is related to current goals
            if 'current_goals' in context:
                for goal in context['current_goals']:
                    if str(goal).lower() in str(item).lower():
                        relevance += 0.3
            
            # Check if item is related to current task
            if 'current_task' in context:
                if str(context['current_task']).lower() in str(item).lower():
                    relevance += 0.2
                    
            # Check emotional relevance
            if 'emotional_significance' in context:
                relevance += context['emotional_significance'] * 0.1
        
        # Cap relevance score
        return min(1.0, relevance)

    def get_attention_metrics(self) -> Dict[str, Any]:
        """
        Get current attention metrics for monitoring and adjustment
        """
        return {
            'focus_strength': self.state.focus_strength,
            'attention_span': self.state.attention_span,
            'distraction_resistance': self.state.distraction_resistance,
            'fatigue_level': self.state.fatigue_level,
            'cognitive_load': self.state.cognitive_load,
            'focus_duration': self.state.focus_duration,
            'estimated_remaining_focus': self.state.focus_duration * (1 - self.state.fatigue_level)
        }