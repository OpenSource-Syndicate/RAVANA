"""
Dynamic working memory parameters that adapt based on task complexity,
emotional state, cognitive load, and individual capacity.
"""
import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class MemoryItemState(Enum):
    """
    Represents the state of an item in working memory
    """
    ACTIVE = "active"
    DECAYING = "decaying"
    INHIBITED = "inhibited"
    REHEARSED = "rehearsed"


@dataclass
class WorkingMemoryState:
    """
    Represents the current state of working memory system with dynamic parameters
    """
    capacity: int = 7  # Miller's magical number
    decay_rate: float = 0.1  # How quickly items fade from working memory (per second)
    rehearsal_efficiency: float = 0.8  # Effectiveness of maintenance strategies (0-1)
    interference_resistance: float = 0.6  # Resistance to proactive/retroactive interference (0-1)
    emotional_valence_impact: float = 0.2  # How emotion affects capacity (0-1)
    stress_modulation: float = 0.1  # Stress impact on working memory (0-1)
    cognitive_load_factor: float = 0.3  # Impact of cognitive load (0-1)
    attention_focus_factor: float = 0.5  # Impact of attention focus (0-1)


class WorkingMemoryDynamics:
    """
    Implements dynamic working memory parameters that adapt based on:
    - Task complexity
    - Emotional state
    - Cognitive load
    - Individual capacity
    """
    def __init__(self):
        self.state = WorkingMemoryState()
        self.memory_items: List[Dict[str, Any]] = []  # Items currently in working memory
        self.item_history: List[Dict[str, Any]] = []  # Recently evicted items
        self.last_update_time = datetime.datetime.now()
        
        # Memory item dynamics
        self.max_item_age = 300  # Maximum time an item stays in working memory (seconds)
        self.rehearsal_cost = 0.1  # Cognitive cost of rehearsal (affects load)
        self.interference_threshold = 0.7  # Threshold for proactive interference
        
        # Emotional impact parameters
        self.emotional_impact = {
            'positive': {'capacity_boost': 0.1, 'decay_slowdown': 0.1},
            'negative': {'capacity_reduction': 0.15, 'decay_speedup': 0.2},
            'high_arousal': {'capacity_reduction': 0.2, 'interference_increase': 0.15}
        }

    def update_working_memory_dynamics(self, context: Dict[str, Any]):
        """
        Update working memory parameters based on current context and experience
        """
        current_time = datetime.datetime.now()
        time_delta = (current_time - self.last_update_time).total_seconds()
        self.last_update_time = current_time

        # Update state based on context
        if 'cognitive_load' in context:
            self.state.cognitive_load_factor = min(1.0, max(0.0, context['cognitive_load']))
            
        if 'attention_focus' in context:
            self.state.attention_focus_factor = min(1.0, max(0.0, context['attention_focus']))
            
        if 'emotional_state' in context:
            self._apply_emotional_impacts(context['emotional_state'])
            
        if 'stress_level' in context:
            self.state.stress_modulation = min(1.0, max(0.0, context['stress_level']))

        # Apply cognitive load effects to memory parameters
        self._apply_cognitive_load_effects()
        
        # Apply attention effects
        self._apply_attention_effects()
        
        # Process existing items (decay, maintenance, etc.)
        self._process_existing_items(time_delta)

        # Record state change for historical analysis
        state_snapshot = {
            'timestamp': current_time,
            'capacity': self.state.capacity,
            'decay_rate': self.state.decay_rate,
            'rehearsal_efficiency': self.state.rehearsal_efficiency,
            'interference_resistance': self.state.interference_resistance,
            'active_items': len(self.memory_items),
            'cognitive_load_factor': self.state.cognitive_load_factor
        }
        
        self.item_history.append({
            'timestamp': current_time,
            'state_snapshot': state_snapshot,
            'context': context
        })

        # Keep only recent history
        if len(self.item_history) > 100:
            self.item_history = self.item_history[-100:]

    def _apply_emotional_impacts(self, emotional_state: Dict[str, Any]):
        """
        Apply emotional state impacts to working memory parameters
        """
        valence = emotional_state.get('valence', 0.0)  # -1 to 1
        arousal = emotional_state.get('arousal', 0.5)  # 0 to 1
        
        # Reset state modifications
        self.state.capacity = 7
        self.state.decay_rate = 0.1
        
        # Apply valence effects
        if valence > 0.3:  # Positive emotional state
            self.state.capacity += 1
            self.state.decay_rate *= 0.9  # Slower decay
            self.state.emotional_valence_impact = 0.2
        elif valence < -0.3:  # Negative emotional state
            self.state.capacity = max(3, self.state.capacity - 1)
            self.state.decay_rate *= 1.2  # Faster decay
            self.state.emotional_valence_impact = -0.15

        # Apply arousal effects
        if arousal > 0.7:  # High arousal
            self.state.capacity = max(4, self.state.capacity - 2)
            self.state.decay_rate *= 1.3  # Much faster decay
            self.state.emotional_valence_impact = -0.3

    def _apply_cognitive_load_effects(self):
        """
        Apply cognitive load effects to working memory parameters
        """
        load_factor = self.state.cognitive_load_factor
        
        # Higher load reduces capacity
        capacity_reduction = int(load_factor * 2)  # Reduce by up to 2 items
        self.state.capacity = max(3, 7 - capacity_reduction)
        
        # Higher load increases decay rate
        self.state.decay_rate = min(0.5, 0.1 + (load_factor * 0.2))
        
        # Higher load reduces interference resistance
        self.state.interference_resistance = max(0.1, 0.6 - (load_factor * 0.3))

    def _apply_attention_effects(self):
        """
        Apply attention effects to working memory parameters
        """
        attention_factor = self.state.attention_focus_factor
        
        # Better attention improves rehearsal efficiency
        self.state.rehearsal_efficiency = 0.5 + (attention_factor * 0.3)
        
        # Better attention improves interference resistance
        self.state.interference_resistance = max(0.3, 0.3 + (attention_factor * 0.3))

    def _process_existing_items(self, time_delta: float):
        """
        Process existing items in working memory (decay, maintenance, etc.)
        """
        items_to_remove = []
        
        for i, item in enumerate(self.memory_items):
            # Update item age
            item['age'] = item.get('age', 0) + time_delta
            
            # Apply decay
            if item['state'] == MemoryItemState.DECAYING.value:
                item['activation'] = max(0.0, item['activation'] - 
                                       self.state.decay_rate * time_delta)
            elif item['state'] == MemoryItemState.REHEARSED.value:
                # Rehearsed items decay more slowly
                item['activation'] = max(0.0, item['activation'] - 
                                       self.state.decay_rate * 0.5 * time_delta)
            
            # Move to item history if activation too low or too old
            if item['activation'] < 0.1 or item['age'] > self.max_item_age:
                # Add to history before removal
                self.item_history.append({
                    'timestamp': datetime.datetime.now(),
                    'item': item.copy(),
                    'reason': 'activation_expired' if item['activation'] < 0.1 else 'age_expired'
                })
                
                items_to_remove.append(i)
        
        # Remove expired items (in reverse order to maintain indices)
        for i in reversed(items_to_remove):
            del self.memory_items[i]

    def store_items(self, items: List[Any], context: Dict[str, Any] = None) -> List[bool]:
        """
        Store items in working memory with dynamic allocation
        """
        if context:
            self.update_working_memory_dynamics(context)
        
        # Determine available capacity
        available_slots = max(0, self.state.capacity - len(self.memory_items))
        
        # Score items by priority/relevance
        scored_items = []
        for item in items:
            priority = self._calculate_item_priority(item, context)
            scored_items.append((item, priority))
        
        # Sort by priority
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Store items based on priority and available capacity
        results = []
        for i, (item, priority) in enumerate(scored_items):
            if i < available_slots:
                # Add item to working memory
                memory_item = {
                    'content': item,
                    'activation': 1.0,  # Full activation when first stored
                    'priority': priority,
                    'age': 0,
                    'state': MemoryItemState.ACTIVE.value,
                    'timestamp': datetime.datetime.now(),
                    'context': context or {}
                }
                self.memory_items.append(memory_item)
                results.append(True)
            else:
                # No capacity available
                results.append(False)
        
        return results

    def retrieve_items(self, query: str = None, max_items: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve items from working memory based on query and activation level
        """
        # Update dynamics for current state
        self.update_working_memory_dynamics({})
        
        # Filter items based on query if provided
        if query:
            filtered_items = []
            for item in self.memory_items:
                if query.lower() in str(item['content']).lower():
                    filtered_items.append(item)
        else:
            filtered_items = self.memory_items[:]
        
        # Sort by activation level (highest first)
        filtered_items.sort(key=lambda x: x['activation'], reverse=True)
        
        # Return top items
        max_items = max_items or len(filtered_items)
        return filtered_items[:max_items]

    def rehearse_items(self, item_indices: List[int]) -> List[bool]:
        """
        Rehearse specific items to maintain them in working memory
        """
        results = []
        for idx in item_indices:
            if 0 <= idx < len(self.memory_items):
                item = self.memory_items[idx]
                # Apply rehearsal effect
                item['activation'] = min(1.0, item['activation'] + self.state.rehearsal_efficiency * 0.3)
                item['state'] = MemoryItemState.REHEARSED.value
                results.append(True)
            else:
                results.append(False)
        
        return results

    def _calculate_item_priority(self, item: Any, context: Dict[str, Any]) -> float:
        """
        Calculate priority for storing an item in working memory
        """
        priority = 0.5  # Base priority
        
        # Add context-based priority boosts
        if context:
            # Boost if item is related to current goals
            if 'current_goals' in context:
                for goal in context['current_goals']:
                    if str(goal).lower() in str(item).lower():
                        priority += 0.3
            
            # Boost if item is related to current task
            if 'current_task' in context:
                if str(context['current_task']).lower() in str(item).lower():
                    priority += 0.2
                    
            # Boost if item is relevant to context
            if 'context_relevance' in context:
                priority += context['context_relevance'] * 0.1
        
        # Cap priority
        return min(1.0, priority)

    def get_working_memory_metrics(self) -> Dict[str, Any]:
        """
        Get current working memory metrics for monitoring and adjustment
        """
        avg_activation = sum(item['activation'] for item in self.memory_items) / len(self.memory_items) if self.memory_items else 0.0
        
        return {
            'current_capacity': len(self.memory_items),
            'max_capacity': self.state.capacity,
            'decay_rate': self.state.decay_rate,
            'rehearsal_efficiency': self.state.rehearsal_efficiency,
            'interference_resistance': self.state.interference_resistance,
            'avg_activation': avg_activation,
            'items_in_memory': len(self.memory_items),
            'cognitive_load_factor': self.state.cognitive_load_factor,
            'attention_focus_factor': self.state.attention_focus_factor
        }

    def clear(self):
        """
        Clear all items from working memory
        """
        # Move current items to history
        for item in self.memory_items:
            self.item_history.append({
                'timestamp': datetime.datetime.now(),
                'item': item.copy(),
                'reason': 'manual_clear'
            })
        
        self.memory_items = []