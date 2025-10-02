"""Tests for emotional intelligence functionality."""

import pytest
from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence


class TestEmotionalIntelligence:
    """Test emotional intelligence module."""

    @pytest.fixture
    def ei_system(self):
        """Create emotional intelligence system for testing."""
        return EmotionalIntelligence()

    def test_ei_initialization(self, ei_system):
        """Test emotional intelligence initializes correctly."""
        assert ei_system is not None
        assert ei_system.mood_vector is not None
        assert isinstance(ei_system.mood_vector, dict)
        assert len(ei_system.mood_vector) > 0

    def test_update_mood(self, ei_system):
        """Test updating mood values."""
        initial_value = ei_system.mood_vector.get('Confident', 0.0)
        ei_system.update_mood('Confident', 0.2)
        new_value = ei_system.mood_vector['Confident']
        
        assert new_value != initial_value

    def test_get_dominant_mood(self, ei_system):
        """Test getting dominant mood."""
        # Set specific moods
        ei_system.update_mood('Curious', 0.8)
        ei_system.update_mood('Confident', 0.3)
        
        dominant = ei_system.get_dominant_mood()
        
        assert dominant is not None
        assert isinstance(dominant, str)

    def test_mood_decay(self, ei_system):
        """Test mood decay functionality."""
        ei_system.update_mood('Excited', 0.8)
        initial_value = ei_system.mood_vector['Excited']
        
        ei_system.decay_moods(decay=0.1)
        new_value = ei_system.mood_vector['Excited']
        
        assert new_value < initial_value

    def test_blend_moods(self, ei_system):
        """Test mood blending."""
        ei_system.update_mood('Confident', 0.7)
        ei_system.update_mood('Curious', 0.7)
        
        ei_system.blend_moods()
        
        # Check if blended mood exists
        assert 'Inspired' in ei_system.mood_vector

    def test_process_action_natural(self, ei_system):
        """Test processing natural language action output."""
        action_output = "The agent discovered a new approach to solving the problem."
        
        initial_mood = ei_system.get_mood_vector().copy()
        ei_system.process_action_natural(action_output)
        new_mood = ei_system.get_mood_vector()
        
        # Mood should have changed
        assert initial_mood != new_mood

    def test_influence_behavior(self, ei_system):
        """Test behavior influence based on mood."""
        ei_system.update_mood('Frustrated', 0.8)
        
        behavior = ei_system.influence_behavior()
        
        assert behavior is not None
        assert isinstance(behavior, dict)

    def test_set_persona(self, ei_system):
        """Test setting emotional persona."""
        ei_system.set_persona("Optimistic")
        
        assert ei_system.persona is not None

    def test_log_emotional_event(self, ei_system):
        """Test logging emotional events."""
        mood_changes = {'Confident': 0.2, 'Curious': 0.1}
        triggers = ['success', 'learning']
        context = "Successfully completed a task"
        
        ei_system.log_emotional_event(mood_changes, triggers, context)
        
        assert len(ei_system.emotional_events) > 0

    def test_get_emotional_context(self, ei_system):
        """Test getting emotional context."""
        ei_system.update_mood('Curious', 0.7)
        
        context = ei_system.get_emotional_context()
        
        assert context is not None
        assert 'dominant_mood' in context
        assert 'mood_vector' in context

    def test_learn_from_emotional_outcomes(self, ei_system):
        """Test learning from emotional outcomes."""
        action_result = {
            'result': 'success',
            'mood_before': {'Confident': 0.5},
            'mood_after': {'Confident': 0.7}
        }
        
        ei_system.learn_from_emotional_outcomes(action_result)
        
        # Should not raise exception
        assert True
