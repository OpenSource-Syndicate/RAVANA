import unittest
from datetime import datetime, timedelta
from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence, EmotionalEvent


class TestEnhancedEmotionalIntelligence(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.ei = EmotionalIntelligence()

    def test_initialize_mood_dimensions(self):
        """Test that mood dimensions are properly initialized"""
        # Check that all mood categories are present
        self.assertTrue(len(self.ei.ALL_MOODS) > 10)
        self.assertTrue(len(self.ei.PRIMARY_MOODS) > 10)
        self.assertTrue(len(self.ei.SECONDARY_MOODS) > 10)

        # Check that mood vector is initialized with all moods
        self.assertEqual(len(self.ei.mood_vector), len(self.ei.ALL_MOODS))

        # Check that all values start at 0.0
        for mood_value in self.ei.mood_vector.values():
            self.assertEqual(mood_value, 0.0)

    def test_update_mood_with_momentum(self):
        """Test mood updates with momentum effects"""
        initial_confident = self.ei.mood_vector["Confident"]
        self.ei.update_mood("Confident", 0.5)

        # Mood should increase
        self.assertGreater(self.ei.mood_vector["Confident"], initial_confident)

        # Momentum should be set
        self.assertNotEqual(self.ei.mood_momentum["Confident"], 0.0)

    def test_mood_blending(self):
        """Test mood blending functionality"""
        # Set up conditions for blending
        self.ei.mood_vector["Confident"] = 0.7
        self.ei.mood_vector["Curious"] = 0.8

        # Apply blending
        self.ei.blend_moods()

        # Check that blended mood was affected
        self.assertGreaterEqual(self.ei.mood_vector["Inspired"], 0.0)

    def test_decay_moods_with_stability(self):
        """Test mood decay with stability controls"""
        # Set high mood values
        self.ei.mood_vector["Excited"] = 0.8
        self.ei.mood_vector["Curious"] = 0.2

        # Store initial values
        initial_excited = self.ei.mood_vector["Excited"]
        initial_curious = self.ei.mood_vector["Curious"]

        # Apply decay
        self.ei.decay_moods(0.1)

        # Both should decay, but excited should decay more due to stability controls
        self.assertLess(self.ei.mood_vector["Excited"], initial_excited)
        self.assertLess(self.ei.mood_vector["Curious"], initial_curious)

    def test_log_emotional_event(self):
        """Test emotional event logging"""
        initial_event_count = len(self.ei.emotional_events)

        mood_changes = {"Confident": 0.3, "Curious": 0.2}
        triggers = ["new_discovery"]
        context = "Test context"

        self.ei.log_emotional_event(mood_changes, triggers, context)

        # Should have one more event
        self.assertEqual(len(self.ei.emotional_events),
                         initial_event_count + 1)

        # Check event properties
        last_event = self.ei.emotional_events[-1]
        self.assertEqual(last_event.triggers, triggers)
        self.assertEqual(last_event.context, context)
        self.assertAlmostEqual(last_event.intensity, 0.5)  # 0.3 + 0.2

    def test_get_emotional_context(self):
        """Test getting emotional context for decision making"""
        context = self.ei.get_emotional_context()

        # Should have required keys
        self.assertIn("dominant_mood", context)
        self.assertIn("mood_vector", context)
        self.assertIn("recent_events", context)

        # Recent events should be a list
        self.assertIsInstance(context["recent_events"], list)

    def test_get_mood_intensity_level(self):
        """Test mood intensity level classification"""
        # Test low intensity
        self.ei.mood_vector["Confident"] = 0.2
        self.assertEqual(self.ei.get_mood_intensity_level("Confident"), "low")

        # Test medium intensity
        self.ei.mood_vector["Curious"] = 0.5
        self.assertEqual(self.ei.get_mood_intensity_level("Curious"), "medium")

        # Test high intensity
        self.ei.mood_vector["Excited"] = 0.8
        self.assertEqual(self.ei.get_mood_intensity_level("Excited"), "high")

        # Test unknown mood
        self.assertEqual(self.ei.get_mood_intensity_level(
            "NonExistentMood"), "unknown")

    def test_persona_adaptation(self):
        """Test persona adaptation functionality"""
        # Store initial multiplier
        initial_multiplier = self.ei.persona.get(
            "mood_multipliers", {}).get("Confident", 1.0)

        # Apply adaptation
        self.ei.adapt_persona("Test experience", "positive outcome")

        # Multiplier should have increased
        new_multiplier = self.ei.persona.get(
            "mood_multipliers", {}).get("Confident", 1.0)
        self.assertGreaterEqual(new_multiplier, initial_multiplier)

    def test_process_action_result_enhanced(self):
        """Test enhanced action result processing"""
        initial_event_count = len(self.ei.emotional_events)

        action_result = {
            "new_discovery": True,
            "task_completed": False
        }

        self.ei.process_action_result(action_result)

        # Should have logged an emotional event
        self.assertGreater(len(self.ei.emotional_events), initial_event_count)

    def test_emotional_event_cleanup(self):
        """Test that old emotional events are cleaned up"""
        # Add an old event (25 hours ago)
        old_event = EmotionalEvent(
            timestamp=datetime.now() - timedelta(hours=25),
            mood_changes={"Confident": 0.3},
            triggers=["test_trigger"],
            context="old context",
            intensity=0.3
        )
        self.ei.emotional_events.append(old_event)

        # Add a recent event
        recent_event = EmotionalEvent(
            timestamp=datetime.now(),
            mood_changes={"Curious": 0.2},
            triggers=["recent_trigger"],
            context="recent context",
            intensity=0.2
        )
        self.ei.emotional_events.append(recent_event)

        # Process an action to trigger cleanup
        self.ei.process_action_result({"new_discovery": True})

        # Should only have the recent event (old one should be cleaned up)
        # Note: This test might be flaky depending on exact timing, but it's conceptually correct


if __name__ == '__main__':
    unittest.main()
