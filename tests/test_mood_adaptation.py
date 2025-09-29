"""
Test cases for the mood adaptation functionality in the PromptManager
"""
from core.prompt_manager import PromptManager
import sys
import os
import unittest

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestMoodAdaptation(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.prompt_manager = PromptManager()

    def test_curious_mood_adaptation(self):
        """Test mood adaptation for curious emotional state."""
        context = {
            "agent_name": "RAVANA-TEST",
            "task_summary": "Exploring new concepts",
            "outcome": "Discovered interesting patterns",
            "current_mood": {"primary_emotion": "curious"},
            "related_memories": "Previous exploration results"
        }

        # Get a prompt and apply post-processing manually to check mood adaptation
        template = self.prompt_manager.repository.retrieve_template(
            "self_reflection")
        rendered_prompt = template.render(context)
        adapted_prompt = self.prompt_manager._post_process_prompt(
            rendered_prompt, context)

        self.assertIn("[MOOD ADAPTATION]", adapted_prompt)
        self.assertIn(
            "Embrace exploration and creative thinking", adapted_prompt)

    def test_focused_mood_adaptation(self):
        """Test mood adaptation for focused emotional state."""
        context = {
            "agent_name": "RAVANA-TEST",
            "task_summary": "Working on detailed analysis",
            "outcome": "Completed precise calculations",
            "current_mood": {"primary_emotion": "focused"},
            "related_memories": "Previous analytical tasks"
        }

        # Get a prompt and apply post-processing manually to check mood adaptation
        template = self.prompt_manager.repository.retrieve_template(
            "self_reflection")
        rendered_prompt = template.render(context)
        adapted_prompt = self.prompt_manager._post_process_prompt(
            rendered_prompt, context)

        self.assertIn("[MOOD ADAPTATION]", adapted_prompt)
        self.assertIn("Maintain precision and detailed analysis",
                      adapted_prompt)

    def test_reflective_mood_adaptation(self):
        """Test mood adaptation for reflective emotional state."""
        context = {
            "agent_name": "RAVANA-TEST",
            "task_summary": "Completed a major task",
            "outcome": "Mixed results requiring analysis",
            "current_mood": {"primary_emotion": "reflective"},
            "related_memories": "Previous task outcomes"
        }

        # Get a prompt and apply post-processing manually to check mood adaptation
        template = self.prompt_manager.repository.retrieve_template(
            "self_reflection")
        rendered_prompt = template.render(context)
        adapted_prompt = self.prompt_manager._post_process_prompt(
            rendered_prompt, context)

        self.assertIn("[MOOD ADAPTATION]", adapted_prompt)
        self.assertIn(
            "Engage in introspection and learning from experience", adapted_prompt)

    def test_cautious_mood_adaptation(self):
        """Test mood adaptation for cautious emotional state."""
        context = {
            "agent_name": "RAVANA-TEST",
            "task_summary": "Handling sensitive information",
            "outcome": "Processed with care",
            "current_mood": {"primary_emotion": "cautious"},
            "related_memories": "Previous safety protocols"
        }

        # Get a prompt and apply post-processing manually to check mood adaptation
        template = self.prompt_manager.repository.retrieve_template(
            "self_reflection")
        rendered_prompt = template.render(context)
        adapted_prompt = self.prompt_manager._post_process_prompt(
            rendered_prompt, context)

        self.assertIn("[MOOD ADAPTATION]", adapted_prompt)
        self.assertIn(
            "Prioritize risk assessment and safety considerations", adapted_prompt)

    def test_unknown_mood_no_adaptation(self):
        """Test that unknown moods don't cause errors."""
        context = {
            "agent_name": "RAVANA-TEST",
            "task_summary": "Normal task",
            "outcome": "Standard outcome",
            "current_mood": {"primary_emotion": "unknown_emotion"},
            "related_memories": "Standard memories"
        }

        # Get a prompt and apply post-processing manually
        template = self.prompt_manager.repository.retrieve_template(
            "self_reflection")
        rendered_prompt = template.render(context)
        adapted_prompt = self.prompt_manager._post_process_prompt(
            rendered_prompt, context)

        # Should not contain mood adaptation for unknown emotions
        self.assertNotIn("[MOOD ADAPTATION]", adapted_prompt)

    def test_no_mood_context(self):
        """Test that missing mood context doesn't cause errors."""
        context = {
            "agent_name": "RAVANA-TEST",
            "task_summary": "Normal task",
            "outcome": "Standard outcome",
            "related_memories": "Standard memories"
            # No current_mood provided
        }

        # Get a prompt and apply post-processing manually
        template = self.prompt_manager.repository.retrieve_template(
            "self_reflection")
        rendered_prompt = template.render(context)
        adapted_prompt = self.prompt_manager._post_process_prompt(
            rendered_prompt, context)

        # Should not contain mood adaptation when no mood is provided
        self.assertNotIn("[MOOD ADAPTATION]", adapted_prompt)


if __name__ == '__main__':
    unittest.main()
