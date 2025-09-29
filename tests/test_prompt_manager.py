"""
Test cases for the enhanced PromptManager system
"""
from core.prompt_manager import PromptManager, PromptTemplate
import sys
import os
import unittest
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestPromptManager(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.prompt_manager = PromptManager()

    def test_prompt_manager_initialization(self):
        """Test that PromptManager initializes correctly with default templates."""
        # Check that default templates are registered
        templates = self.prompt_manager.repository.list_templates()
        self.assertIn("self_reflection", templates)
        self.assertIn("decision_making", templates)
        self.assertIn("experimentation", templates)
        self.assertIn("code_generation", templates)

    def test_prompt_template_creation(self):
        """Test creation and storage of prompt templates."""
        template_content = "Test prompt with {variable}"
        template = PromptTemplate(
            name="test_template",
            template=template_content,
            metadata={"category": "test", "description": "Test template"}
        )

        self.assertEqual(template.name, "test_template")
        self.assertEqual(template.template, template_content)
        self.assertEqual(template.metadata["category"], "test")

    def test_prompt_template_rendering(self):
        """Test rendering of prompt templates with context."""
        template_content = "Hello {name}, you are {age} years old."
        template = PromptTemplate(
            name="greeting_template",
            template=template_content,
            metadata={"category": "test"}
        )

        context = {"name": "Alice", "age": 30}
        rendered = template.render(context)
        expected = "Hello Alice, you are 30 years old."
        self.assertEqual(rendered, expected)

    def test_get_prompt_by_name(self):
        """Test retrieving and rendering prompts by name."""
        context = {
            "agent_name": "RAVANA-TEST",
            "task_summary": "Testing the system",
            "outcome": "Test completed successfully",
            "current_mood": "curious",
            "related_memories": "Previous test results"
        }

        prompt = self.prompt_manager.get_prompt("self_reflection", context)
        self.assertIn("[ROLE DEFINITION]", prompt)
        self.assertIn("RAVANA-TEST", prompt)
        self.assertIn("[CONTEXT]", prompt)
        self.assertIn("[TASK INSTRUCTIONS]", prompt)
        # The mood adaptation is added during post-processing, not in the template itself
        # so we won't check for it here

    def test_prompt_validation(self):
        """Test prompt validation functionality."""
        valid_prompt = """
[ROLE DEFINITION]
You are a test agent.

[CONTEXT]
This is a test context.

[TASK INSTRUCTIONS]
Perform a test task.
"""

        invalid_prompt = "This is too short"

        self.assertTrue(self.prompt_manager.validate_prompt(valid_prompt))
        self.assertFalse(self.prompt_manager.validate_prompt(invalid_prompt))

    def test_register_new_template(self):
        """Test registering a new prompt template."""
        template_content = "New template with {parameter}"
        self.prompt_manager.register_prompt_template(
            "new_test_template",
            template_content,
            {"category": "test", "version": "1.0"}
        )

        # Retrieve and verify the template
        templates = self.prompt_manager.repository.list_templates()
        self.assertIn("new_test_template", templates)

        prompt = self.prompt_manager.get_prompt(
            "new_test_template", {"parameter": "value"})
        self.assertIn("New template with value", prompt)

    def test_enhanced_reflection_prompt(self):
        """Test the enhanced self-reflection prompt structure."""
        context = {
            "agent_name": "RAVANA-TEST",
            "task_summary": "Completed a complex reasoning task",
            "outcome": "Successfully solved the problem",
            "current_mood": "reflective",
            "related_memories": "Previous similar tasks"
        }

        prompt = self.prompt_manager.get_prompt("self_reflection", context)

        # Check for required sections
        self.assertIn("[ROLE DEFINITION]", prompt)
        self.assertIn("[CONTEXT]", prompt)
        self.assertIn("[TASK INSTRUCTIONS]", prompt)
        self.assertIn("[REASONING FRAMEWORK]", prompt)
        self.assertIn("[OUTPUT REQUIREMENTS]", prompt)
        self.assertIn("[SAFETY CONSTRAINTS]", prompt)
        # The mood adaptation is added during post-processing, not in the template itself
        # so we won't check for it here

        # Check for context variables
        self.assertIn("RAVANA-TEST", prompt)
        self.assertIn("Completed a complex reasoning task", prompt)
        self.assertIn("Successfully solved the problem", prompt)
        self.assertIn("reflective", prompt)

    def test_enhanced_decision_prompt(self):
        """Test the enhanced decision-making prompt structure."""
        context = {
            "agent_name": "RAVANA-DECISION",
            "current_situation": "Need to choose next action",
            "active_goals": json.dumps([{"id": 1, "description": "Test goal"}]),
            "current_hypotheses": json.dumps(["Hypothesis 1"]),
            "action_list": json.dumps(["action1", "action2"]),
            "current_mood": "focused",  # Add the missing context variable
            "safety_constraints": ["Follow ethical guidelines", "Avoid harmful actions"]
        }

        prompt = self.prompt_manager.get_prompt("decision_making", context)

        # Check for required sections
        self.assertIn("[ROLE DEFINITION]", prompt)
        self.assertIn("[CONTEXT]", prompt)
        self.assertIn("[TASK INSTRUCTIONS]", prompt)
        self.assertIn("[REASONING FRAMEWORK]", prompt)
        self.assertIn("[OUTPUT REQUIREMENTS]", prompt)
        self.assertIn("[SAFETY CONSTRAINTS]", prompt)
        self.assertIn("RAVANA-DECISION", prompt)
        self.assertIn("Need to choose next action", prompt)
        self.assertIn("focused", prompt)  # Check for the mood context variable

    def test_enhanced_experimentation_prompt(self):
        """Test the enhanced experimentation prompt structure."""
        context = {
            "agent_name": "RAVANA-EXPERIMENT",
            "experiment_objective": "Test quantum tunneling",
            "relevant_theory": "Quantum mechanics",
            "resource_constraints": "Limited computational resources",
            "safety_requirements": "Standard safety protocols"
        }

        prompt = self.prompt_manager.get_prompt("experimentation", context)

        # Check for required sections
        self.assertIn("[ROLE DEFINITION]", prompt)
        self.assertIn("[CONTEXT]", prompt)
        self.assertIn("[TASK INSTRUCTIONS]", prompt)
        self.assertIn("[REASONING FRAMEWORK]", prompt)
        self.assertIn("[OUTPUT REQUIREMENTS]", prompt)
        self.assertIn("[SAFETY CONSTRAINTS]", prompt)
        self.assertIn("RAVANA-EXPERIMENT", prompt)
        self.assertIn("Test quantum tunneling", prompt)

    def test_enhanced_coding_prompt(self):
        """Test the enhanced coding prompt structure."""
        context = {
            "agent_name": "RAVANA-CODER",
            "task_description": "Implement a sorting algorithm",
            "requirements": "Must be efficient and well-documented",
            "constraints": "Use Python only",
            "target_environment": "Standard Python 3.9"
        }

        prompt = self.prompt_manager.get_prompt("code_generation", context)

        # Check for required sections
        self.assertIn("[ROLE DEFINITION]", prompt)
        self.assertIn("[CONTEXT]", prompt)
        self.assertIn("[TASK INSTRUCTIONS]", prompt)
        self.assertIn("[REASONING FRAMEWORK]", prompt)
        self.assertIn("[OUTPUT REQUIREMENTS]", prompt)
        self.assertIn("[SAFETY CONSTRAINTS]", prompt)
        self.assertIn("RAVANA-CODER", prompt)
        self.assertIn("Implement a sorting algorithm", prompt)


if __name__ == '__main__':
    unittest.main()
