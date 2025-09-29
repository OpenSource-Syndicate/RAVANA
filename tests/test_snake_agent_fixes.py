#!/usr/bin/env python3
"""
Test Suite for Snake Agent Bug Fixes and Unlimited Token Generation

This test validates:
1. AttributeError fix in _update_mood method
2. Unlimited token generation functionality
3. State validation and error handling
4. Configuration updates
"""

from core.snake_llm import SnakeCodingLLM, SnakeReasoningLLM, SnakeConfigValidator
from core.config import Config
from core.snake_agent import SnakeAgent, SnakeAgentState, FileSystemMonitor
import pytest
import asyncio
import unittest
from unittest.mock import Mock, AsyncMock, patch
import json
import tempfile
import os
from pathlib import Path

# Import the fixed modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))


class TestSnakeAgentAttributeFix(unittest.TestCase):
    """Test the AttributeError fix in Snake Agent"""

    def setUp(self):
        """Set up test environment"""
        self.mock_agi_system = Mock()
        self.mock_agi_system.workspace_path = tempfile.mkdtemp()

    def test_mood_update_with_valid_state(self):
        """Test that _update_mood correctly accesses state.experiment_success_rate"""
        agent = SnakeAgent(self.mock_agi_system)

        # Set various success rates and test mood updates
        test_cases = [
            (0.9, "confident"),
            (0.7, "curious"),
            (0.3, "cautious"),
            (0.1, "frustrated")
        ]

        for success_rate, expected_mood in test_cases:
            agent.state.experiment_success_rate = success_rate

            # Should not raise AttributeError
            try:
                agent._update_mood()
                self.assertEqual(agent.state.mood, expected_mood)
                print(
                    f"✓ Success rate {success_rate} → mood '{expected_mood}'")
            except AttributeError as e:
                self.fail(
                    f"AttributeError still occurs with success rate {success_rate}: {e}")

    def test_mood_update_with_missing_attribute(self):
        """Test mood update when experiment_success_rate is missing"""
        agent = SnakeAgent(self.mock_agi_system)

        # Remove the attribute to simulate the original bug scenario
        if hasattr(agent.state, 'experiment_success_rate'):
            delattr(agent.state, 'experiment_success_rate')

        # Should not crash and should initialize the attribute
        agent._update_mood()

        # Should have initialized the attribute to 0.0
        self.assertTrue(hasattr(agent.state, 'experiment_success_rate'))
        self.assertEqual(agent.state.experiment_success_rate, 0.0)
        self.assertEqual(agent.state.mood, "frustrated")  # 0.0 < 0.2
        print("✓ Missing attribute handled gracefully")

    def test_state_validation(self):
        """Test state validation functionality"""
        agent = SnakeAgent(self.mock_agi_system)

        # Test with valid state
        self.assertTrue(agent._validate_state())
        print("✓ State validation works with valid state")

        # Test with missing attribute
        delattr(agent.state, 'mood')
        self.assertFalse(agent._validate_state())

        # Test state reinitialization
        agent._reinitialize_state()
        self.assertTrue(agent._validate_state())
        self.assertEqual(agent.state.mood, "curious")
        print("✓ State reinitialization works correctly")

    def test_experiment_success_rate_update(self):
        """Test that experiment success rate updates correctly"""
        agent = SnakeAgent(self.mock_agi_system)
        initial_rate = agent.state.experiment_success_rate

        # Test successful experiment
        agent._update_experiment_success_rate(True)
        self.assertGreater(agent.state.experiment_success_rate, initial_rate)

        # Test failed experiment
        current_rate = agent.state.experiment_success_rate
        agent._update_experiment_success_rate(False)
        self.assertLess(agent.state.experiment_success_rate, current_rate)

        print("✓ Experiment success rate updates correctly")


class TestUnlimitedTokenGeneration(unittest.TestCase):
    """Test unlimited token generation functionality"""

    def setUp(self):
        """Set up test environment"""
        # Mock Ollama server responses
        self.mock_response = {
            "response": "This is a long response that would exceed normal token limits...",
            "model": "test-model",
            "created_at": "2025-08-22T12:00:00Z",
            "done": True
        }

    def test_config_unlimited_mode(self):
        """Test that configuration supports unlimited mode"""
        # Test SNAKE_CODING_MODEL config
        coding_config = Config.SNAKE_CODING_MODEL
        self.assertIn('unlimited_mode', coding_config)
        self.assertTrue(coding_config.get('unlimited_mode', False))

        # Test SNAKE_REASONING_MODEL config
        reasoning_config = Config.SNAKE_REASONING_MODEL
        self.assertIn('unlimited_mode', reasoning_config)
        self.assertTrue(reasoning_config.get('unlimited_mode', False))

        # Test that max_tokens can be None for unlimited
        self.assertIsNone(coding_config.get('max_tokens'))
        self.assertIsNone(reasoning_config.get('max_tokens'))

        print("✓ Configuration supports unlimited token generation")

    @patch('aiohttp.ClientSession.post')
    @patch('core.snake_llm.SnakeConfigValidator.validate_ollama_connection')
    async def test_unlimited_ollama_call(self, mock_validate, mock_post):
        """Test that Ollama API is called with unlimited tokens"""
        # Setup mocks
        mock_validate.return_value = True
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value.__aenter__.return_value = mock_response

        # Create LLM instance
        coding_llm = SnakeCodingLLM()
        coding_llm._initialized = True  # Skip initialization

        # Make a call
        await coding_llm._call_ollama("test prompt", unlimited=True)

        # Verify the call was made with num_predict = -1 (unlimited)
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        payload = call_args[1]['json']

        self.assertEqual(payload['options']['num_predict'], -1)
        print("✓ Ollama API called with unlimited tokens (num_predict = -1)")

    @patch('aiohttp.ClientSession.post')
    @patch('core.snake_llm.SnakeConfigValidator.validate_ollama_connection')
    async def test_coding_methods_use_unlimited(self, mock_validate, mock_post):
        """Test that coding methods use unlimited token generation"""
        # Setup mocks
        mock_validate.return_value = True
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = self.mock_response
        mock_post.return_value.__aenter__.return_value = mock_response

        coding_llm = SnakeCodingLLM()
        coding_llm._initialized = True

        test_code = "def hello(): print('Hello, World!')"

        # Test all methods that should use unlimited tokens
        await coding_llm.analyze_code(test_code)
        await coding_llm.generate_improvement("some analysis", test_code)
        await coding_llm.review_code_safety(test_code)

        # All calls should have used unlimited tokens
        self.assertEqual(mock_post.call_count, 3)

        for call in mock_post.call_args_list:
            payload = call[1]['json']
            self.assertEqual(payload['options']['num_predict'], -1)

        print("✓ All coding methods use unlimited token generation")


class TestErrorHandling(unittest.TestCase):
    """Test error handling improvements"""

    def setUp(self):
        """Set up test environment"""
        self.mock_agi_system = Mock()
        self.mock_agi_system.workspace_path = tempfile.mkdtemp()

    async def test_analysis_cycle_error_recovery(self):
        """Test that analysis cycle recovers from errors gracefully"""
        agent = SnakeAgent(self.mock_agi_system)
        agent.file_monitor = Mock()

        # Mock various methods to raise exceptions
        agent.file_monitor.scan_for_changes = Mock(
            side_effect=Exception("File monitor error"))
        agent._should_perform_periodic_analysis = Mock(
            side_effect=Exception("Periodic analysis error"))

        # Should not crash despite multiple errors
        try:
            await agent._execute_analysis_cycle()
            print("✓ Analysis cycle handles errors gracefully")
        except Exception as e:
            self.fail(f"Analysis cycle crashed despite error handling: {e}")

    def test_state_persistence_with_corruption(self):
        """Test state persistence handles corruption gracefully"""
        agent = SnakeAgent(self.mock_agi_system)

        # Create a corrupted state file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json}')
            corrupted_state_file = f.name

        try:
            agent.state_file = Path(corrupted_state_file)

            # Should not crash when loading corrupted state
            asyncio.run(agent._load_state())

            # Should have default state
            self.assertIsNotNone(agent.state)
            self.assertEqual(agent.state.mood, "curious")
            print("✓ Corrupted state file handled gracefully")

        finally:
            os.unlink(corrupted_state_file)


async def run_integration_test():
    """Integration test for complete Snake Agent functionality"""
    print("\n=== Running Integration Test ===")

    # Create mock AGI system
    mock_agi = Mock()
    mock_agi.workspace_path = tempfile.mkdtemp()

    try:
        # Create Snake Agent
        agent = SnakeAgent(mock_agi)

        # Test initialization (without actually connecting to Ollama)
        with patch('core.snake_llm.SnakeConfigValidator.validate_ollama_connection', return_value=True):
            with patch('core.snake_llm.OllamaClient.pull_model_if_needed', return_value=True):
                init_success = await agent.initialize()

                if init_success:
                    print("✓ Snake Agent initialization successful")
                else:
                    print(
                        "⚠ Snake Agent initialization failed (expected without Ollama)")

        # Test state management
        original_rate = agent.state.experiment_success_rate
        agent._update_experiment_success_rate(True)
        agent._update_mood()

        print(
            f"✓ State management: rate {original_rate} → {agent.state.experiment_success_rate}, mood: {agent.state.mood}")

        # Test error handling
        agent.state = None  # Simulate corrupted state
        await agent._execute_analysis_cycle()  # Should not crash

        print("✓ Error handling: survived null state")

        # Test status report
        status = await agent.get_status()
        print(f"✓ Status report: {json.dumps(status, indent=2, default=str)}")

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        raise
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(mock_agi.workspace_path, ignore_errors=True)


def run_all_tests():
    """Run all test suites"""
    print("=" * 60)
    print("SNAKE AGENT BUG FIX & UNLIMITED TOKEN GENERATION TESTS")
    print("=" * 60)

    # Run unit tests
    test_classes = [
        TestSnakeAgentAttributeFix,
        TestUnlimitedTokenGeneration,
        TestErrorHandling
    ]

    for test_class in test_classes:
        print(f"\n--- Running {test_class.__name__} ---")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)

        if result.wasSuccessful():
            print(f"✓ {test_class.__name__} passed all tests")
        else:
            print(f"✗ {test_class.__name__} had failures")

    # Run integration test
    print(f"\n--- Running Integration Test ---")
    asyncio.run(run_integration_test())

    print("\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
