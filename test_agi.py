#!/usr/bin/env python3
"""
Test script for the AGI system
"""

import os
import sys
import unittest
import threading
import time
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the AGI system
from main import AGISystem

class TestAGISystem(unittest.TestCase):
    """Test cases for the AGI system"""
    
    @patch('main.EmotionalIntelligence')
    @patch('main.GoalPlanner')
    @patch('main.EventDetector')
    @patch('main.setup_db')
    def setUp(self, mock_setup_db, mock_event_detector, mock_goal_planner, mock_emotional_intelligence):
        """Set up the test environment"""
        self.agi = AGISystem()
        # Mock the memory server thread
        self.agi.start_memory_server = MagicMock()
    
    def test_initialization(self):
        """Test that the AGI system initializes correctly"""
        self.assertIsNotNone(self.agi.emotional_intelligence)
        self.assertIsNotNone(self.agi.goal_planner)
        self.assertIsNotNone(self.agi.event_detector)
        self.assertFalse(self.agi.running)
    
    @patch('main.requests.post')
    def test_process_memory(self, mock_post):
        """Test the process_memory method"""
        # Mock the response from the memory server
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"memories": ["Test memory"]}
        mock_post.return_value = mock_response
        
        # Call the method
        memories = self.agi.process_memory("Test input")
        
        # Check the result
        self.assertEqual(memories, ["Test memory"])
        mock_post.assert_called()
    
    @patch('main.requests.post')
    def test_retrieve_relevant_memories(self, mock_post):
        """Test the retrieve_relevant_memories method"""
        # Mock the response from the memory server
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"relevant_memories": [{"id": 1, "text": "Test memory", "similarity": 0.9}]}
        mock_post.return_value = mock_response
        
        # Call the method
        memories = self.agi.retrieve_relevant_memories("Test query")
        
        # Check the result
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["text"], "Test memory")
        mock_post.assert_called()
    
    @patch('main.plan_from_context')
    def test_create_goal_from_context(self, mock_plan_from_context):
        """Test the create_goal_from_context method"""
        # Mock the response from the planner
        mock_plan_from_context.return_value = 1
        
        # Call the method
        goal_id = self.agi.create_goal_from_context("Test context")
        
        # Check the result
        self.assertEqual(goal_id, 1)
        mock_plan_from_context.assert_called_with("Test context")
    
    @patch('main.CuriosityTrigger.trigger')
    def test_trigger_curiosity(self, mock_trigger):
        """Test the trigger_curiosity method"""
        # Mock the response from the curiosity trigger
        mock_trigger.return_value = ("Test article", "Test prompt")
        
        # Call the method
        article, prompt = self.agi.trigger_curiosity(["Test topic"])
        
        # Check the result
        self.assertEqual(article, "Test article")
        self.assertEqual(prompt, "Test prompt")
        mock_trigger.assert_called_with(["Test topic"], 0.7)
    
    @patch('main.reflect_on_task')
    def test_reflect_on_action(self, mock_reflect_on_task):
        """Test the reflect_on_action method"""
        # Mock the response from the reflection module
        mock_reflect_on_task.return_value = {"reflection": "Test reflection"}
        
        # Call the method
        reflection = self.agi.reflect_on_action("Test task", "Test outcome")
        
        # Check the result
        self.assertEqual(reflection, {"reflection": "Test reflection"})
        mock_reflect_on_task.assert_called_with("Test task", "Test outcome")
    
    @patch('main.compress_knowledge')
    def test_compress_knowledge_logs(self, mock_compress_knowledge):
        """Test the compress_knowledge_logs method"""
        # Mock the response from the knowledge compression module
        mock_compress_knowledge.return_value = {"summary": "Test summary"}
        
        # Call the method
        compressed = self.agi.compress_knowledge_logs([{"log": "Test log"}])
        
        # Check the result
        self.assertEqual(compressed, {"summary": "Test summary"})
        mock_compress_knowledge.assert_called_with([{"log": "Test log"}])
    
    def test_process_input_create_goal(self):
        """Test the process_input method with a create goal request"""
        # Mock the necessary methods
        self.agi.emotional_intelligence.process_action_natural = MagicMock()
        self.agi.process_memory = MagicMock(return_value=["Test memory"])
        self.agi.retrieve_relevant_memories = MagicMock(return_value=[])
        self.agi.emotional_intelligence.get_dominant_mood = MagicMock(return_value="Curious")
        self.agi.emotional_intelligence.influence_behavior = MagicMock(return_value={})
        self.agi.create_goal_from_context = MagicMock(return_value=1)
        
        # Call the method
        response = self.agi.process_input("create goal: Learn Python")
        
        # Check the result
        self.assertIn("Created new goal with ID: 1", response)
        self.agi.create_goal_from_context.assert_called_with("create goal: Learn Python")
    
    def test_process_input_experiment(self):
        """Test the process_input method with an experiment request"""
        # Mock the necessary methods
        self.agi.emotional_intelligence.process_action_natural = MagicMock()
        self.agi.process_memory = MagicMock(return_value=["Test memory"])
        self.agi.retrieve_relevant_memories = MagicMock(return_value=[])
        self.agi.emotional_intelligence.get_dominant_mood = MagicMock(return_value="Curious")
        self.agi.emotional_intelligence.influence_behavior = MagicMock(return_value={})
        self.agi.run_experiment = MagicMock(return_value="Test result")
        
        # Call the method
        response = self.agi.process_input("experiment: Test gravity")
        
        # Check the result
        self.assertIn("Experiment result: Test result", response)
        self.agi.run_experiment.assert_called_with("experiment: Test gravity")
    
    def test_process_input_default(self):
        """Test the process_input method with a default request"""
        # Mock the necessary methods
        self.agi.emotional_intelligence.process_action_natural = MagicMock()
        self.agi.process_memory = MagicMock(return_value=["Test memory"])
        self.agi.retrieve_relevant_memories = MagicMock(return_value=[])
        self.agi.emotional_intelligence.get_dominant_mood = MagicMock(return_value="Curious")
        self.agi.emotional_intelligence.influence_behavior = MagicMock(return_value={})
        
        # Call the method
        response = self.agi.process_input("Hello, how are you?")
        
        # Check the result
        self.assertIn("Processed input. Current mood: Curious", response)
        self.assertIn("Memories stored: 1", response)

if __name__ == "__main__":
    unittest.main() 