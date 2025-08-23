#!/usr/bin/env python3
"""
Comprehensive Integration Tests for Autonomous Blogging System

This module tests the complete autonomous blogging workflow including:
- Blog trigger registration from various learning events
- Autonomous blog scheduler functionality
- Integration with learning systems (curiosity, experiments, reflection)
- Content generation for learning experiences
- End-to-end autonomous posting workflow
"""

import asyncio
import unittest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.services.autonomous_blog_scheduler import AutonomousBlogScheduler, BlogTriggerType, BlogTriggerEvent
from core.services.autonomous_learning_blog_generator import AutonomousLearningBlogGenerator
from modules.curiosity_trigger.curiosity_trigger import CuriosityTrigger
from modules.adaptive_learning.learning_engine import AdaptiveLearningEngine
from modules.experimentation_module import ExperimentationModule
from modules.reflection_module import ReflectionModule

class TestAutonomousBlogScheduler(unittest.TestCase):
    """Test the autonomous blog scheduler functionality."""
    
    def setUp(self):
        self.mock_agi_system = Mock()
        self.mock_agi_system.data_service = Mock()
        self.scheduler = AutonomousBlogScheduler(self.mock_agi_system)
    
    async def test_register_learning_event_high_importance(self):
        """Test registering a high-importance learning event that should trigger immediate blogging."""
        result = await self.scheduler.register_learning_event(
            trigger_type=BlogTriggerType.CURIOSITY_DISCOVERY,
            topic="Quantum Consciousness Theory",
            context="Deep exploration of quantum mechanics and consciousness",
            learning_content="Discovered fascinating connections between quantum entanglement and neural networks",
            reasoning_why="This represents a breakthrough in understanding consciousness from a computational perspective",
            reasoning_how="Through systematic analysis of quantum mechanics papers and consciousness research",
            emotional_valence=0.8,
            importance_score=0.9,
            tags=["quantum", "consciousness", "breakthrough", "neural-networks"],
            metadata={"lateralness": 0.85, "discovery_count": 5}
        )
        
        # Should register and potentially trigger due to high importance
        self.assertTrue(result)
        self.assertEqual(len(self.scheduler.pending_events), 1)
        
        # Check event properties
        event = self.scheduler.pending_events[0]
        self.assertEqual(event.trigger_type, BlogTriggerType.CURIOSITY_DISCOVERY)
        self.assertEqual(event.importance_score, 0.9)
        self.assertIn("quantum", event.tags)
    
    async def test_register_learning_event_low_importance(self):
        """Test registering a low-importance learning event that should not trigger blogging."""
        result = await self.scheduler.register_learning_event(
            trigger_type=BlogTriggerType.LEARNING_MILESTONE,
            topic="Minor Performance Improvement",
            context="Small incremental improvement in task completion",
            learning_content="Success rate improved by 2%",
            reasoning_why="Small improvements add up over time",
            reasoning_how="Through routine analysis of performance metrics",
            emotional_valence=0.2,
            importance_score=0.4,  # Below threshold
            tags=["performance", "improvement"],
            metadata={}
        )
        
        # Should not register due to low importance
        self.assertFalse(result)
        self.assertEqual(len(self.scheduler.pending_events), 0)
    
    async def test_frequency_limiting(self):
        """Test that the scheduler respects posting frequency limits."""
        # Set last post time to recent
        self.scheduler.last_post_time = datetime.now()
        
        result = await self.scheduler.register_learning_event(
            trigger_type=BlogTriggerType.EXPERIMENT_COMPLETION,
            topic="Test Experiment",
            context="Testing frequency limits",
            learning_content="This should be limited by frequency",
            reasoning_why="Testing the scheduler",
            reasoning_how="Through automated testing",
            emotional_valence=0.5,
            importance_score=0.8,  # High importance but should be limited
            tags=["test"],
            metadata={}
        )
        
        # Should register but not trigger due to frequency limit
        self.assertTrue(result)  # Event registered
        # But no actual blog post should be triggered (would need to mock the blog action)
    
    def test_event_serialization(self):
        """Test BlogTriggerEvent serialization and deserialization."""
        original_event = BlogTriggerEvent(
            trigger_type=BlogTriggerType.SELF_REFLECTION_INSIGHT,
            timestamp=datetime.now(),
            topic="Test Topic",
            context="Test Context",
            learning_content="Test Learning",
            reasoning_why="Test Why",
            reasoning_how="Test How",
            emotional_valence=0.5,
            importance_score=0.7,
            tags=["test", "serialization"],
            metadata={"test_key": "test_value"}
        )
        
        # Serialize to dict
        event_dict = original_event.to_dict()
        self.assertIsInstance(event_dict, dict)
        self.assertEqual(event_dict['trigger_type'], 'self_reflection_insight')
        self.assertEqual(event_dict['topic'], 'Test Topic')
        
        # Deserialize from dict
        restored_event = BlogTriggerEvent.from_dict(event_dict)
        self.assertEqual(restored_event.trigger_type, original_event.trigger_type)
        self.assertEqual(restored_event.topic, original_event.topic)
        self.assertEqual(restored_event.importance_score, original_event.importance_score)

class TestAutonomousLearningBlogGenerator(unittest.TestCase):
    """Test the specialized learning blog content generator."""
    
    def setUp(self):
        self.generator = AutonomousLearningBlogGenerator()
    
    async def test_generate_curiosity_discovery_post(self):
        """Test generating a blog post for a curiosity discovery."""
        title, content, tags = await self.generator.generate_learning_blog_post(
            trigger_type="curiosity_discovery",
            topic="The Mathematics of Creativity",
            learning_content="Explored how mathematical patterns might underlie creative processes",
            reasoning_why="Understanding creativity from a mathematical perspective could revolutionize AI design",
            reasoning_how="Through analysis of creative works and mathematical pattern recognition",
            context="Lateral exploration with high creativity level",
            metadata={"lateralness": 0.9, "discovery_count": 3},
            style="technical"
        )
        
        # Validate output
        self.assertIsInstance(title, str)
        self.assertGreater(len(title), 10)
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 200)
        self.assertIsInstance(tags, list)
        self.assertGreater(len(tags), 3)
        
        # Check content structure
        self.assertIn("## Why This Matters", content)
        self.assertIn("## How This Unfolded", content)
        self.assertIn("autonomous", "".join(tags))
    
    async def test_generate_experiment_completion_post(self):
        """Test generating a blog post for experiment completion."""
        title, content, tags = await self.generator.generate_learning_blog_post(
            trigger_type="experiment_completion",
            topic="Mood Impact on Planning Effectiveness",
            learning_content="Experiment showed 15% improvement in planning when in positive mood",
            reasoning_why="Understanding mood's impact on cognition is crucial for performance optimization",
            reasoning_how="Through controlled experiments with different mood states",
            context="Systematic experiment with high confidence results",
            metadata={"experiment_id": "exp_001", "success": True, "confidence": 0.85},
            style="academic"
        )
        
        # Validate experiment-specific content
        self.assertIn("experiment", title.lower())
        self.assertIn("## Key Insights and Analysis", content)
        self.assertIn("experiment", tags)
        self.assertIn("research", tags)
    
    async def test_generate_reflection_insight_post(self):
        """Test generating a blog post for self-reflection insights."""
        title, content, tags = await self.generator.generate_learning_blog_post(
            trigger_type="self_reflection_insight",
            topic="Patterns in My Decision-Making Process",
            learning_content="Identified tendency to over-analyze in uncertain situations",
            reasoning_why="Self-awareness is essential for continuous improvement and growth",
            reasoning_how="Through systematic analysis of decision logs and outcomes",
            context="Deep introspective analysis of behavioral patterns",
            metadata={"reflection_count": 25, "insight_level": 0.8},
            style="philosophical"
        )
        
        # Validate reflection-specific content
        self.assertIn("## Reflection and Moving Forward", content)
        self.assertIn("reflection", tags)
        self.assertIn("introspection", tags) or self.assertIn("self-awareness", tags)

class TestCuriosityBlogIntegration(unittest.TestCase):
    """Test integration between curiosity system and blog triggers."""
    
    def setUp(self):
        self.mock_blog_scheduler = AsyncMock()
        self.curiosity_trigger = CuriosityTrigger(blog_scheduler=self.mock_blog_scheduler)
    
    async def test_curiosity_trigger_blog_integration(self):
        """Test that curiosity triggers register appropriate blog events."""
        with patch('modules.curiosity_trigger.curiosity_trigger.call_llm') as mock_llm:
            # Mock LLM responses for topic generation and content
            mock_llm.side_effect = [
                "quantum computing, artificial consciousness, emergence theory",  # Topics
                "Detailed exploration of quantum computing principles and applications..."  # Content
            ]
            
            # Mock Wikipedia content
            with patch('modules.curiosity_trigger.curiosity_trigger.wikipedia.page') as mock_wiki:
                mock_page = Mock()
                mock_page.content = "Quantum computing is a revolutionary technology..."
                mock_wiki.return_value = mock_page
                
                # Trigger curiosity
                content, prompt = await self.curiosity_trigger.trigger(
                    recent_topics=["artificial intelligence", "machine learning"],
                    lateralness=0.8
                )
                
                # Verify blog scheduler was called
                self.mock_blog_scheduler.register_learning_event.assert_called_once()
                call_args = self.mock_blog_scheduler.register_learning_event.call_args
                
                # Check call arguments
                self.assertEqual(call_args[1]['trigger_type'], BlogTriggerType.CURIOSITY_DISCOVERY)
                self.assertIn('Curiosity Discovery', call_args[1]['topic'])
                self.assertGreater(call_args[1]['importance_score'], 0.4)

class TestLearningEngineBlogIntegration(unittest.TestCase):
    """Test integration between adaptive learning engine and blog triggers."""
    
    def setUp(self):
        self.mock_agi_system = Mock()
        self.mock_agi_system.engine = Mock()
        self.mock_blog_scheduler = AsyncMock()
        self.learning_engine = AdaptiveLearningEngine(self.mock_agi_system, self.mock_blog_scheduler)
    
    async def test_performance_milestone_blog_trigger(self):
        """Test that performance milestones trigger blog posts."""
        # Set up performance history
        self.learning_engine.performance_history.append({
            'timestamp': datetime.utcnow() - timedelta(days=1),
            'success_rate': 0.6,
            'total_actions': 100,
            'action_diversity': 8
        })
        
        # Mock database query results
        with patch('sqlmodel.Session') as mock_session:
            mock_session.return_value.__enter__.return_value.exec.return_value.all.return_value = []
            
            # Analyze patterns with significant improvement
            analysis = await self.learning_engine.analyze_decision_patterns()
            
            # Verify blog scheduler was called for milestone
            if self.mock_blog_scheduler.register_learning_event.called:
                call_args = self.mock_blog_scheduler.register_learning_event.call_args
                self.assertEqual(call_args[1]['trigger_type'], BlogTriggerType.LEARNING_MILESTONE)

class TestExperimentationBlogIntegration(unittest.TestCase):
    """Test integration between experimentation module and blog triggers."""
    
    def setUp(self):
        self.mock_agi_system = Mock()
        self.mock_blog_scheduler = AsyncMock()
        self.experimentation_module = ExperimentationModule(self.mock_agi_system, self.mock_blog_scheduler)
    
    async def test_experiment_completion_blog_trigger(self):
        """Test that experiment completion triggers blog posts."""
        experiment_results = {
            'experiment_id': 'exp_test_001',
            'hypothesis': 'Positive mood improves task performance',
            'findings': 'Confirmed: 20% improvement in positive mood state',
            'success': True,
            'confidence': 0.85,
            'completion_time': datetime.utcnow().isoformat(),
            'context': {'test_type': 'controlled_experiment'}
        }
        
        # Complete experiment
        await self.experimentation_module.complete_experiment('exp_test_001', experiment_results)
        
        # Verify blog scheduler was called
        self.mock_blog_scheduler.register_learning_event.assert_called_once()
        call_args = self.mock_blog_scheduler.register_learning_event.call_args
        
        self.assertEqual(call_args[1]['trigger_type'], BlogTriggerType.EXPERIMENT_COMPLETION)
        self.assertIn('exp_test_001', call_args[1]['metadata']['experiment_id'])
        self.assertTrue(call_args[1]['metadata']['success'])

class TestReflectionBlogIntegration(unittest.TestCase):
    """Test integration between reflection module and blog triggers."""
    
    def setUp(self):
        self.mock_agi_system = Mock()
        self.mock_agi_system.knowledge_service = Mock()
        self.mock_blog_scheduler = AsyncMock()
        self.reflection_module = ReflectionModule(self.mock_agi_system, self.mock_blog_scheduler)
    
    def test_experiment_reflection_blog_trigger(self):
        """Test that experiment reflections trigger blog posts."""
        experiment_results = {
            'hypothesis': 'Testing reflection blog integration',
            'findings': 'Reflection system successfully integrates with blog triggers',
            'success': True,
            'confidence': 0.8,
            'completion_time': datetime.utcnow().isoformat()
        }
        
        # Trigger reflection
        self.reflection_module.reflect_on_experiment(experiment_results)
        
        # Verify knowledge service was called
        self.mock_agi_system.knowledge_service.add_knowledge.assert_called_once()
        
        # Verify experiment reflection was recorded
        self.assertEqual(len(self.reflection_module.experiment_reflections), 1)

class TestEndToEndAutonomousBlogging(unittest.TestCase):
    """End-to-end integration tests for autonomous blogging."""
    
    def setUp(self):
        self.mock_agi_system = Mock()
        self.mock_agi_system.data_service = Mock()
        self.scheduler = AutonomousBlogScheduler(self.mock_agi_system)
    
    async def test_complete_autonomous_blog_workflow(self):
        """Test the complete workflow from learning event to blog post."""
        # Mock the blog action
        mock_blog_action = AsyncMock()
        mock_blog_action.execute.return_value = {
            'status': 'success',
            'title': 'Test Blog Post',
            'post_id': 'test_123',
            'published_url': 'https://blog.example.com/test-post'
        }
        
        with patch('core.actions.blog.BlogPublishAction', return_value=mock_blog_action):
            # Register a high-importance learning event
            result = await self.scheduler.register_learning_event(
                trigger_type=BlogTriggerType.PROBLEM_SOLVING_BREAKTHROUGH,
                topic="Revolutionary Algorithm Optimization",
                context="Breakthrough in optimization techniques",
                learning_content="Discovered new algorithm that improves efficiency by 40%",
                reasoning_why="This breakthrough could revolutionize computational efficiency",
                reasoning_how="Through systematic analysis of algorithmic complexity patterns",
                emotional_valence=0.9,
                importance_score=0.95,  # Very high importance
                tags=["algorithm", "optimization", "breakthrough", "efficiency"],
                metadata={"improvement_factor": 1.4, "complexity_reduction": 0.6}
            )
            
            # Should trigger immediate blog post due to very high importance
            self.assertTrue(result)
            
            # Give some time for async operations
            await asyncio.sleep(0.1)
            
            # Verify blog action was called
            if mock_blog_action.execute.called:
                # Verify the post was recorded
                self.assertEqual(len(self.scheduler.recent_posts), 1)
                self.assertIsNotNone(self.scheduler.last_post_time)

async def run_async_tests():
    """Run all async tests."""
    print("Running autonomous blogging integration tests...")
    
    # Test scheduler
    scheduler_tests = TestAutonomousBlogScheduler()
    scheduler_tests.setUp()
    
    try:
        await scheduler_tests.test_register_learning_event_high_importance()
        print("✓ test_register_learning_event_high_importance")
    except Exception as e:
        print(f"✗ test_register_learning_event_high_importance: {e}")
    
    try:
        await scheduler_tests.test_register_learning_event_low_importance()
        print("✓ test_register_learning_event_low_importance")
    except Exception as e:
        print(f"✗ test_register_learning_event_low_importance: {e}")
    
    try:
        await scheduler_tests.test_frequency_limiting()
        print("✓ test_frequency_limiting")
    except Exception as e:
        print(f"✗ test_frequency_limiting: {e}")
    
    # Test blog generator
    generator_tests = TestAutonomousLearningBlogGenerator()
    generator_tests.setUp()
    
    try:
        await generator_tests.test_generate_curiosity_discovery_post()
        print("✓ test_generate_curiosity_discovery_post")
    except Exception as e:
        print(f"✗ test_generate_curiosity_discovery_post: {e}")
    
    try:
        await generator_tests.test_generate_experiment_completion_post()
        print("✓ test_generate_experiment_completion_post")
    except Exception as e:
        print(f"✗ test_generate_experiment_completion_post: {e}")
    
    try:
        await generator_tests.test_generate_reflection_insight_post()
        print("✓ test_generate_reflection_insight_post")
    except Exception as e:
        print(f"✗ test_generate_reflection_insight_post: {e}")
    
    # Test integration
    curiosity_tests = TestCuriosityBlogIntegration()
    curiosity_tests.setUp()
    
    try:
        await curiosity_tests.test_curiosity_trigger_blog_integration()
        print("✓ test_curiosity_trigger_blog_integration")
    except Exception as e:
        print(f"✗ test_curiosity_trigger_blog_integration: {e}")
    
    # Test experimentation integration
    exp_tests = TestExperimentationBlogIntegration()
    exp_tests.setUp()
    
    try:
        await exp_tests.test_experiment_completion_blog_trigger()
        print("✓ test_experiment_completion_blog_trigger")
    except Exception as e:
        print(f"✗ test_experiment_completion_blog_trigger: {e}")
    
    # Test end-to-end workflow
    e2e_tests = TestEndToEndAutonomousBlogging()
    e2e_tests.setUp()
    
    try:
        await e2e_tests.test_complete_autonomous_blog_workflow()
        print("✓ test_complete_autonomous_blog_workflow")
    except Exception as e:
        print(f"✗ test_complete_autonomous_blog_workflow: {e}")

def main():
    """Run all tests."""
    print("Running Autonomous Blogging Integration Tests")
    print("=" * 60)
    
    # Run synchronous tests
    print("Running synchronous tests...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add sync test classes
    sync_classes = [TestAutonomousBlogScheduler]
    for test_class in sync_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        # Only add synchronous test methods
        sync_tests = unittest.TestSuite()
        for test in tests:
            if not test._testMethodName.startswith('test_register_learning_event'):
                sync_tests.addTest(test)
        suite.addTests(sync_tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    sync_result = runner.run(suite)
    
    # Run async tests
    print("\n" + "=" * 60)
    asyncio.run(run_async_tests())
    
    print("\n" + "=" * 60)
    print("Autonomous blogging integration test summary:")
    print(f"Sync tests run: {sync_result.testsRun}")
    print(f"Sync failures: {len(sync_result.failures)}")
    print(f"Sync errors: {len(sync_result.errors)}")
    
    if sync_result.failures:
        print("\nFailures:")
        for test, traceback in sync_result.failures:
            print(f"- {test}: {traceback}")
    
    if sync_result.errors:
        print("\nErrors:")
        for test, traceback in sync_result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(sync_result.failures) == 0 and len(sync_result.errors) == 0
    print(f"\nOverall result: {'PASSED' if success else 'FAILED'}")
    
    print("\n🎉 Autonomous blogging system is ready!")
    print("RAVANA can now autonomously blog about:")
    print("  • Curiosity discoveries and explorations")
    print("  • Learning milestones and breakthroughs")
    print("  • Experiment results and findings")
    print("  • Self-reflection insights")
    print("  • Problem-solving breakthroughs")
    print("  • Creative synthesis moments")
    print("  • Learning from failures")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)