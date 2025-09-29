#!/usr/bin/env python3
"""
Comprehensive Unit Tests for RAVANA Blog Integration Components

This module provides unit tests for:
- BlogAPIInterface
- BlogContentGenerator
- BlogPublishAction
- BlogContentValidator
"""

from core.actions.blog_content_validator import BlogContentValidator, ContentValidationError
from core.actions.blog import BlogPublishAction
from core.actions.blog_content_generator import BlogContentGenerator, BlogContentError
from core.actions.blog_api import BlogAPIInterface, BlogAPIError, CircuitBreaker
import asyncio
import unittest
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCircuitBreaker(unittest.TestCase):
    """Test the CircuitBreaker class."""

    def setUp(self):
        self.breaker = CircuitBreaker(failure_threshold=3, timeout=60)

    def test_initial_state(self):
        """Test circuit breaker initial state."""
        self.assertEqual(self.breaker.state, "CLOSED")
        self.assertTrue(self.breaker.can_execute())
        self.assertEqual(self.breaker.failure_count, 0)

    def test_success_recording(self):
        """Test recording successful operations."""
        self.breaker.failure_count = 2
        self.breaker.record_success()
        self.assertEqual(self.breaker.failure_count, 0)
        self.assertEqual(self.breaker.state, "CLOSED")

    def test_failure_recording(self):
        """Test recording failed operations."""
        # Record failures below threshold
        for i in range(2):
            self.breaker.record_failure()
            self.assertEqual(self.breaker.state, "CLOSED")

        # Record failure that triggers circuit opening
        self.breaker.record_failure()
        self.assertEqual(self.breaker.state, "OPEN")
        self.assertFalse(self.breaker.can_execute())

    def test_half_open_state(self):
        """Test half-open state behavior."""
        # Force circuit to open
        for i in range(3):
            self.breaker.record_failure()

        # Simulate timeout passing
        import time
        self.breaker.last_failure_time = time.time() - 61

        # Should transition to half-open
        self.assertTrue(self.breaker.can_execute())


class TestBlogAPIInterface(unittest.TestCase):
    """Test the BlogAPIInterface class."""

    def setUp(self):
        self.api = BlogAPIInterface()

    def test_initialization(self):
        """Test API interface initialization."""
        self.assertIsNotNone(self.api.api_url)
        self.assertIsNotNone(self.api.auth_token)
        self.assertIsInstance(self.api.circuit_breaker, CircuitBreaker)
        self.assertGreater(self.api.timeout, 0)
        self.assertGreater(self.api.retry_attempts, 0)

    def test_validate_config_success(self):
        """Test configuration validation with valid config."""
        result = self.api.validate_config()
        self.assertTrue(result)

    @patch('core.config.Config.BLOG_ENABLED', False)
    def test_validate_config_disabled(self):
        """Test configuration validation when blog is disabled."""
        api = BlogAPIInterface()
        result = api.validate_config()
        self.assertFalse(result)

    @patch('core.config.Config.BLOG_API_URL', '')
    def test_validate_config_no_url(self):
        """Test configuration validation with missing URL."""
        api = BlogAPIInterface()
        result = api.validate_config()
        self.assertFalse(result)

    @patch('core.config.Config.BLOG_AUTH_TOKEN', '')
    def test_validate_config_no_token(self):
        """Test configuration validation with missing token."""
        api = BlogAPIInterface()
        result = api.validate_config()
        self.assertFalse(result)

    async def test_publish_post_circuit_breaker_open(self):
        """Test publish_post when circuit breaker is open."""
        self.api.circuit_breaker.state = "OPEN"

        with self.assertRaises(BlogAPIError) as context:
            await self.api.publish_post("Test", "Content", ["tag"])

        self.assertEqual(context.exception.error_type, "CIRCUIT_BREAKER_OPEN")

    @patch('aiohttp.ClientSession.post')
    async def test_publish_post_success(self, mock_post):
        """Test successful blog post publication."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(
            return_value='{"success": true, "post_id": "123", "published_url": "http://example.com"}')
        mock_response.headers = {}
        mock_post.return_value.__aenter__.return_value = mock_response

        result = await self.api.publish_post("Test Title", "Test Content", ["test"])

        self.assertTrue(result["success"])
        self.assertEqual(result["post_id"], "123")
        self.assertEqual(result["published_url"], "http://example.com")

    @patch('aiohttp.ClientSession.post')
    async def test_publish_post_auth_failure(self, mock_post):
        """Test publication with authentication failure."""
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value='Unauthorized')
        mock_response.headers = {}
        mock_post.return_value.__aenter__.return_value = mock_response

        with self.assertRaises(BlogAPIError) as context:
            await self.api.publish_post("Test", "Content", ["tag"])

        self.assertEqual(context.exception.error_type, "AUTHENTICATION_FAILED")

    @patch('aiohttp.ClientSession.post')
    async def test_publish_post_rate_limit(self, mock_post):
        """Test publication with rate limiting."""
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.text = AsyncMock(return_value='Rate limited')
        mock_response.headers = {'Retry-After': '60'}
        mock_post.return_value.__aenter__.return_value = mock_response

        with self.assertRaises(BlogAPIError) as context:
            await self.api.publish_post("Test", "Content", ["tag"])

        self.assertEqual(context.exception.error_type, "RATE_LIMITED")
        self.assertEqual(context.exception.retry_after, 60)


class TestBlogContentValidator(unittest.TestCase):
    """Test the BlogContentValidator class."""

    def setUp(self):
        self.validator = BlogContentValidator()

    def test_validate_title_success(self):
        """Test successful title validation."""
        title = "Understanding Time Dilation in Physics"
        content = "# Introduction\n\nThis is a test content with multiple paragraphs.\n\n## Section 1\n\nSome detailed content here that meets the minimum length requirements for testing."
        tags = ["physics", "time", "science"]

        validated_title, validated_content, validated_tags, report = \
            self.validator.validate_and_sanitize(title, content, tags)

        self.assertEqual(validated_title, title)
        self.assertTrue(report["validation_passed"])
        self.assertGreater(report["quality_score"], 0)

    def test_validate_title_too_short(self):
        """Test title validation with too short title."""
        with self.assertRaises(ContentValidationError):
            self.validator._validate_and_sanitize_title("Hi", {})

    def test_validate_title_empty(self):
        """Test title validation with empty title."""
        with self.assertRaises(ContentValidationError):
            self.validator._validate_and_sanitize_title("", {})

    def test_validate_content_too_short(self):
        """Test content validation with too short content."""
        short_content = "Too short"
        with self.assertRaises(ContentValidationError):
            self.validator._validate_and_sanitize_content(short_content, {})

    def test_validate_content_empty(self):
        """Test content validation with empty content."""
        with self.assertRaises(ContentValidationError):
            self.validator._validate_and_sanitize_content("", {})

    def test_security_threat_removal(self):
        """Test removal of security threats from content."""
        malicious_content = "# Title\n\n<script>alert('xss')</script>\n\nSafe content here that is long enough to pass validation tests."
        clean_content = self.validator._remove_security_threats(
            malicious_content, "content")
        self.assertNotIn("<script>", clean_content)
        self.assertNotIn("alert", clean_content)

    def test_smart_truncation(self):
        """Test smart content truncation."""
        long_content = "Paragraph one.\n\nParagraph two.\n\nParagraph three." * 1000
        truncated = self.validator._smart_truncate(long_content, 200)

        self.assertLess(len(truncated), len(long_content))
        self.assertIn("Content truncated", truncated)

    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        high_quality_content = """
        # Main Title
        
        This is a comprehensive blog post with **bold text** and *italic text*.
        
        ## Section 1
        
        Here's some `inline code` and a code block:
        
        ```python
        def example():
            return "Hello World"
        ```
        
        ## Section 2
        
        Here's a [link](http://example.com) and a list:
        
        - Item 1
        - Item 2
        - Item 3
        
        This content has multiple paragraphs and good structure.
        """

        score = self.validator._calculate_quality_score(high_quality_content)
        self.assertGreater(score, 5.0)  # Should get a good score

    def test_readability_score_calculation(self):
        """Test readability score calculation."""
        readable_content = "This is simple text. It has short sentences. Easy to read."
        score = self.validator._calculate_readability_score(readable_content)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 10.0)

    def test_quick_validate_success(self):
        """Test quick validation with valid content."""
        result = self.validator.quick_validate(
            "Valid Title",
            "This is valid content that meets the minimum length requirements for the blog system.",
            ["valid", "tags"]
        )
        self.assertTrue(result)

    def test_quick_validate_failure(self):
        """Test quick validation with invalid content."""
        result = self.validator.quick_validate("", "Short", [])
        self.assertFalse(result)


class TestBlogContentGenerator(unittest.TestCase):
    """Test the BlogContentGenerator class."""

    def setUp(self):
        self.generator = BlogContentGenerator()

    def test_initialization(self):
        """Test content generator initialization."""
        self.assertIsNotNone(self.generator.validator)
        self.assertGreater(self.generator.max_content_length, 0)
        self.assertGreater(self.generator.min_content_length, 0)
        self.assertIn(self.generator.default_style,
                      self.generator.available_styles)

    def test_get_style_guidance(self):
        """Test style guidance generation."""
        for style in self.generator.available_styles:
            guidance = self.generator._get_style_guidance(style)
            self.assertIsInstance(guidance, str)
            self.assertGreater(len(guidance), 10)

    def test_parse_llm_content_response_json(self):
        """Test parsing LLM response in JSON format."""
        json_response = '{"title": "Test Title", "content": "Test content here"}'
        title, content = self.generator._parse_llm_content_response(
            json_response)

        self.assertEqual(title, "Test Title")
        self.assertEqual(content, "Test content here")

    def test_parse_llm_content_response_markdown(self):
        """Test parsing LLM response in markdown format."""
        markdown_response = "# Test Title\n\nThis is the content of the blog post."
        title, content = self.generator._parse_llm_content_response(
            markdown_response)

        self.assertEqual(title, "Test Title")
        self.assertEqual(content, "This is the content of the blog post.")

    def test_extract_simple_tags(self):
        """Test simple tag extraction."""
        content = "This is about machine learning and artificial intelligence in Python programming."
        topic = "AI Development"

        tags = self.generator._extract_simple_tags(content, topic)

        self.assertIn("ai development", tags)
        self.assertTrue(
            any(tag in ["machine learning", "ai", "python"] for tag in tags))

    @patch('core.actions.blog_content_generator.async_safe_call_llm')
    async def test_extract_tags_with_llm(self, mock_llm):
        """Test tag extraction using LLM."""
        mock_llm.return_value = '["physics", "time", "relativity", "science"]'

        tags = await self.generator._extract_tags("Content about physics", "physics")

        self.assertIn("physics", tags)
        self.assertIn("time", tags)
        mock_llm.assert_called_once()

    def test_mood_context_formatting(self):
        """Test mood context formatting."""
        mock_context = {
            "relevant_memories": [{"content": "Memory 1"}, {"content": "Memory 2"}],
            "recent_experiments": [{"content": "Experiment 1"}]
        }

        formatted = self.generator._format_memory_context(mock_context)

        self.assertIn("Memory 1", formatted)
        self.assertIn("Experiment 1", formatted)


class TestBlogPublishAction(unittest.TestCase):
    """Test the BlogPublishAction class."""

    def setUp(self):
        self.mock_system = Mock()
        self.mock_data_service = Mock()
        self.action = BlogPublishAction(
            self.mock_system, self.mock_data_service)

    def test_initialization(self):
        """Test action initialization."""
        self.assertEqual(self.action.name, "publish_blog_post")
        self.assertIsNotNone(self.action.description)
        self.assertIsInstance(self.action.parameters, list)
        self.assertGreater(len(self.action.parameters), 0)
        self.assertIsNotNone(self.action.api_interface)
        self.assertIsNotNone(self.action.content_generator)

    def test_parameter_schema(self):
        """Test parameter schema correctness."""
        params = {param['name']: param for param in self.action.parameters}

        # Required parameters
        self.assertIn('topic', params)
        self.assertTrue(params['topic']['required'])

        # Optional parameters
        self.assertIn('style', params)
        self.assertIn('context', params)
        self.assertIn('custom_tags', params)
        self.assertIn('dry_run', params)

    @patch('core.config.Config.BLOG_ENABLED', False)
    async def test_execute_blog_disabled(self):
        """Test execution when blog is disabled."""
        action = BlogPublishAction(self.mock_system, self.mock_data_service)
        result = await action.execute(topic="Test Topic")

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["error"], "BLOG_DISABLED")

    async def test_execute_missing_topic(self):
        """Test execution with missing topic."""
        result = await self.action.execute()

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"], "MISSING_TOPIC")

    @patch.object(BlogContentGenerator, 'generate_post')
    async def test_execute_dry_run(self, mock_generate):
        """Test dry run execution."""
        mock_generate.return_value = (
            "Test Title", "Test content here that is long enough", ["test", "physics"])

        result = await self.action.execute(
            topic="Test Topic",
            style="technical",
            dry_run=True
        )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["title"], "Test Title")
        mock_generate.assert_called_once()

    def test_calculate_emotional_valence(self):
        """Test emotional valence calculation."""
        # Test different action types
        self.assertEqual(
            self.action._calculate_emotional_valence({}, "published"), 0.8)
        self.assertEqual(
            self.action._calculate_emotional_valence({}, "dry_run"), 0.4)
        self.assertEqual(self.action._calculate_emotional_valence(
            {}, "content_error"), -0.6)
        self.assertEqual(
            self.action._calculate_emotional_valence({}, "api_error"), -0.6)
        self.assertEqual(self.action._calculate_emotional_valence(
            {}, "unexpected_error"), -0.8)
        self.assertEqual(
            self.action._calculate_emotional_valence({}, "unknown"), 0.0)

    def test_format_memory_content(self):
        """Test memory content formatting."""
        result = {"title": "Test Title", "topic": "Test Topic",
                  "tags": ["test"], "published_url": "http://example.com"}

        content = self.action._format_memory_content(result, "published")

        self.assertIn("Test Title", content)
        self.assertIn("Test Topic", content)
        self.assertIn("http://example.com", content)

    def test_get_configuration_info(self):
        """Test configuration info retrieval."""
        config = self.action.get_configuration_info()

        self.assertIsInstance(config, dict)
        self.assertIn("enabled", config)
        self.assertIn("api_url", config)
        self.assertIn("available_styles", config)
        self.assertIn("content_length_limits", config)


class TestIntegration(unittest.TestCase):
    """Integration tests for blog components."""

    async def test_full_workflow_dry_run(self):
        """Test complete workflow in dry run mode."""
        # Create mock system and services
        mock_system = Mock()
        mock_data_service = Mock()

        # Create action
        action = BlogPublishAction(mock_system, mock_data_service)

        # Mock LLM response
        with patch('core.actions.blog_content_generator.async_safe_call_llm') as mock_llm:
            mock_llm.return_value = '''
            {
                "title": "Understanding Machine Learning",
                "content": "# Understanding Machine Learning\\n\\nMachine learning is a fascinating field that enables computers to learn from data. This comprehensive guide explores the fundamentals and applications of ML in modern technology.\\n\\n## Key Concepts\\n\\nThe core concepts include supervised learning, unsupervised learning, and reinforcement learning. Each approach has its own strengths and use cases.\\n\\n## Applications\\n\\nML is used in various domains including natural language processing, computer vision, and autonomous systems."
            }
            '''

            # Execute dry run
            result = await action.execute(
                topic="Machine Learning Fundamentals",
                style="technical",
                context="Introduction to ML concepts",
                custom_tags=["ai", "ml"],
                dry_run=True
            )

            # Verify result
            self.assertEqual(result["status"], "success")
            self.assertTrue(result["dry_run"])
            self.assertIn("Understanding Machine Learning", result["title"])
            self.assertGreater(result["content_length"], 100)
            self.assertIn("ai", result["tags"])


async def run_async_tests():
    """Run all async tests."""
    # Get all async test methods
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    for test_class in [TestBlogAPIInterface, TestBlogContentGenerator, TestBlogPublishAction, TestIntegration]:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run async tests manually
    print("Running async tests...")

    # Test BlogAPIInterface async methods
    api_tests = TestBlogAPIInterface()
    api_tests.setUp()

    try:
        await api_tests.test_publish_post_circuit_breaker_open()
        print("✓ test_publish_post_circuit_breaker_open")
    except Exception as e:
        print(f"✗ test_publish_post_circuit_breaker_open: {e}")

    try:
        await api_tests.test_publish_post_success()
        print("✓ test_publish_post_success")
    except Exception as e:
        print(f"✗ test_publish_post_success: {e}")

    try:
        await api_tests.test_publish_post_auth_failure()
        print("✓ test_publish_post_auth_failure")
    except Exception as e:
        print(f"✗ test_publish_post_auth_failure: {e}")

    # Test BlogContentGenerator async methods
    gen_tests = TestBlogContentGenerator()
    gen_tests.setUp()

    try:
        await gen_tests.test_extract_tags_with_llm()
        print("✓ test_extract_tags_with_llm")
    except Exception as e:
        print(f"✗ test_extract_tags_with_llm: {e}")

    # Test BlogPublishAction async methods
    action_tests = TestBlogPublishAction()
    action_tests.setUp()

    try:
        await action_tests.test_execute_blog_disabled()
        print("✓ test_execute_blog_disabled")
    except Exception as e:
        print(f"✗ test_execute_blog_disabled: {e}")

    try:
        await action_tests.test_execute_missing_topic()
        print("✓ test_execute_missing_topic")
    except Exception as e:
        print(f"✗ test_execute_missing_topic: {e}")

    try:
        await action_tests.test_execute_dry_run()
        print("✓ test_execute_dry_run")
    except Exception as e:
        print(f"✗ test_execute_dry_run: {e}")

    # Test integration
    integration_tests = TestIntegration()

    try:
        await integration_tests.test_full_workflow_dry_run()
        print("✓ test_full_workflow_dry_run")
    except Exception as e:
        print(f"✗ test_full_workflow_dry_run: {e}")


def main():
    """Run all tests."""
    print("Running RAVANA Blog Integration Unit Tests")
    print("=" * 50)

    # Run synchronous tests
    print("Running synchronous tests...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add sync test classes
    sync_classes = [TestCircuitBreaker, TestBlogContentValidator]
    for test_class in sync_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    sync_result = runner.run(suite)

    # Run async tests
    print("\n" + "=" * 50)
    asyncio.run(run_async_tests())

    print("\n" + "=" * 50)
    print("Unit test summary:")
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

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
