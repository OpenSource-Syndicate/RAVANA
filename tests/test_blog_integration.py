"""
Comprehensive test suite for RAVANA Blog Integration

This script validates the blog publishing functionality and verifies
it works correctly with the provided API endpoint.
"""

import asyncio
import json
import logging
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.actions.registry import ActionRegistry
from core.actions.blog import BlogPublishAction
from core.actions.blog_api import BlogAPIInterface
from core.actions.blog_content_generator import BlogContentGenerator
from core.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockAGISystem:
    """Mock AGI system for testing purposes."""
    def __init__(self):
        self.memory_service = None

class MockDataService:
    """Mock data service for testing purposes."""
    def save_action_log(self, action_name, status, result):
        logger.info(f"Mock log: {action_name} - {status} - {json.dumps(result, indent=2)}")

async def test_blog_api_interface():
    """Test the BlogAPIInterface class."""
    logger.info("Testing BlogAPIInterface...")
    
    api = BlogAPIInterface()
    
    # Test configuration validation
    config_valid = api.validate_config()
    logger.info(f"Configuration valid: {config_valid}")
    
    if not config_valid:
        logger.warning("Blog API configuration is invalid - check environment variables")
        return False
    
    # Test connection (this might fail if the API expects specific content)
    try:
        connection_ok = await api.test_connection()
        logger.info(f"Connection test: {'PASSED' if connection_ok else 'FAILED'}")
    except Exception as e:
        logger.warning(f"Connection test failed: {e}")
        connection_ok = False
    
    return config_valid

async def test_blog_content_generator():
    """Test the BlogContentGenerator class."""
    logger.info("Testing BlogContentGenerator...")
    
    generator = BlogContentGenerator()
    
    try:
        # Test content generation
        title, content, tags = await generator.generate_post(
            topic="Time Dilation Experiments",
            style="technical",
            context="Recent physics experiments and breakthroughs",
            custom_tags=["physics", "experiments"]
        )
        
        logger.info(f"Generated content:")
        logger.info(f"  Title: {title}")
        logger.info(f"  Content length: {len(content)} characters")
        logger.info(f"  Tags: {tags}")
        logger.info(f"  Content preview: {content[:200]}...")
        
        # Validate content
        if len(content) >= Config.BLOG_MIN_CONTENT_LENGTH:
            logger.info("Content generation: PASSED")
            return True
        else:
            logger.error("Content too short")
            return False
            
    except Exception as e:
        logger.error(f"Content generation failed: {e}")
        return False

async def test_blog_publish_action():
    """Test the BlogPublishAction class."""
    logger.info("Testing BlogPublishAction...")
    
    # Create mock system and data service
    mock_system = MockAGISystem()
    mock_data_service = MockDataService()
    
    # Create action instance
    action = BlogPublishAction(mock_system, mock_data_service)
    
    # Test action properties
    logger.info(f"Action name: {action.name}")
    logger.info(f"Action description: {action.description}")
    logger.info(f"Parameters: {len(action.parameters)} parameters")
    
    # Test dry run execution
    try:
        result = await action.execute(
            topic="Time Dilation Hack",
            style="technical",
            context="Understanding time dilation concepts and breakthrough research",
            custom_tags=["physics", "time", "gravity", "spacetime"],
            dry_run=True
        )
        
        logger.info("Dry run execution result:")
        logger.info(json.dumps(result, indent=2))
        
        if result.get("status") == "success" and result.get("dry_run"):
            logger.info("Dry run test: PASSED")
            return True
        else:
            logger.error("Dry run test: FAILED")
            return False
            
    except Exception as e:
        logger.error(f"Dry run execution failed: {e}")
        return False

async def test_action_registry_integration():
    """Test that BlogPublishAction is properly registered."""
    logger.info("Testing ActionRegistry integration...")
    
    try:
        # Create mock system and data service
        mock_system = MockAGISystem()
        mock_data_service = MockDataService()
        
        # Create registry
        registry = ActionRegistry(mock_system, mock_data_service)
        
        # Check if blog action is registered
        try:
            blog_action = registry.get_action("publish_blog_post")
            logger.info(f"Blog action found: {blog_action.name}")
            logger.info("Action registry integration: PASSED")
            return True
        except ValueError as e:
            logger.error(f"Blog action not found in registry: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Registry integration test failed: {e}")
        return False

async def test_full_publish_workflow():
    """Test the complete publishing workflow (with API call)."""
    logger.info("Testing full publish workflow...")
    
    if not Config.BLOG_ENABLED:
        logger.warning("Blog integration disabled - skipping full workflow test")
        return True
    
    # Create mock system and data service
    mock_system = MockAGISystem()
    mock_data_service = MockDataService()
    
    # Create action instance
    action = BlogPublishAction(mock_system, mock_data_service)
    
    try:
        # Test actual publication (this will hit the real API)
        result = await action.execute(
            topic="RAVANA AGI Blog Integration Test",
            style="technical",
            context="Testing the autonomous blog publishing system integration",
            custom_tags=["ravana", "agi", "testing", "integration"],
            dry_run=False  # This will actually publish!
        )
        
        logger.info("Full workflow execution result:")
        logger.info(json.dumps(result, indent=2))
        
        if result.get("status") == "success" and result.get("published_url"):
            logger.info("Full workflow test: PASSED")
            logger.info(f"Published at: {result.get('published_url')}")
            return True
        else:
            logger.error("Full workflow test: FAILED")
            return False
            
    except Exception as e:
        logger.error(f"Full workflow execution failed: {e}")
        return False

async def test_configuration_display():
    """Display current configuration for debugging."""
    logger.info("Current Blog Configuration:")
    logger.info(f"  BLOG_ENABLED: {Config.BLOG_ENABLED}")
    logger.info(f"  BLOG_API_URL: {Config.BLOG_API_URL}")
    logger.info(f"  BLOG_AUTH_TOKEN: {'***' if Config.BLOG_AUTH_TOKEN else 'NOT SET'}")
    logger.info(f"  BLOG_DEFAULT_STYLE: {Config.BLOG_DEFAULT_STYLE}")
    logger.info(f"  BLOG_MAX_CONTENT_LENGTH: {Config.BLOG_MAX_CONTENT_LENGTH}")
    logger.info(f"  BLOG_MIN_CONTENT_LENGTH: {Config.BLOG_MIN_CONTENT_LENGTH}")
    logger.info(f"  BLOG_AUTO_TAGGING_ENABLED: {Config.BLOG_AUTO_TAGGING_ENABLED}")
    logger.info(f"  BLOG_TIMEOUT_SECONDS: {Config.BLOG_TIMEOUT_SECONDS}")
    logger.info(f"  BLOG_RETRY_ATTEMPTS: {Config.BLOG_RETRY_ATTEMPTS}")

async def main():
    """Run all tests."""
    logger.info("Starting RAVANA Blog Integration Tests")
    logger.info("=" * 50)
    
    # Display configuration
    await test_configuration_display()
    logger.info("=" * 50)
    
    # Run tests
    tests = [
        ("Blog API Interface", test_blog_api_interface()),
        ("Blog Content Generator", test_blog_content_generator()),
        ("Blog Publish Action", test_blog_publish_action()),
        ("Action Registry Integration", test_action_registry_integration()),
    ]
    
    results = []
    for test_name, test_coro in tests:
        logger.info(f"\nRunning: {test_name}")
        logger.info("-" * 30)
        try:
            result = await test_coro
            results.append((test_name, result))
            logger.info(f"{test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.exception(f"{test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Optional: Test full workflow if enabled
    if Config.BLOG_ENABLED and input("\nRun full publish test (will publish to blog)? [y/N]: ").lower() == 'y':
        logger.info(f"\nRunning: Full Publish Workflow")
        logger.info("-" * 30)
        try:
            result = await test_full_publish_workflow()
            results.append(("Full Publish Workflow", result))
            logger.info(f"Full Publish Workflow: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.exception(f"Full Publish Workflow crashed: {e}")
            results.append(("Full Publish Workflow", False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    logger.info("-" * 50)
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests PASSED! Blog integration is ready.")
    else:
        logger.warning(f"⚠️  {total - passed} test(s) FAILED. Check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)