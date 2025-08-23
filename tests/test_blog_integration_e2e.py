#!/usr/bin/env python3
"""
End-to-End Integration Tests for RAVANA Blog Integration

This module provides comprehensive integration tests including:
- Mock API server testing
- End-to-end workflow validation
- Error scenario testing
- Performance testing
- Real API compatibility testing
"""

import asyncio
import json
import sys
import os
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch
import aiohttp
from aiohttp import web
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.actions.blog import BlogPublishAction
from core.actions.blog_api import BlogAPIInterface, BlogAPIError
from core.actions.blog_content_generator import BlogContentGenerator

class MockBlogAPIServer:
    """Mock blog API server for testing."""
    
    def __init__(self, port=8899):
        self.port = port
        self.app = web.Application()
        self.setup_routes()
        self.request_count = 0
        self.last_request = None
        
    def setup_routes(self):
        """Setup API routes."""
        self.app.router.add_post('/api/publish', self.handle_publish)
        self.app.router.add_get('/health', self.handle_health)
        
    async def handle_health(self, request):
        """Health check endpoint."""
        return web.json_response({"status": "healthy"})
        
    async def handle_publish(self, request):
        """Handle blog publish requests."""
        self.request_count += 1
        
        # Parse request
        try:
            data = await request.json()
            self.last_request = data
        except Exception:
            return web.json_response(
                {"success": False, "message": "Invalid JSON"},
                status=400
            )
        
        # Check authorization
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return web.json_response(
                {"success": False, "message": "Missing or invalid authorization"},
                status=401
            )
        
        token = auth_header[7:]  # Remove "Bearer "
        if token != "test_token_123":
            return web.json_response(
                {"success": False, "message": "Invalid token"},
                status=401
            )
        
        # Validate required fields
        required_fields = ['title', 'content', 'tags']
        for field in required_fields:
            if field not in data:
                return web.json_response(
                    {"success": False, "message": f"Missing field: {field}"},
                    status=400
                )
        
        # Simulate different responses based on title
        title = data.get('title', '')
        
        if title == "RATE_LIMIT_TEST":
            return web.json_response(
                {"success": False, "message": "Rate limited"},
                status=429,
                headers={'Retry-After': '30'}
            )
        
        if title == "SERVER_ERROR_TEST":
            return web.json_response(
                {"success": False, "message": "Internal server error"},
                status=500
            )
        
        if title == "TIMEOUT_TEST":
            await asyncio.sleep(5)  # Simulate timeout
            
        # Successful response
        return web.json_response({
            "success": True,
            "message": "Post published successfully",
            "post_id": f"post_{self.request_count}",
            "published_url": f"http://test-blog.com/posts/post_{self.request_count}"
        })
    
    async def start(self):
        """Start the mock server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, 'localhost', self.port)
        await self.site.start()
        print(f"Mock blog API server started on http://localhost:{self.port}")
        
    async def stop(self):
        """Stop the mock server."""
        if hasattr(self, 'runner'):
            await self.runner.cleanup()

class BlogIntegrationTester:
    """Comprehensive blog integration tester."""
    
    def __init__(self):
        self.mock_server = MockBlogAPIServer()
        self.test_results = []
        
    async def setup(self):
        """Setup test environment."""
        await self.mock_server.start()
        
        # Wait a moment for server to start
        await asyncio.sleep(0.1)
        
    async def teardown(self):
        """Cleanup test environment."""
        await self.mock_server.stop()
        
    async def run_all_tests(self):
        """Run all integration tests."""
        print("🧪 Running Blog Integration Tests")
        print("=" * 50)
        
        await self.setup()
        
        try:
            # Test categories
            await self.test_basic_functionality()
            await self.test_error_scenarios()
            await self.test_content_generation()
            await self.test_validation_scenarios()
            await self.test_performance()
            await self.test_real_api_compatibility()
            
        finally:
            await self.teardown()
            
        self.print_summary()
        
    async def test_basic_functionality(self):
        """Test basic blog publishing functionality."""
        print("\\n📝 Testing Basic Functionality")
        print("-" * 30)
        
        # Test 1: Successful blog post publication
        try:
            api = BlogAPIInterface()
            # Override with test server
            api.api_url = f"http://localhost:{self.mock_server.port}/api/publish"
            api.auth_token = "test_token_123"
            
            result = await api.publish_post(
                "Test Blog Post",
                "# Test Content\\n\\nThis is test content for the blog integration.",
                ["test", "integration"]
            )
            
            assert result["success"] == True
            assert "post_id" in result
            assert "published_url" in result
            
            self.test_results.append(("Basic Publication", "PASSED", ""))
            print("✓ Basic publication test passed")
            
        except Exception as e:
            self.test_results.append(("Basic Publication", "FAILED", str(e)))
            print(f"✗ Basic publication test failed: {e}")
        
        # Test 2: End-to-end action workflow
        try:
            mock_system = Mock()
            mock_data_service = Mock()
            
            action = BlogPublishAction(mock_system, mock_data_service)
            action.api_interface.api_url = f"http://localhost:{self.mock_server.port}/api/publish"
            action.api_interface.auth_token = "test_token_123"
            
            # Mock LLM response
            with patch('core.actions.blog_content_generator.async_safe_call_llm') as mock_llm:
                mock_llm.return_value = '''
                {
                    "title": "Integration Test Post",
                    "content": "# Integration Test\\n\\nThis is a comprehensive integration test for the RAVANA blog system. The content includes multiple sections and demonstrates the full workflow from content generation to publication.\\n\\n## Features Tested\\n\\n- Content generation\\n- API integration\\n- Error handling\\n- Validation"
                }
                '''
                
                result = await action.execute(
                    topic="Blog Integration Testing",
                    style="technical",
                    dry_run=False
                )
                
                assert result["status"] == "success"
                assert "published_url" in result
                assert result["dry_run"] == False
                
                self.test_results.append(("End-to-End Workflow", "PASSED", ""))
                print("✓ End-to-end workflow test passed")
                
        except Exception as e:
            self.test_results.append(("End-to-End Workflow", "FAILED", str(e)))
            print(f"✗ End-to-end workflow test failed: {e}")
    
    async def test_error_scenarios(self):
        """Test various error scenarios."""
        print("\\n⚠️  Testing Error Scenarios")
        print("-" * 30)
        
        api = BlogAPIInterface()
        api.api_url = f"http://localhost:{self.mock_server.port}/api/publish"
        
        # Test 1: Authentication failure
        try:
            api.auth_token = "invalid_token"
            
            try:
                await api.publish_post("Test", "Content", ["test"])
                self.test_results.append(("Auth Failure", "FAILED", "Should have raised exception"))
                print("✗ Auth failure test failed: Should have raised exception")
            except BlogAPIError as e:
                if e.error_type == "AUTHENTICATION_FAILED":
                    self.test_results.append(("Auth Failure", "PASSED", ""))
                    print("✓ Auth failure test passed")
                else:
                    self.test_results.append(("Auth Failure", "FAILED", f"Wrong error type: {e.error_type}"))
                    print(f"✗ Auth failure test failed: Wrong error type: {e.error_type}")
                    
        except Exception as e:
            self.test_results.append(("Auth Failure", "FAILED", str(e)))
            print(f"✗ Auth failure test failed: {e}")
        
        # Test 2: Rate limiting
        try:
            api.auth_token = "test_token_123"
            
            try:
                await api.publish_post("RATE_LIMIT_TEST", "Content", ["test"])
                self.test_results.append(("Rate Limiting", "FAILED", "Should have raised exception"))
                print("✗ Rate limiting test failed: Should have raised exception")
            except BlogAPIError as e:
                if e.error_type == "RATE_LIMITED" and e.retry_after == 30:
                    self.test_results.append(("Rate Limiting", "PASSED", ""))
                    print("✓ Rate limiting test passed")
                else:
                    self.test_results.append(("Rate Limiting", "FAILED", f"Wrong error handling: {e.error_type}"))
                    print(f"✗ Rate limiting test failed: Wrong error handling: {e.error_type}")
                    
        except Exception as e:
            self.test_results.append(("Rate Limiting", "FAILED", str(e)))
            print(f"✗ Rate limiting test failed: {e}")
        
        # Test 3: Server error
        try:
            try:
                await api.publish_post("SERVER_ERROR_TEST", "Content", ["test"])
                self.test_results.append(("Server Error", "FAILED", "Should have raised exception"))
                print("✗ Server error test failed: Should have raised exception")
            except BlogAPIError as e:
                if e.error_type == "SERVER_ERROR":
                    self.test_results.append(("Server Error", "PASSED", ""))
                    print("✓ Server error test passed")
                else:
                    self.test_results.append(("Server Error", "FAILED", f"Wrong error type: {e.error_type}"))
                    print(f"✗ Server error test failed: Wrong error type: {e.error_type}")
                    
        except Exception as e:
            self.test_results.append(("Server Error", "FAILED", str(e)))
            print(f"✗ Server error test failed: {e}")
        
        # Test 4: Circuit breaker
        try:
            # Force circuit breaker to open by recording failures
            for _ in range(5):
                api.circuit_breaker.record_failure()
            
            try:
                await api.publish_post("Test", "Content", ["test"])
                self.test_results.append(("Circuit Breaker", "FAILED", "Should have raised exception"))
                print("✗ Circuit breaker test failed: Should have raised exception")
            except BlogAPIError as e:
                if e.error_type == "CIRCUIT_BREAKER_OPEN":
                    self.test_results.append(("Circuit Breaker", "PASSED", ""))
                    print("✓ Circuit breaker test passed")
                else:
                    self.test_results.append(("Circuit Breaker", "FAILED", f"Wrong error type: {e.error_type}"))
                    print(f"✗ Circuit breaker test failed: Wrong error type: {e.error_type}")
                    
        except Exception as e:
            self.test_results.append(("Circuit Breaker", "FAILED", str(e)))
            print(f"✗ Circuit breaker test failed: {e}")
    
    async def test_content_generation(self):
        """Test content generation scenarios."""
        print("\\n📝 Testing Content Generation")
        print("-" * 30)
        
        # Test 1: Content generation with memory context
        try:
            # Mock memory service
            mock_memory_service = Mock()
            mock_memory_service.get_relevant_memories = AsyncMock(return_value=Mock(relevant_memories=[]))
            
            generator = BlogContentGenerator(memory_service=mock_memory_service)
            
            with patch('core.actions.blog_content_generator.async_safe_call_llm') as mock_llm:
                mock_llm.side_effect = [
                    '''
                    {
                        "title": "AI and Machine Learning Fundamentals",
                        "content": "# AI and Machine Learning Fundamentals\\n\\nArtificial Intelligence and Machine Learning are transforming technology. This comprehensive guide explores key concepts and applications.\\n\\n## Core Concepts\\n\\nMachine learning algorithms enable computers to learn from data without explicit programming. Key types include supervised, unsupervised, and reinforcement learning.\\n\\n## Applications\\n\\nAI is revolutionizing industries from healthcare to finance, enabling predictive analytics and automated decision-making."
                    }
                    ''',
                    '["ai", "machine-learning", "technology", "algorithms"]'
                ]
                
                title, content, tags = await generator.generate_post(
                    topic="AI and Machine Learning",
                    style="technical",
                    context="Introduction to fundamental concepts"
                )
                
                assert len(title) > 5
                assert len(content) > 100
                assert len(tags) > 0
                assert "ai" in [tag.lower() for tag in tags]
                
                self.test_results.append(("Content Generation", "PASSED", ""))
                print("✓ Content generation test passed")
                
        except Exception as e:
            self.test_results.append(("Content Generation", "FAILED", str(e)))
            print(f"✗ Content generation test failed: {e}")
        
        # Test 2: Multiple writing styles
        try:
            generator = BlogContentGenerator()
            
            styles_tested = []
            for style in ["technical", "casual", "creative"]:
                guidance = generator._get_style_guidance(style)
                assert len(guidance) > 10
                styles_tested.append(style)
            
            assert len(styles_tested) == 3
            
            self.test_results.append(("Style Variations", "PASSED", ""))
            print("✓ Style variations test passed")
            
        except Exception as e:
            self.test_results.append(("Style Variations", "FAILED", str(e)))
            print(f"✗ Style variations test failed: {e}")
    
    async def test_validation_scenarios(self):
        """Test content validation scenarios."""
        print("\\n✅ Testing Validation Scenarios")
        print("-" * 30)
        
        # Test 1: Content validation success
        try:
            from core.actions.blog_content_validator import BlogContentValidator
            
            validator = BlogContentValidator()
            
            title = "Understanding Quantum Computing"
            content = """# Understanding Quantum Computing
            
Quantum computing represents a paradigm shift in computational technology. This article explores the fundamental principles and potential applications.

## Quantum Principles

Quantum computers leverage quantum mechanical phenomena such as superposition and entanglement to process information in ways classical computers cannot.

## Applications

Potential applications include cryptography, optimization problems, and drug discovery. The field is rapidly evolving with significant investments from major technology companies.

## Challenges

Current challenges include quantum decoherence, error rates, and the need for extremely low temperatures to maintain quantum states.
            """
            tags = ["quantum", "computing", "technology", "physics"]
            
            validated_title, validated_content, validated_tags, report = \\
                validator.validate_and_sanitize(title, content, tags)
            
            assert report["validation_passed"] == True
            assert report["quality_score"] > 3.0
            assert len(validated_tags) > 0
            
            self.test_results.append(("Content Validation", "PASSED", ""))
            print("✓ Content validation test passed")
            
        except Exception as e:
            self.test_results.append(("Content Validation", "FAILED", str(e)))
            print(f"✗ Content validation test failed: {e}")
        
        # Test 2: Security sanitization
        try:
            validator = BlogContentValidator()
            
            malicious_content = """# Test Post
            
This is a test post with <script>alert('xss')</script> malicious content.

<iframe src="http://evil.com"></iframe>

Normal content continues here.
            """
            
            clean_content = validator._remove_security_threats(malicious_content, "content")
            
            assert "<script>" not in clean_content
            assert "<iframe>" not in clean_content
            assert "Normal content" in clean_content
            
            self.test_results.append(("Security Sanitization", "PASSED", ""))
            print("✓ Security sanitization test passed")
            
        except Exception as e:
            self.test_results.append(("Security Sanitization", "FAILED", str(e)))
            print(f"✗ Security sanitization test failed: {e}")
    
    async def test_performance(self):
        """Test performance scenarios."""
        print("\\n⚡ Testing Performance")
        print("-" * 30)
        
        # Test 1: Response time
        try:
            api = BlogAPIInterface()
            api.api_url = f"http://localhost:{self.mock_server.port}/api/publish"
            api.auth_token = "test_token_123"
            
            start_time = time.time()
            result = await api.publish_post(
                "Performance Test",
                "This is a performance test content",
                ["performance", "test"]
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            assert result["success"] == True
            assert response_time < 2.0  # Should complete within 2 seconds
            
            self.test_results.append(("Response Time", "PASSED", f"{response_time:.2f}s"))
            print(f"✓ Response time test passed ({response_time:.2f}s)")
            
        except Exception as e:
            self.test_results.append(("Response Time", "FAILED", str(e)))
            print(f"✗ Response time test failed: {e}")
        
        # Test 2: Multiple concurrent requests
        try:
            api = BlogAPIInterface()
            api.api_url = f"http://localhost:{self.mock_server.port}/api/publish"
            api.auth_token = "test_token_123"
            
            # Create multiple concurrent requests
            tasks = []
            for i in range(3):
                task = api.publish_post(
                    f"Concurrent Test {i}",
                    f"This is concurrent test content {i}",
                    ["concurrent", "test", f"test{i}"]
                )
                tasks.append(task)
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            total_time = end_time - start_time
            
            # Check all succeeded
            success_count = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
            
            assert success_count == 3
            assert total_time < 5.0  # All should complete within 5 seconds
            
            self.test_results.append(("Concurrent Requests", "PASSED", f"{success_count}/3 in {total_time:.2f}s"))
            print(f"✓ Concurrent requests test passed ({success_count}/3 in {total_time:.2f}s)")
            
        except Exception as e:
            self.test_results.append(("Concurrent Requests", "FAILED", str(e)))
            print(f"✗ Concurrent requests test failed: {e}")
    
    async def test_real_api_compatibility(self):
        """Test compatibility with real API format."""
        print("\\n🌐 Testing Real API Compatibility")
        print("-" * 30)
        
        # Test 1: Request format compatibility
        try:
            # Capture the actual request made to our mock server
            api = BlogAPIInterface()
            api.api_url = f"http://localhost:{self.mock_server.port}/api/publish"
            api.auth_token = "test_token_123"
            
            await api.publish_post(
                "Time Dilation Hack",
                "# Understanding Time Dilation\\n\\nTime dilation is a fascinating concept from Einstein's theory of relativity...",
                ["physics", "time", "gravity", "spacetime"]
            )
            
            # Check the captured request format
            request = self.mock_server.last_request
            
            assert "title" in request
            assert "content" in request
            assert "tags" in request
            assert request["title"] == "Time Dilation Hack"
            assert isinstance(request["tags"], list)
            assert "physics" in request["tags"]
            
            self.test_results.append(("API Format Compatibility", "PASSED", ""))
            print("✓ API format compatibility test passed")
            
        except Exception as e:
            self.test_results.append(("API Format Compatibility", "FAILED", str(e)))
            print(f"✗ API format compatibility test failed: {e}")
        
        # Test 2: cURL command equivalence
        try:
            # Test that our implementation produces the same result as the provided cURL command
            expected_payload = {
                "title": "Time Dilation Hack",
                "content": "# Understanding Time Dilation\\n\\nTime dilation is a fascinating concept...",
                "tags": ["physics", "time", "gravity", "spacetime"]
            }
            
            api = BlogAPIInterface()
            api.api_url = f"http://localhost:{self.mock_server.port}/api/publish"
            api.auth_token = "test_token_123"
            
            result = await api.publish_post(
                expected_payload["title"],
                expected_payload["content"],
                expected_payload["tags"]
            )
            
            # Verify the request was formatted correctly
            actual_request = self.mock_server.last_request
            
            assert actual_request["title"] == expected_payload["title"]
            assert isinstance(actual_request["content"], str)
            assert actual_request["tags"] == expected_payload["tags"]
            assert result["success"] == True
            
            self.test_results.append(("cURL Equivalence", "PASSED", ""))
            print("✓ cURL equivalence test passed")
            
        except Exception as e:
            self.test_results.append(("cURL Equivalence", "FAILED", str(e)))
            print(f"✗ cURL equivalence test failed: {e}")
    
    def print_summary(self):
        """Print test summary."""
        print("\\n" + "=" * 50)
        print("🧪 INTEGRATION TEST SUMMARY")
        print("=" * 50)
        
        passed = 0
        failed = 0
        
        for test_name, status, details in self.test_results:
            status_icon = "✓" if status == "PASSED" else "✗"
            detail_str = f" ({details})" if details else ""
            print(f"{status_icon} {test_name:.<40} {status}{detail_str}")
            
            if status == "PASSED":
                passed += 1
            else:
                failed += 1
        
        print("-" * 50)
        print(f"Total tests: {len(self.test_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success rate: {(passed / len(self.test_results) * 100):.1f}%")
        
        if failed == 0:
            print("\\n🎉 All integration tests PASSED!")
        else:
            print(f"\\n⚠️  {failed} test(s) FAILED. Check the details above.")
        
        return failed == 0

async def main():
    """Run all integration tests."""
    tester = BlogIntegrationTester()
    success = await tester.run_all_tests()
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)