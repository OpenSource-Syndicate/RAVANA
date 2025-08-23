"""
Blog API Interface for RAVANA AGI System

This module provides secure HTTP communication with the RAVANA blog platform,
handling authentication, request formatting, and response processing.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import aiohttp
from core.config import Config

logger = logging.getLogger(__name__)

class BlogAPIError(Exception):
    """Custom exception for blog API related errors."""
    def __init__(self, message: str, error_type: str = "UNKNOWN", retry_after: Optional[int] = None):
        super().__init__(message)
        self.error_type = error_type
        self.retry_after = retry_after

class CircuitBreaker:
    """Simple circuit breaker implementation for API resilience."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def can_execute(self) -> bool:
        """Check if the circuit allows execution."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"
        
    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class BlogAPIInterface:
    """
    Interface for communicating with the RAVANA blog platform API.
    
    Handles secure authentication, request formatting, response processing,
    and implements retry logic with exponential backoff for resilience.
    """
    
    def __init__(self):
        self.api_url = Config.BLOG_API_URL
        self.auth_token = Config.BLOG_AUTH_TOKEN
        self.timeout = Config.BLOG_TIMEOUT_SECONDS
        self.retry_attempts = Config.BLOG_RETRY_ATTEMPTS
        self.backoff_factor = Config.BLOG_RETRY_BACKOFF_FACTOR
        self.max_retry_delay = Config.BLOG_MAX_RETRY_DELAY
        self.circuit_breaker = CircuitBreaker()
        self.rate_limit_reset = None
        
    async def publish_post(self, title: str, content: str, tags: List[str]) -> Dict[str, Any]:
        """
        Publishes a blog post to the RAVANA blog platform.
        
        Args:
            title: The title of the blog post
            content: The markdown content of the blog post
            tags: List of tags to associate with the post
            
        Returns:
            Dict containing the API response with success status, message, and post details
            
        Raises:
            BlogAPIError: If the API request fails after all retry attempts
        """
        payload = {
            "title": title,
            "content": content,
            "tags": tags
        }
        
        logger.info(f"Publishing blog post: '{title}' with {len(tags)} tags")
        
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise BlogAPIError(
                "Circuit breaker is OPEN - too many recent failures",
                error_type="CIRCUIT_BREAKER_OPEN",
                retry_after=300
            )
        
        # Check rate limiting
        if self.rate_limit_reset and time.time() < self.rate_limit_reset:
            wait_time = int(self.rate_limit_reset - time.time())
            raise BlogAPIError(
                f"Rate limited - retry after {wait_time} seconds",
                error_type="RATE_LIMITED",
                retry_after=wait_time
            )
        
        for attempt in range(self.retry_attempts):
            try:
                result = await self._make_request(payload)
                logger.info(f"Blog post published successfully: {result.get('published_url', 'URL not provided')}")
                self.circuit_breaker.record_success()
                return result
                
            except BlogAPIError as e:
                self.circuit_breaker.record_failure()
                
                # Don't retry certain error types
                if e.error_type in ["AUTHENTICATION_FAILED", "CIRCUIT_BREAKER_OPEN", "INVALID_REQUEST"]:
                    logger.error(f"Non-retryable error: {e}")
                    raise
                
                # Handle rate limiting specially
                if e.error_type == "RATE_LIMITED" and e.retry_after:
                    self.rate_limit_reset = time.time() + e.retry_after
                    if attempt == self.retry_attempts - 1:
                        raise
                    logger.warning(f"Rate limited, waiting {e.retry_after} seconds...")
                    await asyncio.sleep(e.retry_after)
                    continue
                
                if attempt == self.retry_attempts - 1:
                    logger.error(f"Failed to publish blog post after {self.retry_attempts} attempts: {e}")
                    raise
                
                delay = min(
                    self.backoff_factor ** attempt,
                    self.max_retry_delay
                )
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
        
    async def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Makes the actual HTTP request to the blog API.
        
        Args:
            payload: The request payload containing title, content, and tags
            
        Returns:
            Parsed JSON response from the API
            
        Raises:
            BlogAPIError: If the request fails or returns an error response
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}",
            "User-Agent": "RAVANA-AGI/1.0"
        }
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.api_url,
                    json=payload,
                    headers=headers
                ) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        try:
                            result = json.loads(response_text)
                            return self._handle_response(result)
                        except json.JSONDecodeError as e:
                            raise BlogAPIError(f"Invalid JSON response: {e}", "INVALID_RESPONSE")
                    
                    elif response.status == 401:
                        raise BlogAPIError("Authentication failed - check auth token", "AUTHENTICATION_FAILED")
                    
                    elif response.status == 400:
                        raise BlogAPIError(f"Invalid request: {response_text}", "INVALID_REQUEST")
                    
                    elif response.status == 429:
                        # Extract retry-after header if available
                        retry_after = response.headers.get('Retry-After', '60')
                        try:
                            retry_after_seconds = int(retry_after)
                        except ValueError:
                            retry_after_seconds = 60
                        
                        raise BlogAPIError(
                            "Rate limit exceeded", 
                            "RATE_LIMITED", 
                            retry_after=retry_after_seconds
                        )
                    
                    elif response.status >= 500:
                        raise BlogAPIError(f"Server error ({response.status}): {response_text}", "SERVER_ERROR")
                    
                    else:
                        raise BlogAPIError(f"HTTP {response.status}: {response_text}", "HTTP_ERROR")
                        
        except aiohttp.ClientError as e:
            raise BlogAPIError(f"Network error: {e}", "NETWORK_ERROR")
        
        except asyncio.TimeoutError:
            raise BlogAPIError(f"Request timeout after {self.timeout} seconds", "TIMEOUT")
    
    def _handle_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes and validates the API response.
        
        Args:
            response: Raw JSON response from the API
            
        Returns:
            Processed response dictionary
            
        Raises:
            BlogAPIError: If the response indicates an error
        """
        if not isinstance(response, dict):
            raise BlogAPIError("Invalid response format - expected JSON object", "INVALID_RESPONSE")
        
        success = response.get("success", False)
        message = response.get("message", "No message provided")
        
        if not success:
            raise BlogAPIError(f"API error: {message}", "API_ERROR")
        
        # Ensure required fields are present
        result = {
            "success": True,
            "message": message,
            "post_id": response.get("post_id"),
            "published_url": response.get("published_url"),
            "timestamp": time.time()
        }
        
        logger.debug(f"Blog API response: {result}")
        return result
    
    def validate_config(self) -> bool:
        """
        Validates the blog API configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        if not Config.BLOG_ENABLED:
            logger.info("Blog integration is disabled")
            return False
            
        if not self.api_url:
            logger.error("Blog API URL is not configured")
            return False
            
        if not self.auth_token:
            logger.error("Blog auth token is not configured")
            return False
            
        if self.timeout <= 0:
            logger.error("Invalid timeout configuration")
            return False
            
        return True
    
    async def test_connection(self) -> bool:
        """
        Tests the connection to the blog API with a minimal request.
        
        Returns:
            True if connection test passes, False otherwise
        """
        if not self.validate_config():
            return False
            
        try:
            # Use a minimal test payload
            test_payload = {
                "title": "Connection Test",
                "content": "# Test\nThis is a connection test.",
                "tags": ["test"]
            }
            
            # Make a test request but expect it might fail due to test content
            await self._make_request(test_payload)
            return True
            
        except BlogAPIError as e:
            logger.warning(f"Blog API connection test failed: {e}")
            return False
        
        except Exception as e:
            logger.error(f"Unexpected error during connection test: {e}")
            return False