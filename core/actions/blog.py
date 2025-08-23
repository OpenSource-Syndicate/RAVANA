"""
Blog Publishing Action for RAVANA AGI System

This module implements the main BlogPublishAction that integrates content generation,
API communication, and memory management for autonomous blog publishing.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List
from core.actions.action import Action
from core.actions.blog_api import BlogAPIInterface, BlogAPIError
from core.actions.blog_content_generator import BlogContentGenerator, BlogContentError
from core.config import Config

logger = logging.getLogger(__name__)

class BlogPublishAction(Action):
    """
    Action for publishing blog posts to the RAVANA blog platform.
    
    This action orchestrates the complete blog publishing workflow:
    1. Content generation using LLM with memory context
    2. Content validation and formatting
    3. API communication with the blog platform
    4. Result logging and memory storage
    """
    
    def __init__(self, system: 'AGISystem', data_service: 'DataService'):
        super().__init__(system, data_service)
        self.api_interface = BlogAPIInterface()
        self.content_generator = BlogContentGenerator(
            memory_service=getattr(system, 'memory_service', None),
            data_service=data_service,
            system=system
        )
    
    @property
    def name(self) -> str:
        return "publish_blog_post"
    
    @property
    def description(self) -> str:
        return (
            "Publishes a blog post to the RAVANA blog platform. "
            "Generates content using AI, formats it in markdown, "
            "and publishes it with appropriate tags."
        )
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "topic",
                "type": "string",
                "description": "Main topic or subject for the blog post",
                "required": True,
            },
            {
                "name": "style",
                "type": "string",
                "description": f"Writing style: {', '.join(Config.BLOG_CONTENT_STYLES)}",
                "required": False,
            },
            {
                "name": "context",
                "type": "string",
                "description": "Additional context or specific aspects to focus on",
                "required": False,
            },
            {
                "name": "custom_tags",
                "type": "array",
                "description": "Custom tags to include (in addition to auto-generated)",
                "required": False,
            },
            {
                "name": "dry_run",
                "type": "boolean",
                "description": "If true, generates content but doesn't publish to the blog",
                "required": False,
            },
        ]
    
    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes the blog publishing workflow.
        
        Args:
            **kwargs: Action parameters including topic, style, context, custom_tags, dry_run
            
        Returns:
            Dict containing execution results, published URL, and metadata
        """
        try:
            # Validate configuration
            if not Config.BLOG_ENABLED:
                return {
                    "status": "skipped",
                    "message": "Blog publishing is disabled in configuration",
                    "error": "BLOG_DISABLED"
                }
            
            # Extract and validate parameters
            topic = kwargs.get("topic", "").strip()
            if not topic:
                return {
                    "status": "error",
                    "message": "Topic parameter is required and cannot be empty",
                    "error": "MISSING_TOPIC"
                }
            
            style = kwargs.get("style", Config.BLOG_DEFAULT_STYLE)
            context = kwargs.get("context")
            custom_tags = kwargs.get("custom_tags", [])
            dry_run = kwargs.get("dry_run", False)
            
            logger.info(f"Starting blog publish action: topic='{topic}', style='{style}', dry_run={dry_run}")
            
            # Generate content
            generation_start = datetime.now()
            title, content, tags = await self.content_generator.generate_post(
                topic=topic,
                style=style,
                context=context,
                custom_tags=custom_tags
            )
            generation_time = (datetime.now() - generation_start).total_seconds()
            
            # Prepare result metadata
            result = {
                "status": "success",
                "title": title,
                "content_length": len(content),
                "tags": tags,
                "generation_time_seconds": generation_time,
                "style": style,
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
            }
            
            # If dry run, return without publishing
            if dry_run:
                result.update({
                    "message": "Content generated successfully (dry run - not published)",
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                    "dry_run": True
                })
                await self._log_blog_action(result, "dry_run")
                return result
            
            # Validate API configuration
            if not self.api_interface.validate_config():
                return {
                    "status": "error",
                    "message": "Blog API configuration is invalid",
                    "error": "INVALID_CONFIG"
                }
            
            # Publish to blog platform
            publish_start = datetime.now()
            api_result = await self.api_interface.publish_post(title, content, tags)
            publish_time = (datetime.now() - publish_start).total_seconds()
            
            # Update result with publication details
            result.update({
                "message": "Blog post published successfully",
                "published_url": api_result.get("published_url"),
                "post_id": api_result.get("post_id"),
                "publish_time_seconds": publish_time,
                "api_response": api_result,
                "dry_run": False
            })
            
            # Log successful publication
            await self._log_blog_action(result, "published")
            
            logger.info(f"Blog post published successfully: '{title}' -> {result.get('published_url')}")
            return result
            
        except BlogContentError as e:
            error_result = {
                "status": "error",
                "message": f"Content generation failed: {e}",
                "error": "CONTENT_GENERATION_FAILED",
                "topic": kwargs.get("topic", ""),
                "timestamp": datetime.now().isoformat(),
            }
            await self._log_blog_action(error_result, "content_error")
            logger.error(f"Blog content generation failed: {e}")
            return error_result
            
        except BlogAPIError as e:
            error_result = {
                "status": "error",
                "message": f"Blog API error: {e}",
                "error": "API_ERROR",
                "topic": kwargs.get("topic", ""),
                "timestamp": datetime.now().isoformat(),
            }
            await self._log_blog_action(error_result, "api_error")
            logger.error(f"Blog API error: {e}")
            return error_result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "message": f"Unexpected error: {e}",
                "error": "UNEXPECTED_ERROR",
                "topic": kwargs.get("topic", ""),
                "timestamp": datetime.now().isoformat(),
            }
            await self._log_blog_action(error_result, "unexpected_error")
            logger.exception(f"Unexpected error in blog publish action: {e}")
            return error_result
    
    async def _log_blog_action(self, result: Dict[str, Any], action_type: str) -> None:
        """
        Logs the blog action result to the data service and memory system.
        
        Args:
            result: The action execution result
            action_type: Type of action (published, dry_run, content_error, api_error, unexpected_error)
        """
        try:
            # Log to data service
            if self.data_service:
                log_entry = {
                    "action": "publish_blog_post",
                    "action_type": action_type,
                    "result": result,
                    "timestamp": datetime.now().isoformat(),
                }
                
                # Use existing data service logging mechanism with correct parameters
                await asyncio.to_thread(
                    getattr(self.data_service, 'save_action_log', lambda *args: None),
                    "publish_blog_post",
                    log_entry,  # params
                    result.get("status", "unknown"),  # status
                    result  # result
                )
            
            # Log to memory service for future context
            if hasattr(self.system, 'memory_service') and self.system.memory_service:
                memory_content = self._format_memory_content(result, action_type)
                memory_tags = ["blog", "publishing", action_type]
                
                # Add topic-specific tags
                if result.get("tags"):
                    memory_tags.extend(result["tags"][:3])  # Limit to first 3 tags
                
                # Store in memory service
                try:
                    memories_to_save = [{
                        "text": memory_content,
                        "type": "episodic",
                        "tags": memory_tags,
                        "emotional_valence": self._calculate_emotional_valence(result, action_type),
                        "created_at": datetime.now().isoformat()
                    }]
                    
                    await self.system.memory_service.save_memories(memories_to_save)
                    logger.debug(f"Blog action logged to memory: {memory_content[:100]}...")
                    
                except Exception as e:
                    logger.warning(f"Failed to log to memory service: {e}")
                
        except Exception as e:
            logger.warning(f"Failed to log blog action: {e}")
    
    def _format_memory_content(self, result: Dict[str, Any], action_type: str) -> str:
        """
        Formats the blog action result for memory storage.
        """
        if action_type == "published":
            return (
                f"Published blog post: '{result.get('title', 'Untitled')}' "
                f"about {result.get('topic', 'unknown topic')} "
                f"({result.get('content_length', 0)} chars, {len(result.get('tags', []))} tags). "
                f"URL: {result.get('published_url', 'N/A')}"
            )
        elif action_type == "dry_run":
            return (
                f"Generated blog content (dry run): '{result.get('title', 'Untitled')}' "
                f"about {result.get('topic', 'unknown topic')} "
                f"({result.get('content_length', 0)} chars, {len(result.get('tags', []))} tags)"
            )
        elif action_type in ["content_error", "api_error", "unexpected_error"]:
            return (
                f"Blog publishing failed ({action_type}): {result.get('message', 'Unknown error')} "
                f"for topic '{result.get('topic', 'unknown')}'"
            )
        else:
            return f"Blog action completed: {action_type} - {result.get('message', 'No message')}"
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Tests the blog API connection without publishing content.
        
        Returns:
            Dict containing connection test results
        """
        try:
            if not Config.BLOG_ENABLED:
                return {
                    "status": "skipped",
                    "message": "Blog integration is disabled",
                    "connected": False
                }
            
            config_valid = self.api_interface.validate_config()
            if not config_valid:
                return {
                    "status": "error",
                    "message": "Blog API configuration is invalid",
                    "connected": False
                }
            
            connection_ok = await self.api_interface.test_connection()
            
            return {
                "status": "success" if connection_ok else "error",
                "message": "Connection test successful" if connection_ok else "Connection test failed",
                "connected": connection_ok,
                "api_url": Config.BLOG_API_URL,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.exception(f"Blog connection test failed: {e}")
            return {
                "status": "error",
                "message": f"Connection test error: {e}",
                "connected": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_emotional_valence(self, result: Dict[str, Any], action_type: str) -> float:
        """
        Calculates emotional valence for memory storage based on action result.
        
        Args:
            result: The action execution result
            action_type: Type of action performed
            
        Returns:
            Float between -1.0 (negative) and 1.0 (positive)
        """
        if action_type == "published":
            return 0.8  # Publishing successfully is positive
        elif action_type == "dry_run":
            return 0.4  # Content generation success is mildly positive
        elif action_type in ["content_error", "api_error"]:
            return -0.6  # Errors are negative
        elif action_type == "unexpected_error":
            return -0.8  # Unexpected errors are very negative
        else:
            return 0.0  # Neutral for unknown types
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """
        Returns current blog configuration information.
        
        Returns:
            Dict containing configuration details
        """
        return {
            "enabled": Config.BLOG_ENABLED,
            "api_url": Config.BLOG_API_URL,
            "default_style": Config.BLOG_DEFAULT_STYLE,
            "available_styles": Config.BLOG_CONTENT_STYLES,
            "content_length_limits": {
                "min": Config.BLOG_MIN_CONTENT_LENGTH,
                "max": Config.BLOG_MAX_CONTENT_LENGTH
            },
            "auto_tagging_enabled": Config.BLOG_AUTO_TAGGING_ENABLED,
            "max_tags": Config.BLOG_MAX_TAGS,
            "timeout_seconds": Config.BLOG_TIMEOUT_SECONDS,
            "retry_attempts": Config.BLOG_RETRY_ATTEMPTS,
            "auto_publish_enabled": Config.BLOG_AUTO_PUBLISH_ENABLED,
            "require_approval": Config.BLOG_REQUIRE_APPROVAL
        }