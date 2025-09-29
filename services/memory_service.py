"""
Unified Memory Service for RAVANA AGI System

This module provides a comprehensive memory service that combines the functionality of
the original memory service with the enhanced memory service capabilities.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from core.enhanced_memory_service import EnhancedMemoryService, MemoryType, Memory
from core.embeddings_manager import embeddings_manager, ModelPurpose
from core.config import Config
from core.shutdown_coordinator import Shutdownable

logger = logging.getLogger(__name__)


class MemoryService(EnhancedMemoryService, Shutdownable):
    """
    Unified Memory Service that combines basic memory operations with enhanced capabilities.
    """
    
    def __init__(self):
        # Initialize the enhanced memory service functionality
        EnhancedMemoryService.__init__(self)
        
        # Additional service-specific attributes if needed
        self.service_host = "localhost"
        self.service_port = 8001

    # EnhancedMemoryService methods are inherited, but we can override as needed
    # For example, if we need to hook into memory operations for additional processing:
    
    async def save_memories(self, memories) -> bool:
        """
        Save memories to the memory store.
        
        Args:
            memories: List of memory objects or dictionaries to save
        
        Returns:
            True if successfully saved
        """
        return await super().save_memories(memories)

    async def get_relevant_memories(self, query_text: str, top_k: int = 5, 
                                   memory_types: List[MemoryType] = None,
                                   min_importance: float = 0.3) -> List[Tuple[Memory, float]]:
        """
        Retrieve relevant memories based on query text.
        
        Args:
            query_text: Text to find relevant memories for
            top_k: Number of top memories to return
            memory_types: Filter by specific memory types
            min_importance: Minimum importance score to include
        
        Returns:
            List of tuples (memory, similarity_score)
        """
        return await self.retrieve_relevant_memories(
            query=query_text,
            top_k=top_k,
            memory_types=memory_types,
            min_importance=min_importance
        )

    async def extract_memories(self, user_input: str, ai_output: str) -> Any:
        """
        Extract memories from user input and AI output.
        
        Args:
            user_input: Input from the user
            ai_output: Output from the AI
        
        Returns:
            Extracted memories
        """
        content = f"User: {user_input}\nAI: {ai_output}"
        return await self.create_memory_from_content(
            content=content,
            memory_type=MemoryType.CONVERSATIONAL,
            context={"interaction_type": "user_ai"}
        )

    # Shutdownable interface implementation
    async def prepare_shutdown(self) -> bool:
        """
        Prepare MemoryService for shutdown.

        Returns:
            bool: True if preparation was successful
        """
        logger.info("Preparing MemoryService for shutdown...")
        try:
            # No specific preparation needed for MemoryService
            logger.info("MemoryService prepared for shutdown")
            return True
        except Exception as e:
            logger.error(f"Error preparing MemoryService for shutdown: {e}")
            return False

    async def shutdown(self, timeout: float = 30.0) -> bool:
        """
        Shutdown MemoryService with timeout.

        Args:
            timeout: Maximum time to wait for shutdown

        Returns:
            bool: True if shutdown was successful
        """
        logger.info(f"Shutting down MemoryService with timeout {timeout}s...")
        try:
            # Perform cleanup with timeout
            # In this implementation, there's no specific cleanup needed
            # since EnhancedMemoryService doesn't have external processes to manage

            logger.info("MemoryService shutdown completed successfully")
            return True
        except asyncio.TimeoutError:
            logger.warning("MemoryService shutdown timed out")
            return False
        except Exception as e:
            logger.error(f"Error during MemoryService shutdown: {e}")
            return False

    def get_shutdown_metrics(self) -> dict:
        """
        Get shutdown-related metrics for the MemoryService.

        Returns:
            Dict containing shutdown metrics
        """
        try:
            stats = self.get_memory_statistics()
            return {
                "status": stats.get("status", "unknown"),
                "total_memories": stats.get("total_memories", 0),
                "memory_types": stats.get("type_distribution", {}),
                "total_content_size": stats.get("total_content_size", 0),
            }
        except Exception as e:
            logger.error(f"Error getting MemoryService shutdown metrics: {e}")
            return {"error": str(e)}
