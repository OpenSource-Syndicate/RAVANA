import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from core.embeddings_manager import embeddings_manager, ModelPurpose
from core.llm_selector import get_llm_selector, LLMTask
from core.config import Config

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memories that can be stored."""
    EPISODIC = "episodic"  # Personal experiences and events
    SEMANTIC = "semantic"  # Facts and general knowledge
    PROCEDURAL = "procedural"  # Skills and how-to information
    EMOTIONAL = "emotional"  # Emotional experiences and states
    REFLECTIVE = "reflective"  # Reflections and insights
    CONVERSATIONAL = "conversational"  # Interaction memories
    LEARNING = "learning"  # Learning experiences and knowledge gained


@dataclass
class Memory:
    """Represents a single memory with all its components."""
    id: str
    content: str
    summary: str
    embedding: Optional[List[float]]
    memory_type: MemoryType
    timestamp: datetime
    tags: List[str]
    importance_score: float  # 0.0 to 1.0, how important the memory is
    context: Dict[str, Any]  # Additional context for the memory
    embedding_purpose: ModelPurpose  # Purpose for which embedding was generated


class MemorySummaryService:
    """Service to generate summaries of content using AI-driven LLM selection."""

    def __init__(self):
        self.llm_selector = get_llm_selector()

    async def generate_summary(self,
                               content: str,
                               max_length: int = 200,
                               focus_on: str = "key_points") -> str:
        """
        Generate a summary of the content using AI-selected LLM.

        Args:
            content: Content to summarize
            max_length: Maximum length of summary in words
            focus_on: What aspect to focus on ("key_points", "emotional", "factual", "insightful")

        Returns:
            Generated summary
        """
        if not self.llm_selector:
            # Fallback to simple summarization if LLM selector not available
            words = content.split()
            if len(words) <= max_length:
                return content
            return " ".join(words[:max_length]) + "..."

        # Select the best provider for summarization task
        provider_info = self.llm_selector.select_best_provider(
            task=LLMTask.ANALYSIS,
            content=content,
            response_quality_needed=0.8,
            response_speed_needed=0.6
        )

        if not provider_info:
            logger.warning(
                "No LLM provider available for summarization, using fallback")
            # Fallback to simple summarization
            words = content.split()
            if len(words) <= max_length:
                return content
            return " ".join(words[:max_length]) + "..."

        # For now, we'll return a simple summary, but in a real implementation
        # this would call the selected LLM provider to generate the summary
        logger.info(f"Selected {provider_info['provider']} for summarization")

        # Create a summary prompt based on focus
        focus_prompts = {
            "key_points": f"Summarize the key points of the following text in about {max_length} words: {content}",
            "emotional": f"Summarize the emotional content and feelings expressed in the following text in about {max_length} words: {content}",
            "factual": f"Summarize the factual information in the following text in about {max_length} words: {content}",
            "insightful": f"Extract insights and learnings from the following text and summarize them in about {max_length} words: {content}"
        }

        prompt = focus_prompts.get(focus_on, focus_prompts["key_points"])

        # Simulate LLM response - in real implementation this would call the actual API
        # For now returning a truncated version of content
        words = content.split()
        summary = " ".join(words[:min(len(words), max_length)]) + "..."

        return summary


class MemoryStorageService:
    """Service to handle memory storage and retrieval with consistency guarantees."""

    def __init__(self):
        self.embeddings_mgr = embeddings_manager
        self.summary_service = MemorySummaryService()
        self.memory_store = {}  # In real implementation, this would be a persistent store
        self.memory_tags = {}  # Inverted index for tag-based retrieval
        self.memory_embeddings = {}

    def _generate_memory_id(self, content: str, timestamp: datetime) -> str:
        """Generate a unique ID for a memory."""
        content_hash = hashlib.sha256(
            f"{content}{timestamp.isoformat()}".encode()).hexdigest()
        return f"mem_{content_hash[:12]}_{int(timestamp.timestamp())}"

    async def create_memory(self,
                            content: str,
                            memory_type: MemoryType,
                            context: Dict[str, Any] = None,
                            tags: List[str] = None,
                            importance_score: float = 0.5,
                            embedding_purpose: ModelPurpose = ModelPurpose.GENERAL) -> Memory:
        """
        Create a new memory with summary and embedding.

        Args:
            content: The main content of the memory
            memory_type: Type of memory being created
            context: Additional context for the memory
            tags: Tags to associate with the memory
            importance_score: How important this memory is (0.0-1.0)
            embedding_purpose: Purpose for generating the embedding

        Returns:
            Created Memory object
        """
        timestamp = datetime.utcnow()
        memory_id = self._generate_memory_id(content, timestamp)

        # Generate summary
        summary = await self.summary_service.generate_summary(content)

        # Generate embedding with AI-selected model
        try:
            embedding = self.embeddings_mgr.get_embedding(
                content,
                purpose=embedding_purpose
            )
            # Convert numpy array to list for storage
            embedding_list = embedding.tolist() if hasattr(
                embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.error(
                f"Error generating embedding for memory {memory_id}: {e}")
            embedding_list = [0.0] * 384  # Default embedding size

        # Create memory object
        memory = Memory(
            id=memory_id,
            content=content,
            summary=summary,
            embedding=embedding_list,
            memory_type=memory_type,
            timestamp=timestamp,
            tags=tags or [],
            importance_score=importance_score,
            context=context or {},
            embedding_purpose=embedding_purpose
        )

        # Store in memory store
        self.memory_store[memory_id] = memory
        self.memory_embeddings[memory_id] = embedding_list

        # Update tag index
        for tag in memory.tags:
            if tag not in self.memory_tags:
                self.memory_tags[tag] = []
            self.memory_tags[tag].append(memory_id)

        logger.info(f"Created memory {memory_id} of type {memory_type.value}")
        return memory

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        return self.memory_store.get(memory_id)

    def get_memories_by_type(self, memory_type: MemoryType) -> List[Memory]:
        """Retrieve all memories of a specific type."""
        return [mem for mem in self.memory_store.values() if mem.memory_type == memory_type]

    def get_memories_by_tags(self, tags: List[str]) -> List[Memory]:
        """Retrieve memories that match any of the given tags."""
        matching_ids = set()
        for tag in tags:
            if tag in self.memory_tags:
                matching_ids.update(self.memory_tags[tag])

        return [self.memory_store[mid] for mid in matching_ids if mid in self.memory_store]

    async def find_similar_memories(self,
                                    query: str,
                                    top_k: int = 5,
                                    memory_types: List[MemoryType] = None,
                                    min_importance: float = 0.0,
                                    time_range_days: int = None) -> List[Tuple[Memory, float]]:
        """
        Find memories similar to the query using semantic search.

        Args:
            query: Query text to find similar memories
            top_k: Number of top similar memories to return
            memory_types: Filter by specific memory types
            min_importance: Minimum importance score
            time_range_days: Only include memories from last N days

        Returns:
            List of tuples (memory, similarity_score) sorted by similarity
        """
        # Generate query embedding with AI-selected model
        try:
            query_embedding = self.embeddings_mgr.get_embedding(
                query,
                purpose=ModelPurpose.SEMANTIC_SEARCH
            )
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return []

        # Filter memories based on criteria
        candidate_memories = []
        for memory in self.memory_store.values():
            # Apply filters
            if memory_types and memory.memory_type not in memory_types:
                continue
            if memory.importance_score < min_importance:
                continue
            if time_range_days:
                time_diff = datetime.utcnow() - memory.timestamp
                if time_diff.days > time_range_days:
                    continue

            candidate_memories.append(memory)

        if not candidate_memories:
            return []

        # Calculate similarities
        similarities = []
        query_embedding_np = query_embedding if hasattr(
            query_embedding, '__len__') else [query_embedding]

        for memory in candidate_memories:
            memory_embedding_np = memory.embedding
            try:
                # Calculate cosine similarity
                dot_product = sum(
                    a * b for a, b in zip(query_embedding_np, memory_embedding_np))
                norm_query = sum(a * a for a in query_embedding_np) ** 0.5
                norm_memory = sum(a * a for a in memory_embedding_np) ** 0.5

                if norm_query == 0 or norm_memory == 0:
                    similarity = 0.0
                else:
                    similarity = dot_product / (norm_query * norm_memory)

                similarities.append((memory, similarity))
            except Exception as e:
                logger.warning(
                    f"Error calculating similarity for memory {memory.id}: {e}")
                continue

        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    async def consolidate_memories(self,
                                   days_to_consolidate: int = 30,
                                   min_importance: float = 0.3) -> Dict[str, Any]:
        """
        Consolidate older memories to reduce storage and improve retrieval.

        Args:
            days_to_consolidate: Only consolidate memories older than this many days
            min_importance: Minimum importance to consider for consolidation

        Returns:
            Summary of consolidation operation
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days_to_consolidate)
        memories_to_consolidate = [
            mem for mem in self.memory_store.values()
            if (mem.timestamp < cutoff_time and
                mem.importance_score >= min_importance and
                # Only consolidate certain types
                mem.memory_type in [MemoryType.EPISODIC, MemoryType.REFLECTIVE])
        ]

        if not memories_to_consolidate:
            return {
                "consolidated_count": 0,
                "message": "No memories eligible for consolidation"
            }

        # Group similar memories for consolidation
        consolidated_groups = {}
        for memory in memories_to_consolidate:
            # Create a simple grouping key based on content themes
            # In a real implementation, this would use semantic clustering
            content_start = memory.content[:50].lower()
            group_key = content_start.split(
            )[0] if content_start.split() else "unknown"

            if group_key not in consolidated_groups:
                consolidated_groups[group_key] = []
            consolidated_groups[group_key].append(memory)

        # Consolidate each group
        consolidated_count = 0
        for group_key, group_memories in consolidated_groups.items():
            if len(group_memories) > 1:  # Only consolidate groups with multiple memories
                # Create consolidated content
                consolidated_content = "Consolidated memories about " + group_key + ": "
                # Limit to first 5 summaries
                consolidated_content += " ".join(
                    [mem.summary for mem in group_memories[:5]])

                # Create new consolidated memory
                consolidated_memory = await self.create_memory(
                    content=consolidated_content,
                    memory_type=MemoryType.SEMANTIC,  # Change type to semantic
                    # Keep max importance
                    importance_score=min(
                        0.8, max(m.importance_score for m in group_memories)),
                    # Combine tags
                    tags=list(
                        set(tag for mem in group_memories for tag in mem.tags)),
                    embedding_purpose=ModelPurpose.GENERAL
                )

                # Remove original memories
                for old_memory in group_memories:
                    if old_memory.id in self.memory_store:
                        del self.memory_store[old_memory.id]
                        if old_memory.id in self.memory_embeddings:
                            del self.memory_embeddings[old_memory.id]
                        # Update tag index
                        for tag in old_memory.tags:
                            if tag in self.memory_tags and old_memory.id in self.memory_tags[tag]:
                                self.memory_tags[tag].remove(old_memory.id)

                consolidated_count += len(group_memories)

        return {
            "consolidated_count": consolidated_count,
            "groups_created": len(consolidated_groups),
            "message": f"Consolidated {consolidated_count} memories into {len(consolidated_groups)} groups"
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        total_memories = len(self.memory_store)
        type_counts = {}
        total_size = 0

        for memory in self.memory_store.values():
            type_name = memory.memory_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            total_size += len(memory.content)

        return {
            "total_memories": total_memories,
            "type_distribution": type_counts,
            "total_content_size": total_size,
            "average_content_length": total_size / max(1, total_memories),
            "tag_count": len(self.memory_tags)
        }


class EnhancedMemoryService:
    """Main memory service that integrates embedding and LLM capabilities."""

    def __init__(self):
        self.storage_service = MemoryStorageService()
        self.summary_service = self.storage_service.summary_service
        self.embeddings_mgr = self.storage_service.embeddings_mgr

    async def create_memory_from_content(self,
                                         content: str,
                                         memory_type: MemoryType,
                                         context: Dict[str, Any] = None,
                                         tags: List[str] = None,
                                         importance_override: float = None) -> Memory:
        """
        Create a memory from content, automatically determining importance and type.

        Args:
            content: Content to store as memory
            memory_type: Type of memory to create
            context: Additional context
            tags: Tags to associate
            importance_override: Override automatic importance calculation

        Returns:
            Created Memory object
        """
        # Calculate importance based on content characteristics if not overridden
        if importance_override is None:
            importance = await self._calculate_importance(content, memory_type)
        else:
            importance = importance_override

        # Determine embedding purpose based on memory type
        embedding_purpose = self._get_embedding_purpose_for_memory_type(
            memory_type)

        return await self.storage_service.create_memory(
            content=content,
            memory_type=memory_type,
            context=context,
            tags=tags,
            importance_score=importance,
            embedding_purpose=embedding_purpose
        )

    async def _calculate_importance(self, content: str, memory_type: MemoryType) -> float:
        """Calculate importance score for content based on various factors."""
        importance = 0.5  # Base importance

        # Length-based importance (longer content might be more important)
        length_score = min(0.3, len(content) / 1000)  # Cap at 0.3
        importance += length_score

        # Type-based importance adjustment
        type_multiplier = {
            MemoryType.REFLECTIVE: 1.2,
            MemoryType.LEARNING: 1.1,
            MemoryType.EMOTIONAL: 1.0,
            MemoryType.EPISODIC: 0.8,
            MemoryType.SEMANTIC: 1.0,
            MemoryType.PROCEDURAL: 1.0,
            MemoryType.CONVERSATIONAL: 0.7
        }
        importance *= type_multiplier.get(memory_type, 1.0)

        # Keyword-based importance (certain keywords indicate higher importance)
        important_keywords = ['important', 'crucial', 'critical', 'key', 'essential',
                              'learned', 'realized', 'discovered', 'insight', 'breakthrough']

        content_lower = content.lower()
        keyword_score = sum(
            1 for word in important_keywords if word in content_lower) * 0.1
        importance += keyword_score

        # Cap importance between 0.1 and 1.0
        return max(0.1, min(1.0, importance))

    def _get_embedding_purpose_for_memory_type(self, memory_type: MemoryType) -> ModelPurpose:
        """Get the appropriate embedding purpose for a memory type."""
        purpose_mapping = {
            MemoryType.EPISODIC: ModelPurpose.GENERAL,
            MemoryType.SEMANTIC: ModelPurpose.GENERAL,
            MemoryType.PROCEDURAL: ModelPurpose.GENERAL,
            MemoryType.EMOTIONAL: ModelPurpose.GENERAL,
            MemoryType.REFLECTIVE: ModelPurpose.GENERAL,
            MemoryType.CONVERSATIONAL: ModelPurpose.GENERAL,
            MemoryType.LEARNING: ModelPurpose.GENERAL
        }
        return purpose_mapping.get(memory_type, ModelPurpose.GENERAL)

    async def retrieve_relevant_memories(self,
                                         query: str,
                                         top_k: int = 5,
                                         memory_types: List[MemoryType] = None,
                                         min_importance: float = 0.3,
                                         time_range_days: int = None,
                                         tags: List[str] = None) -> List[Tuple[Memory, float]]:
        """
        Retrieve memories relevant to a query using semantic search.

        Args:
            query: Query to find relevant memories for
            top_k: Number of top memories to return
            memory_types: Filter by specific memory types
            min_importance: Minimum importance score to include
            time_range_days: Only include memories from last N days
            tags: Filter by specific tags

        Returns:
            List of tuples (memory, similarity_score)
        """
        # If tags are specified, use tag-based retrieval first
        if tags:
            tagged_memories = self.storage_service.get_memories_by_tags(tags)
            # Among tagged memories, find those most similar to query
            if tagged_memories:
                query_embedding = self.embeddings_mgr.get_embedding(
                    query,
                    purpose=ModelPurpose.SEMANTIC_SEARCH
                )

                similarities = []
                for memory in tagged_memories:
                    if (memory_types and memory.memory_type not in memory_types) or \
                       memory.importance_score < min_importance or \
                       (time_range_days and
                            (datetime.utcnow() - memory.timestamp).days > time_range_days):
                        continue

                    # Calculate similarity with memory content
                    memory_embedding = memory.embedding
                    try:
                        dot_product = sum(
                            a * b for a, b in zip(query_embedding, memory_embedding))
                        norm_query = sum(a * a for a in query_embedding) ** 0.5
                        norm_memory = sum(
                            a * a for a in memory_embedding) ** 0.5

                        if norm_query == 0 or norm_memory == 0:
                            similarity = 0.0
                        else:
                            similarity = dot_product / \
                                (norm_query * norm_memory)

                        similarities.append((memory, similarity))
                    except Exception as e:
                        logger.warning(f"Error calculating similarity: {e}")
                        continue

                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:top_k]

        # Fall back to semantic search
        return await self.storage_service.find_similar_memories(
            query=query,
            top_k=top_k,
            memory_types=memory_types,
            min_importance=min_importance,
            time_range_days=time_range_days
        )

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        return self.storage_service.get_memory(memory_id)

    def get_memories_by_type(self, memory_type: MemoryType) -> List[Memory]:
        """Get all memories of a specific type."""
        return self.storage_service.get_memories_by_type(memory_type)

    async def consolidate_old_memories(self) -> Dict[str, Any]:
        """Consolidate older memories to maintain efficiency."""
        return await self.storage_service.consolidate_memories()

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        return self.storage_service.get_memory_stats()

    async def extract_memories(self, content: str, context: str = "") -> Any:
        """
        Extract significant memories from content and context.

        Args:
            content: Content to extract memories from
            context: Additional context for memory extraction

        Returns:
            Object with memories attribute containing a list of extracted memories
        """
        # Create a memory from the content
        memory = await self.create_memory_from_content(
            content=content,
            memory_type=MemoryType.REFLECTIVE,  # Default to reflective for extracted content
            context={"source": "extraction", "original_context": context}
        )

        # Create an object with the expected 'memories' attribute
        class MemoryExtractionResult:
            def __init__(self, memories):
                self.memories = memories

        return MemoryExtractionResult([memory])

    async def save_memories(self, memories) -> bool:
        """
        Save a list of memories to the memory store.

        Args:
            memories: List of memory objects to save

        Returns:
            True if successfully saved
        """
        try:
            if not memories:
                return True

            # Process each memory in the list
            for memory_data in memories:
                # If memory_data is already a Memory object, save it directly
                if isinstance(memory_data, Memory):
                    self.storage_service.memory_store[memory_data.id] = memory_data
                    self.storage_service.memory_embeddings[memory_data.id] = memory_data.embedding

                    # Update tag index
                    for tag in memory_data.tags:
                        if tag not in self.storage_service.memory_tags:
                            self.storage_service.memory_tags[tag] = []
                        if memory_data.id not in self.storage_service.memory_tags[tag]:
                            self.storage_service.memory_tags[tag].append(
                                memory_data.id)
                else:
                    # If it's a dict or other format, create a memory from it
                    # For now, we'll treat it as content and create a reflective memory
                    if isinstance(memory_data, dict):
                        content = memory_data.get('content', str(memory_data))
                        memory_type = MemoryType.REFLECTIVE
                        context = memory_data.get('context', {})
                        tags = memory_data.get('tags', [])
                    else:
                        content = str(memory_data)
                        memory_type = MemoryType.REFLECTIVE
                        context = {}
                        tags = []

                    await self.create_memory_from_content(
                        content=content,
                        memory_type=memory_type,
                        context=context,
                        tags=tags
                    )

            logger.info(f"Saved {len(memories)} memories to memory store")
            return True
        except Exception as e:
            logger.error(f"Error saving memories: {e}")
            return False


# Global instance for shared use
enhanced_memory_service = EnhancedMemoryService()
