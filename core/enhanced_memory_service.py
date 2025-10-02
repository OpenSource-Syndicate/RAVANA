import logging
import hashlib
from datetime import datetime, timedelta, timezone
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
    SHORT_TERM = "short_term"  # Temporary memories, recently formed
    WORKING = "working"  # Active memories in current thought processes
    LONG_TERM = "long_term"  # Consolidated memories, stable storage


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
        # Hierarchical memory stores
        self.short_term_memory = {}  # Temporary memories (last few minutes)
        self.working_memory = {}    # Active memories in current thought processes (last few hours)
        self.long_term_memory = {}  # Consolidated memories (persisted long-term)
        self.episodic_memory = {}   # Personal experiences and events
        self.semantic_memory = {}   # Factual knowledge and concepts
        self.procedural_memory = {} # Skills and how-to knowledge
        self.emotional_memory = {}  # Emotionally tagged experiences
        self.memory_tags = {}  # Inverted index for tag-based retrieval
        self.memory_embeddings = {}
        
        # Configuration for memory management
        self.short_term_duration = 300  # 5 minutes in seconds
        self.working_memory_duration = 7200  # 2 hours in seconds
        
        # Memory consolidation settings
        self.consolidation_threshold = 0.6  # Minimum importance for consolidation
        self.consolidation_interval_hours = 2  # How often to trigger consolidation

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
        timestamp = datetime.now(timezone.utc)
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

        # Store in appropriate memory tier based on type and importance
        await self._store_memory_in_hierarchy(memory)

        # Update tag index
        for tag in memory.tags:
            if tag not in self.memory_tags:
                self.memory_tags[tag] = []
            self.memory_tags[tag].append(memory_id)

        logger.info(f"Created memory {memory_id} of type {memory_type.value}")
        return memory

    async def _store_memory_in_hierarchy(self, memory: Memory):
        """Store memory in the appropriate hierarchical level based on type and importance."""
        # Determine memory tier based on type and importance
        if memory.memory_type in [MemoryType.SHORT_TERM] or memory.importance_score < 0.3:
            # Short-term memories: recently formed, low importance, or explicitly short-term
            self.short_term_memory[memory.id] = memory
            self.memory_embeddings[memory.id] = memory.embedding
        elif memory.memory_type in [MemoryType.WORKING] or memory.importance_score > 0.7:
            # Working memories: high importance, or explicitly working memory
            self.working_memory[memory.id] = memory
            self.memory_embeddings[memory.id] = memory.embedding
        elif memory.memory_type == MemoryType.EPISODIC:
            # Episodic memories: personal events and experiences
            self.episodic_memory[memory.id] = memory
            self.memory_embeddings[memory.id] = memory.embedding
        elif memory.memory_type == MemoryType.SEMANTIC:
            # Semantic memories: facts, concepts, and general knowledge
            self.semantic_memory[memory.id] = memory
            self.memory_embeddings[memory.id] = memory.embedding
        elif memory.memory_type == MemoryType.PROCEDURAL:
            # Procedural memories: skills and how-to knowledge
            self.procedural_memory[memory.id] = memory
            self.memory_embeddings[memory.id] = memory.embedding
        elif memory.memory_type == MemoryType.EMOTIONAL:
            # Emotional memories: experiences tagged with emotional context
            self.emotional_memory[memory.id] = memory
            self.memory_embeddings[memory.id] = memory.embedding
        else:
            # Default to long-term memory for other types
            self.long_term_memory[memory.id] = memory
            self.memory_embeddings[memory.id] = memory.embedding

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID from any tier."""
        # Search in all memory tiers
        for memory_store in [self.short_term_memory, self.working_memory, self.long_term_memory, 
                             self.episodic_memory, self.semantic_memory, self.procedural_memory, 
                             self.emotional_memory]:
            if memory_id in memory_store:
                return memory_store[memory_id]
        return None

    def get_memories_by_type(self, memory_type: MemoryType) -> List[Memory]:
        """Retrieve all memories of a specific type."""
        # Special handling for each memory type to use dedicated storage
        if memory_type == MemoryType.EPISODIC:
            return list(self.episodic_memory.values())
        elif memory_type == MemoryType.SEMANTIC:
            return list(self.semantic_memory.values())
        elif memory_type == MemoryType.PROCEDURAL:
            return list(self.procedural_memory.values())
        elif memory_type == MemoryType.EMOTIONAL:
            return list(self.emotional_memory.values())
        elif memory_type == MemoryType.WORKING:
            return list(self.working_memory.values())
        elif memory_type == MemoryType.SHORT_TERM:
            return list(self.short_term_memory.values())
        else:
            # For other types, check all general storage
            all_memories = {**self.short_term_memory, **self.working_memory, **self.long_term_memory}
            return [mem for mem in all_memories.values() if mem.memory_type == memory_type]

    def get_memories_by_tags(self, tags: List[str]) -> List[Memory]:
        """Retrieve memories that match any of the given tags."""
        matching_ids = set()
        for tag in tags:
            if tag in self.memory_tags:
                matching_ids.update(self.memory_tags[tag])

        all_memories = {**self.short_term_memory, **self.working_memory, **self.long_term_memory}
        return [all_memories[mid] for mid in matching_ids if mid in all_memories]

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

        # Get all memories from all tiers - include specialized memory types
        all_memories = {
            **self.short_term_memory, 
            **self.working_memory, 
            **self.long_term_memory,
            **self.episodic_memory,
            **self.semantic_memory,
            **self.procedural_memory,
            **self.emotional_memory
        }

        # Filter memories based on criteria
        candidate_memories = []
        for memory in all_memories.values():
            # Apply filters
            if memory_types and memory.memory_type not in memory_types:
                continue
            if memory.importance_score < min_importance:
                continue
            if time_range_days:
                time_diff = datetime.now(timezone.utc) - memory.timestamp
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

    async def transfer_memory_to_working(self, memory_id: str) -> bool:
        """Transfer a memory from long-term to working memory."""
        if memory_id in self.long_term_memory:
            memory = self.long_term_memory.pop(memory_id)
            self.working_memory[memory_id] = memory
            logger.info(f"Transferred memory {memory_id} from long-term to working memory")
            return True
        elif memory_id in self.short_term_memory:
            memory = self.short_term_memory.pop(memory_id)
            self.working_memory[memory_id] = memory
            logger.info(f"Transferred memory {memory_id} from short-term to working memory")
            return True
        return False

    async def transfer_memory_to_long_term(self, memory_id: str) -> bool:
        """Transfer a memory from working to long-term memory (consolidation)."""
        if memory_id in self.working_memory:
            memory = self.working_memory.pop(memory_id)
            self.long_term_memory[memory_id] = memory
            logger.info(f"Transferred memory {memory_id} from working to long-term memory")
            return True
        elif memory_id in self.short_term_memory:
            memory = self.short_term_memory.pop(memory_id)
            self.long_term_memory[memory_id] = memory
            logger.info(f"Transferred memory {memory_id} from short-term to long-term memory")
            return True
        return False

    async def cleanup_expired_memories(self):
        """Remove expired memories from short-term and working memory."""
        current_time = datetime.now(timezone.utc)
        expired_count = 0

        # Check short-term memory for expired items
        expired_ids = []
        for memory_id, memory in self.short_term_memory.items():
            time_diff = (current_time - memory.timestamp).total_seconds()
            if time_diff > self.short_term_duration:
                expired_ids.append(memory_id)

        for memory_id in expired_ids:
            del self.short_term_memory[memory_id]
            if memory_id in self.memory_embeddings:
                del self.memory_embeddings[memory_id]
            # Remove from tag index
            memory = self.get_memory(memory_id)  # Need to get the memory object to access tags
            if memory:
                for tag in memory.tags:
                    if tag in self.memory_tags and memory_id in self.memory_tags[tag]:
                        self.memory_tags[tag].remove(memory_id)
            expired_count += 1

        # Check working memory for expired items
        expired_ids = []
        for memory_id, memory in self.working_memory.items():
            time_diff = (current_time - memory.timestamp).total_seconds()
            if time_diff > self.working_memory_duration:
                expired_ids.append(memory_id)

        for memory_id in expired_ids:
            # Only remove if importance is low (important memories should be preserved)
            memory = self.working_memory[memory_id]
            if memory.importance_score < 0.5:
                del self.working_memory[memory_id]
                if memory_id in self.memory_embeddings:
                    del self.memory_embeddings[memory_id]
                # Remove from tag index
                for tag in memory.tags:
                    if tag in self.memory_tags and memory_id in self.memory_tags[tag]:
                        self.memory_tags[tag].remove(memory_id)
                expired_count += 1

        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired memories")

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
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days_to_consolidate)
        
        # Get all memories from all tiers
        all_memories = {**self.short_term_memory, **self.working_memory, **self.long_term_memory}
        
        memories_to_consolidate = [
            mem for mem in all_memories.values()
            if (mem.timestamp < cutoff_time and
                mem.importance_score >= min_importance and
                # Only consolidate certain types
                mem.memory_type in [MemoryType.EPISODIC, MemoryType.REFLECTIVE, MemoryType.SHORT_TERM, MemoryType.WORKING])
        ]

        if not memories_to_consolidate:
            return {
                "consolidated_count": 0,
                "message": "No memories eligible for consolidation"
            }

        # Group similar memories using semantic clustering
        # This enhanced version uses embeddings to find semantically similar memories
        consolidated_groups = await self._semantic_clustering(memories_to_consolidate)

        # Consolidate each group
        consolidated_count = 0
        for group_memories in consolidated_groups:
            if len(group_memories) > 1:  # Only consolidate groups with multiple memories
                # Create consolidated content using semantic summarization
                consolidated_content = await self._create_semantic_summary(group_memories)
                
                # Calculate importance based on the importance of individual memories and group size
                avg_importance = sum(m.importance_score for m in group_memories) / len(group_memories)
                max_importance = max(m.importance_score for m in group_memories)
                consolidated_importance = min(0.9, (avg_importance + max_importance) / 2)

                # Create new consolidated memory
                consolidated_memory = await self.create_memory(
                    content=consolidated_content,
                    memory_type=MemoryType.LONG_TERM,  # Change type to long-term
                    # Calculate importance based on the individual memories
                    importance_score=consolidated_importance,
                    # Combine and extend tags
                    tags=self._consolidate_tags(group_memories),
                    embedding_purpose=ModelPurpose.GENERAL
                )

                # Remove original memories from their respective stores
                for old_memory in group_memories:
                    # Remove from short-term memory if it exists there
                    if old_memory.id in self.short_term_memory:
                        del self.short_term_memory[old_memory.id]
                    # Remove from working memory if it exists there
                    elif old_memory.id in self.working_memory:
                        del self.working_memory[old_memory.id]
                    # Remove from long-term memory if it exists there
                    elif old_memory.id in self.long_term_memory:
                        del self.long_term_memory[old_memory.id]
                    
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
            "message": f"Consolidated {consolidated_count} memories into {len(consolidated_groups)} semantically coherent groups"
        }

    async def _semantic_clustering(self, memories: List[Memory], threshold: float = 0.7) -> List[List[Memory]]:
        """
        Group memories based on semantic similarity using embeddings.
        
        Args:
            memories: List of memories to cluster
            threshold: Similarity threshold for clustering
            
        Returns:
            List of memory clusters (each cluster is a list of memories)
        """
        if not memories:
            return []

        # Get embeddings for all memories
        memory_embeddings = []
        for mem in memories:
            # Use the stored embedding if available
            stored_emb = self.memory_embeddings.get(mem.id)
            if stored_emb is not None:
                memory_embeddings.append((mem, stored_emb))
            else:
                # Generate embedding if not stored
                try:
                    emb = self.embeddings_mgr.get_embedding(mem.content, purpose=ModelPurpose.SEMANTIC_SEARCH)
                    emb_list = emb.tolist() if hasattr(emb, 'tolist') else list(emb)
                    memory_embeddings.append((mem, emb_list))
                except Exception as e:
                    logger.warning(f"Failed to get embedding for memory {mem.id}: {e}")
                    continue

        # Perform semantic clustering
        clusters = []
        used_memories = set()

        for i, (mem1, emb1) in enumerate(memory_embeddings):
            if mem1.id in used_memories:
                continue

            cluster = [mem1]
            used_memories.add(mem1.id)

            for j, (mem2, emb2) in enumerate(memory_embeddings[i+1:], i+1):
                if mem2.id in used_memories:
                    continue

                # Calculate cosine similarity
                try:
                    dot_product = sum(a * b for a, b in zip(emb1, emb2))
                    norm1 = sum(a * a for a in emb1) ** 0.5
                    norm2 = sum(a * a for a in emb2) ** 0.5

                    if norm1 == 0 or norm2 == 0:
                        similarity = 0.0
                    else:
                        similarity = dot_product / (norm1 * norm2)

                    if similarity >= threshold:
                        cluster.append(mem2)
                        used_memories.add(mem2.id)
                except Exception as e:
                    logger.warning(f"Error calculating similarity between {mem1.id} and {mem2.id}: {e}")

            if len(cluster) > 0:
                clusters.append(cluster)

        return clusters

    async def _create_semantic_summary(self, group_memories: List[Memory]) -> str:
        """
        Create a semantic summary of a group of related memories.
        
        Args:
            group_memories: List of semantically related memories
            
        Returns:
            Consolidated content summary
        """
        if not group_memories:
            return ""

        # Sort by importance to prioritize important memories in summary
        sorted_memories = sorted(group_memories, key=lambda m: m.importance_score, reverse=True)

        # Extract key themes and concepts from the memories
        theme_prompt = f"""
        Below are several related memories that should be consolidated into a single semantic memory.
        
        Memories:
        {chr(10).join([f"- {mem.summary}" for mem in sorted_memories[:5]])}
        
        Please create a comprehensive summary that captures the key themes, insights, and
        important information from these related memories. The summary should be concise
        but preserve the essential meaning and context.
        
        Consolidated Summary:
        """

        # Use LLM to generate a semantic summary
        try:
            from core.llm import call_llm
            summary = await asyncio.to_thread(call_llm, theme_prompt)
            if summary and summary.strip():
                return summary.strip()
        except Exception as e:
            logger.warning(f"Failed to generate semantic summary using LLM: {e}")

        # Fallback: simple concatenation with key points
        title = f"Consolidated memory group: {sorted_memories[0].summary[:50]}..."
        content_parts = [title]
        
        # Add important content from each memory
        for mem in sorted_memories[:5]:  # Limit to top 5 memories
            content_parts.append(f"Key point: {mem.summary}")
        
        return "\n".join(content_parts)

    def _consolidate_tags(self, group_memories: List[Memory]) -> List[str]:
        """
        Consolidate tags from a group of memories, preserving the most relevant ones.
        
        Args:
            group_memories: List of memories to consolidate tags from
            
        Returns:
            Consolidated list of tags
        """
        # Count tag occurrences and importance-weighted scores
        tag_scores = {}
        for mem in group_memories:
            for tag in mem.tags:
                # Weight by memory importance
                current_score = tag_scores.get(tag, 0)
                tag_scores[tag] = current_score + mem.importance_score

        # Sort by score and return top tags (with a minimum threshold)
        sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, score in sorted_tags if score >= 0.5]  # Minimum threshold

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        # Combined memory counts
        all_memories = {
            **self.short_term_memory, 
            **self.working_memory, 
            **self.long_term_memory,
            **self.episodic_memory,
            **self.semantic_memory,
            **self.procedural_memory,
            **self.emotional_memory
        }
        total_memories = len(all_memories)
        type_counts = {}
        total_size = 0

        for memory in all_memories.values():
            type_name = memory.memory_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            total_size += len(memory.content)

        # Tier-specific counts
        tier_counts = {
            "short_term": len(self.short_term_memory),
            "working": len(self.working_memory),
            "long_term": len(self.long_term_memory),
            "episodic": len(self.episodic_memory),
            "semantic": len(self.semantic_memory),
            "procedural": len(self.procedural_memory),
            "emotional": len(self.emotional_memory)
        }

        return {
            "total_memories": total_memories,
            "tier_distribution": tier_counts,
            "type_distribution": type_counts,
            "total_content_size": total_size,
            "average_content_length": total_size / max(1, total_memories) if total_memories > 0 else 0,
            "tag_count": len(self.memory_tags)
        }


class EnhancedMemoryService:
    """Main memory service that integrates embedding and LLM capabilities."""

    def __init__(self):
        self.storage_service = MemoryStorageService()
        self.summary_service = self.storage_service.summary_service
        self.embeddings_mgr = self.storage_service.embeddings_mgr

    async def cleanup_expired_memories(self):
        """Clean up expired memories from short-term and working memory."""
        await self.storage_service.cleanup_expired_memories()

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
                            (datetime.now(timezone.utc) - memory.timestamp).days > time_range_days):
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
                # If memory_data is already a Memory object, save it by storing in appropriate hierarchy
                if isinstance(memory_data, Memory):
                    await self.storage_service._store_memory_in_hierarchy(memory_data)
                    
                    # Update embedding cache
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
                    # For now, we'll treat it as content and create a memory from the specified type
                    if isinstance(memory_data, dict):
                        content = memory_data.get('content', str(memory_data))
                        # Check if a memory type is specified in the dict
                        memory_type = memory_data.get('type', MemoryType.REFLECTIVE)
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

    async def link_memories_by_context(self, memory_id: str, related_memory_ids: List[str], 
                                      strength: float = 0.5) -> bool:
        """
        Create contextual links between memories based on shared context.
        
        Args:
            memory_id: ID of the primary memory
            related_memory_ids: List of IDs of related memories
            strength: Strength of the contextual link (0.0-1.0)
            
        Returns:
            True if linking was successful
        """
        try:
            # For now, we'll enhance the memory's context field to note the relationships
            primary_memory = self.storage_service.get_memory(memory_id)
            if not primary_memory:
                return False
                
            # Update the context of the primary memory to include links
            if 'related_memories' not in primary_memory.context:
                primary_memory.context['related_memories'] = {}
            
            for related_id in related_memory_ids:
                primary_memory.context['related_memories'][related_id] = {
                    'relationship_type': 'contextual',
                    'strength': strength,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            # Also update the related memory to reference back to the primary
            for related_id in related_memory_ids:
                related_memory = self.storage_service.get_memory(related_id)
                if related_memory:
                    if 'related_memories' not in related_memory.context:
                        related_memory.context['related_memories'] = {}
                    related_memory.context['related_memories'][memory_id] = {
                        'relationship_type': 'contextual',
                        'strength': strength,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
            logger.info(f"Created contextual links between memory {memory_id} and {len(related_memory_ids)} other memories")
            return True
        except Exception as e:
            logger.error(f"Error linking memories by context: {e}")
            return False

    async def link_memories_by_emotion(self, memory_id: str, related_memory_ids: List[str], 
                                      emotion_type: str, strength: float = 0.5) -> bool:
        """
        Create emotional links between memories based on shared emotional content.
        
        Args:
            memory_id: ID of the primary memory
            related_memory_ids: List of IDs of related memories
            emotion_type: Type of emotion that links these memories
            strength: Strength of the emotional link (0.0-1.0)
            
        Returns:
            True if linking was successful
        """
        try:
            # Update the context of the primary memory to include emotional links
            primary_memory = self.storage_service.get_memory(memory_id)
            if not primary_memory:
                return False
                
            if 'emotional_links' not in primary_memory.context:
                primary_memory.context['emotional_links'] = {}
            
            for related_id in related_memory_ids:
                primary_memory.context['emotional_links'][related_id] = {
                    'emotion_type': emotion_type,
                    'strength': strength,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            # Also update the related memory to reference back to the primary
            for related_id in related_memory_ids:
                related_memory = self.storage_service.get_memory(related_id)
                if related_memory:
                    if 'emotional_links' not in related_memory.context:
                        related_memory.context['emotional_links'] = {}
                    related_memory.context['emotional_links'][memory_id] = {
                        'emotion_type': emotion_type,
                        'strength': strength,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
            logger.info(f"Created emotional links based on '{emotion_type}' between memory {memory_id} and {len(related_memory_ids)} other memories")
            return True
        except Exception as e:
            logger.error(f"Error linking memories by emotion: {e}")
            return False

    async def find_temporally_linked_memories(self, memory_id: str, time_window_minutes: int = 60) -> List[Memory]:
        """
        Find memories that occurred within a specific time window of the given memory.
        
        Args:
            memory_id: ID of the reference memory
            time_window_minutes: Time window in minutes to search for related memories
            
        Returns:
            List of temporally related memories
        """
        try:
            reference_memory = self.storage_service.get_memory(memory_id)
            if not reference_memory:
                return []
                
            reference_time = reference_memory.timestamp
            time_start = reference_time - timedelta(minutes=time_window_minutes/2)
            time_end = reference_time + timedelta(minutes=time_window_minutes/2)
            
            # Get all memories from all tiers
            all_memories = {**self.storage_service.short_term_memory, 
                           **self.storage_service.working_memory, 
                           **self.storage_service.long_term_memory}
            
            temporally_related = []
            for memory in all_memories.values():
                if time_start <= memory.timestamp <= time_end and memory.id != memory_id:
                    temporally_related.append(memory)
                    
            logger.info(f"Found {len(temporally_related)} temporally related memories for memory {memory_id}")
            return temporally_related
        except Exception as e:
            logger.error(f"Error finding temporally linked memories: {e}")
            return []

    async def retrieve_contextually_relevant_memories(self, 
                                                     query: str, 
                                                     emotion_context: Dict[str, float] = None,
                                                     time_range_minutes: int = 1440,  # 24 hours default
                                                     top_k: int = 10) -> List[Tuple[Memory, float]]:
        """
        Retrieve memories using multiple relevance criteria: semantic, emotional, and temporal.
        
        Args:
            query: Query text for semantic relevance
            emotion_context: Current emotional state to match against emotional memories
            time_range_minutes: Time range to consider for temporal relevance
            top_k: Maximum number of memories to return
            
        Returns:
            List of tuples (memory, relevance_score) sorted by combined relevance
        """
        try:
            # Get semantically similar memories
            semantic_memories = await self.retrieve_relevant_memories(
                query=query,
                top_k=top_k*2,  # Get more than needed for combination
                time_range_days=time_range_minutes/1440  # Convert minutes to days
            )
            
            # If emotion context is provided, factor emotional relevance
            if emotion_context:
                # Enhance semantic results with emotional relevance
                enhanced_results = []
                for memory, semantic_score in semantic_memories:
                    # Calculate emotional relevance if memory has emotional context
                    emotional_relevance = 0.0
                    if hasattr(memory, 'context') and 'emotions' in memory.context:
                        memory_emotions = memory.context.get('emotions', {})
                        # Calculate similarity between current emotion context and memory's emotional content
                        for emotion, current_intensity in emotion_context.items():
                            memory_intensity = memory_emotions.get(emotion, 0.0)
                            # Weight by both intensities
                            emotional_relevance += current_intensity * memory_intensity
                        # Normalize emotional relevance (assuming emotional_relevance can be up to number of emotions * 1.0)
                        emotional_relevance = min(1.0, emotional_relevance / len(emotion_context))
                    
                    # Combine semantic and emotional scores
                    combined_score = 0.7 * semantic_score + 0.3 * emotional_relevance
                    enhanced_results.append((memory, combined_score))
                
                # Sort by combined score and return top_k
                enhanced_results.sort(key=lambda x: x[1], reverse=True)
                return enhanced_results[:top_k]
            else:
                # Return just semantic results if no emotion context provided
                return semantic_memories[:top_k]
                
        except Exception as e:
            logger.error(f"Error retrieving contextually relevant memories: {e}")
            # Fallback to just semantic search
            return await self.retrieve_relevant_memories(
                query=query,
                top_k=top_k,
                time_range_days=time_range_minutes/1440
            )

    async def update_working_memory(self, 
                                  new_content: str = None, 
                                  memory_id: str = None,
                                  content_override: str = None,
                                  tags: List[str] = None) -> Optional[Memory]:
        """
        Update the working memory with new information or modify existing working memory.
        
        Args:
            new_content: New content to add to working memory
            memory_id: ID of existing memory to update
            content_override: New content for existing memory
            tags: Tags to associate with the working memory
            
        Returns:
            Updated or created Memory object
        """
        try:
            # If updating existing memory
            if memory_id and content_override:
                existing_memory = self.working_memory.get(memory_id)
                if existing_memory:
                    # Update content and re-embed
                    updated_memory = Memory(
                        id=existing_memory.id,
                        content=content_override,
                        summary=await self.summary_service.generate_summary(content_override),
                        embedding=self.embeddings_mgr.get_embedding(content_override, purpose=ModelPurpose.GENERAL).tolist() if content_override else existing_memory.embedding,
                        memory_type=MemoryType.WORKING,
                        timestamp=datetime.now(timezone.utc),
                        tags=tags or existing_memory.tags,
                        importance_score=existing_memory.importance_score,
                        context=existing_memory.context,
                        embedding_purpose=ModelPurpose.GENERAL
                    )
                    self.working_memory[memory_id] = updated_memory
                    self.memory_embeddings[memory_id] = updated_memory.embedding
                    logger.info(f"Updated working memory {memory_id}")
                    return updated_memory
            # If adding new content
            elif new_content:
                new_memory = await self._create_working_memory(new_content, tags or [])
                logger.info(f"Added new content to working memory: {new_memory.id}")
                return new_memory
            # If just retrieving
            else:
                return self.working_memory.get(memory_id) if memory_id else None
        except Exception as e:
            logger.error(f"Error updating working memory: {e}")
            return None

    async def _create_working_memory(self, content: str, tags: List[str] = None) -> Memory:
        """Create a new working memory item."""
        memory_id = self._generate_memory_id(content, datetime.now(timezone.utc))
        
        # Generate embedding and summary
        embedding = self.embeddings_mgr.get_embedding(content, purpose=ModelPurpose.GENERAL)
        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        summary = await self.summary_service.generate_summary(content)
        
        # Create memory with appropriate importance
        memory = Memory(
            id=memory_id,
            content=content,
            summary=summary,
            embedding=embedding_list,
            memory_type=MemoryType.WORKING,
            timestamp=datetime.now(timezone.utc),
            tags=tags or [],
            importance_score=0.8,  # Higher importance for working memory
            context={"created_in_working_memory": True},
            embedding_purpose=ModelPurpose.GENERAL
        )
        
        # Store in working memory
        self.working_memory[memory_id] = memory
        self.memory_embeddings[memory_id] = memory.embedding
        
        # Update tag index
        for tag in memory.tags:
            if tag not in self.memory_tags:
                self.memory_tags[tag] = []
            self.memory_tags[tag].append(memory_id)
        
        return memory

    def get_working_memory_contents(self) -> List[Memory]:
        """Get all currently active working memories."""
        return list(self.working_memory.values())

    def clear_working_memory(self, preserve_important: bool = True):
        """
        Clear working memory, optionally preserving high-importance items.
        
        Args:
            preserve_important: If True, keep memories with importance >= 0.7
        """
        if preserve_important:
            # Only remove low-importance working memories
            to_remove = []
            for memory_id, memory in self.working_memory.items():
                if memory.importance_score < 0.7:
                    to_remove.append(memory_id)
            
            for memory_id in to_remove:
                del self.working_memory[memory_id]
                if memory_id in self.memory_embeddings:
                    del self.memory_embeddings[memory_id]
                # Update tag index
                memory = self.get_memory(memory_id)
                if memory:
                    for tag in memory.tags:
                        if tag in self.memory_tags and memory_id in self.memory_tags[tag]:
                            self.memory_tags[tag].remove(memory_id)
        else:
            # Clear all working memory
            for memory_id in list(self.working_memory.keys()):
                if memory_id in self.memory_embeddings:
                    del self.memory_embeddings[memory_id]
                # Update tag index
                memory = self.get_memory(memory_id)
                if memory:
                    for tag in memory.tags:
                        if tag in self.memory_tags and memory_id in self.memory_tags[tag]:
                            self.memory_tags[tag].remove(memory_id)
            
            self.working_memory.clear()
            
        logger.info(f"Cleared working memory (preserved important: {preserve_important})")


# Global instance for shared use
enhanced_memory_service = EnhancedMemoryService()
