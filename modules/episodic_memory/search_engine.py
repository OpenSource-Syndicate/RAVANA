"""
AdvancedSearchEngine for multi-modal memory system.
Implements hybrid search combining vector and text similarity, plus cross-modal search.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, List
import numpy as np

from .models import (
    MemoryRecord, ContentType, MemoryType, SearchRequest, SearchResponse,
    SearchResult, SearchMode, CrossModalSearchRequest
)
from .postgresql_store import PostgreSQLStore
from .embedding_service import EmbeddingService
from .whisper_processor import WhisperAudioProcessor

logger = logging.getLogger(__name__)


class AdvancedSearchEngine:
    """
    Advanced search engine with hybrid and cross-modal search capabilities.
    Combines vector similarity search with full-text search and supports cross-modal retrieval.
    """

    def __init__(self,
                 postgres_store: PostgreSQLStore,
                 embedding_service: EmbeddingService,
                 whisper_processor: Optional[WhisperAudioProcessor] = None):
        """
        Initialize the advanced search engine.

        Args:
            postgres_store: PostgreSQL store instance
            embedding_service: Embedding service instance
            whisper_processor: Optional Whisper processor for audio queries
        """
        self.postgres = postgres_store
        self.embeddings = embedding_service
        self.whisper = whisper_processor

        # Search configuration
        self.vector_weight = 0.7
        self.text_weight = 0.3
        self.rerank_threshold = 0.5

        logger.info("Initialized AdvancedSearchEngine")

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Perform search based on the search mode specified in the request.

        Args:
            request: Search request with parameters

        Returns:
            Search response with results
        """
        start_time = time.time()

        try:
            if request.search_mode == SearchMode.VECTOR:
                results = await self._vector_search(request)
            elif request.search_mode == SearchMode.TEXT:
                results = await self._text_search(request)
            elif request.search_mode == SearchMode.HYBRID:
                results = await self._hybrid_search(request)
            elif request.search_mode == SearchMode.CROSS_MODAL:
                results = await self._cross_modal_search(request)
            else:
                raise ValueError(
                    f"Unsupported search mode: {request.search_mode}")

            # Apply final filtering and ranking
            filtered_results = await self._filter_and_rank_results(results, request)

            search_time_ms = int((time.time() - start_time) * 1000)

            return SearchResponse(
                results=filtered_results,
                total_found=len(filtered_results),
                search_time_ms=search_time_ms,
                search_mode=request.search_mode,
                query_metadata={
                    "query": request.query,
                    "content_types": [ct.value for ct in request.content_types] if request.content_types else None,
                    "similarity_threshold": request.similarity_threshold
                }
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return SearchResponse(
                results=[],
                total_found=0,
                search_time_ms=int((time.time() - start_time) * 1000),
                search_mode=request.search_mode,
                query_metadata={"error": str(e)}
            )

    async def _vector_search(self, request: SearchRequest) -> List[SearchResult]:
        """Perform pure vector similarity search."""
        # Generate query embedding
        query_embedding = await self.embeddings.generate_text_embedding(request.query)

        # Search using text embeddings by default
        results = await self.postgres.vector_search(
            embedding=query_embedding,
            embedding_type="text",
            limit=request.limit * 2,  # Get more results for post-processing
            similarity_threshold=request.similarity_threshold,
            content_types=request.content_types
        )

        # Convert to SearchResult objects
        search_results = []
        for i, (memory_record, similarity) in enumerate(results):
            search_results.append(SearchResult(
                memory_record=memory_record,
                similarity_score=similarity,
                rank=i + 1,
                search_metadata={"search_type": "vector",
                                 "embedding_type": "text"}
            ))

        return search_results[:request.limit]

    async def _text_search(self, request: SearchRequest) -> List[SearchResult]:
        """Perform pure full-text search."""
        results = await self.postgres.text_search(
            query_text=request.query,
            limit=request.limit,
            content_types=request.content_types
        )

        # Convert to SearchResult objects
        search_results = []
        for i, (memory_record, score) in enumerate(results):
            # Normalize text search score to 0-1 range
            normalized_score = min(1.0, max(0.0, score / 10.0))
            search_results.append(SearchResult(
                memory_record=memory_record,
                similarity_score=normalized_score,
                rank=i + 1,
                search_metadata={"search_type": "text", "text_score": score}
            ))

        return search_results

    async def _hybrid_search(self, request: SearchRequest) -> List[SearchResult]:
        """Perform hybrid search combining vector and text search."""
        # Run both searches concurrently
        vector_task = asyncio.create_task(self._vector_search(request))
        text_task = asyncio.create_task(self._text_search(request))

        vector_results, text_results = await asyncio.gather(vector_task, text_task)

        # Merge and re-rank results
        merged_results = await self._merge_search_results(vector_results, text_results)

        return merged_results[:request.limit]

    async def _cross_modal_search(self, request: SearchRequest) -> List[SearchResult]:
        """Perform cross-modal search using unified embeddings."""
        if not request.query_content_type or not request.target_content_types:
            # Fallback to regular vector search
            return await self._vector_search(request)

        # Generate appropriate embedding based on query content type
        if request.query_content_type == ContentType.TEXT:
            query_embedding = await self.embeddings.generate_text_embedding(request.query)
        elif request.query_content_type == ContentType.AUDIO and self.whisper:
            # Process audio query
            audio_result = await self.whisper.process_audio(request.query)
            query_embedding = await self.embeddings.generate_text_embedding(
                audio_result.get("transcript", "")
            )
        elif request.query_content_type == ContentType.IMAGE:
            query_embedding = await self.embeddings.generate_image_embedding(request.query)
        else:
            # Fallback to text embedding
            query_embedding = await self.embeddings.generate_text_embedding(request.query)

        # Search using unified embeddings
        results = await self.postgres.vector_search(
            embedding=query_embedding,
            embedding_type="unified",
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
            content_types=request.target_content_types
        )

        # Convert to SearchResult objects
        search_results = []
        for i, (memory_record, similarity) in enumerate(results):
            search_results.append(SearchResult(
                memory_record=memory_record,
                similarity_score=similarity,
                rank=i + 1,
                search_metadata={
                    "search_type": "cross_modal",
                    "query_type": request.query_content_type.value,
                    "target_types": [ct.value for ct in request.target_content_types]
                }
            ))

        return search_results

    async def _merge_search_results(self,
                                    vector_results: List[SearchResult],
                                    text_results: List[SearchResult]) -> List[SearchResult]:
        """Merge and re-rank results from vector and text search."""
        # Create a mapping of memory_id to results
        result_map = {}

        # Add vector results
        for result in vector_results:
            memory_id = result.memory_record.id
            result_map[memory_id] = {
                "memory_record": result.memory_record,
                "vector_score": result.similarity_score,
                "text_score": 0.0,
                "vector_rank": result.rank,
                "text_rank": None
            }

        # Add/update with text results
        for result in text_results:
            memory_id = result.memory_record.id
            if memory_id in result_map:
                result_map[memory_id]["text_score"] = result.similarity_score
                result_map[memory_id]["text_rank"] = result.rank
            else:
                result_map[memory_id] = {
                    "memory_record": result.memory_record,
                    "vector_score": 0.0,
                    "text_score": result.similarity_score,
                    "vector_rank": None,
                    "text_rank": result.rank
                }

        # Calculate hybrid scores
        merged_results = []
        for memory_id, data in result_map.items():
            # Calculate weighted hybrid score
            hybrid_score = (
                data["vector_score"] * self.vector_weight +
                data["text_score"] * self.text_weight
            )

            # Create merged result
            merged_result = SearchResult(
                memory_record=data["memory_record"],
                similarity_score=hybrid_score,
                rank=0,  # Will be set after sorting
                search_metadata={
                    "search_type": "hybrid",
                    "vector_score": data["vector_score"],
                    "text_score": data["text_score"],
                    "vector_rank": data["vector_rank"],
                    "text_rank": data["text_rank"],
                    "hybrid_score": hybrid_score
                }
            )
            merged_results.append(merged_result)

        # Sort by hybrid score
        merged_results.sort(key=lambda x: x.similarity_score, reverse=True)

        # Update ranks
        for i, result in enumerate(merged_results):
            result.rank = i + 1

        return merged_results

    async def _filter_and_rank_results(self,
                                       results: List[SearchResult],
                                       request: SearchRequest) -> List[SearchResult]:
        """Apply final filtering and ranking to search results."""
        filtered_results = []

        for result in results:
            # Apply similarity threshold
            if result.similarity_score < request.similarity_threshold:
                continue

            # Apply memory type filtering
            if request.memory_types:
                if result.memory_record.memory_type not in request.memory_types:
                    continue

            # Apply tag filtering
            if request.tags:
                memory_tags = set(result.memory_record.tags)
                request_tags = set(request.tags)
                if not memory_tags.intersection(request_tags):
                    continue

            # Apply date filtering
            if request.created_after:
                if result.memory_record.created_at < request.created_after:
                    continue

            if request.created_before:
                if result.memory_record.created_at > request.created_before:
                    continue

            filtered_results.append(result)

        # Re-rank and limit results
        return filtered_results[:request.limit]

    async def cross_modal_search(self, request: CrossModalSearchRequest) -> List[SearchResult]:
        """
        Specialized cross-modal search with different query and target types.

        Args:
            request: Cross-modal search request

        Returns:
            List of search results
        """
        try:
            # Generate query embedding based on type
            if request.query_type == ContentType.TEXT:
                query_embedding = await self.embeddings.generate_text_embedding(request.query_content)
            elif request.query_type == ContentType.AUDIO and self.whisper:
                audio_result = await self.whisper.process_audio(request.query_content)
                # Use transcript for cross-modal search
                query_embedding = await self.embeddings.generate_text_embedding(
                    audio_result.get("transcript", "")
                )
            elif request.query_type == ContentType.IMAGE:
                query_embedding = await self.embeddings.generate_image_embedding(request.query_content)
            else:
                raise ValueError(
                    f"Unsupported query type: {request.query_type}")

            # Search using unified embeddings
            results = await self.postgres.vector_search(
                embedding=query_embedding,
                embedding_type="unified",
                limit=request.limit,
                similarity_threshold=request.similarity_threshold,
                content_types=request.target_types
            )

            # Convert to SearchResult objects
            search_results = []
            for i, (memory_record, similarity) in enumerate(results):
                search_results.append(SearchResult(
                    memory_record=memory_record,
                    similarity_score=similarity,
                    rank=i + 1,
                    search_metadata={
                        "search_type": "cross_modal_specialized",
                        "query_type": request.query_type.value,
                        "target_types": [ct.value for ct in request.target_types]
                    }
                ))

            return search_results

        except Exception as e:
            logger.error(f"Cross-modal search failed: {e}")
            return []

    async def find_similar_memories(self,
                                    memory_record: MemoryRecord,
                                    limit: int = 10,
                                    similarity_threshold: float = 0.7) -> List[SearchResult]:
        """
        Find memories similar to a given memory record.

        Args:
            memory_record: Reference memory record
            limit: Maximum results to return
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of similar memory records
        """
        try:
            # Use the best available embedding
            if memory_record.unified_embedding:
                embedding = memory_record.unified_embedding
                embedding_type = "unified"
            elif memory_record.text_embedding:
                embedding = memory_record.text_embedding
                embedding_type = "text"
            elif memory_record.image_embedding:
                embedding = memory_record.image_embedding
                embedding_type = "image"
            elif memory_record.audio_embedding:
                embedding = memory_record.audio_embedding
                embedding_type = "audio"
            else:
                logger.warning("No embeddings available for similarity search")
                return []

            # Search for similar memories
            results = await self.postgres.vector_search(
                embedding=embedding,
                embedding_type=embedding_type,
                limit=limit + 1,  # +1 to exclude the original
                similarity_threshold=similarity_threshold
            )

            # Convert to SearchResult objects and exclude the original
            search_results = []
            for i, (similar_record, similarity) in enumerate(results):
                if similar_record.id != memory_record.id:  # Exclude the original
                    search_results.append(SearchResult(
                        memory_record=similar_record,
                        similarity_score=similarity,
                        rank=len(search_results) + 1,
                        search_metadata={
                            "search_type": "similarity",
                            "reference_id": str(memory_record.id),
                            "embedding_type": embedding_type
                        }
                    ))

            return search_results[:limit]

        except Exception as e:
            logger.error(f"Similar memories search failed: {e}")
            return []

    async def semantic_clustering(self,
                                  content_types: Optional[List[ContentType]] = None,
                                  cluster_threshold: float = 0.8,
                                  min_cluster_size: int = 2) -> Dict[str, List[MemoryRecord]]:
        """
        Perform semantic clustering of memories.

        Args:
            content_types: Filter by content types
            cluster_threshold: Similarity threshold for clustering
            min_cluster_size: Minimum cluster size

        Returns:
            Dictionary of clusters
        """
        try:
            # This is a simplified clustering implementation
            # In production, you might use more sophisticated clustering algorithms

            # Get all memories (limited for performance)
            all_memories_query = """
                SELECT * FROM memory_records 
                WHERE unified_embedding IS NOT NULL
                ORDER BY created_at DESC 
                LIMIT 1000
            """

            if content_types:
                content_type_values = [ct.value for ct in content_types]
                all_memories_query = f"""
                    SELECT * FROM memory_records 
                    WHERE unified_embedding IS NOT NULL
                    AND content_type = ANY($1)
                    ORDER BY created_at DESC 
                    LIMIT 1000
                """

            # This would require direct database access for clustering
            # For now, return empty clusters
            logger.info("Semantic clustering would be implemented here")
            return {}

        except Exception as e:
            logger.error(f"Semantic clustering failed: {e}")
            return {}

    def configure_search_weights(self, vector_weight: float, text_weight: float):
        """
        Configure the weights for hybrid search.

        Args:
            vector_weight: Weight for vector similarity (0-1)
            text_weight: Weight for text search (0-1)
        """
        total_weight = vector_weight + text_weight
        if total_weight > 0:
            self.vector_weight = vector_weight / total_weight
            self.text_weight = text_weight / total_weight
            logger.info(
                f"Updated search weights: vector={self.vector_weight:.2f}, text={self.text_weight:.2f}")
        else:
            logger.warning(
                "Invalid weights provided, keeping current configuration")

    async def get_search_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics and performance metrics."""
        try:
            # Get basic statistics from the database
            stats = await self.postgres.get_memory_statistics()

            # Add search engine specific stats
            stats.update({
                "vector_weight": self.vector_weight,
                "text_weight": self.text_weight,
                "embedding_cache_stats": self.embeddings.get_cache_stats()
            })

            return stats

        except Exception as e:
            logger.error(f"Failed to get search statistics: {e}")
            return {}
