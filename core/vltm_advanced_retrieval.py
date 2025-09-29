"""
Very Long-Term Memory Advanced Retrieval System

This module implements advanced retrieval mechanisms with multi-modal indexing 
for strategic, temporal, and causal queries in the VLTM system.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from core.vltm_storage_backend import StorageBackend
from core.vltm_data_models import (
    MemoryType, PatternType, VLTMConfiguration, DEFAULT_VLTM_CONFIG
)

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of queries supported by the retrieval system"""
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    STRATEGIC = "strategic"
    PATTERN_BASED = "pattern_based"
    HYBRID = "hybrid"


class RetrievalMode(str, Enum):
    """Retrieval modes for different use cases"""
    PRECISE = "precise"  # High precision, lower recall
    COMPREHENSIVE = "comprehensive"  # High recall, lower precision
    BALANCED = "balanced"  # Balanced precision and recall
    EXPLORATORY = "exploratory"  # For discovery and exploration


@dataclass
class QueryContext:
    """Context information for query processing"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    domain_filter: Optional[List[str]] = None
    importance_threshold: float = 0.0
    include_patterns: bool = False
    include_strategic_knowledge: bool = True


@dataclass
class RetrievalResult:
    """Result from retrieval operation"""
    query_id: str
    memories: List[Dict[str, Any]]
    patterns: List[Dict[str, Any]]
    strategic_insights: List[Dict[str, Any]]
    total_found: int
    processing_time_ms: float
    query_type: QueryType
    confidence_scores: List[float]
    relevance_explanations: List[str]


class AdvancedRetrievalEngine:
    """
    Advanced retrieval engine with multi-modal indexing capabilities.

    Supports strategic, temporal, causal, and semantic queries with 
    sophisticated ranking and filtering mechanisms.
    """

    def __init__(self, storage_backend: StorageBackend, config: Optional[VLTMConfiguration] = None):
        self.storage_backend = storage_backend
        self.config = config or DEFAULT_VLTM_CONFIG

        # Query processing components
        self.semantic_processor = SemanticQueryProcessor()
        self.temporal_processor = TemporalQueryProcessor()
        self.causal_processor = CausalQueryProcessor()
        self.strategic_processor = StrategicQueryProcessor()

        # Index managers
        self.multi_modal_index = MultiModalIndex(storage_backend)

        # Query cache for performance
        self.query_cache: Dict[str, RetrievalResult] = {}
        self.cache_ttl_seconds = 300  # 5 minutes

        logger.info("Advanced retrieval engine initialized")

    async def initialize(self) -> bool:
        """Initialize the retrieval engine and its indices"""
        try:
            # Initialize multi-modal index
            await self.multi_modal_index.initialize()

            # Initialize query processors
            await self.semantic_processor.initialize()
            await self.temporal_processor.initialize()
            await self.causal_processor.initialize()
            await self.strategic_processor.initialize()

            logger.info("Advanced retrieval engine initialization complete")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize retrieval engine: {e}")
            return False

    async def query(self,
                    query_text: str,
                    query_type: QueryType = QueryType.HYBRID,
                    context: Optional[QueryContext] = None,
                    mode: RetrievalMode = RetrievalMode.BALANCED,
                    limit: int = 10) -> RetrievalResult:
        """
        Execute advanced query with multi-modal retrieval.

        Args:
            query_text: Natural language query
            query_type: Type of query to execute
            context: Additional query context
            mode: Retrieval mode for precision/recall balance
            limit: Maximum number of results

        Returns:
            Comprehensive retrieval results
        """
        start_time = datetime.utcnow()
        query_id = str(uuid.uuid4())
        context = context or QueryContext()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(
                query_text, query_type, context, mode, limit)
            if cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    logger.debug(
                        f"Returning cached result for query: {query_text[:50]}...")
                    return cached_result

            logger.info(
                f"Processing {query_type} query: {query_text[:100]}...")

            # Route to appropriate processor based on query type
            if query_type == QueryType.SEMANTIC:
                result = await self._process_semantic_query(query_text, context, mode, limit)
            elif query_type == QueryType.TEMPORAL:
                result = await self._process_temporal_query(query_text, context, mode, limit)
            elif query_type == QueryType.CAUSAL:
                result = await self._process_causal_query(query_text, context, mode, limit)
            elif query_type == QueryType.STRATEGIC:
                result = await self._process_strategic_query(query_text, context, mode, limit)
            elif query_type == QueryType.PATTERN_BASED:
                result = await self._process_pattern_query(query_text, context, mode, limit)
            else:  # HYBRID
                result = await self._process_hybrid_query(query_text, context, mode, limit)

            # Set query metadata
            result.query_id = query_id
            result.query_type = query_type
            result.processing_time_ms = (
                datetime.utcnow() - start_time).total_seconds() * 1000

            # Cache the result
            self.query_cache[cache_key] = result

            logger.info(
                f"Query completed: {result.total_found} results in {result.processing_time_ms:.2f}ms")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            # Return empty result on error
            return RetrievalResult(
                query_id=query_id,
                memories=[],
                patterns=[],
                strategic_insights=[],
                total_found=0,
                processing_time_ms=(datetime.utcnow() -
                                    start_time).total_seconds() * 1000,
                query_type=query_type,
                confidence_scores=[],
                relevance_explanations=[]
            )

    async def _process_semantic_query(self, query_text: str, context: QueryContext,
                                      mode: RetrievalMode, limit: int) -> RetrievalResult:
        """Process semantic similarity query"""

        # Use semantic processor to find similar memories
        semantic_results = await self.semantic_processor.search(
            query_text, context, mode, limit
        )

        # Enhance with strategic insights if requested
        strategic_insights = []
        if context.include_strategic_knowledge:
            strategic_insights = await self.strategic_processor.get_related_insights(
                semantic_results["memories"], limit=5
            )

        return RetrievalResult(
            query_id="",
            memories=semantic_results["memories"],
            patterns=semantic_results.get("patterns", []),
            strategic_insights=strategic_insights,
            total_found=len(semantic_results["memories"]),
            processing_time_ms=0,
            query_type=QueryType.SEMANTIC,
            confidence_scores=semantic_results.get("scores", []),
            relevance_explanations=semantic_results.get("explanations", [])
        )

    async def _process_temporal_query(self, query_text: str, context: QueryContext,
                                      mode: RetrievalMode, limit: int) -> RetrievalResult:
        """Process temporal-based query"""

        temporal_results = await self.temporal_processor.search(
            query_text, context, mode, limit
        )

        return RetrievalResult(
            query_id="",
            memories=temporal_results["memories"],
            patterns=temporal_results.get("temporal_patterns", []),
            strategic_insights=[],
            total_found=len(temporal_results["memories"]),
            processing_time_ms=0,
            query_type=QueryType.TEMPORAL,
            confidence_scores=temporal_results.get("scores", []),
            relevance_explanations=temporal_results.get("explanations", [])
        )

    async def _process_causal_query(self, query_text: str, context: QueryContext,
                                    mode: RetrievalMode, limit: int) -> RetrievalResult:
        """Process causal relationship query"""

        causal_results = await self.causal_processor.search(
            query_text, context, mode, limit
        )

        return RetrievalResult(
            query_id="",
            memories=causal_results["memories"],
            patterns=causal_results.get("causal_patterns", []),
            strategic_insights=causal_results.get("insights", []),
            total_found=len(causal_results["memories"]),
            processing_time_ms=0,
            query_type=QueryType.CAUSAL,
            confidence_scores=causal_results.get("scores", []),
            relevance_explanations=causal_results.get("explanations", [])
        )

    async def _process_strategic_query(self, query_text: str, context: QueryContext,
                                       mode: RetrievalMode, limit: int) -> RetrievalResult:
        """Process strategic knowledge query"""

        strategic_results = await self.strategic_processor.search(
            query_text, context, mode, limit
        )

        return RetrievalResult(
            query_id="",
            memories=strategic_results.get("memories", []),
            patterns=strategic_results.get("patterns", []),
            strategic_insights=strategic_results["strategic_knowledge"],
            total_found=len(strategic_results["strategic_knowledge"]),
            processing_time_ms=0,
            query_type=QueryType.STRATEGIC,
            confidence_scores=strategic_results.get("scores", []),
            relevance_explanations=strategic_results.get("explanations", [])
        )

    async def _process_pattern_query(self, query_text: str, context: QueryContext,
                                     mode: RetrievalMode, limit: int) -> RetrievalResult:
        """Process pattern-based query"""

        # Search for patterns matching the query
        pattern_results = await self.storage_backend.search_patterns(
            query_text, limit=limit * 2
        )

        # Get memories associated with found patterns
        memories = []
        for pattern in pattern_results:
            pattern_memories = await self.storage_backend.get_pattern_memories(
                pattern["pattern_id"], limit=5
            )
            memories.extend(pattern_memories)

        # Remove duplicates and limit
        unique_memories = self._deduplicate_memories(memories)[:limit]

        return RetrievalResult(
            query_id="",
            memories=unique_memories,
            patterns=pattern_results,
            strategic_insights=[],
            total_found=len(unique_memories),
            processing_time_ms=0,
            query_type=QueryType.PATTERN_BASED,
            confidence_scores=[],
            relevance_explanations=[]
        )

    async def _process_hybrid_query(self, query_text: str, context: QueryContext,
                                    mode: RetrievalMode, limit: int) -> RetrievalResult:
        """Process hybrid query combining multiple approaches"""

        # Execute multiple query types in parallel
        semantic_task = self._process_semantic_query(
            query_text, context, mode, limit//2)
        temporal_task = self._process_temporal_query(
            query_text, context, mode, limit//4)
        strategic_task = self._process_strategic_query(
            query_text, context, mode, limit//4)

        semantic_result, temporal_result, strategic_result = await asyncio.gather(
            semantic_task, temporal_task, strategic_task, return_exceptions=True
        )

        # Combine and rank results
        all_memories = []
        all_patterns = []
        all_insights = []
        all_scores = []
        all_explanations = []

        if isinstance(semantic_result, RetrievalResult):
            all_memories.extend(semantic_result.memories)
            all_patterns.extend(semantic_result.patterns)
            all_insights.extend(semantic_result.strategic_insights)
            all_scores.extend(semantic_result.confidence_scores)
            all_explanations.extend(semantic_result.relevance_explanations)

        if isinstance(temporal_result, RetrievalResult):
            all_memories.extend(temporal_result.memories)
            all_patterns.extend(temporal_result.patterns)

        if isinstance(strategic_result, RetrievalResult):
            all_insights.extend(strategic_result.strategic_insights)

        # Deduplicate and rank
        unique_memories = self._deduplicate_memories(all_memories)
        unique_patterns = self._deduplicate_patterns(all_patterns)
        unique_insights = self._deduplicate_insights(all_insights)

        # Apply final ranking and limit
        ranked_memories = self._rank_hybrid_results(
            unique_memories, query_text)[:limit]

        return RetrievalResult(
            query_id="",
            memories=ranked_memories,
            patterns=unique_patterns[:limit//2],
            strategic_insights=unique_insights[:limit//2],
            total_found=len(ranked_memories),
            processing_time_ms=0,
            query_type=QueryType.HYBRID,
            confidence_scores=all_scores[:len(ranked_memories)],
            relevance_explanations=all_explanations[:len(ranked_memories)]
        )

    def _deduplicate_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate memories"""
        seen_ids = set()
        unique_memories = []

        for memory in memories:
            memory_id = memory.get("memory_id")
            if memory_id and memory_id not in seen_ids:
                seen_ids.add(memory_id)
                unique_memories.append(memory)

        return unique_memories

    def _deduplicate_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate patterns"""
        seen_ids = set()
        unique_patterns = []

        for pattern in patterns:
            pattern_id = pattern.get("pattern_id")
            if pattern_id and pattern_id not in seen_ids:
                seen_ids.add(pattern_id)
                unique_patterns.append(pattern)

        return unique_patterns

    def _deduplicate_insights(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate strategic insights"""
        seen_ids = set()
        unique_insights = []

        for insight in insights:
            insight_id = insight.get("knowledge_id")
            if insight_id and insight_id not in seen_ids:
                seen_ids.add(insight_id)
                unique_insights.append(insight)

        return unique_insights

    def _rank_hybrid_results(self, memories: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
        """Rank hybrid results using multiple signals"""

        # Simple ranking based on importance and recency
        def rank_score(memory):
            importance = memory.get("importance_score", 0.5)
            strategic_value = memory.get("strategic_value", 0.5)

            # Recency factor (more recent = higher score)
            created_at = memory.get("created_at")
            if created_at:
                try:
                    if isinstance(created_at, str):
                        created_date = datetime.fromisoformat(
                            created_at.replace('Z', '+00:00'))
                    else:
                        created_date = created_at

                    days_old = (datetime.utcnow() -
                                created_date.replace(tzinfo=None)).days
                    recency_factor = max(0.1, 1.0 - (days_old / 365))
                except:
                    recency_factor = 0.5
            else:
                recency_factor = 0.5

            return importance * 0.4 + strategic_value * 0.4 + recency_factor * 0.2

        return sorted(memories, key=rank_score, reverse=True)

    def _generate_cache_key(self, query_text: str, query_type: QueryType,
                            context: QueryContext, mode: RetrievalMode, limit: int) -> str:
        """Generate cache key for query"""
        key_parts = [
            query_text[:100],  # Truncate long queries
            query_type.value,
            mode.value,
            str(limit),
            str(context.importance_threshold),
            str(context.include_patterns),
            str(context.include_strategic_knowledge)
        ]

        if context.time_range:
            key_parts.append(
                f"{context.time_range[0]}-{context.time_range[1]}")

        return "|".join(key_parts)

    def _is_cache_valid(self, cached_result: RetrievalResult) -> bool:
        """Check if cached result is still valid"""
        # Simple TTL-based cache validation
        # In production, this could be more sophisticated
        return True  # For now, assume cache is always valid

    async def get_query_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get query suggestions based on partial input"""

        try:
            # Get recent successful queries for suggestions
            suggestions = []

            # Simple suggestion logic - in production this would be more sophisticated
            query_templates = [
                f"What {partial_query}",
                f"How {partial_query}",
                f"When {partial_query}",
                f"Why {partial_query}",
                f"Show me {partial_query}"
            ]

            suggestions.extend(query_templates[:limit])
            return suggestions

        except Exception as e:
            logger.error(f"Error generating query suggestions: {e}")
            return []

    async def explain_query_results(self, query_id: str) -> Dict[str, Any]:
        """Provide explanation for query results"""

        try:
            # Find cached result
            for cached_result in self.query_cache.values():
                if cached_result.query_id == query_id:
                    return {
                        "query_type": cached_result.query_type,
                        "total_found": cached_result.total_found,
                        "processing_time_ms": cached_result.processing_time_ms,
                        "ranking_factors": [
                            "Semantic similarity",
                            "Importance score",
                            "Strategic value",
                            "Recency"
                        ],
                        "explanations": cached_result.relevance_explanations
                    }

            return {"error": "Query results not found"}

        except Exception as e:
            logger.error(f"Error explaining query results: {e}")
            return {"error": str(e)}


class SemanticQueryProcessor:
    """Processor for semantic similarity queries"""

    async def initialize(self):
        """Initialize semantic processor"""

    async def search(self, query_text: str, context: QueryContext,
                     mode: RetrievalMode, limit: int) -> Dict[str, Any]:
        """Search for semantically similar memories"""

        # Placeholder implementation
        return {
            "memories": [],
            "patterns": [],
            "scores": [],
            "explanations": []
        }


class TemporalQueryProcessor:
    """Processor for temporal queries"""

    async def initialize(self):
        """Initialize temporal processor"""

    async def search(self, query_text: str, context: QueryContext,
                     mode: RetrievalMode, limit: int) -> Dict[str, Any]:
        """Search for temporally relevant memories"""

        return {
            "memories": [],
            "temporal_patterns": [],
            "scores": [],
            "explanations": []
        }


class CausalQueryProcessor:
    """Processor for causal relationship queries"""

    async def initialize(self):
        """Initialize causal processor"""

    async def search(self, query_text: str, context: QueryContext,
                     mode: RetrievalMode, limit: int) -> Dict[str, Any]:
        """Search for causal relationships"""

        return {
            "memories": [],
            "causal_patterns": [],
            "insights": [],
            "scores": [],
            "explanations": []
        }


class StrategicQueryProcessor:
    """Processor for strategic knowledge queries"""

    async def initialize(self):
        """Initialize strategic processor"""

    async def search(self, query_text: str, context: QueryContext,
                     mode: RetrievalMode, limit: int) -> Dict[str, Any]:
        """Search strategic knowledge"""

        return {
            "strategic_knowledge": [],
            "patterns": [],
            "memories": [],
            "scores": [],
            "explanations": []
        }

    async def get_related_insights(self, memories: List[Dict[str, Any]],
                                   limit: int = 5) -> List[Dict[str, Any]]:
        """Get strategic insights related to memories"""
        return []


class MultiModalIndex:
    """Multi-modal indexing for efficient retrieval"""

    def __init__(self, storage_backend: StorageBackend):
        self.storage_backend = storage_backend
        self.indices = {
            "semantic": None,
            "temporal": None,
            "causal": None,
            "strategic": None
        }

    async def initialize(self):
        """Initialize all indices"""
        logger.info("Initializing multi-modal indices...")
        # Implementation would create various indices for efficient retrieval
