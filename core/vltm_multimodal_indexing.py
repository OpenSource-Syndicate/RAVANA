"""
Very Long-Term Memory Multi-Modal Indexing System

This module implements multi-dimensional indexing for efficient memory retrieval
across temporal, semantic, causal, and strategic dimensions.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict

from core.vltm_storage_backend import StorageBackend
from core.vltm_data_models import MemoryType, PatternType, VLTMConfiguration

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    """Types of indices for multi-modal retrieval"""
    TEMPORAL = "temporal"
    SEMANTIC = "semantic" 
    CAUSAL = "causal"
    STRATEGIC = "strategic"
    PATTERN = "pattern"
    IMPORTANCE = "importance"


class IndexDimension(str, Enum):
    """Dimensional aspects for indexing"""
    TIME_BASED = "time_based"
    CONTENT_BASED = "content_based"
    RELATIONSHIP_BASED = "relationship_based"
    VALUE_BASED = "value_based"


@dataclass
class IndexEntry:
    """Entry in a multi-modal index"""
    memory_id: str
    index_type: IndexType
    dimension_value: Any
    timestamp: datetime
    importance_score: float
    metadata: Dict[str, Any]


@dataclass
class IndexStats:
    """Statistics for index operations"""
    total_entries: int = 0
    index_size_bytes: int = 0
    avg_lookup_time_ms: float = 0.0
    cache_hit_ratio: float = 0.0
    last_rebuild_time: Optional[datetime] = None


class MultiModalIndex:
    """
    Multi-dimensional indexing system for VLTM efficient retrieval.
    
    Provides indexing across:
    - Temporal dimension (time-based access patterns)
    - Semantic dimension (content similarity and themes)
    - Causal dimension (cause-effect relationships)
    - Strategic dimension (importance and strategic value)
    - Pattern dimension (behavioral and structural patterns)
    """
    
    def __init__(self, storage_backend: StorageBackend, config: Optional[VLTMConfiguration] = None):
        self.storage_backend = storage_backend
        self.config = config
        
        # Index structures for different dimensions
        self.indices: Dict[IndexType, Dict[str, Any]] = {
            IndexType.TEMPORAL: {},
            IndexType.SEMANTIC: {},
            IndexType.CAUSAL: {},
            IndexType.STRATEGIC: {},
            IndexType.PATTERN: {},
            IndexType.IMPORTANCE: {}
        }
        
        # Index statistics
        self.index_stats: Dict[IndexType, IndexStats] = {}
        
        # Cache for frequent queries
        self.query_cache: Dict[str, Any] = {}
        self.cache_ttl_seconds = 300
        
        # Index managers
        self.temporal_indexer = TemporalIndexer()
        self.semantic_indexer = SemanticIndexer()
        self.causal_indexer = CausalIndexer()
        self.strategic_indexer = StrategicIndexer()
        self.pattern_indexer = PatternIndexer()
        
        logger.info("MultiModalIndex initialized")
    
    async def initialize(self) -> bool:
        """Initialize all index dimensions"""
        try:
            # Initialize individual indexers
            await self.temporal_indexer.initialize()
            await self.semantic_indexer.initialize()
            await self.causal_indexer.initialize()
            await self.strategic_indexer.initialize()
            await self.pattern_indexer.initialize()
            
            # Build initial indices
            await self._build_initial_indices()
            
            # Initialize statistics
            await self._initialize_index_stats()
            
            logger.info("MultiModalIndex initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MultiModalIndex: {e}")
            return False
    
    async def _build_initial_indices(self):
        """Build initial indices from existing memories"""
        try:
            # Get all memories for indexing
            memories = await self.storage_backend.get_all_memories(limit=10000)
            
            logger.info(f"Building initial indices for {len(memories)} memories")
            
            # Index memories across all dimensions
            for memory in memories:
                await self.index_memory(memory)
            
            logger.info("Initial indices built successfully")
            
        except Exception as e:
            logger.error(f"Error building initial indices: {e}")
    
    async def _initialize_index_stats(self):
        """Initialize statistics for all index types"""
        for index_type in IndexType:
            self.index_stats[index_type] = IndexStats(
                total_entries=len(self.indices[index_type]),
                last_rebuild_time=datetime.utcnow()
            )
    
    async def index_memory(self, memory: Dict[str, Any]) -> bool:
        """Index a memory across all dimensions"""
        try:
            memory_id = memory.get("memory_id")
            if not memory_id:
                return False
            
            # Index in temporal dimension
            await self.temporal_indexer.index_memory(memory, self.indices[IndexType.TEMPORAL])
            
            # Index in semantic dimension
            await self.semantic_indexer.index_memory(memory, self.indices[IndexType.SEMANTIC])
            
            # Index in causal dimension
            await self.causal_indexer.index_memory(memory, self.indices[IndexType.CAUSAL])
            
            # Index in strategic dimension
            await self.strategic_indexer.index_memory(memory, self.indices[IndexType.STRATEGIC])
            
            # Index in pattern dimension
            await self.pattern_indexer.index_memory(memory, self.indices[IndexType.PATTERN])
            
            # Index in importance dimension
            await self._index_by_importance(memory)
            
            # Update statistics
            await self._update_index_stats(memory)
            
            return True
            
        except Exception as e:
            logger.error(f"Error indexing memory {memory.get('memory_id')}: {e}")
            return False
    
    async def _index_by_importance(self, memory: Dict[str, Any]):
        """Index memory by importance score"""
        try:
            memory_id = memory.get("memory_id")
            importance_score = memory.get("importance_score", 0.5)
            
            # Create importance buckets
            if importance_score >= 0.8:
                bucket = "critical"
            elif importance_score >= 0.6:
                bucket = "high"
            elif importance_score >= 0.4:
                bucket = "medium"
            else:
                bucket = "low"
            
            if bucket not in self.indices[IndexType.IMPORTANCE]:
                self.indices[IndexType.IMPORTANCE][bucket] = []
            
            self.indices[IndexType.IMPORTANCE][bucket].append({
                "memory_id": memory_id,
                "importance_score": importance_score,
                "indexed_at": datetime.utcnow()
            })
            
        except Exception as e:
            logger.error(f"Error indexing by importance: {e}")
    
    async def _update_index_stats(self, memory: Dict[str, Any]):
        """Update index statistics after adding a memory"""
        for index_type in IndexType:
            if index_type in self.index_stats:
                self.index_stats[index_type].total_entries += 1
    
    async def query_temporal(self, time_range: Tuple[datetime, datetime], 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Query memories by temporal dimension"""
        try:
            return await self.temporal_indexer.query(
                self.indices[IndexType.TEMPORAL], time_range, limit
            )
        except Exception as e:
            logger.error(f"Error in temporal query: {e}")
            return []
    
    async def query_semantic(self, query_text: str, 
                           similarity_threshold: float = 0.7,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Query memories by semantic similarity"""
        try:
            return await self.semantic_indexer.query(
                self.indices[IndexType.SEMANTIC], query_text, similarity_threshold, limit
            )
        except Exception as e:
            logger.error(f"Error in semantic query: {e}")
            return []
    
    async def query_causal(self, cause_pattern: str, 
                         effect_pattern: str,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Query memories by causal relationships"""
        try:
            return await self.causal_indexer.query(
                self.indices[IndexType.CAUSAL], cause_pattern, effect_pattern, limit
            )
        except Exception as e:
            logger.error(f"Error in causal query: {e}")
            return []
    
    async def query_strategic(self, strategic_value_threshold: float = 0.6,
                            memory_types: Optional[List[MemoryType]] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """Query memories by strategic importance"""
        try:
            return await self.strategic_indexer.query(
                self.indices[IndexType.STRATEGIC], strategic_value_threshold, memory_types, limit
            )
        except Exception as e:
            logger.error(f"Error in strategic query: {e}")
            return []
    
    async def query_patterns(self, pattern_type: PatternType,
                           pattern_frequency_threshold: int = 2,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """Query memories by pattern matching"""
        try:
            return await self.pattern_indexer.query(
                self.indices[IndexType.PATTERN], pattern_type, pattern_frequency_threshold, limit
            )
        except Exception as e:
            logger.error(f"Error in pattern query: {e}")
            return []
    
    async def query_by_importance(self, importance_level: str,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """Query memories by importance level"""
        try:
            if importance_level not in self.indices[IndexType.IMPORTANCE]:
                return []
            
            importance_entries = self.indices[IndexType.IMPORTANCE][importance_level]
            
            # Get memory details for the entries
            results = []
            for entry in importance_entries[:limit]:
                memory = await self.storage_backend.get_memory(entry["memory_id"])
                if memory:
                    results.append(memory)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in importance query: {e}")
            return []
    
    async def multi_dimensional_query(self, 
                                    temporal_range: Optional[Tuple[datetime, datetime]] = None,
                                    semantic_query: Optional[str] = None,
                                    importance_threshold: float = 0.5,
                                    memory_types: Optional[List[MemoryType]] = None,
                                    limit: int = 50) -> List[Dict[str, Any]]:
        """Execute multi-dimensional query across indices"""
        try:
            # Collect candidate memories from each dimension
            candidates = set()
            
            # Temporal candidates
            if temporal_range:
                temporal_results = await self.query_temporal(temporal_range, limit * 2)
                candidates.update(m.get("memory_id") for m in temporal_results if m.get("memory_id"))
            
            # Semantic candidates
            if semantic_query:
                semantic_results = await self.query_semantic(semantic_query, limit=limit * 2)
                semantic_ids = set(m.get("memory_id") for m in semantic_results if m.get("memory_id"))
                
                if candidates:
                    candidates = candidates.intersection(semantic_ids)
                else:
                    candidates = semantic_ids
            
            # Strategic candidates
            strategic_results = await self.query_strategic(importance_threshold, memory_types, limit * 2)
            strategic_ids = set(m.get("memory_id") for m in strategic_results if m.get("memory_id"))
            
            if candidates:
                candidates = candidates.intersection(strategic_ids)
            else:
                candidates = strategic_ids
            
            # Retrieve full memory objects for final candidates
            final_results = []
            for memory_id in list(candidates)[:limit]:
                memory = await self.storage_backend.get_memory(memory_id)
                if memory:
                    final_results.append(memory)
            
            # Sort by relevance (importance score)
            final_results.sort(key=lambda m: m.get("importance_score", 0), reverse=True)
            
            return final_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in multi-dimensional query: {e}")
            return []
    
    async def rebuild_index(self, index_type: IndexType) -> bool:
        """Rebuild a specific index"""
        try:
            logger.info(f"Rebuilding {index_type} index...")
            
            # Clear existing index
            self.indices[index_type] = {}
            
            # Get all memories
            memories = await self.storage_backend.get_all_memories(limit=10000)
            
            # Rebuild index based on type
            indexer_map = {
                IndexType.TEMPORAL: self.temporal_indexer,
                IndexType.SEMANTIC: self.semantic_indexer,
                IndexType.CAUSAL: self.causal_indexer,
                IndexType.STRATEGIC: self.strategic_indexer,
                IndexType.PATTERN: self.pattern_indexer
            }
            
            if index_type in indexer_map:
                indexer = indexer_map[index_type]
                for memory in memories:
                    await indexer.index_memory(memory, self.indices[index_type])
            elif index_type == IndexType.IMPORTANCE:
                for memory in memories:
                    await self._index_by_importance(memory)
            
            # Update statistics
            self.index_stats[index_type] = IndexStats(
                total_entries=len(self.indices[index_type]),
                last_rebuild_time=datetime.utcnow()
            )
            
            logger.info(f"Rebuilt {index_type} index with {self.index_stats[index_type].total_entries} entries")
            return True
            
        except Exception as e:
            logger.error(f"Error rebuilding {index_type} index: {e}")
            return False
    
    async def rebuild_all_indices(self) -> bool:
        """Rebuild all indices"""
        try:
            success_count = 0
            for index_type in IndexType:
                if await self.rebuild_index(index_type):
                    success_count += 1
            
            logger.info(f"Rebuilt {success_count}/{len(IndexType)} indices successfully")
            return success_count == len(IndexType)
            
        except Exception as e:
            logger.error(f"Error rebuilding all indices: {e}")
            return False
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get comprehensive index statistics"""
        total_entries = sum(stats.total_entries for stats in self.index_stats.values())
        
        return {
            "total_entries_across_all_indices": total_entries,
            "index_types_count": len(IndexType),
            "individual_index_stats": {
                index_type.value: {
                    "total_entries": stats.total_entries,
                    "index_size_bytes": stats.index_size_bytes,
                    "avg_lookup_time_ms": stats.avg_lookup_time_ms,
                    "cache_hit_ratio": stats.cache_hit_ratio,
                    "last_rebuild_time": stats.last_rebuild_time.isoformat() if stats.last_rebuild_time else None
                }
                for index_type, stats in self.index_stats.items()
            },
            "cache_size": len(self.query_cache),
            "supported_dimensions": [dim.value for dim in IndexDimension]
        }
    
    async def optimize_indices(self) -> Dict[str, Any]:
        """Optimize all indices for better performance"""
        try:
            optimization_results = {}
            
            for index_type in IndexType:
                # Simple optimization: remove stale entries
                before_count = len(self.indices[index_type])
                
                # Clear cache entries older than TTL
                current_time = datetime.utcnow()
                
                # Optimize temporal index
                if index_type == IndexType.TEMPORAL:
                    optimized_count = await self._optimize_temporal_index()
                    optimization_results[index_type.value] = {
                        "before_count": before_count,
                        "after_count": optimized_count,
                        "optimization_type": "temporal_cleanup"
                    }
                
                # For other indices, perform general cleanup
                else:
                    # Simple cleanup - in production would be more sophisticated
                    optimization_results[index_type.value] = {
                        "before_count": before_count,
                        "after_count": before_count,  # No change for now
                        "optimization_type": "no_optimization"
                    }
            
            # Clear stale query cache
            cache_before = len(self.query_cache)
            self.query_cache.clear()  # Simple cache clear
            
            optimization_results["query_cache"] = {
                "cache_entries_cleared": cache_before,
                "cache_entries_remaining": len(self.query_cache)
            }
            
            logger.info("Index optimization completed")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error optimizing indices: {e}")
            return {"error": str(e)}
    
    async def _optimize_temporal_index(self) -> int:
        """Optimize temporal index by removing old entries"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=365)  # Keep 1 year
            
            optimized_index = {}
            for time_key, entries in self.indices[IndexType.TEMPORAL].items():
                if isinstance(entries, list):
                    # Filter entries newer than cutoff
                    recent_entries = [
                        entry for entry in entries 
                        if entry.get("timestamp", datetime.min) > cutoff_time
                    ]
                    if recent_entries:
                        optimized_index[time_key] = recent_entries
                else:
                    optimized_index[time_key] = entries
            
            self.indices[IndexType.TEMPORAL] = optimized_index
            return len(optimized_index)
            
        except Exception as e:
            logger.error(f"Error optimizing temporal index: {e}")
            return 0


class TemporalIndexer:
    """Indexer for temporal dimension"""
    
    async def initialize(self):
        """Initialize temporal indexer"""
        pass
    
    async def index_memory(self, memory: Dict[str, Any], temporal_index: Dict[str, Any]):
        """Index memory by temporal attributes"""
        try:
            memory_id = memory.get("memory_id")
            created_at = memory.get("created_at")
            
            if not created_at:
                return
            
            # Index by year-month for efficient range queries
            time_key = created_at.strftime("%Y-%m") if isinstance(created_at, datetime) else str(created_at)[:7]
            
            if time_key not in temporal_index:
                temporal_index[time_key] = []
            
            temporal_index[time_key].append({
                "memory_id": memory_id,
                "timestamp": created_at,
                "memory_type": memory.get("memory_type"),
                "importance_score": memory.get("importance_score", 0.5)
            })
            
        except Exception as e:
            logger.error(f"Error in temporal indexing: {e}")
    
    async def query(self, temporal_index: Dict[str, Any], 
                   time_range: Tuple[datetime, datetime], limit: int) -> List[Dict[str, Any]]:
        """Query temporal index"""
        try:
            start_time, end_time = time_range
            results = []
            
            for time_key, entries in temporal_index.items():
                for entry in entries:
                    entry_time = entry.get("timestamp")
                    if isinstance(entry_time, datetime) and start_time <= entry_time <= end_time:
                        results.append(entry)
            
            # Sort by timestamp and limit
            results.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in temporal query: {e}")
            return []


class SemanticIndexer:
    """Indexer for semantic dimension"""
    
    async def initialize(self):
        """Initialize semantic indexer"""
        pass
    
    async def index_memory(self, memory: Dict[str, Any], semantic_index: Dict[str, Any]):
        """Index memory by semantic content"""
        try:
            memory_id = memory.get("memory_id")
            content = str(memory.get("content", ""))
            
            # Extract keywords for semantic indexing
            words = content.lower().split()
            keywords = [word for word in words if len(word) > 3]
            
            for keyword in keywords[:10]:  # Limit to top 10 keywords
                if keyword not in semantic_index:
                    semantic_index[keyword] = []
                
                semantic_index[keyword].append({
                    "memory_id": memory_id,
                    "relevance_score": 1.0,  # Simple scoring
                    "content_preview": content[:100]
                })
                
        except Exception as e:
            logger.error(f"Error in semantic indexing: {e}")
    
    async def query(self, semantic_index: Dict[str, Any], query_text: str, 
                   similarity_threshold: float, limit: int) -> List[Dict[str, Any]]:
        """Query semantic index"""
        try:
            query_words = query_text.lower().split()
            results = []
            
            for word in query_words:
                if word in semantic_index:
                    results.extend(semantic_index[word])
            
            # Remove duplicates and limit
            seen_ids = set()
            unique_results = []
            for result in results:
                memory_id = result.get("memory_id")
                if memory_id not in seen_ids:
                    seen_ids.add(memory_id)
                    unique_results.append(result)
            
            return unique_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in semantic query: {e}")
            return []


class CausalIndexer:
    """Indexer for causal dimension"""
    
    async def initialize(self):
        """Initialize causal indexer"""
        pass
    
    async def index_memory(self, memory: Dict[str, Any], causal_index: Dict[str, Any]):
        """Index memory by causal relationships"""
        try:
            memory_id = memory.get("memory_id")
            content = str(memory.get("content", "")).lower()
            
            # Simple causal pattern detection
            causal_indicators = ["caused", "led to", "resulted in", "because", "due to"]
            
            for indicator in causal_indicators:
                if indicator in content:
                    if indicator not in causal_index:
                        causal_index[indicator] = []
                    
                    causal_index[indicator].append({
                        "memory_id": memory_id,
                        "causal_indicator": indicator,
                        "content_context": content[max(0, content.find(indicator)-50):content.find(indicator)+50]
                    })
                    
        except Exception as e:
            logger.error(f"Error in causal indexing: {e}")
    
    async def query(self, causal_index: Dict[str, Any], cause_pattern: str, 
                   effect_pattern: str, limit: int) -> List[Dict[str, Any]]:
        """Query causal index"""
        try:
            results = []
            
            # Search for causal relationships
            for indicator, entries in causal_index.items():
                for entry in entries:
                    context = entry.get("content_context", "")
                    if cause_pattern.lower() in context or effect_pattern.lower() in context:
                        results.append(entry)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in causal query: {e}")
            return []


class StrategicIndexer:
    """Indexer for strategic dimension"""
    
    async def initialize(self):
        """Initialize strategic indexer"""
        pass
    
    async def index_memory(self, memory: Dict[str, Any], strategic_index: Dict[str, Any]):
        """Index memory by strategic value"""
        try:
            memory_id = memory.get("memory_id")
            memory_type = memory.get("memory_type")
            strategic_value = memory.get("strategic_value", 0.5)
            
            # Group by memory type and strategic value
            type_key = str(memory_type)
            if type_key not in strategic_index:
                strategic_index[type_key] = []
            
            strategic_index[type_key].append({
                "memory_id": memory_id,
                "strategic_value": strategic_value,
                "memory_type": memory_type
            })
            
        except Exception as e:
            logger.error(f"Error in strategic indexing: {e}")
    
    async def query(self, strategic_index: Dict[str, Any], strategic_value_threshold: float,
                   memory_types: Optional[List[MemoryType]], limit: int) -> List[Dict[str, Any]]:
        """Query strategic index"""
        try:
            results = []
            
            target_types = [str(mt) for mt in memory_types] if memory_types else list(strategic_index.keys())
            
            for type_key in target_types:
                if type_key in strategic_index:
                    for entry in strategic_index[type_key]:
                        if entry.get("strategic_value", 0) >= strategic_value_threshold:
                            results.append(entry)
            
            # Sort by strategic value
            results.sort(key=lambda x: x.get("strategic_value", 0), reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in strategic query: {e}")
            return []


class PatternIndexer:
    """Indexer for pattern dimension"""
    
    async def initialize(self):
        """Initialize pattern indexer"""
        pass
    
    async def index_memory(self, memory: Dict[str, Any], pattern_index: Dict[str, Any]):
        """Index memory by patterns"""
        try:
            memory_id = memory.get("memory_id")
            patterns = memory.get("patterns", [])
            
            for pattern in patterns:
                pattern_type = pattern.get("pattern_type", "unknown")
                
                if pattern_type not in pattern_index:
                    pattern_index[pattern_type] = []
                
                pattern_index[pattern_type].append({
                    "memory_id": memory_id,
                    "pattern": pattern,
                    "frequency": pattern.get("frequency", 1)
                })
                
        except Exception as e:
            logger.error(f"Error in pattern indexing: {e}")
    
    async def query(self, pattern_index: Dict[str, Any], pattern_type: PatternType,
                   frequency_threshold: int, limit: int) -> List[Dict[str, Any]]:
        """Query pattern index"""
        try:
            pattern_key = str(pattern_type)
            results = []
            
            if pattern_key in pattern_index:
                for entry in pattern_index[pattern_key]:
                    if entry.get("frequency", 0) >= frequency_threshold:
                        results.append(entry)
            
            # Sort by frequency
            results.sort(key=lambda x: x.get("frequency", 0), reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error in pattern query: {e}")
            return []