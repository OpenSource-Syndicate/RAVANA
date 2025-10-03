"""
Very Long-Term Memory Store

This module implements the main VeryLongTermMemoryStore class that serves as the
primary interface for the Snake Agent's Very Long-Term Memory System. It integrates
storage backend, memory classification, and provides high-level memory operations.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from core.vltm_data_models import (
    VeryLongTermMemory, MemoryPattern, StrategicKnowledge,
    MemoryRecord, PatternRecord, MemoryType, PatternType,
    VLTMConfiguration, DEFAULT_VLTM_CONFIG, ConsolidationRequest,
    ConsolidationResult, StrategicQuery, MemoryImportanceLevel
)
from core.vltm_storage_backend import StorageBackend
from core.vltm_memory_classifier import MemoryClassifier, ImportanceEvaluator, ClassificationFeatures

logger = logging.getLogger(__name__)


class VeryLongTermMemoryStore:
    """
    Main interface for very long-term memory operations.

    Provides high-level methods for storing, retrieving, and managing memories
    while coordinating between storage backend and classification components.
    """

    def __init__(self, config: Optional[VLTMConfiguration] = None, base_storage_dir: str = "vltm_storage"):
        """
        Initialize the very long-term memory store.

        Args:
            config: VLTM configuration, uses default if None
            base_storage_dir: Base directory for storage
        """
        self.config = config or DEFAULT_VLTM_CONFIG
        self.base_storage_dir = base_storage_dir

        # Core components
        self.storage_backend: Optional[StorageBackend] = None
        self.memory_classifier: Optional[MemoryClassifier] = None
        self.importance_evaluator: Optional[ImportanceEvaluator] = None

        # State tracking
        self.initialized = False
        self.total_memories_stored = 0
        self.total_patterns_extracted = 0
        self.total_strategic_knowledge = 0
        self.last_consolidation_time: Optional[datetime] = None

        # Performance metrics
        self.operation_metrics = {
            "store_operations": 0,
            "retrieve_operations": 0,
            "search_operations": 0,
            "classification_operations": 0,
            "total_processing_time": 0.0
        }

        logger.info("VeryLongTermMemoryStore initialized")

    async def initialize(self) -> bool:
        """Initialize all components of the memory store"""
        try:
            logger.info("Initializing Very Long-Term Memory Store...")

            # Validate configuration
            config_issues = self.config.validate_configuration()
            if config_issues:
                logger.error(
                    f"Configuration validation failed: {config_issues}")
                return False

            # Initialize storage backend
            self.storage_backend = StorageBackend(
                self.config, self.base_storage_dir)
            if not await self.storage_backend.initialize():
                logger.error("Failed to initialize storage backend")
                return False

            # Initialize memory classifier
            self.memory_classifier = MemoryClassifier(self.config)

            # Initialize importance evaluator
            self.importance_evaluator = ImportanceEvaluator()

            # Load existing statistics
            await self._load_statistics()

            self.initialized = True
            logger.info("Very Long-Term Memory Store initialized successfully")
            return True

        except Exception as e:
            logger.error(
                f"Failed to initialize VLTM store: {e}", exc_info=True)
            return False

    async def store_memory(
        self,
        content: Dict[str, Any],
        memory_type: Optional[MemoryType] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_session: str = "unknown",
        related_memories: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Store a new memory in the very long-term memory system.

        Args:
            content: Memory content to store
            memory_type: Type of memory (will be classified if None)
            metadata: Additional metadata
            source_session: Source session identifier
            related_memories: List of related memory IDs

        Returns:
            Memory ID if successful, None if failed
        """
        if not self.initialized:
            logger.error("VLTM store not initialized")
            return None

        start_time = datetime.utcnow()

        try:
            # Generate memory ID
            memory_id = str(uuid.uuid4())

            # Create memory record
            memory_record = MemoryRecord(
                memory_id=memory_id,
                memory_type=memory_type or MemoryType.CODE_PATTERN,  # Default, will be classified
                content=content,
                metadata=metadata or {},
                source_session=source_session,
                related_memories=related_memories or []
            )

            # Classify memory if type not provided
            if memory_type is None:
                classification_result = self.memory_classifier.classify_memory(
                    memory_record)
                memory_record.memory_type = classification_result["memory_type"]
                memory_record.importance_score = classification_result["importance_score"]
                memory_record.strategic_value = classification_result["strategic_value"]

                # Store classification metadata
                memory_record.metadata.update({
                    "classification": classification_result,
                    "classified_at": datetime.utcnow().isoformat()
                })

                self.operation_metrics["classification_operations"] += 1

            # Store memory
            success = await self.storage_backend.store_memory(memory_record)

            if success:
                self.total_memories_stored += 1
                self.operation_metrics["store_operations"] += 1

                # Log successful storage
                logger.info(f"Stored memory {memory_id} (type: {memory_record.memory_type}, "
                            f"importance: {memory_record.importance_score:.3f})")

                # Update processing time
                processing_time = (datetime.utcnow() -
                                   start_time).total_seconds()
                self.operation_metrics["total_processing_time"] += processing_time

                return memory_id
            else:
                logger.error(f"Failed to store memory {memory_id}")
                return None

        except Exception as e:
            logger.error(f"Error storing memory: {e}", exc_info=True)
            return None

    async def retrieve_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: Memory ID to retrieve

        Returns:
            Memory data including content and metadata, None if not found
        """
        if not self.initialized:
            logger.error("VLTM store not initialized")
            return None

        start_time = datetime.utcnow()

        try:
            memory = await self.storage_backend.retrieve_memory(memory_id)

            if memory:
                self.operation_metrics["retrieve_operations"] += 1

                # Parse content and metadata
                content = json.loads(memory.compressed_content)
                metadata = json.loads(memory.metadata)
                related_memories = json.loads(memory.related_memories)

                result = {
                    "memory_id": memory.memory_id,
                    "memory_type": memory.memory_type,
                    "content": content,
                    "metadata": metadata,
                    "created_at": memory.created_at.isoformat(),
                    "last_accessed": memory.last_accessed.isoformat(),
                    "access_count": memory.access_count,
                    "importance_score": memory.importance_score,
                    "strategic_value": memory.strategic_value,
                    "source_session": memory.source_session,
                    "related_memories": related_memories
                }

                # Update processing time
                processing_time = (datetime.utcnow() -
                                   start_time).total_seconds()
                self.operation_metrics["total_processing_time"] += processing_time

                return result
            else:
                logger.debug(f"Memory {memory_id} not found")
                return None

        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None

    async def search_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        min_importance: float = 0.0,
        min_strategic_value: float = 0.0,
        limit: int = 20,
        use_semantic_search: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search memories using text query and filters.

        Args:
            query: Search query text
            memory_types: Filter by memory types
            min_importance: Minimum importance score
            min_strategic_value: Minimum strategic value
            limit: Maximum number of results
            use_semantic_search: Whether to use semantic search

        Returns:
            List of matching memories with scores
        """
        if not self.initialized:
            logger.error("VLTM store not initialized")
            return []

        start_time = datetime.utcnow()

        try:
            results = []

            if use_semantic_search:
                # Perform semantic search
                semantic_results = await self.storage_backend.semantic_search_memories(query, limit * 2)

                # Get full memory data for semantic results
                for semantic_result in semantic_results:
                    memory_id = semantic_result["memory_id"]
                    memory_data = await self.retrieve_memory(memory_id)

                    if memory_data:
                        # Apply filters
                        if memory_types and memory_data["memory_type"] not in memory_types:
                            continue
                        if memory_data["importance_score"] < min_importance:
                            continue
                        if memory_data["strategic_value"] < min_strategic_value:
                            continue

                        # Add search metadata
                        memory_data["search_score"] = 1.0 - \
                            semantic_result["distance"]
                        memory_data["search_metadata"] = semantic_result["metadata"]

                        results.append(memory_data)

                        if len(results) >= limit:
                            break

            # If semantic search didn't find enough results, try database search
            if len(results) < limit:
                remaining_limit = limit - len(results)

                if memory_types:
                    for memory_type in memory_types:
                        db_results = await self.storage_backend.retrieve_memories_by_type(
                            memory_type, remaining_limit, min_importance
                        )

                        for db_memory in db_results:
                            if db_memory.strategic_value >= min_strategic_value:
                                memory_data = await self._convert_db_memory_to_dict(db_memory)
                                # Default score for DB results
                                memory_data["search_score"] = 0.5
                                results.append(memory_data)

                                if len(results) >= limit:
                                    break

                        if len(results) >= limit:
                            break

            self.operation_metrics["search_operations"] += 1

            # Update processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.operation_metrics["total_processing_time"] += processing_time

            # Sort by search score
            results.sort(key=lambda x: x.get("search_score", 0), reverse=True)

            logger.debug(f"Found {len(results)} memories for query: {query}")
            return results[:limit]

        except Exception as e:
            logger.error(f"Error searching memories: {e}", exc_info=True)
            return []

    async def store_pattern(
        self,
        pattern_type: PatternType,
        description: str,
        pattern_data: Dict[str, Any],
        confidence_score: float,
        supporting_memories: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Store a new pattern extracted from memories.

        Args:
            pattern_type: Type of pattern
            description: Pattern description
            pattern_data: Pattern-specific data
            confidence_score: Confidence in pattern validity
            supporting_memories: List of supporting memory IDs

        Returns:
            Pattern ID if successful, None if failed
        """
        if not self.initialized:
            logger.error("VLTM store not initialized")
            return None

        try:
            pattern_id = str(uuid.uuid4())

            pattern_record = PatternRecord(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                description=description,
                pattern_data=pattern_data,
                confidence_score=confidence_score,
                supporting_memories=supporting_memories or []
            )

            success = await self.storage_backend.store_pattern(pattern_record)

            if success:
                self.total_patterns_extracted += 1
                logger.info(f"Stored pattern {pattern_id} (type: {pattern_type}, "
                            f"confidence: {confidence_score:.3f})")
                return pattern_id
            else:
                logger.error(f"Failed to store pattern {pattern_id}")
                return None

        except Exception as e:
            logger.error(f"Error storing pattern: {e}", exc_info=True)
            return None

    async def store_strategic_knowledge(
        self,
        domain: str,
        summary: str,
        knowledge_structure: Dict[str, Any],
        confidence_level: float,
        source_patterns: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Store strategic knowledge derived from patterns.

        Args:
            domain: Knowledge domain (e.g., "architecture", "performance")
            summary: Knowledge summary
            knowledge_structure: Structured knowledge data
            confidence_level: Confidence in knowledge validity
            source_patterns: List of source pattern IDs

        Returns:
            Knowledge ID if successful, None if failed
        """
        if not self.initialized:
            logger.error("VLTM store not initialized")
            return None

        try:
            knowledge_id = str(uuid.uuid4())

            success = await self.storage_backend.store_strategic_knowledge(
                knowledge_id=knowledge_id,
                domain=domain,
                summary=summary,
                confidence=confidence_level,
                knowledge_structure=knowledge_structure,
                source_patterns=source_patterns
            )

            if success:
                self.total_strategic_knowledge += 1
                logger.info(f"Stored strategic knowledge {knowledge_id} "
                            f"(domain: {domain}, confidence: {confidence_level:.3f})")
                return knowledge_id
            else:
                logger.error(
                    f"Failed to store strategic knowledge {knowledge_id}")
                return None

        except Exception as e:
            logger.error(
                f"Error storing strategic knowledge: {e}", exc_info=True)
            return None

    async def query_strategic_knowledge(self, query: StrategicQuery) -> List[Dict[str, Any]]:
        """
        Query strategic knowledge using structured query.

        Args:
            query: Strategic knowledge query

        Returns:
            List of matching strategic knowledge entries
        """
        if not self.initialized:
            logger.error("VLTM store not initialized")
            return []

        try:
            results = []

            # Query by domains if specified
            if query.knowledge_domains:
                for domain in query.knowledge_domains:
                    domain_results = await self.storage_backend.retrieve_strategic_knowledge_by_domain(
                        domain, query.min_confidence, query.max_results
                    )

                    for strategic in domain_results:
                        knowledge_data = await self._convert_strategic_knowledge_to_dict(strategic)
                        results.append(knowledge_data)
            else:
                # If no specific domains, try semantic search on strategic knowledge
                # This would be implemented with vector search on strategic knowledge
                pass

            # Sort by confidence level
            results.sort(key=lambda x: x.get(
                "confidence_level", 0), reverse=True)

            return results[:query.max_results]

        except Exception as e:
            logger.error(f"Error querying strategic knowledge: {e}")
            return []

    async def get_memories_by_type(self, memory_type: 'MemoryType', limit: int = 100, importance_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Get memories by type, with optional limit and importance filtering."""
        if not self.initialized:
            logger.error("VLTM store not initialized")
            return []

        try:
            # Use storage backend to retrieve memories by type
            db_memories = await self.storage_backend.retrieve_memories_by_type(
                memory_type, limit, importance_threshold
            )

            # Convert DB memories to dictionary format
            results = []
            for db_memory in db_memories:
                memory_data = await self._convert_db_memory_to_dict(db_memory)
                results.append(memory_data)

            logger.info(f"Retrieved {len(results)} memories of type {memory_type}")
            return results

        except Exception as e:
            logger.error(f"Error retrieving memories by type {memory_type}: {e}", exc_info=True)
            return []

    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        if not self.initialized:
            return {"error": "not_initialized"}

        try:
            # Get storage statistics
            storage_stats = await self.storage_backend.get_storage_statistics()

            # Get classification statistics
            classification_stats = self.memory_classifier.get_classification_statistics()

            # Combine with local statistics
            return {
                "storage": storage_stats,
                "classification": classification_stats,
                "operations": self.operation_metrics.copy(),
                "totals": {
                    "memories_stored": self.total_memories_stored,
                    "patterns_extracted": self.total_patterns_extracted,
                    "strategic_knowledge": self.total_strategic_knowledge
                },
                "last_consolidation_time": self.last_consolidation_time.isoformat() if self.last_consolidation_time else None,
                "initialized": self.initialized,
                "config_summary": {
                    "retention_policies": len(self.config.retention_policies),
                    "consolidation_schedule": self.config.consolidation_schedule
                }
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    async def _convert_db_memory_to_dict(self, db_memory: VeryLongTermMemory) -> Dict[str, Any]:
        """Convert database memory object to dictionary"""
        return {
            "memory_id": db_memory.memory_id,
            "memory_type": db_memory.memory_type,
            "content": json.loads(db_memory.compressed_content),
            "metadata": json.loads(db_memory.metadata),
            "created_at": db_memory.created_at.isoformat(),
            "last_accessed": db_memory.last_accessed.isoformat(),
            "access_count": db_memory.access_count,
            "importance_score": db_memory.importance_score,
            "strategic_value": db_memory.strategic_value,
            "source_session": db_memory.source_session,
            "related_memories": json.loads(db_memory.related_memories)
        }

    async def _convert_strategic_knowledge_to_dict(self, strategic: StrategicKnowledge) -> Dict[str, Any]:
        """Convert strategic knowledge object to dictionary"""
        return {
            "knowledge_id": strategic.knowledge_id,
            "knowledge_domain": strategic.knowledge_domain,
            "knowledge_summary": strategic.knowledge_summary,
            "confidence_level": strategic.confidence_level,
            "last_updated": strategic.last_updated.isoformat(),
            "source_patterns": json.loads(strategic.source_patterns),
            "knowledge_structure": json.loads(strategic.knowledge_structure),
            "validation_score": strategic.validation_score,
            "application_count": strategic.application_count
        }

    async def _load_statistics(self):
        """Load existing statistics from storage"""
        try:
            storage_stats = await self.storage_backend.get_storage_statistics()

            # Update local counters from storage
            self.total_memories_stored = storage_stats.get("memory_count", 0)
            self.total_patterns_extracted = storage_stats.get(
                "pattern_count", 0)
            self.total_strategic_knowledge = storage_stats.get(
                "strategic_knowledge_count", 0)

            logger.info(f"Loaded statistics: {self.total_memories_stored} memories, "
                        f"{self.total_patterns_extracted} patterns, "
                        f"{self.total_strategic_knowledge} strategic knowledge entries")

        except Exception as e:
            logger.warning(f"Could not load existing statistics: {e}")

    async def cleanup(self):
        """Clean up resources"""
        try:
            logger.info("Cleaning up VLTM store...")

            if self.storage_backend:
                await self.storage_backend.cleanup()

            self.initialized = False
            logger.info("VLTM store cleanup completed")

        except Exception as e:
            logger.error(f"Error during VLTM store cleanup: {e}")

    def is_initialized(self) -> bool:
        """Check if the store is initialized"""
        return self.initialized

    def get_config(self) -> VLTMConfiguration:
        """Get current configuration"""
        return self.config
