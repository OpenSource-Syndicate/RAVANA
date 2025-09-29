"""
Very Long-Term Memory Storage Backend

This module implements the storage backend for the Snake Agent's Very Long-Term Memory System,
providing persistent storage using PostgreSQL, file system storage for compressed memories,
and vector store integration for semantic search.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import gzip

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel, select, and_, or_, desc, asc
import chromadb
from chromadb.config import Settings

from core.vltm_data_models import (
    VeryLongTermMemory, MemoryPattern, MemoryConsolidation, StrategicKnowledge,
    ConsolidationMetrics, MemoryType, PatternType, ConsolidationType,
    MemoryRecord, PatternRecord, VLTMConfiguration
)
from core.config import Config

logger = logging.getLogger(__name__)


class StorageBackend:
    """
    Storage backend for very long-term memory system.

    Provides unified interface for PostgreSQL database operations,
    file system storage for compressed memories, and vector operations.
    """

    def __init__(self, config: VLTMConfiguration, base_storage_dir: str = "vltm_storage"):
        """
        Initialize the storage backend.

        Args:
            config: VLTM configuration
            base_storage_dir: Base directory for file storage
        """
        self.config = config
        self.base_storage_dir = Path(base_storage_dir)

        # Storage directories
        self.compressed_storage_dir = self.base_storage_dir / "compressed"
        self.archive_storage_dir = self.base_storage_dir / "archives"
        self.temp_storage_dir = self.base_storage_dir / "temp"

        # Create directories
        for directory in [self.compressed_storage_dir, self.archive_storage_dir, self.temp_storage_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Database connections
        self.engine = None
        self.async_engine = None
        self.async_session_maker = None

        # Vector store
        self.chroma_client = None
        self.memory_collection = None
        self.pattern_collection = None
        self.strategic_collection = None

        # Performance metrics
        self.operation_count = 0
        self.total_storage_size = 0
        self.last_cleanup_time = datetime.utcnow()

    async def initialize(self) -> bool:
        """Initialize all storage components"""
        try:
            logger.info("Initializing VLTM storage backend...")

            # Initialize database connections
            if not await self._initialize_database():
                logger.error("Failed to initialize database")
                return False

            # Initialize vector store
            if not await self._initialize_vector_store():
                logger.error("Failed to initialize vector store")
                return False

            # Run initial maintenance
            await self._initial_maintenance()

            logger.info("VLTM storage backend initialized successfully")
            return True

        except Exception as e:
            logger.error(
                f"Error initializing storage backend: {e}", exc_info=True)
            return False

    async def _initialize_database(self) -> bool:
        """Initialize database connections (PostgreSQL or SQLite)"""
        try:
            # Get database configuration
            db_config = Config()
            database_url = getattr(
                db_config, 'DATABASE_URL', 'sqlite:///ravana_agi.db')

            # Handle async engine based on database type
            if database_url.startswith('postgresql://'):
                async_database_url = database_url.replace(
                    'postgresql://', 'postgresql+asyncpg://')
            elif database_url.startswith('sqlite://'):
                # Use regular engine for SQLite since async SQLite support is limited
                async_database_url = database_url
            else:
                async_database_url = database_url

            # Create engines
            self.engine = create_engine(database_url, echo=False)

            # For SQLite, we use the regular engine in async context
            if database_url.startswith('sqlite://'):
                from sqlalchemy.ext.asyncio import create_async_engine
                # Store the sync sessionmaker separately for SQLite
                from sqlalchemy.orm import sessionmaker as sync_sessionmaker
                self.async_engine = None  # We'll handle SQLite differently
                self._sync_sessionmaker = sync_sessionmaker(
                    self.engine, expire_on_commit=False)
            else:
                self.async_engine = create_async_engine(
                    async_database_url, echo=False)
                self.async_session_maker = async_sessionmaker(
                    self.async_engine, class_=AsyncSession, expire_on_commit=False
                )

            # Handle table creation based on database type
            if database_url.startswith('sqlite://'):
                # For SQLite, use sync approach within async context
                from sqlalchemy import create_engine as sync_create_engine
                temp_engine = sync_create_engine(database_url)
                SQLModel.metadata.create_all(temp_engine)
                temp_engine.dispose()
            else:
                # For PostgreSQL, use the async approach
                async with self.async_engine.begin() as conn:
                    await conn.run_sync(SQLModel.metadata.create_all)

            # Test connection
            if database_url.startswith('sqlite://'):
                # For SQLite test connection
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
            else:
                async with self.async_session_maker() as session:
                    await session.execute(text("SELECT 1"))

            logger.info("Database connections initialized")
            return True

        except Exception as e:
            logger.error(f"Database initialization failed: {e}", exc_info=True)
            return False

    async def _initialize_vector_store(self) -> bool:
        """Initialize ChromaDB vector store"""
        try:
            # Initialize ChromaDB client
            chroma_settings = Settings(
                persist_directory=str(self.base_storage_dir / "chromadb"),
                anonymized_telemetry=False
            )
            self.chroma_client = chromadb.Client(chroma_settings)

            # Create collections for different types of vectors
            self.memory_collection = self.chroma_client.get_or_create_collection(
                name="vltm_memories",
                metadata={"description": "Very long-term memory embeddings"}
            )

            self.pattern_collection = self.chroma_client.get_or_create_collection(
                name="vltm_patterns",
                metadata={"description": "Memory pattern embeddings"}
            )

            self.strategic_collection = self.chroma_client.get_or_create_collection(
                name="vltm_strategic",
                metadata={"description": "Strategic knowledge embeddings"}
            )

            logger.info("Vector store initialized")
            return True

        except Exception as e:
            logger.error(f"Vector store initialization failed: {e}")
            return False

    async def _initial_maintenance(self):
        """Run initial maintenance tasks"""
        try:
            # Calculate storage size
            await self._calculate_storage_size()

            # Clean up temporary files
            await self._cleanup_temp_files()

            logger.info("Initial maintenance completed")

        except Exception as e:
            logger.warning(f"Initial maintenance failed: {e}")

    # Memory Storage Operations

    async def store_memory(self, memory_record: MemoryRecord) -> bool:
        """Store a memory record in the database and file system"""
        try:
            # Handle database type appropriately
            if self.async_engine is None:  # SQLite case
                # Use synchronous session for SQLite
                with self._sync_sessionmaker() as session:
                    # Create database record
                    db_memory = VeryLongTermMemory(
                        memory_id=memory_record.memory_id,
                        memory_type=memory_record.memory_type,
                        compressed_content=json.dumps(memory_record.content),
                        metadata=json.dumps(memory_record.metadata),
                        importance_score=memory_record.importance_score,
                        strategic_value=memory_record.strategic_value,
                        source_session=memory_record.source_session,
                        related_memories=json.dumps(
                            memory_record.related_memories)
                    )

                    session.add(db_memory)
                    session.commit()
            else:  # PostgreSQL case
                async with self.async_session_maker() as session:
                    # Create database record
                    db_memory = VeryLongTermMemory(
                        memory_id=memory_record.memory_id,
                        memory_type=memory_record.memory_type,
                        compressed_content=json.dumps(memory_record.content),
                        metadata=json.dumps(memory_record.metadata),
                        importance_score=memory_record.importance_score,
                        strategic_value=memory_record.strategic_value,
                        source_session=memory_record.source_session,
                        related_memories=json.dumps(
                            memory_record.related_memories)
                    )

                    session.add(db_memory)
                    await session.commit()

            # Store compressed content to file system if large
            content_size = len(json.dumps(memory_record.content))
            if content_size > 10000:  # Store large content separately
                await self._store_compressed_content(memory_record.memory_id, memory_record.content)

            # Add to vector store
            await self._add_memory_to_vector_store(memory_record)

            self.operation_count += 1
            logger.debug(f"Stored memory: {memory_record.memory_id}")
            return True

        except Exception as e:
            logger.error(
                f"Error storing memory {memory_record.memory_id}: {e}")
            return False

    async def retrieve_memory(self, memory_id: str) -> Optional[VeryLongTermMemory]:
        """Retrieve a memory record by ID"""
        try:
            # Handle database type appropriately
            if self.async_engine is None:  # SQLite case
                with self._sync_sessionmaker() as session:
                    statement = select(VeryLongTermMemory).where(
                        VeryLongTermMemory.memory_id == memory_id)
                    result = session.execute(statement)
                    memory = result.scalar_one_or_none()

                    if memory:
                        # Update access count and timestamp
                        memory.access_count += 1
                        memory.last_accessed = datetime.utcnow()
                        session.commit()

                    return memory
            else:  # PostgreSQL case
                async with self.async_session_maker() as session:
                    statement = select(VeryLongTermMemory).where(
                        VeryLongTermMemory.memory_id == memory_id)
                    result = await session.execute(statement)
                    memory = result.scalar_one_or_none()

                    if memory:
                        # Update access count and timestamp
                        memory.access_count += 1
                        memory.last_accessed = datetime.utcnow()
                        await session.commit()

                    return memory

        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None

    async def retrieve_memories_by_type(
        self,
        memory_type: MemoryType,
        limit: int = 100,
        min_importance: float = 0.0
    ) -> List[VeryLongTermMemory]:
        """Retrieve memories by type with optional filtering"""
        try:
            # Handle database type appropriately
            if self.async_engine is None:  # SQLite case
                with self._sync_sessionmaker() as session:
                    statement = (
                        select(VeryLongTermMemory)
                        .where(
                            and_(
                                VeryLongTermMemory.memory_type == memory_type,
                                VeryLongTermMemory.importance_score >= min_importance
                            )
                        )
                        .order_by(desc(VeryLongTermMemory.importance_score))
                        .limit(limit)
                    )

                    result = session.execute(statement)
                    return list(result.scalars().all())
            else:  # PostgreSQL case
                async with self.async_session_maker() as session:
                    statement = (
                        select(VeryLongTermMemory)
                        .where(
                            and_(
                                VeryLongTermMemory.memory_type == memory_type,
                                VeryLongTermMemory.importance_score >= min_importance
                            )
                        )
                        .order_by(desc(VeryLongTermMemory.importance_score))
                        .limit(limit)
                    )

                    result = await session.execute(statement)
                    return list(result.scalars().all())

        except Exception as e:
            logger.error(
                f"Error retrieving memories by type {memory_type}: {e}")
            return []

    async def retrieve_recent_memories(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> List[VeryLongTermMemory]:
        """Retrieve recent memories"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            # Handle database type appropriately
            if self.async_engine is None:  # SQLite case
                with self._sync_sessionmaker() as session:
                    statement = (
                        select(VeryLongTermMemory)
                        .where(VeryLongTermMemory.created_at >= cutoff_time)
                        .order_by(desc(VeryLongTermMemory.created_at))
                        .limit(limit)
                    )

                    result = session.execute(statement)
                    return list(result.scalars().all())
            else:  # PostgreSQL case
                async with self.async_session_maker() as session:
                    statement = (
                        select(VeryLongTermMemory)
                        .where(VeryLongTermMemory.created_at >= cutoff_time)
                        .order_by(desc(VeryLongTermMemory.created_at))
                        .limit(limit)
                    )

                    result = await session.execute(statement)
                    return list(result.scalars().all())

        except Exception as e:
            logger.error(f"Error retrieving recent memories: {e}")
            return []

    # Pattern Storage Operations

    async def store_pattern(self, pattern_record: PatternRecord) -> bool:
        """Store a pattern record"""
        try:
            # Handle database type appropriately
            if self.async_engine is None:  # SQLite case
                with self._sync_sessionmaker() as session:
                    db_pattern = MemoryPattern(
                        pattern_id=pattern_record.pattern_id,
                        pattern_type=pattern_record.pattern_type,
                        pattern_description=pattern_record.description,
                        confidence_score=pattern_record.confidence_score,
                        pattern_data=json.dumps(pattern_record.pattern_data),
                        supporting_memories=json.dumps(
                            pattern_record.supporting_memories)
                    )

                    session.add(db_pattern)
                    session.commit()
            else:  # PostgreSQL case
                async with self.async_session_maker() as session:
                    db_pattern = MemoryPattern(
                        pattern_id=pattern_record.pattern_id,
                        pattern_type=pattern_record.pattern_type,
                        pattern_description=pattern_record.description,
                        confidence_score=pattern_record.confidence_score,
                        pattern_data=json.dumps(pattern_record.pattern_data),
                        supporting_memories=json.dumps(
                            pattern_record.supporting_memories)
                    )

                    session.add(db_pattern)
                    await session.commit()

            # Add to vector store
            await self._add_pattern_to_vector_store(pattern_record)

            logger.debug(f"Stored pattern: {pattern_record.pattern_id}")
            return True

        except Exception as e:
            logger.error(
                f"Error storing pattern {pattern_record.pattern_id}: {e}")
            return False

    async def retrieve_patterns_by_type(
        self,
        pattern_type: PatternType,
        min_confidence: float = 0.5,
        limit: int = 50
    ) -> List[MemoryPattern]:
        """Retrieve patterns by type"""
        try:
            # Handle database type appropriately
            if self.async_engine is None:  # SQLite case
                with self._sync_sessionmaker() as session:
                    statement = (
                        select(MemoryPattern)
                        .where(
                            and_(
                                MemoryPattern.pattern_type == pattern_type,
                                MemoryPattern.confidence_score >= min_confidence
                            )
                        )
                        .order_by(desc(MemoryPattern.confidence_score))
                        .limit(limit)
                    )

                    result = session.execute(statement)
                    return list(result.scalars().all())
            else:  # PostgreSQL case
                async with self.async_session_maker() as session:
                    statement = (
                        select(MemoryPattern)
                        .where(
                            and_(
                                MemoryPattern.pattern_type == pattern_type,
                                MemoryPattern.confidence_score >= min_confidence
                            )
                        )
                        .order_by(desc(MemoryPattern.confidence_score))
                        .limit(limit)
                    )

                    result = await session.execute(statement)
                    return list(result.scalars().all())

        except Exception as e:
            logger.error(
                f"Error retrieving patterns by type {pattern_type}: {e}")
            return []

    # Strategic Knowledge Operations

    async def store_strategic_knowledge(
        self,
        knowledge_id: str,
        domain: str,
        summary: str,
        confidence: float,
        knowledge_structure: Dict[str, Any],
        source_patterns: List[str] = None
    ) -> bool:
        """Store strategic knowledge"""
        try:
            # Handle database type appropriately
            if self.async_engine is None:  # SQLite case
                with self._sync_sessionmaker() as session:
                    strategic = StrategicKnowledge(
                        knowledge_id=knowledge_id,
                        knowledge_domain=domain,
                        knowledge_summary=summary,
                        confidence_level=confidence,
                        knowledge_structure=json.dumps(knowledge_structure),
                        source_patterns=json.dumps(source_patterns or [])
                    )

                    session.add(strategic)
                    session.commit()
            else:  # PostgreSQL case
                async with self.async_session_maker() as session:
                    strategic = StrategicKnowledge(
                        knowledge_id=knowledge_id,
                        knowledge_domain=domain,
                        knowledge_summary=summary,
                        confidence_level=confidence,
                        knowledge_structure=json.dumps(knowledge_structure),
                        source_patterns=json.dumps(source_patterns or [])
                    )

                    session.add(strategic)
                    await session.commit()

            # Add to vector store
            await self._add_strategic_to_vector_store(strategic)

            logger.debug(f"Stored strategic knowledge: {knowledge_id}")
            return True

        except Exception as e:
            logger.error(
                f"Error storing strategic knowledge {knowledge_id}: {e}")
            return False

    async def retrieve_strategic_knowledge_by_domain(
        self,
        domain: str,
        min_confidence: float = 0.5,
        limit: int = 20
    ) -> List[StrategicKnowledge]:
        """Retrieve strategic knowledge by domain"""
        try:
            # Handle database type appropriately
            if self.async_engine is None:  # SQLite case
                with self._sync_sessionmaker() as session:
                    statement = (
                        select(StrategicKnowledge)
                        .where(
                            and_(
                                StrategicKnowledge.knowledge_domain == domain,
                                StrategicKnowledge.confidence_level >= min_confidence
                            )
                        )
                        .order_by(desc(StrategicKnowledge.confidence_level))
                        .limit(limit)
                    )

                    result = session.execute(statement)
                    return list(result.scalars().all())
            else:  # PostgreSQL case
                async with self.async_session_maker() as session:
                    statement = (
                        select(StrategicKnowledge)
                        .where(
                            and_(
                                StrategicKnowledge.knowledge_domain == domain,
                                StrategicKnowledge.confidence_level >= min_confidence
                            )
                        )
                        .order_by(desc(StrategicKnowledge.confidence_level))
                        .limit(limit)
                    )

                    result = await session.execute(statement)
                    return list(result.scalars().all())

        except Exception as e:
            logger.error(
                f"Error retrieving strategic knowledge for domain {domain}: {e}")
            return []

    # Consolidation Operations

    async def record_consolidation(
        self,
        consolidation_id: str,
        consolidation_type: ConsolidationType,
        memories_processed: int,
        patterns_extracted: int,
        compression_ratio: float,
        processing_time: float,
        success: bool = True,
        error_message: str = None,
        results: Dict[str, Any] = None
    ) -> bool:
        """Record a consolidation operation"""
        try:
            # Handle database type appropriately
            if self.async_engine is None:  # SQLite case
                with self._sync_sessionmaker() as session:
                    consolidation = MemoryConsolidation(
                        consolidation_id=consolidation_id,
                        consolidation_type=consolidation_type,
                        memories_processed=memories_processed,
                        patterns_extracted=patterns_extracted,
                        compression_ratio=compression_ratio,
                        processing_time_seconds=processing_time,
                        success=success,
                        error_message=error_message,
                        consolidation_results=json.dumps(results or {})
                    )

                    session.add(consolidation)
                    session.commit()
            else:  # PostgreSQL case
                async with self.async_session_maker() as session:
                    consolidation = MemoryConsolidation(
                        consolidation_id=consolidation_id,
                        consolidation_type=consolidation_type,
                        memories_processed=memories_processed,
                        patterns_extracted=patterns_extracted,
                        compression_ratio=compression_ratio,
                        processing_time_seconds=processing_time,
                        success=success,
                        error_message=error_message,
                        consolidation_results=json.dumps(results or {})
                    )

                    session.add(consolidation)
                    await session.commit()

                logger.debug(f"Recorded consolidation: {consolidation_id}")
                return True

        except Exception as e:
            logger.error(
                f"Error recording consolidation {consolidation_id}: {e}")
            return False

    # Vector Store Operations

    async def _add_memory_to_vector_store(self, memory_record: MemoryRecord):
        """Add memory to vector store for semantic search"""
        try:
            # Create text representation for embedding
            text_content = f"{memory_record.memory_type.value} {json.dumps(memory_record.content)}"

            # Add to collection (ChromaDB will generate embeddings)
            self.memory_collection.add(
                documents=[text_content],
                metadatas=[{
                    "memory_id": memory_record.memory_id,
                    "memory_type": memory_record.memory_type.value,
                    "importance_score": memory_record.importance_score,
                    "strategic_value": memory_record.strategic_value
                }],
                ids=[memory_record.memory_id]
            )

        except Exception as e:
            logger.warning(f"Error adding memory to vector store: {e}")

    async def _add_pattern_to_vector_store(self, pattern_record: PatternRecord):
        """Add pattern to vector store"""
        try:
            text_content = f"{pattern_record.pattern_type.value} {pattern_record.description}"

            self.pattern_collection.add(
                documents=[text_content],
                metadatas=[{
                    "pattern_id": pattern_record.pattern_id,
                    "pattern_type": pattern_record.pattern_type.value,
                    "confidence_score": pattern_record.confidence_score
                }],
                ids=[pattern_record.pattern_id]
            )

        except Exception as e:
            logger.warning(f"Error adding pattern to vector store: {e}")

    async def _add_strategic_to_vector_store(self, strategic: StrategicKnowledge):
        """Add strategic knowledge to vector store"""
        try:
            text_content = f"{strategic.knowledge_domain} {strategic.knowledge_summary}"

            self.strategic_collection.add(
                documents=[text_content],
                metadatas=[{
                    "knowledge_id": strategic.knowledge_id,
                    "knowledge_domain": strategic.knowledge_domain,
                    "confidence_level": strategic.confidence_level
                }],
                ids=[strategic.knowledge_id]
            )

        except Exception as e:
            logger.warning(
                f"Error adding strategic knowledge to vector store: {e}")

    async def semantic_search_memories(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform semantic search on memories"""
        try:
            results = self.memory_collection.query(
                query_texts=[query],
                n_results=limit
            )

            return [
                {
                    "memory_id": results['ids'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else 0,
                    "metadata": results['metadatas'][0][i]
                }
                for i in range(len(results['ids'][0]))
            ]

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    # File System Operations

    async def _store_compressed_content(self, memory_id: str, content: Dict[str, Any]):
        """Store large content in compressed file"""
        try:
            file_path = self.compressed_storage_dir / f"{memory_id}.json.gz"

            # Compress and save
            with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                json.dump(content, f, indent=2, default=str)

            logger.debug(f"Stored compressed content for memory: {memory_id}")

        except Exception as e:
            logger.error(f"Error storing compressed content: {e}")

    async def _load_compressed_content(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Load compressed content from file"""
        try:
            file_path = self.compressed_storage_dir / f"{memory_id}.json.gz"

            if not file_path.exists():
                return None

            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Error loading compressed content: {e}")
            return None

    # Maintenance Operations

    async def _calculate_storage_size(self):
        """Calculate total storage size"""
        try:
            total_size = 0
            for directory in [self.compressed_storage_dir, self.archive_storage_dir]:
                for file_path in directory.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size

            self.total_storage_size = total_size
            logger.debug(
                f"Total storage size: {total_size / (1024*1024):.2f} MB")

        except Exception as e:
            logger.warning(f"Error calculating storage size: {e}")

    async def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for file_path in self.temp_storage_dir.glob('*'):
                if file_path.is_file():
                    # Remove files older than 1 hour
                    file_age = datetime.utcnow().timestamp() - file_path.stat().st_mtime
                    if file_age > 3600:  # 1 hour
                        file_path.unlink()

        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")

    async def cleanup(self):
        """Clean up storage backend resources"""
        try:
            logger.info("Cleaning up VLTM storage backend...")

            if self.async_engine:
                await self.async_engine.dispose()

            if self.engine:
                self.engine.dispose()

            logger.info("VLTM storage backend cleanup completed")

        except Exception as e:
            logger.error(f"Error during storage backend cleanup: {e}")

    # Statistics and Monitoring

    async def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            # Handle database type appropriately
            if self.async_engine is None:  # SQLite case
                with self._sync_sessionmaker() as session:
                    # Count records
                    memory_count = session.execute(
                        text("SELECT COUNT(*) FROM very_long_term_memories"))
                    pattern_count = session.execute(
                        text("SELECT COUNT(*) FROM memory_patterns"))
                    strategic_count = session.execute(
                        text("SELECT COUNT(*) FROM strategic_knowledge"))
                    consolidation_count = session.execute(
                        text("SELECT COUNT(*) FROM memory_consolidations"))

                    # Calculate storage size
                    await self._calculate_storage_size()

                    return {
                        "memory_count": memory_count.scalar(),
                        "pattern_count": pattern_count.scalar(),
                        "strategic_knowledge_count": strategic_count.scalar(),
                        "consolidation_count": consolidation_count.scalar(),
                        "total_storage_size_mb": self.total_storage_size / (1024 * 1024),
                        "operation_count": self.operation_count,
                        "last_cleanup_time": self.last_cleanup_time.isoformat()
                    }
            else:  # PostgreSQL case
                async with self.async_session_maker() as session:
                    # Count records
                    memory_count = await session.execute(text("SELECT COUNT(*) FROM very_long_term_memories"))
                    pattern_count = await session.execute(text("SELECT COUNT(*) FROM memory_patterns"))
                    strategic_count = await session.execute(text("SELECT COUNT(*) FROM strategic_knowledge"))
                    consolidation_count = await session.execute(text("SELECT COUNT(*) FROM memory_consolidations"))

                    # Calculate storage size
                    await self._calculate_storage_size()

                    return {
                        "memory_count": memory_count.scalar(),
                        "pattern_count": pattern_count.scalar(),
                        "strategic_knowledge_count": strategic_count.scalar(),
                        "consolidation_count": consolidation_count.scalar(),
                        "total_storage_size_mb": self.total_storage_size / (1024 * 1024),
                        "operation_count": self.operation_count,
                        "last_cleanup_time": self.last_cleanup_time.isoformat()
                    }

        except Exception as e:
            logger.error(f"Error getting storage statistics: {e}")
            return {}
