"""
PostgreSQLStore for multi-modal memory system.
Handles database operations with pgvector support for similarity search.
"""

import logging
import asyncio
import json
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
import uuid

try:
    import asyncpg
    import numpy as np
    ASYNCPG_AVAILABLE = True
except ImportError as e:
    ASYNCPG_AVAILABLE = False
    asyncpg = None
    np = None
    logging.warning(f"AsyncPG not available: {e}")

from .models import (
    MemoryRecord, ContentType, MemoryType, SearchRequest, 
    AudioMetadata, ImageMetadata, VideoMetadata, ConsolidationResult
)

logger = logging.getLogger(__name__)

class PostgreSQLStore:
    """
    PostgreSQL store with pgvector support for multi-modal memory storage and retrieval.
    """
    
    def __init__(self, 
                 database_url: str,
                 pool_size: int = 10,
                 max_connections: int = 20):
        """
        Initialize PostgreSQL store.
        
        Args:
            database_url: PostgreSQL connection URL
            pool_size: Connection pool size
            max_connections: Maximum connections
        """
        if not ASYNCPG_AVAILABLE:
            raise ImportError("AsyncPG not available. Install with: pip install asyncpg")
        
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_connections = max_connections
        self.pool = None
        
        logger.info(f"Initialized PostgreSQLStore with pool_size={pool_size}")
    
    async def initialize(self):
        """Initialize database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.pool_size,
                max_size=self.max_connections,
                command_timeout=60
            )
            logger.info("PostgreSQL connection pool created successfully")
            
            # Test connection
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT version()")
                logger.info(f"Connected to PostgreSQL: {result}")
                
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise
    
    async def close(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
    
    async def save_memory_record(self, memory_record: MemoryRecord) -> MemoryRecord:
        """
        Save a memory record to the database.
        
        Args:
            memory_record: Memory record to save
            
        Returns:
            Saved memory record with ID assigned
        """
        try:
            async with self.pool.acquire() as conn:
                # Generate ID if not provided
                if memory_record.id is None:
                    memory_record.id = uuid.uuid4()
                
                # Convert embeddings to vectors
                text_embedding = memory_record.text_embedding
                image_embedding = memory_record.image_embedding
                audio_embedding = memory_record.audio_embedding
                unified_embedding = memory_record.unified_embedding
                
                # Insert main record
                query = """
                    INSERT INTO memory_records (
                        id, content_type, content_text, content_metadata, file_path,
                        text_embedding, image_embedding, audio_embedding, unified_embedding,
                        created_at, last_accessed, access_count, memory_type,
                        emotional_valence, confidence_score, tags
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
                    )
                """
                
                await conn.execute(
                    query,
                    memory_record.id,
                    memory_record.content_type.value,
                    memory_record.content_text,
                    json.dumps(memory_record.content_metadata),
                    memory_record.file_path,
                    text_embedding,
                    image_embedding,
                    audio_embedding,
                    unified_embedding,
                    memory_record.created_at or datetime.utcnow(),
                    memory_record.last_accessed or datetime.utcnow(),
                    memory_record.access_count,
                    memory_record.memory_type.value,
                    memory_record.emotional_valence,
                    memory_record.confidence_score,
                    memory_record.tags
                )
                
                # Save type-specific metadata
                await self._save_type_specific_metadata(conn, memory_record)
                
                logger.info(f"Saved memory record: {memory_record.id}")
                return memory_record
                
        except Exception as e:
            logger.error(f"Failed to save memory record: {e}")
            raise
    
    async def _save_type_specific_metadata(self, conn, memory_record: MemoryRecord):
        """Save type-specific metadata tables."""
        if memory_record.content_type == ContentType.AUDIO and memory_record.audio_metadata:
            await self._save_audio_metadata(conn, memory_record.id, memory_record.audio_metadata)
        elif memory_record.content_type == ContentType.IMAGE and memory_record.image_metadata:
            await self._save_image_metadata(conn, memory_record.id, memory_record.image_metadata)
        elif memory_record.content_type == ContentType.VIDEO and memory_record.video_metadata:
            await self._save_video_metadata(conn, memory_record.id, memory_record.video_metadata)
    
    async def _save_audio_metadata(self, conn, memory_id: uuid.UUID, audio_metadata: AudioMetadata):
        """Save audio-specific metadata."""
        query = """
            INSERT INTO audio_memories (
                memory_id, transcript, language_code, confidence_scores,
                duration_seconds, audio_features, sample_rate, channels
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (memory_id) DO UPDATE SET
                transcript = EXCLUDED.transcript,
                language_code = EXCLUDED.language_code,
                confidence_scores = EXCLUDED.confidence_scores,
                duration_seconds = EXCLUDED.duration_seconds,
                audio_features = EXCLUDED.audio_features,
                sample_rate = EXCLUDED.sample_rate,
                channels = EXCLUDED.channels
        """
        
        await conn.execute(
            query,
            memory_id,
            audio_metadata.transcript,
            audio_metadata.language_code,
            json.dumps(audio_metadata.confidence_scores),
            audio_metadata.duration_seconds,
            json.dumps(audio_metadata.audio_features),
            audio_metadata.sample_rate,
            audio_metadata.channels
        )
    
    async def _save_image_metadata(self, conn, memory_id: uuid.UUID, image_metadata: ImageMetadata):
        """Save image-specific metadata."""
        query = """
            INSERT INTO image_memories (
                memory_id, width, height, object_detections,
                scene_description, image_hash, color_palette, image_features
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (memory_id) DO UPDATE SET
                width = EXCLUDED.width,
                height = EXCLUDED.height,
                object_detections = EXCLUDED.object_detections,
                scene_description = EXCLUDED.scene_description,
                image_hash = EXCLUDED.image_hash,
                color_palette = EXCLUDED.color_palette,
                image_features = EXCLUDED.image_features
        """
        
        await conn.execute(
            query,
            memory_id,
            image_metadata.width,
            image_metadata.height,
            json.dumps(image_metadata.object_detections),
            image_metadata.scene_description,
            image_metadata.image_hash,
            json.dumps(image_metadata.color_palette),
            json.dumps(image_metadata.image_features)
        )
    
    async def _save_video_metadata(self, conn, memory_id: uuid.UUID, video_metadata: VideoMetadata):
        """Save video-specific metadata."""
        query = """
            INSERT INTO video_memories (
                memory_id, duration_seconds, frame_rate, width, height,
                video_features, thumbnail_path
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (memory_id) DO UPDATE SET
                duration_seconds = EXCLUDED.duration_seconds,
                frame_rate = EXCLUDED.frame_rate,
                width = EXCLUDED.width,
                height = EXCLUDED.height,
                video_features = EXCLUDED.video_features,
                thumbnail_path = EXCLUDED.thumbnail_path
        """
        
        await conn.execute(
            query,
            memory_id,
            video_metadata.duration_seconds,
            video_metadata.frame_rate,
            video_metadata.width,
            video_metadata.height,
            json.dumps(video_metadata.video_features),
            video_metadata.thumbnail_path
        )
    
    async def get_memory_record(self, memory_id: uuid.UUID) -> Optional[MemoryRecord]:
        """
        Retrieve a memory record by ID.
        
        Args:
            memory_id: Memory record ID
            
        Returns:
            Memory record if found, None otherwise
        """
        try:
            async with self.pool.acquire() as conn:
                # Get main record
                query = """
                    SELECT * FROM memory_records WHERE id = $1
                """
                row = await conn.fetchrow(query, memory_id)
                
                if not row:
                    return None
                
                # Convert to MemoryRecord
                memory_record = await self._row_to_memory_record(conn, row)
                return memory_record
                
        except Exception as e:
            logger.error(f"Failed to get memory record {memory_id}: {e}")
            return None
    
    async def _row_to_memory_record(self, conn, row) -> MemoryRecord:
        """Convert database row to MemoryRecord object."""
        memory_record = MemoryRecord(
            id=row['id'],
            content_type=ContentType(row['content_type']),
            content_text=row['content_text'],
            content_metadata=json.loads(row['content_metadata']) if row['content_metadata'] else {},
            file_path=row['file_path'],
            text_embedding=row['text_embedding'],
            image_embedding=row['image_embedding'],
            audio_embedding=row['audio_embedding'],
            unified_embedding=row['unified_embedding'],
            created_at=row['created_at'],
            last_accessed=row['last_accessed'],
            access_count=row['access_count'],
            memory_type=MemoryType(row['memory_type']),
            emotional_valence=row['emotional_valence'],
            confidence_score=row['confidence_score'],
            tags=row['tags'] or []
        )
        
        # Load type-specific metadata
        await self._load_type_specific_metadata(conn, memory_record)
        
        return memory_record
    
    async def _load_type_specific_metadata(self, conn, memory_record: MemoryRecord):
        """Load type-specific metadata."""
        if memory_record.content_type == ContentType.AUDIO:
            memory_record.audio_metadata = await self._load_audio_metadata(conn, memory_record.id)
        elif memory_record.content_type == ContentType.IMAGE:
            memory_record.image_metadata = await self._load_image_metadata(conn, memory_record.id)
        elif memory_record.content_type == ContentType.VIDEO:
            memory_record.video_metadata = await self._load_video_metadata(conn, memory_record.id)
    
    async def _load_audio_metadata(self, conn, memory_id: uuid.UUID) -> Optional[AudioMetadata]:
        """Load audio metadata."""
        query = "SELECT * FROM audio_memories WHERE memory_id = $1"
        row = await conn.fetchrow(query, memory_id)
        
        if row:
            return AudioMetadata(
                transcript=row['transcript'],
                language_code=row['language_code'],
                confidence_scores=json.loads(row['confidence_scores']) if row['confidence_scores'] else {},
                duration_seconds=row['duration_seconds'],
                audio_features=json.loads(row['audio_features']) if row['audio_features'] else {},
                sample_rate=row['sample_rate'],
                channels=row['channels']
            )
        return None
    
    async def _load_image_metadata(self, conn, memory_id: uuid.UUID) -> Optional[ImageMetadata]:
        """Load image metadata."""
        query = "SELECT * FROM image_memories WHERE memory_id = $1"
        row = await conn.fetchrow(query, memory_id)
        
        if row:
            return ImageMetadata(
                width=row['width'],
                height=row['height'],
                object_detections=json.loads(row['object_detections']) if row['object_detections'] else {},
                scene_description=row['scene_description'],
                image_hash=row['image_hash'],
                color_palette=json.loads(row['color_palette']) if row['color_palette'] else {},
                image_features=json.loads(row['image_features']) if row['image_features'] else {}
            )
        return None
    
    async def _load_video_metadata(self, conn, memory_id: uuid.UUID) -> Optional[VideoMetadata]:
        """Load video metadata."""
        query = "SELECT * FROM video_memories WHERE memory_id = $1"
        row = await conn.fetchrow(query, memory_id)
        
        if row:
            return VideoMetadata(
                duration_seconds=row['duration_seconds'],
                frame_rate=row['frame_rate'],
                width=row['width'],
                height=row['height'],
                video_features=json.loads(row['video_features']) if row['video_features'] else {},
                thumbnail_path=row['thumbnail_path']
            )
        return None
    
    async def vector_search(self, 
                           embedding: List[float],
                           embedding_type: str = "text",
                           limit: int = 10,
                           similarity_threshold: float = 0.7,
                           content_types: Optional[List[ContentType]] = None) -> List[Tuple[MemoryRecord, float]]:
        """
        Perform vector similarity search.
        
        Args:
            embedding: Query embedding
            embedding_type: Type of embedding ("text", "image", "audio", "unified")
            limit: Maximum results to return
            similarity_threshold: Minimum similarity threshold
            content_types: Filter by content types
            
        Returns:
            List of (memory_record, similarity_score) tuples
        """
        try:
            async with self.pool.acquire() as conn:
                # Build query based on embedding type
                embedding_column = f"{embedding_type}_embedding"
                
                where_conditions = [f"{embedding_column} IS NOT NULL"]
                params = [embedding]
                param_count = 1
                
                if content_types:
                    param_count += 1
                    where_conditions.append(f"content_type = ANY(${param_count})")
                    params.append([ct.value for ct in content_types])
                
                param_count += 1
                where_conditions.append(f"1 - ({embedding_column} <=> ${param_count}) >= ${param_count + 1}")
                params.extend([embedding, similarity_threshold])
                
                query = f"""
                    SELECT *, 1 - ({embedding_column} <=> $1) as similarity
                    FROM memory_records 
                    WHERE {' AND '.join(where_conditions)}
                    ORDER BY {embedding_column} <=> $1
                    LIMIT ${param_count + 2}
                """
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                # Convert to MemoryRecord objects
                results = []
                for row in rows:
                    memory_record = await self._row_to_memory_record(conn, row)
                    similarity = float(row['similarity'])
                    results.append((memory_record, similarity))
                
                # Update access statistics
                if results:
                    memory_ids = [r[0].id for r in results]
                    await self._update_access_stats(conn, memory_ids)
                
                logger.info(f"Vector search returned {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def text_search(self,
                         query_text: str,
                         limit: int = 10,
                         content_types: Optional[List[ContentType]] = None) -> List[Tuple[MemoryRecord, float]]:
        """
        Perform full-text search.
        
        Args:
            query_text: Search query
            limit: Maximum results to return
            content_types: Filter by content types
            
        Returns:
            List of (memory_record, similarity_score) tuples
        """
        try:
            async with self.pool.acquire() as conn:
                where_conditions = ["search_vector @@ plainto_tsquery($1)"]
                params = [query_text]
                param_count = 1
                
                if content_types:
                    param_count += 1
                    where_conditions.append(f"content_type = ANY(${param_count})")
                    params.append([ct.value for ct in content_types])
                
                query = f"""
                    SELECT *, ts_rank(search_vector, plainto_tsquery($1)) as text_score
                    FROM memory_records
                    WHERE {' AND '.join(where_conditions)}
                    ORDER BY text_score DESC
                    LIMIT ${param_count + 1}
                """
                params.append(limit)
                
                rows = await conn.fetch(query, *params)
                
                # Convert to MemoryRecord objects
                results = []
                for row in rows:
                    memory_record = await self._row_to_memory_record(conn, row)
                    score = float(row['text_score'])
                    results.append((memory_record, score))
                
                # Update access statistics
                if results:
                    memory_ids = [r[0].id for r in results]
                    await self._update_access_stats(conn, memory_ids)
                
                logger.info(f"Text search returned {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []
    
    async def _update_access_stats(self, conn, memory_ids: List[uuid.UUID]):
        """Update access statistics for memory records."""
        query = """
            UPDATE memory_records 
            SET last_accessed = NOW(), access_count = access_count + 1
            WHERE id = ANY($1)
        """
        await conn.execute(query, memory_ids)
    
    async def delete_memory_record(self, memory_id: uuid.UUID) -> bool:
        """
        Delete a memory record.
        
        Args:
            memory_id: Memory record ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            async with self.pool.acquire() as conn:
                query = "DELETE FROM memory_records WHERE id = $1"
                result = await conn.execute(query, memory_id)
                
                deleted = result.split()[-1] == '1'
                if deleted:
                    logger.info(f"Deleted memory record: {memory_id}")
                
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete memory record {memory_id}: {e}")
            return False
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        try:
            async with self.pool.acquire() as conn:
                # Get basic stats
                stats_query = """
                    SELECT 
                        COUNT(*) as total_memories,
                        COUNT(DISTINCT content_type) as content_types,
                        AVG(confidence_score) as avg_confidence,
                        MIN(created_at) as oldest_memory,
                        MAX(created_at) as newest_memory,
                        AVG(access_count) as avg_access_count
                    FROM memory_records
                """
                stats = await conn.fetchrow(stats_query)
                
                # Get breakdown by content type
                type_query = """
                    SELECT content_type, COUNT(*) as count
                    FROM memory_records
                    GROUP BY content_type
                """
                type_breakdown = await conn.fetch(type_query)
                
                return {
                    "total_memories": stats['total_memories'],
                    "content_types": dict(type_breakdown),
                    "avg_confidence": float(stats['avg_confidence']) if stats['avg_confidence'] else 0.0,
                    "oldest_memory": stats['oldest_memory'],
                    "newest_memory": stats['newest_memory'],
                    "avg_access_count": float(stats['avg_access_count']) if stats['avg_access_count'] else 0.0
                }
                
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    async def cleanup_old_memories(self, days_old: int = 30, keep_minimum: int = 1000) -> int:
        """
        Clean up old, rarely accessed memories.
        
        Args:
            days_old: Age threshold in days
            keep_minimum: Minimum number of memories to keep
            
        Returns:
            Number of memories deleted
        """
        try:
            async with self.pool.acquire() as conn:
                # Check current count
                count_query = "SELECT COUNT(*) FROM memory_records"
                current_count = await conn.fetchval(count_query)
                
                if current_count <= keep_minimum:
                    return 0
                
                # Delete old memories
                cutoff_date = datetime.utcnow() - timedelta(days=days_old)
                delete_query = """
                    DELETE FROM memory_records 
                    WHERE created_at < $1 
                    AND access_count <= 1
                    AND id NOT IN (
                        SELECT id FROM memory_records 
                        ORDER BY last_accessed DESC 
                        LIMIT $2
                    )
                """
                
                result = await conn.execute(delete_query, cutoff_date, keep_minimum)
                deleted_count = int(result.split()[-1])
                
                logger.info(f"Cleaned up {deleted_count} old memories")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            return 0