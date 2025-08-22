"""
Very Long-Term Memory Data Migration Utilities

This module implements data migration utilities to populate the VLTM system 
from existing episodic memory and knowledge compression systems.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass

from core.vltm_store import VeryLongTermMemoryStore
from core.vltm_data_models import MemoryType, MemoryRecord
from services.memory_service import MemoryService
from services.knowledge_service import KnowledgeService

logger = logging.getLogger(__name__)


@dataclass
class MigrationConfig:
    """Configuration for memory migration"""
    batch_size: int = 50
    episodic_cutoff_days: Optional[int] = 30
    knowledge_cutoff_days: Optional[int] = 30
    min_confidence: float = 0.3
    dry_run: bool = False
    skip_duplicates: bool = True


@dataclass
class MigrationStats:
    """Migration statistics"""
    episodic_migrated: int = 0
    knowledge_migrated: int = 0
    total_failed: int = 0
    processing_time: float = 0.0


class VLTMDataMigrator:
    """Data migration utility for VLTM system"""
    
    def __init__(self, vltm_store: VeryLongTermMemoryStore, 
                 memory_service: MemoryService, knowledge_service: KnowledgeService,
                 config: Optional[MigrationConfig] = None):
        self.vltm_store = vltm_store
        self.memory_service = memory_service
        self.knowledge_service = knowledge_service
        self.config = config or MigrationConfig()
        self.migration_id = str(uuid.uuid4())
        self.migrated_ids: Set[str] = set()
        
    async def migrate_all_data(self) -> MigrationStats:
        """Perform complete data migration"""
        start_time = datetime.utcnow()
        stats = MigrationStats()
        
        try:
            # Migrate episodic memories
            episodic_stats = await self._migrate_episodic_memories()
            stats.episodic_migrated = episodic_stats["migrated"]
            stats.total_failed += episodic_stats["failed"]
            
            # Migrate knowledge data
            knowledge_stats = await self._migrate_knowledge_data()
            stats.knowledge_migrated = knowledge_stats["migrated"]
            stats.total_failed += knowledge_stats["failed"]
            
            stats.processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.info(f"Migration completed: {stats.episodic_migrated + stats.knowledge_migrated} memories migrated")
            return stats
            
        except Exception as e:
            logger.error(f"Migration error: {e}")
            return stats
    
    async def _migrate_episodic_memories(self) -> Dict[str, int]:
        """Migrate episodic memory data"""
        logger.info("Migrating episodic memories...")
        stats = {"migrated": 0, "failed": 0}
        
        try:
            # Fetch sample episodic memories (in production, would query actual DB)
            episodic_memories = await self._fetch_sample_episodic_data()
            
            for memory_data in episodic_memories:
                try:
                    vltm_memory = self._convert_episodic_to_vltm(memory_data)
                    if not vltm_memory:
                        continue
                    
                    if self.config.skip_duplicates and await self._is_duplicate(vltm_memory):
                        continue
                    
                    if not self.config.dry_run:
                        memory_id = await self.vltm_store.store_memory(
                            content=vltm_memory["content"],
                            memory_type=vltm_memory["memory_type"],
                            metadata=vltm_memory["metadata"],
                            source_session="episodic_migration"
                        )
                        if memory_id:
                            self.migrated_ids.add(memory_id)
                            stats["migrated"] += 1
                        else:
                            stats["failed"] += 1
                    else:
                        stats["migrated"] += 1
                        
                except Exception as e:
                    stats["failed"] += 1
                    logger.error(f"Error processing episodic memory: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error migrating episodic memories: {e}")
            return stats
    
    async def _fetch_sample_episodic_data(self) -> List[Dict[str, Any]]:
        """Fetch sample episodic memory data"""
        # Sample data representing episodic memory structure
        return [
            {
                "id": str(uuid.uuid4()),
                "content_text": "System optimized query performance by 40%",
                "content_type": "text",
                "created_at": datetime.utcnow() - timedelta(days=5),
                "memory_type": "episodic",
                "confidence_score": 0.8,
                "tags": ["optimization", "performance"]
            },
            {
                "id": str(uuid.uuid4()),
                "content_text": "Database connection error during peak hours",
                "content_type": "text", 
                "created_at": datetime.utcnow() - timedelta(days=2),
                "memory_type": "episodic",
                "confidence_score": 0.9,
                "tags": ["error", "database"]
            }
        ]
    
    def _convert_episodic_to_vltm(self, episodic_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert episodic memory to VLTM format"""
        try:
            content_text = episodic_data.get("content_text", "")
            if len(content_text.strip()) < 10:
                return None
            
            memory_type = self._classify_episodic_content(content_text)
            
            content = {
                "original_text": content_text,
                "migration_info": {
                    "source": "episodic_memory",
                    "original_id": episodic_data.get("id"),
                    "migrated_at": datetime.utcnow().isoformat(),
                    "migration_id": self.migration_id
                }
            }
            
            metadata = {
                "episodic_migration": True,
                "original_created_at": episodic_data.get("created_at").isoformat() if episodic_data.get("created_at") else None,
                "tags": episodic_data.get("tags", [])
            }
            
            return {"content": content, "memory_type": memory_type, "metadata": metadata}
            
        except Exception as e:
            logger.error(f"Error converting episodic memory: {e}")
            return None
    
    def _classify_episodic_content(self, content: str) -> MemoryType:
        """Classify episodic content into VLTM memory type"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["optimization", "improved", "performance"]):
            return MemoryType.SUCCESSFUL_IMPROVEMENT
        elif any(word in content_lower for word in ["failed", "error", "crash"]):
            return MemoryType.FAILED_EXPERIMENT
        elif any(word in content_lower for word in ["architecture", "design"]):
            return MemoryType.ARCHITECTURAL_INSIGHT
        elif any(word in content_lower for word in ["critical", "urgent"]):
            return MemoryType.CRITICAL_FAILURE
        else:
            return MemoryType.CODE_PATTERN
    
    async def _migrate_knowledge_data(self) -> Dict[str, int]:
        """Migrate knowledge compression data"""
        logger.info("Migrating knowledge data...")
        stats = {"migrated": 0, "failed": 0}
        
        try:
            # Fetch knowledge data from service
            knowledge_entries = []
            categories = ["compression", "system", "optimization"]
            
            for category in categories:
                try:
                    category_data = self.knowledge_service.get_knowledge_by_category(category, limit=20)
                    knowledge_entries.extend(category_data)
                except Exception as e:
                    logger.warning(f"Error fetching knowledge for {category}: {e}")
            
            for knowledge_data in knowledge_entries:
                try:
                    vltm_memory = self._convert_knowledge_to_vltm(knowledge_data)
                    if not vltm_memory:
                        continue
                    
                    if self.config.skip_duplicates and await self._is_duplicate(vltm_memory):
                        continue
                    
                    if not self.config.dry_run:
                        memory_id = await self.vltm_store.store_memory(
                            content=vltm_memory["content"],
                            memory_type=vltm_memory["memory_type"],
                            metadata=vltm_memory["metadata"],
                            source_session="knowledge_migration"
                        )
                        if memory_id:
                            self.migrated_ids.add(memory_id)
                            stats["migrated"] += 1
                        else:
                            stats["failed"] += 1
                    else:
                        stats["migrated"] += 1
                        
                except Exception as e:
                    stats["failed"] += 1
                    logger.error(f"Error processing knowledge entry: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error migrating knowledge data: {e}")
            return stats
    
    def _convert_knowledge_to_vltm(self, knowledge_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert knowledge entry to VLTM format"""
        try:
            summary = knowledge_data.get("summary", "")
            if len(summary.strip()) < 20:
                return None
            
            memory_type = self._classify_knowledge_content(knowledge_data)
            
            content = {
                "knowledge_summary": summary,
                "migration_info": {
                    "source": "knowledge_service",
                    "original_id": knowledge_data.get("id"),
                    "migrated_at": datetime.utcnow().isoformat(),
                    "migration_id": self.migration_id
                }
            }
            
            metadata = {
                "knowledge_migration": True,
                "original_category": knowledge_data.get("category"),
                "original_source": knowledge_data.get("source")
            }
            
            return {"content": content, "memory_type": memory_type, "metadata": metadata}
            
        except Exception as e:
            logger.error(f"Error converting knowledge entry: {e}")
            return None
    
    def _classify_knowledge_content(self, knowledge_data: Dict[str, Any]) -> MemoryType:
        """Classify knowledge content into VLTM memory type"""
        category = knowledge_data.get("category", "").lower()
        summary = knowledge_data.get("summary", "").lower()
        
        if category in ["compression", "system"] or "strategic" in summary:
            return MemoryType.STRATEGIC_KNOWLEDGE
        elif "architecture" in summary:
            return MemoryType.ARCHITECTURAL_INSIGHT
        elif "learning" in summary:
            return MemoryType.META_LEARNING_RULE
        elif "optimization" in summary:
            return MemoryType.SUCCESSFUL_IMPROVEMENT
        else:
            return MemoryType.STRATEGIC_KNOWLEDGE
    
    async def _is_duplicate(self, vltm_memory: Dict[str, Any]) -> bool:
        """Check if memory is a duplicate"""
        try:
            content = vltm_memory["content"]
            search_text = (content.get("original_text") or content.get("knowledge_summary", ""))[:100]
            
            if not search_text:
                return False
            
            similar_memories = await self.vltm_store.search_memories(
                query=search_text, limit=3, similarity_threshold=0.85
            )
            return len(similar_memories) > 0
            
        except Exception:
            return False
    
    async def incremental_migration(self, hours: int = 24) -> MigrationStats:
        """Perform incremental migration of recent data"""
        logger.info(f"Starting incremental migration for last {hours} hours")
        
        # Temporarily adjust cutoff for incremental
        original_episodic = self.config.episodic_cutoff_days
        original_knowledge = self.config.knowledge_cutoff_days
        
        self.config.episodic_cutoff_days = hours / 24
        self.config.knowledge_cutoff_days = hours / 24
        
        try:
            return await self.migrate_all_data()
        finally:
            self.config.episodic_cutoff_days = original_episodic
            self.config.knowledge_cutoff_days = original_knowledge