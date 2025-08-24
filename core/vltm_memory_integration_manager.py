"""
Very Long-Term Memory Integration Manager

This module implements the MemoryIntegrationManager that coordinates between 
existing memory systems (episodic, semantic) and the very long-term memory system.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

from core.vltm_store import VeryLongTermMemoryStore
from core.vltm_consolidation_engine import MemoryConsolidationEngine
from core.vltm_data_models import MemoryType, MemoryImportance, VLTMConfiguration
from core.vltm_advanced_retrieval import AdvancedRetrievalEngine
from services.memory_service import MemoryService
from services.knowledge_service import KnowledgeService

logger = logging.getLogger(__name__)


class MemoryFlowDirection(str, Enum):
    """Direction of memory flow"""
    TO_VLTM = "to_vltm"
    FROM_VLTM = "from_vltm"
    BIDIRECTIONAL = "bidirectional"


class IntegrationMode(str, Enum):
    """Integration mode for memory systems"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    HYBRID = "hybrid"
    SELECTIVE = "selective"


@dataclass
class MemoryBridge:
    """Configuration for memory system bridge"""
    source_system: str
    target_system: str
    flow_direction: MemoryFlowDirection
    memory_types: List[MemoryType]
    sync_interval_minutes: int = 60
    batch_size: int = 100
    enabled: bool = True


@dataclass
class IntegrationStats:
    """Statistics for memory integration operations"""
    memories_synchronized: int = 0
    patterns_extracted: int = 0
    knowledge_consolidated: int = 0
    failed_operations: int = 0
    processing_time_seconds: float = 0.0
    last_sync_timestamp: Optional[datetime] = None


class MemoryIntegrationManager:
    """
    Coordinates memory flow between existing memory systems and VLTM.
    
    This manager ensures seamless integration between:
    - Episodic memory system (short-term storage)
    - Semantic/Knowledge system (structured knowledge)
    - Very Long-Term Memory system (strategic patterns and insights)
    """
    
    def __init__(self, 
                 vltm_store: VeryLongTermMemoryStore,
                 consolidation_engine: MemoryConsolidationEngine,
                 retrieval_engine: AdvancedRetrievalEngine,
                 memory_service: MemoryService,
                 knowledge_service: KnowledgeService,
                 config: Optional[VLTMConfiguration] = None):
        
        self.vltm_store = vltm_store
        self.consolidation_engine = consolidation_engine
        self.retrieval_engine = retrieval_engine
        self.memory_service = memory_service
        self.knowledge_service = knowledge_service
        self.config = config
        
        # Memory bridges configuration
        self.memory_bridges: List[MemoryBridge] = []
        self.integration_stats = IntegrationStats()
        
        # Synchronization state
        self.is_running = False
        self.sync_tasks: Dict[str, asyncio.Task] = {}
        self.last_sync_checkpoints: Dict[str, datetime] = {}
        
        # Memory flow tracking
        self.memory_flow_log: List[Dict[str, Any]] = []
        self.max_flow_log_entries = 1000
        
        logger.info("Memory Integration Manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize the memory integration manager"""
        try:
            # Setup default memory bridges
            await self._setup_default_bridges()
            
            # Initialize synchronization checkpoints
            await self._load_sync_checkpoints()
            
            # Start monitoring tasks
            await self._start_monitoring()
            
            self.is_running = True
            logger.info("Memory Integration Manager initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Memory Integration Manager: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the integration manager gracefully"""
        try:
            self.is_running = False
            
            # Cancel all sync tasks
            for task_name, task in self.sync_tasks.items():
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        logger.info(f"Cancelled sync task: {task_name}")
            
            # Save sync checkpoints
            await self._save_sync_checkpoints()
            
            logger.info("Memory Integration Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during Memory Integration Manager shutdown: {e}")
    
    async def _setup_default_bridges(self):
        """Setup default memory bridges between systems"""
        
        # Bridge: Episodic Memory → VLTM
        episodic_to_vltm = MemoryBridge(
            source_system="episodic_memory",
            target_system="vltm",
            flow_direction=MemoryFlowDirection.TO_VLTM,
            memory_types=[
                MemoryType.SUCCESSFUL_IMPROVEMENT,
                MemoryType.FAILED_EXPERIMENT,
                MemoryType.CRITICAL_FAILURE,
                MemoryType.ARCHITECTURAL_INSIGHT
            ],
            sync_interval_minutes=30,
            batch_size=50
        )
        
        # Bridge: Knowledge System → VLTM
        knowledge_to_vltm = MemoryBridge(
            source_system="knowledge_service",
            target_system="vltm",
            flow_direction=MemoryFlowDirection.TO_VLTM,
            memory_types=[
                MemoryType.STRATEGIC_KNOWLEDGE,
                MemoryType.META_LEARNING_RULE,
                MemoryType.CODE_PATTERN
            ],
            sync_interval_minutes=60,
            batch_size=25
        )
        
        # Bridge: VLTM → Knowledge System (strategic insights)
        vltm_to_knowledge = MemoryBridge(
            source_system="vltm",
            target_system="knowledge_service",
            flow_direction=MemoryFlowDirection.FROM_VLTM,
            memory_types=[MemoryType.STRATEGIC_KNOWLEDGE],
            sync_interval_minutes=120,
            batch_size=10
        )
        
        self.memory_bridges = [episodic_to_vltm, knowledge_to_vltm, vltm_to_knowledge]
        logger.info(f"Setup {len(self.memory_bridges)} default memory bridges")
    
    async def _start_monitoring(self):
        """Start monitoring tasks for memory synchronization"""
        
        for bridge in self.memory_bridges:
            if bridge.enabled:
                task_name = f"sync_{bridge.source_system}_to_{bridge.target_system}"
                
                # Create sync task for this bridge
                task = asyncio.create_task(
                    self._sync_bridge_continuously(bridge),
                    name=task_name
                )
                self.sync_tasks[task_name] = task
                
                logger.info(f"Started monitoring task: {task_name}")
    
    async def _sync_bridge_continuously(self, bridge: MemoryBridge):
        """Continuously sync a memory bridge"""
        
        while self.is_running:
            try:
                await self._sync_memory_bridge(bridge)
                
                # Wait for the next sync interval
                await asyncio.sleep(bridge.sync_interval_minutes * 60)
                
            except asyncio.CancelledError:
                logger.info(f"Sync task cancelled for bridge: {bridge.source_system} → {bridge.target_system}")
                break
            except Exception as e:
                logger.error(f"Error in bridge sync: {e}")
                # Wait before retrying
                await asyncio.sleep(60)
    
    async def _sync_memory_bridge(self, bridge: MemoryBridge):
        """Synchronize memories across a specific bridge"""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting sync: {bridge.source_system} → {bridge.target_system}")
            
            if bridge.flow_direction == MemoryFlowDirection.TO_VLTM:
                if bridge.source_system == "episodic_memory":
                    await self._sync_episodic_to_vltm(bridge)
                elif bridge.source_system == "knowledge_service":
                    await self._sync_knowledge_to_vltm(bridge)
                    
            elif bridge.flow_direction == MemoryFlowDirection.FROM_VLTM:
                if bridge.target_system == "knowledge_service":
                    await self._sync_vltm_to_knowledge(bridge)
                    
            # Update sync checkpoint
            self.last_sync_checkpoints[f"{bridge.source_system}_{bridge.target_system}"] = start_time
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Sync completed in {processing_time:.2f}s")
            
        except Exception as e:
            self.integration_stats.failed_operations += 1
            logger.error(f"Error syncing bridge: {e}")
    
    async def _sync_episodic_to_vltm(self, bridge: MemoryBridge):
        """Sync episodic memories to VLTM"""
        
        checkpoint_key = f"{bridge.source_system}_{bridge.target_system}"
        last_sync = self.last_sync_checkpoints.get(checkpoint_key)
        
        # Get recent episodic memories
        recent_memories = await self._get_recent_episodic_memories(
            since=last_sync, 
            limit=bridge.batch_size
        )
        
        memories_synced = 0
        for memory_data in recent_memories:
            try:
                # Filter by memory types
                memory_type = self._classify_episodic_memory(memory_data)
                if memory_type not in bridge.memory_types:
                    continue
                
                # Convert to VLTM format
                vltm_memory = await self._convert_episodic_to_vltm(memory_data, memory_type)
                if not vltm_memory:
                    continue
                
                # Check importance threshold
                if not await self._meets_vltm_importance_threshold(vltm_memory):
                    continue
                
                # Store in VLTM
                memory_id = await self.vltm_store.store_memory(
                    content=vltm_memory["content"],
                    memory_type=memory_type,
                    metadata=vltm_memory["metadata"],
                    source_session="episodic_sync"
                )
                
                if memory_id:
                    memories_synced += 1
                    self._log_memory_flow(
                        source="episodic_memory",
                        target="vltm",
                        memory_id=memory_id,
                        memory_type=memory_type
                    )
                
            except Exception as e:
                logger.warning(f"Failed to sync episodic memory: {e}")
                self.integration_stats.failed_operations += 1
        
        self.integration_stats.memories_synchronized += memories_synced
        logger.info(f"Synced {memories_synced} episodic memories to VLTM")
    
    async def _sync_knowledge_to_vltm(self, bridge: MemoryBridge):
        """Sync knowledge data to VLTM"""
        
        checkpoint_key = f"{bridge.source_system}_{bridge.target_system}"
        last_sync = self.last_sync_checkpoints.get(checkpoint_key)
        
        # Get recent knowledge entries
        recent_knowledge = await self._get_recent_knowledge_entries(
            since=last_sync,
            limit=bridge.batch_size
        )
        
        knowledge_synced = 0
        for knowledge_data in recent_knowledge:
            try:
                # Convert to VLTM format
                vltm_memory = await self._convert_knowledge_to_vltm(knowledge_data)
                if not vltm_memory:
                    continue
                
                # Check if already exists
                if await self._vltm_memory_exists(vltm_memory):
                    continue
                
                # Store in VLTM
                memory_id = await self.vltm_store.store_memory(
                    content=vltm_memory["content"],
                    memory_type=vltm_memory["memory_type"],
                    metadata=vltm_memory["metadata"],
                    source_session="knowledge_sync"
                )
                
                if memory_id:
                    knowledge_synced += 1
                    self._log_memory_flow(
                        source="knowledge_service",
                        target="vltm",
                        memory_id=memory_id,
                        memory_type=vltm_memory["memory_type"]
                    )
                
            except Exception as e:
                logger.warning(f"Failed to sync knowledge entry: {e}")
                self.integration_stats.failed_operations += 1
        
        self.integration_stats.knowledge_consolidated += knowledge_synced
        logger.info(f"Synced {knowledge_synced} knowledge entries to VLTM")
    
    async def _sync_vltm_to_knowledge(self, bridge: MemoryBridge):
        """Sync strategic insights from VLTM to knowledge system"""
        
        # Get strategic knowledge from VLTM
        strategic_memories = await self.vltm_store.get_memories_by_type(
            memory_type=MemoryType.STRATEGIC_KNOWLEDGE,
            limit=bridge.batch_size,
            importance_threshold=MemoryImportance.HIGH.value
        )
        
        insights_synced = 0
        for memory in strategic_memories:
            try:
                # Convert VLTM memory to knowledge format
                knowledge_entry = await self._convert_vltm_to_knowledge(memory)
                if not knowledge_entry:
                    continue
                
                # Check if already exists in knowledge system
                if await self._knowledge_entry_exists(knowledge_entry):
                    continue
                
                # Store in knowledge system
                knowledge_id = await self.knowledge_service.store_knowledge(
                    content=knowledge_entry["content"],
                    category=knowledge_entry["category"],
                    metadata=knowledge_entry["metadata"]
                )
                
                if knowledge_id:
                    insights_synced += 1
                    self._log_memory_flow(
                        source="vltm",
                        target="knowledge_service",
                        memory_id=memory["memory_id"],
                        memory_type=MemoryType.STRATEGIC_KNOWLEDGE
                    )
                
            except Exception as e:
                logger.warning(f"Failed to sync strategic insight: {e}")
                self.integration_stats.failed_operations += 1
        
        logger.info(f"Synced {insights_synced} strategic insights to knowledge system")
    
    async def _get_recent_episodic_memories(self, since: Optional[datetime], limit: int) -> List[Dict[str, Any]]:
        """Get recent episodic memories for synchronization"""
        
        # Simulate episodic memory retrieval
        # In production, this would query the actual episodic memory system
        memories = []
        
        cutoff_time = since or (datetime.utcnow() - timedelta(hours=24))
        
        # Sample episodic memories
        sample_memories = [
            {
                "id": str(uuid.uuid4()),
                "content": "Successfully optimized database query performance by 45%",
                "timestamp": datetime.utcnow() - timedelta(hours=2),
                "confidence": 0.9,
                "tags": ["optimization", "database", "performance"]
            },
            {
                "id": str(uuid.uuid4()),
                "content": "Encountered memory leak in background process",
                "timestamp": datetime.utcnow() - timedelta(hours=6),
                "confidence": 0.8,
                "tags": ["error", "memory", "process"]
            }
        ]
        
        # Filter by timestamp
        for memory in sample_memories:
            if memory["timestamp"] > cutoff_time:
                memories.append(memory)
        
        return memories[:limit]
    
    async def _get_recent_knowledge_entries(self, since: Optional[datetime], limit: int) -> List[Dict[str, Any]]:
        """Get recent knowledge entries for synchronization"""
        
        try:
            # Get knowledge entries from service
            knowledge_entries = []
            categories = ["system", "optimization", "patterns", "architecture"]
            
            for category in categories:
                entries = self.knowledge_service.get_knowledge_by_category(
                    category=category, 
                    limit=limit // len(categories)
                )
                knowledge_entries.extend(entries)
            
            return knowledge_entries
            
        except Exception as e:
            logger.error(f"Error getting recent knowledge entries: {e}")
            return []
    
    def _classify_episodic_memory(self, memory_data: Dict[str, Any]) -> MemoryType:
        """Classify episodic memory into VLTM memory type"""
        
        content = memory_data.get("content", "").lower()
        tags = memory_data.get("tags", [])
        
        # Classification logic
        if any(word in content for word in ["optimized", "improved", "enhanced"]):
            return MemoryType.SUCCESSFUL_IMPROVEMENT
        elif any(word in content for word in ["error", "failed", "crash"]):
            if any(word in content for word in ["critical", "severe"]):
                return MemoryType.CRITICAL_FAILURE
            else:
                return MemoryType.FAILED_EXPERIMENT
        elif any(word in content for word in ["architecture", "design", "pattern"]):
            return MemoryType.ARCHITECTURAL_INSIGHT
        elif "optimization" in tags:
            return MemoryType.SUCCESSFUL_IMPROVEMENT
        else:
            return MemoryType.CODE_PATTERN
    
    async def _convert_episodic_to_vltm(self, memory_data: Dict[str, Any], memory_type: MemoryType) -> Optional[Dict[str, Any]]:
        """Convert episodic memory to VLTM format"""
        
        try:
            content = {
                "original_content": memory_data.get("content"),
                "source_system": "episodic_memory",
                "integration_info": {
                    "synced_at": datetime.utcnow().isoformat(),
                    "original_id": memory_data.get("id"),
                    "confidence": memory_data.get("confidence", 0.5)
                }
            }
            
            metadata = {
                "episodic_sync": True,
                "original_timestamp": memory_data.get("timestamp").isoformat() if memory_data.get("timestamp") else None,
                "tags": memory_data.get("tags", [])
            }
            
            return {
                "content": content,
                "memory_type": memory_type,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error converting episodic memory: {e}")
            return None
    
    async def _convert_knowledge_to_vltm(self, knowledge_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert knowledge entry to VLTM format"""
        
        try:
            summary = knowledge_data.get("summary", "")
            if len(summary.strip()) < 20:
                return None
            
            # Determine memory type based on content
            memory_type = MemoryType.STRATEGIC_KNOWLEDGE
            if "pattern" in summary.lower():
                memory_type = MemoryType.CODE_PATTERN
            elif "learning" in summary.lower():
                memory_type = MemoryType.META_LEARNING_RULE
            
            content = {
                "knowledge_summary": summary,
                "source_system": "knowledge_service",
                "integration_info": {
                    "synced_at": datetime.utcnow().isoformat(),
                    "original_id": knowledge_data.get("id"),
                    "category": knowledge_data.get("category")
                }
            }
            
            metadata = {
                "knowledge_sync": True,
                "original_category": knowledge_data.get("category"),
                "source": knowledge_data.get("source")
            }
            
            return {
                "content": content,
                "memory_type": memory_type,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error converting knowledge entry: {e}")
            return None
    
    async def _convert_vltm_to_knowledge(self, vltm_memory: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert VLTM memory to knowledge system format"""
        
        try:
            content = vltm_memory.get("content", {})
            
            # Extract knowledge summary
            knowledge_summary = (
                content.get("knowledge_summary") or 
                content.get("strategic_insight") or
                content.get("original_content", "")
            )
            
            if len(knowledge_summary.strip()) < 20:
                return None
            
            return {
                "content": knowledge_summary,
                "category": "strategic_vltm",
                "metadata": {
                    "vltm_origin": True,
                    "memory_id": vltm_memory.get("memory_id"),
                    "importance_score": vltm_memory.get("importance_score"),
                    "synced_at": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error converting VLTM to knowledge: {e}")
            return None
    
    async def _meets_vltm_importance_threshold(self, vltm_memory: Dict[str, Any]) -> bool:
        """Check if memory meets VLTM importance threshold"""
        
        # Simple importance evaluation
        confidence = vltm_memory.get("metadata", {}).get("confidence", 0.5)
        tags = vltm_memory.get("metadata", {}).get("tags", [])
        
        # High importance if high confidence or critical tags
        critical_tags = ["optimization", "critical", "architecture", "performance"]
        has_critical_tags = any(tag in critical_tags for tag in tags)
        
        return confidence > 0.7 or has_critical_tags
    
    async def _vltm_memory_exists(self, vltm_memory: Dict[str, Any]) -> bool:
        """Check if VLTM memory already exists"""
        
        try:
            content = vltm_memory["content"]
            search_text = (
                content.get("knowledge_summary") or 
                content.get("original_content", "")
            )[:100]
            
            if not search_text:
                return False
            
            # Search for similar memories
            similar_memories = await self.vltm_store.search_memories(
                query=search_text,
                limit=3,
                similarity_threshold=0.85
            )
            
            return len(similar_memories) > 0
            
        except Exception:
            return False
    
    async def _knowledge_entry_exists(self, knowledge_entry: Dict[str, Any]) -> bool:
        """Check if knowledge entry already exists"""
        
        try:
            # Simple existence check based on content similarity
            content = knowledge_entry["content"][:100]
            
            existing_entries = self.knowledge_service.search_knowledge(
                query=content,
                limit=3
            )
            
            return len(existing_entries) > 0
            
        except Exception:
            return False
    
    def _log_memory_flow(self, source: str, target: str, memory_id: str, memory_type: MemoryType):
        """Log memory flow for tracking"""
        
        flow_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": source,
            "target": target,
            "memory_id": memory_id,
            "memory_type": memory_type.value,
            "flow_id": str(uuid.uuid4())
        }
        
        self.memory_flow_log.append(flow_entry)
        
        # Trim log if too large
        if len(self.memory_flow_log) > self.max_flow_log_entries:
            self.memory_flow_log = self.memory_flow_log[-self.max_flow_log_entries//2:]
    
    async def _load_sync_checkpoints(self):
        """Load synchronization checkpoints"""
        
        # In production, this would load from persistent storage
        self.last_sync_checkpoints = {
            "episodic_memory_vltm": datetime.utcnow() - timedelta(hours=24),
            "knowledge_service_vltm": datetime.utcnow() - timedelta(hours=24),
            "vltm_knowledge_service": datetime.utcnow() - timedelta(hours=24)
        }
        
        logger.info("Loaded sync checkpoints")
    
    async def _save_sync_checkpoints(self):
        """Save synchronization checkpoints"""
        
        # In production, this would save to persistent storage
        logger.info("Saved sync checkpoints")
    
    async def trigger_manual_sync(self, source_system: str, target_system: str) -> Dict[str, Any]:
        """Trigger manual synchronization between systems"""
        
        try:
            # Find the appropriate bridge
            bridge = None
            for b in self.memory_bridges:
                if b.source_system == source_system and b.target_system == target_system:
                    bridge = b
                    break
            
            if not bridge:
                return {"error": f"No bridge found for {source_system} → {target_system}"}
            
            # Perform sync
            await self._sync_memory_bridge(bridge)
            
            return {
                "success": True,
                "source": source_system,
                "target": target_system,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in manual sync: {e}")
            return {"error": str(e)}
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        
        return {
            "is_running": self.is_running,
            "active_bridges": len([b for b in self.memory_bridges if b.enabled]),
            "running_tasks": len([t for t in self.sync_tasks.values() if not t.done()]),
            "statistics": {
                "memories_synchronized": self.integration_stats.memories_synchronized,
                "patterns_extracted": self.integration_stats.patterns_extracted,
                "knowledge_consolidated": self.integration_stats.knowledge_consolidated,
                "failed_operations": self.integration_stats.failed_operations,
                "last_sync": self.integration_stats.last_sync_timestamp.isoformat() if self.integration_stats.last_sync_timestamp else None
            },
            "memory_bridges": [
                {
                    "source": bridge.source_system,
                    "target": bridge.target_system,
                    "flow_direction": bridge.flow_direction.value,
                    "enabled": bridge.enabled,
                    "sync_interval_minutes": bridge.sync_interval_minutes
                }
                for bridge in self.memory_bridges
            ],
            "recent_memory_flows": self.memory_flow_log[-10:]  # Last 10 flows
        }
    
    async def add_memory_bridge(self, bridge: MemoryBridge) -> bool:
        """Add a new memory bridge"""
        
        try:
            # Validate bridge configuration
            if not bridge.source_system or not bridge.target_system:
                raise ValueError("Bridge must have valid source and target systems")
            
            # Add to bridges list
            self.memory_bridges.append(bridge)
            
            # Start monitoring if enabled and manager is running
            if bridge.enabled and self.is_running:
                task_name = f"sync_{bridge.source_system}_to_{bridge.target_system}"
                task = asyncio.create_task(
                    self._sync_bridge_continuously(bridge),
                    name=task_name
                )
                self.sync_tasks[task_name] = task
                
                logger.info(f"Added and started new memory bridge: {task_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding memory bridge: {e}")
            return False
    
    async def remove_memory_bridge(self, source_system: str, target_system: str) -> bool:
        """Remove a memory bridge"""
        
        try:
            # Find and remove bridge
            bridge_removed = False
            for i, bridge in enumerate(self.memory_bridges):
                if bridge.source_system == source_system and bridge.target_system == target_system:
                    del self.memory_bridges[i]
                    bridge_removed = True
                    break
            
            if not bridge_removed:
                return False
            
            # Cancel corresponding task
            task_name = f"sync_{source_system}_to_{target_system}"
            if task_name in self.sync_tasks:
                task = self.sync_tasks[task_name]
                if not task.done():
                    task.cancel()
                del self.sync_tasks[task_name]
                
                logger.info(f"Removed memory bridge: {task_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing memory bridge: {e}")
            return False