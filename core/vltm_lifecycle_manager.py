"""
Very Long-Term Memory Lifecycle Manager

This module implements the memory lifecycle management for the Snake Agent's
Very Long-Term Memory System, handling memory promotion between layers,
archival policies, and cleanup procedures.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from core.vltm_data_models import (
    VeryLongTermMemory, MemoryType, MemoryImportanceLevel,
    VLTMConfiguration, MemoryRetentionPolicy
)
from core.vltm_storage_backend import StorageBackend
from core.vltm_memory_classifier import MemoryClassifier

logger = logging.getLogger(__name__)


class MemoryLifecycleStage(Enum):
    """Stages in memory lifecycle"""
    WORKING = "working"           # Active in working memory
    SHORT_TERM = "short_term"     # Recently created, temporary
    LONG_TERM = "long_term"       # Validated, important memories
    VERY_LONG_TERM = "very_long_term"  # Strategic, persistent memories
    ARCHIVED = "archived"         # Compressed, rarely accessed
    EXPIRED = "expired"           # Ready for deletion


@dataclass
class PromotionCriteria:
    """Criteria for memory promotion between lifecycle stages"""
    min_age_days: int
    min_importance_score: float
    min_strategic_value: float
    min_access_count: int
    memory_types: List[MemoryType]
    additional_conditions: Dict[str, Any]


@dataclass
class ArchivalPolicy:
    """Policy for memory archival"""
    memory_type: MemoryType
    archive_after_days: int
    min_importance_for_retention: float
    compression_level: str  # "light", "medium", "heavy"
    delete_after_archive_days: Optional[int]


@dataclass
class LifecycleAction:
    """Represents a lifecycle action on a memory"""
    action_id: str
    memory_id: str
    action_type: str  # "promote", "archive", "delete", "compress"
    from_stage: MemoryLifecycleStage
    to_stage: MemoryLifecycleStage
    reason: str
    timestamp: datetime
    metadata: Dict[str, Any]


class MemoryLifecycleManager:
    """
    Manages the lifecycle of memories in the very long-term memory system.

    Handles promotion between memory layers, archival policies, cleanup
    procedures, and maintains optimal memory organization across the
    memory hierarchy.
    """

    def __init__(self, config: VLTMConfiguration, storage_backend: StorageBackend):
        """
        Initialize the memory lifecycle manager.

        Args:
            config: VLTM configuration
            storage_backend: Storage backend for memory operations
        """
        self.config = config
        self.storage_backend = storage_backend
        self.memory_classifier = MemoryClassifier(config)

        # Lifecycle policies
        self.promotion_criteria = self._initialize_promotion_criteria()
        self.archival_policies = self._initialize_archival_policies()

        # State tracking
        self.lifecycle_actions: List[LifecycleAction] = []
        self.lifecycle_statistics = defaultdict(int)
        self.last_lifecycle_run: Optional[datetime] = None

        # Performance settings
        self.batch_size = 500
        self.max_actions_per_run = 1000

        logger.info("Memory lifecycle manager initialized")

    def _initialize_promotion_criteria(self) -> Dict[str, PromotionCriteria]:
        """Initialize promotion criteria for different memory types"""

        criteria = {}

        # Working to Short-term (automatic, immediate)
        criteria["working_to_short_term"] = PromotionCriteria(
            min_age_days=0,
            min_importance_score=0.0,
            min_strategic_value=0.0,
            min_access_count=0,
            memory_types=list(MemoryType),
            additional_conditions={}
        )

        # Short-term to Long-term
        criteria["short_term_to_long_term"] = PromotionCriteria(
            min_age_days=1,
            min_importance_score=0.6,
            min_strategic_value=0.4,
            min_access_count=1,
            memory_types=[
                MemoryType.SUCCESSFUL_IMPROVEMENT,
                MemoryType.FAILED_EXPERIMENT,
                MemoryType.CODE_PATTERN,
                MemoryType.BEHAVIORAL_PATTERN
            ],
            additional_conditions={"validation_required": True}
        )

        # Long-term to Very Long-term
        criteria["long_term_to_very_long_term"] = PromotionCriteria(
            min_age_days=7,
            min_importance_score=0.7,
            min_strategic_value=0.6,
            min_access_count=2,
            memory_types=[
                MemoryType.STRATEGIC_KNOWLEDGE,
                MemoryType.ARCHITECTURAL_INSIGHT,
                MemoryType.EVOLUTION_PATTERN,
                MemoryType.META_LEARNING_RULE,
                MemoryType.CRITICAL_FAILURE
            ],
            additional_conditions={"pattern_validated": True}
        )

        # Direct promotion to Very Long-term (critical memories)
        criteria["direct_to_very_long_term"] = PromotionCriteria(
            min_age_days=0,
            min_importance_score=0.9,
            min_strategic_value=0.8,
            min_access_count=0,
            memory_types=[
                MemoryType.CRITICAL_FAILURE,
                MemoryType.STRATEGIC_KNOWLEDGE
            ],
            additional_conditions={"immediate_promotion": True}
        )

        return criteria

    def _initialize_archival_policies(self) -> Dict[MemoryType, ArchivalPolicy]:
        """Initialize archival policies for different memory types"""

        policies = {}

        # Strategic memories - never delete, light compression after 1 year
        policies[MemoryType.STRATEGIC_KNOWLEDGE] = ArchivalPolicy(
            memory_type=MemoryType.STRATEGIC_KNOWLEDGE,
            archive_after_days=365,
            min_importance_for_retention=0.5,
            compression_level="light",
            delete_after_archive_days=None  # Never delete
        )

        policies[MemoryType.CRITICAL_FAILURE] = ArchivalPolicy(
            memory_type=MemoryType.CRITICAL_FAILURE,
            archive_after_days=180,
            min_importance_for_retention=0.7,
            compression_level="light",
            delete_after_archive_days=None  # Never delete
        )

        policies[MemoryType.ARCHITECTURAL_INSIGHT] = ArchivalPolicy(
            memory_type=MemoryType.ARCHITECTURAL_INSIGHT,
            archive_after_days=365,
            min_importance_for_retention=0.6,
            compression_level="medium",
            delete_after_archive_days=None  # Never delete
        )

        # Tactical memories - compress and eventual deletion
        policies[MemoryType.SUCCESSFUL_IMPROVEMENT] = ArchivalPolicy(
            memory_type=MemoryType.SUCCESSFUL_IMPROVEMENT,
            archive_after_days=180,
            min_importance_for_retention=0.5,
            compression_level="medium",
            delete_after_archive_days=730  # Delete after 2 years
        )

        policies[MemoryType.FAILED_EXPERIMENT] = ArchivalPolicy(
            memory_type=MemoryType.FAILED_EXPERIMENT,
            archive_after_days=90,
            min_importance_for_retention=0.4,
            compression_level="heavy",
            delete_after_archive_days=365  # Delete after 1 year
        )

        policies[MemoryType.CODE_PATTERN] = ArchivalPolicy(
            memory_type=MemoryType.CODE_PATTERN,
            archive_after_days=180,
            min_importance_for_retention=0.4,
            compression_level="medium",
            delete_after_archive_days=365  # Delete after 1 year
        )

        policies[MemoryType.BEHAVIORAL_PATTERN] = ArchivalPolicy(
            memory_type=MemoryType.BEHAVIORAL_PATTERN,
            archive_after_days=120,
            min_importance_for_retention=0.3,
            compression_level="medium",
            delete_after_archive_days=180  # Delete after 6 months
        )

        return policies

    async def run_lifecycle_management(self) -> Dict[str, Any]:
        """
        Run a complete lifecycle management cycle.

        Returns:
            Results of the lifecycle management run
        """
        try:
            start_time = datetime.utcnow()
            logger.info("Starting memory lifecycle management run")

            results = {
                "promoted_memories": 0,
                "archived_memories": 0,
                "deleted_memories": 0,
                "compressed_memories": 0,
                "actions_taken": [],
                "processing_time_seconds": 0.0,
                "success": True,
                "error_message": None
            }

            # Step 1: Promote memories between lifecycle stages
            promotion_results = await self._promote_memories()
            results["promoted_memories"] = promotion_results["total_promoted"]
            results["actions_taken"].extend(promotion_results["actions"])

            # Step 2: Archive old memories according to policies
            archival_results = await self._archive_memories()
            results["archived_memories"] = archival_results["total_archived"]
            results["actions_taken"].extend(archival_results["actions"])

            # Step 3: Compress archived memories
            compression_results = await self._compress_memories()
            results["compressed_memories"] = compression_results["total_compressed"]
            results["actions_taken"].extend(compression_results["actions"])

            # Step 4: Delete expired memories
            deletion_results = await self._delete_expired_memories()
            results["deleted_memories"] = deletion_results["total_deleted"]
            results["actions_taken"].extend(deletion_results["actions"])

            # Step 5: Update lifecycle statistics
            await self._update_lifecycle_statistics(results)

            self.last_lifecycle_run = start_time
            results["processing_time_seconds"] = (
                datetime.utcnow() - start_time).total_seconds()

            logger.info(f"Lifecycle management completed: "
                        f"{results['promoted_memories']} promoted, "
                        f"{results['archived_memories']} archived, "
                        f"{results['compressed_memories']} compressed, "
                        f"{results['deleted_memories']} deleted")

            return results

        except Exception as e:
            logger.error(f"Error in lifecycle management: {e}", exc_info=True)
            return {
                "promoted_memories": 0,
                "archived_memories": 0,
                "deleted_memories": 0,
                "compressed_memories": 0,
                "actions_taken": [],
                "processing_time_seconds": 0.0,
                "success": False,
                "error_message": str(e)
            }

    async def _promote_memories(self) -> Dict[str, Any]:
        """Promote memories between lifecycle stages"""

        try:
            promoted_count = 0
            actions = []

            # Get recent memories for promotion evaluation
            recent_memories = await self.storage_backend.retrieve_recent_memories(
                hours=168, limit=self.batch_size  # Last week
            )

            for memory in recent_memories:
                promotion_action = await self._evaluate_memory_promotion(memory)

                if promotion_action:
                    success = await self._execute_promotion_action(promotion_action)
                    if success:
                        promoted_count += 1
                        actions.append(promotion_action)
                        self.lifecycle_actions.append(promotion_action)

            return {
                "total_promoted": promoted_count,
                "actions": actions
            }

        except Exception as e:
            logger.error(f"Error promoting memories: {e}")
            return {"total_promoted": 0, "actions": []}

    async def _evaluate_memory_promotion(self, memory: VeryLongTermMemory) -> Optional[LifecycleAction]:
        """Evaluate if a memory should be promoted"""

        try:
            memory_age = datetime.utcnow() - memory.created_at

            # Check direct promotion to very long-term for critical memories
            direct_criteria = self.promotion_criteria["direct_to_very_long_term"]
            if (memory.memory_type in direct_criteria.memory_types and
                memory.importance_score >= direct_criteria.min_importance_score and
                    memory.strategic_value >= direct_criteria.min_strategic_value):

                return LifecycleAction(
                    action_id=str(uuid.uuid4()),
                    memory_id=memory.memory_id,
                    action_type="promote",
                    from_stage=MemoryLifecycleStage.SHORT_TERM,
                    to_stage=MemoryLifecycleStage.VERY_LONG_TERM,
                    reason="Direct promotion - critical memory",
                    timestamp=datetime.utcnow(),
                    metadata={
                        "importance_score": memory.importance_score,
                        "strategic_value": memory.strategic_value,
                        "memory_type": memory.memory_type
                    }
                )

            # Check long-term to very long-term promotion
            vlt_criteria = self.promotion_criteria["long_term_to_very_long_term"]
            if (memory_age.days >= vlt_criteria.min_age_days and
                memory.memory_type in vlt_criteria.memory_types and
                memory.importance_score >= vlt_criteria.min_importance_score and
                memory.strategic_value >= vlt_criteria.min_strategic_value and
                    memory.access_count >= vlt_criteria.min_access_count):

                return LifecycleAction(
                    action_id=str(uuid.uuid4()),
                    memory_id=memory.memory_id,
                    action_type="promote",
                    from_stage=MemoryLifecycleStage.LONG_TERM,
                    to_stage=MemoryLifecycleStage.VERY_LONG_TERM,
                    reason="Age and importance criteria met",
                    timestamp=datetime.utcnow(),
                    metadata={
                        "age_days": memory_age.days,
                        "importance_score": memory.importance_score,
                        "strategic_value": memory.strategic_value,
                        "access_count": memory.access_count
                    }
                )

            # Check short-term to long-term promotion
            lt_criteria = self.promotion_criteria["short_term_to_long_term"]
            if (memory_age.days >= lt_criteria.min_age_days and
                memory.memory_type in lt_criteria.memory_types and
                memory.importance_score >= lt_criteria.min_importance_score and
                    memory.strategic_value >= lt_criteria.min_strategic_value):

                return LifecycleAction(
                    action_id=str(uuid.uuid4()),
                    memory_id=memory.memory_id,
                    action_type="promote",
                    from_stage=MemoryLifecycleStage.SHORT_TERM,
                    to_stage=MemoryLifecycleStage.LONG_TERM,
                    reason="Validation period passed",
                    timestamp=datetime.utcnow(),
                    metadata={
                        "age_days": memory_age.days,
                        "importance_score": memory.importance_score,
                        "strategic_value": memory.strategic_value
                    }
                )

            return None

        except Exception as e:
            logger.error(
                f"Error evaluating promotion for memory {memory.memory_id}: {e}")
            return None

    async def _execute_promotion_action(self, action: LifecycleAction) -> bool:
        """Execute a memory promotion action"""

        try:
            # For now, promotion is mainly about metadata updates
            # In a full implementation, this would involve moving data between storage tiers

            logger.info(
                f"Promoting memory {action.memory_id} from {action.from_stage} to {action.to_stage}")

            # Update memory metadata to reflect new lifecycle stage
            # This is a simplified implementation
            return True

        except Exception as e:
            logger.error(
                f"Error executing promotion action {action.action_id}: {e}")
            return False

    async def _archive_memories(self) -> Dict[str, Any]:
        """Archive old memories according to archival policies"""

        try:
            archived_count = 0
            actions = []

            for memory_type, policy in self.archival_policies.items():
                # Get memories of this type that are candidates for archival
                type_memories = await self.storage_backend.retrieve_memories_by_type(
                    memory_type, limit=200, min_importance=0.0
                )

                cutoff_date = datetime.utcnow() - timedelta(days=policy.archive_after_days)

                for memory in type_memories:
                    if (memory.created_at <= cutoff_date and
                            memory.importance_score >= policy.min_importance_for_retention):

                        action = LifecycleAction(
                            action_id=str(uuid.uuid4()),
                            memory_id=memory.memory_id,
                            action_type="archive",
                            from_stage=MemoryLifecycleStage.VERY_LONG_TERM,
                            to_stage=MemoryLifecycleStage.ARCHIVED,
                            reason=f"Archival policy - age {(datetime.utcnow() - memory.created_at).days} days",
                            timestamp=datetime.utcnow(),
                            metadata={
                                "policy": policy.memory_type,
                                "compression_level": policy.compression_level,
                                "age_days": (datetime.utcnow() - memory.created_at).days
                            }
                        )

                        success = await self._execute_archival_action(action, policy)
                        if success:
                            archived_count += 1
                            actions.append(action)
                            self.lifecycle_actions.append(action)

            return {
                "total_archived": archived_count,
                "actions": actions
            }

        except Exception as e:
            logger.error(f"Error archiving memories: {e}")
            return {"total_archived": 0, "actions": []}

    async def _execute_archival_action(self, action: LifecycleAction, policy: ArchivalPolicy) -> bool:
        """Execute a memory archival action"""

        try:
            # Archive the memory with appropriate compression
            logger.info(
                f"Archiving memory {action.memory_id} with {policy.compression_level} compression")

            # In a full implementation, this would:
            # 1. Compress the memory content
            # 2. Move to archive storage
            # 3. Update metadata
            # 4. Remove from active storage if needed

            return True

        except Exception as e:
            logger.error(
                f"Error executing archival action {action.action_id}: {e}")
            return False

    async def _compress_memories(self) -> Dict[str, Any]:
        """Compress memories in archive storage"""

        try:
            compressed_count = 0
            actions = []

            # Get archived memories that need compression
            # This is a simplified implementation
            # In practice, would query archived memories needing compression

            return {
                "total_compressed": compressed_count,
                "actions": actions
            }

        except Exception as e:
            logger.error(f"Error compressing memories: {e}")
            return {"total_compressed": 0, "actions": []}

    async def _delete_expired_memories(self) -> Dict[str, Any]:
        """Delete memories that have exceeded their retention period"""

        try:
            deleted_count = 0
            actions = []

            for memory_type, policy in self.archival_policies.items():
                if policy.delete_after_archive_days is None:
                    continue  # Never delete these memories

                # Calculate deletion cutoff date
                deletion_cutoff = datetime.utcnow() - timedelta(
                    days=policy.archive_after_days + policy.delete_after_archive_days
                )

                # Get memories of this type for deletion evaluation
                type_memories = await self.storage_backend.retrieve_memories_by_type(
                    memory_type, limit=100, min_importance=0.0
                )

                for memory in type_memories:
                    if (memory.created_at <= deletion_cutoff and
                            memory.importance_score < policy.min_importance_for_retention):

                        action = LifecycleAction(
                            action_id=str(uuid.uuid4()),
                            memory_id=memory.memory_id,
                            action_type="delete",
                            from_stage=MemoryLifecycleStage.ARCHIVED,
                            to_stage=MemoryLifecycleStage.EXPIRED,
                            reason=f"Retention period expired - {(datetime.utcnow() - memory.created_at).days} days old",
                            timestamp=datetime.utcnow(),
                            metadata={
                                "policy": policy.memory_type,
                                "age_days": (datetime.utcnow() - memory.created_at).days,
                                "importance_score": memory.importance_score
                            }
                        )

                        success = await self._execute_deletion_action(action)
                        if success:
                            deleted_count += 1
                            actions.append(action)
                            self.lifecycle_actions.append(action)

            return {
                "total_deleted": deleted_count,
                "actions": actions
            }

        except Exception as e:
            logger.error(f"Error deleting expired memories: {e}")
            return {"total_deleted": 0, "actions": []}

    async def _execute_deletion_action(self, action: LifecycleAction) -> bool:
        """Execute a memory deletion action"""

        try:
            logger.info(f"Deleting expired memory {action.memory_id}")

            # In a full implementation, this would:
            # 1. Remove memory from all storage locations
            # 2. Remove related patterns and references
            # 3. Update indices
            # 4. Log the deletion

            return True

        except Exception as e:
            logger.error(
                f"Error executing deletion action {action.action_id}: {e}")
            return False

    async def _update_lifecycle_statistics(self, results: Dict[str, Any]):
        """Update lifecycle management statistics"""

        self.lifecycle_statistics["total_runs"] += 1
        self.lifecycle_statistics["total_promoted"] += results["promoted_memories"]
        self.lifecycle_statistics["total_archived"] += results["archived_memories"]
        self.lifecycle_statistics["total_compressed"] += results["compressed_memories"]
        self.lifecycle_statistics["total_deleted"] += results["deleted_memories"]

        if results["success"]:
            self.lifecycle_statistics["successful_runs"] += 1
        else:
            self.lifecycle_statistics["failed_runs"] += 1

    # Public interface methods

    async def promote_memory_manually(self, memory_id: str, to_stage: MemoryLifecycleStage, reason: str) -> bool:
        """Manually promote a memory to a specific lifecycle stage"""

        try:
            memory = await self.storage_backend.retrieve_memory(memory_id)
            if not memory:
                logger.warning(
                    f"Memory {memory_id} not found for manual promotion")
                return False

            action = LifecycleAction(
                action_id=str(uuid.uuid4()),
                memory_id=memory_id,
                action_type="promote",
                from_stage=MemoryLifecycleStage.LONG_TERM,  # Assume current stage
                to_stage=to_stage,
                reason=f"Manual promotion: {reason}",
                timestamp=datetime.utcnow(),
                metadata={"manual": True, "reason": reason}
            )

            success = await self._execute_promotion_action(action)
            if success:
                self.lifecycle_actions.append(action)
                logger.info(
                    f"Manually promoted memory {memory_id} to {to_stage}")

            return success

        except Exception as e:
            logger.error(
                f"Error in manual promotion of memory {memory_id}: {e}")
            return False

    def get_lifecycle_statistics(self) -> Dict[str, Any]:
        """Get lifecycle management statistics"""

        return {
            "statistics": dict(self.lifecycle_statistics),
            "last_run": self.last_lifecycle_run.isoformat() if self.last_lifecycle_run else None,
            "promotion_criteria_count": len(self.promotion_criteria),
            "archival_policies_count": len(self.archival_policies),
            "recent_actions": len([a for a in self.lifecycle_actions if
                                   (datetime.utcnow() - a.timestamp).days <= 7]),
            "configuration": {
                "batch_size": self.batch_size,
                "max_actions_per_run": self.max_actions_per_run
            }
        }

    def get_recent_actions(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent lifecycle actions"""

        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        recent_actions = [
            {
                "action_id": action.action_id,
                "memory_id": action.memory_id,
                "action_type": action.action_type,
                "from_stage": action.from_stage.value,
                "to_stage": action.to_stage.value,
                "reason": action.reason,
                "timestamp": action.timestamp.isoformat(),
                "metadata": action.metadata
            }
            for action in self.lifecycle_actions
            if action.timestamp >= cutoff_time
        ]

        return recent_actions

    def get_memory_stage_distribution(self) -> Dict[str, int]:
        """Get distribution of memories across lifecycle stages"""

        # This would require querying the storage backend for memory counts
        # by lifecycle stage. For now, return a placeholder
        return {
            "working": 0,
            "short_term": 0,
            "long_term": 0,
            "very_long_term": 0,
            "archived": 0
        }
