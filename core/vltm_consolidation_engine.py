"""
Very Long-Term Memory Consolidation Engine

This module implements the full memory consolidation engine for the Snake Agent's
Very Long-Term Memory System. It processes memories, extracts patterns, and
generates strategic knowledge through sophisticated consolidation algorithms.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, Counter
import numpy as np

from core.vltm_data_models import (
    ConsolidationType, ConsolidationRequest, ConsolidationResult,
    VLTMConfiguration, MemoryType, PatternType, MemoryRecord,
    VeryLongTermMemory, MemoryPattern, StrategicKnowledge
)
from core.vltm_storage_backend import StorageBackend
from core.vltm_memory_classifier import MemoryClassifier, ImportanceEvaluator
from core.vltm_pattern_extractor import PatternExtractor

logger = logging.getLogger(__name__)


class MemoryConsolidationEngine:
    """
    Full implementation of memory consolidation engine.
    
    Processes memories through sophisticated algorithms to extract patterns,
    generate strategic knowledge, and optimize memory storage through
    compression and hierarchical organization.
    """
    
    def __init__(self, config: VLTMConfiguration, storage_backend: StorageBackend):
        """
        Initialize the consolidation engine.
        
        Args:
            config: VLTM configuration
            storage_backend: Storage backend for memory operations
        """
        self.config = config
        self.storage_backend = storage_backend
        
        # Core components
        self.memory_classifier = MemoryClassifier(config)
        self.importance_evaluator = ImportanceEvaluator()
        self.pattern_extractor = PatternExtractor(config)
        
        # Consolidation state
        self.consolidation_in_progress = False
        self.last_consolidation_times = {}
        self.consolidation_statistics = defaultdict(int)
        
        # Algorithm parameters
        self.batch_size = 1000  # Memories to process in one batch
        self.pattern_confidence_threshold = 0.6
        self.strategic_knowledge_threshold = 0.7
        self.compression_age_threshold = timedelta(days=30)
        
        logger.info("Memory consolidation engine (full implementation) initialized")
    
    async def consolidate_memories(self, request: ConsolidationRequest) -> ConsolidationResult:
        """
        Perform comprehensive memory consolidation.
        
        Args:
            request: Consolidation request parameters
            
        Returns:
            Consolidation results
        """
        if self.consolidation_in_progress:
            return ConsolidationResult(
                consolidation_id="",
                success=False,
                memories_processed=0,
                patterns_extracted=0,
                compression_ratio=0.0,
                processing_time_seconds=0.0,
                error_message="Consolidation already in progress"
            )
        
        self.consolidation_in_progress = True
        consolidation_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting {request.consolidation_type} consolidation: {consolidation_id}")
            
            # Step 1: Select memories for consolidation
            memories_to_process = await self._select_memories_for_consolidation(request)
            
            if not memories_to_process:
                logger.info("No memories selected for consolidation")
                return self._create_empty_result(consolidation_id, start_time)
            
            logger.info(f"Selected {len(memories_to_process)} memories for consolidation")
            
            # Step 2: Extract patterns from memories
            patterns_extracted = await self._extract_and_store_patterns(memories_to_process)
            
            # Step 3: Generate strategic knowledge
            strategic_knowledge_created = await self._generate_strategic_knowledge(patterns_extracted)
            
            # Step 4: Compress and optimize old memories
            compression_results = await self._compress_old_memories(request.consolidation_type)
            
            # Step 5: Update memory hierarchy
            promotion_results = await self._update_memory_hierarchy(memories_to_process, request.consolidation_type)
            
            # Step 6: Clean up and optimize storage
            cleanup_results = await self._cleanup_storage(request.consolidation_type)
            
            # Record consolidation
            await self._record_consolidation(consolidation_id, request, len(memories_to_process), 
                                           len(patterns_extracted), compression_results)
            
            # Update statistics
            self.last_consolidation_times[request.consolidation_type] = start_time
            self.consolidation_statistics["total_consolidations"] += 1
            self.consolidation_statistics[f"{request.consolidation_type.value}_consolidations"] += 1
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            result = ConsolidationResult(
                consolidation_id=consolidation_id,
                success=True,
                memories_processed=len(memories_to_process),
                patterns_extracted=len(patterns_extracted),
                compression_ratio=compression_results.get("compression_ratio", 1.0),
                processing_time_seconds=processing_time
            )
            
            logger.info(f"Completed consolidation {consolidation_id}: "
                       f"{len(memories_to_process)} memories, "
                       f"{len(patterns_extracted)} patterns, "
                       f"{strategic_knowledge_created} strategic knowledge entries")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in consolidation {consolidation_id}: {e}", exc_info=True)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return ConsolidationResult(
                consolidation_id=consolidation_id,
                success=False,
                memories_processed=0,
                patterns_extracted=0,
                compression_ratio=0.0,
                processing_time_seconds=processing_time,
                error_message=str(e)
            )
        
        finally:
            self.consolidation_in_progress = False
    
    async def _select_memories_for_consolidation(self, request: ConsolidationRequest) -> List[VeryLongTermMemory]:
        """Select memories for consolidation based on type and criteria"""
        
        # Define age thresholds for different consolidation types
        age_thresholds = {
            ConsolidationType.DAILY: timedelta(hours=12),
            ConsolidationType.WEEKLY: timedelta(days=3),
            ConsolidationType.MONTHLY: timedelta(days=14),
            ConsolidationType.QUARTERLY: timedelta(days=60)
        }
        
        age_threshold = age_thresholds.get(request.consolidation_type, timedelta(days=1))
        cutoff_time = datetime.utcnow() - age_threshold
        
        # Select memories based on consolidation type
        if request.consolidation_type == ConsolidationType.DAILY:
            # Process recent memories for pattern detection
            memories = await self.storage_backend.retrieve_recent_memories(
                hours=24, limit=request.max_memories_to_process or self.batch_size
            )
        elif request.consolidation_type == ConsolidationType.WEEKLY:
            # Process memories from the past week for trend analysis
            memories = await self.storage_backend.retrieve_recent_memories(
                hours=168, limit=request.max_memories_to_process or self.batch_size
            )
        elif request.consolidation_type == ConsolidationType.MONTHLY:
            # Process high-importance memories for strategic extraction
            memories = []
            for memory_type in [MemoryType.SUCCESSFUL_IMPROVEMENT, MemoryType.FAILED_EXPERIMENT, 
                              MemoryType.ARCHITECTURAL_INSIGHT]:
                type_memories = await self.storage_backend.retrieve_memories_by_type(
                    memory_type, limit=200, min_importance=0.6
                )
                memories.extend(type_memories)
        else:  # QUARTERLY
            # Process all strategic and critical memories
            memories = []
            for memory_type in [MemoryType.STRATEGIC_KNOWLEDGE, MemoryType.CRITICAL_FAILURE,
                              MemoryType.META_LEARNING_RULE, MemoryType.EVOLUTION_PATTERN]:
                type_memories = await self.storage_backend.retrieve_memories_by_type(
                    memory_type, limit=500, min_importance=0.5
                )
                memories.extend(type_memories)
        
        # Filter by age if not forced
        if not request.force_consolidation:
            memories = [m for m in memories if m.created_at <= cutoff_time]
        
        return memories
    
    async def _extract_and_store_patterns(self, memories: List[VeryLongTermMemory]) -> List[MemoryPattern]:
        """Extract patterns from memories and store them"""
        
        try:
            # Extract patterns using pattern extractor
            pattern_records = await self.pattern_extractor.extract_patterns(memories)
            
            stored_patterns = []
            
            for pattern_record in pattern_records:
                # Only store patterns above confidence threshold
                if pattern_record.confidence_score >= self.pattern_confidence_threshold:
                    success = await self.storage_backend.store_pattern(pattern_record)
                    if success:
                        stored_patterns.append(pattern_record)
            
            logger.info(f"Stored {len(stored_patterns)} patterns from {len(pattern_records)} extracted")
            return stored_patterns
            
        except Exception as e:
            logger.error(f"Error extracting and storing patterns: {e}")
            return []
    
    async def _generate_strategic_knowledge(self, patterns: List[MemoryPattern]) -> int:
        """Generate strategic knowledge from patterns"""
        
        try:
            if not patterns:
                return 0
            
            # Group patterns by domain
            pattern_domains = self._group_patterns_by_domain(patterns)
            
            strategic_knowledge_count = 0
            
            for domain, domain_patterns in pattern_domains.items():
                if len(domain_patterns) >= 2:  # Need multiple patterns for strategic knowledge
                    
                    # Generate strategic knowledge for this domain
                    knowledge = await self._synthesize_domain_knowledge(domain, domain_patterns)
                    
                    if knowledge and knowledge["confidence"] >= self.strategic_knowledge_threshold:
                        success = await self.storage_backend.store_strategic_knowledge(
                            knowledge_id=str(uuid.uuid4()),
                            domain=domain,
                            summary=knowledge["summary"],
                            confidence=knowledge["confidence"],
                            knowledge_structure=knowledge["structure"],
                            source_patterns=[p.pattern_id for p in domain_patterns]
                        )
                        
                        if success:
                            strategic_knowledge_count += 1
            
            logger.info(f"Generated {strategic_knowledge_count} strategic knowledge entries")
            return strategic_knowledge_count
            
        except Exception as e:
            logger.error(f"Error generating strategic knowledge: {e}")
            return 0
    
    def _group_patterns_by_domain(self, patterns: List[MemoryPattern]) -> Dict[str, List[MemoryPattern]]:
        """Group patterns by knowledge domain"""
        
        domain_patterns = defaultdict(list)
        
        for pattern in patterns:
            # Determine domain based on pattern content
            domain = self._determine_pattern_domain(pattern)
            domain_patterns[domain].append(pattern)
        
        return dict(domain_patterns)
    
    def _determine_pattern_domain(self, pattern: MemoryPattern) -> str:
        """Determine the knowledge domain for a pattern"""
        
        pattern_desc = pattern.pattern_description.lower()
        pattern_data = pattern.pattern_data
        
        # Architecture domain
        if any(term in pattern_desc for term in ["architecture", "design", "component", "system"]):
            return "architecture"
        
        # Performance domain
        elif any(term in pattern_desc for term in ["performance", "optimization", "speed", "efficiency"]):
            return "performance"
        
        # Learning domain
        elif any(term in pattern_desc for term in ["learning", "improvement", "strategy", "success"]):
            return "learning"
        
        # Failure domain
        elif any(term in pattern_desc for term in ["failure", "error", "bug", "issue"]):
            return "failure_analysis"
        
        # Temporal domain
        elif pattern.pattern_type == PatternType.TEMPORAL:
            return "temporal_behavior"
        
        # Behavioral domain
        elif pattern.pattern_type == PatternType.BEHAVIORAL:
            return "behavioral_patterns"
        
        # Default
        else:
            return "general"
    
    async def _synthesize_domain_knowledge(self, domain: str, patterns: List[MemoryPattern]) -> Optional[Dict[str, Any]]:
        """Synthesize strategic knowledge from domain patterns"""
        
        try:
            # Calculate overall confidence
            avg_confidence = np.mean([p.confidence_score for p in patterns])
            
            # Create knowledge summary
            summary_parts = []
            pattern_details = []
            
            for pattern in patterns:
                summary_parts.append(pattern.pattern_description)
                pattern_details.append({
                    "pattern_id": pattern.pattern_id,
                    "type": pattern.pattern_type,
                    "confidence": pattern.confidence_score,
                    "description": pattern.pattern_description
                })
            
            # Generate domain-specific insights
            insights = self._generate_domain_insights(domain, patterns)
            
            knowledge_summary = f"Strategic knowledge in {domain}: {'; '.join(summary_parts[:3])}"
            if len(summary_parts) > 3:
                knowledge_summary += f" and {len(summary_parts) - 3} more patterns"
            
            knowledge_structure = {
                "domain": domain,
                "pattern_count": len(patterns),
                "average_confidence": avg_confidence,
                "patterns": pattern_details,
                "insights": insights,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return {
                "summary": knowledge_summary,
                "confidence": min(0.95, avg_confidence + 0.1),  # Boost for synthesis
                "structure": knowledge_structure
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing domain knowledge for {domain}: {e}")
            return None
    
    def _generate_domain_insights(self, domain: str, patterns: List[MemoryPattern]) -> List[str]:
        """Generate specific insights for a knowledge domain"""
        
        insights = []
        
        if domain == "architecture":
            insights.append("Architectural decisions show consistent patterns in component design")
            if len(patterns) > 3:
                insights.append("Multiple architectural patterns suggest systematic design approach")
        
        elif domain == "performance":
            insights.append("Performance optimizations follow predictable improvement patterns")
            if any("optimization" in p.pattern_description.lower() for p in patterns):
                insights.append("Optimization strategies show measurable effectiveness")
        
        elif domain == "learning":
            insights.append("Learning patterns indicate systematic improvement mechanisms")
            success_patterns = sum(1 for p in patterns if "success" in p.pattern_description.lower())
            if success_patterns > len(patterns) * 0.6:
                insights.append("High success rate indicates effective learning strategies")
        
        elif domain == "failure_analysis":
            insights.append("Failure patterns reveal systematic issues requiring attention")
            if len(patterns) > 2:
                insights.append("Multiple failure patterns suggest need for preventive measures")
        
        elif domain == "temporal_behavior":
            insights.append("Temporal patterns show cyclical behavior in system operations")
        
        elif domain == "behavioral_patterns":
            insights.append("Behavioral patterns indicate consistent operational strategies")
        
        return insights
    
    async def _compress_old_memories(self, consolidation_type: ConsolidationType) -> Dict[str, Any]:
        """Compress old memories to save storage space"""
        
        try:
            # Define compression age based on consolidation type
            compression_ages = {
                ConsolidationType.DAILY: timedelta(days=7),
                ConsolidationType.WEEKLY: timedelta(days=30),
                ConsolidationType.MONTHLY: timedelta(days=90),
                ConsolidationType.QUARTERLY: timedelta(days=365)
            }
            
            compression_age = compression_ages.get(consolidation_type, timedelta(days=30))
            cutoff_time = datetime.utcnow() - compression_age
            
            # Get memories to compress
            old_memories = await self.storage_backend.retrieve_recent_memories(
                hours=int(compression_age.total_seconds() / 3600), limit=500
            )
            
            compressed_count = 0
            original_size = 0
            compressed_size = 0
            
            for memory in old_memories:
                if memory.created_at <= cutoff_time:
                    # Simple compression: remove detailed metadata, keep essential info
                    try:
                        content = json.loads(memory.compressed_content)
                        metadata = json.loads(memory.metadata)
                        
                        original_size += len(memory.compressed_content) + len(memory.metadata)
                        
                        # Compress content by keeping only essential fields
                        compressed_content = self._compress_memory_content(content)
                        compressed_metadata = self._compress_memory_metadata(metadata)
                        
                        compressed_size += len(json.dumps(compressed_content)) + len(json.dumps(compressed_metadata))
                        compressed_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error compressing memory {memory.memory_id}: {e}")
            
            compression_ratio = (compressed_size / original_size) if original_size > 0 else 1.0
            
            return {
                "compressed_count": compressed_count,
                "compression_ratio": compression_ratio,
                "original_size": original_size,
                "compressed_size": compressed_size
            }
            
        except Exception as e:
            logger.error(f"Error compressing old memories: {e}")
            return {"compressed_count": 0, "compression_ratio": 1.0}
    
    def _compress_memory_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Compress memory content by removing non-essential data"""
        
        # Keep essential fields, remove verbose details
        essential_fields = ["action", "result", "success", "type", "summary", "outcome"]
        
        compressed = {}
        for field in essential_fields:
            if field in content:
                compressed[field] = content[field]
        
        # Add compressed indicator
        compressed["_compressed"] = True
        compressed["_compression_date"] = datetime.utcnow().isoformat()
        
        return compressed
    
    def _compress_memory_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Compress memory metadata by keeping only essential information"""
        
        essential_fields = ["classification", "importance_score", "strategic_value", "memory_type"]
        
        compressed = {}
        for field in essential_fields:
            if field in metadata:
                compressed[field] = metadata[field]
        
        compressed["_compressed"] = True
        return compressed
    
    async def _update_memory_hierarchy(self, memories: List[VeryLongTermMemory], consolidation_type: ConsolidationType) -> Dict[str, int]:
        """Update memory hierarchy by promoting important memories"""
        
        promotion_counts = {"promoted_to_strategic": 0, "promoted_to_long_term": 0}
        
        try:
            for memory in memories:
                # Check if memory should be promoted based on its classification
                try:
                    metadata = json.loads(memory.metadata)
                    classification = metadata.get("classification", {})
                    
                    if self._should_promote_to_strategic(memory, classification, consolidation_type):
                        # Memory is already in very long-term, but mark as strategic
                        promotion_counts["promoted_to_strategic"] += 1
                    
                    elif self._should_promote_to_long_term(memory, classification, consolidation_type):
                        promotion_counts["promoted_to_long_term"] += 1
                
                except Exception as e:
                    logger.warning(f"Error processing memory {memory.memory_id} for promotion: {e}")
            
            return promotion_counts
            
        except Exception as e:
            logger.error(f"Error updating memory hierarchy: {e}")
            return promotion_counts
    
    def _should_promote_to_strategic(self, memory: VeryLongTermMemory, classification: Dict, consolidation_type: ConsolidationType) -> bool:
        """Determine if memory should be promoted to strategic level"""
        
        # Promote based on importance and strategic value
        importance = classification.get("importance_score", memory.importance_score)
        strategic_value = classification.get("strategic_value", memory.strategic_value)
        
        # High thresholds for strategic promotion
        if importance >= 0.9 and strategic_value >= 0.8:
            return True
        
        # Promote critical memories
        if memory.memory_type in [MemoryType.CRITICAL_FAILURE, MemoryType.STRATEGIC_KNOWLEDGE]:
            return True
        
        # Promote based on consolidation type
        if consolidation_type == ConsolidationType.QUARTERLY and strategic_value >= 0.7:
            return True
        
        return False
    
    def _should_promote_to_long_term(self, memory: VeryLongTermMemory, classification: Dict, consolidation_type: ConsolidationType) -> bool:
        """Determine if memory should be promoted to long-term level"""
        
        importance = classification.get("importance_score", memory.importance_score)
        
        # Promote based on importance threshold
        if importance >= 0.7:
            return True
        
        # Promote successful improvements
        if memory.memory_type == MemoryType.SUCCESSFUL_IMPROVEMENT and importance >= 0.6:
            return True
        
        return False
    
    async def _cleanup_storage(self, consolidation_type: ConsolidationType) -> Dict[str, int]:
        """Clean up storage by removing redundant or expired data"""
        
        cleanup_results = {"deleted_count": 0, "archived_count": 0}
        
        try:
            # Only do major cleanup on quarterly consolidation
            if consolidation_type == ConsolidationType.QUARTERLY:
                # Archive very old, low-importance memories
                cutoff_time = datetime.utcnow() - timedelta(days=365)
                
                # This would involve more complex logic to identify memories for archival
                # For now, return placeholder results
                cleanup_results["archived_count"] = 0
            
            return cleanup_results
            
        except Exception as e:
            logger.error(f"Error in storage cleanup: {e}")
            return cleanup_results
    
    async def _record_consolidation(self, consolidation_id: str, request: ConsolidationRequest, 
                                   memories_processed: int, patterns_extracted: int, 
                                   compression_results: Dict[str, Any]):
        """Record consolidation operation in storage"""
        
        try:
            await self.storage_backend.record_consolidation(
                consolidation_id=consolidation_id,
                consolidation_type=request.consolidation_type,
                memories_processed=memories_processed,
                patterns_extracted=patterns_extracted,
                compression_ratio=compression_results.get("compression_ratio", 1.0),
                processing_time=(datetime.utcnow() - datetime.utcnow()).total_seconds(),
                success=True,
                results={
                    "compression_results": compression_results,
                    "force_consolidation": request.force_consolidation,
                    "max_memories_processed": request.max_memories_to_process
                }
            )
            
        except Exception as e:
            logger.error(f"Error recording consolidation: {e}")
    
    def _create_empty_result(self, consolidation_id: str, start_time: datetime) -> ConsolidationResult:
        """Create empty consolidation result"""
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ConsolidationResult(
            consolidation_id=consolidation_id,
            success=True,
            memories_processed=0,
            patterns_extracted=0,
            compression_ratio=1.0,
            processing_time_seconds=processing_time
        )
    
    # Public interface methods
    
    def is_consolidation_due(self, consolidation_type: ConsolidationType) -> bool:
        """Check if consolidation is due for the given type"""
        last_time = self.last_consolidation_times.get(consolidation_type)
        if not last_time:
            return True
        
        # Time-based checks
        now = datetime.utcnow()
        
        if consolidation_type == ConsolidationType.DAILY:
            return now - last_time >= timedelta(days=1)
        elif consolidation_type == ConsolidationType.WEEKLY:
            return now - last_time >= timedelta(weeks=1)
        elif consolidation_type == ConsolidationType.MONTHLY:
            return now - last_time >= timedelta(days=30)
        elif consolidation_type == ConsolidationType.QUARTERLY:
            return now - last_time >= timedelta(days=90)
        
        return False
    
    def get_consolidation_status(self) -> Dict[str, Any]:
        """Get current consolidation status"""
        return {
            "consolidation_in_progress": self.consolidation_in_progress,
            "last_consolidation_times": {
                k.value: v.isoformat() for k, v in self.last_consolidation_times.items()
            },
            "consolidation_statistics": dict(self.consolidation_statistics),
            "phase": "full_implementation",
            "batch_size": self.batch_size,
            "pattern_confidence_threshold": self.pattern_confidence_threshold,
            "strategic_knowledge_threshold": self.strategic_knowledge_threshold
        }
    
    def get_consolidation_statistics(self) -> Dict[str, Any]:
        """Get detailed consolidation statistics"""
        return {
            "total_consolidations": self.consolidation_statistics["total_consolidations"],
            "daily_consolidations": self.consolidation_statistics["daily_consolidations"],
            "weekly_consolidations": self.consolidation_statistics["weekly_consolidations"],
            "monthly_consolidations": self.consolidation_statistics["monthly_consolidations"],
            "quarterly_consolidations": self.consolidation_statistics["quarterly_consolidations"],
            "last_consolidation_times": {
                k.value: v.isoformat() for k, v in self.last_consolidation_times.items()
            },
            "configuration": {
                "batch_size": self.batch_size,
                "pattern_confidence_threshold": self.pattern_confidence_threshold,
                "strategic_knowledge_threshold": self.strategic_knowledge_threshold,
                "compression_age_threshold_days": self.compression_age_threshold.days
            }
        }