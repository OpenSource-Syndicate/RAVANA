"""
Very Long-Term Memory Compression Engine

This module implements compression strategies for aging memories,
including pattern abstraction and lossy compression for long-term storage.
"""

import asyncio
import json
import logging
import uuid
import zlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.vltm_storage_backend import StorageBackend
from core.vltm_data_models import MemoryType, MemoryImportance, VLTMConfiguration

logger = logging.getLogger(__name__)


class CompressionLevel(str, Enum):
    """Levels of compression for memory aging"""
    NONE = "none"
    LIGHT = "light"          # Lossless compression
    MODERATE = "moderate"    # Pattern abstraction with minimal loss
    HEAVY = "heavy"          # Aggressive compression with acceptable loss
    EXTREME = "extreme"      # Maximum compression for archival


class CompressionStrategy(str, Enum):
    """Compression strategies for different memory types"""
    LOSSLESS = "lossless"
    PATTERN_ABSTRACTION = "pattern_abstraction"
    SEMANTIC_COMPRESSION = "semantic_compression"
    TEMPORAL_COMPRESSION = "temporal_compression"
    FREQUENCY_BASED = "frequency_based"


@dataclass
class CompressionRule:
    """Rule for memory compression based on age and importance"""
    memory_type: MemoryType
    age_days_threshold: int
    importance_threshold: float
    compression_level: CompressionLevel
    compression_strategy: CompressionStrategy
    preserve_patterns: bool = True
    preserve_strategic_value: bool = True


@dataclass
class CompressionStats:
    """Statistics for compression operations"""
    memories_compressed: int = 0
    patterns_extracted: int = 0
    storage_saved_bytes: int = 0
    compression_ratio: float = 0.0
    processing_time_seconds: float = 0.0
    errors_count: int = 0


class CompressionEngine:
    """
    Advanced compression engine for aging memories in VLTM.
    
    Implements multiple compression strategies:
    - Age-based progressive compression
    - Pattern abstraction for similar memories
    - Semantic compression preserving meaning
    - Temporal compression for sequences
    """
    
    def __init__(self, storage_backend: StorageBackend, config: Optional[VLTMConfiguration] = None):
        self.storage_backend = storage_backend
        self.config = config
        self.compression_rules: List[CompressionRule] = []
        self.pattern_cache: Dict[str, Any] = {}
        self.compression_stats = CompressionStats()
        
        # Compression algorithms
        self.compressors = {
            CompressionStrategy.LOSSLESS: self._lossless_compression,
            CompressionStrategy.PATTERN_ABSTRACTION: self._pattern_abstraction,
            CompressionStrategy.SEMANTIC_COMPRESSION: self._semantic_compression,
            CompressionStrategy.TEMPORAL_COMPRESSION: self._temporal_compression,
            CompressionStrategy.FREQUENCY_BASED: self._frequency_based_compression
        }
        
        logger.info("Compression Engine initialized")
    
    async def initialize(self) -> bool:
        """Initialize compression engine with default rules"""
        try:
            await self._setup_default_compression_rules()
            await self._initialize_pattern_cache()
            logger.info("Compression Engine initialization complete")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Compression Engine: {e}")
            return False
    
    async def _setup_default_compression_rules(self):
        """Setup default compression rules for different memory types"""
        
        rules = [
            CompressionRule(
                memory_type=MemoryType.STRATEGIC_KNOWLEDGE,
                age_days_threshold=180,
                importance_threshold=0.7,
                compression_level=CompressionLevel.LIGHT,
                compression_strategy=CompressionStrategy.LOSSLESS,
                preserve_patterns=True,
                preserve_strategic_value=True
            ),
            CompressionRule(
                memory_type=MemoryType.SUCCESSFUL_IMPROVEMENT,
                age_days_threshold=90,
                importance_threshold=0.6,
                compression_level=CompressionLevel.MODERATE,
                compression_strategy=CompressionStrategy.PATTERN_ABSTRACTION,
                preserve_patterns=True,
                preserve_strategic_value=True
            ),
            CompressionRule(
                memory_type=MemoryType.FAILED_EXPERIMENT,
                age_days_threshold=60,
                importance_threshold=0.4,
                compression_level=CompressionLevel.HEAVY,
                compression_strategy=CompressionStrategy.SEMANTIC_COMPRESSION,
                preserve_patterns=False,
                preserve_strategic_value=False
            ),
            CompressionRule(
                memory_type=MemoryType.CODE_PATTERN,
                age_days_threshold=30,
                importance_threshold=0.5,
                compression_level=CompressionLevel.MODERATE,
                compression_strategy=CompressionStrategy.TEMPORAL_COMPRESSION,
                preserve_patterns=True,
                preserve_strategic_value=False
            ),
            CompressionRule(
                memory_type=MemoryType.META_LEARNING_RULE,
                age_days_threshold=365,
                importance_threshold=0.8,
                compression_level=CompressionLevel.LIGHT,
                compression_strategy=CompressionStrategy.LOSSLESS,
                preserve_patterns=True,
                preserve_strategic_value=True
            )
        ]
        
        self.compression_rules = rules
        logger.info(f"Setup {len(self.compression_rules)} compression rules")
    
    async def _initialize_pattern_cache(self):
        """Initialize pattern cache for compression"""
        try:
            patterns = await self.storage_backend.get_all_patterns(limit=1000)
            for pattern in patterns:
                pattern_id = pattern.get("pattern_id")
                if pattern_id:
                    self.pattern_cache[pattern_id] = {
                        "pattern_type": pattern.get("pattern_type"),
                        "abstraction": pattern.get("abstraction"),
                        "frequency": pattern.get("frequency", 1)
                    }
            logger.info(f"Loaded {len(self.pattern_cache)} patterns to cache")
        except Exception as e:
            logger.warning(f"Could not load patterns to cache: {e}")
    
    async def compress_aged_memories(self, max_batch_size: int = 100) -> CompressionStats:
        """Compress memories based on aging rules"""
        start_time = datetime.utcnow()
        stats = CompressionStats()
        
        try:
            logger.info("Starting aged memory compression...")
            
            for rule in self.compression_rules:
                batch_stats = await self._compress_memories_by_rule(rule, max_batch_size)
                stats.memories_compressed += batch_stats.memories_compressed
                stats.patterns_extracted += batch_stats.patterns_extracted
                stats.storage_saved_bytes += batch_stats.storage_saved_bytes
                stats.errors_count += batch_stats.errors_count
            
            if stats.storage_saved_bytes > 0:
                stats.compression_ratio = stats.storage_saved_bytes / (stats.storage_saved_bytes + 1000000)
            
            stats.processing_time_seconds = (datetime.utcnow() - start_time).total_seconds()
            
            self.compression_stats.memories_compressed += stats.memories_compressed
            self.compression_stats.patterns_extracted += stats.patterns_extracted
            self.compression_stats.storage_saved_bytes += stats.storage_saved_bytes
            
            logger.info(f"Compression completed: {stats.memories_compressed} memories processed")
            return stats
            
        except Exception as e:
            logger.error(f"Error during memory compression: {e}")
            stats.errors_count += 1
            return stats
    
    async def _compress_memories_by_rule(self, rule: CompressionRule, max_batch_size: int) -> CompressionStats:
        """Compress memories matching a specific rule"""
        stats = CompressionStats()
        cutoff_date = datetime.utcnow() - timedelta(days=rule.age_days_threshold)
        
        try:
            eligible_memories = await self.storage_backend.get_memories_for_compression(
                memory_type=rule.memory_type,
                before_date=cutoff_date,
                importance_threshold=rule.importance_threshold,
                limit=max_batch_size
            )
            
            logger.info(f"Found {len(eligible_memories)} memories for {rule.memory_type} compression")
            
            memory_groups = await self._group_similar_memories(eligible_memories)
            
            for group in memory_groups:
                try:
                    compression_result = await self._compress_memory_group(group, rule)
                    if compression_result:
                        stats.memories_compressed += len(group)
                        stats.patterns_extracted += compression_result.get("patterns_extracted", 0)
                        stats.storage_saved_bytes += compression_result.get("bytes_saved", 0)
                except Exception as e:
                    logger.warning(f"Error compressing memory group: {e}")
                    stats.errors_count += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in rule-based compression: {e}")
            stats.errors_count += 1
            return stats
    
    async def _group_similar_memories(self, memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group similar memories for batch compression"""
        groups = []
        used_memories = set()
        
        for i, memory in enumerate(memories):
            if i in used_memories:
                continue
            
            group = [memory]
            used_memories.add(i)
            
            for j, other_memory in enumerate(memories[i+1:], i+1):
                if j in used_memories:
                    continue
                
                if await self._are_memories_similar(memory, other_memory):
                    group.append(other_memory)
                    used_memories.add(j)
            
            groups.append(group)
        
        return groups
    
    async def _are_memories_similar(self, memory1: Dict[str, Any], memory2: Dict[str, Any]) -> bool:
        """Check if two memories are similar enough to group together"""
        try:
            content1 = str(memory1.get("content", ""))
            content2 = str(memory2.get("content", ""))
            
            common_words = set(content1.lower().split()) & set(content2.lower().split())
            total_words = len(set(content1.lower().split()) | set(content2.lower().split()))
            
            if total_words == 0:
                return False
            
            similarity = len(common_words) / total_words
            return similarity > 0.3
            
        except Exception:
            return False
    
    async def _compress_memory_group(self, memory_group: List[Dict[str, Any]], 
                                   rule: CompressionRule) -> Optional[Dict[str, Any]]:
        """Compress a group of similar memories"""
        try:
            compressor = self.compressors.get(rule.compression_strategy)
            if not compressor:
                logger.warning(f"Unknown compression strategy: {rule.compression_strategy}")
                return None
            
            compression_result = await compressor(memory_group, rule)
            
            if compression_result and compression_result.get("success", False):
                await self._store_compressed_memories(compression_result, rule)
                await self._archive_original_memories(memory_group)
                return compression_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error compressing memory group: {e}")
            return None
    
    async def _lossless_compression(self, memory_group: List[Dict[str, Any]], 
                                  rule: CompressionRule) -> Dict[str, Any]:
        """Apply lossless compression to memory group"""
        try:
            serialized_data = json.dumps(memory_group, default=str)
            original_size = len(serialized_data.encode('utf-8'))
            
            compressed_data = zlib.compress(serialized_data.encode('utf-8'))
            compressed_size = len(compressed_data)
            
            compression_ratio = (original_size - compressed_size) / original_size if original_size > 0 else 0
            
            return {
                "success": True,
                "compression_method": "lossless",
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
                "compressed_data": compressed_data,
                "memory_count": len(memory_group),
                "patterns_extracted": 0,
                "bytes_saved": original_size - compressed_size
            }
            
        except Exception as e:
            logger.error(f"Lossless compression failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _pattern_abstraction(self, memory_group: List[Dict[str, Any]], 
                                 rule: CompressionRule) -> Dict[str, Any]:
        """Extract patterns and create abstract representation"""
        try:
            patterns = await self._extract_group_patterns(memory_group)
            
            abstract_memory = {
                "pattern_id": str(uuid.uuid4()),
                "pattern_type": "abstracted_group",
                "original_count": len(memory_group),
                "common_patterns": patterns,
                "representative_content": await self._create_representative_content(memory_group),
                "memory_ids": [m.get("memory_id") for m in memory_group],
                "abstraction_timestamp": datetime.utcnow().isoformat()
            }
            
            serialized_abstract = json.dumps(abstract_memory, default=str)
            abstract_size = len(serialized_abstract.encode('utf-8'))
            original_size = sum(len(json.dumps(m, default=str).encode('utf-8')) for m in memory_group)
            
            return {
                "success": True,
                "compression_method": "pattern_abstraction", 
                "original_size": original_size,
                "compressed_size": abstract_size,
                "compression_ratio": (original_size - abstract_size) / original_size if original_size > 0 else 0,
                "abstract_memory": abstract_memory,
                "memory_count": len(memory_group),
                "patterns_extracted": len(patterns),
                "bytes_saved": original_size - abstract_size
            }
            
        except Exception as e:
            logger.error(f"Pattern abstraction failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _semantic_compression(self, memory_group: List[Dict[str, Any]], 
                                  rule: CompressionRule) -> Dict[str, Any]:
        """Apply semantic compression preserving meaning"""
        try:
            semantic_summary = await self._create_semantic_summary(memory_group)
            
            key_insights = []
            for memory in memory_group:
                content = memory.get("content", {})
                if isinstance(content, dict) and "insight" in str(content).lower():
                    key_insights.append(content)
            
            compressed_memory = {
                "compression_id": str(uuid.uuid4()),
                "compression_type": "semantic",
                "semantic_summary": semantic_summary,
                "key_insights": key_insights[:5],
                "original_count": len(memory_group),
                "memory_ids": [m.get("memory_id") for m in memory_group],
                "compression_timestamp": datetime.utcnow().isoformat()
            }
            
            serialized_compressed = json.dumps(compressed_memory, default=str)
            compressed_size = len(serialized_compressed.encode('utf-8'))
            original_size = sum(len(json.dumps(m, default=str).encode('utf-8')) for m in memory_group)
            
            return {
                "success": True,
                "compression_method": "semantic_compression",
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": (original_size - compressed_size) / original_size if original_size > 0 else 0,
                "compressed_memory": compressed_memory,
                "memory_count": len(memory_group),
                "patterns_extracted": 0,
                "bytes_saved": original_size - compressed_size
            }
            
        except Exception as e:
            logger.error(f"Semantic compression failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _temporal_compression(self, memory_group: List[Dict[str, Any]], 
                                  rule: CompressionRule) -> Dict[str, Any]:
        """Apply temporal compression for sequential memories"""
        try:
            sorted_memories = sorted(memory_group, key=lambda m: m.get("created_at", datetime.min))
            
            temporal_sequence = {
                "sequence_id": str(uuid.uuid4()),
                "sequence_type": "temporal_compression",
                "start_time": sorted_memories[0].get("created_at") if sorted_memories else None,
                "end_time": sorted_memories[-1].get("created_at") if sorted_memories else None,
                "event_count": len(sorted_memories),
                "key_events": [
                    {
                        "timestamp": m.get("created_at"),
                        "summary": str(m.get("content", ""))[:200],
                        "importance": m.get("importance_score", 0.5)
                    }
                    for m in sorted_memories[::max(1, len(sorted_memories)//5)]
                ],
                "compression_timestamp": datetime.utcnow().isoformat()
            }
            
            serialized_sequence = json.dumps(temporal_sequence, default=str)
            compressed_size = len(serialized_sequence.encode('utf-8'))
            original_size = sum(len(json.dumps(m, default=str).encode('utf-8')) for m in memory_group)
            
            return {
                "success": True,
                "compression_method": "temporal_compression",
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": (original_size - compressed_size) / original_size if original_size > 0 else 0,
                "temporal_sequence": temporal_sequence,
                "memory_count": len(memory_group),
                "patterns_extracted": 1,
                "bytes_saved": original_size - compressed_size
            }
            
        except Exception as e:
            logger.error(f"Temporal compression failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _frequency_based_compression(self, memory_group: List[Dict[str, Any]], 
                                         rule: CompressionRule) -> Dict[str, Any]:
        """Apply frequency-based compression for repeated patterns"""
        try:
            content_frequency = {}
            for memory in memory_group:
                content = str(memory.get("content", ""))
                words = content.lower().split()
                
                for word in words:
                    if len(word) > 3:
                        content_frequency[word] = content_frequency.get(word, 0) + 1
            
            high_freq_patterns = {
                word: count for word, count in content_frequency.items() 
                if count >= len(memory_group) * 0.3
            }
            
            frequency_compressed = {
                "compression_id": str(uuid.uuid4()),
                "compression_type": "frequency_based",
                "high_frequency_patterns": high_freq_patterns,
                "pattern_summary": ", ".join(list(high_freq_patterns.keys())[:5]),
                "original_count": len(memory_group),
                "representative_memories": [
                    memory for memory in memory_group 
                    if memory.get("importance_score", 0) > 0.7
                ][:3],
                "compression_timestamp": datetime.utcnow().isoformat()
            }
            
            serialized_compressed = json.dumps(frequency_compressed, default=str)
            compressed_size = len(serialized_compressed.encode('utf-8'))
            original_size = sum(len(json.dumps(m, default=str).encode('utf-8')) for m in memory_group)
            
            return {
                "success": True,
                "compression_method": "frequency_based_compression",
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": (original_size - compressed_size) / original_size if original_size > 0 else 0,
                "frequency_compressed": frequency_compressed,
                "memory_count": len(memory_group),
                "patterns_extracted": len(high_freq_patterns),
                "bytes_saved": original_size - compressed_size
            }
            
        except Exception as e:
            logger.error(f"Frequency-based compression failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _extract_group_patterns(self, memory_group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract common patterns from memory group"""
        patterns = []
        
        try:
            content_texts = [str(m.get("content", "")) for m in memory_group]
            
            word_frequency = {}
            for text in content_texts:
                words = text.lower().split()
                for word in words:
                    if len(word) > 4:
                        word_frequency[word] = word_frequency.get(word, 0) + 1
            
            common_themes = [
                word for word, count in word_frequency.items() 
                if count > len(memory_group) * 0.4
            ]
            
            if common_themes:
                patterns.append({
                    "pattern_type": "common_themes",
                    "themes": common_themes,
                    "frequency": len(common_themes)
                })
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Error extracting group patterns: {e}")
            return patterns
    
    async def _create_representative_content(self, memory_group: List[Dict[str, Any]]) -> str:
        """Create representative content from memory group"""
        try:
            all_content = " ".join([str(m.get("content", "")) for m in memory_group])
            summary = all_content[:300] + "..." if len(all_content) > 300 else all_content
            return summary
        except Exception as e:
            logger.warning(f"Error creating representative content: {e}")
            return "Representative content could not be generated"
    
    async def _create_semantic_summary(self, memory_group: List[Dict[str, Any]]) -> str:
        """Create semantic summary of memory group"""
        try:
            concepts = set()
            
            for memory in memory_group:
                content = str(memory.get("content", ""))
                words = content.split()
                for word in words:
                    if len(word) > 5 and word.lower() not in ['the', 'and', 'for', 'with']:
                        concepts.add(word.lower())
            
            top_concepts = list(concepts)[:10]
            summary = f"Memory group containing {len(memory_group)} memories related to: " + ", ".join(top_concepts)
            return summary
            
        except Exception as e:
            logger.warning(f"Error creating semantic summary: {e}")
            return f"Semantic summary of {len(memory_group)} related memories"
    
    async def _store_compressed_memories(self, compression_result: Dict[str, Any], rule: CompressionRule):
        """Store compressed memory representation"""
        try:
            if rule.compression_level in [CompressionLevel.LIGHT, CompressionLevel.MODERATE]:
                await self.storage_backend.store_compressed_memory(
                    compressed_data=compression_result,
                    compression_method=compression_result.get("compression_method"),
                    original_count=compression_result.get("memory_count", 0)
                )
            else:
                await self.storage_backend.archive_compressed_memory(
                    compressed_data=compression_result,
                    archive_location="vltm_compressed_archive"
                )
            
            logger.debug(f"Stored compressed memory using {compression_result.get('compression_method')}")
            
        except Exception as e:
            logger.error(f"Error storing compressed memories: {e}")
    
    async def _archive_original_memories(self, memory_group: List[Dict[str, Any]]):
        """Archive original memories after successful compression"""
        try:
            memory_ids = [m.get("memory_id") for m in memory_group if m.get("memory_id")]
            await self.storage_backend.mark_memories_compressed(memory_ids)
            logger.debug(f"Archived {len(memory_ids)} original memories")
        except Exception as e:
            logger.error(f"Error archiving original memories: {e}")
    
    async def decompress_memory(self, compressed_memory_id: str) -> Optional[List[Dict[str, Any]]]:
        """Decompress a compressed memory back to original format"""
        try:
            compressed_data = await self.storage_backend.get_compressed_memory(compressed_memory_id)
            if not compressed_data:
                return None
            
            compression_method = compressed_data.get("compression_method")
            
            if compression_method == "lossless":
                return await self._decompress_lossless(compressed_data)
            elif compression_method == "pattern_abstraction":
                return await self._decompress_pattern_abstraction(compressed_data)
            else:
                logger.warning(f"Cannot decompress method: {compression_method}")
                return None
                
        except Exception as e:
            logger.error(f"Error decompressing memory: {e}")
            return None
    
    async def _decompress_lossless(self, compressed_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Decompress lossless compressed data"""
        try:
            compressed_bytes = compressed_data.get("compressed_data")
            if not compressed_bytes:
                return None
            
            decompressed_data = zlib.decompress(compressed_bytes)
            original_memories = json.loads(decompressed_data.decode('utf-8'))
            return original_memories
            
        except Exception as e:
            logger.error(f"Error in lossless decompression: {e}")
            return None
    
    async def _decompress_pattern_abstraction(self, compressed_data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Reconstruct memories from pattern abstraction (partial recovery)"""
        try:
            abstract_memory = compressed_data.get("abstract_memory", {})
            
            # Can only return abstract representation, not original memories
            reconstructed = [{
                "memory_id": f"reconstructed_{i}",
                "content": abstract_memory.get("representative_content", ""),
                "patterns": abstract_memory.get("common_patterns", []),
                "original_count": abstract_memory.get("original_count", 0),
                "reconstruction_note": "Reconstructed from pattern abstraction"
            }]
            
            return reconstructed
            
        except Exception as e:
            logger.error(f"Error in pattern abstraction decompression: {e}")
            return None
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression engine statistics"""
        return {
            "total_memories_compressed": self.compression_stats.memories_compressed,
            "total_patterns_extracted": self.compression_stats.patterns_extracted,
            "total_storage_saved_bytes": self.compression_stats.storage_saved_bytes,
            "compression_rules_count": len(self.compression_rules),
            "pattern_cache_size": len(self.pattern_cache),
            "supported_strategies": list(self.compressors.keys())
        }
    
    async def add_compression_rule(self, rule: CompressionRule) -> bool:
        """Add a new compression rule"""
        try:
            self.compression_rules.append(rule)
            logger.info(f"Added compression rule for {rule.memory_type}")
            return True
        except Exception as e:
            logger.error(f"Error adding compression rule: {e}")
            return False
    
    async def remove_compression_rule(self, memory_type: MemoryType) -> bool:
        """Remove compression rule for a memory type"""
        try:
            original_count = len(self.compression_rules)
            self.compression_rules = [rule for rule in self.compression_rules if rule.memory_type != memory_type]
            removed_count = original_count - len(self.compression_rules)
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} compression rule(s) for {memory_type}")
                return True
            else:
                logger.warning(f"No compression rule found for {memory_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing compression rule: {e}")
            return False