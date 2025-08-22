"""
Very Long-Term Memory Pattern Extractor

This module implements pattern extraction algorithms for the Snake Agent's
Very Long-Term Memory System, identifying temporal, causal, and behavioral
patterns from memory data for consolidation into strategic knowledge.
"""

import asyncio
import json
import logging
import re
import uuid
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
from scipy import stats

from core.vltm_data_models import (
    VeryLongTermMemory, MemoryPattern, PatternType, MemoryType,
    PatternRecord, VLTMConfiguration
)

logger = logging.getLogger(__name__)


@dataclass
class TemporalPattern:
    """Represents a temporal pattern found in memory data"""
    pattern_id: str
    pattern_type: str
    time_period: str
    frequency: float
    confidence: float
    description: str
    supporting_memories: List[str]
    temporal_data: Dict[str, Any]


@dataclass
class CausalPattern:
    """Represents a causal relationship pattern"""
    pattern_id: str
    cause_events: List[str]
    effect_events: List[str]
    confidence: float
    time_delay: Optional[float]
    strength: float
    description: str
    supporting_memories: List[str]
    causal_data: Dict[str, Any]


@dataclass
class BehavioralPattern:
    """Represents a behavioral pattern in system or agent actions"""
    pattern_id: str
    behavior_type: str
    behavior_sequence: List[str]
    frequency: int
    success_rate: float
    confidence: float
    description: str
    supporting_memories: List[str]
    behavioral_data: Dict[str, Any]


class PatternExtractor:
    """
    Extracts patterns from memory data for consolidation.
    
    Implements temporal, causal, and behavioral pattern recognition
    algorithms to identify recurring patterns that can be consolidated
    into strategic knowledge.
    """
    
    def __init__(self, config: VLTMConfiguration):
        """Initialize the pattern extractor."""
        self.config = config
        
        # Pattern detection thresholds
        self.temporal_confidence_threshold = 0.7
        self.causal_confidence_threshold = 0.6
        self.behavioral_confidence_threshold = 0.65
        self.min_pattern_support = 3
        
        # Pattern tracking
        self.extracted_patterns = []
        self.pattern_statistics = defaultdict(int)
        
        logger.info("Pattern extractor initialized")
    
    async def extract_patterns(self, memories: List[VeryLongTermMemory]) -> List[PatternRecord]:
        """Extract all types of patterns from a collection of memories."""
        if not memories:
            logger.warning("No memories provided for pattern extraction")
            return []
        
        try:
            logger.info(f"Extracting patterns from {len(memories)} memories")
            all_patterns = []
            
            # Extract temporal patterns
            temporal_patterns = await self.extract_temporal_patterns(memories)
            all_patterns.extend(temporal_patterns)
            
            # Extract causal patterns
            causal_patterns = await self.extract_causal_patterns(memories)
            all_patterns.extend(causal_patterns)
            
            # Extract behavioral patterns
            behavioral_patterns = await self.extract_behavioral_patterns(memories)
            all_patterns.extend(behavioral_patterns)
            
            # Update statistics
            self.pattern_statistics["total_patterns_extracted"] += len(all_patterns)
            self.pattern_statistics["temporal_patterns"] += len(temporal_patterns)
            self.pattern_statistics["causal_patterns"] += len(causal_patterns)
            self.pattern_statistics["behavioral_patterns"] += len(behavioral_patterns)
            
            logger.info(f"Extracted {len(all_patterns)} patterns: "
                       f"{len(temporal_patterns)} temporal, "
                       f"{len(causal_patterns)} causal, "
                       f"{len(behavioral_patterns)} behavioral")
            
            return all_patterns
            
        except Exception as e:
            logger.error(f"Error extracting patterns: {e}", exc_info=True)
            return []
    
    async def extract_temporal_patterns(self, memories: List[VeryLongTermMemory]) -> List[PatternRecord]:
        """Extract temporal patterns from memories"""
        try:
            patterns = []
            
            # Group memories by time periods
            time_series_data = self._prepare_temporal_data(memories)
            
            # Detect cyclic patterns
            for period_type, period_data in time_series_data.items():
                if len(period_data) < 4:
                    continue
                
                counts = [len(period_memories) for period_memories in period_data.values()]
                if len(counts) >= 4:
                    cycle_strength = self._calculate_cyclicity(counts)
                    
                    if cycle_strength > self.temporal_confidence_threshold:
                        pattern_id = str(uuid.uuid4())
                        supporting_memories = [m.memory_id for m in memories[:10]]
                        
                        pattern = PatternRecord(
                            pattern_id=pattern_id,
                            pattern_type=PatternType.TEMPORAL,
                            description=f"Cyclic {period_type} pattern in memory activity",
                            confidence_score=cycle_strength,
                            pattern_data={
                                "cycle_type": "activity_cycle",
                                "period": period_type,
                                "strength": cycle_strength,
                                "pattern_details": {
                                    "average_count": np.mean(counts),
                                    "std_dev": np.std(counts)
                                }
                            },
                            supporting_memories=supporting_memories
                        )
                        patterns.append(pattern)
            
            # Detect trend patterns
            for period_type, period_data in time_series_data.items():
                if len(period_data) >= 3:
                    sorted_periods = sorted(period_data.keys())
                    counts = [len(period_data[period]) for period in sorted_periods]
                    
                    if len(counts) >= 3:
                        x = np.arange(len(counts))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, counts)
                        
                        trend_strength = abs(r_value)
                        if p_value < 0.05 and trend_strength > 0.5:
                            trend_direction = "increasing" if slope > 0 else "decreasing"
                            pattern_id = str(uuid.uuid4())
                            
                            pattern = PatternRecord(
                                pattern_id=pattern_id,
                                pattern_type=PatternType.TEMPORAL,
                                description=f"{trend_direction.capitalize()} {period_type} trend",
                                confidence_score=trend_strength,
                                pattern_data={
                                    "trend_type": "activity_trend",
                                    "direction": trend_direction,
                                    "period": period_type,
                                    "slope": slope,
                                    "r_squared": r_value ** 2
                                },
                                supporting_memories=[m.memory_id for m in memories[:10]]
                            )
                            patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error extracting temporal patterns: {e}")
            return []
    
    async def extract_causal_patterns(self, memories: List[VeryLongTermMemory]) -> List[PatternRecord]:
        """Extract causal relationship patterns from memories"""
        try:
            patterns = []
            causal_events = self._prepare_causal_data(memories)
            
            # Look for cause-effect pairs within time windows
            time_window = timedelta(hours=1)
            
            for i, cause in enumerate(causal_events):
                for j, effect in enumerate(causal_events[i+1:], i+1):
                    time_diff = effect["timestamp"] - cause["timestamp"]
                    
                    if timedelta(0) < time_diff <= time_window:
                        strength = self._calculate_causal_strength(cause, effect, time_diff)
                        
                        if strength > self.causal_confidence_threshold:
                            pattern_id = str(uuid.uuid4())
                            
                            pattern = PatternRecord(
                                pattern_id=pattern_id,
                                pattern_type=PatternType.CAUSAL,
                                description=f"Causal: {self._describe_event(cause)} → {self._describe_event(effect)}",
                                confidence_score=strength,
                                pattern_data={
                                    "cause_events": [self._describe_event(cause)],
                                    "effect_events": [self._describe_event(effect)],
                                    "time_delay": time_diff.total_seconds(),
                                    "strength": strength
                                },
                                supporting_memories=[cause["memory_id"], effect["memory_id"]]
                            )
                            patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error extracting causal patterns: {e}")
            return []
    
    async def extract_behavioral_patterns(self, memories: List[VeryLongTermMemory]) -> List[PatternRecord]:
        """Extract behavioral patterns from memories"""
        try:
            patterns = []
            behavioral_sequences = self._prepare_behavioral_data(memories)
            
            # Extract common subsequences
            all_sequences = [seq["sequence"] for seq in behavioral_sequences if len(seq["sequence"]) >= 2]
            
            if not all_sequences:
                return patterns
            
            subsequence_counts = Counter()
            
            for sequence in all_sequences:
                for length in range(2, min(5, len(sequence) + 1)):
                    for i in range(len(sequence) - length + 1):
                        subseq = tuple(sequence[i:i+length])
                        subsequence_counts[subseq] += 1
            
            min_frequency = max(2, len(all_sequences) // 10)
            
            for subsequence, frequency in subsequence_counts.items():
                if frequency >= min_frequency:
                    # Calculate success rate
                    pattern_successes = 0
                    pattern_total = 0
                    supporting_memories = []
                    
                    for seq_data in behavioral_sequences:
                        if self._contains_subsequence(seq_data["sequence"], list(subsequence)):
                            pattern_total += 1
                            if seq_data["success_count"] > seq_data["failure_count"]:
                                pattern_successes += 1
                            supporting_memories.extend([m.memory_id for m in seq_data["memories"][:2]])
                    
                    success_rate = pattern_successes / pattern_total if pattern_total > 0 else 0
                    confidence = min(0.95, frequency / len(all_sequences) + success_rate * 0.3)
                    
                    if confidence >= self.behavioral_confidence_threshold:
                        pattern_id = str(uuid.uuid4())
                        behavior_type = "success_sequence" if success_rate > 0.7 else \
                                      "failure_sequence" if success_rate < 0.3 else "mixed_sequence"
                        
                        pattern = PatternRecord(
                            pattern_id=pattern_id,
                            pattern_type=PatternType.BEHAVIORAL,
                            description=f"{behavior_type.replace('_', ' ').title()}: {' → '.join(subsequence)}",
                            confidence_score=confidence,
                            pattern_data={
                                "behavior_type": behavior_type,
                                "behavior_sequence": list(subsequence),
                                "frequency": frequency,
                                "success_rate": success_rate
                            },
                            supporting_memories=supporting_memories[:10]
                        )
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error extracting behavioral patterns: {e}")
            return []
    
    # Helper methods
    
    def _prepare_temporal_data(self, memories: List[VeryLongTermMemory]) -> Dict[str, Dict]:
        """Prepare temporal data for pattern analysis"""
        daily_data = defaultdict(list)
        weekly_data = defaultdict(list)
        monthly_data = defaultdict(list)
        
        for memory in memories:
            created_date = memory.created_at.date()
            week_key = f"{created_date.year}-W{created_date.isocalendar()[1]}"
            month_key = f"{created_date.year}-{created_date.month:02d}"
            
            daily_data[str(created_date)].append(memory)
            weekly_data[week_key].append(memory)
            monthly_data[month_key].append(memory)
        
        return {"daily": daily_data, "weekly": weekly_data, "monthly": monthly_data}
    
    def _calculate_cyclicity(self, values: List[float]) -> float:
        """Calculate cyclicity strength using autocorrelation"""
        if len(values) < 4:
            return 0.0
        
        try:
            normalized = np.array(values)
            normalized = (normalized - np.mean(normalized)) / (np.std(normalized) + 1e-8)
            
            correlations = []
            for lag in range(1, min(4, len(normalized))):
                if lag < len(normalized):
                    correlation = np.corrcoef(normalized[:-lag], normalized[lag:])[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))
            
            return max(correlations) if correlations else 0.0
        except Exception:
            return 0.0
    
    def _prepare_causal_data(self, memories: List[VeryLongTermMemory]) -> List[Dict[str, Any]]:
        """Prepare data for causal analysis"""
        causal_events = []
        
        for memory in memories:
            try:
                content = json.loads(memory.compressed_content)
                
                event_data = {
                    "memory_id": memory.memory_id,
                    "timestamp": memory.created_at,
                    "memory_type": memory.memory_type,
                    "content": content,
                    "success_indicators": self._extract_success_indicators(content),
                    "failure_indicators": self._extract_failure_indicators(content),
                    "action_types": self._extract_action_types(content)
                }
                causal_events.append(event_data)
            except Exception:
                continue
        
        causal_events.sort(key=lambda x: x["timestamp"])
        return causal_events
    
    def _calculate_causal_strength(self, cause: Dict, effect: Dict, time_diff: timedelta) -> float:
        """Calculate strength of causal relationship"""
        try:
            # Temporal proximity factor
            temporal_factor = max(0, 1 - (time_diff.total_seconds() / 3600))
            
            # Content similarity factor
            cause_text = json.dumps(cause["content"]).lower()
            effect_text = json.dumps(effect["content"]).lower()
            
            cause_words = set(re.findall(r'\b\w+\b', cause_text))
            effect_words = set(re.findall(r'\b\w+\b', effect_text))
            
            if len(cause_words) == 0 or len(effect_words) == 0:
                content_similarity = 0
            else:
                overlap = len(cause_words.intersection(effect_words))
                content_similarity = overlap / min(len(cause_words), len(effect_words))
            
            # Logical relationship factor
            logic_factor = 0.5
            if cause["failure_indicators"] and effect["success_indicators"]:
                logic_factor = 0.8
            elif cause["action_types"] and effect["success_indicators"]:
                logic_factor = 0.7
            
            strength = (temporal_factor * 0.4 + content_similarity * 0.3 + logic_factor * 0.3)
            return min(1.0, max(0.0, strength))
        except Exception:
            return 0.0
    
    def _prepare_behavioral_data(self, memories: List[VeryLongTermMemory]) -> List[Dict[str, Any]]:
        """Prepare data for behavioral pattern analysis"""
        session_groups = defaultdict(list)
        
        for memory in memories:
            session_key = memory.source_session
            if not session_key or session_key == "unknown":
                session_key = memory.created_at.date().isoformat()
            session_groups[session_key].append(memory)
        
        behavioral_sequences = []
        for session, session_memories in session_groups.items():
            if len(session_memories) < 2:
                continue
            
            session_memories.sort(key=lambda m: m.created_at)
            
            sequence_data = {
                "session": session,
                "memories": session_memories,
                "sequence": self._extract_behavior_sequence(session_memories),
                "success_count": sum(1 for m in session_memories if self._is_success_memory(m)),
                "failure_count": sum(1 for m in session_memories if self._is_failure_memory(m))
            }
            behavioral_sequences.append(sequence_data)
        
        return behavioral_sequences
    
    def _extract_success_indicators(self, content: Dict) -> List[str]:
        """Extract success indicators from content"""
        success_keywords = ["success", "improved", "optimized", "fixed", "resolved", "completed"]
        content_text = json.dumps(content).lower()
        return [kw for kw in success_keywords if kw in content_text]
    
    def _extract_failure_indicators(self, content: Dict) -> List[str]:
        """Extract failure indicators from content"""
        failure_keywords = ["failed", "error", "exception", "crash", "bug", "issue"]
        content_text = json.dumps(content).lower()
        return [kw for kw in failure_keywords if kw in content_text]
    
    def _extract_action_types(self, content: Dict) -> List[str]:
        """Extract action types from content"""
        action_keywords = ["refactor", "implement", "optimize", "test", "deploy", "analyze"]
        content_text = json.dumps(content).lower()
        return [kw for kw in action_keywords if kw in content_text]
    
    def _describe_event(self, event: Dict) -> str:
        """Create a description of an event"""
        event_type = event.get("memory_type", "unknown")
        actions = ", ".join(event.get("action_types", [])[:2])
        return f"{event_type}({actions})" if actions else str(event_type)
    
    def _extract_behavior_sequence(self, memories: List[VeryLongTermMemory]) -> List[str]:
        """Extract behavior sequence from ordered memories"""
        sequence = []
        for memory in memories:
            try:
                content = json.loads(memory.compressed_content)
                behavior = self._extract_primary_behavior(content)
                if behavior:
                    sequence.append(behavior)
            except Exception:
                continue
        return sequence
    
    def _extract_primary_behavior(self, content: Dict) -> str:
        """Extract primary behavior from content"""
        content_text = str(content).lower()
        if "optimize" in content_text:
            return "optimize"
        elif "test" in content_text:
            return "test"
        elif "implement" in content_text:
            return "implement"
        elif "analyze" in content_text:
            return "analyze"
        else:
            return "general_action"
    
    def _is_success_memory(self, memory: VeryLongTermMemory) -> bool:
        """Check if memory indicates success"""
        try:
            content = json.loads(memory.compressed_content)
            return len(self._extract_success_indicators(content)) > 0
        except Exception:
            return False
    
    def _is_failure_memory(self, memory: VeryLongTermMemory) -> bool:
        """Check if memory indicates failure"""
        try:
            content = json.loads(memory.compressed_content)
            return len(self._extract_failure_indicators(content)) > 0
        except Exception:
            return False
    
    def _contains_subsequence(self, sequence: List[str], subsequence: List[str]) -> bool:
        """Check if sequence contains subsequence"""
        if len(subsequence) > len(sequence):
            return False
        
        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i:i+len(subsequence)] == subsequence:
                return True
        return False
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get pattern extraction statistics"""
        return dict(self.pattern_statistics)