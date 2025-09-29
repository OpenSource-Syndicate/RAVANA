"""
Very Long-Term Memory Classifier

This module implements memory classification algorithms for the Snake Agent's
Very Long-Term Memory System, including strategic/tactical classification,
importance evaluation, and memory promotion criteria.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np

from core.vltm_data_models import (
    MemoryType, MemoryImportanceLevel, MemoryRecord, VLTMConfiguration
)

logger = logging.getLogger(__name__)


class ClassificationCriteria(Enum):
    """Criteria for memory classification"""
    SUCCESS_IMPACT = "success_impact"
    FAILURE_SEVERITY = "failure_severity"
    FREQUENCY_OF_OCCURRENCE = "frequency_of_occurrence"
    ARCHITECTURAL_RELEVANCE = "architectural_relevance"
    LEARNING_VALUE = "learning_value"
    STRATEGIC_IMPORTANCE = "strategic_importance"
    TEMPORAL_RELEVANCE = "temporal_relevance"
    CAUSAL_RELATIONSHIP = "causal_relationship"


@dataclass
class ClassificationFeatures:
    """Features extracted from memory content for classification"""
    content_keywords: Set[str]
    success_indicators: List[str]
    failure_indicators: List[str]
    architectural_terms: List[str]
    performance_metrics: Dict[str, float]
    temporal_markers: List[str]
    causal_chains: List[Dict[str, Any]]
    improvement_evidence: List[str]
    error_patterns: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "content_keywords": list(self.content_keywords),
            "success_indicators": self.success_indicators,
            "failure_indicators": self.failure_indicators,
            "architectural_terms": self.architectural_terms,
            "performance_metrics": self.performance_metrics,
            "temporal_markers": self.temporal_markers,
            "causal_chains": self.causal_chains,
            "improvement_evidence": self.improvement_evidence,
            "error_patterns": self.error_patterns
        }


class MemoryClassifier:
    """
    Classifies memories into strategic/tactical categories and evaluates importance.

    Uses rule-based classification combined with statistical analysis to determine
    memory importance, strategic value, and appropriate retention policies.
    """

    def __init__(self, config: VLTMConfiguration):
        """
        Initialize the memory classifier.

        Args:
            config: VLTM configuration containing classification thresholds
        """
        self.config = config

        # Classification thresholds
        self.strategic_threshold = 0.7
        self.tactical_threshold = 0.4
        self.critical_importance_threshold = 0.9

        # Keyword patterns for different categories
        self._init_keyword_patterns()

        # Statistical tracking
        self.classification_history = []
        self.memory_type_frequencies = {}
        self.importance_distributions = {}

        logger.info("Memory classifier initialized")

    def _init_keyword_patterns(self):
        """Initialize keyword patterns for classification"""

        # Strategic keywords
        self.strategic_keywords = {
            "architecture": [
                "architecture", "design pattern", "system design", "scalability",
                "microservices", "monolith", "distributed", "component", "module",
                "interface", "api", "protocol", "framework", "infrastructure"
            ],
            "performance": [
                "performance", "optimization", "bottleneck", "latency", "throughput",
                "memory usage", "cpu usage", "response time", "benchmark", "profiling"
            ],
            "learning": [
                "learning", "adaptation", "evolution", "improvement", "pattern",
                "strategy", "meta-learning", "knowledge", "insight", "discovery"
            ],
            "critical_systems": [
                "critical", "failure", "crash", "corruption", "security", "vulnerability",
                "breach", "attack", "recovery", "backup", "disaster", "emergency"
            ]
        }

        # Tactical keywords
        self.tactical_keywords = {
            "code_changes": [
                "refactor", "bug fix", "feature", "implementation", "function",
                "method", "class", "variable", "parameter", "return", "exception"
            ],
            "experiments": [
                "experiment", "test", "trial", "hypothesis", "result", "outcome",
                "validation", "verification", "analysis", "measurement"
            ],
            "improvements": [
                "improvement", "enhancement", "upgrade", "update", "modification",
                "change", "adjustment", "tweak", "fix", "patch"
            ]
        }

        # Success indicators
        self.success_indicators = [
            "successful", "improved", "optimized", "enhanced", "fixed", "resolved",
            "achieved", "accomplished", "completed", "working", "stable", "reliable",
            "faster", "better", "efficient", "effective", "reduced", "increased"
        ]

        # Failure indicators
        self.failure_indicators = [
            "failed", "error", "exception", "crash", "bug", "issue", "problem",
            "broken", "incorrect", "invalid", "timeout", "deadlock", "memory leak",
            "corrupted", "unstable", "unreliable", "slow", "degraded"
        ]

        # Architectural terms
        self.architectural_terms = [
            "component", "service", "module", "layer", "tier", "interface",
            "gateway", "proxy", "adapter", "facade", "observer", "strategy",
            "factory", "singleton", "repository", "controller", "model", "view"
        ]

    def classify_memory(self, memory_record: MemoryRecord) -> Dict[str, Any]:
        """
        Classify a memory record and determine its importance and strategic value.

        Args:
            memory_record: Memory record to classify

        Returns:
            Classification results including type, importance, and strategic value
        """
        try:
            # Extract features from memory content
            features = self._extract_features(memory_record)

            # Determine memory type based on content analysis
            memory_type = self._determine_memory_type(features, memory_record)

            # Calculate importance score
            importance_score = self._calculate_importance_score(
                features, memory_record)

            # Calculate strategic value
            strategic_value = self._calculate_strategic_value(
                features, memory_record)

            # Determine importance level
            importance_level = self._determine_importance_level(
                importance_score)

            # Create classification result
            classification_result = {
                "memory_type": memory_type,
                "importance_score": importance_score,
                "strategic_value": strategic_value,
                "importance_level": importance_level,
                "features": features.to_dict(),
                "classification_timestamp": datetime.utcnow().isoformat(),
                "classifier_version": "1.0"
            }

            # Update statistics
            self._update_classification_statistics(classification_result)

            logger.debug(f"Classified memory {memory_record.memory_id}: "
                         f"type={memory_type}, importance={importance_score:.3f}, "
                         f"strategic={strategic_value:.3f}")

            return classification_result

        except Exception as e:
            logger.error(
                f"Error classifying memory {memory_record.memory_id}: {e}")
            return self._get_default_classification()

    def _extract_features(self, memory_record: MemoryRecord) -> ClassificationFeatures:
        """Extract classification features from memory content"""

        # Convert content to text for analysis
        content_text = json.dumps(memory_record.content, default=str).lower()
        metadata_text = json.dumps(memory_record.metadata, default=str).lower()
        full_text = f"{content_text} {metadata_text}"

        # Extract keywords
        content_keywords = self._extract_keywords(full_text)

        # Identify success indicators
        success_indicators = [
            indicator for indicator in self.success_indicators
            if indicator in full_text
        ]

        # Identify failure indicators
        failure_indicators = [
            indicator for indicator in self.failure_indicators
            if indicator in full_text
        ]

        # Identify architectural terms
        architectural_terms = [
            term for term in self.architectural_terms
            if term in full_text
        ]

        # Extract performance metrics
        performance_metrics = self._extract_performance_metrics(
            memory_record.content)

        # Extract temporal markers
        temporal_markers = self._extract_temporal_markers(full_text)

        # Extract causal chains
        causal_chains = self._extract_causal_chains(memory_record.content)

        # Extract improvement evidence
        improvement_evidence = self._extract_improvement_evidence(full_text)

        # Extract error patterns
        error_patterns = self._extract_error_patterns(full_text)

        return ClassificationFeatures(
            content_keywords=content_keywords,
            success_indicators=success_indicators,
            failure_indicators=failure_indicators,
            architectural_terms=architectural_terms,
            performance_metrics=performance_metrics,
            temporal_markers=temporal_markers,
            causal_chains=causal_chains,
            improvement_evidence=improvement_evidence,
            error_patterns=error_patterns
        )

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract relevant keywords from text"""
        # Simple keyword extraction using patterns
        keywords = set()

        # Extract strategic keywords
        for category, terms in self.strategic_keywords.items():
            for term in terms:
                if term in text:
                    keywords.add(
                        f"strategic_{category}_{term.replace(' ', '_')}")

        # Extract tactical keywords
        for category, terms in self.tactical_keywords.items():
            for term in terms:
                if term in text:
                    keywords.add(
                        f"tactical_{category}_{term.replace(' ', '_')}")

        return keywords

    def _extract_performance_metrics(self, content: Dict[str, Any]) -> Dict[str, float]:
        """Extract performance metrics from content"""
        metrics = {}

        # Look for numeric values with performance-related keys
        performance_keys = [
            "time", "duration", "latency", "response_time", "throughput",
            "memory", "cpu", "disk", "bandwidth", "rate", "count", "size"
        ]

        def extract_from_dict(data, prefix=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    key_lower = key.lower()
                    if any(perf_key in key_lower for perf_key in performance_keys):
                        if isinstance(value, (int, float)):
                            metrics[f"{prefix}{key}"] = float(value)
                    elif isinstance(value, dict):
                        extract_from_dict(value, f"{prefix}{key}_")

        extract_from_dict(content)
        return metrics

    def _extract_temporal_markers(self, text: str) -> List[str]:
        """Extract temporal markers from text"""
        temporal_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # Date patterns
            r'\b\d{2}:\d{2}:\d{2}\b',  # Time patterns
            r'\b(before|after|during|since|until)\b',  # Temporal prepositions
            r'\b(yesterday|today|tomorrow|now|then)\b',  # Temporal adverbs
            # Duration
            r'\b(\d+)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b'
        ]

        markers = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            markers.extend(matches)

        return markers

    def _extract_causal_chains(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract causal relationship chains from content"""
        causal_chains = []

        # Look for causal indicators in content
        causal_indicators = ["because", "due to",
                             "caused by", "resulted in", "led to", "triggered"]

        content_str = json.dumps(content, default=str).lower()

        for indicator in causal_indicators:
            if indicator in content_str:
                # Simple causal chain extraction
                # In a more sophisticated implementation, this would use NLP
                causal_chains.append({
                    "indicator": indicator,
                    "context": "causal_relationship_detected",
                    "confidence": 0.6
                })

        return causal_chains

    def _extract_improvement_evidence(self, text: str) -> List[str]:
        """Extract evidence of improvements from text"""
        improvement_patterns = [
            r'\b(improved|optimized|enhanced|fixed|resolved)\s+\w+',
            r'\b\d+%\s+(better|faster|slower|improvement)',
            r'\breduced\s+\w+\s+by\s+\d+',
            r'\bincreased\s+\w+\s+by\s+\d+'
        ]

        evidence = []
        for pattern in improvement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            evidence.extend(matches)

        return evidence

    def _extract_error_patterns(self, text: str) -> List[str]:
        """Extract error patterns from text"""
        error_patterns = [
            r'\b\w*error\w*\b',
            r'\b\w*exception\w*\b',
            r'\b\w*failure\w*\b',
            r'\bstack\s+trace\b',
            r'\bcode\s+\d+\b'
        ]

        patterns = []
        for pattern in error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            patterns.extend(matches)

        return patterns

    def _determine_memory_type(self, features: ClassificationFeatures, memory_record: MemoryRecord) -> MemoryType:
        """Determine the memory type based on features"""

        # Count strategic vs tactical indicators
        strategic_score = 0
        tactical_score = 0

        # Analyze keywords
        for keyword in features.content_keywords:
            if keyword.startswith("strategic"):
                strategic_score += 1
            elif keyword.startswith("tactical"):
                tactical_score += 1

        # Critical failure detection
        if (len(features.failure_indicators) > 2 and
                any("critical" in indicator for indicator in features.failure_indicators)):
            return MemoryType.CRITICAL_FAILURE

        # Strategic knowledge detection
        if (strategic_score > tactical_score and
                len(features.architectural_terms) > 2):
            return MemoryType.STRATEGIC_KNOWLEDGE

        # Architecture insight detection
        if len(features.architectural_terms) > 3:
            return MemoryType.ARCHITECTURAL_INSIGHT

        # Success/failure classification
        if len(features.success_indicators) > len(features.failure_indicators):
            if len(features.improvement_evidence) > 0:
                return MemoryType.SUCCESSFUL_IMPROVEMENT
            else:
                return MemoryType.BEHAVIORAL_PATTERN
        elif len(features.failure_indicators) > 0:
            return MemoryType.FAILED_EXPERIMENT

        # Pattern detection
        if len(features.causal_chains) > 0:
            return MemoryType.EVOLUTION_PATTERN

        # Default to code pattern for specific implementations
        return MemoryType.CODE_PATTERN

    def _calculate_importance_score(self, features: ClassificationFeatures, memory_record: MemoryRecord) -> float:
        """Calculate importance score (0.0 to 1.0)"""

        score = 0.5  # Base score

        # Success/improvement bonus
        if len(features.success_indicators) > 0:
            score += 0.1 * min(len(features.success_indicators), 3)

        # Failure severity penalty (but still important to remember)
        if len(features.failure_indicators) > 0:
            score += 0.15 * min(len(features.failure_indicators), 2)

        # Architectural relevance bonus
        if len(features.architectural_terms) > 0:
            score += 0.2 * min(len(features.architectural_terms) / 5, 1.0)

        # Performance impact bonus
        if features.performance_metrics:
            score += 0.1

        # Causal relationship bonus
        if len(features.causal_chains) > 0:
            score += 0.15

        # Improvement evidence bonus
        if len(features.improvement_evidence) > 0:
            score += 0.2 * min(len(features.improvement_evidence) / 3, 1.0)

        # Ensure score is in valid range
        return max(0.0, min(1.0, score))

    def _calculate_strategic_value(self, features: ClassificationFeatures, memory_record: MemoryRecord) -> float:
        """Calculate strategic value (0.0 to 1.0)"""

        value = 0.3  # Base value

        # Strategic keyword bonus
        strategic_keywords = [
            k for k in features.content_keywords if k.startswith("strategic")]
        if strategic_keywords:
            value += 0.3 * min(len(strategic_keywords) / 5, 1.0)

        # Architectural terms bonus
        if len(features.architectural_terms) > 2:
            value += 0.25

        # Learning value bonus
        if any("learning" in keyword for keyword in features.content_keywords):
            value += 0.2

        # Long-term pattern bonus
        if len(features.causal_chains) > 0:
            value += 0.15

        # System-wide impact indicators
        system_impact_terms = ["system", "global",
                               "architecture", "framework", "infrastructure"]
        content_text = json.dumps(memory_record.content, default=str).lower()

        impact_count = sum(
            1 for term in system_impact_terms if term in content_text)
        if impact_count > 0:
            value += 0.1 * min(impact_count / 3, 1.0)

        # Ensure value is in valid range
        return max(0.0, min(1.0, value))

    def _determine_importance_level(self, importance_score: float) -> MemoryImportanceLevel:
        """Determine importance level based on score"""
        if importance_score >= 0.9:
            return MemoryImportanceLevel.CRITICAL
        elif importance_score >= 0.7:
            return MemoryImportanceLevel.HIGH
        elif importance_score >= 0.4:
            return MemoryImportanceLevel.MEDIUM
        else:
            return MemoryImportanceLevel.LOW

    def _update_classification_statistics(self, classification_result: Dict[str, Any]):
        """Update classification statistics for analysis and improvement"""

        # Track classification history
        self.classification_history.append({
            "timestamp": datetime.utcnow(),
            "memory_type": classification_result["memory_type"],
            "importance_score": classification_result["importance_score"],
            "strategic_value": classification_result["strategic_value"]
        })

        # Keep only recent history (last 1000 classifications)
        if len(self.classification_history) > 1000:
            self.classification_history = self.classification_history[-1000:]

        # Update frequency counts
        memory_type = classification_result["memory_type"]
        self.memory_type_frequencies[memory_type] = self.memory_type_frequencies.get(
            memory_type, 0) + 1

        # Update importance distributions
        importance_level = classification_result["importance_level"]
        self.importance_distributions[importance_level] = self.importance_distributions.get(
            importance_level, 0) + 1

    def _get_default_classification(self) -> Dict[str, Any]:
        """Get default classification for error cases"""
        return {
            "memory_type": MemoryType.CODE_PATTERN,
            "importance_score": 0.5,
            "strategic_value": 0.3,
            "importance_level": MemoryImportanceLevel.MEDIUM,
            "features": {},
            "classification_timestamp": datetime.utcnow().isoformat(),
            "classifier_version": "1.0",
            "error": "classification_failed"
        }

    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get classification statistics for monitoring and analysis"""

        if not self.classification_history:
            return {"status": "no_data"}

        # Calculate statistics from recent history
        recent_scores = [entry["importance_score"]
                         for entry in self.classification_history[-100:]]
        recent_strategic = [entry["strategic_value"]
                            for entry in self.classification_history[-100:]]

        return {
            "total_classifications": len(self.classification_history),
            "memory_type_frequencies": dict(self.memory_type_frequencies),
            "importance_distributions": dict(self.importance_distributions),
            "avg_importance_score": np.mean(recent_scores) if recent_scores else 0,
            "avg_strategic_value": np.mean(recent_strategic) if recent_strategic else 0,
            "importance_score_std": np.std(recent_scores) if recent_scores else 0,
            "strategic_value_std": np.std(recent_strategic) if recent_strategic else 0,
            "last_classification_time": self.classification_history[-1]["timestamp"].isoformat()
        }

    def should_promote_to_long_term(self, memory_record: MemoryRecord, classification: Dict[str, Any]) -> bool:
        """Determine if memory should be promoted to long-term storage"""

        importance_score = classification.get("importance_score", 0.5)
        strategic_value = classification.get("strategic_value", 0.3)
        memory_type = classification.get(
            "memory_type", MemoryType.CODE_PATTERN)

        # Always promote critical and strategic memories
        if memory_type in [MemoryType.CRITICAL_FAILURE, MemoryType.STRATEGIC_KNOWLEDGE]:
            return True

        # Promote high importance memories
        if importance_score >= self.strategic_threshold:
            return True

        # Promote high strategic value memories
        if strategic_value >= self.strategic_threshold:
            return True

        # Check retention policy
        policy = self.config.retention_policies.get(memory_type.value)
        if policy and importance_score >= policy.importance_threshold:
            return True

        return False

    def should_promote_to_very_long_term(self, memory_record: MemoryRecord, classification: Dict[str, Any], age_days: int) -> bool:
        """Determine if memory should be promoted to very long-term storage"""

        importance_score = classification.get("importance_score", 0.5)
        strategic_value = classification.get("strategic_value", 0.3)
        memory_type = classification.get(
            "memory_type", MemoryType.CODE_PATTERN)

        # Always promote permanent retention types
        permanent_types = [
            MemoryType.STRATEGIC_KNOWLEDGE,
            MemoryType.CRITICAL_FAILURE,
            MemoryType.ARCHITECTURAL_INSIGHT,
            MemoryType.META_LEARNING_RULE
        ]

        if memory_type in permanent_types:
            return age_days >= 30  # After 30 days of validation

        # Promote high-value memories after sufficient time
        if strategic_value >= 0.8 and age_days >= 90:
            return True

        # Promote proven successful improvements
        if (memory_type == MemoryType.SUCCESSFUL_IMPROVEMENT and
                importance_score >= 0.7 and age_days >= 60):
            return True

        return False


class ImportanceEvaluator:
    """
    Evaluates memory importance using multiple criteria and weights.

    Provides more sophisticated importance evaluation than the basic classifier,
    using statistical analysis and historical data.
    """

    def __init__(self):
        """Initialize the importance evaluator"""

        # Evaluation weights
        self.success_weight = 0.3
        self.failure_weight = 0.25
        self.frequency_weight = 0.2
        self.architectural_weight = 0.15
        self.temporal_weight = 0.1

        # Historical data for context
        self.evaluation_history = []

        logger.info("Importance evaluator initialized")

    def evaluate_importance(
        self,
        memory_record: MemoryRecord,
        features: ClassificationFeatures,
        historical_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Evaluate memory importance using multiple criteria.

        Args:
            memory_record: Memory record to evaluate
            features: Extracted features
            historical_context: Historical context for comparison

        Returns:
            Importance score (0.0 to 1.0)
        """

        # Calculate component scores
        success_score = self._calculate_success_score(features)
        failure_score = self._calculate_failure_score(features)
        frequency_score = self._calculate_frequency_score(
            memory_record, historical_context)
        architectural_score = self._calculate_architectural_score(features)
        temporal_score = self._calculate_temporal_score(features)

        # Calculate weighted composite score
        composite_score = (
            success_score * self.success_weight +
            failure_score * self.failure_weight +
            frequency_score * self.frequency_weight +
            architectural_score * self.architectural_weight +
            temporal_score * self.temporal_weight
        )

        # Store evaluation for future context
        self.evaluation_history.append({
            "timestamp": datetime.utcnow(),
            "memory_id": memory_record.memory_id,
            "composite_score": composite_score,
            "component_scores": {
                "success": success_score,
                "failure": failure_score,
                "frequency": frequency_score,
                "architectural": architectural_score,
                "temporal": temporal_score
            }
        })

        return min(1.0, max(0.0, composite_score))

    def _calculate_success_score(self, features: ClassificationFeatures) -> float:
        """Calculate score based on success indicators"""
        success_count = len(features.success_indicators)
        improvement_count = len(features.improvement_evidence)

        # Normalize scores
        success_norm = min(1.0, success_count / 3.0)
        improvement_norm = min(1.0, improvement_count / 2.0)

        return (success_norm + improvement_norm) / 2.0

    def _calculate_failure_score(self, features: ClassificationFeatures) -> float:
        """Calculate score based on failure indicators (failures are also important to remember)"""
        failure_count = len(features.failure_indicators)
        error_pattern_count = len(features.error_patterns)

        # Failures are important for learning
        failure_norm = min(1.0, failure_count / 2.0)
        error_norm = min(1.0, error_pattern_count / 3.0)

        return (failure_norm + error_norm) / 2.0

    def _calculate_frequency_score(self, memory_record: MemoryRecord, historical_context: Optional[Dict[str, Any]]) -> float:
        """Calculate score based on frequency of similar memories"""
        # This would be enhanced with actual frequency analysis
        # For now, return a default moderate score
        return 0.5

    def _calculate_architectural_score(self, features: ClassificationFeatures) -> float:
        """Calculate score based on architectural relevance"""
        arch_term_count = len(features.architectural_terms)
        arch_keyword_count = len(
            [k for k in features.content_keywords if "strategic_architecture" in k])

        arch_norm = min(1.0, arch_term_count / 5.0)
        keyword_norm = min(1.0, arch_keyword_count / 3.0)

        return (arch_norm + keyword_norm) / 2.0

    def _calculate_temporal_score(self, features: ClassificationFeatures) -> float:
        """Calculate score based on temporal relevance"""
        temporal_count = len(features.temporal_markers)
        causal_count = len(features.causal_chains)

        temporal_norm = min(1.0, temporal_count / 3.0)
        causal_norm = min(1.0, causal_count / 2.0)

        return (temporal_norm + causal_norm) / 2.0

    def calculate_composite_score(self, individual_scores: Dict[str, float]) -> float:
        """Calculate composite score from individual criterion scores"""

        weights = {
            "success_impact": self.success_weight,
            "failure_severity": self.failure_weight,
            "frequency_of_occurrence": self.frequency_weight,
            "architectural_relevance": self.architectural_weight,
            "temporal_relevance": self.temporal_weight
        }

        weighted_sum = sum(
            individual_scores.get(criterion, 0.5) * weight
            for criterion, weight in weights.items()
        )

        return min(1.0, max(0.0, weighted_sum))
