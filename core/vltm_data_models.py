"""
Very Long-Term Memory Data Models

This module defines the data models and schemas for the Snake Agent's 
Very Long-Term Memory System, including memory records, patterns, 
consolidations, and strategic knowledge structures.
"""

from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import json
from pydantic import BaseModel, validator


class MemoryType(str, Enum):
    """Types of memories in the very long-term memory system"""
    STRATEGIC_KNOWLEDGE = "strategic_knowledge"
    ARCHITECTURAL_INSIGHT = "architectural_insight"
    EVOLUTION_PATTERN = "evolution_pattern"
    META_LEARNING_RULE = "meta_learning_rule"
    CRITICAL_FAILURE = "critical_failure"
    SUCCESSFUL_IMPROVEMENT = "successful_improvement"
    FAILED_EXPERIMENT = "failed_experiment"
    CODE_PATTERN = "code_pattern"
    BEHAVIORAL_PATTERN = "behavioral_pattern"


class PatternType(str, Enum):
    """Types of patterns extracted from memories"""
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    BEHAVIORAL = "behavioral"
    STRUCTURAL = "structural"
    PERFORMANCE = "performance"


class ConsolidationType(str, Enum):
    """Types of memory consolidation processes"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    EVENT_TRIGGERED = "event_triggered"


class MemoryImportanceLevel(str, Enum):
    """Importance levels for memory classification"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# First define the junction tables
class ConsolidationPattern(SQLModel, table=True):
    """Junction table linking memory consolidations and patterns"""
    __tablename__ = "consolidation_patterns"

    consolidation_id: str = Field(
        foreign_key="memory_consolidations.consolidation_id", primary_key=True)
    pattern_id: str = Field(
        foreign_key="memory_patterns.pattern_id", primary_key=True)
    extraction_confidence: float = Field(default=1.0)


class PatternStrategicKnowledge(SQLModel, table=True):
    """Junction table linking memory patterns and strategic knowledge"""
    __tablename__ = "pattern_strategic_knowledge"

    pattern_id: str = Field(
        foreign_key="memory_patterns.pattern_id", primary_key=True)
    knowledge_id: str = Field(
        foreign_key="strategic_knowledge.knowledge_id", primary_key=True)
    contribution_weight: float = Field(default=1.0)


# Data Models for Database Tables
class VeryLongTermMemory(SQLModel, table=True):
    """Core very long-term memory record"""
    __tablename__ = "very_long_term_memories"

    memory_id: str = Field(primary_key=True)
    memory_type: MemoryType
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    promoted_at: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    strategic_value: float = Field(default=0.5, ge=0.0, le=1.0)
    compressed_content: str  # JSON string of compressed memory data
    metadata_info: str = Field(default="{}")  # JSON string of metadata
    source_session: str = Field(default="unknown")
    # JSON array of related memory IDs
    related_memories: str = Field(default="[]")
    retention_category: str = Field(
        default="permanent")  # retention policy category

    # Relationships
    patterns: List["MemoryPattern"] = Relationship(
        back_populates="source_memory")

    @validator('compressed_content', 'metadata_info', 'related_memories')
    def validate_json_fields(cls, v):
        """Validate that JSON fields contain valid JSON"""
        if isinstance(v, str):
            try:
                json.loads(v)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format")
        return v


class MemoryPattern(SQLModel, table=True):
    """Patterns extracted from memories"""
    __tablename__ = "memory_patterns"

    pattern_id: str = Field(primary_key=True)
    pattern_type: PatternType
    pattern_description: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    pattern_data: str  # JSON string of pattern-specific data
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    supporting_memories: str = Field(default="[]")  # JSON array of memory IDs
    validation_count: int = Field(default=0)
    last_validated: Optional[datetime] = None

    # Foreign key to source memory
    source_memory_id: Optional[str] = Field(
        foreign_key="very_long_term_memories.memory_id")

    # Relationships
    source_memory: Optional[VeryLongTermMemory] = Relationship(
        back_populates="patterns")
    # Fixed the relationship to use the junction table
    consolidations: List["MemoryConsolidation"] = Relationship(
        back_populates="extracted_patterns",
        link_model=ConsolidationPattern  # Using the junction table
    )
    strategic_knowledge: List["StrategicKnowledge"] = Relationship(
        back_populates="patterns",
        link_model=PatternStrategicKnowledge  # Using the junction table
    )


class MemoryConsolidation(SQLModel, table=True):
    """Records of memory consolidation processes"""
    __tablename__ = "memory_consolidations"

    consolidation_id: str = Field(primary_key=True)
    consolidation_date: datetime = Field(default_factory=datetime.utcnow)
    consolidation_type: ConsolidationType
    memories_processed: int = Field(default=0)
    patterns_extracted: int = Field(default=0)
    compression_ratio: float = Field(default=1.0)
    consolidation_results: str = Field(default="{}")  # JSON string of results
    processing_time_seconds: float = Field(default=0.0)
    success: bool = Field(default=True)
    error_message: Optional[str] = None

    # Relationships
    # Fixed the relationship to use the junction table
    extracted_patterns: List[MemoryPattern] = Relationship(
        back_populates="consolidations",
        link_model=ConsolidationPattern  # Using the junction table
    )


class StrategicKnowledge(SQLModel, table=True):
    """Strategic knowledge derived from patterns"""
    __tablename__ = "strategic_knowledge"

    knowledge_id: str = Field(primary_key=True)
    knowledge_domain: str  # e.g., "architecture", "performance", "learning"
    knowledge_summary: str
    confidence_level: float = Field(ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    source_patterns: str = Field(default="[]")  # JSON array of pattern IDs
    # JSON string of structured knowledge
    knowledge_structure: str = Field(default="{}")
    validation_score: float = Field(default=0.5, ge=0.0, le=1.0)
    # How many times this knowledge was applied
    application_count: int = Field(default=0)

    # Relationships - Fixed to use the junction table
    patterns: List[MemoryPattern] = Relationship(
        back_populates="strategic_knowledge",
        link_model=PatternStrategicKnowledge  # Using the junction table
    )


class ConsolidationMetrics(SQLModel, table=True):
    """Performance metrics for consolidation processes"""
    __tablename__ = "consolidation_metrics"

    metric_id: str = Field(primary_key=True)
    consolidation_id: str = Field(
        foreign_key="memory_consolidations.consolidation_id")
    metric_name: str
    metric_value: float
    metric_unit: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Pydantic Models for API and Internal Use

class MemoryRecord(BaseModel):
    """Pydantic model for memory records in API operations"""
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    importance_score: float = 0.5
    strategic_value: float = 0.5
    source_session: str = "unknown"
    related_memories: List[str] = []


class PatternRecord(BaseModel):
    """Pydantic model for pattern records"""
    pattern_id: str
    pattern_type: PatternType
    description: str
    confidence_score: float
    pattern_data: Dict[str, Any]
    supporting_memories: List[str] = []


class ConsolidationRequest(BaseModel):
    """Request model for memory consolidation"""
    consolidation_type: ConsolidationType
    memory_age_threshold: Optional[int] = None  # days
    max_memories_to_process: Optional[int] = None
    force_consolidation: bool = False


class ConsolidationResult(BaseModel):
    """Result model for consolidation operations"""
    consolidation_id: str
    success: bool
    memories_processed: int
    patterns_extracted: int
    compression_ratio: float
    processing_time_seconds: float
    error_message: Optional[str] = None


class StrategicQuery(BaseModel):
    """Query model for strategic knowledge retrieval"""
    query_text: str
    knowledge_domains: List[str] = []
    min_confidence: float = 0.5
    max_results: int = 10
    include_patterns: bool = False


class MemoryRetentionPolicy(BaseModel):
    """Configuration for memory retention policies"""
    memory_type: MemoryType
    retention_period_days: Optional[int] = None  # None means permanent
    importance_threshold: float = 0.0
    compression_after_days: int = 30
    archive_after_days: Optional[int] = None


class VLTMConfiguration(BaseModel):
    """Configuration for very long-term memory system"""
    retention_policies: Dict[str, MemoryRetentionPolicy] = {}
    consolidation_schedule: Dict[str, str] = {
        "daily": "02:00",
        "weekly": "sunday_03:00",
        "monthly": "first_sunday_04:00",
        "quarterly": "first_day_05:00"
    }
    compression_settings: Dict[str, Any] = {
        "light_compression_after_days": 30,
        "medium_compression_after_days": 180,
        "heavy_compression_after_days": 365
    }
    performance_settings: Dict[str, Any] = {
        "max_consolidation_time_seconds": 3600,
        "max_memories_per_consolidation": 10000,
        "indexing_batch_size": 1000
    }

    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Validate retention policies
        for policy_name, policy in self.retention_policies.items():
            if policy.retention_period_days is not None and policy.retention_period_days < 1:
                issues.append(
                    f"Invalid retention period for {policy_name}: must be >= 1 day")

            if policy.compression_after_days < 1:
                issues.append(
                    f"Invalid compression period for {policy_name}: must be >= 1 day")

        # Validate performance settings
        if self.performance_settings.get("max_consolidation_time_seconds", 0) < 60:
            issues.append("Max consolidation time must be at least 60 seconds")

        if self.performance_settings.get("max_memories_per_consolidation", 0) < 100:
            issues.append(
                "Max memories per consolidation must be at least 100")

        return issues


# Default configuration
DEFAULT_VLTM_CONFIG = VLTMConfiguration(
    retention_policies={
        "strategic_knowledge": MemoryRetentionPolicy(
            memory_type=MemoryType.STRATEGIC_KNOWLEDGE,
            retention_period_days=None,  # Permanent
            importance_threshold=0.7,
            compression_after_days=90
        ),
        "critical_failures": MemoryRetentionPolicy(
            memory_type=MemoryType.CRITICAL_FAILURE,
            retention_period_days=None,  # Permanent
            importance_threshold=0.9,
            compression_after_days=30
        ),
        "successful_improvements": MemoryRetentionPolicy(
            memory_type=MemoryType.SUCCESSFUL_IMPROVEMENT,
            retention_period_days=730,  # 2 years
            importance_threshold=0.6,
            compression_after_days=180
        ),
        "code_patterns": MemoryRetentionPolicy(
            memory_type=MemoryType.CODE_PATTERN,
            retention_period_days=365,  # 1 year
            importance_threshold=0.5,
            compression_after_days=90
        )
    }
)
