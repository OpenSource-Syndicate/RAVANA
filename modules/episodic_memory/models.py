"""
Pydantic models for multi-modal memory system.
Defines data structures for memory records, search requests, and responses.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid


class ContentType(str, Enum):
    """Enumeration of supported content types."""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    VIDEO = "video"


class MemoryType(str, Enum):
    """Enumeration of memory types."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    CONSOLIDATED = "consolidated"
    WORKING = "working"


class SearchMode(str, Enum):
    """Enumeration of search modes."""
    VECTOR = "vector"
    TEXT = "text"
    HYBRID = "hybrid"
    CROSS_MODAL = "cross_modal"


class AudioMetadata(BaseModel):
    """Metadata specific to audio content."""
    transcript: Optional[str] = None
    language_code: Optional[str] = None
    confidence_scores: Dict[str, float] = Field(default_factory=dict)
    duration_seconds: Optional[float] = None
    audio_features: Dict[str, Any] = Field(default_factory=dict)
    sample_rate: Optional[int] = None
    channels: Optional[int] = None


class ImageMetadata(BaseModel):
    """Metadata specific to image content."""
    width: Optional[int] = None
    height: Optional[int] = None
    object_detections: Dict[str, Any] = Field(default_factory=dict)
    scene_description: Optional[str] = None
    image_hash: Optional[str] = None
    color_palette: Dict[str, Any] = Field(default_factory=dict)
    image_features: Dict[str, Any] = Field(default_factory=dict)


class VideoMetadata(BaseModel):
    """Metadata specific to video content."""
    duration_seconds: Optional[float] = None
    frame_rate: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    video_features: Dict[str, Any] = Field(default_factory=dict)
    thumbnail_path: Optional[str] = None


class MemoryRecord(BaseModel):
    """Main memory record model."""
    id: Optional[uuid.UUID] = None
    content_type: ContentType
    content_text: Optional[str] = None
    content_metadata: Dict[str, Any] = Field(default_factory=dict)
    file_path: Optional[str] = None

    # Embeddings
    text_embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None
    audio_embedding: Optional[List[float]] = None
    unified_embedding: Optional[List[float]] = None

    # Metadata
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    memory_type: MemoryType = MemoryType.EPISODIC
    emotional_valence: Optional[float] = None
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)

    # Type-specific metadata
    audio_metadata: Optional[AudioMetadata] = None
    image_metadata: Optional[ImageMetadata] = None
    video_metadata: Optional[VideoMetadata] = None

    @validator('emotional_valence')
    def validate_emotional_valence(cls, v):
        if v is not None and not (-1.0 <= v <= 1.0):
            raise ValueError('emotional_valence must be between -1.0 and 1.0')
        return v

    @validator('content_text')
    def validate_content_text(cls, v, values):
        content_type = values.get('content_type')
        if content_type == ContentType.TEXT and not v:
            raise ValueError('content_text is required for text content type')
        return v

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: str
        }


class SearchRequest(BaseModel):
    """Request model for memory search operations."""
    query: str = Field(..., min_length=1, max_length=1000)
    content_types: Optional[List[ContentType]] = None
    memory_types: Optional[List[MemoryType]] = None
    search_mode: SearchMode = SearchMode.HYBRID
    limit: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    include_metadata: bool = True
    tags: Optional[List[str]] = None

    # Cross-modal search specific
    query_content_type: Optional[ContentType] = None
    target_content_types: Optional[List[ContentType]] = None

    # Date filtering
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None


class CrossModalSearchRequest(BaseModel):
    """Request model for cross-modal search operations."""
    query_content: str = Field(...,
                               description="File path or content for query")
    query_type: ContentType
    target_types: List[ContentType]
    limit: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """Individual search result."""
    memory_record: MemoryRecord
    similarity_score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    search_metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Response model for search operations."""
    results: List[SearchResult]
    total_found: int
    search_time_ms: int
    search_mode: SearchMode
    query_metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationRequest(BaseModel):
    """Request for memory extraction from conversations."""
    user_input: str = Field(..., min_length=1)
    ai_output: str = Field(..., min_length=1)
    context: Optional[str] = None
    extract_emotions: bool = True
    memory_type: MemoryType = MemoryType.EPISODIC


class MemoriesList(BaseModel):
    """List of extracted memories."""
    memories: List[str]
    memory_type: MemoryType = MemoryType.EPISODIC
    confidence_scores: Optional[List[float]] = None
    emotional_valences: Optional[List[float]] = None


class ConsolidateRequest(BaseModel):
    """Request for memory consolidation."""
    memory_ids: Optional[List[uuid.UUID]] = None
    max_memories_to_process: int = Field(default=50, ge=1, le=200)
    consolidation_strategy: str = Field(default="llm_based")
    preserve_original: bool = True


class ConsolidationResult(BaseModel):
    """Result of memory consolidation operation."""
    consolidated_memories: List[MemoryRecord]
    original_memory_ids: List[uuid.UUID]
    consolidation_summary: str
    created_at: datetime


class StatusResponse(BaseModel):
    """Generic status response."""
    status: str
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthCheckResponse(BaseModel):
    """Health check response with system status."""
    status: str
    database_connected: bool
    chroma_connected: bool
    embedding_service_ready: bool
    memory_count: int
    uptime_seconds: int
    details: Dict[str, Any] = Field(default_factory=dict)


class FileUploadRequest(BaseModel):
    """Request for file upload and processing."""
    file_type: ContentType
    file_name: str
    file_size: int
    context: Optional[str] = None
    extract_text: bool = True
    generate_embeddings: bool = True
    tags: List[str] = Field(default_factory=list)


class ProcessingResult(BaseModel):
    """Result of file processing operation."""
    memory_record: MemoryRecord
    processing_time_ms: int
    success: bool
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)

# Batch operation models


class BatchProcessRequest(BaseModel):
    """Request for batch processing of multiple files."""
    file_paths: List[str]
    content_types: Optional[List[ContentType]] = None
    parallel_processing: bool = True
    max_workers: int = Field(default=4, ge=1, le=16)


class BatchProcessResult(BaseModel):
    """Result of batch processing operation."""
    results: List[ProcessingResult]
    total_processed: int
    successful_count: int
    failed_count: int
    total_time_ms: int

# Statistics and analytics models


class MemoryStatistics(BaseModel):
    """Memory system statistics."""
    total_memories: int
    by_content_type: Dict[ContentType, int]
    by_memory_type: Dict[MemoryType, int]
    storage_size_mb: float
    avg_confidence_score: float
    most_accessed_memories: List[MemoryRecord]
    recent_additions: List[MemoryRecord]
    consolidation_stats: Dict[str, Any]


class PerformanceMetrics(BaseModel):
    """Performance metrics for the memory system."""
    avg_search_time_ms: float
    avg_embedding_generation_time_ms: float
    avg_storage_time_ms: float
    cache_hit_rate: float
    embedding_cache_size: int
    database_connection_pool_size: int
