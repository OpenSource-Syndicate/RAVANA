# Memory Systems



## Update Summary
**Changes Made**   
- Updated documentation to reflect corrected many-to-many relationship declarations in VLTM data models
- Added new section on Very Long-Term Memory (VLTM) system and its integration with existing memory systems
- Enhanced architectural diagrams to include VLTM components and integration patterns
- Added detailed information about memory bridges, flow direction, and synchronization mechanisms
- Updated code examples to reflect the use of junction tables for many-to-many relationships
- Added information about memory classification, conversion, and importance threshold evaluation
- Integrated LLM reliability improvements from core/llm.py into memory system documentation
- Added details about enhanced LLM error handling, JSON parsing, and response validation in memory operations

## Table of Contents
1. [Introduction](#introduction)
2. [Memory Architecture Overview](#memory-architecture-overview)
3. [Episodic Memory System](#episodic-memory-system)
4. [Multi-Modal Embedding Service](#multi-modal-embedding-service)
5. [Advanced Search Engine](#advanced-search-engine)
6. [Multi-Modal Memory Orchestration](#multi-modal-memory-orchestration)
7. [Semantic Memory and Knowledge Compression](#semantic-memory-and-knowledge-compression)
8. [MemoryService Interface and CRUD Operations](#memoryservice-interface-and-crud-operations)
9. [Memory Retrieval Patterns](#memory-retrieval-patterns)
10. [Consolidation Triggers and Retention Policies](#consolidation-triggers-and-retention-policies)
11. [Performance Considerations](#performance-considerations)
12. [Debugging Memory Issues](#debugging-memory-issues)
13. [Very Long-Term Memory System](#very-long-term-memory-system)
14. [Conclusion](#conclusion)

## Introduction
The RAVANA system implements a dual-memory architecture combining episodic and semantic memory systems to enable long-term learning and contextual awareness. This document details the design, implementation, and operational characteristics of these memory systems, focusing on their storage mechanisms, retrieval patterns, and integration points. The system leverages PostgreSQL with pgvector for similarity-based retrieval and employs LLM-driven knowledge compression to transform raw experiences into structured semantic summaries. Recent enhancements have introduced multi-modal memory processing with support for text, audio, and image content, enabling cross-modal search and unified embedding generation. Additionally, the system now includes a Very Long-Term Memory (VLTM) system that integrates with existing memory systems through configurable memory bridges, enabling strategic knowledge consolidation and cross-system synchronization. The memory system has been enhanced with improved LLM reliability features including detailed logging, enhanced error handling, and robust JSON parsing to ensure consistent memory operations.

## Memory Architecture Overview

``mermaid
graph TB
subgraph "Memory Systems"
EM[Episodic Memory]
SM[Semantic Memory]
VLTM[Very Long-Term Memory]
end
subgraph "Processing"
EX[Memory Extraction]
CO[Consolidation]
KC[Knowledge Compression]
ES[Embedding Service]
SE[Search Engine]
MI[Memory Integration]
end
subgraph "Storage"
PG[PostgreSQL with pgvector]
SQ[SQLModel Database]
FS[File System]
end
subgraph "Access"
MS[MemoryService]
MMS[MultiModalService]
KS[KnowledgeService]
VIM[VLTM Integration Manager]
end
UserInput --> EX
EX --> EM
EM --> CO
CO --> PG
KC --> SM
SM --> SQ
MS --> EM
MMS --> EM
KS --> SM
EM --> |Vector Search| SE
SM --> |Semantic Search| KS
PG --> EM
SQ --> SM
FS --> SM
ES --> PG
SE --> PG
MMS --> ES
MMS --> SE
EM --> MI
SM --> MI
MI --> VLTM
VLTM --> KS
VIM --> MI
style EM fill:#f9f,stroke:#333
style SM fill:#bbf,stroke:#333
style VLTM fill:#9f9,stroke:#333
```

**Diagram sources**
- [memory.py](file://modules/episodic_memory/memory.py#L0-L401)
- [multi_modal_service.py](file://modules/episodic_memory/multi_modal_service.py#L0-L656)
- [knowledge_service.py](file://services/knowledge_service.py#L0-L255)
- [postgresql_store.py](file://modules/episodic_memory/postgresql_store.py#L0-L590)
- [vltm_memory_integration_manager.py](file://core/vltm_memory_integration_manager.py#L0-L779)

**Section sources**
- [memory.py](file://modules/episodic_memory/memory.py#L0-L401)
- [multi_modal_service.py](file://modules/episodic_memory/multi_modal_service.py#L0-L656)
- [postgresql_store.py](file://modules/episodic_memory/postgresql_store.py#L0-L590)
- [vltm_memory_integration_manager.py](file://core/vltm_memory_integration_manager.py#L0-L779)

## Episodic Memory System

The episodic memory system captures and stores specific events and interactions as discrete memory records. Each memory is stored with rich metadata and indexed using vector embeddings for similarity-based retrieval. Recent updates have replaced ChromaDB with PostgreSQL enhanced with pgvector extension, enabling robust multi-modal storage and advanced querying capabilities.

### Storage Mechanism with PostgreSQL and SentenceTransformers

Episodic memories are stored in PostgreSQL with pgvector extension, providing a production-grade database solution for vector similarity search. The system uses SentenceTransformers to generate embeddings for memory texts, enabling semantic search capabilities.

``mermaid
classDiagram
class MemoryRecord {
+uuid id
+ContentType content_type
+string content_text
+string file_path
+List[float] text_embedding
+List[float] image_embedding
+List[float] audio_embedding
+List[float] unified_embedding
+datetime created_at
+datetime last_accessed
+int access_count
+MemoryType memory_type
+float emotional_valence
+List[str] tags
}
class PostgreSQLStore {
+save_memory_record(record)
+get_memory_record(id)
+vector_search(embedding)
+get_memory_statistics()
}
class MultiModalMemoryService {
+process_text_memory(text)
+process_audio_memory(path)
+process_image_memory(path)
+search_memories(request)
}
PostgreSQLStore --> MemoryRecord : "stores"
MultiModalMemoryService --> PostgreSQLStore : "uses"
MultiModalMemoryService --> MemoryRecord : "manages"
```

**Diagram sources**
- [models.py](file://modules/episodic_memory/models.py#L0-L250)
- [postgresql_store.py](file://modules/episodic_memory/postgresql_store.py#L0-L590)
- [multi_modal_service.py](file://modules/episodic_memory/multi_modal_service.py#L0-L656)

**Section sources**
- [memory.py](file://modules/episodic_memory/memory.py#L0-L401)
- [postgresql_store.py](file://modules/episodic_memory/postgresql_store.py#L0-L590)
- [models.py](file://modules/episodic_memory/models.py#L0-L250)

#### Embedding Generation and Storage
The system uses the `all-MiniLM-L6-v2` SentenceTransformer model to generate 384-dimensional text embeddings. For multi-modal content, specialized embedding generation is implemented:

```python
class EmbeddingService:
    def __init__(self, text_model_name: str = "all-MiniLM-L6-v2"):
        self.text_model_name = text_model_name
        self.text_embedding_dim = 384
        self.image_embedding_dim = 512
        self.audio_embedding_dim = 512
        self.unified_embedding_dim = 1024
```

Memories are stored with comprehensive metadata including creation timestamp, access statistics, content type, and confidence scores:

```python
memory_record = MemoryRecord(
    content_type=content_type,
    content_text=content_text,
    file_path=file_path,
    memory_type=memory_type,
    tags=tags or [],
    emotional_valence=emotional_valence,
    confidence_score=confidence_score,
    created_at=datetime.utcnow()
)
```

#### Memory Extraction Process
The system extracts memories from conversations using an LLM-powered extraction process. The `extract_memories_from_conversation` method analyzes user-AI interactions and identifies key information to store:

```python
async def extract_memories_from_conversation(self, request: ConversationRequest) -> MemoriesList:
    prompt = f"""
    You are a memory extraction module for an AGI. Your task is to analyze a conversation 
    and identify key pieces of information to be stored in the AGI's long-term memory.

    Focus on extracting:
    - Key facts and information
    - User preferences and characteristics
    - Important goals, plans, or intentions
    - Notable events or experiences
    - Emotional context if relevant

    Guidelines:
    - Each memory should be a single, self-contained statement
    - Keep memories concise (under 30 words)
    - Prefer information that is likely to be relevant long-term
    - Do not store transitory conversational details
    - Output as a JSON object with a "memories" array

    Conversation:
    User: {request.user_input}
    AI: {request.ai_output}
    """
```

The extraction focuses on key facts, user preferences, major goals, and core beliefs while filtering out transitory conversational details.

## Multi-Modal Embedding Service

The EmbeddingService provides multi-modal embedding generation for text, audio, and image content, enabling cross-modal retrieval and unified embedding creation.

### Multi-Modal Embedding Generation

The embedding service supports multiple content types with specialized processing:

``mermaid
flowchart TD
A[Input Content] --> B{Content Type}
B --> C[Text]
B --> D[Audio]
B --> E[Image]
C --> F[Generate Text Embedding<br>using SentenceTransformer]
D --> G[Process Audio with Whisper<br>Extract Features]
E --> H[Extract Image Features<br>Color, Dimensions, Histogram]
F --> I[Store text_embedding]
G --> J[Store audio_embedding]
H --> K[Store image_embedding]
I --> L[Generate Unified Embedding]
J --> L
K --> L
L --> M[Store unified_embedding]
```

**Diagram sources**
- [embedding_service.py](file://modules/episodic_memory/embedding_service.py#L0-L498)
- [models.py](file://modules/episodic_memory/models.py#L0-L250)

**Section sources**
- [embedding_service.py](file://modules/episodic_memory/embedding_service.py#L0-L498)

#### Text Embedding Implementation
Text embeddings are generated using SentenceTransformers with caching for performance:

```python
async def generate_text_embedding(self, text: str) -> List[float]:
    # Check cache first
    cached = self.cache.get(text, self.text_model_name)
    if cached is not None:
        return cached
    
    self._load_text_model()
    
    try:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: self.text_model.encode(text, convert_to_tensor=False, normalize_embeddings=True)
        )
        
        embedding_list = embedding.tolist()
        self.cache.put(text, self.text_model_name, embedding_list)
        
        return embedding_list
        
    except Exception as e:
        logger.error(f"Text embedding generation failed: {e}")
        return [0.0] * self.text_embedding_dim
```

#### Audio Embedding Implementation
Audio embeddings are generated from Whisper transcription and audio features:

```python
async def generate_audio_embedding(self, audio_features: Dict[str, Any]) -> List[float]:
    features = []
    
    # Extract numerical features from audio analysis
    if "mfcc" in audio_features:
        mfcc_data = audio_features["mfcc"]
        if "mean" in mfcc_data:
            features.extend(mfcc_data["mean"])
        if "std" in mfcc_data:
            features.extend(mfcc_data["std"])
    
    if "spectral_centroid" in audio_features:
        sc = audio_features["spectral_centroid"]
        features.extend([sc.get("mean", 0.0), sc.get("std", 0.0)])
    
    # Pad or truncate to desired dimension
    if len(features) < self.audio_embedding_dim:
        features.extend([0.0] * (self.audio_embedding_dim - len(features)))
    else:
        features = features[:self.audio_embedding_dim]
    
    return features
```

#### Image Embedding Implementation
Image embeddings are generated from visual features (placeholder for CLIP in production):

```python
async def generate_image_embedding(self, image_path: str) -> List[float]:
    try:
        image = Image.open(image_path).convert('RGB')
        
        # Extract basic statistics
        img_array = np.array(image)
        features = []
        
        # Color statistics
        for channel in range(3):  # RGB
            channel_data = img_array[:, :, channel].flatten()
            features.extend([
                float(np.mean(channel_data)),
                float(np.std(channel_data)),
                float(np.median(channel_data)),
                float(np.percentile(channel_data, 25)),
                float(np.percentile(channel_data, 75))
            ])
        
        # Image dimensions
        features.extend([
            float(image.width),
            float(image.height),
            float(image.width * image.height)  # Area
        ])
        
        # Histogram features
        hist, _ = np.histogram(img_array.flatten(), bins=32, range=(0, 256))
        hist_normalized = hist / np.sum(hist)
        features.extend(hist_normalized.tolist())
        
        # Pad or truncate to desired dimension
        if len(features) < self.image_embedding_dim:
            features.extend([0.0] * (self.image_embedding_dim - len(features)))
        else:
            features = features[:self.image_embedding_dim]
        
        return features
        
    except Exception as e:
        logger.error(f"Image embedding generation failed for {image_path}: {e}")
        return [0.0] * self.image_embedding_dim
```

#### Unified Embedding Generation
The system generates unified embeddings by combining modalities with weighted fusion:

```python
async def generate_unified_embedding(self, memory_record: MemoryRecord) -> List[float]:
    unified = []
    
    # Combine embeddings with weights
    text_weight = 0.4
    image_weight = 0.3
    audio_weight = 0.3
    
    # Text embedding (weighted)
    if memory_record.text_embedding:
        text_emb = np.array(memory_record.text_embedding) * text_weight
        unified.extend(text_emb.tolist())
    else:
        unified.extend([0.0] * int(self.unified_embedding_dim * text_weight))
    
    # Image embedding (weighted)
    if memory_record.image_embedding:
        image_emb = np.array(memory_record.image_embedding) * image_weight
        unified.extend(image_emb[:int(self.unified_embedding_dim * image_weight)].tolist())
    else:
        unified.extend([0.0] * int(self.unified_embedding_dim * image_weight))
    
    # Audio embedding (weighted)
    if memory_record.audio_embedding:
        audio_emb = np.array(memory_record.audio_embedding) * audio_weight
        unified.extend(audio_emb[:int(self.unified_embedding_dim * audio_weight)].tolist())
    else:
        unified.extend([0.0] * int(self.unified_embedding_dim * audio_weight))
    
    # Normalize the unified embedding
    unified_array = np.array(unified)
    norm = np.linalg.norm(unified_array)
    if norm > 0:
        unified_array = unified_array / norm
    
    return unified_array.tolist()
```

## Advanced Search Engine

The AdvancedSearchEngine provides sophisticated search capabilities including cross-modal search, similarity search, and hybrid search modes.

### Cross-Modal Search Implementation

The search engine supports cross-modal queries where different content types can be used to search across modalities:

```python
async def cross_modal_search(self, request: CrossModalSearchRequest) -> List[SearchResult]:
    # Generate query embedding based on type
    if request.query_type == ContentType.TEXT:
        query_embedding = await self.embeddings.generate_text_embedding(request.query_content)
    elif request.query_type == ContentType.AUDIO and self.whisper:
        audio_result = await self.whisper.process_audio(request.query_content)
        query_embedding = await self.embeddings.generate_text_embedding(
            audio_result.get("transcript", "")
        )
    elif request.query_type == ContentType.IMAGE:
        query_embedding = await self.embeddings.generate_image_embedding(request.query_content)
    else:
        raise ValueError(f"Unsupported query type: {request.query_type}")
    
    # Search using unified embeddings
    results = await self.postgres.vector_search(
        embedding=query_embedding,
        embedding_type="unified",
        limit=request.limit,
        similarity_threshold=request.similarity_threshold,
        content_types=request.target_types
    )
    
    # Convert to SearchResult objects
    search_results = []
    for i, (memory_record, similarity) in enumerate(results):
        search_results.append(SearchResult(
            memory_record=memory_record,
            similarity_score=similarity,
            rank=i + 1,
            search_metadata={
                "search_type": "cross_modal_specialized",
                "query_type": request.query_type.value,
                "target_types": [ct.value for ct in request.target_types]
            }
        ))
    
    return search_results
```

### Similarity Search Implementation
Find memories similar to a given memory record:

```python
async def find_similar_memories(self, 
                              memory_record: MemoryRecord, 
                              limit: int = 10,
                              similarity_threshold: float = 0.7) -> List[SearchResult]:
    # Use the best available embedding
    if memory_record.unified_embedding:
        embedding = memory_record.unified_embedding
        embedding_type = "unified"
    elif memory_record.text_embedding:
        embedding = memory_record.text_embedding
        embedding_type = "text"
    elif memory_record.image_embedding:
        embedding = memory_record.image_embedding
        embedding_type = "image"
    elif memory_record.audio_embedding:
        embedding = memory_record.audio_embedding
        embedding_type = "audio"
    else:
        logger.warning("No embeddings available for similarity search")
        return []
    
    # Search for similar memories
    results = await self.postgres.vector_search(
        embedding=embedding,
        embedding_type=embedding_type,
        limit=limit + 1,  # +1 to exclude the original
        similarity_threshold=similarity_threshold
    )
    
    # Convert to SearchResult objects and exclude the original
    search_results = []
    for i, (similar_record, similarity) in enumerate(results):
        if similar_record.id != memory_record.id:  # Exclude the original
            search_results.append(SearchResult(
                memory_record=similar_record,
                similarity_score=similarity,
                rank=len(search_results) + 1,
                search_metadata={
                    "search_type": "similarity",
                    "reference_id": str(memory_record.id),
                    "embedding_type": embedding_type
                }
            ))
    
    return search_results[:limit]
```

### Hybrid Search Configuration
Configure weights for hybrid search combining vector and text search:

```python
def configure_search_weights(self, vector_weight: float, text_weight: float):
    """
    Configure the weights for hybrid search.
    
    Args:
        vector_weight: Weight for vector similarity (0-1)
        text_weight: Weight for text search (0-1)
    """
    total_weight = vector_weight + text_weight
    if total_weight > 0:
        self.vector_weight = vector_weight / total_weight
        self.text_weight = text_weight / total_weight
        logger.info(f"Updated search weights: vector={self.vector_weight:.2f}, text={self.text_weight:.2f}")
    else:
        logger.warning("Invalid weights provided, keeping current configuration")
```

## Multi-Modal Memory Orchestration

The MultiModalMemoryService orchestrates all components of the memory system, providing a unified interface for multi-modal operations.

### Service Architecture

``mermaid
classDiagram
class MultiModalMemoryService {
+initialize()
+close()
+process_text_memory(text)
+process_audio_memory(path)
+process_image_memory(path)
+extract_memories_from_conversation(request)
+search_memories(request)
+find_similar_memories(id)
+batch_process_files(request)
+health_check()
}
class PostgreSQLStore {
+save_memory_record()
+get_memory_record()
+vector_search()
}
class EmbeddingService {
+generate_text_embedding()
+generate_audio_embedding()
+generate_image_embedding()
+generate_unified_embedding()
}
class WhisperAudioProcessor {
+process_audio()
}
class AdvancedSearchEngine {
+search()
+cross_modal_search()
+find_similar_memories()
}
MultiModalMemoryService --> PostgreSQLStore : "uses"
MultiModalMemoryService --> EmbeddingService : "uses"
MultiModalMemoryService --> WhisperAudioProcessor : "uses"
MultiModalMemoryService --> AdvancedSearchEngine : "uses"
```

**Diagram sources**
- [multi_modal_service.py](file://modules/episodic_memory/multi_modal_service.py#L0-L656)
- [postgresql_store.py](file://modules/episodic_memory/postgresql_store.py#L0-L590)
- [embedding_service.py](file://modules/episodic_memory/embedding_service.py#L0-L498)
- [search_engine.py](file://modules/episodic_memory/search_engine.py#L0-L508)

**Section sources**
- [multi_modal_service.py](file://modules/episodic_memory/multi_modal_service.py#L0-L656)

#### Text Memory Processing
Process and store text-based memories:

```python
async def process_text_memory(self, 
                            text: str,
                            memory_type: MemoryType = MemoryType.EPISODIC,
                            tags: Optional[List[str]] = None,
                            emotional_valence: Optional[float] = None) -> MemoryRecord:
    # Create memory record
    memory_record = MemoryRecord(
        content_type=ContentType.TEXT,
        content_text=text,
        memory_type=memory_type,
        tags=tags or [],
        emotional_valence=emotional_valence,
        created_at=datetime.utcnow()
    )
    
    # Generate embeddings
    memory_record = await self.embedding_service.generate_embeddings(memory_record)
    
    # Save to database
    saved_record = await self.postgres_store.save_memory_record(memory_record)
    
    logger.info(f"Processed text memory: {saved_record.id}")
    return saved_record
```

#### Audio Memory Processing
Process and store audio memories with Whisper transcription:

```python
async def process_audio_memory(self, 
                             audio_path: str,
                             context: Optional[str] = None,
                             memory_type: MemoryType = MemoryType.EPISODIC,
                             tags: Optional[List[str]] = None) -> MemoryRecord:
    # Process audio with Whisper
    audio_result = await self.whisper_processor.process_audio(audio_path, context)
    
    # Create audio metadata
    audio_metadata = self.whisper_processor.create_audio_metadata(audio_result)
    
    # Create memory record
    memory_record = MemoryRecord(
        content_type=ContentType.AUDIO,
        content_text=audio_result.get("transcript"),
        file_path=audio_path,
        memory_type=memory_type,
        tags=tags or [],
        confidence_score=audio_result.get("confidence", 0.8),
        audio_metadata=audio_metadata,
        created_at=datetime.utcnow()
    )
    
    # Generate embeddings
    memory_record = await self.embedding_service.generate_embeddings(memory_record)
    
    # Save to database
    saved_record = await self.postgres_store.save_memory_record(memory_record)
    
    logger.info(f"Processed audio memory: {saved_record.id}")
    return saved_record
```

#### Image Memory Processing
Process and store image memories:

```python
async def process_image_memory(self, 
                             image_path: str,
                             description: Optional[str] = None,
                             memory_type: MemoryType = MemoryType.EPISODIC,
                             tags: Optional[List[str]] = None) -> MemoryRecord:
    # Create basic image metadata
    from PIL import Image
    with Image.open(image_path) as img:
        width, height = img.size
    
    image_metadata = ImageMetadata(
        width=width,
        height=height,
        scene_description=description
    )
    
    # Create memory record
    memory_record = MemoryRecord(
        content_type=ContentType.IMAGE,
        content_text=description,
        file_path=image_path,
        memory_type=memory_type,
        tags=tags or [],
        image_metadata=image_metadata,
        created_at=datetime.utcnow()
    )
    
    # Generate embeddings
    memory_record = await self.embedding_service.generate_embeddings(memory_record)
    
    # Save to database
    saved_record = await self.postgres_store.save_memory_record(memory_record)
    
    logger.info(f"Processed image memory: {saved_record.id}")
    return saved_record
```

#### Batch Processing
Process multiple files in batch with parallel processing support:

```python
async def batch_process_files(self, request: BatchProcessRequest) -> BatchProcessResult:
    # Process files
    if request.parallel_processing:
        # Process in parallel with limited concurrency
        semaphore = asyncio.Semaphore(request.max_workers)
        
        async def process_with_semaphore(task):
            async with semaphore:
                return await task
        
        parallel_tasks = [process_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
    else:
        # Process sequentially
        for task in tasks:
            result = await task
            results.append(result)
    
    # Process results
    processing_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processing_results.append(ProcessingResult(
                memory_record=None,
                processing_time_ms=0,
                success=False,
                error_message=str(result)
            ))
            failed_count += 1
        else:
            processing_results.append(result)
            if result.success:
                successful_count += 1
            else:
                failed_count += 1
    
    return BatchProcessResult(
        results=processing_results,
        total_processed=len(request.file_paths),
        successful_count=successful_count,
        failed_count=failed_count,
        total_time_ms=int(total_time)
    )
```

## Semantic Memory and Knowledge Compression

The semantic memory system transforms episodic experiences into structured knowledge through a compression pipeline that identifies patterns, generalizes information, and creates concise summaries.

### Knowledge Compression Pipeline

The knowledge compression pipeline converts raw experiences into semantic summaries using LLM-driven analysis:

``mermaid
flowchart TD
A[Raw Experiences] --> B{Knowledge<br>Compression}
B --> C[Pattern Recognition]
C --> D[Merge Related Facts]
D --> E[Deduplicate Redundant Info]
E --> F[Generalize Specifics]
F --> G[Create Summary]
G --> H[Store in Semantic Memory]
```

**Diagram sources**
- [main.py](file://modules/knowledge_compression/main.py#L0-L42)
- [compression_prompts.py](file://modules/knowledge_compression/compression_prompts.py#L0-L5)

**Section sources**
- [main.py](file://modules/knowledge_compression/main.py#L0-L42)
- [compressed_memory.py](file://modules/knowledge_compression/compressed_memory.py#L0-L17)

#### Compression Implementation
The compression process is implemented in the `compress_knowledge` function, which uses an LLM to summarize accumulated logs:

```python
def compress_knowledge(logs):
    prompt = COMPRESSION_PROMPT.format(logs=json.dumps(logs, indent=2))
    summary = call_llm(prompt)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "summary": summary
    }
    save_summary(entry)
    return entry
```

The compression prompt instructs the LLM to produce structured summaries of new facts learned, key outcomes, and next goals:

```python
COMPRESSION_PROMPT = (
    "You are an AI tasked with summarizing accumulated knowledge and logs. "
    "Given the following logs, produce a concise summary report of new facts learned, key outcomes, and next goals.\n"
    "Logs: {logs}\n"
    "Respond in a clear, structured format."
)
```

#### Storage Mechanism
Compressed knowledge is stored as JSON files on the filesystem, with each summary entry containing a timestamp and the LLM-generated summary:

```python
def save_summary(entry):
    data = load_summaries()
    data.append(entry)
    with open(COMPRESSED_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
```

The system also integrates with SQLModel for database-backed semantic memory storage, where summaries are stored in a relational database with metadata:

```python
class Summary(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: str
    summary_text: str
    source: str
    category: str
    content_hash: str = Field(unique=True)
```

## MemoryService Interface and CRUD Operations

The `MemoryService` provides a unified interface for memory management operations, abstracting the underlying storage mechanisms.

``mermaid
classDiagram
class MemoryService {
+get_relevant_memories(query_text)
+save_memories(memories)
+extract_memories(user_input, ai_output)
+consolidate_memories()
}
class MemoryAPI {
+get_relevant_memories_api(request)
+save_memories_api(request)
+extract_memories_api(request)
+consolidate_memories_api(request)
}
MemoryService --> MemoryAPI : "delegates"
```

**Diagram sources**
- [memory_service.py](file://services/memory_service.py#L0-L20)
- [memory.py](file://modules/episodic_memory/memory.py#L0-L401)

**Section sources**
- [memory_service.py](file://services/memory_service.py#L0-L20)

### CRUD Operations

The MemoryService implements the following CRUD operations:

#### Create: save_memories
Stores new memories in the episodic memory system:

```python
async def save_memories(self, memories):
    await asyncio.to_thread(save_memories, memories)
```

The operation runs in a separate thread to avoid blocking the event loop.

#### Read: get_relevant_memories
Retrieves memories relevant to a query using vector similarity search:

```python
async def get_relevant_memories(self, query_text: str):
    return await get_relevant_memories_api({"query_text": query_text})
```

#### Update: extract_memories
Extracts and updates memories from new interactions:

```python
async def extract_memories(self, user_input: str, ai_output: str):
    return await extract_memories_api({"user_input": user_input, "ai_output": ai_output})
```

#### Delete: consolidate_memories
Removes redundant memories during consolidation:

```python
async def consolidate_memories(self):
    from modules.episodic_memory.memory import ConsolidateRequest
    return await consolidate_memories_api(ConsolidateRequest())
```

## Memory Retrieval Patterns

The system implements similarity-based retrieval patterns for efficient memory access.

### Vector Search Implementation

Memory retrieval uses PostgreSQL's pgvector extension for vector search capabilities:

```python
async def vector_search(self, 
                       embedding: List[float],
                       embedding_type: str = "text",
                       limit: int = 10,
                       similarity_threshold: float = 0.7,
                       content_types: Optional[List[ContentType]] = None) -> List[Tuple[MemoryRecord, float]]:
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
```

### Retrieval Parameters

The retrieval process is configurable through the following parameters:

- **top_n**: Maximum number of memories to return (default: 5)
- **similarity_threshold**: Minimum similarity score for inclusion (default: 0.7)

The similarity threshold acts as a filter to ensure only highly relevant memories are retrieved, preventing information overload.

### Access Pattern Tracking

The system tracks memory access patterns by updating metadata on retrieval:

```python
# Update access metadata for retrieved memories
if ids_to_update:
    chroma_collection.update(ids=ids_to_update, metadatas=metadatas_to_update)
```

Each retrieved memory has its `last_accessed` timestamp and `access_count` updated, enabling usage-based retention policies.

## Consolidation Triggers and Retention Policies

The system implements automated memory consolidation to prevent memory bloat and improve efficiency.

### Consolidation Process

The consolidation process uses an LLM to merge, deduplicate, and generalize memories:

```python
@app.post("/consolidate_memories/", response_model=StatusResponse, tags=["Memories"])
async def consolidate_memories_api(request: ConsolidateRequest):
    memories_data = chroma_collection.get(
        limit=request.max_memories_to_process,
        include=["metadatas"]
    )
    
    prompt = PROMPT_FOR_CONSOLIDATION + "\n" + json.dumps(memories_to_process, indent=2)
    llm_response_str = await asyncio.to_thread(call_llm, prompt)
    consolidation_plan = parse_llm_json_response(llm_response_str)
    
    # Save new consolidated memories
    if consolidation_plan["consolidated"]:
        save_memories(consolidation_plan["consolidated"], memory_type='long-term-consolidated')
    
    # Delete old memories
    if consolidation_plan["to_delete"]:
        chroma_collection.delete(ids=unique_to_delete_ids)
```

The consolidation prompt provides specific instructions for merging related memories, removing duplicates, and generalizing specific facts.

### Trigger Mechanism

Consolidation is triggered programmatically by the system:

```python
async def consolidate_memories(self):
    return await consolidate_memories_api(ConsolidateRequest())
```

In the core system, consolidation is called at strategic points in the execution flow:

```python
consolidation_result = await self.memory_service.consolidate_memories()
```

### Retention Policies

The system implements retention through:
- **Usage-based prioritization**: Frequently accessed memories are retained
- **Redundancy elimination**: Duplicate or overlapping memories are removed
- **Temporal relevance**: Older, less accessed memories are prioritized for consolidation

The system currently fetches a random batch of memories for consolidation due to ChromaDB's lack of metadata-based sorting, but this could be enhanced with custom indexing.

## Performance Considerations

The memory system incorporates several performance optimizations and scalability considerations.

### Vector Search Optimization

``mermaid
flowchart TD
A[Query Text] --> B[Generate Embedding]
B --> C[Vector Search in PostgreSQL]
C --> D[Filter by Similarity]
D --> E[Update Access Metadata]
E --> F[Return Results]
```

**Diagram sources**
- [postgresql_store.py](file://modules/episodic_memory/postgresql_store.py#L0-L590)

**Section sources**
- [postgresql_store.py](file://modules/episodic_memory/postgresql_store.py#L0-L590)

#### Indexing Strategies
- PostgreSQL with pgvector automatically indexes embeddings for fast similarity search
- GIN indexes are used for tag-based filtering
- The system could benefit from implementing HNSW or other approximate nearest neighbor algorithms for larger datasets

#### Performance Metrics
- Query latency: Optimized through in-memory vector indexing
- Memory footprint: Controlled through periodic consolidation
- Throughput: Async operations prevent blocking the main event loop

### Semantic Search with FAISS

For semantic memory, the system uses FAISS for efficient vector search:

```python
# Initialize FAISS index for semantic search
self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
```

The FAISS index is persisted to disk and automatically loaded on startup:

```python
if os.path.exists(self.index_file) and os.path.exists(self.id_map_file):
    self.faiss_index = faiss.read_index(self.index_file)
    with open(self.id_map_file, "rb") as f:
        self.id_map = pickle.load(f)
```

### Memory Bloat Prevention

The system prevents memory bloat through:
- **Consolidation**: Regular merging of related memories
- **Deduplication**: Removal of redundant information
- **Access tracking**: Usage-based retention prioritization
- **Batch processing**: Limiting the number of memories processed at once

## Debugging Memory Issues

The system provides several mechanisms for debugging memory-related issues.

### Health Monitoring

The memory service includes a health check endpoint:

```python
async def health_check(self) -> Dict[str, Any]:
    """Perform comprehensive health check."""
    try:
        # Check database connection
        db_stats = await self.postgres_store.get_memory_statistics()
        db_connected = bool(db_stats)
        
        # Check embedding service
        test_embedding = await self.embedding_service.generate_text_embedding("test")
        embedding_ready = len(test_embedding) > 0
        
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "status": "healthy" if db_connected and embedding_ready else "degraded",
            "database_connected": db_connected,
            "embedding_service_ready": embedding_ready,
            "memory_count": db_stats.get("total_memories", 0),
            "uptime_seconds": int(uptime),
            "initialized": self.initialized
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "initialized": self.initialized
        }
```

This endpoint verifies database connectivity and reports the current memory count.

### Diagnostic Endpoints

Additional diagnostic capabilities include:
- **list_memories_api**: Retrieves all stored memories for inspection
- **Logging**: Comprehensive logging of memory operations
- **Status responses**: Detailed operation results with metadata

### Common Issues and Solutions

#### Retrieval Inaccuracies
- **Cause**: Low similarity threshold or poor embedding quality
- **Solution**: Adjust similarity_threshold parameter or retrain embeddings

#### Memory Leaks
- **Cause**: Failed consolidation or improper memory deletion
- **Solution**: Verify consolidation process and check deletion logs

#### Performance Degradation
- **Cause**: Large memory database without proper indexing
- **Solution**: Implement approximate nearest neighbor search or database partitioning

## Very Long-Term Memory System

The Very Long-Term Memory (VLTM) system provides strategic knowledge management and cross-system memory integration. It uses SQLModel with junction tables to properly implement many-to-many relationships between memory patterns, consolidations, and strategic knowledge.

### Data Model Relationships

The VLTM data models use junction tables to correctly implement many-to-many relationships:

``mermaid
classDiagram
class VeryLongTermMemory {
+memory_id: str
+memory_type: MemoryType
+created_at: datetime
+last_accessed: datetime
+access_count: int
+importance_score: float
+compressed_content: str
+metadata_info: str
}
class MemoryPattern {
+pattern_id: str
+pattern_type: PatternType
+pattern_description: str
+confidence_score: float
+pattern_data: str
+discovered_at: datetime
}
class MemoryConsolidation {
+consolidation_id: str
+consolidation_date: datetime
+consolidation_type: ConsolidationType
+memories_processed: int
+patterns_extracted: int
+compression_ratio: float
+success: bool
}
class StrategicKnowledge {
+knowledge_id: str
+knoledge_domain: str
+knoledge_summary: str
+confidence_level: float
+last_updated: datetime
}
class ConsolidationPattern {
+consolidation_id: str
+pattern_id: str
+extraction_confidence: float
}
class PatternStrategicKnowledge {
+pattern_id: str
+knoledge_id: str
+contribution_weight: float
}

VeryLongTermMemory "1" -- "0..*" MemoryPattern : contains
MemoryPattern "0..*" -- "0..*" MemoryConsolidation : extracted in
MemoryPattern "0..*" -- "0..*" StrategicKnowledge : contributes to
MemoryConsolidation "1" -- "0..*" ConsolidationPattern : has
ConsolidationPattern "0..*" -- "0..*" MemoryPattern : links
StrategicKnowledge "1" -- "0..*" PatternStrategicKnowledge : has
PatternStrategicKnowledge "0..*" -- "0..*" MemoryPattern : links
```

**Diagram sources**
- [vltm_data_models.py](file://core/vltm_data_models.py#L0-L325) - *Updated in recent commit*

**Section sources**
- [vltm_data_models.py](file://core/vltm_data_models.py#L0-L325) - *Updated in recent commit*

#### Junction Table Implementation
The system uses junction tables to properly implement many-to-many relationships:

```python
class ConsolidationPattern(SQLModel, table=True):
    """Junction table linking memory consolidations and patterns"""
    __tablename__ = "consolidation_patterns"
    
    consolidation_id: str = Field(foreign_key="memory_consolidations.consolidation_id", primary_key=True)
    pattern_id: str = Field(foreign_key="memory_patterns.pattern_id", primary_key=True)
    extraction_confidence: float = Field(default=1.0)


class PatternStrategicKnowledge(SQLModel, table=True):
    """Junction table linking memory patterns and strategic knowledge"""
    __tablename__ = "pattern_strategic_knowledge"
    
    pattern_id: str = Field(foreign_key="memory_patterns.pattern_id", primary_key=True)
    knowledge_id: str = Field(foreign_key="strategic_knowledge.knowledge_id", primary_key=True)
    contribution_weight: float = Field(default=1.0)
```

The relationships are properly defined using the `link_model` parameter:

```python
class MemoryPattern(SQLModel, table=True):
    # Fixed the relationship to use the junction table
    consolidations: List["MemoryConsolidation"] = Relationship(
        back_populates="extracted_patterns",
        link_model=ConsolidationPattern  # Using the junction table
    )
    strategic_knowledge: List["StrategicKnowledge"] = Relationship(
        back_populates="patterns",
        link_model=PatternStrategicKnowledge  # Using the junction table
    )
```

### Memory Integration Manager

The MemoryIntegrationManager coordinates memory flow between existing memory systems and the VLTM system.

#### Integration Architecture

``mermaid
classDiagram
class MemoryIntegrationManager {
+initialize()
+shutdown()
+trigger_manual_sync()
+get_integration_status()
}
class MemoryBridge {
+source_system: str
+target_system: str
+flow_direction: MemoryFlowDirection
+memory_types: List[MemoryType]
+sync_interval_minutes: int
+batch_size: int
+enabled: bool
}
class IntegrationStats {
+memories_synchronized: int
+patterns_extracted: int
+knoledge_consolidated: int
+failed_operations: int
+processing_time_seconds: float
+last_sync_timestamp: datetime
}
class MemoryFlowDirection {
+TO_VLTM: "to_vltm"
+FROM_VLTM: "from_vltm"
+BIDIRECTIONAL: "bidirectional"
}
class IntegrationMode {
+REAL_TIME: "real_time"
+BATCH: "batch"
+HYBRID: "hybrid"
+SELECTIVE: "selective"
}

MemoryIntegrationManager --> MemoryBridge : "manages"
MemoryIntegrationManager --> IntegrationStats : "tracks"
MemoryIntegrationManager --> MemoryFlowDirection : "uses"
MemoryIntegrationManager --> IntegrationMode : "uses"
```

**Diagram sources**
- [vltm_memory_integration_manager.py](file://core/vltm_memory_integration_manager.py#L0-L779) - *Updated in recent commit*

**Section sources**
- [vltm_memory_integration_manager.py](file://core/vltm_memory_integration_manager.py#L0-L779) - *Updated in recent commit*

#### Memory Bridge Configuration
The system uses configurable memory bridges to control memory flow:

```python
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
```

Default bridges are set up during initialization:

```python
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
```

#### Memory Synchronization
The integration manager continuously synchronizes memories across bridges:

```python
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
```

#### Memory Classification and Conversion
The system classifies episodic memories into VLTM memory types:

```python
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
```

Memories are converted between systems with appropriate metadata:

```python
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
```

## Conclusion
The RAVANA memory system implements a sophisticated dual-architecture approach combining episodic and semantic memory systems. The episodic memory captures specific experiences using PostgreSQL with pgvector and SentenceTransformers for vector-based storage and retrieval, while the semantic memory system uses LLM-driven knowledge compression to create structured summaries. The system has been enhanced with multi-modal capabilities, supporting text, audio, and image content with cross-modal search and unified embedding generation. The MultiModalMemoryService provides a comprehensive interface for memory operations, and the system incorporates automated consolidation to prevent memory bloat. Performance is optimized through vector indexing and async operations, with comprehensive logging and diagnostic capabilities for debugging. Additionally, the system now includes a Very Long-Term Memory (VLTM) system that properly implements many-to-many relationships using junction tables and provides strategic knowledge management through the MemoryIntegrationManager. This architecture enables the system to maintain long-term context, learn from experiences, and provide increasingly personalized responses over time.

**Referenced Files in This Document**   
- [memory.py](file://modules/episodic_memory/memory.py#L0-L401) - *Updated in recent commit*
- [client.py](file://modules/episodic_memory/client.py#L0-L150) - *Updated in recent commit*
- [embedding_service.py](file://modules/episodic_memory/embedding_service.py#L0-L498) - *Added in recent commit*
- [search_engine.py](file://modules/episodic_memory/search_engine.py#L0-L508) - *Added in recent commit*
- [multi_modal_service.py](file://modules/episodic_memory/multi_modal_service.py#L0-L656) - *Added in recent commit*
- [models.py](file://modules/episodic_memory/models.py#L0-L250) - *Added in recent commit*
- [postgresql_store.py](file://modules/episodic_memory/postgresql_store.py#L0-L590) - *Added in recent commit*
- [memory_service.py](file://services/memory_service.py#L0-L20)
- [compressed_memory.py](file://modules/knowledge_compression/compressed_memory.py#L0-L17)
- [main.py](file://modules/knowledge_compression/main.py#L0-L42)
- [compression_prompts.py](file://modules/knowledge_compression/compression_prompts.py#L0-L5)
- [test_memory.py](file://modules/episodic_memory/test_memory.py#L0-L30)
- [knowledge_service.py](file://services/knowledge_service.py#L0-L255)
- [vltm_data_models.py](file://core/vltm_data_models.py#L0-L325) - *Updated in recent commit*
- [vltm_memory_integration_manager.py](file://core/vltm_memory_integration_manager.py#L0-L779) - *Updated in recent commit*
- [llm.py](file://core/llm.py#L0-L1636) - *Updated in recent commit*