# RAVANA Multi-Modal Memory System

Enhanced episodic memory system with multi-modal capabilities including audio processing via Whisper, advanced semantic search, and PostgreSQL-based long-term storage.

## Features

- **Multi-Modal Support**: Text, audio, image, and video content processing
- **Whisper Integration**: State-of-the-art audio transcription and feature extraction
- **Advanced Semantic Search**: Hybrid vector + text search with cross-modal capabilities
- **PostgreSQL + pgvector**: High-performance vector similarity search
- **Unified Embeddings**: Cross-modal semantic understanding with ImageBind-style approach
- **Backward Compatibility**: Maintains compatibility with existing ChromaDB-based system

## Architecture

The system consists of several key components:

- **MultiModalMemoryService**: Main orchestration service
- **EmbeddingService**: Multi-modal embedding generation
- **WhisperAudioProcessor**: Audio transcription and feature extraction
- **AdvancedSearchEngine**: Hybrid and cross-modal search capabilities
- **PostgreSQLStore**: Vector database operations with pgvector

## Installation

### Prerequisites

1. **PostgreSQL with pgvector extension**
   ```bash
   # Install PostgreSQL (version 12+)
   # Install pgvector extension: https://github.com/pgvector/pgvector
   ```

2. **Python Dependencies**
   ```bash
   cd modules/episodic_memory
   pip install -r requirements.txt
   ```

### Database Setup

1. **Create PostgreSQL Database**
   ```sql
   CREATE DATABASE ravana_memory;
   CREATE USER ravana_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE ravana_memory TO ravana_user;
   ```

2. **Set Environment Variables**
   ```bash
   export POSTGRES_URL="postgresql://ravana_user:your_password@localhost:5432/ravana_memory"
   ```

3. **Run Database Migration**
   ```bash
   python setup_database.py --database-url $POSTGRES_URL
   ```

   Options:
   - `--force`: Force recreation of schema
   - `--skip-migration`: Skip ChromaDB migration
   - `--backup-chromadb /path/to/backup`: Backup existing ChromaDB before migration
   - `--verify-only`: Only verify existing migration

## Usage

### Starting the Service

```bash
# Set environment variables
export POSTGRES_URL="postgresql://ravana_user:password@localhost:5432/ravana_memory"

# Start the enhanced memory service
python memory.py
```

The service will run on `http://localhost:8001` with both legacy and new multi-modal endpoints.

### API Endpoints

#### Legacy Endpoints (Backward Compatible)
- `POST /extract_memories/` - Extract memories from conversation
- `POST /save_memories/` - Save memory list
- `POST /get_relevant_memories/` - Search memories
- `GET /health` - Health check

#### New Multi-Modal Endpoints
- `POST /memories/audio/` - Process audio files with Whisper
- `POST /memories/image/` - Process image files
- `POST /search/advanced/` - Advanced search with multiple modes
- `POST /search/cross-modal/` - Cross-modal search
- `GET /memories/{id}/similar` - Find similar memories
- `POST /batch/process/` - Batch file processing
- `GET /statistics/` - System statistics

### Client Usage

```python
from client import (
    upload_audio_memory, upload_image_memory, advanced_search,
    cross_modal_search, hybrid_search, vector_search
)

# Upload audio file
result = upload_audio_memory(
    "path/to/audio.wav",
    context="Meeting discussion about project planning",
    tags=["meeting", "planning"]
)

# Upload image
result = upload_image_memory(
    "path/to/image.jpg",
    description="Whiteboard sketch of system architecture",
    tags=["architecture", "design"]
)

# Advanced search
results = advanced_search(
    query="project planning",
    search_mode="hybrid",
    content_types=["text", "audio"],
    limit=10
)

# Cross-modal search
results = cross_modal_search(
    query_content="meeting notes",
    query_type="text",
    target_types=["audio", "image"],
    limit=5
)
```

### Configuration

#### Environment Variables

```bash
# Database configuration
export POSTGRES_URL="postgresql://user:pass@localhost:5432/db"

# Model configuration
export WHISPER_MODEL_SIZE="base"  # tiny, base, small, medium, large
export TEXT_MODEL_NAME="all-MiniLM-L6-v2"
export DEVICE="cuda"  # cpu, cuda, auto
```

#### Service Configuration

The service can run in two modes:
1. **Legacy Mode**: Only ChromaDB-based functionality (if PostgreSQL not available)
2. **Multi-Modal Mode**: Full multi-modal capabilities with PostgreSQL

## Data Models

### Memory Record Structure

```python
{
    "id": "uuid",
    "content_type": "text|audio|image|video",
    "content_text": "extracted or provided text",
    "file_path": "path/to/original/file",
    "text_embedding": [384 dimensions],
    "image_embedding": [512 dimensions],
    "audio_embedding": [512 dimensions],
    "unified_embedding": [1024 dimensions],
    "created_at": "2024-01-01T12:00:00Z",
    "memory_type": "episodic|semantic|consolidated",
    "confidence_score": 0.95,
    "tags": ["tag1", "tag2"],
    "audio_metadata": {
        "transcript": "transcribed text",
        "language_code": "en",
        "duration_seconds": 30.5,
        "audio_features": {...}
    }
}
```

### Search Request

```python
{
    "query": "search terms",
    "search_mode": "hybrid|vector|text|cross_modal",
    "content_types": ["text", "audio"],
    "limit": 10,
    "similarity_threshold": 0.7,
    "tags": ["optional", "filters"]
}
```

## Performance Considerations

### Embedding Cache
- Text embeddings are cached to avoid recomputation
- Cache size configurable (default: 1000 entries)
- LRU eviction policy

### Database Optimization
- Vector indexes using IVFFlat algorithm
- Full-text search indexes for text content
- Connection pooling for concurrent requests

### Audio Processing
- Maximum audio duration: 5 minutes (configurable)
- Automatic resampling to 16kHz for Whisper
- Temporary file cleanup

## Monitoring and Logging

### Health Checks
```bash
curl http://localhost:8001/health
```

### Statistics
```bash
curl http://localhost:8001/statistics/
```

### Logging Configuration
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Migration from Legacy System

### Automatic Migration
The setup script automatically migrates existing ChromaDB memories:

```bash
python setup_database.py --backup-chromadb ./chromadb_backup
```

### Manual Migration
1. Backup existing ChromaDB
2. Set up PostgreSQL with pgvector
3. Run migration script
4. Verify migration with `--verify-only`
5. Update environment variables
6. Restart service

## Troubleshooting

### Common Issues

1. **pgvector not found**
   ```bash
   # Install pgvector extension
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   make
   make install
   ```

2. **Whisper dependencies**
   ```bash
   pip install torch torchaudio
   pip install openai-whisper librosa soundfile
   ```

3. **Memory issues with large audio files**
   - Reduce `WHISPER_MODEL_SIZE` to "tiny" or "base"
   - Use `DEVICE="cpu"` if GPU memory is limited
   - Set maximum audio duration limits

4. **Database connection issues**
   ```bash
   # Check connection
   psql $POSTGRES_URL -c "SELECT version();"
   ```

### Debug Mode
```bash
export LOG_LEVEL=DEBUG
python memory.py
```

## Development

### Running Tests
```bash
pytest test_multimodal_memory.py -v
```

### Code Structure
```
episodic_memory/
├── memory.py                 # Main FastAPI application
├── models.py                 # Pydantic data models
├── multi_modal_service.py    # Main service orchestrator
├── embedding_service.py      # Multi-modal embeddings
├── whisper_processor.py      # Audio processing
├── postgresql_store.py       # Database operations
├── search_engine.py          # Advanced search
├── setup_database.py         # Migration utilities
├── schema.sql               # Database schema
├── requirements.txt         # Dependencies
├── client.py                # API client
└── test_multimodal_memory.py # Test suite
```

## Contributing

1. Ensure all tests pass
2. Follow existing code style
3. Add tests for new features
4. Update documentation

## License

Same as RAVANA project license.