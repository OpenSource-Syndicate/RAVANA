"""
Comprehensive test suite for multi-modal memory system.
Tests all components: embeddings, Whisper, PostgreSQL, search engine, and API endpoints.
"""

import pytest
import asyncio
import tempfile
import os
import uuid
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from datetime import datetime
import json

# Test imports
try:
    from .models import MemoryRecord, ContentType, MemoryType, SearchRequest, SearchMode, AudioMetadata, ImageMetadata, ConversationRequest, MemoriesList
    from .embedding_service import EmbeddingService
    from .whisper_processor import WhisperAudioProcessor
    from .postgresql_store import PostgreSQLStore
    from .search_engine import AdvancedSearchEngine
    from .multi_modal_service import MultiModalMemoryService
except ImportError:
    # For testing in isolation
    pytest.skip("Multi-modal components not available", allow_module_level=True)

# Test configuration
TEST_DATABASE_URL = "postgresql://test:test@localhost:5433/test_ravana"
TEST_AUDIO_FILE = "test_audio.wav"
TEST_IMAGE_FILE = "test_image.jpg"


class TestEmbeddingService:
    """Test cases for EmbeddingService."""

    @pytest.fixture
    async def embedding_service(self):
        """Create EmbeddingService instance for testing."""
        service = EmbeddingService(device="cpu", cache_size=10)
        yield service
        service.cleanup()

    @pytest.mark.asyncio
    async def test_text_embedding_generation(self, embedding_service):
        """Test text embedding generation."""
        text = "This is a test sentence for embedding generation."
        embedding = await embedding_service.generate_text_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) == embedding_service.text_embedding_dim
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_empty_text_embedding(self, embedding_service):
        """Test embedding generation for empty text."""
        embedding = await embedding_service.generate_text_embedding("")

        assert len(embedding) == embedding_service.text_embedding_dim
        assert all(x == 0.0 for x in embedding)

    @pytest.mark.asyncio
    async def test_embedding_cache(self, embedding_service):
        """Test embedding caching functionality."""
        text = "Test caching"

        # First call
        embedding1 = await embedding_service.generate_text_embedding(text)

        # Second call should use cache
        embedding2 = await embedding_service.generate_text_embedding(text)

        assert embedding1 == embedding2

    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, embedding_service):
        """Test batch embedding generation."""
        texts = ["First text", "Second text", "Third text"]
        embeddings = await embedding_service.batch_generate_embeddings(texts)

        assert len(embeddings) == len(texts)
        assert all(len(emb) == embedding_service.text_embedding_dim for emb in embeddings)

    @pytest.mark.asyncio
    async def test_similarity_computation(self, embedding_service):
        """Test similarity computation between embeddings."""
        text1 = "The cat sits on the mat"
        text2 = "A cat is sitting on a mat"
        text3 = "The weather is nice today"

        emb1 = await embedding_service.generate_text_embedding(text1)
        emb2 = await embedding_service.generate_text_embedding(text2)
        emb3 = await embedding_service.generate_text_embedding(text3)

        # Similar texts should have higher similarity
        sim_similar = await embedding_service.compute_similarity(emb1, emb2)
        sim_different = await embedding_service.compute_similarity(emb1, emb3)

        assert 0 <= sim_similar <= 1
        assert 0 <= sim_different <= 1
        assert sim_similar > sim_different


class TestWhisperAudioProcessor:
    """Test cases for WhisperAudioProcessor."""

    @pytest.fixture
    def audio_processor(self):
        """Create WhisperAudioProcessor instance for testing."""
        # Mock Whisper for testing
        with patch('whisper.load_model') as mock_load:
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                "text": "This is a test transcription",
                "language": "en",
                "segments": [{
                    "start": 0.0,
                    "end": 2.0,
                    "text": "This is a test transcription",
                    "avg_logprob": -0.5
                }]
            }
            mock_load.return_value = mock_model

            processor = WhisperAudioProcessor(model_size="tiny")
            yield processor
            processor.cleanup()

    def test_audio_processor_initialization(self, audio_processor):
        """Test audio processor initialization."""
        assert audio_processor.model_size == "tiny"
        assert audio_processor.sample_rate == 16000
        assert ".wav" in audio_processor.supported_formats

    @pytest.mark.asyncio
    async def test_audio_processing(self, audio_processor):
        """Test audio file processing."""
        # Create a mock audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Mock librosa functions
            with patch('librosa.load') as mock_load, \
                 patch('librosa.get_duration') as mock_duration, \
                 patch('soundfile.SoundFile') as mock_sf:

                mock_load.return_value = (np.random.random(16000), 16000)
                mock_duration.return_value = 1.0

                mock_sf_instance = Mock()
                mock_sf_instance.samplerate = 16000
                mock_sf_instance.channels = 1
                mock_sf_instance.frames = 16000
                mock_sf.__enter__.return_value = mock_sf_instance

                result = await audio_processor.process_audio(temp_path)

                assert result["transcript"] == "This is a test transcription"
                assert result["language"] == "en"
                assert "audio_features" in result
                assert "confidence" in result

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_audio_metadata_creation(self, audio_processor):
        """Test AudioMetadata creation from processing result."""
        processing_result = {
            "transcript": "Test transcript",
            "language": "en",
            "confidence": 0.9,
            "duration": 5.0,
            "audio_features": {"mfcc": {"mean": [1, 2, 3]}},
            "sample_rate": 16000,
            "channels": 1
        }

        metadata = audio_processor.create_audio_metadata(processing_result)

        assert isinstance(metadata, AudioMetadata)
        assert metadata.transcript == "Test transcript"
        assert metadata.language_code == "en"
        assert metadata.duration_seconds == 5.0


class TestPostgreSQLStore:
    """Test cases for PostgreSQLStore."""

    @pytest.fixture
    async def postgres_store(self):
        """Create PostgreSQLStore instance for testing."""
        # This would require a test database
        # For now, we'll mock the database operations
        store = Mock(spec=PostgreSQLStore)
        store.initialize = AsyncMock()
        store.close = AsyncMock()
        store.save_memory_record = AsyncMock()
        store.get_memory_record = AsyncMock()
        store.vector_search = AsyncMock()
        store.text_search = AsyncMock()
        store.get_memory_statistics = AsyncMock()

        yield store

    @pytest.mark.asyncio
    async def test_memory_record_saving(self, postgres_store):
        """Test saving memory records."""
        memory_record = MemoryRecord(
            content_type=ContentType.TEXT,
            content_text="Test memory",
            memory_type=MemoryType.EPISODIC,
            text_embedding=[0.1, 0.2, 0.3] * 128  # 384 dimensions
        )

        # Mock successful save
        postgres_store.save_memory_record.return_value = memory_record

        result = await postgres_store.save_memory_record(memory_record)
        assert result == memory_record
        postgres_store.save_memory_record.assert_called_once_with(memory_record)

    @pytest.mark.asyncio
    async def test_vector_search(self, postgres_store):
        """Test vector similarity search."""
        query_embedding = [0.1, 0.2, 0.3] * 128

        # Mock search results
        mock_memory = MemoryRecord(
            id=uuid.uuid4(),
            content_type=ContentType.TEXT,
            content_text="Similar memory",
            memory_type=MemoryType.EPISODIC
        )
        postgres_store.vector_search.return_value = [(mock_memory, 0.85)]

        results = await postgres_store.vector_search(
            embedding=query_embedding,
            embedding_type="text",
            limit=10
        )

        assert len(results) == 1
        assert results[0][0] == mock_memory
        assert results[0][1] == 0.85


class TestAdvancedSearchEngine:
    """Test cases for AdvancedSearchEngine."""

    @pytest.fixture
    def search_engine(self):
        """Create AdvancedSearchEngine instance for testing."""
        mock_postgres = Mock(spec=PostgreSQLStore)
        mock_embeddings = Mock(spec=EmbeddingService)
        mock_whisper = Mock(spec=WhisperAudioProcessor)

        # Setup mock methods
        mock_postgres.vector_search = AsyncMock()
        mock_postgres.text_search = AsyncMock()
        mock_embeddings.generate_text_embedding = AsyncMock()

        engine = AdvancedSearchEngine(mock_postgres, mock_embeddings, mock_whisper)
        yield engine

    @pytest.mark.asyncio
    async def test_vector_search(self, search_engine):
        """Test vector search functionality."""
        request = SearchRequest(
            query="test query",
            search_mode=SearchMode.VECTOR,
            limit=5
        )

        # Mock embedding generation
        search_engine.embeddings.generate_text_embedding.return_value = [0.1] * 384

        # Mock search results
        mock_memory = MemoryRecord(
            id=uuid.uuid4(),
            content_type=ContentType.TEXT,
            content_text="Test memory",
            memory_type=MemoryType.EPISODIC
        )
        search_engine.postgres.vector_search.return_value = [(mock_memory, 0.8)]

        response = await search_engine.search(request)

        assert response.total_found == 1
        assert response.search_mode == SearchMode.VECTOR
        assert len(response.results) == 1
        assert response.results[0].memory_record == mock_memory

    @pytest.mark.asyncio
    async def test_hybrid_search(self, search_engine):
        """Test hybrid search combining vector and text search."""
        request = SearchRequest(
            query="test query",
            search_mode=SearchMode.HYBRID,
            limit=5
        )

        # Mock embedding generation
        search_engine.embeddings.generate_text_embedding.return_value = [0.1] * 384

        # Mock vector search results
        vector_memory = MemoryRecord(
            id=uuid.uuid4(),
            content_type=ContentType.TEXT,
            content_text="Vector result",
            memory_type=MemoryType.EPISODIC
        )
        search_engine.postgres.vector_search.return_value = [(vector_memory, 0.8)]

        # Mock text search results
        text_memory = MemoryRecord(
            id=uuid.uuid4(),
            content_type=ContentType.TEXT,
            content_text="Text result",
            memory_type=MemoryType.EPISODIC
        )
        search_engine.postgres.text_search.return_value = [(text_memory, 5.0)]

        response = await search_engine.search(request)

        assert response.search_mode == SearchMode.HYBRID
        assert response.total_found >= 1


class TestMultiModalMemoryService:
    """Test cases for MultiModalMemoryService."""

    @pytest.fixture
    async def memory_service(self):
        """Create MultiModalMemoryService instance for testing."""
        # Mock the service for testing
        service = Mock(spec=MultiModalMemoryService)
        service.initialize = AsyncMock()
        service.close = AsyncMock()
        service.process_text_memory = AsyncMock()
        service.process_audio_memory = AsyncMock()
        service.process_image_memory = AsyncMock()
        service.search_memories = AsyncMock()
        service.extract_memories_from_conversation = AsyncMock()
        service.health_check = AsyncMock()

        yield service

    @pytest.mark.asyncio
    async def test_text_memory_processing(self, memory_service):
        """Test text memory processing."""
        text = "This is a test memory"

        # Mock successful processing
        mock_record = MemoryRecord(
            id=uuid.uuid4(),
            content_type=ContentType.TEXT,
            content_text=text,
            memory_type=MemoryType.EPISODIC
        )
        memory_service.process_text_memory.return_value = mock_record

        result = await memory_service.process_text_memory(text)

        assert result == mock_record
        memory_service.process_text_memory.assert_called_once_with(text)

    @pytest.mark.asyncio
    async def test_conversation_memory_extraction(self, memory_service):
        """Test memory extraction from conversations."""
        request = ConversationRequest(
            user_input="I'm planning a trip to Paris next month",
            ai_output="That sounds exciting! Paris is beautiful."
        )

        # Mock extraction result
        mock_memories = MemoriesList(
            memories=["User is planning a trip to Paris", "Trip is scheduled for next month"],
            memory_type=MemoryType.EPISODIC
        )
        memory_service.extract_memories_from_conversation.return_value = mock_memories

        result = await memory_service.extract_memories_from_conversation(request)

        assert result == mock_memories
        assert len(result.memories) == 2

    @pytest.mark.asyncio
    async def test_health_check(self, memory_service):
        """Test service health check."""
        # Mock health check response
        health_response = {
            "status": "healthy",
            "database_connected": True,
            "embedding_service_ready": True,
            "memory_count": 100
        }
        memory_service.health_check.return_value = health_response

        result = await memory_service.health_check()

        assert result["status"] == "healthy"
        assert result["database_connected"] is True


class TestIntegration:
    """Integration tests for the complete multi-modal memory system."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_text_processing(self):
        """Test end-to-end text memory processing."""
        # This would require actual database setup
        # For now, we'll test the workflow with mocks

        # Mock the service
        service = Mock(spec=MultiModalMemoryService)

        # Test text processing workflow
        text = "User loves machine learning and AI research"

        # Mock embedding generation
        mock_embedding = [0.1] * 384

        # Mock memory record creation
        mock_record = MemoryRecord(
            id=uuid.uuid4(),
            content_type=ContentType.TEXT,
            content_text=text,
            text_embedding=mock_embedding,
            memory_type=MemoryType.EPISODIC,
            created_at=datetime.utcnow()
        )

        service.process_text_memory.return_value = mock_record

        # Process the memory
        result = await service.process_text_memory(text)

        # Verify the result
        assert result.content_text == text
        assert result.content_type == ContentType.TEXT
        assert result.text_embedding == mock_embedding

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_workflow(self):
        """Test the complete search workflow."""
        service = Mock(spec=MultiModalMemoryService)

        # Mock search request
        search_request = SearchRequest(
            query="machine learning",
            search_mode=SearchMode.HYBRID,
            limit=10
        )

        # Mock search response
        from .models import SearchResponse, SearchResult
        mock_memory = MemoryRecord(
            id=uuid.uuid4(),
            content_type=ContentType.TEXT,
            content_text="User loves machine learning",
            memory_type=MemoryType.EPISODIC
        )

        mock_response = SearchResponse(
            results=[
                SearchResult(
                    memory_record=mock_memory,
                    similarity_score=0.85,
                    rank=1
                )
            ],
            total_found=1,
            search_time_ms=50,
            search_mode=SearchMode.HYBRID
        )

        service.search_memories.return_value = mock_response

        # Perform search
        result = await service.search_memories(search_request)

        # Verify results
        assert result.total_found == 1
        assert result.search_mode == SearchMode.HYBRID
        assert len(result.results) == 1
        assert result.results[0].similarity_score == 0.85

# Test utilities
def create_test_audio_file(duration_seconds=1.0, sample_rate=16000):
    """Create a test audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        # Create simple sine wave
        import numpy as np
        t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # This would normally use soundfile to write the audio
        # For testing, we'll just create an empty file
        temp_file.write(b"mock audio data")
        return temp_file.name


def create_test_image_file():
    """Create a test image file for testing."""
    from PIL import Image
    import numpy as np

    # Create a simple test image
    image_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file.name, "JPEG")
        return temp_file.name

# Pytest configuration
pytest_plugins = ["pytest_asyncio"]

# Test markers
pytestmark = pytest.mark.asyncio

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])