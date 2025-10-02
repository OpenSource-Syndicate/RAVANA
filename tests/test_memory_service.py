"""Tests for memory service functionality."""

import pytest
import asyncio
from services.memory_service import MemoryService
from core.enhanced_memory_service import MemoryType, Memory


class TestMemoryService:
    """Test memory service functionality."""

    @pytest.fixture
    def memory_service(self):
        """Create memory service for testing."""
        service = MemoryService()
        return service

    def test_memory_service_initialization(self, memory_service):
        """Test memory service initializes correctly."""
        assert memory_service is not None
        assert hasattr(memory_service, 'save_memories')
        assert hasattr(memory_service, 'retrieve_relevant_memories')

    @pytest.mark.asyncio
    async def test_save_memories(self, memory_service, sample_memory_data):
        """Test saving memories."""
        memories = [
            {'content': mem, 'type': MemoryType.EPISODIC}
            for mem in sample_memory_data
        ]
        
        result = await memory_service.save_memories(memories)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_retrieve_relevant_memories(self, memory_service, sample_memory_data):
        """Test retrieving relevant memories."""
        # First save some memories
        memories = [
            {'content': mem, 'type': MemoryType.EPISODIC}
            for mem in sample_memory_data
        ]
        await memory_service.save_memories(memories)
        
        # Then retrieve
        results = await memory_service.get_relevant_memories(
            query_text="quantum computing",
            top_k=5
        )
        
        assert results is not None
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_extract_memories(self, memory_service):
        """Test memory extraction from conversation."""
        user_input = "I'm working on a quantum computing project"
        ai_output = "That sounds fascinating! Tell me more."
        
        result = await memory_service.extract_memories(user_input, ai_output)
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_memory_types(self, memory_service):
        """Test different memory types."""
        memory_types = [
            MemoryType.EPISODIC,
            MemoryType.SEMANTIC,
            MemoryType.PROCEDURAL,
            MemoryType.CONVERSATIONAL,
            MemoryType.EMOTIONAL,
            MemoryType.REFLECTIVE,
            MemoryType.SHORT_TERM,
            MemoryType.WORKING,
            MemoryType.LONG_TERM
        ]
        
        for mem_type in memory_types:
            memories = [{
                'content': f'Test {mem_type.value} memory',
                'type': mem_type
            }]
            result = await memory_service.save_memories(memories)
            assert result is True

    @pytest.mark.asyncio
    async def test_memory_type_retrieval(self, memory_service):
        """Test retrieval of specific memory types."""
        # Save memories of different types
        test_memories = [
            {'content': 'This is an episodic memory', 'type': MemoryType.EPISODIC},
            {'content': 'This is a semantic memory', 'type': MemoryType.SEMANTIC},
            {'content': 'This is an emotional memory', 'type': MemoryType.EMOTIONAL},
            {'content': 'This is a working memory', 'type': MemoryType.WORKING}
        ]
        
        await memory_service.save_memories(test_memories)
        
        # Test retrieval by type
        episodic_memories = memory_service.get_memories_by_type(MemoryType.EPISODIC)
        assert len(episodic_memories) >= 1
        assert any('episodic' in mem.content.lower() for mem in episodic_memories)
        
        emotional_memories = memory_service.get_memories_by_type(MemoryType.EMOTIONAL)
        assert len(emotional_memories) >= 1
        assert any('emotional' in mem.content.lower() for mem in emotional_memories)

    @pytest.mark.asyncio
    async def test_memory_shutdown(self, memory_service):
        """Test memory service graceful shutdown."""
        prepared = await memory_service.prepare_shutdown()
        assert prepared is True
        
        shutdown = await memory_service.shutdown(timeout=10.0)
        assert shutdown is True

    def test_memory_shutdown_metrics(self, memory_service):
        """Test getting shutdown metrics."""
        metrics = memory_service.get_shutdown_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, dict)
