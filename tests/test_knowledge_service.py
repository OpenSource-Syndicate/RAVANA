"""Tests for knowledge service functionality."""

import pytest
import asyncio
from services.knowledge_service import KnowledgeService


class TestKnowledgeService:
    """Test knowledge service functionality."""

    @pytest.fixture
    def knowledge_service(self, test_engine):
        """Create knowledge service for testing."""
        service = KnowledgeService(test_engine)
        return service

    def test_knowledge_service_initialization(self, knowledge_service):
        """Test knowledge service initializes correctly."""
        assert knowledge_service is not None
        assert knowledge_service.engine is not None
        assert knowledge_service.embedding_model is not None

    def test_add_knowledge(self, knowledge_service):
        """Test adding knowledge."""
        content = "Quantum computing uses quantum bits that can be in superposition."
        source = "test_source"
        category = "quantum_physics"
        
        result = knowledge_service.add_knowledge(content, source, category)
        
        assert result is not None
        assert 'summary' in result
        assert 'timestamp' in result

    def test_add_duplicate_knowledge(self, knowledge_service):
        """Test adding duplicate knowledge."""
        content = "Test content for duplication"
        
        # Add first time
        result1 = knowledge_service.add_knowledge(content, "test", "test")
        assert result1['duplicate'] is False
        
        # Add second time (duplicate)
        result2 = knowledge_service.add_knowledge(content, "test", "test")
        assert result2['duplicate'] is True

    def test_get_knowledge_by_category(self, knowledge_service):
        """Test retrieving knowledge by category."""
        # Add some knowledge first
        knowledge_service.add_knowledge("Test content", "test", "test_category")
        
        results = knowledge_service.get_knowledge_by_category("test_category", limit=10)
        
        assert results is not None
        assert isinstance(results, list)

    def test_get_recent_knowledge(self, knowledge_service):
        """Test retrieving recent knowledge."""
        # Add some knowledge first
        knowledge_service.add_knowledge("Recent content", "test", "test")
        
        results = knowledge_service.get_recent_knowledge(hours=24, limit=20)
        
        assert results is not None
        assert isinstance(results, list)

    def test_search_knowledge(self, knowledge_service):
        """Test searching knowledge."""
        # Add searchable content
        knowledge_service.add_knowledge(
            "Quantum entanglement is a phenomenon",
            "test",
            "physics"
        )
        
        results = knowledge_service.search_knowledge("quantum", limit=10)
        
        assert results is not None
        assert isinstance(results, list)

    def test_compress_and_save_knowledge(self, knowledge_service):
        """Test knowledge compression."""
        # Add some knowledge to compress
        for i in range(5):
            knowledge_service.add_knowledge(
                f"Test content {i}",
                "test",
                "test"
            )
        
        summary = knowledge_service.compress_and_save_knowledge()
        
        assert summary is not None
        assert 'summary' in summary
        assert 'timestamp' in summary

    @pytest.mark.asyncio
    async def test_knowledge_shutdown(self, knowledge_service):
        """Test knowledge service graceful shutdown."""
        prepared = await knowledge_service.prepare_shutdown()
        assert prepared is True
        
        shutdown = await knowledge_service.shutdown(timeout=10.0)
        assert shutdown is True

    def test_knowledge_shutdown_metrics(self, knowledge_service):
        """Test getting shutdown metrics."""
        metrics = knowledge_service.get_shutdown_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, dict)
