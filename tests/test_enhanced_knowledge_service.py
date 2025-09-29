"""
Tests for enhanced knowledge service functionality.
"""

import pytest
import asyncio
from services.knowledge_service import KnowledgeService
from database.engine import create_db_and_tables, engine


@pytest.fixture
def knowledge_service():
    """Create a knowledge service for testing."""
    return KnowledgeService(engine)


@pytest.mark.asyncio
async def test_add_knowledge_basic(knowledge_service):
    """Test basic knowledge addition."""
    content = "This is a test piece of knowledge about artificial intelligence."

    result = await asyncio.to_thread(
        knowledge_service.add_knowledge,
        content=content,
        source="test",
        category="test_category"
    )

    assert result is not None
    assert 'summary' in result
    assert result['source'] == 'test'
    assert result['category'] == 'test_category'
    assert result['duplicate'] == False


@pytest.mark.asyncio
async def test_add_knowledge_duplicate(knowledge_service):
    """Test duplicate knowledge detection."""
    content = "This is duplicate content for testing."

    # Add first time
    result1 = await asyncio.to_thread(
        knowledge_service.add_knowledge,
        content=content,
        source="test",
        category="test_category"
    )

    # Add same content again
    result2 = await asyncio.to_thread(
        knowledge_service.add_knowledge,
        content=content,
        source="test",
        category="test_category"
    )

    assert result1['duplicate'] == False
    assert result2['duplicate'] == True


@pytest.mark.asyncio
async def test_get_knowledge_by_category(knowledge_service):
    """Test retrieving knowledge by category."""
    # Add some test knowledge
    await asyncio.to_thread(
        knowledge_service.add_knowledge,
        content="Test content for category retrieval",
        source="test",
        category="retrieval_test"
    )

    results = knowledge_service.get_knowledge_by_category("retrieval_test")
    assert len(results) > 0
    assert results[0]['category'] == 'retrieval_test'


@pytest.mark.asyncio
async def test_search_knowledge(knowledge_service):
    """Test knowledge search functionality."""
    # Add searchable content
    await asyncio.to_thread(
        knowledge_service.add_knowledge,
        content="Machine learning algorithms are powerful tools for data analysis",
        source="test",
        category="search_test"
    )

    results = knowledge_service.search_knowledge("machine learning")
    assert len(results) > 0
    assert any("machine learning" in r['summary'].lower() for r in results)


def test_knowledge_service_initialization():
    """Test that knowledge service initializes properly."""
    ks = KnowledgeService(engine)
    assert ks.engine is not None


if __name__ == "__main__":
    # Create database tables for testing
    create_db_and_tables()
    pytest.main([__file__])
