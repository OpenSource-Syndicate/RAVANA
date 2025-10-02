"""Pytest configuration and shared fixtures for RAVANA AGI tests."""

import pytest
import asyncio
import os
import sys
from pathlib import Path
from sqlmodel import create_engine, SQLModel, Session
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test database URL
TEST_DATABASE_URL = "sqlite:///test_ravana.db"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine."""
    engine = create_engine(TEST_DATABASE_URL, echo=False)
    SQLModel.metadata.create_all(engine)
    yield engine
    engine.dispose()
    # Cleanup test database
    if os.path.exists("test_ravana.db"):
        os.remove("test_ravana.db")


@pytest.fixture
def test_session(test_engine):
    """Create test database session."""
    with Session(test_engine) as session:
        yield session
        session.rollback()


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    from core.config import Config
    config = Config()
    # Ensure config is properly initialized for tests
    return config


@pytest.fixture
def sample_mood_vector():
    """Sample mood vector for testing."""
    return {
        "Confident": 0.5,
        "Curious": 0.7,
        "Frustrated": 0.2,
        "Excited": 0.6
    }


@pytest.fixture
def sample_memory_data():
    """Sample memory data for testing."""
    return [
        "User prefers technical explanations",
        "System learned about quantum computing",
        "Important deadline on Friday"
    ]


@pytest.fixture
def sample_action_output():
    """Sample action output for testing."""
    return {
        "task_completed": True,
        "status": "success",
        "action": "test_action",
        "result": "Action completed successfully",
        "timestamp": datetime.now().isoformat()
    }
