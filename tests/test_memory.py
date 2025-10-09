import os
import pytest
from modules.episodic_memory.client import extract_memories, save_memories, get_relevant_memories, health_check

# Set embedding model for test (optional)
os.environ['EMBEDDING_MODEL'] = 'all-MiniLM-L6-v2'


def test_health():
    """Test that the memory system health check works."""
    result = health_check()
    print('Health check:', result)


def test_extract_memories():
    """Test extracting memories from conversations."""
    user_input = "I love Italian food and I'm planning a trip to Rome next summer."
    ai_output = "That's wonderful! Rome is beautiful in the summer."
    result = extract_memories(user_input, ai_output)
    print('Extracted memories:', result)
    
    # Basic assertions to verify the extraction worked
    assert result is not None
    assert isinstance(result, dict)
    if 'memories' in result:
        assert isinstance(result['memories'], list)


def test_save_memories():
    """Test saving memories to storage."""
    # Test saving episodic memories
    episodic_memories = [{"type": "episodic", "content": "User mentioned loving Italian food."}]
    result = save_memories(episodic_memories, memory_type='episodic')
    print('Save episodic memories:', result)
    assert result is not None
    
    # Test saving semantic memories
    semantic_memories = [{"type": "semantic", "content": "User loves Italian food."}]
    result = save_memories(semantic_memories, memory_type='semantic')
    print('Save semantic memories:', result)
    assert result is not None


def test_get_relevant_memories():
    """Test retrieving relevant memories based on queries."""
    # Save some test memories first
    test_memories = [
        {"type": "episodic", "content": "User loves Italian food."},
        {"type": "episodic", "content": "User is planning a trip to Rome."}
    ]
    save_result = save_memories(test_memories, memory_type='episodic')
    
    # Test retrieving relevant memories
    result1 = get_relevant_memories("What does the user like to eat?", threshold=0.3)
    print('Relevant memories for food query:', result1)
    assert result1 is not None
    
    result2 = get_relevant_memories("Where is the user planning to travel?", threshold=0.3)
    print('Relevant memories for travel query:', result2)
    assert result2 is not None