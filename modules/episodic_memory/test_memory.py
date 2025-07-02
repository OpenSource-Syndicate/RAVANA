import os
from client import extract_memories, save_memories, get_relevant_memories, health_check

# Set embedding model for test (optional)
os.environ['EMBEDDING_MODEL'] = 'all-MiniLM-L6-v2'

def test_health():
    print('Health check:', health_check())

def test_extract_memories():
    user_input = "I love Italian food and I'm planning a trip to Rome next summer."
    ai_output = "That's wonderful! Rome is beautiful in the summer."
    result = extract_memories(user_input, ai_output)
    print('Extracted memories:', result)
    return result

def test_save_memories(memories, memory_type):
    result = save_memories(memories, memory_type=memory_type)
    print(f'Save memories (type={memory_type}):', result)
    return result

def test_get_relevant_memories(query, top_n=3, threshold=0.6):
    result = get_relevant_memories(query, top_n=top_n, similarity_threshold=threshold)
    print(f'Relevant memories for "{query}":', result)
    return result

if __name__ == "__main__":
    test_health()
    extracted = test_extract_memories()
    if extracted and 'memories' in extracted:
        test_save_memories(extracted['memories'], memory_type='episodic')
    test_save_memories(["User loves Italian food."], memory_type='semantic')
    test_get_relevant_memories("What does the user like to eat?")
    test_get_relevant_memories("Where is the user planning to travel?") 