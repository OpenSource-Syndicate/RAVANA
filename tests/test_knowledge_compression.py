import os
import json
from unittest.mock import patch
from modules.knowledge_compression.main import compress_knowledge
from modules.knowledge_compression.compressed_memory import COMPRESSED_FILE, load_summaries


def test_compress_knowledge():
    """Test compress_knowledge with a mock LLM call."""
    logs = [
        {"timestamp": "2024-07-01T12:00:00Z",
            "task": "Task 1", "result": "Success"},
        {"timestamp": "2024-07-02T13:00:00Z", "task": "Task 2", "result": "Failure"}
    ]
    with patch('modules.knowledge_compression.main.call_llm', return_value="Summary: Task 1 succeeded, Task 2 failed. Next: Improve reliability."):
        entry = compress_knowledge(logs)
        print("Compressed Summary (mocked):", json.dumps(entry, indent=2))
        assert 'summary' in entry
        assert 'Summary:' in entry['summary']


def test_compress_knowledge_real_api():
    """Test compress_knowledge with the real LLM API (uses actual API call)."""
    logs = [
        {"timestamp": "2024-07-01T12:00:00Z",
            "task": "Task 1", "result": "Success"},
        {"timestamp": "2024-07-02T13:00:00Z", "task": "Task 2", "result": "Failure"}
    ]
    entry = compress_knowledge(logs)
    print("Compressed Summary (real API):", json.dumps(entry, indent=2))
    assert 'summary' in entry
    assert isinstance(entry['summary'], str)
    assert len(entry['summary']) > 0


def test_load_summaries():
    """Test loading all summaries from the compressed memory store."""
    summaries = load_summaries()
    print("All Summaries:", json.dumps(summaries, indent=2))
    assert isinstance(summaries, list)


def cleanup_test_summaries():
    """Remove the compressed_memory.json file after tests to clean up test data."""
    if os.path.exists(COMPRESSED_FILE):
        os.remove(COMPRESSED_FILE)


if __name__ == "__main__":
    test_compress_knowledge()
    test_compress_knowledge_real_api()
    test_load_summaries()
    cleanup_test_summaries()
