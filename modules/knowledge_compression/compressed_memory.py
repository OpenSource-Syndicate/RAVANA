import os
import json

COMPRESSED_FILE = os.path.join(
    os.path.dirname(__file__), 'compressed_memory.json')


def save_summary(entry):
    """Append a summary entry to the JSON file."""
    data = load_summaries()
    data.append(entry)
    with open(COMPRESSED_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_summaries():
    """Load all summary entries from the JSON file."""
    if not os.path.exists(COMPRESSED_FILE):
        return []
    with open(COMPRESSED_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)
