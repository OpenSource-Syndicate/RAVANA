import os
import json

REFLECTIONS_FILE = os.path.join(os.path.dirname(__file__), 'reflections.json')

def save_reflection(entry):
    """Append a reflection entry to the JSON file."""
    data = load_reflections()
    data.append(entry)
    with open(REFLECTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_reflections():
    """Load all reflection entries from the JSON file."""
    if not os.path.exists(REFLECTIONS_FILE):
        return []
    with open(REFLECTIONS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f) 