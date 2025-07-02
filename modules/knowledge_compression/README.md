# Knowledge Compression Module (Module 9)

This module summarizes accumulated knowledge and logs on a periodic schedule to prevent memory bloat and provide concise progress updates.

## Usage

```
python main.py --logs path/to/logs.json
```

## API
- `compress_knowledge(logs: list) -> dict`: Generate and store a compressed summary.
- `load_summaries() -> list`: Load all stored summaries.

## LLM
Uses the shared `llm.py` and `config.json` from `modules/memory-3`. 