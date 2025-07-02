# Self-Reflection & Self-Modification Module (Module 8 & 13)

**New: Self-Modification (Advanced Module 13)**

This module can now identify buggy or improvable components from its own reflection logs and, using LLMs, propose and test code rewrites. All changes are tested in a sandbox before being applied to the main codebase. Lazy or incomplete LLM responses are detected and filtered out. No change is accepted unless all tests pass and the patch addresses the original issue.

This module generates self-reflective journal entries after major tasks using an LLM. Reflections are stored for future review and meta-knowledge.

## Usage

```
python main.py --task "Summarize news articles" --outcome "Successfully extracted key points, but missed some minor details."
```

## API
- `reflect_on_task(task_summary: str, outcome: str) -> dict`: Generate and store a reflection.
- `load_reflections() -> list`: Load all stored reflections.

## LLM
Uses the shared `llm.py` and `config.json` from `modules/memory-3`. 

## Enhanced Self-Reflection Process

This module now supports a structured self-reflection process after each major task. The process includes:

- **Reflection Prompt:** The LLM is prompted to answer:
  1. What worked?
  2. What failed?
  3. What surprised me?
  4. What do I still need to learn?
- **LangChain Integration:** The module uses LangChain to chain tasks: Planning → Execution → Reflection. Custom prompts guide the reflection phase.
- **Logging:** All reflections are logged in a database for review and future goal setting. Useful reflections are fed back as context for future tasks, becoming part of the system's meta-knowledge.

## Installation

To use the enhanced self-reflection features, install LangChain:

```bash
pip install langchain
```

--- 