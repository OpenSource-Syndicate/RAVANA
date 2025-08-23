# Ollama Timeout Update Summary

## Changes Made

1. **Core Configuration Update** (`core/config.py`):
   - Updated `SNAKE_OLLAMA_TIMEOUT` from 120 seconds to 3000 seconds (50 minutes)
   - Updated the default timeout values in both `SNAKE_CODING_MODEL` and `SNAKE_REASONING_MODEL` configurations from 300 seconds to 3000 seconds
   - Added comment indicating the extended timeout duration (50 minutes)

2. **Documentation Update** (`SNAKE_AGENT_SETUP.md`):
   - Updated all environment configuration examples to use 3000 seconds (50 minutes) instead of 120 seconds
   - Updated Windows batch configuration example
   - Updated Linux/macOS shell configuration example

3. **Error Message Enhancement** (`core/snake_llm.py`):
   - Enhanced the timeout error message to include both seconds and minutes for better clarity

## Impact

- The Snake Agent will now wait up to 50 minutes for Ollama responses instead of the previous 5 minutes
- This allows for more complex reasoning tasks and larger model responses
- The system will no longer timeout prematurely on long-running LLM operations

## Environment Variable

To customize the timeout value, set the `SNAKE_OLLAMA_TIMEOUT` environment variable:
```bash
export SNAKE_OLLAMA_TIMEOUT=3000  # 50 minutes in seconds
```

## Verification

All modified files have been checked for syntax errors and are valid.