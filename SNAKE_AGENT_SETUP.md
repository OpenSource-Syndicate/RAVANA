# Snake Agent Setup Guide

## Overview

The Snake Agent is an autonomous background agent that continuously monitors, analyzes, and experiments with the RAVANA codebase. It uses local LLM models via Ollama to perform code analysis and suggest improvements while maintaining safety through sandboxed experimentation.

## Prerequisites

### 1. Ollama Installation

First, install Ollama on your system:

**Windows:**
```bash
# Download and install from https://ollama.ai/download/windows
# Or use winget
winget install Ollama.Ollama
```

**macOS:**
```bash
# Download and install from https://ollama.ai/download/mac
# Or use Homebrew
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Start Ollama Service

Start the Ollama service (this needs to run in the background):

```bash
ollama serve
```

The service will be available at `http://localhost:11434` by default.

## Model Setup

### Required Models

The Snake Agent requires two specialized models:

1. **Coding Model**: For code analysis and improvement generation
2. **Reasoning Model**: For safety evaluation and decision making

### Download Models

```bash
# Download the coding model (recommended)
ollama pull deepseek-coder:6.7b

# Download the reasoning model (recommended)
ollama pull llama3.1:8b

# Verify models are installed
ollama list
```

### Alternative Model Options

You can choose different models based on your hardware capabilities:

**Coding Models:**
- `deepseek-coder:1.3b` - Lightweight (2GB RAM)
- `deepseek-coder:6.7b` - Balanced (8GB RAM) - **Recommended**
- `codellama:7b` - Alternative option (8GB RAM)
- `starcoder2:3b` - Efficient (4GB RAM)

**Reasoning Models:**
- `llama3.1:8b` - Balanced (8GB RAM) - **Recommended**
- `qwen2.5:7b` - Strong reasoning (7GB RAM)
- `mistral:7b` - Fast and efficient (7GB RAM)
- `gemma2:9b` - Google's model (9GB RAM)

## Environment Configuration

### Required Environment Variables

Set the following environment variables to configure the Snake Agent:

```bash
# Basic Configuration
export SNAKE_AGENT_ENABLED=True
export SNAKE_AGENT_INTERVAL=300  # Analysis interval in seconds (5 minutes)

# Ollama Configuration
export SNAKE_OLLAMA_BASE_URL=http://localhost:11434
export SNAKE_OLLAMA_TIMEOUT=3000
export SNAKE_OLLAMA_KEEP_ALIVE=5m

# Model Selection
export SNAKE_CODING_MODEL=deepseek-coder:6.7b
export SNAKE_REASONING_MODEL=llama3.1:8b

# Model Parameters
export SNAKE_CODING_TEMPERATURE=0.1      # Low for precise code
export SNAKE_REASONING_TEMPERATURE=0.3   # Higher for creativity
export SNAKE_CODING_MAX_TOKENS=4096
export SNAKE_REASONING_MAX_TOKENS=2048

# Safety Configuration
export SNAKE_SANDBOX_TIMEOUT=60          # Sandbox execution timeout
export SNAKE_MAX_FILE_SIZE=1048576       # Max file size (1MB)
export SNAKE_APPROVAL_REQUIRED=True      # Require RAVANA approval

# Communication
export SNAKE_COMM_CHANNEL=memory_service
export SNAKE_COMM_PRIORITY_THRESHOLD=0.8

# Graceful Shutdown
export SNAKE_SHUTDOWN_TIMEOUT=30
export SNAKE_STATE_PERSISTENCE=True
```

### Windows Configuration

For Windows, create a `snake_config.bat` file:

```batch
@echo off
set SNAKE_AGENT_ENABLED=True
set SNAKE_AGENT_INTERVAL=300
set SNAKE_OLLAMA_BASE_URL=http://localhost:11434
set SNAKE_OLLAMA_TIMEOUT=3000
set SNAKE_OLLAMA_KEEP_ALIVE=5m
set SNAKE_CODING_MODEL=deepseek-coder:6.7b
set SNAKE_REASONING_MODEL=llama3.1:8b
set SNAKE_CODING_TEMPERATURE=0.1
set SNAKE_REASONING_TEMPERATURE=0.3
set SNAKE_CODING_MAX_TOKENS=4096
set SNAKE_REASONING_MAX_TOKENS=2048
set SNAKE_SANDBOX_TIMEOUT=60
set SNAKE_MAX_FILE_SIZE=1048576
set SNAKE_APPROVAL_REQUIRED=True
set SNAKE_COMM_CHANNEL=memory_service
set SNAKE_COMM_PRIORITY_THRESHOLD=0.8
set SNAKE_SHUTDOWN_TIMEOUT=30
set SNAKE_STATE_PERSISTENCE=True

echo Snake Agent environment configured
```

Run before starting RAVANA:
```batch
snake_config.bat
python main.py
```

### Linux/macOS Configuration

Create a `snake_config.sh` file:

```bash
#!/bin/bash
export SNAKE_AGENT_ENABLED=True
export SNAKE_AGENT_INTERVAL=300
export SNAKE_OLLAMA_BASE_URL=http://localhost:11434
export SNAKE_OLLAMA_TIMEOUT=3000
export SNAKE_OLLAMA_KEEP_ALIVE=5m
export SNAKE_CODING_MODEL=deepseek-coder:6.7b
export SNAKE_REASONING_MODEL=llama3.1:8b
export SNAKE_CODING_TEMPERATURE=0.1
export SNAKE_REASONING_TEMPERATURE=0.3
export SNAKE_CODING_MAX_TOKENS=4096
export SNAKE_REASONING_MAX_TOKENS=2048
export SNAKE_SANDBOX_TIMEOUT=60
export SNAKE_MAX_FILE_SIZE=1048576
export SNAKE_APPROVAL_REQUIRED=True
export SNAKE_COMM_CHANNEL=memory_service
export SNAKE_COMM_PRIORITY_THRESHOLD=0.8
export SNAKE_SHUTDOWN_TIMEOUT=30
export SNAKE_STATE_PERSISTENCE=True

echo "Snake Agent environment configured"
```

Run before starting RAVANA:
```bash
source snake_config.sh
python main.py
```

## Verification

### Test Snake Agent Configuration

Run the validation test to ensure everything is configured correctly:

```bash
python tests/test_snake_agent.py
```

### Check Ollama Connection

Test Ollama connectivity:

```bash
curl http://localhost:11434/api/tags
```

Should return a JSON response with available models.

### Verify Models

Check that your models are available:

```bash
ollama list
```

Should show your downloaded models.

## Hardware Requirements

### Minimum Requirements
- **RAM**: 8GB (for recommended models)
- **Storage**: 10GB free space for models
- **CPU**: Modern multi-core processor
- **Network**: Internet connection for initial model download

### Recommended Requirements
- **RAM**: 16GB or more
- **Storage**: 20GB free space
- **CPU**: 8+ cores
- **GPU**: Optional (Ollama can use GPU acceleration)

## Usage

### Starting RAVANA with Snake Agent

1. **Start Ollama service:**
   ```bash
   ollama serve
   ```

2. **Configure environment:**
   ```bash
   # Linux/macOS
   source snake_config.sh
   
   # Windows
   snake_config.bat
   ```

3. **Start RAVANA:**
   ```bash
   python main.py
   ```

The Snake Agent will automatically start in the background when RAVANA begins its autonomous loop.

### Monitoring Snake Agent

Check Snake Agent status:

```bash
# View logs for Snake Agent activity
tail -f ravana_agi.log | grep -i "snake"
```

### Snake Agent Operations

The Snake Agent performs these operations automatically:

1. **File Monitoring**: Continuously monitors Python files for changes
2. **Code Analysis**: Analyzes code quality, performance, and architecture
3. **Safe Experimentation**: Tests improvements in isolated sandbox
4. **Communication**: Sends findings to RAVANA main system
5. **Learning**: Tracks success rates and adapts behavior

## Configuration Options

### Agent Behavior

- `SNAKE_AGENT_INTERVAL`: How often to perform analysis (seconds)
- `SNAKE_APPROVAL_REQUIRED`: Whether to require approval before implementing changes
- `SNAKE_COMM_PRIORITY_THRESHOLD`: Minimum priority for communications

### Safety Settings

- `SNAKE_SANDBOX_TIMEOUT`: Maximum time for sandbox operations
- `SNAKE_MAX_FILE_SIZE`: Maximum file size to analyze
- `SNAKE_BLACKLIST_PATHS`: Comma-separated paths to avoid

### Performance Tuning

- `SNAKE_CODING_TEMPERATURE`: Lower = more conservative code generation
- `SNAKE_REASONING_TEMPERATURE`: Higher = more creative reasoning
- `SNAKE_*_MAX_TOKENS`: Maximum tokens for model responses

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```
   Error: Cannot connect to Ollama at http://localhost:11434
   ```
   - **Solution**: Ensure Ollama service is running (`ollama serve`)
   - Check firewall settings
   - Verify URL in `SNAKE_OLLAMA_BASE_URL`

2. **Model Not Found**
   ```
   Error: Model deepseek-coder:6.7b not found
   ```
   - **Solution**: Download the model (`ollama pull deepseek-coder:6.7b`)
   - Check model name spelling
   - Verify with `ollama list`

3. **Snake Agent Not Starting**
   ```
   Snake Agent initialization failed
   ```
   - **Solution**: Check environment variables
   - Verify model availability
   - Check RAVANA logs for detailed errors

4. **High Memory Usage**
   - **Solution**: Use smaller models (e.g., `deepseek-coder:1.3b`)
   - Reduce `SNAKE_*_MAX_TOKENS` values
   - Increase `SNAKE_AGENT_INTERVAL`

5. **Sandbox Timeouts**
   ```
   Sandbox execution timed out
   ```
   - **Solution**: Increase `SNAKE_SANDBOX_TIMEOUT`
   - Check system resource availability
   - Review code complexity being tested

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python main.py
```

### Health Check

Create a health check script (`check_snake.py`):

```python
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.snake_llm import SnakeConfigValidator

async def main():
    print("🐍 Snake Agent Health Check")
    print("=" * 40)
    
    report = SnakeConfigValidator.get_startup_report()
    
    print(f"Ollama Connected: {'✅' if report['ollama_connected'] else '❌'}")
    print(f"Coding Model Available: {'✅' if report['coding_model_available'] else '❌'}")
    print(f"Reasoning Model Available: {'✅' if report['reasoning_model_available'] else '❌'}")
    print(f"Configuration Valid: {'✅' if report['config_valid'] else '❌'}")
    
    if report['available_models']:
        print(f"\nAvailable Models:")
        for model in report['available_models']:
            print(f"  - {model}")
    
    if report['config_valid']:
        print("\n🎉 Snake Agent is ready!")
        return True
    else:
        print("\n⚠️  Snake Agent needs configuration")
        return False

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
```

Run the health check:
```bash
python check_snake.py
```

## Advanced Configuration

### Custom Model URLs

If using a remote Ollama instance:

```bash
export SNAKE_OLLAMA_BASE_URL=http://remote-host:11434
```

### Multiple Coding Models

For different analysis types, you can specify alternative models:

```bash
export SNAKE_CODING_MODEL=deepseek-coder:6.7b
export SNAKE_ALTERNATIVE_CODING_MODEL=codellama:7b
```

### Performance Monitoring

Enable performance tracking:

```bash
export SNAKE_PERFORMANCE_MONITORING=True
export SNAKE_METRICS_INTERVAL=3600  # Log metrics every hour
```

## Security Considerations

1. **Sandbox Isolation**: All code experiments run in isolated environments
2. **Safety Analysis**: Code is analyzed for dangerous patterns before execution
3. **Approval Workflow**: Set `SNAKE_APPROVAL_REQUIRED=True` for production
4. **Resource Limits**: Configure timeouts and file size limits
5. **Network Isolation**: Sandbox environments have limited network access

## Integration with RAVANA

The Snake Agent integrates seamlessly with RAVANA's existing systems:

- **Memory Service**: Stores analysis results and communications
- **Shared State**: Accesses RAVANA's current context
- **Action Manager**: Can propose new actions
- **Graceful Shutdown**: Properly shuts down with RAVANA
- **State Persistence**: Maintains state across restarts

## Getting Help

1. **Check Logs**: Look in `ravana_agi.log` for detailed information
2. **Run Tests**: Execute `python tests/test_snake_agent.py`
3. **Health Check**: Run the health check script
4. **Documentation**: Review this guide and code comments
5. **GitHub Issues**: Report problems on the RAVANA repository

---

The Snake Agent represents a significant step toward autonomous code improvement in AGI systems. By following this guide, you'll have a powerful autonomous agent continuously working to improve your RAVANA system.