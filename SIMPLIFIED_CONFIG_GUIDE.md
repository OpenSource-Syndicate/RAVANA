# Simplified Configuration Guide for RAVANA AGI

## Overview

This guide explains the simplified configuration system for the RAVANA AGI project. The goal is to reduce complexity while maintaining essential functionality.

## Key Changes

### 1. Consolidated Configuration
- **Before**: Multiple configuration files (`core/config.json`, `analysis/config.json`, `core/config.py`) with complex nested structures
- **After**: Single simplified configuration file (`simple_config.json`) with clear, logical groupings

### 2. Reduced Complexity
- Simplified model configuration from separate coding/reasoning models to a single primary model
- Consolidated API keys to essential providers only
- Streamlined settings into logical categories

### 3. Environment Variable Support
- All configuration values can be overridden via environment variables
- Maintains compatibility with existing deployment practices

## Configuration Structure

The simplified configuration is organized into these main sections:

### System
- `name`: Name of the system (default: "RAVANA")
- `database_url`: Database connection string (default: "sqlite:///ravana_agi.db")
- `log_level`: Logging level (default: "INFO")

### Operational
- `curiosity_chance`: Probability of curiosity-driven behavior (0.0 to 1.0)
- `reflection_chance`: Probability of reflection (0.0 to 1.0)
- `loop_sleep_duration`: Sleep duration between loops in seconds
- `max_iterations`: Maximum iterations for processes

### Models
- `primary`: Configuration for the main AI model
  - `provider`: AI provider (e.g., "ollama", "openai")
  - `model_name`: Name of the model
  - `base_url`: API endpoint
  - `temperature`: Creativity parameter
  - `max_tokens`: Maximum tokens to generate
  - `timeout`: Request timeout in seconds

### API Keys
- `electronhub`: API key for ElectronHub provider
- `zuki`: API key for Zuki provider
- `gemini`: API key for Google Gemini provider

### Memory
- `embedding_model`: Model used for embeddings
- `embedding_use_cuda`: Whether to use CUDA for embeddings

### Agents
- `snake_agent`: Configuration for the Snake Agent
  - `enabled`: Whether the agent is enabled
  - `interval`: Interval between agent runs in seconds

### Services
- `blog`: Blog publishing service
  - `enabled`: Whether blog publishing is enabled
  - `api_url`: Blog API endpoint
- `conversational_ai`: Conversational AI service
  - `enabled`: Whether conversational AI is enabled

### Scheduling
- `data_collection_interval`: How often to collect data (seconds)
- `event_detection_interval`: How often to detect events (seconds)
- `knowledge_compression_interval`: How often to compress knowledge (seconds)

### Persona
- `name`: Name of the AI persona
- `creativity`: Creativity level (0.0 to 1.0)

### Shutdown
- `timeout`: Shutdown timeout in seconds
- `graceful_shutdown_enabled`: Whether graceful shutdown is enabled

## Usage

### Loading Configuration

```python
from core.simple_config import get_config

config = get_config()

# Access configuration values
print(config.system_name)
print(config.primary_model['model_name'])
```

### Environment Variables

All configuration values can be overridden using environment variables:

```bash
export SYSTEM_NAME="MyRAVANA"
export PRIMARY_MODEL_NAME="llama3.1:70b"
export CURIOSITY_CHANCE=0.5
```

### Configuration File

The configuration can also be loaded from a JSON file:

```python
from core.simple_config import SimpleConfig

config = SimpleConfig()
config.load_from_file("simple_config.json")
```

## Migration Guide

To migrate from the old configuration system:

1. Update your imports to use `core.simple_config` instead of `core.config`
2. Replace complex model configurations with the simplified primary model
3. Consolidate API keys to the essential providers
4. Use the new logical groupings for configuration values

## Benefits

1. **Easier Maintenance**: Single configuration file with clear structure
2. **Better Readability**: Logical groupings make it easy to find settings
3. **Simplified Code**: Reduced complexity in configuration handling
4. **Environment Support**: Full compatibility with environment variables
5. **Backward Compatibility**: Existing environment variables still work

## Example Configuration

```json
{
  "system": {
    "name": "RAVANA",
    "database_url": "sqlite:///ravana_agi.db",
    "log_level": "INFO"
  },
  "operational": {
    "curiosity_chance": 0.4,
    "reflection_chance": 0.15,
    "loop_sleep_duration": 7,
    "max_iterations": 15
  }
  // ... other sections
}
```

This simplified configuration system maintains all essential functionality while reducing complexity and improving maintainability.