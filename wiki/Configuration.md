# Configuration



## Update Summary
**Changes Made**   
- Added comprehensive documentation for enhanced graceful shutdown configuration options
- Updated LLM provider configuration section with new Gemini API key details
- Added new section on shutdown coordinator functionality and phases
- Updated environment variables section with new shutdown-related settings
- Added new diagram for shutdown process flow
- Updated referenced files list with new and modified files

## Table of Contents
1. [Configuration Overview](#configuration-overview)
2. [Core Configuration Files](#core-configuration-files)
3. [Config Class Implementation](#config-class-implementation)
4. [LLM Provider Configuration](#llm-provider-configuration)
5. [Module-Specific Settings](#module-specific-settings)
6. [Configuration Loading and Validation](#configuration-loading-and-validation)
7. [Environment Variables and Overrides](#environment-variables-and-overrides)
8. [Sensitive Data Handling](#sensitive-data-handling)
9. [Advanced Configuration Management](#advanced-configuration-management)
10. [Configuration Examples](#configuration-examples)
11. [Common Configuration Issues](#common-configuration-issues)
12. [Best Practices](#best-practices)

## Configuration Overview

The RAVANA system implements a comprehensive configuration management system that combines JSON configuration files with environment variable overrides. The system supports multiple configuration sources, including global settings, module-specific configurations, and runtime environment variables. This layered approach enables flexible deployment across different environments while maintaining security and consistency.

The configuration system serves as the central nervous system for the AGI framework, controlling everything from LLM provider selection to emotional intelligence parameters and autonomous behavior patterns. Two primary mechanisms work together: the static `config.json` files that define default settings and provider configurations, and the dynamic `Config` class that loads, validates, and provides access to settings throughout the application.

**Section sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L2-L182)
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json#L1-L193)

## Core Configuration Files

The system utilizes multiple JSON configuration files to manage different aspects of the AGI framework. The primary configuration file `core/config.json` contains LLM provider credentials, API endpoints, and model availability information. This file serves as the authoritative source for external service integrations.

``mermaid
graph TD
A[Configuration Files] --> B[core/config.json]
A --> C[modules/conversational_ai/config.json]
A --> D[modules/emotional_intellegence/config.json]
A --> E[modules/emotional_intellegence/persona.json]
B --> F[LLM Providers]
B --> G[API Keys]
B --> H[Model Lists]
C --> I[Platform Integration]
C --> J[Communication Settings]
D --> K[Emotional Triggers]
D --> L[Behavior Influences]
E --> M[Persona Definitions]
E --> N[Mood Multipliers]
```

**Diagram sources**
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json#L1-L193)
- [conversational_ai/config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\config.json#L1-L27)
- [emotional_intellegence/config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\config.json#L1-L202)
- [persona.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\persona.json#L1-L86)

**Section sources**
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json#L1-L193)
- [conversational_ai/config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\config.json#L1-L27)
- [emotional_intellegence/config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\config.json#L1-L202)
- [persona.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\persona.json#L1-L86)

## Config Class Implementation

The `Config` class in `core/config.py` serves as the central configuration manager, providing a unified interface for accessing settings from both JSON files and environment variables. This class implements a hierarchical override system where environment variables take precedence over JSON file values, enabling flexible deployment configurations.

```python
class Config:
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///ravana_agi.db")
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    CURIOSITY_CHANCE = float(os.environ.get("CURIOSITY_CHANCE", 0.3))
    # ... additional settings
```

The class uses Python's `os.environ.get()` method to retrieve environment variables with fallback to default values. Type conversion is explicitly handled for different data types: strings remain as-is, numeric values are converted using `int()` or `float()`, and boolean values are parsed from string representations ("true", "1", "yes") to Python boolean values.

``mermaid
classDiagram
class Config {
+DATABASE_URL : str
+LOG_LEVEL : str
+CURIOSITY_CHANCE : float
+REFLECTION_CHANCE : float
+LOOP_SLEEP_DURATION : int
+EMOTIONAL_PERSONA : str
+SNAKE_AGENT_ENABLED : bool
+BLOG_ENABLED : bool
+get_setting(key : str) : Any
+validate_configuration() : List[str]
}
class ConfigLoader {
+load_from_json(path : str) : Dict
+load_from_env(prefix : str) : Dict
+merge_configurations() : Dict
+validate_settings(config : Dict) : bool
}
Config "1" -- "1" ConfigLoader : uses
```

**Diagram sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L2-L182)

**Section sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L2-L182)

## LLM Provider Configuration

The system supports multiple LLM providers through the `core/config.json` file, which defines API endpoints, authentication credentials, and available models for each provider. This configuration enables the system to route requests to different providers based on model availability, cost considerations, and performance requirements.

### Provider Configuration Structure

Each provider in the configuration file follows a consistent structure with the following key components:

- **api_key**: Authentication token for the provider API
- **base_url**: Endpoint URL for API requests
- **models**: List of available models that can be used

```json
{
  "zuki": {
    "api_key": "zu-ab9fba2aeef85c7ecb217b00ce7ca1fe",
    "base_url": "https://api.zukijourney.com/v1",
    "models": [
      "gpt-4o:online",
      "gpt-4o",
      "deepseek-chat",
      "deepseek-reasoner"
    ]
  }
}
```

### Multi-Key Fallback System

The Gemini provider implements a sophisticated multi-key fallback system that allows the system to rotate through multiple API keys when rate limits are reached:

```json
"gemini": {
  "api_keys": [
    {
      "id": "gemini_key_1",
      "key": "AIzaSyBW-aVU-x7JCjBJVVKjPGUacups0-GBHvQ",
      "priority": 1,
      "failure_count": 0,
      "last_success": null
    },
    {
      "id": "gemini_key_2",
      "key": "AIzaSyBW-aVU-x7JCjBJVVKjPGUacups0-GBHvQ",
      "priority": 2,
      "failure_count": 0,
      "last_success": null
    },
    {
      "id": "gemini_key_3",
      "key": "AIzaSyC8hTaXE6jkWDFZyZDybFEoHfPR-4VSV-k",
      "priority": 3,
      "failure_count": 0,
      "last_success": null
    },
    {
      "id": "gemini_key_10",
      "key": "AIzaSyBCQic67-TJmSXUVTCiW3SFU5fLZx7enII",
      "priority": 10,
      "failure_count": 0,
      "last_success": null
    }
  ],
  "rate_limit": {
    "requests_per_minute": 60,
    "cooldown_period": 300,
    "max_retries": 3,
    "backoff_factor": 2.0
  },
  "fallback": {
    "enabled": true,
    "timeout": 30,
    "max_key_failures": 5
  }
}
```

This configuration enables resilient operation by automatically switching to backup keys when the primary key encounters rate limiting or failures. The recent update has expanded the Gemini configuration to include 10 API keys with priority-based rotation.

``mermaid
sequenceDiagram
participant Client as "Application"
participant Config as "Config System"
participant Provider as "LLM Provider"
Client->>Config : Request model usage
Config->>Config : Select provider based on model availability
Config->>Provider : Send API request with primary key
alt Rate limit exceeded
Provider-->>Config : 429 Too Many Requests
Config->>Config : Increment failure count
Config->>Config : Select next priority key
Config->>Provider : Retry with backup key
else Success
Provider-->>Config : 200 OK with response
end
Config-->>Client : Return response or error
```

**Diagram sources**
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json#L1-L193)
- [llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py#L46-L114)

**Section sources**
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json#L1-L193)
- [llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py#L46-L114)

## Module-Specific Settings

Different modules within the RAVANA system have specialized configuration requirements that are managed through dedicated configuration files and settings.

### Emotional Intelligence Configuration

The emotional intelligence module uses two configuration files to define emotional states and persona behaviors:

```json
{
  "emotional_intelligence_config": {
    "primary_emotions": {
      "joy_based": ["Confident", "Excited", "Inspired", "Satisfied"],
      "interest_based": ["Curious", "Reflective", "Intrigued", "Engaged"]
    },
    "secondary_emotions": [
      "Hopeful", "Grateful", "Proud", "Guilty"
    ],
    "emotion_intensity_levels": {
      "low": [0.0, 0.33],
      "medium": [0.34, 0.66],
      "high": [0.67, 1.0]
    },
    "triggers": {
      "new_discovery": "The agent has discovered new information or learned something novel.",
      "task_completed": "A task or goal has been successfully completed."
    },
    "mood_updates": {
      "new_discovery": {
        "Curious": 0.2,
        "Excited": 0.15,
        "Inspired": 0.1
      }
    },
    "behavior_influences": {
      "Confident": {
        "risk_tolerance": "high",
        "exploration_tendency": "high",
        "planning_depth": "medium"
      }
    }
  }
}
```

### Conversational AI Configuration

The conversational AI module manages platform integrations and communication settings:

```json
{
  "discord_token": "MTQwODY4Nzk2NjgwODk2OTIxOA.GAH1_i.3slf5vQegHTxhTs5k_uFhgDQ2JFpprg2S1njSI",
  "telegram_token": "8210359732:AAFSrU94MscB38WLmius8TiRAgnvkwaKJiI",
  "platforms": {
    "discord": {
      "enabled": true,
      "command_prefix": "!"
    },
    "telegram": {
      "enabled": true,
      "command_prefix": "/"
    }
  },
  "emotional_intelligence": {
    "default_persona": "Balanced",
    "mood_decay_rate": 0.05,
    "context_window_hours": 24
  }
}
```

### Persona Configuration

The `persona.json` file defines different personality types and their emotional response patterns:

```json
{
  "personas": {
    "Optimistic": {
      "mood_multipliers": {
        "Confident": 1.5,
        "Curious": 1.2,
        "Frustrated": 0.5
      },
      "description": "Sees the glass as half full. Bounces back from setbacks quickly.",
      "adaptation_rate": 0.1
    },
    "Analytical": {
      "mood_multipliers": {
        "Confident": 1.1,
        "Curious": 1.8,
        "Frustrated": 0.8
      },
      "description": "Driven by data and logic. Less prone to strong emotional swings.",
      "adaptation_rate": 0.15
    }
  },
  "default_persona": "Balanced"
}
```

**Section sources**
- [emotional_intellegence/config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\config.json#L1-L202)
- [conversational_ai/config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\config.json#L1-L27)
- [persona.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\persona.json#L1-L86)

## Configuration Loading and Validation

The configuration system follows a well-defined process for loading and validating settings from multiple sources. The `Config` class automatically loads settings in the following priority order:

1. Environment variables (highest priority)
2. JSON configuration files
3. Hard-coded default values

### Configuration Validation

The system includes built-in validation for critical configuration parameters. For example, the Snake Agent configuration validates model settings and resource limits:

```python
SNAKE_CODING_MODEL = {
    "provider": "ollama",
    "model_name": os.environ.get("SNAKE_CODING_MODEL", "gpt-oss:20b"),
    "temperature": float(os.environ.get("SNAKE_CODING_TEMPERATURE", "0.1")),
    "max_tokens": None if os.environ.get("SNAKE_CODING_MAX_TOKENS", "unlimited").lower() in ["unlimited", "none", "-1"] else int(os.environ.get("SNAKE_CODING_MAX_TOKENS", "4096")),
    "timeout": int(os.environ.get("SNAKE_OLLAMA_TIMEOUT", 3000))
}
```

### Configuration Access Pattern

Components throughout the system access configuration values using direct attribute access:

```python
from core.config import Config

# Access configuration values
if Config.BLOG_ENABLED:
    publish_content()

model_name = Config.SNAKE_CODING_MODEL["model_name"]
timeout = Config.SNAKE_OLLAMA_TIMEOUT
```

This pattern is used consistently across 25+ files in the codebase, demonstrating the central role of the Config class in the system architecture.

``mermaid
flowchart TD
A[Start] --> B[Load Environment Variables]
B --> C{Variable Exists?}
C --> |Yes| D[Use Environment Value]
C --> |No| E[Use JSON File Value]
E --> F{Value in JSON?}
F --> |Yes| G[Use JSON Value]
F --> |No| H[Use Default Value]
D --> I[Convert to Proper Type]
G --> I
H --> I
I --> J[Store in Config Class]
J --> K[Application Uses Config]
```

**Diagram sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L2-L182)

**Section sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L2-L182)

## Environment Variables and Overrides

The system extensively uses environment variables to enable runtime configuration without modifying configuration files. This approach supports secure deployment practices by keeping sensitive information out of version control.

### Override Mechanism

The override mechanism follows a simple pattern for all configuration values:

```python
# Pattern: os.environ.get("ENV_VAR_NAME", "default_value")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
CURIOSITY_CHANCE = float(os.environ.get("CURIOSITY_CHANCE", 0.3))
SNAKE_AGENT_ENABLED = bool(os.environ.get("SNAKE_AGENT_ENABLED", "True").lower() in ["true", "1", "yes"])
```

### Common Environment Variables

The system recognizes numerous environment variables for different configuration aspects:

**Autonomous Behavior**
- `CURIOSITY_CHANCE`: Probability of initiating curiosity-driven actions (default: 0.3)
- `REFLECTION_CHANCE`: Probability of self-reflection (default: 0.1)
- `LOOP_SLEEP_DURATION`: Delay between autonomous loops (default: 10 seconds)

**Snake Agent Configuration**
- `SNAKE_AGENT_ENABLED`: Whether the Snake Agent is active (default: True)
- `SNAKE_CODING_MODEL`: Ollama model for coding tasks (default: gpt-oss:20b)
- `SNAKE_REASONING_MODEL`: Ollama model for reasoning tasks (default: deepseek-r1:7b)

**Blog Integration**
- `RAVANA_BLOG_ENABLED`: Whether blog publishing is enabled (default: True)
- `RAVANA_BLOG_API_URL`: Endpoint for blog publishing (default: https://ravana-blog.netlify.app/api/publish)
- `RAVANA_BLOG_AUTH_TOKEN`: Authentication token for blog API

**Shutdown Configuration**
- `GRACEFUL_SHUTDOWN_ENABLED`: Whether to perform graceful shutdown (default: True)
- `SHUTDOWN_TIMEOUT`: Maximum time for shutdown process (default: 30 seconds)
- `STATE_PERSISTENCE_ENABLED`: Whether to save state on shutdown (default: True)
- `SHUTDOWN_HEALTH_CHECK_ENABLED`: Whether to perform health checks before shutdown (default: True)
- `SHUTDOWN_BACKUP_ENABLED`: Whether to create backups during shutdown (default: True)
- `SHUTDOWN_BACKUP_COUNT`: Number of backup files to retain (default: 5)
- `SHUTDOWN_STATE_VALIDATION_ENABLED`: Whether to validate state data before persistence (default: True)
- `SHUTDOWN_VALIDATION_ENABLED`: Whether to validate state file integrity (default: True)
- `SHUTDOWN_COMPRESSION_ENABLED`: Whether to enable compression during shutdown (default: True)
- `COMPONENT_PREPARE_TIMEOUT`: Timeout for component preparation phase (default: 10.0 seconds)
- `COMPONENT_SHUTDOWN_TIMEOUT`: Timeout for component shutdown phase (default: 15.0 seconds)
- `MEMORY_SERVICE_SHUTDOWN_TIMEOUT`: Timeout for memory service shutdown (default: 15 seconds)
- `POSTGRES_CONNECTION_TIMEOUT`: Timeout for PostgreSQL connection (default: 10 seconds)
- `CHROMADB_PERSIST_ON_SHUTDOWN`: Whether to persist ChromaDB data on shutdown (default: True)
- `TEMP_FILE_CLEANUP_ENABLED`: Whether to clean up temporary files on shutdown (default: True)
- `ACTION_CACHE_PERSIST`: Whether to persist action cache on shutdown (default: True)
- `RESOURCE_CLEANUP_TIMEOUT`: Timeout for resource cleanup (default: 10 seconds)
- `DATABASE_CLEANUP_TIMEOUT`: Timeout for database cleanup (default: 15 seconds)
- `FORCE_SHUTDOWN_AFTER`: Maximum time before forcing shutdown (default: 60 seconds)

**Section sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L2-L182)

## Sensitive Data Handling

The system implements multiple strategies for handling sensitive configuration data securely.

### API Key Management

API keys are stored in the `config.json` file alongside other provider configuration:

```json
{
  "zuki": {
    "api_key": "zu-ab9fba2aeef85c7ecb217b00ce7ca1fe",
    "base_url": "https://api.zukijourney.com/v1"
  }
}
```

For production deployments, these values should be overridden using environment variables to prevent accidental exposure in version control.

### Token Security

Platform integration tokens are stored in module-specific configuration files:

```json
{
  "discord_token": "MTQwODY4Nzk2NjgwODk2OTIxOA.GAH1_i.3slf5vQegHTxhTs5k_uFhgDQ2JFpprg2S1njSI",
  "telegram_token": "8210359732:AAFSrU94MscB38WLmius8TiRAgnvkwaKJiI"
}
```

These tokens should be managed using environment variables in production environments.

### Secure Defaults

The system implements secure defaults for sensitive settings:

```python
# Blog authentication token (should be overridden in production)
BLOG_AUTH_TOKEN = os.environ.get("RAVANA_BLOG_AUTH_TOKEN", "ravana_secret_token_2024")

# Snake Agent safety configuration
SNAKE_APPROVAL_REQUIRED = bool(os.environ.get("SNAKE_APPROVAL_REQUIRED", "True").lower() in ["true", "1", "yes"])
```

The default authentication token is intentionally weak to encourage users to set their own secure values.

**Section sources**
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json#L1-L193)
- [conversational_ai/config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\config.json#L1-L27)
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L2-L182)

## Advanced Configuration Management

The system includes advanced configuration management capabilities through the VLTM (Very Long-Term Memory) configuration system and enhanced shutdown coordinator.

### Enhanced Shutdown Configuration

The system now features a comprehensive shutdown configuration system with multiple phases and validation steps. The `shutdown_coordinator.py` file implements a multi-phase shutdown process:

```python
class ShutdownCoordinator:
    async def initiate_shutdown(self, reason: str = "manual"):
        """Initiate the graceful shutdown process."""
        # Execute shutdown phases in sequence
        await self._execute_shutdown_phases()
```

### Shutdown Process Flow

The enhanced shutdown process follows a structured sequence of phases:

``mermaid
graph TD
A[Initiate Shutdown] --> B[Pre-Shutdown Validation]
B --> C[Signal Received]
C --> D[Component Notification]
D --> E[Stop Background Tasks]
E --> F[Resource Cleanup]
F --> G[Service Shutdown]
G --> H[State Persistence]
H --> I[Final Validation]
I --> J[Shutdown Complete]
```

### Shutdown Phase Configuration

The system supports configurable timeouts and behaviors for each shutdown phase:

```python
# Shutdown phase configuration
SHUTDOWN_HEALTH_CHECK_ENABLED = bool(os.environ.get("SHUTDOWN_HEALTH_CHECK_ENABLED", "True").lower() in ["true", "1", "yes"])
SHUTDOWN_BACKUP_ENABLED = bool(os.environ.get("SHUTDOWN_BACKUP_ENABLED", "True").lower() in ["true", "1", "yes"])
SHUTDOWN_BACKUP_COUNT = int(os.environ.get("SHUTDOWN_BACKUP_COUNT", 5))
SHUTDOWN_STATE_VALIDATION_ENABLED = bool(os.environ.get("SHUTDOWN_STATE_VALIDATION_ENABLED", "True").lower() in ["true", "1", "yes"])
SHUTDOWN_VALIDATION_ENABLED = bool(os.environ.get("SHUTDOWN_VALIDATION_ENABLED", "True").lower() in ["true", "1", "yes"])
SHUTDOWN_COMPRESSION_ENABLED = bool(os.environ.get("SHUTDOWN_COMPRESSION_ENABLED", "True").lower() in ["true", "1", "yes"])
COMPONENT_PREPARE_TIMEOUT = float(os.environ.get("COMPONENT_PREPARE_TIMEOUT", 10.0))
COMPONENT_SHUTDOWN_TIMEOUT = float(os.environ.get("COMPONENT_SHUTDOWN_TIMEOUT", 15.0))
```

### Shutdown Lifecycle

The system manages the full lifecycle of shutdown operations:

``mermaid
stateDiagram-v2
[*] --> PreShutdownValidation
PreShutdownValidation --> SignalReceived : health checks passed
SignalReceived --> ComponentNotification : signal sent
ComponentNotification --> TasksStopping : components notified
TasksStopping --> ResourceCleanup : tasks stopped
ResourceCleanup --> ServiceShutdown : resources cleaned
ServiceShutdown --> StatePersistence : services shutdown
StatePersistence --> FinalValidation : state persisted
FinalValidation --> ShutdownComplete : validation passed
ShutdownComplete --> [*]
```

### Shutdown History and Logging

The system maintains detailed logs of shutdown operations:

```python
def _log_shutdown_summary(self):
    """Log a summary of the shutdown process."""
    log_data = {
        "shutdown_summary": self.shutdown_state,
        "duration_seconds": elapsed,
        "timestamp": datetime.utcnow().isoformat()
    }
```

This feature enables auditing and troubleshooting of shutdown operations.

**Diagram sources**
- [shutdown_coordinator.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\shutdown_coordinator.py#L100-L760)

**Section sources**
- [shutdown_coordinator.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\shutdown_coordinator.py#L100-L760)
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L100-L182)

## Configuration Examples

### Minimal Configuration

A minimal configuration file for local development:

```json
{
  "zuki": {
    "api_key": "your_zuki_api_key",
    "base_url": "https://api.zukijourney.com/v1",
    "models": ["gpt-4o"]
  },
  "local_situation_generator": {
    "enabled": true,
    "schedule": "daily"
  },
  "main_llm_decision_maker": {
    "enabled": true,
    "preferred_model": "gpt-4o"
  }
}
```

### Production Environment Variables

Recommended environment variables for production deployment:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/ravana_production

# Logging
LOG_LEVEL=WARNING

# Autonomous Behavior
CURIOSITY_CHANCE=0.1
REFLECTION_CHANCE=0.05
LOOP_SLEEP_DURATION=30

# Snake Agent
SNAKE_AGENT_ENABLED=true
SNAKE_CODING_MODEL=deepseek-coder:6.7b
SNAKE_REASONING_MODEL=llama3.1:8b

# Blog Integration
RAVANA_BLOG_ENABLED=true
RAVANA_BLOG_API_URL=https://your-blog-api.com/publish
RAVANA_BLOG_AUTH_TOKEN=your_secure_token_here

# Shutdown Configuration
GRACEFUL_SHUTDOWN_ENABLED=true
SHUTDOWN_TIMEOUT=60
STATE_PERSISTENCE_ENABLED=true
SHUTDOWN_HEALTH_CHECK_ENABLED=true
SHUTDOWN_BACKUP_ENABLED=true
SHUTDOWN_BACKUP_COUNT=10
FORCE_SHUTDOWN_AFTER=120
```

### Development Override

Development-specific overrides to enable debugging:

```bash
# Enable verbose logging
LOG_LEVEL=DEBUG
LOG_FORMAT=JSON

# Faster loops for testing
CURIOSITY_CHANCE=0.8
REFLECTION_CHANCE=0.5
LOOP_SLEEP_DURATION=2

# Disable resource-intensive features
SNAKE_AGENT_ENABLED=false
BLOG_ENABLED=false

# Shorter intervals for testing
DATA_COLLECTION_INTERVAL=300
EVENT_DETECTION_INTERVAL=60

# Enhanced shutdown for testing
SHUTDOWN_HEALTH_CHECK_ENABLED=false
SHUTDOWN_BACKUP_ENABLED=false
SHUTDOWN_STATE_VALIDATION_ENABLED=false
```

**Section sources**
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json#L1-L193)
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L2-L182)

## Common Configuration Issues

### Missing Settings

When required settings are missing, the system uses default values which may not be appropriate for all environments:

```python
# Problem: Using default SQLite database in production
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///ravana_agi.db")
```

**Solution**: Always set `DATABASE_URL` in production to use a proper database:

```bash
DATABASE_URL=postgresql://user:password@host:port/database_name
```

### Type Mismatches

Environment variables are strings, so type conversion errors can occur:

```python
# Problem: String "true" doesn't convert to boolean correctly
SNAKE_AGENT_ENABLED = os.environ.get("SNAKE_AGENT_ENABLED", "True")
```

**Solution**: Use proper boolean parsing:

```python
SNAKE_AGENT_ENABLED = bool(os.environ.get("SNAKE_AGENT_ENABLED", "True").lower() in ["true", "1", "yes"])
```

### API Key Rate Limiting

Multiple API keys help mitigate rate limiting, but improper configuration can still cause issues:

```json
"gemini": {
  "api_keys": [
    {
      "id": "gemini_key_1",
      "key": "your_actual_key_here",  // Don't use placeholder keys
      "priority": 1
    }
  ],
  "rate_limit": {
    "requests_per_minute": 60  // Ensure this matches provider limits
  }
}
```

### Configuration Reload Behavior

The current configuration system loads settings at startup and does not automatically reload when files change. For dynamic configuration updates, use environment variables or the VLTM configuration system.

### Shutdown Configuration Issues

The enhanced shutdown system may encounter issues if not properly configured:

```python
# Problem: Insufficient timeout for component preparation
COMPONENT_PREPARE_TIMEOUT = 1.0  # Too short for complex components

# Problem: Disabled state validation in production
SHUTDOWN_STATE_VALIDATION_ENABLED = false  # Risk of corrupted state files
```

**Solution**: Use appropriate timeouts and enable validation in production:

```bash
COMPONENT_PREPARE_TIMEOUT=30.0
SHUTDOWN_STATE_VALIDATION_ENABLED=true
SHUTDOWN_VALIDATION_ENABLED=true
```

**Section sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L2-L182)
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json#L1-L193)
- [shutdown_coordinator.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\shutdown_coordinator.py#L100-L760)

## Best Practices

### Environment-Specific Configurations

Use different configuration strategies for different environments:

**Development**
- Use environment variables in `.env` files
- Enable verbose logging
- Use faster loop intervals for testing
- Disable resource-intensive features
- Disable shutdown health checks and backups

**Production**
- Use environment variables from secure storage
- Set appropriate resource limits
- Use production-grade databases
- Implement proper error handling and monitoring
- Enable all shutdown validation and backup features
- Use multiple API keys with proper priorities

### Secrets Management

Follow these guidelines for managing sensitive data:

1. Never commit API keys or tokens to version control
2. Use environment variables for sensitive settings
3. Use different keys for development and production
4. Rotate keys periodically
5. Restrict key permissions to minimum required
6. Use secure storage solutions for production secrets

### Configuration Validation

Always validate configuration settings before use:

```python
def validate_configuration():
    """Validate critical configuration settings"""
    errors = []
    
    # Validate database URL
    if "sqlite" in Config.DATABASE_URL and not Config.DATABASE_URL.startswith("sqlite"):
        errors.append("Invalid SQLite database URL format")
    
    # Validate model names
    valid_models = ["gpt-4o", "deepseek-coder", "llama3.1"]
    if Config.SNAKE_CODING_MODEL["model_name"] not in valid_models:
        errors.append(f"Invalid coding model: {Config.SNAKE_CODING_MODEL['model_name']}")
    
    # Validate shutdown timeouts
    if Config.FORCE_SHUTDOWN_AFTER < Config.SHUTDOWN_TIMEOUT:
        errors.append("FORCE_SHUTDOWN_AFTER must be greater than SHUTDOWN_TIMEOUT")
    
    return errors
```

### Documentation and Comments

Maintain clear documentation for all configuration options:

```python
# Autonomous Loop Settings
# Controls the frequency and probability of autonomous behaviors
CURIOSITY_CHANCE = float(os.environ.get("CURIOSITY_CHANCE", 0.3))  # Probability of curiosity-driven actions
REFLECTION_CHANCE = float(os.environ.get("REFLECTION_CHANCE", 0.1))  # Probability of self-reflection
LOOP_SLEEP_DURATION = int(os.environ.get("LOOP_SLEEP_DURATION", 10))  # Seconds between loops

# Enhanced Shutdown Configuration
# Controls the graceful shutdown process with multiple phases and validation
SHUTDOWN_HEALTH_CHECK_ENABLED = bool(os.environ.get("SHUTDOWN_HEALTH_CHECK_ENABLED", "True").lower() in ["true", "1", "yes"])  # Enable pre-shutdown health checks
SHUTDOWN_BACKUP_ENABLED = bool(os.environ.get("SHUTDOWN_BACKUP_ENABLED", "True").lower() in ["true", "1", "yes"])  # Create backups during shutdown
SHUTDOWN_BACKUP_COUNT = int(os.environ.get("SHUTDOWN_BACKUP_COUNT", 5))  # Number of backup files to retain
```

**Section sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L2-L182)

**Referenced Files in This Document**   
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json) - *Updated with new Gemini API keys in commit 281c68a3*
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py) - *Enhanced with new shutdown settings in commit 5d169a6a*
- [shutdown_coordinator.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\shutdown_coordinator.py) - *Added enhanced shutdown functionality*
- [llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py) - *Gemini key management implementation*
- [pyproject.toml](file://c:\Users\ASUS\Documents\GitHub\RAVANA\pyproject.toml) - *Updated with new dependencies*