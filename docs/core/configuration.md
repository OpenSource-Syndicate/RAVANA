# RAVANA AGI Configuration Guide

This guide explains how to configure the RAVANA AGI system for different environments and use cases.

## Configuration Overview

RAVANA AGI uses a hierarchical configuration system that allows for flexible customization:

1. **Default Configuration**: Built-in defaults for all parameters
2. **System Configuration**: Global settings in `core/config.json`
3. **Module Configuration**: Component-specific settings in module directories
4. **Environment Variables**: Runtime overrides through environment variables
5. **Command-line Arguments**: Immediate overrides for specific executions

## Configuration Files

### Core Configuration

The main configuration file is located at `core/config.json`:

```json
{
    "system": {
        "name": "RAVANA AGI",
        "version": "1.0.0",
        "log_level": "INFO",
        "data_directory": "./data",
        "log_directory": "./logs"
    },
    "modules": {
        "enabled": [
            "adaptive_learning",
            "agent_self_reflection",
            "curiosity_trigger",
            "decision_engine",
            "emotional_intellegence",
            "episodic_memory",
            "event_detection",
            "information_processing",
            "knowledge_compression",
            "personality",
            "situation_generator"
        ]
    },
    "services": {
        "data_service": {
            "backend": "sqlite",
            "connection_string": "sqlite:///ravana_agi.db"
        },
        "memory_service": {
            "episodic_backend": "chromadb",
            "semantic_backend": "chromadb"
        }
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2048
    }
}
```

### Module Configuration

Each module can have its own configuration file. For example, `modules/emotional_intellegence/config.json`:

```json
{
    "mood_dimensions": {
        "happiness": {
            "min": -1.0,
            "max": 1.0,
            "default": 0.0
        },
        "curiosity": {
            "min": 0.0,
            "max": 1.0,
            "default": 0.5
        },
        "frustration": {
            "min": 0.0,
            "max": 1.0,
            "default": 0.1
        }
    },
    "personality": {
        "openness": 0.8,
        "conscientiousness": 0.7,
        "extraversion": 0.6,
        "agreeableness": 0.7,
        "neuroticism": 0.3
    }
}
```

## Environment Variables

Configuration can be overridden using environment variables:

```bash
# System settings
export RAVANA_LOG_LEVEL=DEBUG
export RAVANA_DATA_DIR=/var/lib/ravana
export RAVANA_LOG_DIR=/var/log/ravana

# LLM settings
export RAVANA_LLM_PROVIDER=anthropic
export RAVANA_LLM_MODEL=claude-3-opus
export RAVANA_LLM_TEMPERATURE=0.8

# Service settings
export RAVANA_DATABASE_URL=postgresql://user:pass@localhost/ravana
```

## Command-line Arguments

Runtime configuration can be provided through command-line arguments:

```bash
python main.py --log-level DEBUG --llm-model gpt-4-turbo --data-dir /custom/data
```

## Configuration Parameters

### System Parameters

| Parameter | Description | Default | Valid Values |
|-----------|-------------|---------|--------------|
| `system.name` | System identifier | "RAVANA AGI" | String |
| `system.version` | System version | "1.0.0" | String |
| `system.log_level` | Logging verbosity | "INFO" | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `system.data_directory` | Data storage location | "./data" | Valid directory path |
| `system.log_directory` | Log storage location | "./logs" | Valid directory path |

### Module Parameters

| Parameter | Description | Default | Valid Values |
|-----------|-------------|---------|--------------|
| `modules.enabled` | Active modules | All modules | Array of module names |
| `modules.[module].enabled` | Individual module activation | true | Boolean |

### Service Parameters

| Parameter | Description | Default | Valid Values |
|-----------|-------------|---------|--------------|
| `services.data_service.backend` | Data storage backend | "sqlite" | sqlite, postgresql, mysql |
| `services.data_service.connection_string` | Database connection | "sqlite:///ravana_agi.db" | Valid connection string |
| `services.memory_service.episodic_backend` | Episodic memory backend | "chromadb" | chromadb, postgresql |
| `services.memory_service.semantic_backend` | Semantic memory backend | "chromadb" | chromadb, postgresql |

### LLM Parameters

| Parameter | Description | Default | Valid Values |
|-----------|-------------|---------|--------------|
| `llm.provider` | LLM service provider | "openai" | openai, anthropic, google, local |
| `llm.model` | Specific model to use | "gpt-4" | Provider-specific model names |
| `llm.temperature` | Response randomness | 0.7 | 0.0 - 2.0 |
| `llm.max_tokens` | Maximum response length | 2048 | Integer |

## Advanced Configuration

### Custom Module Configuration

To add configuration for a custom module:

1. Create a configuration file in the module directory
2. Define parameters in JSON format
3. Access in code through the configuration service

```python
# In your module
config = self.config.get('my_custom_module', {})
parameter = config.get('my_parameter', 'default_value')
```

### Dynamic Configuration

Configuration can be updated dynamically during runtime:

```python
from core.config import ConfigManager

config_manager = ConfigManager()
config_manager.update('llm.temperature', 0.9)
```

### Configuration Validation

The system validates configuration parameters at startup:

- Type checking for all parameters
- Range validation for numeric values
- Enum validation for predefined options
- Required parameter verification

## Security Configuration

### API Security

API security settings:

```json
{
    "security": {
        "api_keys": {
            "enabled": true,
            "key_file": "./config/api_keys.json"
        },
        "rate_limiting": {
            "enabled": true,
            "requests_per_minute": 60
        },
        "cors": {
            "enabled": true,
            "allowed_origins": ["localhost:3000"]
        }
    }
}
```

### Data Encryption

Data encryption settings:

```json
{
    "encryption": {
        "at_rest": {
            "enabled": true,
            "algorithm": "AES-256"
        },
        "in_transit": {
            "enabled": true,
            "protocol": "TLSv1.3"
        }
    }
}
```

## Performance Tuning

### Memory Settings

Memory optimization parameters:

```json
{
    "performance": {
        "memory": {
            "cache_size": 1000,
            "eviction_policy": "lru",
            "compression_threshold": 1024
        },
        "concurrency": {
            "max_threads": 8,
            "max_processes": 4
        }
    }
}
```

### Database Optimization

Database performance settings:

```json
{
    "database": {
        "connection_pool": {
            "min_connections": 5,
            "max_connections": 20
        },
        "query_optimization": {
            "enable_caching": true,
            "cache_ttl": 300
        }
    }
}
```

## Monitoring Configuration

### Logging Settings

Logging configuration:

```json
{
    "logging": {
        "handlers": {
            "file": {
                "enabled": true,
                "level": "INFO",
                "filename": "ravana.log",
                "max_size": "100MB",
                "backup_count": 5
            },
            "console": {
                "enabled": true,
                "level": "WARNING"
            }
        },
        "format": {
            "timestamp": true,
            "level": true,
            "module": true
        }
    }
}
```

### Metrics Collection

Metrics configuration:

```json
{
    "metrics": {
        "collection": {
            "enabled": true,
            "interval": 60
        },
        "exporters": {
            "prometheus": {
                "enabled": true,
                "endpoint": "/metrics"
            },
            "json": {
                "enabled": true,
                "filename": "metrics.json"
            }
        }
    }
}
```

## Troubleshooting Configuration

### Common Issues

1. **Invalid Configuration Values**
   - Check parameter types and ranges
   - Validate against schema
   - Review documentation for valid values

2. **Missing Required Parameters**
   - Ensure all required parameters are present
   - Check for typos in parameter names
   - Verify configuration file syntax

3. **Environment Variable Conflicts**
   - Check for conflicting environment variables
   - Verify variable name mappings
   - Confirm precedence rules

### Configuration Diagnostics

Use the built-in configuration diagnostic tool:

```bash
python main.py --diagnose-config
```

This will:
- Validate all configuration parameters
- Report any issues or inconsistencies
- Show effective configuration values
- Suggest corrections for common problems

## Best Practices

### Configuration Management

1. **Version Control**: Keep configuration files in version control
2. **Environment Separation**: Use different configs for dev/test/prod
3. **Documentation**: Document all custom parameters
4. **Validation**: Validate configurations before deployment
5. **Backup**: Backup configuration files regularly

### Security Best Practices

1. **Secrets Management**: Never store secrets in config files
2. **Access Control**: Restrict access to configuration files
3. **Encryption**: Encrypt sensitive configuration data
4. **Audit**: Log configuration changes
5. **Review**: Regularly review configuration security

### Performance Best Practices

1. **Caching**: Use appropriate caching strategies
2. **Resource Limits**: Set reasonable resource limits
3. **Monitoring**: Monitor configuration impact on performance
4. **Optimization**: Tune parameters based on usage patterns
5. **Testing**: Test configuration changes before deployment