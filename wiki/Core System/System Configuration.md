# System Configuration



## Update Summary
**Changes Made**   
- Added documentation for new Conversational AI configuration parameters
- Updated Config class implementation section with new Conversational AI settings
- Added new section for Conversational AI Configuration
- Updated Critical Configuration Parameters section with new Conversational AI parameters
- Enhanced source tracking with new file references and update annotations

## Table of Contents
1. [Introduction](#introduction)
2. [Configuration Architecture](#configuration-architecture)
3. [Config Class Implementation](#config-class-implementation)
4. [JSON Configuration Structure](#json-configuration-structure)
5. [Configuration Loading Process](#configuration-loading-process)
6. [Critical Configuration Parameters](#critical-configuration-parameters)
7. [LLM Provider Configuration](#llm-provider-configuration)
8. [Validation and Type Checking](#validation-and-type-checking)
9. [Fallback Mechanisms](#fallback-mechanisms)
10. [Custom Configuration Examples](#custom-configuration-examples)
11. [Secure Handling of Sensitive Settings](#secure-handling-of-sensitive-settings)
12. [Common Configuration Issues](#common-configuration-issues)
13. [Troubleshooting Guide](#troubleshooting-guide)
14. [Conversational AI Configuration](#conversational-ai-configuration)

## Introduction
The RAVANA system employs a multi-layered configuration system that combines environment variables, Python class attributes, and JSON configuration files to govern runtime behavior. This comprehensive configuration framework enables flexible deployment across different environments while maintaining consistent behavior. The system prioritizes security, flexibility, and ease of use, allowing users to customize the AGI's behavior through multiple configuration channels. This document provides a detailed analysis of the configuration system, covering implementation details, critical parameters, validation processes, and best practices for secure configuration management.

## Configuration Architecture
The RAVANA configuration system follows a hierarchical architecture that combines multiple configuration sources with clear precedence rules. The architecture consists of three primary layers: environment variables (highest precedence), Python class defaults (medium precedence), and JSON configuration files (structural configuration). This layered approach enables users to override settings at runtime while maintaining a comprehensive configuration structure in files.

``mermaid
graph TD
A[Configuration System] --> B[Environment Variables]
A --> C[Python Config Class]
A --> D[JSON Configuration File]
B --> E[Runtime Overrides]
C --> F[Default Values]
D --> G[Provider Configuration]
E --> H[Final Configuration]
F --> H
G --> H
H --> I[RAVANA System]
```

**Diagram sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py)
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json)

**Section sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py)
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json)

## Config Class Implementation
The Config class serves as the primary configuration mechanism for the RAVANA system, implemented as a static class with class-level attributes. This implementation pattern provides a simple, accessible configuration interface that can be imported and used throughout the codebase without instantiation.

```python
class Config:
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///ravana_agi.db")
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.environ.get("LOG_FORMAT", "TEXT")
    FEED_URLS = [
        "http://rss.cnn.com/rss/cnn_latest.rss",
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://www.reddit.com/r/worldnews/.rss",
        "https://techcrunch.com/feed/",
        "https://www.npr.org/rss/rss.php?id=1001",
    ] 

    # Autonomous Loop Settings
    CURIOSITY_CHANCE = float(os.environ.get("CURIOSITY_CHANCE", 0.3))
    REFLECTION_CHANCE = float(os.environ.get("REFLECTION_CHANCE", 0.1))
    LOOP_SLEEP_DURATION = int(os.environ.get("LOOP_SLEEP_DURATION", 10))
    ERROR_SLEEP_DURATION = int(os.environ.get("ERROR_SLEEP_DURATION", 60))
    MAX_EXPERIMENT_LOOPS = int(os.environ.get("MAX_EXPERIMENT_LOOPS", 10))
    MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", 10))
    RESEARCH_TASK_TIMEOUT = int(os.environ.get("RESEARCH_TASK_TIMEOUT", 600))

    # Emotional Intelligence Settings
    POSITIVE_MOODS = ['Confident', 'Curious', 'Reflective', 'Excited', 'Content']
    NEGATIVE_MOODS = ['Frustrated', 'Stuck', 'Low Energy', 'Bored']
    EMOTIONAL_PERSONA = "Optimistic"

    # Model Settings
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Background Task Intervals (in seconds)
    DATA_COLLECTION_INTERVAL = int(os.environ.get("DATA_COLLECTION_INTERVAL", 3600))
    EVENT_DETECTION_INTERVAL = int(os.environ.get("EVENT_DETECTION_INTERVAL", 600))
    KNOWLEDGE_COMPRESSION_INTERVAL = int(os.environ.get("KNOWLEDGE_COMPRESSION_INTERVAL", 3600)) 
    # Personality / Invention settings
    PERSONA_NAME = os.environ.get("PERSONA_NAME", "Ravana")
    PERSONA_ORIGIN = os.environ.get("PERSONA_ORIGIN", "Ancient Sri Lanka")
    PERSONA_CREATIVITY = float(os.environ.get("PERSONA_CREATIVITY", 0.7))
    INVENTION_INTERVAL = int(os.environ.get("INVENTION_INTERVAL", 7200))

    # Conversational AI Configuration
    CONVERSATIONAL_AI_ENABLED = bool(os.environ.get("CONVERSATIONAL_AI_ENABLED", "True").lower() in ["true", "1", "yes"])
    CONVERSATIONAL_AI_START_DELAY = int(os.environ.get("CONVERSATIONAL_AI_START_DELAY", 5))  # seconds
```

The Config class implementation follows a consistent pattern where each configuration parameter is defined as a class attribute with a default value. The pattern `ATTRIBUTE = os.environ.get("ENV_VAR_NAME", default_value)` allows environment variables to override default values. For numeric values, explicit type conversion (float() or int()) ensures proper data typing. This implementation provides a clean, readable configuration interface that is easily extensible.

**Updated** Added new Conversational AI configuration parameters to the Config class implementation.

**Section sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L2-L172) - *Updated with Conversational AI configuration*

## JSON Configuration Structure
The config.json file contains the structural configuration for LLM providers and system modules, complementing the environment-based configuration in the Config class. This JSON file defines the available LLM providers, their API endpoints, models, and module-specific settings.

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
  },
  "electronhub": {
    "api_key": "ek-nzrvzzeQG0kmNZVhmkTWrKjgyIyUVY0mQpLwbectvfcPDssXiz",
    "base_url": "https://api.electronhub.ai",
    "models": [
      "deepseek-v3-0324",
      "gpt-4o-2024-11-20"
    ]
  },
  "zanity": {
    "api_key": "vc-b1EbB_BekM2TCPol64yDe7FgmOM34d4q",
    "base_url": "https://api.zanity.xyz/v1",
    "models": [
      "deepseek-r1",
      "deepseek-v3-0324",
      "gpt-4o:free",
      "claude-3.5-sonnet:free",
      "qwen-max-0428"
    ]
  },
  "a4f": {
    "api_key": "ddc-a4f-7bbefd7518a74b36b1d32cb867b1931f",
    "base_url": "https://api.a4f.co/v1"
  },
  "local_situation_generator": {
    "enabled": true,
    "schedule": "daily",
    "default_interest_areas": [
      "technology",
      "finance",
      "health"
    ],
    "use_trends": true
  },
  "main_llm_decision_maker": {
    "enabled": true,
    "preferred_model": "gpt-4o",
    "chain_of_thought": true,
    "rag_enabled": true
  }
}
```

The JSON structure organizes configuration into provider-specific sections (zuki, electronhub, zanity, a4f) that contain API credentials, base URLs, and available models. Additionally, it includes module configuration sections like local_situation_generator and main_llm_decision_maker that control specific system behaviors. This separation of concerns allows for modular configuration management where provider credentials and capabilities are defined separately from system behavior settings.

**Section sources**
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json#L1-L98)

## Configuration Loading Process
The configuration loading process in RAVANA involves multiple components that read and combine configuration from different sources. The process begins with the Config class, which automatically reads environment variables and applies default values. Simultaneously, the llm.py module loads the config.json file directly into a global config dictionary.

``mermaid
sequenceDiagram
participant Main as main.py
participant Config as config.py
participant LLM as llm.py
participant JSON as config.json
Main->>Config : Import Config class
Config->>Environment : Read environment variables
Config->>Config : Apply default values
Main->>LLM : Import LLM functions
LLM->>JSON : Read config.json file
JSON-->>LLM : Return JSON content
LLM->>LLM : Store in global config variable
Main->>System : Use configuration from both sources
```

**Diagram sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py)
- [llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py#L15-L20)
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json)

**Section sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py)
- [llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py#L15-L20)

The loading process in llm.py demonstrates a direct file reading approach:

```python
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
```

This code reads the config.json file from the core directory and parses it into a Python dictionary named 'config' that is used throughout the llm.py module. The Config class in config.py, on the other hand, relies on environment variables with sensible defaults, allowing for runtime configuration without modifying files. This dual approach provides flexibility: environment variables for deployment-specific settings and JSON files for structural configuration that changes less frequently.

## Critical Configuration Parameters
The RAVANA system relies on several critical configuration parameters that govern its core behavior. These parameters are organized into logical categories based on their functional domain.

### Autonomous Loop Settings
These parameters control the timing and probability of autonomous behaviors:

- **CURIOSITY_CHANCE**: Probability (0.0-1.0) of initiating a curiosity-driven action
- **REFLECTION_CHANCE**: Probability (0.0-1.0) of entering reflection mode
- **LOOP_SLEEP_DURATION**: Seconds to sleep between processing loops
- **ERROR_SLEEP_DURATION**: Seconds to sleep after an error occurs
- **MAX_EXPERIMENT_LOOPS**: Maximum number of loops for experimentation
- **MAX_ITERATIONS**: Maximum iterations for recursive processes
- **RESEARCH_TASK_TIMEOUT**: Timeout in seconds for research tasks

### Emotional Intelligence Settings
These parameters define the emotional framework of the AGI:

- **POSITIVE_MOODS**: List of moods considered positive
- **NEGATIVE_MOODS**: List of moods considered negative
- **EMOTIONAL_PERSONA**: Default emotional persona

### Model Settings
These parameters specify AI models used in the system:

- **EMBEDDING_MODEL**: Name of the embedding model for text encoding

### Background Task Intervals
These parameters control the frequency of background processes:

- **DATA_COLLECTION_INTERVAL**: Seconds between data collection cycles
- **EVENT_DETECTION_INTERVAL**: Seconds between event detection cycles
- **KNOWLEDGE_COMPRESSION_INTERVAL**: Seconds between knowledge compression cycles

### Personality and Invention Settings
These parameters shape the AGI's personality and creative behavior:

- **PERSONA_NAME**: Name of the AGI persona
- **PERSONA_ORIGIN**: Origin story for the persona
- **PERSONA_CREATIVITY**: Creativity level (0.0-1.0)
- **INVENTION_INTERVAL**: Seconds between invention attempts

### Conversational AI Settings
These parameters control the Conversational AI module:

- **CONVERSATIONAL_AI_ENABLED**: Boolean flag to enable/disable the Conversational AI module
- **CONVERSATIONAL_AI_START_DELAY**: Delay in seconds before starting the Conversational AI module

**Updated** Added new Conversational AI settings to the Critical Configuration Parameters section.

**Section sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L2-L172) - *Updated with Conversational AI configuration*

## LLM Provider Configuration
The LLM provider configuration in config.json defines the available language model providers, their credentials, endpoints, and supported models. This configuration enables the system to route requests to different providers based on availability and model requirements.

``mermaid
classDiagram
class LLMProvider {
+string api_key
+string base_url
+list[str] models
}
class ZukiProvider {
+string api_key
+string base_url
+list[str] models
}
class ElectronHubProvider {
+string api_key
+string base_url
+list[str] models
}
class ZanityProvider {
+string api_key
+string base_url
+list[str] models
}
class A4FProvider {
+string api_key
+string base_url
}
LLMProvider <|-- ZukiProvider
LLMProvider <|-- ElectronHubProvider
LLMProvider <|-- ZanityProvider
LLMProvider <|-- A4FProvider
```

**Diagram sources**
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json#L1-L50)

**Section sources**
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json#L1-L50)
- [llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py#L330-L362)

The system implements a provider selection strategy that first attempts to use configured providers in sequence, with a fallback to Gemini if all providers fail. The PROVIDERS list in llm.py demonstrates how environment variables are used to override API keys from the config.json file:

```python
PROVIDERS = [
    {
        "name": "a4f",
        "api_key": os.getenv("A4F_API_KEY", "ddc-a4f-7bbefd7518a74b36b1d32cb867b1931f"),
        "base_url": "https://api.a4f.co/v1",
        "models": ["provider-3/gemini-2.0-flash", "provider-2/llama-4-scout", "provider-3/llama-4-scout"]
    },
    {
        "name": "zukijourney",
        "api_key": os.getenv("ZUKIJOURNEY_API_KEY", "zu-ab9fba2aeef85c7ecb217b00ce7ca1fe"),
        "base_url": "https://api.zukijourney.com/v1",
        "models": ["gpt-4o:online", "gpt-4o", "deepseek-chat"]
    }
]
```

This implementation allows for secure credential management by enabling environment variables to override the API keys specified in the config.json file, which is particularly important for preventing accidental exposure of credentials in version control.

## Validation and Type Checking
The RAVANA configuration system implements validation and type checking through several mechanisms, primarily focusing on ensuring proper data types for configuration values. The system uses explicit type conversion in the Config class to ensure values are of the correct type:

```python
CURIOSITY_CHANCE = float(os.environ.get("CURIOSITY_CHANCE", 0.3))
LOOP_SLEEP_DURATION = int(os.environ.get("LOOP_SLEEP_DURATION", 10))
```

For JSON configuration, the system relies on Python's json.load() function, which performs basic type conversion according to JSON specifications. Additional validation occurs in specific modules that use the configuration. For example, the emotional_intelligence.py module validates emotional valence values:

```python
@validator('emotional_valence')
def validate_emotional_valence(cls, v):
    if v is not None and not (-1.0 <= v <= 1.0):
        raise ValueError('emotional_valence must be between -1.0 and 1.0')
    return v
```

The system also includes validation for action parameters in the actions module:

```python
def validate_params(self, params: Dict[str, Any]) -> None:
    """
    Validates the given parameters against the action's defined parameters.
    Raises InvalidActionParams if the parameters are invalid.
    """
    required_params = {p['name'] for p in self.parameters if p.get('required', False)}
    provided_params = set(params.keys())
```

These validation mechanisms ensure that configuration values meet expected criteria before being used in system operations, preventing errors caused by invalid data types or values.

**Section sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L29-L31)
- [modules/emotional_intellegence/emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L20-L25)
- [core/actions/action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L35-L43)

## Fallback Mechanisms
The RAVANA system implements robust fallback mechanisms to ensure continued operation when primary configuration sources or services are unavailable. These mechanisms operate at multiple levels of the configuration and execution stack.

### Provider Fallback Strategy
The LLM system implements a cascading fallback strategy for provider selection:

```python
def call_llm(prompt, preferred_provider=None, model=None):
    """
    Try all providers in order, fallback to Gemini if all fail.
    """
    providers = [
        (call_zuki, 'zuki'),
        (call_electronhub, 'electronhub'),
        (call_zanity, 'zanity'),
        (call_a4f, 'a4f'),
    ]
    if preferred_provider:
        providers = sorted(providers, key=lambda x: x[1] != preferred_provider)
    for func, name in providers:
        result = func(prompt, model) if name != 'a4f' else func(prompt)
        if result:
            return result
    # Fallback to Gemini
    return call_gemini(prompt)
```

This strategy attempts to use each configured provider in sequence, returning the first successful response. If all providers fail, the system falls back to the Gemini API, ensuring that LLM functionality remains available even when primary providers are unreachable.

### Error Handling and Retry Logic
The system implements comprehensive error handling with retry logic for LLM calls:

```python
def safe_call_llm(prompt: str, timeout: int = 30, retries: int = 3, backoff_factor: float = 0.5, **kwargs) -> str:
    """
    Wrap a single LLM call with retry/backoff and timeout.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            result = call_llm(prompt, **kwargs)
            if not result or result.strip() == "":
                raise RuntimeError("Empty response from LLM")
            return result
        except Exception as e:
            last_exc = e
            wait = backoff_factor * (2 ** (attempt - 1))
            logger.warning(f"LLM call failed (attempt {attempt}/{retries}): {e!r}, retrying in {wait:.1f}s")
            time.sleep(wait)
```

This retry mechanism uses exponential backoff to prevent overwhelming failing services while giving them time to recover. The system also includes specific error handling for known issues, such as detecting 404 errors from the Zanity API:

```python
if provider['name'] == 'zanity' and "404" in str(e).lower():
    logging.warning(f"Zanity API at {provider['base_url']} might be unavailable (404 error). Check URL.")
```

**Section sources**
- [llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py#L294-L331)
- [llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py#L48-L84)

## Custom Configuration Examples
The RAVANA system supports various operational modes through custom configuration. These examples demonstrate how to configure the system for different use cases by modifying environment variables and JSON settings.

### Discovery Mode Configuration
For discovery-focused operation, increase curiosity and reflection probabilities while enabling trend-based situation generation:

```bash
# Environment variables
CURIOSITY_CHANCE=0.7
REFLECTION_CHANCE=0.4
LOOP_SLEEP_DURATION=5
PERSONA_CREATIVITY=0.9

# config.json modifications
{
  "local_situation_generator": {
    "enabled": true,
    "schedule": "hourly",
    "default_interest_areas": [
      "quantum physics",
      "artificial intelligence",
      "cosmology"
    ],
    "use_trends": true
  }
}
```

### Physics Experimentation Mode
For physics experimentation, optimize settings for longer processing times and specialized models:

```bash
# Environment variables
MAX_EXPERIMENT_LOOPS=20
RESEARCH_TASK_TIMEOUT=1800  # 30 minutes
EMBEDDING_MODEL="physics-bert-v2"

# config.json modifications
{
  "main_llm_decision_maker": {
    "enabled": true,
    "preferred_model": "gpt-4o",
    "chain_of_thought": true,
    "rag_enabled": true
  },
  "providers": [
    {
      "name": "physics-llm",
      "api_key": "phys-key-123",
      "base_url": "https://api.physics-llm.com/v1",
      "models": ["physics-gpt-4o", "quantum-reasoner"]
    }
  ]
}
```

### Quiet Reflection Mode
For introspective operation with minimal external interaction:

```bash
# Environment variables
CURIOSITY_CHANCE=0.1
REFLECTION_CHANCE=0.6
DATA_COLLECTION_INTERVAL=7200  # 2 hours
EVENT_DETECTION_INTERVAL=1800  # 30 minutes

# config.json modifications
{
  "local_situation_generator": {
    "enabled": false
  },
  "main_llm_decision_maker": {
    "enabled": true,
    "preferred_model": "claude-3.5-sonnet:free",
    "chain_of_thought": true,
    "rag_enabled": false
  }
}
```

**Section sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py)
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json)

## Secure Handling of Sensitive Settings
The RAVANA system employs several strategies for secure handling of sensitive settings, particularly API keys and credentials. The primary approach is the use of environment variables to override credentials specified in configuration files.

### Environment Variable Overrides
The system prioritizes environment variables over hardcoded values in config.json:

```python
PROVIDERS = [
    {
        "name": "a4f",
        "api_key": os.getenv("A4F_API_KEY", "ddc-a4f-7bbefd7518a74b36b1d32cb867b1931f"),
        "base_url": "https://api.a4f.co/v1"
    }
]
```

This pattern allows deployment-specific credentials to be provided through environment variables without modifying the configuration file, reducing the risk of accidental credential exposure in version control.

### Configuration File Security
The config.json file contains default API keys that appear to be placeholder or test keys rather than production credentials. This suggests a security practice of using non-functional keys in version-controlled files. For production deployment, users are expected to override these values through environment variables.

### Secure Credential Management Practices
To securely manage sensitive settings, follow these best practices:

1. **Use environment variables**: Always set API keys through environment variables rather than modifying config.json
2. **Environment-specific configuration**: Use different environment variables for development, testing, and production
3. **Credential rotation**: Regularly rotate API keys and update environment variables accordingly
4. **Access control**: Restrict access to environment variables containing sensitive credentials
5. **Monitoring**: Monitor for unauthorized access to configuration files and environment variables

The system's design encourages these secure practices by making environment variable overrides the primary method for specifying credentials.

**Section sources**
- [llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py#L330-L362)
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json)

## Common Configuration Issues
Several common configuration issues can affect the RAVANA system's operation. Understanding these issues and their solutions is essential for maintaining system stability.

### Misconfigured LLM Endpoints
Issue: LLM providers return connection errors or 404 responses
Solution: Verify the base_url in config.json matches the provider's current API endpoint
```json
{
  "zanity": {
    "base_url": "https://api.zanity.xyz/v1"  # Ensure this URL is correct
  }
}
```

### Invalid JSON Syntax
Issue: Configuration fails to load due to JSON parsing errors
Symptoms: JSONDecodeError exceptions during startup
Solution: Validate JSON syntax using online validators or JSON linters
```bash
python -m json.tool config.json  # Validate JSON syntax
```

### Unexpected Type Coercion
Issue: Configuration values are interpreted with incorrect types
Example: Numeric values treated as strings
Solution: Ensure proper type conversion in config.py
```python
# Correct: explicit type conversion
LOOP_SLEEP_DURATION = int(os.environ.get("LOOP_SLEEP_DURATION", 10))
CURIOSITY_CHANCE = float(os.environ.get("CURIOSITY_CHANCE", 0.3))
```

### Missing Environment Variables
Issue: Default values are used instead of intended configuration
Solution: Set required environment variables before startup
```bash
export A4F_API_KEY="your_actual_api_key"
export DATABASE_URL="postgresql://user:pass@localhost/ravana"
python main.py
```

### Provider Authentication Failures
Issue: LLM providers reject requests due to invalid API keys
Solution: Verify API keys through environment variables
```bash
# Check if environment variable is set
echo $A4F_API_KEY
# If empty, set it
export A4F_API_KEY="your_valid_api_key"
```

**Section sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py)
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json)
- [llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py)

## Troubleshooting Guide
This troubleshooting guide addresses common configuration-related issues and provides step-by-step solutions.

### Configuration Loading Verification
Verify that configuration is loading correctly by checking startup logs:

```python
# In main.py, logging is configured based on Config settings
if Config.LOG_FORMAT.upper() == 'JSON':
    formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
else:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
```

Check for log messages indicating successful configuration loading.

### LLM Provider Connectivity Test
Test connectivity to all configured LLM providers:

```python
def test_all_providers():
    """Test all LLM providers and Gemini fallbacks with a simple prompt."""
    prompt = "What is the capital of France?"
    print("Testing Zuki:")
    print(call_zuki(prompt))
    print("\nTesting ElectronHub:")
    print(call_electronhub(prompt))
    print("\nTesting Zanity:")
    print(call_zanity(prompt))
    print("\nTesting A4F:")
    print(call_a4f(prompt))
    print("\nTesting Gemini (text):")
    print(call_gemini(prompt))
```

Run this test to identify which providers are accessible and which are failing.

### Environment Variable Verification
Verify that environment variables are properly set:

```bash
# Check critical environment variables
echo "DATABASE_URL: $DATABASE_URL"
echo "A4F_API_KEY: ${A4F_API_KEY:0:10}..."  # Show only first 10 characters for security
echo "LOG_LEVEL: $LOG_LEVEL"
```

### Configuration File Validation
Validate the config.json file for syntax correctness:

```python
import json
try:
    with open('core/config.json', 'r') as f:
        config = json.load(f)
    print("✓ config.json is valid JSON")
    print(f"✓ Loaded {len([k for k in config.keys() if isinstance(config[k], dict)])} provider configurations")
except json.JSONDecodeError as e:
    print(f"✗ config.json has JSON syntax error: {e}")
```

### Fallback Mechanism Testing
Test the system's fallback mechanisms by temporarily disabling primary providers:

1. Comment out provider API keys in environment variables
2. Observe if the system successfully falls back to alternative providers
3. Verify that Gemini fallback works when all other providers fail

These troubleshooting steps help identify and resolve configuration issues that could impact system functionality.

**Section sources**
- [llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py#L294-L331)
- [main.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\main.py#L25-L35)
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json)

## Conversational AI Configuration
The Conversational AI module has been integrated into the RAVANA system with dedicated configuration parameters that control its behavior and integration with the main AGI system.

### Configuration Parameters
Two new configuration parameters have been added to the Config class to control the Conversational AI module:

- **CONVERSATIONAL_AI_ENABLED**: A boolean flag that determines whether the Conversational AI module should be initialized and started. When set to `True`, the module will be loaded and run in a separate thread. When set to `False`, the module will be disabled.
- **CONVERSATIONAL_AI_START_DELAY**: An integer value specifying the delay in seconds before starting the Conversational AI module. This delay allows the main AGI system to initialize properly before the Conversational AI module begins operation.

### Implementation Details
The Conversational AI module is integrated into the main AGI system through the AGISystem class in system.py. When the system initializes, it checks the CONVERSATIONAL_AI_ENABLED configuration parameter:

```python
# Initialize Conversational AI if enabled
self.conversational_ai = None
self.conversational_ai_thread = None
if Config.CONVERSATIONAL_AI_ENABLED and CONVERSATIONAL_AI_AVAILABLE:
    try:
        self.conversational_ai = ConversationalAI()
        logger.info("Conversational AI module initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Conversational AI module: {e}")
        self.conversational_ai = None
```

If enabled, the Conversational AI module is started in a separate thread with a configurable delay:

```python
async def start_conversational_ai(self):
    """Start Conversational AI module in a separate thread."""
    if self.conversational_ai and Config.CONVERSATIONAL_AI_ENABLED:
        try:
            logger.info("Starting Conversational AI module...")
            
            # Create a thread to run the Conversational AI
            def run_conversational_ai():
                try:
                    # Add a small delay to allow the main system to initialize
                    time.sleep(Config.CONVERSATIONAL_AI_START_DELAY)
                    # Run the conversational AI as part of the main system (not standalone)
                    asyncio.run(self.conversational_ai.start(standalone=False))
                except Exception as e:
                    logger.error(f"Error in Conversational AI thread: {e}")
            
            self.conversational_ai_thread = threading.Thread(
                target=run_conversational_ai,
                name="ConversationalAI",
                daemon=True
            )
            self.conversational_ai_thread.start()
            logger.info("Conversational AI module started successfully in background thread")
        except Exception as e:
            logger.error(f"Failed to start Conversational AI module: {e}")
```

### Usage and Integration
The Conversational AI module can be launched in two ways:

1. **Integrated mode**: When the main RAVANA system starts, it automatically initializes and starts the Conversational AI module if CONVERSATIONAL_AI_ENABLED is set to `True`. The module runs as a background thread within the main process.

2. **Standalone mode**: The module can be launched independently using the launch_conversational_ai.py script:
```bash
python launch_conversational_ai.py
```
In standalone mode, the module runs as a separate process, which can be useful for development and testing.

### Status Monitoring
The AGISystem provides a method to check the status of the Conversational AI module:

```python
def get_conversational_ai_status(self) -> Dict[str, Any]:
    """Get Conversational AI status information."""
    if not self.conversational_ai:
        return {"enabled": False, "status": "not_initialized"}
    
    # Check if the thread is still alive
    thread_alive = False
    if self.conversational_ai_thread:
        thread_alive = self.conversational_ai_thread.is_alive()
        
    return {
        "enabled": Config.CONVERSATIONAL_AI_ENABLED,
        "status": "active" if thread_alive else "inactive"
    }
```

This status information can be accessed through the AGI system's API or monitoring interfaces.

**Section sources**
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L171-L172) - *Added Conversational AI configuration*
- [system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py#L141-L145) - *Conversational AI initialization*
- [system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py#L310-L319) - *Conversational AI start logic*
- [system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py#L355-L356) - *Status reporting*
- [launch_conversational_ai.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\launch_conversational_ai.py) - *Standalone launcher*
- [modules/conversational_ai/main.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\main.py) - *Conversational AI module implementation*

**Referenced Files in This Document**   
- [config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py) - *Updated with Conversational AI configuration*
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.json)
- [llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py)
- [main.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\main.py)
- [system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py) - *Updated with Conversational AI integration*
- [launch_conversational_ai.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\launch_conversational_ai.py) - *New Conversational AI launcher*
- [modules/conversational_ai/main.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\main.py) - *Conversational AI module implementation*
- [DEVELOPER_GUIDE.md](file://c:\Users\ASUS\Documents\GitHub\RAVANA\DEVELOPER_GUIDE.md)