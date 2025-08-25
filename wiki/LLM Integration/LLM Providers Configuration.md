# LLM Providers Configuration



## Table of Contents
1. [Introduction](#introduction)
2. [Configuration Loading Mechanism](#configuration-loading-mechanism)
3. [Provider Integration and Authentication](#provider-integration-and-authentication)
4. [Gemini Key Management System](#gemini-key-management-system)
5. [Provider Initialization Process](#provider-initialization-process)
6. [Error Handling and Fallback Strategies](#error-handling-and-fallback-strategies)
7. [Security Best Practices](#security-best-practices)
8. [Troubleshooting Guide](#troubleshooting-guide)

## Introduction
This document provides comprehensive documentation for the LLM provider configuration system in the RAVANA project. It details how multiple LLM providers (Zuki, ElectronHub, Zanity, A4F, and Gemini) are integrated, authenticated, and managed within the application. The configuration system supports both environment variables and configuration files for API key management, with sophisticated fallback mechanisms and security practices. The document covers the initialization process, error handling, and best practices for maintaining secure and reliable LLM integrations.

## Configuration Loading Mechanism
The system implements a multi-layered configuration loading mechanism that prioritizes environment variables over configuration files, providing flexibility for different deployment environments.

```mermaid
flowchart TD
Start([Configuration Loading]) --> EnvCheck["Check Environment Variables"]
EnvCheck --> ConfigFile["Load config.json"]
ConfigFile --> Merge["Merge Configurations"]
Merge --> Default["Apply Default Values"]
Default --> Validation["Validate Configuration"]
Validation --> Ready["Configuration Ready"]
style Start fill:#4CAF50,stroke:#388E3C
style Ready fill:#4CAF50,stroke:#388E3C
```

**Diagram sources**
- [core/config.py](file://core/config.py#L29-L129)
- [core/llm.py](file://core/llm.py#L46-L84)

**Section sources**
- [core/config.py](file://core/config.py#L29-L129)

## Provider Integration and Authentication
The system integrates with five LLM providers: Zuki, ElectronHub, Zanity, A4F, and Gemini. Each provider is configured with specific API endpoints, authentication methods, and model options.

### Provider Configuration Structure
The configuration follows a consistent pattern across providers, with API keys, base URLs, and model specifications defined in both environment variables and configuration files.

**Provider Configuration Parameters**
- **api_key**: Authentication credential for the provider
- **base_url**: Endpoint URL for API requests
- **models**: Available models with their capabilities
- **priority**: Order of preference when multiple providers are available

### Environment Variable Overrides
The system allows environment variables to override configuration file settings, enabling secure key management in production environments.

```python
# Example from core/llm.py
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

**Section sources**
- [core/llm.py](file://core/llm.py#L618-L658)
- [core/config.json](file://core/config.json#L55-L115)

## Gemini Key Management System
The Gemini provider features an advanced key management system that supports multiple API keys with priority-based selection, rate limiting detection, and automatic fallback.

### Key Manager Architecture
The `GeminiKeyManager` class implements a sophisticated system for managing multiple API keys with failover capabilities.

```mermaid
classDiagram
class GeminiKeyManager {
+_keys : Dict[str, GeminiKeyStatus]
+_lock : threading.Lock
+rate_limit_indicators : List[str]
+__init__()
+_load_keys_from_config()
+get_available_key() GeminiKeyStatus
+mark_key_success(key_id : str)
+mark_key_failed(key_id : str, error : Exception)
+mark_key_rate_limited(key_id : str, reset_time : datetime)
+is_rate_limit_error(error : Exception) Tuple[bool, Optional[datetime]]
+get_key_statistics() Dict[str, Any]
}
class GeminiKeyStatus {
+key_id : str
+api_key : str
+priority : int
+is_available : bool
+rate_limit_reset_time : Optional[datetime]
+total_requests : int
+failure_count : int
+consecutive_failures : int
+last_success : Optional[datetime]
}
GeminiKeyManager --> GeminiKeyStatus : "manages"
```

**Diagram sources**
- [core/llm.py](file://core/llm.py#L47-L220)

**Section sources**
- [core/llm.py](file://core/llm.py#L46-L114)

### Key Loading Process
The key manager loads API keys from multiple sources in a specific priority order:

1. **Configuration File**: Keys defined in `config.json` with explicit priorities
2. **Environment Variables**: Keys from `GEMINI_API_KEY_1` to `GEMINI_API_KEY_20`
3. **Fallback Key**: Hardcoded key used when no other keys are available

```python
def _load_keys_from_config(self):
    """Load API keys from configuration file."""
    try:
        gemini_config = config.get('gemini', {})
        api_keys = gemini_config.get('api_keys', [])
        
        # Also check environment variables for additional keys
        env_keys = []
        for i in range(1, 21):  # Check for up to 20 environment variables
            env_key = os.getenv(f"GEMINI_API_KEY_{i}")
            if env_key:
                env_keys.append({
                    "id": f"gemini_env_key_{i}",
                    "key": env_key,
                    "priority": 100 + i  # Lower priority than config keys
                })
        
        all_keys = api_keys + env_keys
        
        # Add fallback key if no keys found
        if not all_keys:
            logger.warning("No API keys found in config or environment, using fallback key")
            fallback_key = {
                "id": "fallback_key",
                "key": "AIzaSyAWR9C57V2f2pXFwjtN9jkNYKA_ou5Hdo4",
                "priority": 999
            }
            all_keys.append(fallback_key)
```

**Section sources**
- [core/llm.py](file://core/llm.py#L46-L84)

## Provider Initialization Process
The LLM providers are initialized through a systematic process that handles API key retrieval, base URL configuration, and model selection.

### Initialization Flow
```mermaid
flowchart TD
InitStart([Provider Initialization]) --> KeyCheck["Retrieve API Key"]
KeyCheck --> EnvVar["Check Environment Variables"]
EnvVar --> ConfigFile["Check config.json"]
ConfigFile --> Default["Use Default/Fallback Key"]
Default --> URLConfig["Set Base URL"]
URLConfig --> ModelSelect["Select Default Model"]
ModelSelect --> ClientSetup["Initialize API Client"]
ClientSetup --> Ready["Provider Ready"]
style InitStart fill:#4CAF50,stroke:#388E3C
style Ready fill:#4CAF50,stroke:#388E3C
```

**Diagram sources**
- [core/llm.py](file://core/llm.py#L474-L507)
- [core/config.py](file://core/config.py#L29-L129)

### Provider-Specific Implementation
Each provider has a dedicated function for handling API calls with consistent error handling patterns.

```python
def call_a4f(prompt):
    try:
        api_key = config['a4f']['api_key']
        base_url = config['a4f']['base_url']
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}]}
        r = requests.post(url, headers=headers, json=data, timeout=20)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return None

def call_zanity(prompt, model=None):
    try:
        api_key = config['zanity']['api_key']
        base_url = config['zanity']['base_url']
        model = model or config['zanity']['models'][0]
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        r = requests.post(url, headers=headers, json=data, timeout=20)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return None
```

**Section sources**
- [core/llm.py](file://core/llm.py#L474-L507)

## Error Handling and Fallback Strategies
The system implements comprehensive error handling with multiple fallback layers to ensure reliability.

### Provider Fallback Chain
The `call_llm` function implements a cascading fallback strategy across providers.

```mermaid
sequenceDiagram
participant User
participant LLM as call_llm
participant Zuki
participant ElectronHub
participant Zanity
participant A4F
participant Gemini
User->>LLM : Request
LLM->>Zuki : Try Zuki
alt Zuki Success
Zuki-->>LLM : Response
LLM-->>User : Response
else Zuki Failure
LLM->>ElectronHub : Try ElectronHub
alt ElectronHub Success
ElectronHub-->>LLM : Response
LLM-->>User : Response
else ElectronHub Failure
LLM->>Zanity : Try Zanity
alt Zanity Success
Zanity-->>LLM : Response
LLM-->>User : Response
else Zanity Failure
LLM->>A4F : Try A4F
alt A4F Success
A4F-->>LLM : Response
LLM-->>User : Response
else A4F Failure
LLM->>Gemini : Fallback to Gemini
Gemini-->>LLM : Response
LLM-->>User : Response
end
end
end
end
```

**Diagram sources**
- [core/llm.py](file://core/llm.py#L597-L614)

**Section sources**
- [core/llm.py](file://core/llm.py#L587-L621)

### Enhanced Gemini Error Handling
The Gemini provider includes sophisticated error handling with key rotation and rate limit detection.

```python
def call_gemini_with_fallback(prompt: str, function_type: str = "text", max_retries: int = 3, **kwargs) -> str:
    """
    Enhanced Gemini caller with automatic key rotation and rate limit handling.
    """
    last_error = None
    
    for attempt in range(max_retries):
        key_status = gemini_key_manager.get_available_key()
        
        if not key_status:
            logger.error("No Gemini API keys available for request")
            return f"[All Gemini API keys exhausted: {last_error}]"
        
        try:
            # Call appropriate function based on type
            if function_type == "text":
                result = _call_gemini_text(prompt, key_status.api_key)
            elif function_type == "image":
                image_path = kwargs.get('image_path')
                if not image_path:
                    return "[Error: image_path required for image function]"
                result = _call_gemini_image(image_path, prompt, key_status.api_key)
            # ... other function types
            
            # Mark success and return result
            gemini_key_manager.mark_key_success(key_status.key_id)
            return result
            
        except Exception as e:
            last_error = e
            logger.warning(f"Gemini call failed with key {key_status.key_id[:12]}...: {e}")
            
            # Check if this is a rate limiting error
            is_rate_limited, reset_time = gemini_key_manager.is_rate_limit_error(e)
            
            if is_rate_limited:
                gemini_key_manager.mark_key_rate_limited(key_status.key_id, reset_time)
            else:
                gemini_key_manager.mark_key_failed(key_status.key_id, e)
            
            # Short delay before retry
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
    
    return f"[All Gemini retry attempts failed: {last_error}]"
```

**Section sources**
- [core/llm.py](file://core/llm.py#L226-L292)

## Security Best Practices
The system implements several security best practices for API key management and authentication.

### Key Management Security Features
- **Environment Variable Priority**: API keys can be securely stored in environment variables
- **Multiple Key Support**: Reduces risk of single point of failure
- **Automatic Key Rotation**: Distributes API calls across multiple keys
- **Rate Limit Detection**: Identifies and handles rate-limited keys
- **Failure Tracking**: Monitors key performance and disables failing keys

### Secure Configuration Patterns
The configuration system follows security best practices:

1. **Separation of Secrets**: API keys can be stored in environment variables, separate from code
2. **Default Fallbacks**: Hardcoded fallback keys for development, but environment variables take precedence in production
3. **Key Prioritization**: Configuration file keys have higher priority than environment variables
4. **Thread Safety**: Key manager uses locks to prevent race conditions

**Section sources**
- [core/llm.py](file://core/llm.py#L46-L84)
- [core/config.py](file://core/config.py#L29-L129)

## Troubleshooting Guide
This section provides guidance for diagnosing and resolving common issues with LLM provider configuration.

### Common Authentication Failures
**Issue**: "Authentication failed" or "Invalid API key" errors
**Solutions**:
1. Verify API key format and validity
2. Check environment variable names (e.g., `A4F_API_KEY`, `ZUKIJOURNEY_API_KEY`)
3. Ensure no whitespace in API keys
4. Test key validity with provider's API directly

### Invalid Key Issues
**Issue**: Provider returns "Invalid key" response
**Solutions**:
1. Regenerate API key from provider dashboard
2. Update key in environment variable or config.json
3. Restart application to reload configuration
4. Check key expiration if applicable

### Network Connectivity Issues
**Issue**: "Connection timeout" or "Network error" messages
**Solutions**:
1. Verify internet connectivity
2. Check provider API status page
3. Test with different network connection
4. Increase timeout settings if needed
5. Verify base URL is correct

### Key Exhaustion Problems
**Issue**: "All Gemini API keys exhausted" error
**Solutions**:
1. Add more API keys to config.json or environment variables
2. Check rate limits with provider
3. Implement longer cooldown periods
4. Monitor key usage statistics with `get_gemini_key_statistics()`

### Debugging Tools
The system includes several debugging functions:

```python
def test_all_providers():
    """Test all LLM providers and enhanced Gemini fallbacks with a simple prompt."""
    prompt = "What is the capital of France?"
    print("Testing Zuki:")
    print(call_zuki(prompt))
    print("\nTesting ElectronHub:")
    print(call_electronhub(prompt))
    # ... other providers
    print("\nTesting Enhanced Gemini:")
    print(call_gemini(prompt))

def test_gemini_enhanced():
    """Test the enhanced Gemini system with multiple API keys."""
    # Test key statistics
    stats = get_gemini_key_statistics()
    print(f"Total keys: {stats['total_keys']}")
    print(f"Available keys: {stats['available_keys']}")
```

**Section sources**
- [core/llm.py](file://core/llm.py#L587-L621)
- [tests/simple_gemini_test.py](file://tests/simple_gemini_test.py#L0-L47)

**Referenced Files in This Document**   
- [core/llm.py](file://core/llm.py#L46-L658)
- [core/config.py](file://core/config.py#L29-L129)
- [core/config.json](file://core/config.json#L55-L193)