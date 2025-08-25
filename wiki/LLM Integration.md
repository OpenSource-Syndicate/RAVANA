# LLM Integration



## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Prompt Engineering Strategies](#prompt-engineering-strategies)
7. [Provider Configuration and Authentication](#provider-configuration-and-authentication)
8. [Fallback Mechanisms and Error Handling](#fallback-mechanisms-and-error-handling)
9. [Performance and Cost Management](#performance-and-cost-management)
10. [Security Considerations](#security-considerations)

## Introduction
This document provides a comprehensive overview of the LLM integration system within the RAVANA repository. It details the architecture, supported providers, configuration mechanisms, and advanced features such as prompt engineering, fallback strategies, and security considerations. The system is designed to support multiple LLM providers with robust error handling, rate limiting, and performance optimization.

## Project Structure
The project is organized into several key directories:
- **core**: Contains core functionality including LLM integration, configuration, and state management
- **modules**: Houses specialized modules for decision-making, self-reflection, and experimentation
- **prompts**: Stores JSON templates for various use cases
- **services**: Provides data, knowledge, and memory services
- **database**: Manages database operations and models

The LLM integration is primarily centered in the `core` directory, with supporting components in `modules` and configuration in `prompts`.

```mermaid
graph TD
A[LLM Integration System] --> B[core/llm.py]
A --> C[core/config.json]
A --> D[prompts/]
A --> E[core/prompt_manager.py]
B --> F[Provider Interfaces]
B --> G[Fallback Mechanisms]
B --> H[Error Handling]
C --> I[API Keys]
C --> J[Rate Limiting]
D --> K[Decision Making]
D --> L[Experimentation]
D --> M[Self-Reflection]
```

**Diagram sources**
- [llm.py](file://core/llm.py#L0-L1636)
- [config.json](file://core/config.json#L0-L193)
- [prompts/decision_making.json](file://prompts/decision_making.json#L0-L12)

**Section sources**
- [llm.py](file://core/llm.py#L0-L1636)
- [config.json](file://core/config.json#L0-L193)

## Core Components
The core components of the LLM integration system include:
- **LLM Interface**: Unified interface for calling different LLM providers
- **Provider Adapters**: Specific implementations for Zuki, ElectronHub, Zanity, A4F, and Gemini
- **Prompt Manager**: Handles prompt validation, enhancement, and template management
- **Key Manager**: Manages multiple API keys with rotation and rate limiting
- **Error Handler**: Comprehensive error handling with retry logic and fallback mechanisms

The system is designed with extensibility in mind, allowing for easy addition of new providers and prompt templates.

**Section sources**
- [llm.py](file://core/llm.py#L0-L1636)
- [prompt_manager.py](file://core/prompt_manager.py#L0-L100)

## Architecture Overview
The LLM integration system follows a modular architecture with clear separation of concerns. The core component is the `call_llm` function, which serves as the unified interface for all LLM interactions. This function routes requests to the appropriate provider based on availability and configuration.

```mermaid
graph TD
A[Application] --> B[call_llm]
B --> C{Preferred Provider?}
C --> |Yes| D[Specific Provider]
C --> |No| E[Provider Chain]
D --> F[API Request]
E --> F
F --> G[Response]
G --> H[Error Handling]
H --> I[Retry Logic]
I --> J[Fallback to Gemini]
J --> K[Final Response]
K --> A
```

**Diagram sources**
- [llm.py](file://core/llm.py#L0-L1636)

## Detailed Component Analysis

### LLM Interface and Request Handling
The LLM interface provides a unified method for interacting with multiple providers through the `call_llm` function. This function accepts a prompt and optional model parameter, then attempts to process the request through a chain of providers.

```mermaid
sequenceDiagram
participant App as Application
participant LLM as call_llm
participant Providers as Provider Chain
participant Gemini as Gemini Fallback
App->>LLM : call_llm(prompt, model)
LLM->>Providers : Try Zuki, ElectronHub, Zanity, A4F
alt Provider Success
Providers-->>LLM : Response
LLM-->>App : Return Response
else All Providers Fail
LLM->>Gemini : call_gemini_with_fallback
Gemini-->>LLM : Response or Error
LLM-->>App : Return Response
end
```

**Diagram sources**
- [llm.py](file://core/llm.py#L0-L1636)

**Section sources**
- [llm.py](file://core/llm.py#L0-L1636)

### Provider Configuration and Authentication
The system supports multiple LLM providers with configuration managed through `config.json`. Each provider has its own API key, base URL, and available models.

```mermaid
classDiagram
class LLMProvider {
+str name
+str api_key
+str base_url
+List[str] models
+call(prompt, model)
}
class ZukiProvider {
+call_zuki(prompt, model)
}
class ElectronHubProvider {
+call_electronhub(prompt, model)
}
class ZanityProvider {
+call_zanity(prompt, model)
}
class A4FProvider {
+call_a4f(prompt)
}
class GeminiProvider {
+call_gemini(prompt)
+call_gemini_with_fallback()
}
LLMProvider <|-- ZukiProvider
LLMProvider <|-- ElectronHubProvider
LLMProvider <|-- ZanityProvider
LLMProvider <|-- A4FProvider
LLMProvider <|-- GeminiProvider
```

**Diagram sources**
- [llm.py](file://core/llm.py#L0-L1636)
- [config.json](file://core/config.json#L0-L193)

**Section sources**
- [llm.py](file://core/llm.py#L0-L1636)
- [config.json](file://core/config.json#L0-L193)

## Prompt Engineering Strategies
The system employs structured prompt templates for different use cases, stored in the `prompts` directory. These templates follow a consistent format with role definition, context, task instructions, reasoning framework, output requirements, and safety constraints.

### Decision-Making Prompts
The decision-making prompt template guides the LLM through a structured analysis process:

```json
{
  "name": "decision_making",
  "template": "\n[ROLE DEFINITION]\nYou are {agent_name}, an autonomous AI agent making decisions to achieve your objectives with enhanced reasoning capabilities.\n\n[CONTEXT]\nCurrent situation: {current_situation}\nActive goals: {active_goals}\nCurrent hypotheses: {current_hypotheses}\nEmotional state: {current_mood}\nAvailable actions: {action_list}\n\n[TASK INSTRUCTIONS]\nMake an optimal decision by following this structured approach:\n1. Analyze the situation and identify key factors\n2. Evaluate alignment with goals and hypotheses\n3. Consider multiple approaches and their implications\n4. Assess risks and potential outcomes\n5. Select the optimal action with clear justification\n\n[REASONING FRAMEWORK]\nApply systematic analysis to your decision-making:\n1. Decompose the problem into manageable components\n2. Evaluate each option against success criteria\n3. Consider short-term and long-term consequences\n4. Account for uncertainty and incomplete information\n5. Validate reasoning against logical consistency\n\n[OUTPUT REQUIREMENTS]\nProvide a JSON-formatted response with these fields:\n- analysis: Detailed situation analysis with key factors identified\n- reasoning: Step-by-step reasoning leading to decision\n- confidence: Numerical confidence score (0.0-1.0)\n- risk_assessment: Potential risks and mitigation strategies\n- action: Selected action with parameters\n\n[SAFETY CONSTRAINTS]\n- Ensure actions align with ethical principles\n- Avoid decisions with catastrophic risk potential\n- Consider impact on system stability and reliability\n- Validate against established safety protocols\n"
}
```

**Section sources**
- [prompts/decision_making.json](file://prompts/decision_making.json#L0-L12)

### Experimentation Prompts
The experimentation prompt template follows scientific method principles:

```json
{
  "name": "experimentation",
  "template": "\n[ROLE DEFINITION]\nYou are {agent_name}, a scientific AI agent designing and conducting rigorous experiments to test hypotheses.\n\n[CONTEXT]\nExperiment objective: {experiment_objective}\nRelated knowledge: {relevant_theory}\nAvailable resources: {resource_constraints}\nSafety protocols: {safety_requirements}\n\n[TASK INSTRUCTIONS]\nDesign a comprehensive experiment following these steps:\n1. Formulate a clear hypothesis to test\n2. Design rigorous experimental methodology\n3. Identify required materials and setup\n4. Specify measurement and data collection methods\n5. Define success criteria and validation methods\n6. Analyze potential failure modes and mitigations\n\n[REASONING FRAMEWORK]\nApply scientific method principles:\n1. Ensure hypothesis is falsifiable and specific\n2. Design controls to isolate variables\n3. Plan for replication and verification\n4. Consider alternative explanations\n5. Account for measurement uncertainty\n6. Plan for iterative refinement\n\n[OUTPUT REQUIREMENTS]\nProvide a complete experimental design with:\n- Experiment design: Complete experimental procedure\n- Expected outcomes: Predicted results with rationale\n- Resource requirements: List of needed materials and tools\n- Safety considerations: Risk assessment and safety measures\n- Validation approach: Method for verifying results\n- Failure analysis: Potential failure modes and mitigations\n\n[SAFETY CONSTRAINTS]\n- Adhere to all safety protocols and guidelines\n- Identify and mitigate potential hazards\n- Ensure environmental and ethical compliance\n- Plan for safe termination of problematic experiments\n"
}
```

**Section sources**
- [prompts/experimentation.json](file://prompts/experimentation.json#L0-L12)

### Self-Reflection Prompts
The self-reflection prompt template enables continuous improvement:

```json
{
  "name": "self_reflection",
  "template": "\n[ROLE DEFINITION]\nYou are {agent_name}, an advanced AI agent engaged in continuous self-improvement through structured reflection.\n\n[CONTEXT]\nCurrent situation: {task_summary}\nOutcome: {outcome}\nEmotional state: {current_mood}\nRelevant memories: {related_memories}\n\n[TASK INSTRUCTIONS]\nConduct a thorough self-analysis of your recent task performance using the following questions:\n1. What aspects of your approach were most effective?\n2. Where did you encounter difficulties or failures?\n3. What unexpected insights or discoveries emerged?\n4. What knowledge gaps or skill areas need development?\n5. How can you modify your approach for better results?\n\n[REASONING FRAMEWORK]\nApproach this reflection systematically:\n1. Analyze the task execution and outcomes\n2. Identify patterns in successes and failures\n3. Connect findings to broader learning principles\n4. Generate actionable improvement suggestions\n5. Prioritize recommendations by impact and feasibility\n\n[OUTPUT REQUIREMENTS]\nProvide a detailed, structured response with:\n- Specific examples and evidence\n- Confidence scores for each insight (0.0-1.0)\n- Actionability ratings for improvement suggestions\n- Connections to related memories and experiences\n- Mood-aware reflection depth adjustment\n\n[SAFETY CONSTRAINTS]\n- Be honest and critical in your assessment\n- Focus on learning opportunities rather than justifications\n- Avoid overconfidence in uncertain areas\n- Consider ethical implications of self-modifications\n"
}
```

**Section sources**
- [prompts/self_reflection.json](file://prompts/self_reflection.json#L0-L12)

## Provider Configuration and Authentication
The system supports multiple LLM providers with configuration managed in `config.json`. Each provider has its own API key, base URL, and available models.

### Configuration Structure
The configuration file contains settings for all providers:

```json
{
  "zuki": {
    "api_key": "985160dfa1fd499fd12af708d16552e37a8c6f77cbfb50ae400e3ff33fbd791bc7b3b82625379a1f5ca7568f1ee04eb81a0f8c06f0ba6c276d3dddfe13e9c18d",
    "base_url": "https://api.zukijourney.com/v1",
    "models": [
      "gpt-4o:online",
      "gpt-4o",
      "deepseek-chat",
      "deepseek-reasoner"
    ]
  },
  "electronhub": {
    "api_key": "ek-sVvxMYfdFQ0Kl6Aj2tmV7b8n5v0Y0sDHVsOUZWyx2vbs0AbuAc",
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
  "gemini": {
    "api_keys": [
      {
        "id": "gemini_key_1",
        "key": "AIzaSyBW-aVU-x7JCjBJVVKjPGUacups0-GBHvQ",
        "priority": 1
      },
      {
        "id": "gemini_key_2",
        "key": "AIzaSyBW-aVU-x7JCjBJVVKjPGUacups0-GBHvQ",
        "priority": 2
      }
    ],
    "rate_limit": {
      "requests_per_minute": 60,
      "cooldown_period": 300,
      "max_retries": 3,
      "backoff_factor": 2.0
    }
  }
}
```

### Authentication Mechanisms
Each provider uses standard API key authentication via the Authorization header:

```python
headers = {"Authorization": f"Bearer {api_key}"}
```

The system also supports environment variables for API keys, providing flexibility in deployment:

```python
api_key = os.getenv(f"GEMINI_API_KEY_{i}")
```

**Section sources**
- [config.json](file://core/config.json#L0-L193)
- [llm.py](file://core/llm.py#L0-L1636)

## Fallback Mechanisms and Error Handling
The system implements comprehensive fallback mechanisms and error handling to ensure reliability.

### Gemini Key Management
The `GeminiKeyManager` class manages multiple API keys with rotation and rate limiting:

```mermaid
flowchart TD
A[Request] --> B{Get Available Key}
B --> C[No Keys Available?]
C --> |Yes| D[Return Error]
C --> |No| E[Use Key]
E --> F{Request Success?}
F --> |Yes| G[Mark Success]
F --> |No| H{Rate Limited?}
H --> |Yes| I[Mark Rate Limited]
H --> |No| J[Mark Failed]
I --> K[Wait for Reset]
J --> L{Consecutive Failures >= 5?}
L --> |Yes| M[Mark Unavailable]
L --> |No| N[Continue]
```

**Diagram sources**
- [llm.py](file://core/llm.py#L0-L1636)

**Section sources**
- [llm.py](file://core/llm.py#L0-L1636)

### Unified Fallback Strategy
The `call_llm` function implements a provider chain with fallback to Gemini:

```python
def call_llm(prompt, preferred_provider=None, model=None):
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

**Section sources**
- [llm.py](file://core/llm.py#L0-L1636)

## Performance and Cost Management
The system includes several features to manage performance and cost effectively.

### Rate Limiting and Retry Logic
The system implements exponential backoff with jitter to prevent overwhelming providers:

```python
def safe_call_llm(prompt: str, timeout: int = 30, retries: int = 3, backoff_factor: float = 1.0, **kwargs) -> str:
    for attempt in range(1, retries + 1):
        if attempt > 1:
            jitter = random.uniform(0.1, 0.5) * backoff_factor
            wait = backoff_factor * (2 ** (attempt - 1)) + jitter
            time.sleep(wait)
        # Make API call
        # Handle response
    return "[LLM Error: Unable to generate response. Please try again later.]"
```

### Response Parsing and Validation
The system includes robust response parsing with JSON extraction and error recovery:

```python
def extract_decision(raw_response: str) -> dict:
    block = _extract_json_block(raw_response)
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        fixed_block = _fix_truncated_json(block)
        try:
            data = json.loads(fixed_block)
        except json.JSONDecodeError:
            return {
                "raw_response": raw_response,
                "error": f"JSON decode error: {je}",
                "analysis": "Failed to parse decision",
                "plan": [],
                "action": "log_message",
                "params": {"message": f"Failed to parse decision: {raw_response[:200]}..."}
            }
    return {
        "raw_response": raw_response,
        "analysis": data.get("analysis", "No analysis provided"),
        "plan": data.get("plan", []),
        "action": data.get("action", "log_message"),
        "params": data.get("params", {"message": "No action specified"}),
        "confidence": data.get("confidence", 0.5),
        "reasoning": data.get("reasoning", ""),
    }
```

**Section sources**
- [llm.py](file://core/llm.py#L0-L1636)

## Security Considerations
The system addresses several security aspects in LLM interactions.

### SSL Certificate Handling
The system implements robust SSL certificate management:

```python
def create_ssl_context():
    try:
        ssl_context = ssl.create_default_context()
        return ssl_context
    except Exception as e:
        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            return ssl_context
        except Exception as fallback_error:
            ssl_context = ssl._create_unverified_context()
            return ssl_context
```

### Prompt Injection Risks
The system mitigates prompt injection risks through structured templates and output validation. The prompt templates include clear role definitions and safety constraints that help maintain control over the LLM's behavior.

### Response Validation
The system validates responses to detect lazy or generic responses:

```python
def is_lazy_llm_response(text):
    lazy_phrases = [
        "as an ai language model",
        "i'm unable to",
        "i cannot",
        "i apologize",
        "here is a function",
        "here's an example",
        "please see below",
        "unfortunately",
        "i do not have",
        "i don't have",
        "i am not able",
        "i am unable",
        "i suggest",
        "you can use",
        "to do this, you can",
        "this is a placeholder",
        "[insert",
        "[code block]",
        "[python code]",
        "[insert code here]",
        "[insert explanation here]",
        "[unsupported code language",
        "[python execution error",
        "[shell execution error",
        "[gemini",
        "[error",
        "[exception",
        "[output",
        "[result",
        "[python code result]:\n[python execution error",
    ]
    if not text:
        return True
    text_lower = str(text).strip().lower()
    if not text_lower or len(text_lower) < 10:
        return True
    for phrase in lazy_phrases:
        if phrase in text_lower:
            return True
    if text_lower in ("``", "```"):
        return True
    return False
```

**Section sources**
- [llm.py](file://core/llm.py#L0-L1636)

**Referenced Files in This Document**   
- [llm.py](file://core/llm.py#L0-L1636)
- [config.json](file://core/config.json#L0-L193)
- [decision_making.json](file://prompts/decision_making.json#L0-L12)
- [experimentation.json](file://prompts/experimentation.json#L0-L12)
- [self_reflection.json](file://prompts/self_reflection.json#L0-L12)
- [prompt_manager.py](file://core/prompt_manager.py#L0-L100)