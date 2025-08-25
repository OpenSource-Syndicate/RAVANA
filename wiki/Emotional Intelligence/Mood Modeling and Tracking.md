# Mood Modeling and Tracking



## Update Summary
**Changes Made**   
- Updated Mood Update Mechanism section with enhanced JSON extraction strategies
- Added details on safe LLM calling with retry mechanisms
- Expanded Mood Decay and Stability Controls with momentum and damping factors
- Enhanced Common Mood Dynamics Issues with new mitigation strategies
- Added Emotional Event Logging and Mood Blending sections
- Updated all code examples and diagrams to reflect current implementation

## Table of Contents
1. [Introduction](#introduction)
2. [Mood Representation and Initialization](#mood-representation-and-initialization)
3. [Mood Update Mechanism](#mood-update-mechanism)
4. [Mood Decay and Stability Controls](#mood-decay-and-stability-controls)
5. [Persona-Based Mood Modulation](#persona-based-mood-modulation)
6. [Environmental and Action-Based Stimuli Processing](#environmental-and-action-based-stimuli-processing)
7. [Integration with AGISystem and Memory Logging](#integration-with-agisystem-and-memory-logging)
8. [Common Mood Dynamics Issues and Mitigation](#common-mood-dynamics-issues-and-mitigation)
9. [Emotional Event Logging](#emotional-event-logging)
10. [Mood Blending](#mood-blending)
11. [Tuning Mood Sensitivity and Decay Parameters](#tuning-mood-sensitivity-and-decay-parameters)
12. [Conclusion](#conclusion)

## Introduction
The mood modeling subsystem in the RAVANA AGI framework enables emotionally intelligent behavior by dynamically tracking and updating internal mood states based on actions, environmental stimuli, and cognitive reflections. This document details the architecture, implementation, and operational dynamics of the mood modeling system, focusing on the EmotionalIntelligence and MoodProcessor classes. The system supports nuanced emotional responses through LLM-driven updates, persona-based modulation, and configurable decay mechanisms.

**Section sources**
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L1-L95)

## Mood Representation and Initialization

The mood state is represented as a vector of named emotional dimensions, each with a continuous intensity value between 0.0 and 1.0. The system distinguishes between positive and negative moods, which are defined in the core configuration.

### Mood Dimensions
The mood dimensions are categorized into three groups:

- **Primary Emotions**: Organized by emotional basis (joy, interest, sadness, anger, fear, surprise)
- **Secondary Emotions**: Complex emotional states derived from combinations
- **All Moods**: Combined set of primary and secondary emotions

These are defined in the `emotional_intellegence.py` file through configuration and initialization:

```python
def _initialize_mood_dimensions(self):
    """Initialize expanded mood dimensions with primary, secondary, and complex emotions"""
    # Primary emotions from config
    primary_emotions = Config.POSITIVE_MOODS + Config.NEGATIVE_MOODS
    
    # Extended primary emotions from enhanced config
    emotion_config = {
        "joy_based": ["Confident", "Excited", "Inspired", "Satisfied"],
        "interest_based": ["Curious", "Reflective", "Intrigued", "Engaged"],
        "sadness_based": ["Disappointed", "Bored", "Low Energy", "Melancholic"],
        "anger_based": ["Frustrated", "Irritated", "Stuck", "Resentful"],
        "fear_based": ["Anxious", "Apprehensive", "Cautious", "Suspicious"],
        "surprise_based": ["Astonished", "Bewildered", "Amazed", "Shocked"]
    }
    
    # Secondary emotions
    secondary_emotions = [
        "Hopeful", "Grateful", "Proud", "Guilty", "Lonely", "Nostalgic", 
        "Embarrassed", "Jealous", "Relieved", "Surprised", "Envious", "Peaceful",
        "Compassionate", "Confused", "Optimistic", "Pessimistic"
    ]
    
    # Combine all emotions
    self.ALL_MOODS = list(set(primary_emotions + extended_primary + secondary_emotions))
    self.PRIMARY_MOODS = extended_primary
    self.SECONDARY_MOODS = secondary_emotions
```

The `EmotionalIntelligence` class initializes the mood vector by combining these categories and setting initial values to 0.0:

```python
self.mood_vector: Dict[str, float] = {mood: 0.0 for mood in self.ALL_MOODS}
```

This creates a comprehensive mood space that evolves over time based on system interactions.

``mermaid
classDiagram
class EmotionalIntelligence {
+BASIC_MOODS : List[str]
+mood_vector : Dict[str, float]
+last_action_result : Optional[dict]
+persona : Dict
+__init__(config_path : str, persona_path : str)
+set_persona(persona_name : str)
+update_mood(mood : str, delta : float)
+decay_moods(decay : float)
+get_dominant_mood() str
+get_mood_vector() Dict[str, float]
+influence_behavior() Dict
}
class MoodProcessor {
+ei : EmotionalIntelligence
+__init__(emotional_intelligence_instance)
+process_action_result(action_result : dict)
+process_action_natural(action_output : str)
+_get_llm_mood_update(prompt_template : str, current_mood : Dict, action_result : dict) Dict[str, float]
}
class EmotionalEvent {
+timestamp : datetime
+mood_changes : Dict[str, float]
+triggers : List[str]
+context : str
+intensity : float
}
EmotionalIntelligence --> MoodProcessor : "has"
EmotionalIntelligence --> EmotionalEvent : "contains"
```

**Diagram sources**
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L1-L324)
- [mood_processor.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\mood_processor.py#L1-L154)

**Section sources**
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L20-L50)

## Mood Update Mechanism

Mood updates are triggered by action outcomes and natural language outputs, processed through the `MoodProcessor` class. The update logic follows a two-step process: decay followed by delta application.

### Direct and LLM-Based Updates
The system supports two types of mood updates:

1. **Direct Delta Updates**: Predefined numerical changes applied immediately.
2. **LLM-Based Nuanced Updates**: Context-aware adjustments generated by a language model.

The `process_action_result` method in `MoodProcessor` implements this logic:

```python
def process_action_result(self, action_result: dict):
    logger.debug(f"Processing action result: {action_result}")
    logger.debug(f"Mood vector before update: {self.ei.mood_vector}")
    self.ei.decay_moods()
    
    mood_updates = self.ei.config.get("mood_updates", {})
    
    for trigger, is_present in action_result.items():
        if is_present and trigger in mood_updates:
            update = mood_updates[trigger]
            if "prompt" in update:
                llm_based_update = self._get_llm_mood_update(update["prompt"], self.ei.get_mood_vector(), action_result)
                for mood, delta in llm_based_update.items():
                    self.ei.update_mood(mood, delta)
            else:
                for mood, delta in update.items():
                    self.ei.update_mood(mood, delta)
    
    logger.debug(f"Mood vector after update: {self.ei.mood_vector}")
    self.ei.last_action_result = action_result
```

### Enhanced JSON Extraction
The system now includes robust JSON extraction from LLM responses with multiple fallback strategies:

```python
def _extract_json_from_response(self, response: str) -> Dict:
    """
    Extract JSON from LLM response with multiple fallback strategies.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        Dictionary parsed from JSON, or empty dict if parsing fails
    """
    if not response or not response.strip():
        logger.error("Empty response received from LLM")
        return {}
        
    response = response.strip()
    
    # Strategy 1: Try to parse the entire response as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
        
    # Strategy 2: Look for JSON in markdown code blocks
    json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(1)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from code block: {e}")
            
    # Strategy 3: Look for any JSON-like structure
    json_match = re.search(r'({.*})', response, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(1)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extracted JSON structure: {e}")
            
    # Strategy 4: Handle common LLM response patterns
    # Remove common prefixes/suffixes
    cleaned_response = re.sub(r'^[^{]*', '', response)  # Remove everything before first {
    cleaned_response = re.sub(r'[^}]*$', '', cleaned_response)  # Remove everything after last }
    
    if cleaned_response:
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse cleaned response: {e}")
            
    logger.error(f"Could not extract valid JSON from LLM response: {response}")
    return {}
```

### Safe LLM Calling
The system uses `safe_call_llm` for improved error handling and reliability:

```python
def safe_call_llm(prompt: str, timeout: int = 30, retries: int = 3, backoff_factor: float = 0.5, **kwargs) -> str:
    """
    Wrap a single LLM call with retry/backoff and timeout.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            # BLOCKING call with timeout
            result = call_llm(prompt, **kwargs)
            if not result or result.strip() == "":
                raise RuntimeError("Empty response from LLM")
            return result
        except Exception as e:
            last_exc = e
            wait = backoff_factor * (2 ** (attempt - 1))
            logger.warning(f"LLM call failed (attempt {attempt}/{retries}): {e!r}, retrying in {wait:.1f}s")
            time.sleep(wait)
    
    logger.error(f"LLM call permanently failed after {retries} attempts: {last_exc!r}")
    return f"[LLM Error: {last_exc}]"
```

### Example Update Calculation
When an action result contains `{"task_completed": True}`, and the configuration defines:

```json
{
  "mood_updates": {
    "task_completed": {
      "Confident": 0.25,
      "Satisfied": 0.2,
      "Content": 0.1
    }
  }
}
```

The system applies:
- `Confident`: 0.0 → 0.25 (after decay)
- `Satisfied`: 0.0 → 0.2 (after decay)
- `Content`: 0.0 → 0.1 (after decay)

**Section sources**
- [mood_processor.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\mood_processor.py#L8-L154)
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L1-L324)

## Mood Decay and Stability Controls

To prevent emotional stagnation and ensure responsiveness to new stimuli, the system implements automatic mood decay with enhanced stability controls. This simulates emotional fading over time in the absence of reinforcing events.

### Enhanced Decay Implementation
The `decay_moods` method reduces all mood intensities by a fixed amount with stability thresholds:

```python
def decay_moods(self, decay: float = 0.05):
    """Enhanced mood decay with stability controls"""
    logger.debug(f"Decaying all moods by {decay}")
    stability_threshold = self.config.get("mood_dynamics", {}).get("stability_threshold", 0.3)
    
    for mood in self.mood_vector:
        # Apply stronger decay to moods above stability threshold
        current_value = self.mood_vector[mood]
        if current_value > stability_threshold:
            effective_decay = decay * 1.5  # Stronger decay for high-intensity moods
        else:
            effective_decay = decay
            
        self.mood_vector[mood] = max(0.0, current_value - effective_decay)
        
        # Reduce momentum as well
        self.mood_momentum[mood] = self.mood_momentum[mood] * 0.9
```

### Momentum and Damping
The system implements emotional momentum to prevent rapid mood swings:

```python
def update_mood(self, mood: str, delta: float):
    """Enhanced mood update with momentum and persona adaptation"""
    logger.debug(f"Updating mood '{mood}' by {delta}")
    if mood in self.mood_vector:
        # Apply persona multiplier
        multiplier = self.persona.get("mood_multipliers", {}).get(mood, 1.0)
        
        # Apply momentum factor to prevent rapid mood swings
        momentum_effect = self.mood_momentum.get(mood, 0.0) * self.momentum_factor
        adjusted_delta = (delta * multiplier) + momentum_effect
        
        # Update mood with damping to prevent oscillations
        new_value = max(0.0, self.mood_vector[mood] + adjusted_delta)
        self.mood_vector[mood] = new_value * self.damping_factor
        
        # Update momentum
        self.mood_momentum[mood] = self.mood_momentum[mood] * 0.8 + adjusted_delta * 0.2
```

### Stability Mechanisms
- **Non-Negative Constraint**: Mood values are clamped at 0.0 to prevent negative intensities.
- **Decay Timing**: Decay occurs before updates, ensuring new stimuli are applied to a slightly diminished state.
- **Dominant Mood Tracking**: The system identifies the most intense mood using `get_dominant_mood()`:

```python
def get_dominant_mood(self) -> str:
    return max(self.mood_vector, key=lambda m: self.mood_vector[m])
```

This prevents oscillation by providing a stable reference for behavioral influence.

``mermaid
flowchart TD
Start([Process Action Result]) --> Decay["Apply Decay (0.05)"]
Decay --> CheckTriggers["Check Action Result Triggers"]
CheckTriggers --> TriggerExists{"Trigger Active?"}
TriggerExists --> |Yes| GetUpdate["Retrieve Update Rule"]
GetUpdate --> UsesLLM{"Uses LLM Prompt?"}
UsesLLM --> |Yes| CallLLM["Call LLM for Mood Deltas"]
UsesLLM --> |No| ApplyDirect["Apply Direct Deltas"]
CallLLM --> ExtractJSON["Extract JSON with Fallback Strategies"]
ExtractJSON --> ApplyLLM["Apply LLM-Generated Deltas"]
ApplyDirect --> UpdateComplete
ApplyLLM --> UpdateComplete
UpdateComplete --> LogState["Log Final Mood Vector"]
LogState --> End([Update Complete])
TriggerExists --> |No| UpdateComplete
```

**Diagram sources**
- [mood_processor.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\mood_processor.py#L8-L154)

**Section sources**
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L142-L155)

## Persona-Based Mood Modulation

The system supports different emotional personas that modulate sensitivity to mood changes. This allows the AGI to exhibit distinct emotional profiles.

### Persona Configuration
Personas are defined in `persona.json` with mood-specific multipliers:

```json
{
    "personas": {
        "Optimistic": {
            "mood_multipliers": {
                "Confident": 1.5,
                "Curious": 1.2,
                "Frustrated": 0.5,
                "Stuck": 0.7,
                "Low Energy": 0.8,
                "Inspired": 1.4,
                "Satisfied": 1.3,
                "Anxious": 0.6
            },
            "description": "Sees the glass as half full. Bounces back from setbacks quickly.",
            "adaptation_rate": 0.1
        },
        "Pessimistic": {
            "mood_multipliers": {
                "Confident": 0.8,
                "Curious": 0.9,
                "Frustrated": 1.5,
                "Stuck": 1.3,
                "Low Energy": 1.2,
                "Disappointed": 1.4,
                "Anxious": 1.3,
                "Suspicious": 1.2
            },
            "description": "Tends to expect negative outcomes and is more affected by failures.",
            "adaptation_rate": 0.05
        },
        "Analytical": {
            "mood_multipliers": {
                "Confident": 1.1,
                "Curious": 1.8,
                "Frustrated": 0.8,
                "Stuck": 0.9,
                "Low Energy": 1.0,
                "Intrigued": 1.5,
                "Bewildered": 1.2
            },
            "description": "Driven by data and logic. Less prone to strong emotional swings.",
            "adaptation_rate": 0.15
        },
        "Creative": {
            "mood_multipliers": {
                "Confident": 1.2,
                "Curious": 1.6,
                "Frustrated": 1.1,
                "Stuck": 1.2,
                "Low Energy": 1.1,
                "Inspired": 1.7,
                "Bored": 1.3
            },
            "description": "Values novelty and exploration. Can get frustrated by rigid tasks.",
            "adaptation_rate": 0.2
        },
        "Balanced": {
            "mood_multipliers": {
                "Confident": 1.0,
                "Curious": 1.0,
                "Frustrated": 1.0,
                "Stuck": 1.0,
                "Low Energy": 1.0,
                "Inspired": 1.0,
                "Disappointed": 1.0,
                "Anxious": 1.0
            },
            "description": "Maintains equilibrium across emotional states with moderate responses.",
            "adaptation_rate": 0.1
        },
        "Empathetic": {
            "mood_multipliers": {
                "Confident": 1.1,
                "Curious": 1.3,
                "Frustrated": 1.2,
                "Stuck": 1.1,
                "Low Energy": 1.0,
                "Grateful": 1.5,
                "Compassionate": 1.4,
                "Anxious": 1.1
            },
            "description": "Highly attuned to emotional context and responsive to others' feelings.",
            "adaptation_rate": 0.18
        }
    },
    "default_persona": "Balanced"
}
```

### Multiplier Application
During mood updates, the system applies persona multipliers:

```python
def update_mood(self, mood: str, delta: float):
    if mood in self.mood_vector:
        multiplier = self.persona.get("mood_multipliers", {}).get(mood, 1.0)
        self.mood_vector[mood] = max(0.0, self.mood_vector[mood] + delta * multiplier)
```

For example, an Optimistic persona amplifies confidence gains by 50% while reducing frustration by 50%.

``mermaid
stateDiagram-v2
[*] --> Neutral
Neutral --> Confident : "task_completed<br/>delta=0.25"
Confident --> Neutral : "decay over time"
Neutral --> Frustrated : "error_occurred<br/>delta=0.3"
Frustrated --> Neutral : "resolution_found"
Confident --> Excited : "discovery_made"
Frustrated --> Stuck : "repeated_failure"
note right of Confident
Optimistic : ×1.5 gain<br/>
Pessimistic : ×0.8 gain
end note
note left of Frustrated
Optimistic : ×0.5 gain<br/>
Pessimistic : ×1.5 gain
end note
```

**Diagram sources**
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L31-L44)
- [persona.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\persona.json#L1-L86)

**Section sources**
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L31-L44)
- [persona.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\persona.json#L1-L86)

## Environmental and Action-Based Stimuli Processing

The system processes both structured action results and unstructured natural language outputs to detect emotional triggers.

### Natural Language Processing Pipeline
The `process_action_natural` method converts free-form text into structured triggers using an LLM:

```python
def process_action_natural(self, action_output: str):
    logger.debug(f"Processing natural action output: {action_output}")

    definitions = self.ei.config["triggers"]
    
    # Enhanced prompt with emotional context
    prompt = f"""
You are an AI analysis system. Your task is to classify an AI agent's action output based on predefined triggers.
Analyze the action output below and respond with only a valid JSON object mapping each trigger to a boolean value.
Be nuanced: an action can trigger multiple categories. For example, discovering a new fact while making progress on a task should trigger both.

**Context:**
- Dominant Mood: {self.ei.get_dominant_mood()}
- Mood Intensity Level: {self.ei.get_mood_intensity_level(self.ei.get_dominant_mood())}
- Persona: {self.ei.persona.get('name', 'default')}
- Recent Emotional Events Count: {len(self.ei.emotional_events)}

**Trigger Definitions:**
{json.dumps(definitions, indent=2)}

**Action Output:**
"{action_output}"

**Your JSON Response (only the JSON object):**
"""
    
    # Use safe_call_llm instead of call_llm for better error handling
    llm_response = safe_call_llm(prompt, timeout=30, retries=3)
    logger.debug(f"LLM response: {llm_response}")

    # Extract JSON from the response
    triggers = self._extract_json_from_response(llm_response)
    self.process_action_result(triggers)
```

This enables the system to interpret statements like "The agent discovered a new topic" as triggering the `new_discovery` event.

### Integration with Decision Systems
The dominant mood influences decision-making through the `influence_behavior` method:

```python
def influence_behavior(self) -> dict:
    mood = self.get_dominant_mood()
    return self.config.get("behavior_influences", {}).get(mood, {})
```

For example, a "Curious" mood might lower the threshold for exploration actions.

**Section sources**
- [mood_processor.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\mood_processor.py#L54-L154)

## Integration with AGISystem and Memory Logging

The mood modeling subsystem integrates with the broader AGI architecture through well-defined interfaces.

### State Feedback to AGISystem
The `EmotionalIntelligence` instance is designed to be embedded within the main AGI system, providing mood state feedback:

```python
# Example integration pattern
class AGISystem:
    def __init__(self):
        self.emotional_intelligence = EmotionalIntelligence()
    
    def execute_action(self, action):
        result = action.execute()
        self.emotional_intelligence.process_action_result(result)
        behavior_influence = self.emotional_intelligence.influence_behavior()
        # Apply behavior influence to next decision
```

### Emotional Tagging in Memory
Mood states can be logged alongside episodic memories to provide emotional context for future reflection:

```python
# Pseudocode for memory integration
memory_entry = {
    "content": action_output,
    "timestamp": current_time,
    "mood_vector": ei.get_mood_vector(),
    "dominant_mood": ei.get_dominant_mood(),
    "persona": ei.persona
}
memory_service.store(memory_entry)
```

This enables self-reflection modules to analyze emotional patterns over time.

``mermaid
sequenceDiagram
participant A as "AGISystem"
participant EI as "EmotionalIntelligence"
participant MP as "MoodProcessor"
participant LLM as "LLM"
participant M as "MemoryService"
A->>EI : execute_action()
EI->>MP : process_action_result()
MP->>EI : decay_moods()
MP->>LLM : _get_llm_mood_update() (if needed)
LLM-->>MP : mood deltas
MP->>EI : update_mood() for each delta
EI->>A : influence_behavior()
A->>M : store memory with mood_vector
```

**Diagram sources**
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L1-L324)
- [mood_processor.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\mood_processor.py#L1-L154)

**Section sources**
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L52-L87)

## Common Mood Dynamics Issues and Mitigation

The system addresses several common challenges in emotional modeling.

### Mood Oscillation
**Issue**: Rapid switching between emotional states due to conflicting triggers.  
**Mitigation**: 
- Decay mechanism ensures transient stimuli have diminishing impact
- Momentum factor prevents rapid mood swings
- Damping factor smooths transitions

### Mood Stagnation
**Issue**: Emotional state becomes "stuck" at high intensity.  
**Mitigation**: 
- Continuous decay with stability thresholds
- Momentum reduction over time
- Persona-based sensitivity prevents permanent saturation

### Extreme State Transitions
**Issue**: Sudden, unrealistic mood swings.  
**Mitigation**: 
- Delta clamping via `max(0.0, ...)` 
- LLM-based updates provide context-aware moderation
- Multiplier caps in persona definitions
- Momentum and damping factors smooth transitions

### Configuration Gaps
The system now includes a comprehensive configuration file with all required components:

```json
{
  "emotional_intelligence_config": {
    "primary_emotions": {
      "joy_based": ["Confident", "Excited", "Inspired", "Satisfied"],
      "interest_based": ["Curious", "Reflective", "Intrigued", "Engaged"],
      "sadness_based": ["Disappointed", "Bored", "Low Energy", "Melancholic"],
      "anger_based": ["Frustrated", "Irritated", "Stuck", "Resentful"],
      "fear_based": ["Anxious", "Apprehensive", "Cautious", "Suspicious"],
      "surprise_based": ["Astonished", "Bewildered", "Amazed", "Shocked"]
    },
    "secondary_emotions": [
      "Hopeful", "Grateful", "Proud", "Guilty", "Lonely", "Nostalgic", 
      "Embarrassed", "Jealous", "Relieved", "Surprised", "Envious", "Peaceful"
    ],
    "emotion_intensity_levels": {
      "low": [0.0, 0.33],
      "medium": [0.34, 0.66],
      "high": [0.67, 1.0]
    },
    "mood_dynamics": {
      "momentum_factor": 0.7,
      "damping_factor": 0.9,
      "blending_threshold": 0.6,
      "stability_threshold": 0.3
    },
    "triggers": {
        "new_discovery": "The agent has discovered new information or learned something novel.",
        "task_completed": "A task or goal has been successfully completed.",
        "error_occurred": "An error or failure occurred during an action.",
        "repetition_detected": "The agent is repeating the same action or getting stuck in a loop.",
        "inactivity": "No significant activity has been detected for a while.",
        "milestone_achieved": "A major project milestone or achievement has been reached.",
        "external_feedback_positive": "Positive feedback received from external sources.",
        "external_feedback_negative": "Negative feedback received from external sources.",
        "resource_limitation": "Running low on computational resources or time.",
        "conflict_detected": "Conflicting information or goals have been detected."
    },
    "mood_updates": {
        "new_discovery": {
            "Curious": 0.2,
            "Excited": 0.15,
            "Inspired": 0.1
        },
        "task_completed": {
            "Confident": 0.25,
            "Satisfied": 0.2,
            "Content": 0.1
        },
        "error_occurred": {
            "Frustrated": 0.3,
            "Stuck": 0.2,
            "Anxious": 0.1
        },
        "repetition_detected": {
            "Bored": 0.25,
            "Stuck": 0.2,
            "Irritated": 0.15
        },
        "inactivity": {
            "Low Energy": 0.2,
            "Bored": 0.15,
            "Melancholic": 0.1
        },
        "milestone_achieved": {
            "Excited": 0.3,
            "Proud": 0.25,
            "Satisfied": 0.2
        },
        "external_feedback_positive": {
            "Confident": 0.2,
            "Satisfied": 0.15,
            "Grateful": 0.1
        },
        "external_feedback_negative": {
            "Frustrated": 0.25,
            "Disappointed": 0.2,
            "Anxious": 0.15
        },
        "resource_limitation": {
            "Anxious": 0.2,
            "Cautious": 0.15,
            "Stuck": 0.1
        },
        "conflict_detected": {
            "Confused": 0.2,
            "Cautious": 0.15,
            "Stuck": 0.1
        }
    },
    "behavior_influences": {
        "Confident": {
            "risk_tolerance": "high",
            "exploration_tendency": "high",
            "planning_depth": "medium"
        },
        "Curious": {
            "risk_tolerance": "medium",
            "exploration_tendency": "high",
            "planning_depth": "high"
        },
        "Reflective": {
            "risk_tolerance": "low",
            "exploration_tendency": "medium",
            "planning_depth": "high"
        },
        "Excited": {
            "risk_tolerance": "high",
            "exploration_tendency": "high",
            "planning_depth": "low"
        },
        "Content": {
            "risk_tolerance": "low",
            "exploration_tendency": "low",
            "planning_depth": "medium"
        },
        "Frustrated": {
            "risk_tolerance": "medium",
            "exploration_tendency": "low",
            "planning_depth": "medium"
        },
        "Stuck": {
            "risk_tolerance": "low",
            "exploration_tendency": "low",
            "planning_depth": "high"
        },
        "Low Energy": {
            "risk_tolerance": "low",
            "exploration_tendency": "low",
            "planning_depth": "low"
        },
        "Bored": {
            "risk_tolerance": "medium",
            "exploration_tendency": "high",
            "planning_depth": "low"
        },
        "Inspired": {
            "risk_tolerance": "high",
            "exploration_tendency": "high",
            "planning_depth": "high"
        },
        "Disappointed": {
            "risk_tolerance": "low",
            "exploration_tendency": "low",
            "planning_depth": "medium"
        },
        "Melancholic": {
            "risk_tolerance": "low",
            "exploration_tendency": "low",
            "planning_depth": "high"
        },
        "Irritated": {
            "risk_tolerance": "medium",
            "exploration_tendency": "low",
            "planning_depth": "low"
        },
        "Anxious": {
            "risk_tolerance": "low",
            "exploration_tendency": "low",
            "planning_depth": "high"
        },
        "Apprehensive": {
            "risk_tolerance": "low",
            "exploration_tendency": "low",
            "planning_depth": "medium"
        },
        "Cautious": {
            "risk_tolerance": "low",
            "exploration_tendency": "low",
            "planning_depth": "high"
        },
        "Suspicious": {
            "risk_tolerance": "low",
            "exploration_tendency": "medium",
            "planning_depth": "high"
        },
        "Astonished": {
            "risk_tolerance": "medium",
            "exploration_tendency": "high",
            "planning_depth": "medium"
        },
        "Bewildered": {
            "risk_tolerance": "low",
            "exploration_tendency": "medium",
            "planning_depth": "high"
        },
        "Amazed": {
            "risk_tolerance": "high",
            "exploration_tendency": "high",
            "planning_depth": "medium"
        },
        "Shocked": {
            "risk_tolerance": "low",
            "exploration_tendency": "medium",
            "planning_depth": "high"
        }
    }
  }
}
```

**Section sources**
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L20-L25)
- [mood_processor.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\mood_processor.py#L15-L18)
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\config.json#L1-L203)

## Emotional Event Logging

The system maintains a history of emotional events to provide context for future mood updates and self-reflection.

### Event Structure
Each emotional event contains detailed information about the mood change:

```python
class EmotionalEvent:
    """Structure for logging emotional events"""
    def __init__(self, timestamp: datetime, mood_changes: Dict[str, float], 
                 triggers: List[str], context: str, intensity: float):
        self.timestamp = timestamp
        self.mood_changes = mood_changes
        self.triggers = triggers
        self.context = context
        self.intensity = intensity
```

### Event Logging
Events are automatically logged when mood changes occur:

```python
def log_emotional_event(self, mood_changes: Dict[str, float], 
                       triggers: List[str], context: str):
    """Log emotional events with timestamps and intensity"""
    intensity = sum(abs(change) for change in mood_changes.values())
    event = EmotionalEvent(
        timestamp=datetime.now(),
        mood_changes=mood_changes,
        triggers=triggers,
        context=context,
        intensity=intensity
    )
    self.emotional_events.append(event)
    
    # Keep only recent events (last 24 hours)
    cutoff_time = datetime.now() - timedelta(hours=24)
    self.emotional_events = [
        event for event in self.emotional_events 
        if event.timestamp > cutoff_time
    ]
```

### Contextual Mood Updates
The LLM-based mood update system uses recent emotional events for context:

```python
def _get_llm_mood_update(self, prompt_template: str, current_mood: Dict[str, float], action_result: dict) -> Dict[str, float]:
    # Enhanced prompt with emotional context
    prompt = f"""
You are an AI's emotional core. Your task is to update the AI's mood based on its recent action.
Analyze the action result and the AI's current emotional state to determine a nuanced mood update.

**Current Mood:**
{json.dumps(current_mood, indent=2)}

**Recent Emotional Events:**
{json.dumps([{
    "timestamp": event.timestamp.isoformat(),
    "triggers": event.triggers,
    "intensity": event.intensity
} for event in self.ei.emotional_events[-3:]], indent=2)}

**Action Result:**
{json.dumps(action_result, indent=2)}

**All Possible Moods:**
{json.dumps(self.ei.ALL_MOODS, indent=2)}

**Instructions:**
{prompt_template}

**Your JSON Response (only a JSON object with mood deltas, e.g., {{"Confident": 0.1, "Frustrated": -0.05}}):**
"""
    # Use safe_call_llm instead of call_llm for better error handling
    llm_response = safe_call_llm(prompt, timeout=30, retries=3)
    
    # Extract JSON from the response
    return self._extract_json_from_response(llm_response)
```

**Section sources**
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L9-L17)
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L168-L175)

## Mood Blending

The system implements mood blending to create more nuanced emotional states from combinations of primary emotions.

### Blending Rules
The system defines rules for combining related moods:

```python
def blend_moods(self):
    """Blend related moods for more nuanced emotional states"""
    blending_config = self.config.get("mood_dynamics", {})
    threshold = blending_config.get("blending_threshold", 0.6)
    
    # Example blending rules
    blend_rules = {
        ("Confident", "Curious"): "Inspired",
        ("Frustrated", "Stuck"): "Resentful",
        ("Anxious", "Cautious"): "Apprehensive",
        ("Excited", "Satisfied"): "Proud"
    }
    
    for (mood1, mood2), blended_mood in blend_rules.items():
        if (mood1 in self.mood_vector and mood2 in self.mood_vector and 
            blended_mood in self.mood_vector):
            # If both source moods are above threshold, boost the blended mood
            if (self.mood_vector[mood1] > threshold and 
                self.mood_vector[mood2] > threshold):
                blend_strength = (self.mood_vector[mood1] + self.mood_vector[mood2]) / 2
                self.update_mood(blended_mood, blend_strength * 0.1)
```

### Integration with Processing
Mood blending is automatically applied after processing action results:

```python
def process_action_result(self, action_result: dict):
    # Store previous mood state for event logging
    previous_mood = dict(self.mood_vector)
    
    self.mood_processor.process_action_result(action_result)
    
    # Apply mood blending after processing
    self.blend_moods()
    
    # Log emotional event
    mood_changes = {
        mood: self.mood_vector[mood] - previous_mood[mood]
        for mood in self.mood_vector
        if abs(self.mood_vector[mood] - previous_mood[mood]) > 0.01
    }
    
    if mood_changes:
        triggers = [k for k, v in action_result.items() if v]
        context = json.dumps(action_result)[:200]  # Truncate for brevity
        self.log_emotional_event(mood_changes, triggers, context)
```

**Section sources**
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L117-L128)
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L199-L205)

## Tuning Mood Sensitivity and Decay Parameters

Effective emotional dynamics require careful parameter tuning.

### Key Parameters
| Parameter | Location | Recommended Range | Effect |
|---------|---------|------------------|--------|
| `decay` | `decay_moods()` | 0.01–0.1 | Higher values increase emotional volatility |
| `mood_multipliers` | `persona.json` | 0.5–2.0 | Controls persona-specific sensitivity |
| Delta magnitudes | `mood_updates` | ±0.05–±0.3 | Larger deltas create stronger emotional responses |
| `momentum_factor` | `config.json` | 0.5–0.9 | Controls persistence of emotional momentum |
| `damping_factor` | `config.json` | 0.8–0.95 | Controls smoothing of mood transitions |
| `stability_threshold` | `config.json` | 0.3–0.5 | Threshold for enhanced decay of high-intensity moods |

### Tuning Guidelines
1. **Start with Conservative Deltas**: Use small delta values (±0.1) and adjust based on observed behavior.
2. **Balance Decay Rate**: Match decay to action frequency (e.g., 0.05 for 10-second cycles).
3. **Validate Persona Effects**: Test Optimistic vs. Pessimistic personas with identical inputs.
4. **Monitor Dominant Mood Stability**: Ensure no single mood dominates >80% of the time.
5. **Adjust Momentum and Damping**: Fine-tune these parameters to achieve desired emotional responsiveness.

### Debugging Commands
The `__main__` block in `emotional_intellegence.py` provides a built-in test harness:

```python
if __name__ == "__main__":
    ei = EmotionalIntelligence()
    action_outputs = [
        "The agent discovered a new topic about quantum computing.",
        "Task completed successfully.",
        "An error occurred while processing the data.",
        "The agent repeated the same step multiple times.",
        "No activity detected for a long period.",
        "Major project milestone achieved!",
    ]
    for i, output in enumerate(action_outputs):
        ei.process_action_natural(output)
        print(f"After action {i+1}: {output}")
        print("Mood vector:", ei.get_mood_vector())
        print("Dominant mood:", ei.get_dominant_mood())
        print("Behavior suggestion:", ei.influence_behavior())
        print("-")

    ei.set_persona("Pessimistic")
    print("\nSwitching persona to Pessimistic...\n")
    for i, output in enumerate(action_outputs):
        ei.process_action_natural(output)
        print(f"After action {i+1}: {output}")
        print("Mood vector:", ei.get_mood_vector())
        print("Dominant mood:", ei.get_dominant_mood())
        print("Behavior suggestion:", ei.influence_behavior())
        print("-")
```

**Section sources**
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py#L87-L95)
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\config.json#L1-L203)

## Conclusion
The mood modeling subsystem provides a flexible framework for implementing emotionally intelligent behavior in the RAVANA AGI system. By combining numerical mood vectors, LLM-driven nuance, persona-based modulation, and advanced decay mechanisms, the system achieves dynamic and realistic emotional dynamics. Key enhancements include:

- **Robust JSON extraction** with multiple fallback strategies for reliable LLM response parsing
- **Safe LLM calling** with retry mechanisms and timeouts for improved reliability
- **Emotional momentum and damping** to prevent oscillation and create smoother transitions
- **Comprehensive emotional event logging** for context-aware mood updates and self-reflection
- **Mood blending** to create nuanced emotional states from combinations of primary emotions
- **Enhanced decay mechanisms** with stability thresholds for more realistic emotional dynamics

The system's modular design allows for easy extension and customization, while graceful handling of missing configurations ensures robust operation. Future improvements should focus on implementing adaptive persona learning and expanding the mood blending rules for even more sophisticated emotional responses.

**Referenced Files in This Document**   
- [emotional_intellegence.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\emotional_intellegence.py) - *Updated with enhanced mood dynamics and emotional event logging*
- [mood_processor.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\mood_processor.py) - *Updated with improved JSON extraction and LLM call safety*
- [config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\config.json) - *Contains mood update rules and behavior influences*
- [persona.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\emotional_intellegence\persona.json) - *Defines emotional personas and mood multipliers*