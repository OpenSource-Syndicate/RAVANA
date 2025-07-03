# Ravana AGI Core - System Architecture Deep Dive

This document provides a detailed technical overview of the Ravana AGI system architecture, explaining how the different modules interact to create an autonomous, agentic system.

## Architectural Diagram (High-Level)

```
+-----------------------------------------------------------------+
| AGISystem (main.py - Orchestrator)                              |
|-----------------------------------------------------------------|
| - Initializes all modules with shared resources (e.g., models)  |
| - Manages the main autonomous loop                              |
| - Holds the AGI's current state (e.g., mood)                    |
+--------------------------+--------------------------------------+
                           |
         +-----------------v------------------+
         | Autonomous Agentic Loop          |
         |----------------------------------|
         | 1. SituationGenerator            |
         | 2. DecisionEngine                |
         | 3. Action Execution              |
         | 4. EmotionalIntelligence         |
         | 5. EpisodicMemory (R/W)          |
         | 6. AgentSelfReflection           |
         +----------------------------------+
                           |
+--------------------------+--------------------------------------+
| Core Modules (Independent Components)                           |
|-----------------------------------------------------------------|
| - EmotionalIntelligence  - EpisodicMemory (FastAPI + DB)        |
| - DecisionEngine         - AgentSelfReflection                  |
| - SituationGenerator     - CuriosityTrigger                     |
+-----------------------------------------------------------------+
```

## Core Design Principles

-   **Modularity**: Each component is a self-contained module with a specific responsibility. This allows for independent development, testing, and improvement. For example, the `DecisionEngine` can be replaced with a more advanced planner without affecting the `EpisodicMemory` module.
-   **State-Driven Behavior**: The AGI's actions are not just a direct response to a stimulus. Its internal state, especially the `mood` vector managed by the `EmotionalIntelligence` module, is a key input for decision-making. This prevents deterministic, repetitive behavior and allows for more nuanced responses.
-   **Resource Efficiency**: Large machine learning models (like sentence-transformers) are a significant source of memory overhead. The `AGISystem` loads these models **once** at startup and passes the model objects to any module that needs them. This avoids duplicating large models in memory and is critical for stable, long-term operation.

## Module Deep Dive

-   **AGISystem (`main.py`)**: The central nervous system of the AGI. It initializes all other modules, injects shared dependencies (like the embedding model), and runs the main autonomous loop.

-   **SituationGenerator**: Its sole purpose is to create novel scenarios to prevent the AGI from becoming idle. These situations act as the primary catalyst for the agentic loop, forcing the AGI to think, plan, and act.

-   **DecisionEngine**: This is the AGI's high-level planner. It receives a situation and formulates a strategy to address it. This involves breaking down the problem into sub-goals and outlining a course of action.

-   **EmotionalIntelligence**: Manages the AGI's internal mood, represented as a numerical vector. The mood is updated based on the outcome of actions and the nature of situations. A positive outcome might increase "happiness," while a frustrating one might increase "annoyance." This state is fed back into the `DecisionEngine`, influencing future plans.

-   **EpisodicMemory (`modules/episodic_memory/`)**: The AGI's long-term memory. This module is implemented as a FastAPI service, which exposes endpoints for saving and retrieving memories.
    -   **Technology**: It uses a hybrid approach:
        -   `ChromaDB`: A vector database for fast, semantic similarity search on memory embeddings. This allows the AGI to find relevant past experiences even if they don't share exact keywords.
        -   `SQLite`: Stores the raw text and metadata associated with each memory.
    -   **Functionality**: It handles memory extraction, embedding generation, storage, and retrieval.

-   **AgentSelfReflection**: This module periodically analyzes past interactions stored in memory. It looks for patterns, successes, and failures, generating insights like: "My plans for technical problems have been successful, but my creative writing is repetitive." These insights are stored back into memory and can be used to improve the AGI's core logic.

## The Autonomous Loop in Detail

A single cycle of the autonomous loop is a complete perception-action-learning sequence:

1.  **Perceive**: The `SituationGenerator` creates a prompt (e.g., "Analyze the ethical implications of AI in art.").
2.  **Orient**: The `AGISystem` queries the `EpisodicMemory` for relevant past experiences related to AI ethics, art, etc. The current `mood` is also retrieved.
3.  **Decide**: The situation, relevant memories, and current mood are passed to the `DecisionEngine`. It formulates a plan (e.g., "1. Define AI in art. 2. List ethical pros. 3. List ethical cons. 4. Conclude with a balanced view.").
4.  **Act**: The `AGISystem` executes the plan, generating a detailed text response.
5.  **Update State & Memory**:
    -   The `EmotionalIntelligence` module assesses the response, perhaps slightly increasing the "intellectual satisfaction" component of the mood vector.
    -   The entire interaction (situation, plan, response, new mood) is packaged and sent to the `EpisodicMemory` module to be saved as a new memory.
6.  **Learn**: The `AgentSelfReflection` module is triggered. It might analyze this interaction and generate an insight: "My analysis of ethics improves when I explicitly list pros and cons." This insight becomes a new, powerful memory that can guide future planning.

This loop repeats, creating a flywheel effect where the AGI continuously accumulates experience and refines its own behavior.

## Robust 24/7 Operation

The system is designed for continuous operation via wrapper scripts (`run_autonomous.py`, `start_agi_24_7.bat`). These scripts are not part of the AGI's "brain" but act as an external supervisor.

-   **Automatic Restarts**: If the main Python process crashes due to an unhandled exception, the script will automatically restart it after a short delay.
-   **Heartbeat Monitoring**: The wrapper script monitors the AGI's log files for new entries. If no activity is detected for a configurable period (i.e., the AGI has frozen), the process is killed and restarted.
-   **Resource Logging**: The supervisor script periodically logs the CPU and memory usage of the AGI process, providing valuable data for debugging performance issues like memory leaks.

## Information Flow

1. User input is received by the main system
2. The input is processed by the emotional intelligence module to update the agent's mood
3. The input is stored as memories in the episodic memory module
4. Relevant memories are retrieved to provide context
5. Based on the input and the agent's mood, appropriate actions are taken:
   - Creating goals if requested
   - Running experiments if requested
   - Triggering curiosity if the agent is curious
   - Reflecting on actions if appropriate
   - Compressing knowledge periodically
   - Detecting events from input
   - Processing information from external sources
6. The system responds to the user based on the actions taken

## Extensibility

The modular design allows for easy extension of the AGI system:

1. New modules can be added by creating a new directory in the modules folder
2. The new module can be integrated by updating the imports and initialization in main.py
3. New functionality can be added to existing modules without affecting the rest of the system

## Future Directions

The AGI system can be extended in several ways:

1. **Multi-modal processing**: Adding vision, audio, and other sensory inputs
2. **Improved reasoning**: Enhancing the decision engine with more sophisticated planning algorithms
3. **Social intelligence**: Adding modules for understanding and navigating social dynamics
4. **Embodied cognition**: Connecting the AGI to robotic systems for physical interaction
5. **Meta-learning**: Enabling the AGI to improve its own learning algorithms 