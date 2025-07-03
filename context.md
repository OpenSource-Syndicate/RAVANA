# Ravana AGI Core - System Architecture

This document provides an overview of the Ravana AGI system architecture, explaining how the different modules interact to create an autonomous, agentic system capable of continuous self-improvement.

## System Overview

The Ravana AGI is designed as a modular system where specialized components, or "modules," handle different aspects of intelligence. These modules are orchestrated by a central `AGISystem` that manages the flow of information and decision-making. The system's primary mode of operation is a continuous, autonomous loop.

## The Autonomous Agentic Loop

The core of the system is its autonomous loop, which defines the AGI's process for perceiving, thinking, and acting without direct user intervention.

1.  **Situation Generation**: The loop begins with the `SituationGenerator` module, which creates a novel and challenging scenario. These situations can range from technical problems and ethical dilemmas to creative tasks and hypothetical scenarios, providing a constant stream of new experiences.

2.  **Decision & Planning**: The generated situation is passed to the `DecisionEngine`. This module analyzes the situation and formulates a high-level plan to address it. This involves breaking down the problem into manageable sub-goals and tasks.

3.  **Action & Response**: Based on the plan from the Decision Engine, the `AGISystem` generates a concrete response or action. In the current implementation, this is a textual response outlining the plan, but it can be extended to include executing code, accessing APIs, or controlling other systems.

4.  **Emotional Response**: The `EmotionalIntelligence` module processes the situation and the AGI's response, updating the agent's internal emotional state. This state, represented by a mood vector, can influence future decisions and add a dynamic, more "human" layer to the AGI's behavior.

5.  **Memory Formation**: The entire interaction—including the situation, the plan, the response, and the resulting emotional state—is recorded as a new memory in the `EpisodicMemory` module. This module uses a vector database (ChromaDB) to store memories, allowing for efficient, semantic retrieval of relevant past experiences.

6.  **Self-Reflection**: Finally, the `AgentSelfReflection` module analyzes the outcome of the interaction. It reflects on what was done, whether it was successful, and what could be learned. These reflections are also stored in memory and can be used to improve the AGI's core logic and decision-making processes over time.

This entire loop runs continuously, enabling the AGI to learn, adapt, and potentially evolve its own capabilities without requiring a human operator.

## Core Modules

-   **AGISystem (`main.py`)**: The central coordinator that orchestrates the agentic loop, initializes all modules, and manages the overall state.
-   **SituationGenerator**: Creates diverse and unpredictable situations to drive the AGI's learning process.
-   **DecisionEngine**: Responsible for high-level planning and goal setting.
-   **EmotionalIntelligence**: Manages the AGI's internal mood and emotional state.
-   **EpisodicMemory**: The long-term memory of the AGI, storing all experiences for later retrieval and reflection.
-   **AgentSelfReflection**: Enables the AGI to learn from its past actions and improve its future performance.
-   **CuriosityTrigger**: Can be used to inject novelty and drive exploration, though it is secondary to the `SituationGenerator` in the main loop.
-   **Resource Management**: A key design feature is the centralized loading of large models (like sentence transformers). The main system loads these models once at startup and passes them to the other modules, significantly reducing the memory footprint and improving performance.

## 24/7 Continuous Operation

The system is designed for robust, 24/7 continuous operation, managed by a set of wrapper scripts (`run_autonomous.py`, `start_agi_24_7.bat`, etc.). These scripts provide:

-   **Automatic Restarts**: If the main AGI process crashes or hangs, it is automatically restarted.
-   **Heartbeat Monitoring**: The system's log files are monitored for activity. If the AGI becomes unresponsive, the wrapper script will restart it.
-   **Resource Monitoring**: CPU and memory usage are logged for performance analysis and debugging.

This robust setup ensures that the AGI can run autonomously for extended periods, continuously learning and evolving.

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