# Ravana AGI Core: Developer Guide

Welcome to the developer guide for the Ravana AGI Core. This document provides a deep dive into the system's architecture, components, and development practices to help you contribute to this exciting project.

## 1. Introduction

Ravana AGI Core is an open-source project aimed at creating a fully autonomous, evolving agentic system. The core philosophy is to build a digital organism that can think, learn, and act on its own, driven by internal states and self-generated goals.

### Key Principles:

*   **Autonomy**: The system operates 24/7 without human intervention.
*   **Modularity**: Intelligence is composed of independent, swappable modules.
*   **Emergence**: Complex behaviors arise from the interaction of simpler components.
*   **State-Driven Behavior**: The agent's "mood" and internal state influence its actions.

## 2. Getting Started

This section will guide you through setting up your development environment and running the Ravana AGI system.

### Prerequisites

*   Python 3.8+
*   `uv` for dependency management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/OpenSource-Syndicate/RAVANA.git
    cd RAVANA
    ```

2.  **Set up the virtual environment:**
    ```bash
    uv venv
    ```

3.  **Activate the virtual environment:**
    *   **Windows:**
        ```bash
        .venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```

4.  **Install dependencies:**
    ```bash
    uv pip install -e .
    ```

### Running the System

To run the Ravana AGI in its default autonomous mode, execute the following command:

```bash
uv run main.py
```

This will start the main autonomous loop, and you will see log output in your console.

## 3. Architecture

The Ravana AGI Core is built on a modular, decoupled architecture that allows for independent development and evolution of its components. The system is orchestrated by a central `AGISystem` class, which manages the main autonomous loop and the interaction between the various modules and services.

### High-Level Diagram

```
+-----------------------------------------------------------------+
| AGISystem (core/system.py - Orchestrator)                       |
|-----------------------------------------------------------------|
| - Initializes all modules and services                          |
| - Manages the main autonomous loop                              |
| - Holds the shared state of the AGI (mood, memories, etc.)      |
| - Delegates action execution to the ActionManager               |
+--------------------------+--------------------------------------+
                           |
         +-----------------v------------------+
         |   Autonomous Loop                  |
         |----------------------------------|
         | 1. Handle Behavior Modifiers     |
         | 2. Handle Curiosity              |
         | 3. Generate Situation            |
         | 4. Retrieve Memories             |
         | 5. Make Decision                 |
         | 6. Execute Action & Memorize     |
         | 7. Update Mood & Reflect         |
         +----------------------------------+
                           |
+--------------------------+--------------------------------------+
| Core Modules             | Services                            |
|--------------------------|-------------------------------------|
| - SituationGenerator     | - DataService                       |
| - DecisionEngine         | - KnowledgeService                  |
| - EmotionalIntelligence  | - MemoryService                     |
| - CuriosityTrigger       |                                     |
| - ReflectionModule       |                                     |
| - ExperimentationModule  |                                     |
+--------------------------+--------------------------------------+
```

### Key Components

*   **AGISystem (`core/system.py`)**: The central nervous system of the AGI. It initializes all components, manages the main loop, and holds the shared state.

*   **ActionManager (`core/action_manager.py`)**: Responsible for registering and executing all available actions. It decouples the decision-making logic from the action implementation.

*   **Core Modules (`modules/`)**: These are the building blocks of the AGI's intelligence. Each module is responsible for a specific cognitive function (e.g., decision making, reflection, curiosity).

*   **Services (`services/`)**: These provide access to external resources and data, such as databases, APIs, and file systems.

## 4. Core Modules

The core modules are the heart of the AGI's cognitive abilities. Each module is responsible for a specific function and can be developed and replaced independently.

*   **SituationGenerator (`modules/situation_generator/`)**: This module is responsible for creating novel situations or prompts to keep the AGI engaged and prevent idleness. It acts as the AGI's internal muse, sparking new thoughts and actions.

*   **DecisionEngine (`modules/decision_engine/`)**: The high-level planner of the AGI. It takes the current situation, relevant memories, and the agent's mood as input, and then decides on a course of action. It breaks down goals into executable plans.

*   **EmotionalIntelligence (`modules/emotional_intellegence/`)**: This module manages the AGI's internal emotional state or "mood." It processes the outcomes of actions and updates the mood vector accordingly. The mood, in turn, influences the AGI's decisions and behaviors.

*   **CuriosityTrigger (`modules/curiosity_trigger/`)**: This module drives the AGI's desire to explore and learn. It can generate new topics of interest based on the agent's experiences, prompting it to ask questions or seek out new information.

*   **ReflectionModule (`modules/reflection_module.py`)**: Responsible for self-reflection. It analyzes past experiences and decisions to identify patterns, draw conclusions, and generate hypotheses for improvement.

*   **ExperimentationModule (`modules/experimentation_module.py`)**: This module works closely with the `ReflectionModule`. It takes hypotheses generated during reflection and designs experiments to test them. This allows the AGI to learn and adapt its strategies over time.

## 5. The Autonomous Loop

The main autonomous loop is the central process that drives the AGI's behavior. It's an infinite loop that continuously cycles through a series of cognitive steps. The implementation of the loop can be found in the `run_autonomous_loop` method of the `AGISystem` class (`core/system.py`).

Here is a step-by-step breakdown of the loop:

1.  **Handle Behavior Modifiers**: At the beginning of each iteration, the loop checks for any behavior modifiers generated in the previous cycle. These modifiers are influenced by the agent's mood and can suggest actions like taking a break.

2.  **Handle Curiosity**: There's a chance for the `CuriosityTrigger` to activate, generating new topics of interest for the AGI to explore.

3.  **Generate Situation**: The `SituationGenerator` creates a new situation or prompt to stimulate the AGI. This ensures the agent always has something to think about.

4.  **Retrieve Memories**: Based on the generated situation, the system queries the `MemoryService` to retrieve relevant past experiences and knowledge.

5.  **Make Decision**: The `DecisionEngine` takes the situation, memories, and current mood to make a decision. The output is a plan of action.

6.  **Execute Action & Memorize**: The `ActionManager` executes the chosen action. The interaction (situation, decision, and action output) is then summarized and stored in the agent's memory via the `MemoryService`.

7.  **Update Mood & Reflect**: The outcome of the action is used by the `EmotionalIntelligence` module to update the agent's mood. If the mood doesn't improve, the `ReflectionModule` and `ExperimentationModule` are triggered to analyze the situation and learn from it.

After these steps, the loop pauses for a configurable duration before starting a new iteration.

## 6. Services

The services layer provides a clean interface for interacting with external resources like databases and APIs. These services are used by the `AGISystem` and other modules to manage data, knowledge, and memory.

### DataService (`services/data_service.py`)

The `DataService` is responsible for all interactions with the database for logging and data ingestion. It provides methods to:

*   Fetch and save articles from RSS feeds.
*   Detect and save events from the collected data.
*   Log actions, moods, situations, decisions, and experiments to the database.

### KnowledgeService (`services/knowledge_service.py`)

The `KnowledgeService` is responsible for compressing and summarizing information. It uses a knowledge compression module to create a more compact and efficient knowledge base for the AGI. This is crucial for long-term learning and reasoning.

### MemoryService (`services/memory_service.py`)

The `MemoryService` provides an asynchronous interface to the agent's episodic memory system. It allows the AGI to:

*   Retrieve relevant memories based on a query.
*   Save new memories from interactions.
*   Extract key information from conversations to form new memories.
*   Consolidate and refine memories over time.

## 7. Adding New Actions

The AGI's capabilities can be extended by adding new actions. The system uses an automatic discovery mechanism to find and register actions, making it easy to add new ones.

### Action Structure

All actions must inherit from the `core.actions.action.Action` abstract base class. This class defines the required structure for an action.

Here's an example of a simple "hello_world" action:

```python
# in core/actions/misc.py

from core.actions.action import Action
from typing import Any, Dict, List

class HelloWorldAction(Action):
    @property
    def name(self) -> str:
        return "hello_world"

    @property
    def description(self) -> str:
        return "A simple action that prints a greeting."

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "name",
                "type": "string",
                "description": "The name to include in the greeting.",
                "required": True,
            }
        ]

    async def execute(self, **kwargs: Any) -> Any:
        name = kwargs.get("name")
        return f"Hello, {name}!"

```

### How to Add a New Action

1.  **Create a Python file**: Create a new Python file in the `core/actions/` directory. You can create new subdirectories to organize your actions if needed.

2.  **Define the Action Class**: In the new file, create a class that inherits from `core.actions.action.Action`.

3.  **Implement the Properties**:
    *   `name`: A unique name for your action.
    *   `description`: A clear description of what the action does. This is used in prompts for the LLM.
    *   `parameters`: A list of dictionaries defining the parameters your action accepts. Each parameter should have a `name`, `type`, `description`, and `required` flag.

4.  **Implement the `execute` method**: This method contains the logic for your action. It will receive the parameters as keyword arguments.

That's it! The `ActionRegistry` will automatically discover and register your new action when the application starts. The `DecisionEngine` will then be able to use your new action in its plans.

## 8. Configuration

The Ravana AGI system can be configured using environment variables. The configuration variables are defined in the `core.config.Config` class.

### General Configuration

*   **`DATABASE_URL`**: The connection string for the database.
    *   Default: `sqlite:///ravana_agi.db`

*   **`LOG_LEVEL`**: The logging level for the application.
    *   Default: `INFO`
    *   Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

*   **`LOG_FORMAT`**: The format for the logs.
    *   Default: `TEXT`
    *   Options: `TEXT`, `JSON`

### Autonomous Loop Settings

*   **`CURIOSITY_CHANCE`**: The probability (between 0.0 and 1.0) that the curiosity module will be triggered in each loop.
    *   Default: `0.3`

*   **`LOOP_SLEEP_DURATION`**: The number of seconds the autonomous loop will sleep between iterations.
    *   Default: `10`

*   **`ERROR_SLEEP_DURATION`**: The number of seconds the loop will sleep after encountering an error.
    *   Default: `60`

### Model Settings

*   **`EMBEDDING_MODEL`**: The name of the sentence-transformer model to use for generating embeddings.
    *   Default: `all-MiniLM-L6-v2`

### Background Task Intervals

*   **`DATA_COLLECTION_INTERVAL`**: The interval in seconds for the background task that fetches data from RSS feeds.
    *   Default: `3600` (1 hour)

*   **`EVENT_DETECTION_INTERVAL`**: The interval in seconds for the background task that detects events from the collected data.
    *   Default: `600` (10 minutes)

## 9. Logging and Monitoring

The Ravana AGI system provides detailed logging to help you monitor its behavior and debug issues.

### Log Configuration

The logging system is configured in `main.py`. By default, it logs to both the console and a file named `ravana_agi.log`. The log file is overwritten on each run.

You can configure the log level and format using the `LOG_LEVEL` and `LOG_FORMAT` environment variables.

### JSON Logging

For easier integration with log management tools, you can enable structured JSON logging by setting the `LOG_FORMAT` environment variable to `JSON`.

## 10. Contributing

We welcome contributions from the community! Whether you want to add new features, fix bugs, or improve the documentation, your help is appreciated.

### Contribution Guidelines

1.  **Fork the repository**: Start by forking the main repository to your own GitHub account.

2.  **Create a new branch**: Create a new branch for your feature or bug fix. Use a descriptive name like `feature/your-idea` or `fix/issue-number`.

3.  **Make your changes**: Implement your changes, ensuring you follow the existing code style.

4.  **Write documentation**: If you add a new feature or change existing behavior, be sure to update the documentation accordingly.

5.  **Open a pull request**: Once you're ready, open a pull request against the `main` branch of the original repository. Provide a clear description of your changes and why they are needed.

Thank you for helping to build the future of AGI! 