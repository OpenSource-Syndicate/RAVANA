# Ravana AGI Core - An Autonomous, Agentic System

This project implements a modular, autonomous Artificial General Intelligence (AGI) system named **Ravana AGI Core**. It is designed for continuous 24/7 operation, featuring a sophisticated agentic loop where the system can generate its own tasks, make decisions, reflect on its performance, and evolve over time without direct user input.

## Key Features

- **Fully Autonomous Operation**: The AGI can run continuously, generating situations and responding to them using a complete agentic feedback loop.
- **Agentic Architecture**: The system is built on a modular, agentic design where specialized components handle different aspects of intelligence.
- **Dynamic Decision Making**: An integrated Decision Engine allows the AGI to create plans and goals in response to novel situations.
- **Emotional Intelligence**: A mood-tracking system influences behavior and responses, adding a layer of dynamic personality.
- **Episodic Memory**: All interactions, reflections, and decisions are stored in a long-term, searchable memory.
- **Continuous Self-Improvement**: A self-reflection module enables the agent to analyze its actions and improve its performance over time.
- **Efficient Resource Management**: Models are loaded once and shared across modules to ensure a low memory footprint.

## The Agentic Loop

The core of Ravana AGI is its autonomous loop, which dictates how it perceives, thinks, and acts:

1.  **Situation Generation**: The `SituationGenerator` creates a novel scenario, such as a technical challenge, an ethical dilemma, or a creative task.
2.  **Decision & Planning**: The `DecisionEngine` receives the situation and formulates a high-level plan to address it.
3.  **Action & Response**: The AGI executes the plan, generating a response.
4.  **Emotional Response**: The `EmotionalIntelligence` module updates the AGI's internal mood based on the nature of the situation and the action taken.
5.  **Memory Formation**: The entire interaction—situation, plan, response, and emotional state—is recorded in `EpisodicMemory`.
6.  **Self-Reflection**: The `AgentSelfReflection` module analyzes the outcome, generating insights that can improve future decision-making.

This loop runs continuously, allowing the AGI to learn and adapt without human intervention.

## Getting Started

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/AGI.git
    cd AGI
    ```

2.  **Set up a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows: .\.venv\Scripts\activate
    # On macOS/Linux: source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have a `requirements.txt` file with all necessary packages like `fastapi`, `uvicorn`, `sentence-transformers`, `torch`, `chromadb`, etc.)*

### Running the AGI

The recommended way to run the system is in 24/7 autonomous mode using the provided wrapper scripts.

**On Windows:**
```bash
start_agi_24_7.bat
```

**On macOS/Linux:**
```bash
./run_agi_24_7.py
```

This will start the main AGI process with automatic restart and health monitoring capabilities.

### Monitoring the System

You can monitor the AGI's status and view logs in real-time:

-   `agi_system.log`: Main log for the AGI's thoughts, decisions, and actions.
-   `situation_generator.log`: Logs related to the generation of new situations.
-   `interactions.jsonl`: A structured log of every situation, response, and reflection, perfect for analysis.
-   `autonomous_agi.log`: Log for the wrapper script that monitors the main process.

## Architecture Overview

The system is composed of several core modules that work in concert. For a detailed explanation of the system architecture and the flow of information between modules, please see `context.md`.

## Development Roadmap

Future plans for expanding the AGI's capabilities are outlined in `plan.md`. This includes enhancing the decision-making process, integrating more complex data sources, and enabling more advanced forms of self-improvement.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. 