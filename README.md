# Ravana AGI Core - An Autonomous, Evolving Agentic System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Ravana AGI Core** is an experimental project to build a modular, autonomous Artificial General Intelligence (AGI) system. It is designed for continuous, 24/7 operation, featuring a sophisticated agentic loop where the system can generate its own tasks, make decisions, reflect on its performance, and evolve over time without direct user input.

## Core Philosophy

The design of Ravana AGI is guided by a few core principles:

1.  **Autonomy**: The system should be capable of operating independently, setting its own goals and learning from its own "experiences" generated through a constant feedback loop.
2.  **Modularity**: Intelligence is not monolithic. The system is broken down into specialized, interoperable modules (e.g., decision-making, memory, reflection) that can be improved or replaced independently.
3.  **Emergent Behavior**: We do not hard-code complex behaviors. Instead, we create a system of simple, interacting components from which complex, intelligent behavior can emerge over time.
4.  **State-Driven Development**: The AGI's actions are heavily influenced by its internal state, particularly its "mood." This allows for more dynamic, less predictable, and more "organic" responses.

## Key Features

-   **Fully Autonomous Operation**: Ravana can run continuously, generating situations and responding to them using a complete agentic feedback loop.
-   **Agentic Architecture**: The system is built on a modular, agentic design where specialized components handle different aspects of intelligence. See [context.md](context.md) for a detailed architectural overview.
-   **Dynamic Decision Making**: An integrated Decision Engine allows the AGI to create plans and goals in response to novel situations.
-   **Emotional Intelligence**: A mood-tracking system influences behavior and responses, adding a layer of dynamic personality and preventing repetitive loops.
-   **Episodic & Semantic Memory**: All interactions, reflections, and decisions are stored in a long-term, searchable vector database (ChromaDB), allowing the AGI to recall past experiences based on semantic relevance.
-   **Continuous Self-Improvement**: A self-reflection module enables the agent to analyze its actions and generate insights to improve its future performance.
-   **Efficient Resource Management**: Large models (e.g., sentence transformers for embeddings) are loaded once at startup and shared across all modules to ensure a low memory footprint and high performance.

## The Agentic Loop

The core of Ravana AGI is its autonomous loop, which dictates how it perceives, thinks, and acts. This cycle runs continuously, allowing the AGI to learn and adapt without human intervention.

```
      +-------------------------+
      |  1. Situation           |
      |     Generation          |
      +-----------+-------------+
                  |
                  v
      +-----------+-------------+
      |  2. Decision & Planning |
      | (influenced by mood &   |
      |  memory)                |
      +-----------+-------------+
                  |
                  v
      +-----------+-------------+
      |  3. Action & Response   |
      +-----------+-------------+
                  |
                  v
+-----------------+------------------+
| 4. Emotional    | 5. Memory        |
|    Response     |    Formation     |
| (Update Mood)   | (Save to DB)     |
+-------+---------+--------+---------+
        |                  |
        v                  v
      +--------------------+----------+
      |  6. Self-Reflection &         |
      |     Curiosity (State-Driven)  |
      +-------------------------------+
```

## Getting Started

### Prerequisites
- Python 3.8+
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/OpenSource-Syndicate/RAVANA.git
    cd RAVANA
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    uv venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    uv add -r requirements.txt
    ```

### Running the AGI

The recommended way to run the system is in 24/7 autonomous mode using the provided wrapper scripts. These scripts handle automatic restarts and health monitoring.

**On Windows:**
```bash
uv run main.py
```

**On macOS/Linux:**
```bash
uv run main.py
```

This will start the main AGI process.

## Monitoring the System

You can monitor the AGI's status and view logs in real-time

## Contributing

Contributions are welcome! If you would like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and ensure the code is well-tested and documented.
4.  Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details. 