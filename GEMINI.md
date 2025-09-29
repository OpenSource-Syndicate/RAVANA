# GEMINI.md - RAVANA AGI Project

## Project Overview

This project, RAVANA AGI, is a Python-based autonomous, evolving agentic system. The goal is to create an AI that can run continuously, build and recall memory, model emotions, and self-improve over time. It is designed as a research platform to explore the concepts of autonomous general intelligence.

The architecture is modular, with a clear separation of concerns:
- **`core/`**: Contains the main orchestration, state management, and other core services.
- **`modules/`**: Houses plug-and-play intelligence modules like reflection, experimentation, and conversational AI.
- **`services/`**: Provides abstractions for memory, knowledge, and other high-level services.
- **`database/`**: Manages data persistence with schemas and a database engine.
- **`main.py`**: The main entry point for the application.

## Building and Running

**1. Environment Setup:**

Create a virtual environment and install the required dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
**Note:** The `requirements.txt` file is currently empty. You may need to install dependencies manually based on the imports in the source code.

**2. Running the System:**

The main entry point is `main.py`. You can run it directly to start the autonomous loop:

```bash
python main.py
```

**3. Running Specific Modes:**

The application supports several command-line arguments to run in different modes:

*   **Run a specific physics experiment:**
    ```bash
    python main.py --physics-experiment "<experiment_name>"
    ```
*   **Run in discovery mode:**
    ```bash
    python main.py --discovery-mode
    ```
*   **Run experiment tests:**
    ```bash
    python main.py --test-experiments
    ```
*   **Run a single task:**
    ```bash
    python main.py --single-task "<task_prompt>"
    ```

## Development Conventions

*   **Configuration:** The core configuration is managed in `core/config.py`. It uses environment variables to allow for flexible configuration.
*   **State Management:** The system's state is managed by the `StateManager` class in `core/state_manager.py`. It handles saving, loading, and validating the state.
*   **Modularity:** The project is highly modular, with different functionalities separated into different modules and services. This makes it easy to extend and maintain.
*   **Logging:** The project uses the `logging` module for logging. The log format and level can be configured in `core/config.py`.
*   **Signal Handling:** The application has robust signal handling for graceful shutdown, which is implemented in `main.py`.
