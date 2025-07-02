# AGI System

An integrated Artificial General Intelligence (AGI) system that combines multiple specialized modules to create a cohesive, intelligent agent with memory, emotional intelligence, decision-making capabilities, curiosity, self-reflection, and more.

This project is designed for modularity and continuous operation, allowing for 24/7 autonomous running with robust monitoring and self-recovery capabilities.

## Architecture

The AGI system consists of the following modules:

1. **Episodic Memory** - Stores and retrieves memories from conversations and experiences
2. **Emotional Intelligence** - Manages the agent's emotional state and influences behavior
3. **Decision Engine** - Plans and manages goals, subgoals, and tasks
4. **Curiosity Trigger** - Generates curiosity-driven exploration of topics
5. **Agent Self-Reflection** - Enables the agent to reflect on its actions and improve
6. **Knowledge Compression** - Summarizes accumulated knowledge to prevent memory bloat
7. **Event Detection** - Detects emerging events and trending topics from text data
8. **Information Processing**
   - YouTube Transcription - Transcribes YouTube videos to text
   - Trend Analysis - Analyzes trends from RSS feeds
9. **AGI Experimentation** - Runs experiments to test hypotheses and learn

For a more detailed explanation of the architecture, see [context.md](context.md).

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/AGI.git
    cd AGI
    ```

2. **Set up a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3. **Install dependencies using `pip`:**
    The system now includes several utility scripts that require additional packages.
    ```bash
    pip install requests sentence-transformers wikipedia uvicorn openai google-generativeai psutil tabulate
    ```
    *Note: If you are using `uv`, you can run `uv pip install -r requirements.txt` if you have a `requirements.txt` file with these packages.*

## Usage

### Running in Interactive Mode

To interact with the AGI system directly, run `main.py`:
```bash
python main.py
```
For debug mode:
```bash
python main.py --debug
```
Once the system is running, you can interact with it through the command line.

**Special commands:**
- `exit` - Quit the system.
- `auto` - Start autonomous mode.
- `manual` - Stop autonomous mode.

### Running in 24/7 Autonomous Mode

The recommended way to run the system for continuous operation is using the provided wrapper script.

**On Windows:**
Simply double-click the `start_agi_24_7.bat` file. This will open a new command prompt and run the system with the optimal settings for 24/7 operation.

**On macOS/Linux:**
```bash
./run_agi_24_7.py
```
*(You may need to make the script executable first with `chmod +x run_agi_24_7.py`)*

This script runs `run_autonomous.py` with the `--continuous` flag, which enables automatic restarts and heartbeat monitoring to ensure the system stays online.

### Monitoring the System

While the AGI is running, you can check its status using the `check_agi_status.py` script in a separate terminal:
```bash
python check_agi_status.py
```
For a live, auto-updating view:
```bash
python check_agi_status.py --watch
```
This will display the status of running processes, memory usage, and the latest log entries.

### Troubleshooting

If you encounter issues, several debugging scripts are available:

- **Test the Memory Server**:
    ```bash
    python test_memory_server.py
    ```
- **Test the LLM Connections**:
    ```bash
    python test_llm.py
    ```
- **Test the Sentence Transformers Module**:
    ```bash
    python test_sentence_transformers.py
    ```
- **Run a Deep-Dive Debug on `main.py`**:
    ```bash
    python debug_main.py --mode trace
    ```
    This script can also run in `pdb` mode for interactive debugging.

## Autonomous Mode Details

In autonomous mode, the system:

1. Generates diverse situations using the `SituationGenerator` module.
2. Processes these situations using the main `AGISystem`.
3. Reflects on its responses to improve over time.
4. Stores all interactions in its episodic memory.

### Situation Types

The situation generator can create various types of situations:

- **Trending Topics**: Based on recent news articles from RSS feeds.
- **Curiosity Exploration**: Uses the curiosity trigger module to find interesting topics.
- **Event Response**: Responds to detected events in data.
- **Hypothetical Scenarios**: Generates hypothetical scenarios for the AGI to tackle.
- **Technical Challenges**: Creates technical problems to solve.
- **Ethical Dilemmas**: Presents ethical dilemmas for consideration.
- **Creative Tasks**: Generates creative tasks for the AGI to complete.

## Development

To add a new module:
1. Create a new directory in the `modules` folder.
2. Implement the module functionality.
3. Update the imports and initialization in `main.py`.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

# Ravana AGI Core v1 - Autonomous Mode

## Overview

Ravana AGI is a modular agentic system with continuous learning and self-improvement capabilities. This implementation includes an autonomous mode where the system can run 24/7 without user input. In this mode, one LLM generates situations for the main LLM to tackle.

## Features

- **Autonomous Operation**: The system can run 24/7 without requiring user input
- **Situation Generation**: A dedicated module generates diverse situations for the AGI to tackle
- **Self-Reflection**: The system reflects on its responses to improve over time
- **Memory Storage**: All situations and responses are stored in the episodic memory
- **Modular Design**: Each component runs as a separate service for easy testing and replacement

## Autonomous Mode

In autonomous mode, the system:

1. Generates diverse situations using the situation generator module
2. Processes these situations using the main AGI system
3. Reflects on its responses to improve over time
4. Stores all interactions in its episodic memory

### Situation Types

The situation generator can create various types of situations:

- **Trending Topics**: Based on recent news articles from RSS feeds
- **Curiosity Exploration**: Uses the curiosity trigger module to find interesting topics
- **Event Response**: Responds to detected events in data
- **Hypothetical Scenarios**: Generates hypothetical scenarios for the AGI to tackle
- **Technical Challenges**: Creates technical problems to solve
- **Ethical Dilemmas**: Presents ethical dilemmas for consideration
- **Creative Tasks**: Generates creative tasks for the AGI to complete

## Usage

### Running in Autonomous Mode

You can run the AGI system in autonomous mode using the `run_autonomous.py` script:

```bash
python run_autonomous.py
```

This will start the AGI system in autonomous mode and display its output in real-time.

### Options

- `--debug`: Enable debug logging
- `--duration <seconds>`: Run for a specific duration (default: indefinitely)

Example:

```bash
# Run in autonomous mode with debug logging for 1 hour
python run_autonomous.py --debug --duration 3600
```

### Manual Control

You can also start the AGI system normally and toggle autonomous mode:

```bash
python main.py
```

Once started, you can use the following commands:

- `auto`: Start autonomous mode
- `manual`: Stop autonomous mode
- `exit`: Quit the AGI system

## Architecture

The autonomous mode leverages several modules:

- **Situation Generator**: Generates diverse situations for the AGI to tackle
- **Agent Self-Reflection**: Enables the AGI to reflect on its responses
- **Episodic Memory**: Stores all interactions for future reference
- **Emotional Intelligence**: Provides emotional context for responses
- **Event Detection**: Identifies events in data
- **Information Processing**: Processes information from various sources

## Development

The system is designed to be modular and extensible. You can add new situation types or improve existing ones by modifying the `situation_generator` module.

## License

This project is part of the Ravana AGI Core v1 project. 