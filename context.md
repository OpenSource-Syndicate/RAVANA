# AGI System Architecture

This document provides an overview of the AGI system architecture, explaining how the different modules interact with each other to create a cohesive artificial general intelligence system.

## System Overview

The AGI system is designed as a modular architecture where specialized components handle different aspects of intelligence. These modules work together through a central integration system that coordinates their activities and manages the flow of information between them.

## Core Modules

### 1. Episodic Memory (modules/episodic_memory)

**Purpose**: Stores and retrieves memories from conversations and experiences.

**Key Components**:
- FastAPI server for memory storage and retrieval
- Embedding-based semantic search for finding relevant memories
- SQLite database for persistent storage
- ChromaDB for vector storage and similarity search

**Interactions**:
- Receives input from user conversations
- Extracts key memories using LLM
- Stores memories with embeddings for later retrieval
- Provides relevant memories to other modules when needed

### 2. Emotional Intelligence (modules/emotional_intellegence)

**Purpose**: Manages the agent's emotional state and influences behavior.

**Key Components**:
- Mood tracking system with six basic moods: Curious, Frustrated, Confident, Stuck, Low Energy, Reflective
- Mood vector that represents the intensity of each mood
- Behavior influence system that suggests actions based on dominant mood

**Interactions**:
- Receives action results from other modules
- Updates mood vector based on success/failure of actions
- Provides behavior suggestions to the main system
- Triggers curiosity module when appropriate

### 3. Decision Engine (modules/decision_engine)

**Purpose**: Plans and manages goals, subgoals, and tasks.

**Key Components**:
- Hierarchical goal planner inspired by Hierarchical Task Networks (HTN)
- JSON-based storage of goals, subgoals, and tasks
- LLM-based goal decomposition

**Interactions**:
- Receives high-level goals from user or other modules
- Breaks down goals into subgoals and tasks
- Tracks progress and completion
- Generates new objectives as tasks are completed

### 4. Curiosity Trigger (modules/curiosity_trigger)

**Purpose**: Generates curiosity-driven exploration of topics.

**Key Components**:
- Wikipedia and Reddit data sources for interesting facts
- LLM-based topic suggestion system
- Lateralness parameter to control how unrelated topics should be

**Interactions**:
- Receives recent topics from memory or conversation
- Generates curiosity topics based on lateralness
- Fetches articles about topics
- Provides learning prompts to the main system

### 5. Agent Self-Reflection (modules/agent_self_reflection)

**Purpose**: Enables the agent to reflect on its actions and improve.

**Key Components**:
- Structured self-reflection process after tasks
- LLM-based reflection generation
- Database storage of reflections
- Self-modification capability to improve code based on reflections

**Interactions**:
- Receives task summaries and outcomes from other modules
- Generates reflections on what worked, what failed, what surprised, and what to learn
- Stores reflections for future reference
- Can propose and test code modifications to improve performance

### 6. Knowledge Compression (modules/knowledge_compression)

**Purpose**: Summarizes accumulated knowledge to prevent memory bloat.

**Key Components**:
- Scheduled compression of knowledge logs
- LLM-based summarization
- Storage of compressed summaries

**Interactions**:
- Receives logs from other modules
- Compresses knowledge on a schedule
- Provides concise summaries to the main system
- Helps prevent memory overload

### 7. Event Detection (modules/event_detection)

**Purpose**: Detects emerging events and trending topics from text data.

**Key Components**:
- Topic detection using sentence embeddings and clustering
- Content filtering based on sentiment analysis
- Event alerting system

**Interactions**:
- Receives streams of text data
- Identifies significant clusters as events
- Alerts the main system to important events
- Helps the AGI stay aware of emerging trends

### 8. Information Processing (modules/information_processing)

**Purpose**: Processes information from various sources.

**Submodules**:
- **YouTube Transcription**: Transcribes YouTube videos to text
- **Trend Analysis**: Analyzes trends from RSS feeds

**Interactions**:
- Receives URLs or data sources from user or other modules
- Processes information into text format
- Provides processed information to memory and other modules
- Helps the AGI consume and understand diverse information sources

### 9. AGI Experimentation (modules/agi_experimentation)

**Purpose**: Runs experiments to test hypotheses and learn.

**Key Components**:
- LLM-based experiment design
- Code generation and execution
- Result analysis and learning

**Interactions**:
- Receives experiment ideas from user or curiosity module
- Designs and runs experiments
- Analyzes results and generates insights
- Feeds new knowledge back to memory

## System Integration

The main AGI system (main.py) integrates all these modules into a cohesive system. It:

1. **Initializes all components** at startup
2. **Manages the memory server** in a separate thread
3. **Processes user input** and routes it to appropriate modules
4. **Coordinates module interactions** to create a seamless experience
5. **Maintains system state** across interactions
6. **Supports autonomous operation** for continuous learning and self-improvement

## 24/7 Continuous Operation

The AGI system is designed for robust, 24/7 continuous operation. This is managed by a set of scripts that provide monitoring, automatic restarts, and detailed logging.

### Key Features of Continuous Mode:

-   **Automatic Restart**: If the main AGI process crashes or becomes unresponsive, the `run_autonomous.py` script will automatically restart it.
-   **Heartbeat Monitoring**: A heartbeat mechanism checks if the AGI system is still active by monitoring its log file. If there's no activity for a configurable period, the process is considered hung and will be restarted.
-   **Resource Monitoring**: The `run_autonomous.py` script monitors the CPU and memory usage of the AGI process and its children, logging the data for later analysis.
-   **Port Management**: Before starting, the script checks if the memory server's port (8000) is in use and attempts to terminate any existing processes to prevent conflicts.
-   **Restart Throttling**: To prevent rapid-fire restarts in case of a persistent crash, the system limits the number of restarts within a specific time window.
-   **Graceful Shutdown**: The system is designed to handle `Ctrl+C` (KeyboardInterrupt) gracefully, ensuring that processes are terminated properly.

### How to Run in 24/7 Mode:

The easiest way to start the system in continuous mode is by using the provided wrapper scripts:

-   **On Windows**: `start_agi_24_7.bat`
-   **On macOS/Linux**: `run_agi_24_7.py`

These scripts handle the necessary command-line arguments and ensure the system runs with the optimal settings for long-term stability.

### Monitoring the System:

While the AGI is running, you can use `check_agi_status.py` to get a real-time overview of the system's health, including uptime, resource usage, and the latest log entries.

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