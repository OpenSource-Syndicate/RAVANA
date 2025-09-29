# RAVANA AGI System - Context and Development Guide

## Project Overview

RAVANA is an autonomous, evolving agentic system designed for continuous 24/7 operation, driven by internal states, reflection, and self-generated goals. It is an experiment in autonomous general intelligence that runs continuously without constant human prompting, builds and recalls memory, models emotions as internal signals guiding behavior, reflects, adapts, and improves itself over time.

## Architecture and Components

### Core System Structure
- **`main.py`**: Entry point that initializes and runs the AGI system
- **`core/`**: Orchestrator, state manager, internal services, and configuration
- **`modules/`**: Plug-and-play intelligence modules (reflection, experimentation, conversational AI, etc.)
- **`services/`**: Service-level abstractions: memory, knowledge bases, environment interaction
- **`database/`**: SQLModel-based schemas, models, and the database engine for long-term persistence

### Key Components
1. **AGISystem**: Main orchestrator that manages all components
2. **Snake Agent**: Autonomous code analysis and improvement system that monitors, analyzes, and experiments with the RAVANA codebase in the background
3. **Memory Service**: Enhanced memory system for episodic, semantic, and working memory
4. **Conversational AI**: Multi-platform bot system supporting Discord and Telegram
5. **Experimentation Engine**: System for designing and running experiments to validate hypotheses
6. **Reflection Module**: Self-reflection capabilities allowing the AGI to analyze its performance
7. **Emotional Intelligence**: Mood modeling and emotional responses that influence behavior

### Configuration System
The system uses a comprehensive configuration system in `core/config.py` with:
- Database URL settings
- Autonomous loop parameters (curiosity chance, reflection chance, sleep duration)
- Emotional intelligence settings (positive/negative moods, emotional persona)
- Model settings for embeddings and LLMs
- Snake Agent configuration with enhanced performance settings
- Blog integration configuration
- Conversational AI settings

## Building and Running

### Prerequisites
- Python 3.13 or higher
- Dependencies specified in `pyproject.toml`

### Installation
```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# or for development
pip install -e .
```

### Running the System
```bash
# Start the main system
python main.py

# Run with specific options:
# Run a specific physics experiment
python main.py --physics-experiment "Quantum Tunneling Barrier Analysis"

# Run in discovery mode
python main.py --discovery-mode

# Run experiment tests
python main.py --test-experiments

# Run a single task
python main.py --single-task "Analyze the current state of the codebase"
```

## Key Features

### Autonomous Operation
- Continuous 24/7 operation without human intervention
- Self-managed curiosity and reflection cycles
- Automated task generation and execution
- Mood-based behavior modification

### Memory and Learning
- Enhanced memory service for storing and retrieving experiences
- Adaptive learning engine that improves decision-making over time
- Knowledge compression and consolidation
- Episodic memory for contextual understanding

### Multi-Platform Communication
- Discord and Telegram bot integration
- Bidirectional communication with external users
- Emotional context synchronization between platforms

### Code Analysis and Improvement
- Snake Agent for autonomous codebase analysis
- Automated experiment execution and evaluation
- Safe experimenter for testing code changes
- File system monitoring for changes

### Experimentation and Reflection
- Self-designed experiments to validate hypotheses
- Reflection module for analyzing performance and generating insights
- Systematic knowledge building through experimentation

## Development Conventions

### Code Structure
- Use async/await for all I/O operations
- Implement proper error handling and graceful degradation
- Follow the shutdown coordinator pattern for component cleanup
- Use the enhanced memory service for all memory operations
- Implement proper type hints throughout

### Configuration
- Store all configuration in environment variables or the Config class
- Use the Config class for accessing configuration values
- Follow the provider model pattern for LLM integration

### Testing
- Use pytest for unit and integration tests
- Follow async testing patterns with pytest-asyncio
- Test both success and failure scenarios
- Include performance tests for long-running operations

### Logging
- Use structured logging with appropriate log levels
- Log important state changes and decisions
- Include contextual information in log messages
- Use different log formats based on Config settings

## Key Files and Directories

### Core Components
- `main.py`: System entry point and main event loop
- `core/system.py`: Main AGI system orchestrator
- `core/config.py`: Configuration management
- `core/snake_agent.py`: Autonomous code analysis agent
- `core/snake_agent_enhanced.py`: Enhanced version of the snake agent
- `core/shutdown_coordinator.py`: Graceful shutdown coordination

### Modules
- `modules/reflection_module.py`: Self-reflection capabilities
- `modules/experimentation_module.py`: Experiment design and execution
- `modules/conversational_ai/main.py`: Multi-platform conversation handler
- `modules/emotional_intellegence/`: Mood and emotional modeling
- `modules/personality/`: Personality and creativity modeling

### Services
- `services/memory_service.py`: Memory storage and retrieval
- `services/knowledge_service.py`: Knowledge base management
- `services/data_service.py`: Data collection and processing
- `services/multi_modal_service.py`: Multi-modal processing capabilities

### Database
- `database/models.py`: SQLModel database models
- `database/engine.py`: Database engine and connection management

## Important Notes

1. The system is designed for continuous operation and has extensive shutdown handling
2. The Snake Agent is a key component that autonomously improves the codebase
3. Configuration can be controlled through environment variables
4. The system includes multiple safety mechanisms and graceful error handling
5. Memory and learning are integral to the system's continuous improvement
6. The system maintains detailed logs for debugging and analysis
7. The architecture supports multiple AI model providers with fallback mechanisms