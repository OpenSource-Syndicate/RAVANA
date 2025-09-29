# RAVANA Documentation

Welcome to the documentation for RAVANA AGI - an autonomous, evolving agentic system.

## Table of Contents

* [Getting Started](#getting-started)
* [Architecture](#architecture)
* [Modules](#modules)
* [Services](#services)
* [Configuration](#configuration)
* [Running the System](#running-the-system)
* [Development](#development)

## Getting Started

RAVANA is an experimental autonomous general intelligence system designed to run continuously, build and recall memory, model emotions, and self-improve over time.

To get started:
1. Set up your environment (see [Development](#development))
2. Run the main system: `python main.py`
3. Explore the system's capabilities

## Architecture

RAVANA follows a modular architecture:

* **`core/`** - Core system components including orchestrator, state management, and configuration
* **`modules/`** - Intelligence modules for reflection, experimentation, conversational AI, etc.
* **`services/`** - Service abstractions for memory, knowledge bases, and environment interaction
* **`database/`** - Data persistence with SQLModel-based schemas and engine
* **`scripts/`** - Standalone scripts for special operations
* **`tests/`** - Test suite for validating system behavior
* **`docs/`** - Documentation (this directory)
* **`assets/`** - Static assets like images and other media
* **`wiki/`** - Detailed design documents and architecture breakdowns

## Modules

The system is composed of several key modules:
* **Reflection Module**: Self-reflection capabilities
* **Experimentation Engine**: System for designing and running experiments
* **Conversational AI**: Multi-platform bot system
* **Emotional Intelligence**: Mood modeling and emotional responses
* **Memory Systems**: Enhanced memory for episodic, semantic, and working memory

## Services

Core services include:
* **Memory Service**: Enhanced memory system for storing and retrieving experiences
* **Knowledge Service**: Knowledge base management
* **Data Service**: Data collection and processing
* **Multi-Modal Service**: Multi-modal processing capabilities

## Configuration

Configuration is handled through environment variables and the Config class in `core/config.py`. The system supports:
* Database URL settings
* Autonomous loop parameters
* Emotional intelligence settings
* Model settings for embeddings and LLMs
* Snake Agent configuration
* Blog integration configuration
* Conversational AI settings

## Running the System

To run RAVANA:
```bash
python main.py
```

Additional command-line options:
* `--physics-experiment "<experiment_name>"` - Run a specific physics experiment
* `--discovery-mode` - Run in discovery mode
* `--test-experiments` - Run experiment tests
* `--single-task "<task_prompt>"` - Run a single task

Standalone scripts are available in the `scripts/` directory for specialized operations.

## Development

For development:
1. Create a virtual environment: `python3 -m venv .venv`
2. Activate it: `source .venv/bin/activate` (Linux/Mac) or `source .venv/Scripts/activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests: `pytest`

The system includes extensive logging and debugging capabilities.