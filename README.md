# RAVANA AGI: An Autonomous, Evolving Agentic System

RAVANA is an experimental open-source Artificial General Intelligence (AGI) system designed to think, feel, evolve, and act autonomously. It represents a philosophical and technical exploration into building a digital organism capable of self-directed behavior and continuous evolution.

## Overview

RAVANA AGI is built on a modular, agentic architecture with a continuous six-step loop:
1. Situation Generation
2. Decision & Planning
3. Action & Environment Interaction
4. Mood Update
5. Memory Logging
6. Self-Reflection & Curiosity

Unlike traditional AI systems that respond to user prompts, RAVANA operates continuously, generating its own tasks, making decisions based on its internal state, and evolving through self-reflection and learning.

## Key Features

- **Autonomous Operation**: Continuous 24/7 operation without user intervention
- **Emotional Intelligence**: Mood tracking and emotional state modeling
- **Memory Management**: Episodic and semantic memory systems
- **Self-Reflection**: Continuous learning and self-improvement
- **Curiosity-Driven**: Generates its own tasks and experiments
- **Modular Architecture**: Extensible design with interchangeable components

## System Architecture

The system is organized into several key components:

- **Core System**: Main orchestration and state management
- **Modules**: Specialized functionality (adaptive learning, self-reflection, curiosity, etc.)
- **Services**: Data, knowledge, memory, and multi-modal services
- **Actions**: Executable behaviors and operations
- **Database**: Storage for memory and persistent data

## Getting Started

### Prerequisites

- Python 3.8+
- uv (for dependency management)

### Installation

```bash
git clone https://github.com/OpenSource-Syndicate/RAVANA.git
cd RAVANA
uv venv
# Activate virtual environment:
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
uv pip install -e .
```

### Running RAVANA

```bash
uv run main.py
```

## Documentation

Comprehensive documentation is available in the [docs](docs/) directory:

- [System Architecture](docs/core/architecture.md)
- [Core Components](docs/core/components.md)
- [Module Documentation](docs/modules/)
- [API Reference](docs/api/)
- [Development Guide](docs/development/)

## Contributing

We welcome contributions from researchers, developers, and AI enthusiasts. Please see our [Contributing Guide](docs/development/contributing.md) for more information.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Researchers and developers in the AGI community
- Open-source contributors who make this work possible