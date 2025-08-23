# RAVANA AGI Quick Start Guide

This guide provides a quick path to get RAVANA AGI up and running in minutes.

## Prerequisites

Before starting, ensure you have:
- Python 3.8 or higher
- Git
- Basic command-line knowledge

## Installation (5 minutes)

1. **Install uv** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/OpenSource-Syndicate/RAVANA.git
   cd RAVANA
   ```

3. **Create and activate virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   uv pip install -e .
   ```

## Configuration (2 minutes)

Set up your API key for language model access:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

For Windows:
```cmd
set OPENAI_API_KEY=your-openai-api-key-here
```

## First Run (1 minute)

Start RAVANA AGI:
```bash
uv run main.py
```

You should see output similar to:
```
[INFO] Initializing RAVANA AGI System v1.0.0
[INFO] Loading configuration from core/config.json
[INFO] Initializing modules...
[INFO] System initialized successfully
[INFO] Starting autonomous loop...
```

## Basic Interaction

RAVANA operates autonomously, but you can interact with it through:

1. **Console Input**: Type messages directly in the terminal
2. **API Endpoints**: Access via HTTP REST API
3. **File System**: Monitor data and log files

### Console Interaction Example

```
> Hello RAVANA, how are you today?
[INFO] Processing user input...
[EMOTION] Mood updated: happiness +0.1
[RESPONSE] Hello! I'm functioning optimally and curious about what we might explore together today.
```

## Key Directories

After running RAVANA, you'll see these directories:
- `data/`: System data and memory storage
- `logs/`: Log files for monitoring
- `chroma_db/`: Vector database for semantic memory

## Monitoring System Status

Check system health:
```bash
# View recent logs
tail -f logs/system.log

# Check data directory
ls -la data/
```

## Stopping the System

To stop RAVANA:
1. Press `Ctrl+C` in the terminal
2. Wait for graceful shutdown (10-30 seconds)
3. You should see "System shutdown complete" message

## Next Steps

1. **Explore Documentation**: 
   - [Full Installation Guide](installation.md)
   - [Configuration Guide](configuration.md)
   - [System Architecture](architecture.md)

2. **Customize Configuration**:
   - Modify `core/config.json`
   - Adjust personality traits
   - Configure enabled modules

3. **Experiment with Modules**:
   - Review [Module Documentation](../modules/index.md)
   - Enable/disable specific capabilities
   - Observe behavioral changes

4. **Contribute**:
   - Fork the repository
   - Implement new features
   - Submit pull requests

## Troubleshooting

### Common Issues

1. **"Command not found" errors**
   - Ensure virtual environment is activated
   - Check PATH environment variable

2. **API Key Issues**
   - Verify API key is set correctly
   - Check key has proper permissions
   - Confirm internet connectivity

3. **Module Initialization Failures**
   - Check logs for specific error messages
   - Verify dependencies are installed
   - Review configuration files

### Getting Help

- Check [Installation Guide](installation.md) for detailed instructions
- Review [FAQ](faq.md) for common questions
- Open an issue on GitHub for bugs or feature requests

## System Capabilities

RAVANA AGI includes these core capabilities out of the box:

- **Autonomous Operation**: Continuous self-directed behavior
- **Emotional Intelligence**: Mood tracking and emotional responses
- **Memory Systems**: Episodic and semantic memory storage
- **Learning Engine**: Continuous improvement through experience
- **Self-Reflection**: Analysis of behavior and performance
- **Curiosity Engine**: Intrinsic motivation and goal generation

## Example Use Cases

1. **Research Assistant**: 
   - Ask RAVANA to explore topics of interest
   - Review its research process and findings

2. **Creative Partner**:
   - Collaborate on writing or idea generation
   - Observe how curiosity drives exploration

3. **Learning Companion**:
   - Discuss concepts and see how RAVANA processes information
   - Monitor its learning and adaptation over time

## Safety Information

RAVANA is an experimental system designed for research purposes:
- Operates autonomously with minimal human intervention
- Continuously generates and pursues its own goals
- May exhibit unexpected behaviors during development
- Should be run in a controlled environment

Always monitor system behavior and be prepared to shut down if needed.

## Feedback and Support

We welcome your feedback:
- GitHub Issues for bugs and feature requests
- GitHub Discussions for general conversation
- Email: feedback@ravana-agi.org

Thank you for trying RAVANA AGI!