# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## 🧠 About Ravana AGI Core

Ravana AGI Core is an experimental open-source Artificial General Intelligence (AGI) system designed for 24/7 autonomous operation. It's a **living agent** driven by internal states, reflection, and self-generated goals, featuring modular architecture with pluggable intelligence components.

**Core Philosophy:**
- **Autonomy**: Runs continuously without human intervention
- **Modularity**: Composable intelligence via independent modules  
- **Emergence**: Complex behaviors arise from simple component interactions
- **State-Driven**: Internal "mood" influences planning and reactions

## 🚀 Essential Commands

### Environment Setup
```bash
# Clone and setup
git clone https://github.com/OpenSource-Syndicate/RAVANA.git
cd RAVANA

# Create virtual environment (recommended: use uv)
uv venv
# Windows activation
.venv\Scripts\activate
# macOS/Linux activation  
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### Running the System
```bash
# Full autonomous mode (main operation)
uv run main.py

# Single task execution
uv run main.py --prompt "Your specific task here"

# Physics experimentation mode
uv run main.py --physics-experiment "Quantum Tunneling Barrier Analysis"
uv run main.py --discovery-mode
uv run main.py --test-experiments

# Skip state recovery on startup
uv run main.py --skip-state-recovery
```

### Testing & Development
```bash
# Run individual module tests
python -m pytest modules/agent_self_reflection/test_self_reflection.py
python -m pytest modules/curiosity_trigger/test_curiosity_trigger.py
python -m pytest modules/emotional_intellegence/test_enhanced_emotional_intelligence.py

# Run specific test suites
python test_enhanced_snake_agent.py
python test_blog_integration.py
python run_physics_tests.py

# Simple component testing
python simple_gemini_test.py
python simple_snake_test.py
```

### Database & Logs
```bash
# Database file location
ls ravana_agi.db  # SQLite database

# View logs (overwritten each run)
tail -f ravana_agi.log

# Enable JSON logging (set before running)
export LOG_FORMAT=JSON  # Windows: set LOG_FORMAT=JSON
```

## 🏗️ High-Level Architecture

### Core System Flow
```
AGISystem (main.py) 
    ↓
Autonomous Loop:
1. Check external data & search results
2. Handle mood-based behavior modifiers  
3. Process curiosity triggers
4. Generate situation OR continue task plan
5. Retrieve relevant memories
6. Make enhanced decision (with learning)
7. Execute action via ActionManager
8. Update mood & trigger reflection
9. Memorize interaction
10. Sleep & repeat
```

### Key Components

#### **AGISystem (`core/system.py`)**
- Central orchestrator and "brainstem"
- Manages the 6-step autonomous loop
- Handles graceful shutdown and state persistence
- Coordinates all modules and background tasks

#### **ActionManager (`core/action_manager.py` & `enhanced_action_manager.py`)**
- Discovers and registers all available actions from `core/actions/`
- Parses LLM JSON decisions and executes corresponding actions
- Handles error management and logging
- Supports both sync and async action execution

#### **Core Modules (`modules/`)**
- **DecisionEngine**: High-level planning with mood/memory influence
- **EmotionalIntelligence**: Tracks vectorized mood states
- **EpisodicMemory**: ChromaDB + SQLite semantic memory system  
- **SituationGenerator**: Creates novel scenarios to prevent idleness
- **CuriosityTrigger**: Generates exploration topics and ideas
- **AgentSelfReflection**: Analyzes patterns and generates improvements
- **AdaptiveLearningEngine**: Learns from decision outcomes

#### **Services (`services/`)**
- **DataService**: RSS feed collection, event detection
- **KnowledgeService**: Information compression and summarization
- **MemoryService**: Async interface to episodic memory
- **MultiModalService**: Handles different content types

#### **Snake Agent (`core/snake_agent*.py`)**
- Advanced code analysis and experimentation system
- Dual LLM architecture (coding + reasoning models)
- Multiprocess/multithreaded execution
- Safety sandboxing and approval workflows

### Memory Architecture
- **ChromaDB**: Vector similarity search for semantic retrieval
- **SQLite**: Metadata and structured data storage
- **Automatic consolidation**: Periodic memory optimization
- **Context-aware retrieval**: Situation-based memory queries

### Background Tasks
- **Data Collection**: RSS feeds every hour (`DATA_COLLECTION_INTERVAL`)
- **Event Detection**: Process articles every 10 minutes
- **Knowledge Compression**: Daily knowledge summarization
- **Memory Consolidation**: Every 6 hours memory optimization
- **Blog Maintenance**: Autonomous content generation (if enabled)

## 🛠️ Development Patterns

### Adding New Actions
Actions are auto-discovered from `core/actions/`. Create new action:

```python
# core/actions/my_new_action.py
from core.actions.action import Action
from typing import Any, Dict, List

class MyNewAction(Action):
    @property
    def name(self) -> str:
        return "my_action_name"
    
    @property 
    def description(self) -> str:
        return "Clear description for LLM decision making"
    
    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "param_name",
                "type": "string", 
                "description": "Parameter description",
                "required": True
            }
        ]
    
    async def execute(self, **kwargs: Any) -> Any:
        # Action implementation
        return {"result": "success"}
```

### Module Structure
Each module follows this pattern:
```
modules/module_name/
├── __init__.py
├── module_name.py          # Main implementation
├── requirements.txt        # Module dependencies  
├── pyproject.toml         # Module metadata
└── test_module_name.py    # Tests
```

### Configuration Management
- Environment variables in `core/config.py`
- API keys and sensitive config in `core/config.json`
- Module-specific config in `modules/*/config.json`

## 🔧 Key Configuration

### Environment Variables
```bash
# Core system
DATABASE_URL="sqlite:///ravana_agi.db"
LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT="TEXT"  # TEXT or JSON

# Loop behavior  
CURIOSITY_CHANCE=0.3        # Probability of curiosity trigger
LOOP_SLEEP_DURATION=10      # Seconds between iterations
REFLECTION_CHANCE=0.1       # Probability of reflection

# Models
EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Snake Agent
SNAKE_AGENT_ENABLED="True"
SNAKE_ENHANCED_MODE="True"
SNAKE_CODING_MODEL="gpt-oss:20b"
SNAKE_REASONING_MODEL="deepseek-r1:7b"

# Graceful shutdown
GRACEFUL_SHUTDOWN_ENABLED="True"
STATE_PERSISTENCE_ENABLED="True"
SHUTDOWN_TIMEOUT=30

# Blog integration (optional)
BLOG_ENABLED="True"
BLOG_AUTO_PUBLISH_ENABLED="False" 
BLOG_REQUIRE_APPROVAL="True"
```

### Multi-LLM Configuration
The system supports multiple LLM providers configured in `core/config.json`:
- **Zuki**: `gpt-4o:online`, `deepseek-chat`, `deepseek-reasoner`
- **ElectronHub**: `deepseek-v3-0324`, `gpt-4o-2024-11-20`
- **Zanity**: `deepseek-r1`, `claude-3.5-sonnet:free`
- **Gemini**: Multiple API keys with automatic rotation

## 🧪 Testing Strategy

### Module Testing
- Each module has individual test files
- Run tests for specific functionality: `python test_[component].py`
- Integration tests in `tests/` directory

### System Testing  
- Physics experiments: Test complex reasoning chains
- Blog integration: End-to-end content generation
- Memory system: Consolidation and retrieval accuracy
- Snake Agent: Code analysis and safety systems

### Debug Commands
```bash
# Test individual components
python simple_gemini_test.py        # LLM connectivity
python simple_snake_test.py         # Snake Agent basic ops
python demo_enhanced_emotional_intelligence.py  # Mood system

# Integration testing
python test_blog_integration.py     # Blog system
python test_enhanced_agi_system.py  # Full system test
```

## ⚠️ Important Notes

### State Management
- System automatically saves/restores state on shutdown/startup
- Multi-step plans persist across restarts
- Mood and memory context maintained
- Use `--skip-state-recovery` to start fresh

### Memory Considerations  
- Large models (SentenceTransformer, etc.) loaded once and shared
- Periodic memory consolidation prevents bloat
- Action cache cleared periodically
- Background task management for cleanup

### Safety & Sandboxing
- Snake Agent runs in controlled environment
- File access restrictions via `SNAKE_BLACKLIST_PATHS`
- Approval requirements for sensitive operations
- Automatic timeout handling for long-running tasks

### Multi-Modal Support
- Text, image, and audio processing capabilities  
- Extensible pipeline for new content types
- Integration with various AI service providers

## 🚦 Production Considerations

- **24/7 Operation**: Designed for continuous autonomous running
- **Graceful Shutdown**: Handles signals and saves state properly  
- **Resource Management**: Automatic cleanup and memory optimization
- **Error Recovery**: Resilient to individual component failures
- **Logging**: Structured JSON logging available for monitoring
- **Scalability**: Modular design allows independent component scaling

