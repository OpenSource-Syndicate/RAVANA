# Situation Generator Module

This module generates situations for the AGI system to tackle without user input. It leverages existing modules like trend_analysis, curiosity_trigger, and event_detection to create realistic and challenging scenarios.

## Overview

The Situation Generator is designed to run 24/7 without user input, continuously generating diverse situations for the AGI system to tackle. It uses one LLM to generate situations and the main AGI system's LLM to tackle these situations.

## Features

- **Continuous Operation**: Runs 24/7 without requiring user input
- **Diverse Situation Types**:
  - Trending Topics (from RSS feeds)
  - Curiosity Exploration (using the curiosity trigger module)
  - Event Response (based on detected events)
  - Hypothetical Scenarios
  - Technical Challenges
  - Ethical Dilemmas
  - Creative Tasks
- **Integration with Existing Modules**:
  - Uses trend_analysis to monitor RSS feeds
  - Uses curiosity_trigger to generate interesting topics
  - Uses event_detection to identify events in data
  - Uses agent_self_reflection for LLM calls

## Usage

### Running the Situation Generator

```bash
# Generate a single situation
python situation_generator.py --generate

# Generate a specific type of situation
python situation_generator.py --generate --type trending_topic

# Run continuously
python situation_generator.py --run
```

### Integration with AGI System

To integrate the Situation Generator with the main AGI system, you can import and use the `SituationGenerator` class:

```python
from modules.situation_generator.situation_generator import SituationGenerator

# Initialize the generator
generator = SituationGenerator()

# Start the generator
generator.start()

# Get situations as they are generated
while True:
    situation = generator.get_situation()
    if situation:
        # Process the situation with the main AGI system
        response = agi_system.process_situation(situation)
        print(f"Situation: {situation['prompt']}")
        print(f"Response: {response}")
```

## Situation Types

1. **Trending Topic**: Based on recent news articles from RSS feeds
2. **Curiosity Exploration**: Uses the curiosity trigger module to find interesting topics
3. **Event Response**: Responds to detected events in data
4. **Hypothetical Scenario**: Generates hypothetical scenarios for the AGI to tackle
5. **Technical Challenge**: Creates technical problems to solve
6. **Ethical Dilemma**: Presents ethical dilemmas for consideration
7. **Creative Task**: Generates creative tasks for the AGI to complete

## Dependencies

See `requirements.txt` for a list of dependencies.

## License

This module is part of the Ravana AGI Core v1 project. 