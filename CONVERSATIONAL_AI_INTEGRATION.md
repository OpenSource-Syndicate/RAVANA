# Conversational AI Module Integration Guide

This document explains how to integrate the Conversational AI Module with the main RAVANA AGI system.

## Overview

The Conversational AI Module is designed to work alongside the main RAVANA AGI system, sharing the same memory infrastructure while providing independent conversational capabilities through Discord and Telegram.

## Integration Points

### 1. Shared Memory System

The Conversational AI Module uses the same memory infrastructure as the main RAVANA system:
- PostgreSQL with pgvector for persistent storage
- ChromaDB for vector similarity search
- Shared memory context for consistent personality and knowledge

### 2. Emotional Intelligence

The module extends the existing emotional intelligence system to:
- Adapt responses based on user interests
- Maintain consistent personality across platforms
- Track emotional state per user

### 3. RAVANA Communication Bridge

The module can communicate with the main RAVANA system through:
- Task delegation from users to RAVANA
- Proactive messaging from RAVANA to users
- Status updates and notifications

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r modules/conversational_ai/requirements.txt
   ```

2. Configure the module by editing `modules/conversational_ai/config.json`:
   - Add your Discord bot token
   - Add your Telegram bot token
   - Configure memory paths

## Starting the Module

Run the module with:
```bash
python launch_conversational_ai.py
```

## Integration with Main RAVANA System

### Task Delegation

Users can delegate tasks to RAVANA through the `/task` command (Telegram) or `!task` command (Discord). The module will:
1. Capture the task description
2. Send it to RAVANA through the communication bridge
3. Notify the user when the task is complete

### Proactive Messaging

RAVANA can send messages to users through the communication bridge:
```python
# From main RAVANA system
communicator = RAVANACommunicator()
communicator.notify_user(user_id, "Important update from RAVANA!")
```

## Data Flow

1. User message received through Discord/Telegram
2. Message processed by Emotional Intelligence Engine
3. Context retrieved from Shared Memory System
4. Response generated based on user personality and current mood
5. Response sent back to user through appropriate channel
6. Interactions stored in user-specific memory
7. RAVANA system can initiate messages to users through the communication bridge

## Security Considerations

- User data is isolated in separate directories with access controls
- Bot tokens are stored in environment variables
- Message encryption for sensitive data in transit and at rest
- Compliance with Discord and Telegram API usage policies

## Testing

Run the test suite with:
```bash
python modules/conversational_ai/test_module.py
```

## Extending the Module

### Adding New Platforms

To add support for new messaging platforms:
1. Create a new bot interface in `modules/conversational_ai/bots/`
2. Implement the required methods (`start`, `stop`, `send_message`)
3. Register the bot in `main.py`

### Customizing Emotional Responses

To customize emotional responses:
1. Modify the persona configurations in `modules/emotional_intellegence/persona.json`
2. Update mood processing in `conversational_ei.py`
3. Adjust response generation in `generate_response` method

## Troubleshooting

### Common Issues

1. **Bot not responding**: Check that the bot tokens are correct and the bots have the required permissions
2. **Memory access issues**: Ensure the memory paths are correctly configured
3. **Emotional intelligence errors**: Check that the configuration files are properly formatted

### Logs

Check the logs for detailed error information:
- Discord bot logs: `discord_bot.log`
- Telegram bot logs: `telegram_bot.log`
- Main module logs: `conversational_ai.log`