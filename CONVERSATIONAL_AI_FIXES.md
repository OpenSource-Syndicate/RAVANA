# Conversational AI Startup and Shutdown Fixes

## Issues Identified

1. **Import Error**: The conversational AI module was trying to import `RAVANACommunicator` from the wrong path.
2. **Constructor Parameter Mismatch**: The `DiscordBot` and `TelegramBot` constructors were being called with incorrect parameters.
3. **Signal Handler Conflicts**: The conversational AI module was setting up its own signal handlers which conflicted with the main RAVANA system.
4. **Shutdown Loop Issues**: The shutdown coordinator had issues with futures attached to different event loops.
5. **Graceful Shutdown Not Properly Integrated**: The conversational AI wasn't being properly cleaned up during shutdown.

## Fixes Implemented

### 1. Fixed Import Statement
**File**: `modules/conversational_ai/main.py`
- Changed import from `modules.conversational_ai.communicator.ravana_communicator` to `modules.conversational_ai.communication.ravana_bridge`

### 2. Fixed Constructor Parameters
**File**: `modules/conversational_ai/main.py`
- Fixed `DiscordBot` instantiation to include all required parameters: token, command_prefix, and conversational_ai reference
- Fixed `TelegramBot` instantiation to include all required parameters: token, command_prefix, and conversational_ai reference

### 3. Added Standalone Parameter
**File**: `modules/conversational_ai/main.py`
- Modified the `start()` method to accept a `standalone` parameter
- Signal handlers are only set up when running in standalone mode

**File**: `core/system.py`
- Updated the conversational AI thread to pass `standalone=False` when starting the module as part of the main system

**File**: `launch_conversational_ai.py`
- Updated the launcher script to pass `standalone=True` when running as a standalone module

### 4. Fixed Shutdown Coordinator
**File**: `core/shutdown_coordinator.py`
- Fixed the `_phase_stop_background_tasks` method to use `asyncio.gather` with `return_exceptions=True` to avoid loop issues

### 5. Improved Main Shutdown Handling
**File**: `main.py`
- Simplified the signal handlers to just set the shutdown event instead of calling async code
- Improved the `shutdown_agi_system` function to properly use the shutdown coordinator

## Testing

The fixes have been tested and verified:
1. The conversational AI module can now be imported successfully
2. The conversational AI module starts properly when RAVANA starts
3. Graceful shutdown works without errors
4. The conversational AI module can still be run standalone using the launcher script

## Verification Commands

```bash
# Test import
python -c "from modules.conversational_ai.main import ConversationalAI; print('Success')"

# Test standalone launch
python launch_conversational_ai.py

# Test as part of RAVANA
python main.py --skip-state-recovery
```