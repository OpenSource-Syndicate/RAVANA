# Enhanced Snake Agent



## Update Summary
**Changes Made**   
- Updated documentation to reflect enhanced bot connectivity management and status reporting
- Added new section on Bot Connectivity Management and Status Reporting
- Enhanced troubleshooting guide with bot-specific issues
- Updated section sources to reflect recent code changes
- Added references to conversational AI module files

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Architecture Overview](#architecture-overview)
5. [Detailed Component Analysis](#detailed-component-analysis)
6. [Dependency Analysis](#dependency-analysis)
7. [Performance Considerations](#performance-considerations)
8. [Bot Connectivity Management and Status Reporting](#bot-connectivity-management-and-status-reporting)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Conclusion](#conclusion)

## Introduction
The Enhanced Snake Agent is an autonomous AI system designed to continuously monitor, analyze, and improve code within the RAVANA repository. It leverages a hybrid concurrency model combining threading and multiprocessing to perform real-time file monitoring, code analysis, experimentation, and self-improvement tasks. The agent operates as a background daemon, detecting file changes, initiating analysis workflows, conducting safe code experiments, and proposing or applying improvements based on LLM-driven reasoning. Its modular architecture ensures fault tolerance, performance monitoring, and graceful shutdown capabilities.

## Project Structure
The Enhanced Snake Agent is organized within the `core` directory of the RAVANA repository, with tightly integrated components for logging, threading, process management, and file monitoring. The system follows a modular design where each component has a single responsibility and communicates through well-defined interfaces and queues.

```mermaid
graph TB
subgraph "Enhanced Snake Agent Core"
ESA[EnhancedSnakeAgent]
Config[SnakeAgentConfiguration]
LogManager[SnakeLogManager]
ThreadingManager[SnakeThreadingManager]
ProcessManager[SnakeProcessManager]
FileMonitor[ContinuousFileMonitor]
end
ESA --> Config
ESA --> LogManager
ESA --> ThreadingManager
ESA --> ProcessManager
ESA --> FileMonitor
ThreadingManager --> |Queues| FileMonitor
ProcessManager --> |Distributes Tasks| ThreadingManager
LogManager --> |Logs Events| ESA
FileMonitor --> |Emits Events| ThreadingManager
style ESA fill:#4ECDC4,stroke:#333
style Config fill:#45B7D1,stroke:#333
style LogManager fill:#96CEB4,stroke:#333
style ThreadingManager fill:#FFEAA7,stroke:#333
style ProcessManager fill:#DDA0DD,stroke:#333
style FileMonitor fill:#F7DC6F,stroke:#333
```

**Diagram sources**
- [core/snake_agent_enhanced.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_agent_enhanced.py)
- [core/snake_data_models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_data_models.py#L375-L412)
- [core/snake_log_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_log_manager.py#L105-L371)
- [core/snake_threading_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_threading_manager.py#L0-L199)
- [core/snake_process_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_process_manager.py#L0-L199)
- [core/snake_file_monitor.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_file_monitor.py#L0-L199)

**Section sources**
- [core/snake_agent_enhanced.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_agent_enhanced.py)

## Core Components
The Enhanced Snake Agent consists of several core components that work together to enable autonomous code improvement:

- **EnhancedSnakeAgent**: The main orchestrator that initializes and coordinates all subsystems.
- **SnakeAgentConfiguration**: Central configuration object defining threading, process, and performance parameters.
- **SnakeLogManager**: Thread-safe logging system with structured JSON output and separate log files for different activities.
- **SnakeThreadingManager**: Manages concurrent threads for I/O-bound tasks like file monitoring and communication.
- **SnakeProcessManager**: Handles CPU-intensive operations such as code analysis and experiments using multiprocessing.
- **ContinuousFileMonitor**: Real-time file system watcher that detects changes in the codebase and triggers analysis workflows.

These components are initialized in sequence, with proper error handling and logging throughout the startup process.

**Section sources**
- [core/snake_agent_enhanced.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_agent_enhanced.py)
- [core/snake_data_models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_data_models.py#L375-L412)
- [core/snake_log_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_log_manager.py#L105-L371)
- [core/snake_threading_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_threading_manager.py#L0-L199)
- [core/snake_process_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_process_manager.py#L0-L199)
- [core/snake_file_monitor.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_file_monitor.py#L0-L199)

## Architecture Overview
The Enhanced Snake Agent follows a layered architecture with clear separation between coordination, processing, and monitoring layers. The system uses a producer-consumer pattern across both threads and processes to ensure efficient resource utilization.

```mermaid
graph TD
A[RAVANA Codebase] --> |File Changes| B(ContinuousFileMonitor)
B --> |FileChangeEvent| C(SnakeThreadingManager)
C --> |AnalysisTask| D{Task Priority}
D --> |High/Critical| E(SnakeProcessManager)
D --> |Medium/Low| F[Threaded Analysis]
E --> |ExperimentTask| G[Process Workers]
G --> |Results| E
E --> |ImprovementTask| H[Improvement Process]
H --> |Code Changes| A
C --> |Logging| I(SnakeLogManager)
E --> |Logging| I
B --> |Logging| I
F --> |Logging| I
G --> |Logging| I
H --> |Logging| I
J[EnhancedSnakeAgent] --> |Orchestrates| B
J --> |Orchestrates| C
J --> |Orchestrates| E
J --> |Monitors| I
style A fill:#E1F5FE,stroke:#333
style B fill:#B3E5FC,stroke:#333
style C fill:#81D4FA,stroke:#333
style D fill:#4FC3F7,stroke:#333
style E fill:#29B6F6,stroke:#333
style F fill:#03A9F4,stroke:#333
style G fill:#039BE5,stroke:#333
style H fill:#0288D1,stroke:#333
style I fill:#0277BD,stroke:#333
style J fill:#01579B,stroke:#333,color:white
```

**Diagram sources**
- [core/snake_agent_enhanced.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_agent_enhanced.py)
- [core/snake_file_monitor.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_file_monitor.py#L0-L199)
- [core/snake_threading_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_threading_manager.py#L0-L199)
- [core/snake_process_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_process_manager.py#L0-L199)
- [core/snake_log_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_log_manager.py#L105-L371)

## Detailed Component Analysis

### EnhancedSnakeAgent Analysis
The EnhancedSnakeAgent class serves as the central orchestrator for the entire system. It manages the lifecycle of all components and coordinates their interactions through a main coordination loop.

#### Class Diagram
```mermaid
classDiagram
class EnhancedSnakeAgent {
+agi_system : Any
+snake_config : SnakeAgentConfiguration
+log_manager : SnakeLogManager
+threading_manager : SnakeThreadingManager
+process_manager : SnakeProcessManager
+file_monitor : ContinuousFileMonitor
+running : bool
+initialized : bool
-_shutdown_event : asyncio.Event
-_coordination_lock : asyncio.Lock
+initialize() : bool
+start_autonomous_operation() : None
+stop() : None
+get_status() : Dict[str, Any]
-_setup_component_callbacks() : None
-_coordination_loop() : None
-_perform_health_check() : None
-_log_performance_metrics() : None
-_save_state() : None
-_load_state() : None
-_cleanup() : None
}
class SnakeAgentConfiguration {
+max_threads : int
+max_processes : int
+analysis_threads : int
+file_monitor_interval : float
+process_heartbeat_interval : float
+max_queue_size : int
+task_timeout : float
+cleanup_interval : float
+log_level : str
+enable_performance_monitoring : bool
+auto_recovery : bool
+to_dict() : Dict[str, Any]
+validate() : List[str]
}
class SnakeLogManager {
+start_log_processor() : None
+stop_log_processor(timeout : float) : None
+log_improvement(record : ImprovementRecord) : None
+log_experiment(record : ExperimentRecord) : None
+log_analysis(record : AnalysisRecord) : None
+log_communication(record : CommunicationRecord) : None
+log_system_event(event_type : str, data : Dict[str, Any], level : str, worker_id : str) : None
}
class SnakeThreadingManager {
+initialize() : bool
+start_all_threads() : bool
+start_file_monitor_thread() : bool
+start_analysis_threads(count : int) : bool
+start_communication_thread() : bool
+start_performance_monitor_thread() : bool
+queue_analysis_task(task : AnalysisTask) : None
+set_callbacks(file_change_callback : Callable, analysis_callback : Callable, communication_callback : Callable) : None
}
class SnakeProcessManager {
+initialize() : bool
+start_all_processes() : bool
+start_experiment_processes(count : int) : bool
+start_analysis_processes(count : int) : bool
+start_improvement_process() : bool
+distribute_task(task : Dict[str, Any]) : None
+set_callbacks(experiment_callback : Callable, analysis_callback : Callable, improvement_callback : Callable) : None
}
class ContinuousFileMonitor {
+initialize() : bool
+start_monitoring() : bool
+stop_monitoring() : None
+set_change_callback(callback : Callable) : None
+get_monitoring_status() : Dict[str, Any]
}
EnhancedSnakeAgent --> SnakeAgentConfiguration
EnhancedSnakeAgent --> SnakeLogManager
EnhancedSnakeAgent --> SnakeThreadingManager
EnhancedSnakeAgent --> SnakeProcessManager
EnhancedSnakeAgent --> ContinuousFileMonitor
```

**Diagram sources**
- [core/snake_agent_enhanced.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_agent_enhanced.py)
- [core/snake_data_models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_data_models.py#L375-L412)
- [core/snake_log_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_log_manager.py#L105-L371)
- [core/snake_threading_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_threading_manager.py#L0-L199)
- [core/snake_process_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_process_manager.py#L0-L199)
- [core/snake_file_monitor.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_file_monitor.py#L0-L199)

**Section sources**
- [core/snake_agent_enhanced.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_agent_enhanced.py)

### Initialization and Startup Flow
The Enhanced Snake Agent follows a strict initialization sequence to ensure all components are properly set up before starting autonomous operations.

#### Sequence Diagram
```mermaid
sequenceDiagram
participant ESA as EnhancedSnakeAgent
participant LM as SnakeLogManager
participant TM as SnakeThreadingManager
participant PM as SnakeProcessManager
participant FM as ContinuousFileMonitor
ESA->>ESA : initialize()
ESA->>SnakeAgentConfiguration : validate()
ESA->>LM : Create instance
LM->>LM : start_log_processor()
ESA->>ESA : Log initialization start
ESA->>ESA : Create LLM interfaces
ESA->>TM : Create instance
TM->>TM : initialize()
ESA->>PM : Create instance
PM->>PM : initialize()
PM->>PM : start_result_collector()
ESA->>FM : Create instance
FM->>FM : initialize()
FM->>FM : _initial_file_scan()
FM->>FM : Setup watchdog observer
ESA->>ESA : _setup_component_callbacks()
ESA->>ESA : _load_state()
ESA->>ESA : Set initialized = true
ESA->>ESA : Log initialization complete
ESA-->>ESA : Return True
```

**Diagram sources**
- [core/snake_agent_enhanced.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_agent_enhanced.py)
- [core/snake_log_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_log_manager.py#L105-L371)
- [core/snake_threading_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_threading_manager.py#L0-L199)
- [core/snake_process_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_process_manager.py#L0-L199)
- [core/snake_file_monitor.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_file_monitor.py#L0-L199)

**Section sources**
- [core/snake_agent_enhanced.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_agent_enhanced.py)

### Autonomous Operation Workflow
Once initialized, the Enhanced Snake Agent enters a continuous coordination loop that manages health checks, performance logging, and state persistence.

#### Flowchart
```mermaid
flowchart TD
Start([Start Autonomous Operation]) --> InitCheck{"Initialized?"}
InitCheck --> |No| Initialize[initialize()]
InitCheck --> |Yes| SetRunning[Set running = true]
SetRunning --> StartComponents[Start all components]
StartComponents --> CoordinationLoop[Enter coordination_loop()]
CoordinationLoop --> CheckHealth{"5-min interval?"}
CheckHealth --> |Yes| PerformHealth[perform_health_check()]
CheckHealth --> |No| CheckMetrics{"10-min interval?"}
CheckMetrics --> |Yes| LogMetrics[log_performance_metrics()]
CheckMetrics --> |No| SaveState[save_state()]
PerformHealth --> Wait
LogMetrics --> Wait
SaveState --> Wait
Wait[Wait 10 seconds] --> CoordinationLoop
CoordinationLoop --> |Shutdown requested| Cleanup[Run _cleanup()]
Cleanup --> StopComponents[Stop all components]
StopComponents --> SaveFinalState[_save_state()]
SaveFinalState --> StopLogger[Stop log_manager]
StopLogger --> End([Operation stopped])
style Start fill:#4ECDC4,stroke:#333
style End fill:#4ECDC4,stroke:#333
style InitCheck fill:#FFD700,stroke:#333
style Initialize fill:#FF6B6B,stroke:#333
style SetRunning fill:#4ECDC4,stroke:#333
style StartComponents fill:#4ECDC4,stroke:#333
style CoordinationLoop fill:#45B7D1,stroke:#333
style PerformHealth fill:#96CEB4,stroke:#333
style LogMetrics fill:#96CEB4,stroke:#333
style SaveState fill:#96CEB4,stroke:#333
style Wait fill:#DDA0DD,stroke:#333
style Cleanup fill:#4ECDC4,stroke:#333
style StopComponents fill:#4ECDC4,stroke:#333
style SaveFinalState fill:#4ECDC4,stroke:#333
style StopLogger fill:#4ECDC4,stroke:#333
```

**Diagram sources**
- [core/snake_agent_enhanced.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_agent_enhanced.py)

**Section sources**
- [core/snake_agent_enhanced.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_agent_enhanced.py)

## Dependency Analysis
The Enhanced Snake Agent has a well-defined dependency hierarchy where higher-level components depend on lower-level services. The system avoids circular dependencies through careful interface design and callback patterns.

```mermaid
graph TD
ESA[EnhancedSnakeAgent] --> LM[SnakeLogManager]
ESA --> TM[SnakeThreadingManager]
ESA --> PM[SnakeProcessManager]
ESA --> FM[ContinuousFileMonitor]
ESA --> Config[SnakeAgentConfiguration]
TM --> LM
TM --> Config
PM --> LM
PM --> Config
FM --> LM
FM --> Config
ESA -.->|Callbacks| TM
ESA -.->|Callbacks| PM
ESA -.->|Callbacks| FM
TM -.->|Callbacks| ESA
PM -.->|Callbacks| ESA
FM -.->|Callbacks| ESA
style ESA fill:#01579B,stroke:#333,color:white
style LM fill:#0277BD,stroke:#333,color:white
style TM fill:#0288D1,stroke:#333,color:white
style PM fill:#039BE5,stroke:#333,color:white
style FM fill:#03A9F4,stroke:#333,color:white
style Config fill:#29B6F6,stroke:#333,color:white
```

**Diagram sources**
- [core/snake_agent_enhanced.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_agent_enhanced.py)
- [core/snake_log_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_log_manager.py#L105-L371)
- [core/snake_threading_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_threading_manager.py#L0-L199)
- [core/snake_process_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_process_manager.py#L0-L199)
- [core/snake_file_monitor.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_file_monitor.py#L0-L199)
- [core/snake_data_models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_data_models.py#L375-L412)

**Section sources**
- [core/snake_agent_enhanced.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_agent_enhanced.py)

## Performance Considerations
The Enhanced Snake Agent is designed with performance and resource efficiency in mind:

- **Thread Management**: Configurable thread pool limits prevent resource exhaustion
- **Queue Sizing**: Bounded queues with configurable maximum size prevent memory leaks
- **File Monitoring**: Uses watchdog library with selective file extension monitoring to reduce I/O overhead
- **Logging**: Asynchronous, thread-safe logging with separate structured JSON files for efficient log processing
- **State Persistence**: Regular state saving enables recovery from crashes without losing progress metrics
- **Health Monitoring**: Periodic health checks and performance metrics logging help identify bottlenecks

The agent balances responsiveness with resource conservation by using appropriate concurrency models: threading for I/O-bound tasks and multiprocessing for CPU-intensive operations.

## Bot Connectivity Management and Status Reporting
The Enhanced Snake Agent integrates with the Conversational AI module to provide enhanced bot connectivity management and status reporting. This integration enables detailed monitoring of bot connection states and improved error handling.

### Bot Status Reporting
The system now provides comprehensive status reporting for bot connectivity through the `get_conversational_ai_status` method in the core system module. This method returns detailed information about the connection status of both Discord and Telegram bots.

```python
def get_conversational_ai_status(self) -> Dict[str, Any]:
    """Get Conversational AI status information."""
    if not self.conversational_ai:
        return {"enabled": False, "status": "not_initialized"}
    
    # Check if the conversational AI has been started
    started = getattr(self, '_conversational_ai_started', False)
    
    # Check if bots are connected
    discord_connected = False
    telegram_connected = False
    
    if self.conversational_ai.discord_bot:
        discord_connected = getattr(self.conversational_ai.discord_bot, 'connected', False)
        
    if self.conversational_ai.telegram_bot:
        telegram_connected = getattr(self.conversational_ai.telegram_bot, 'connected', False)
    
    # Determine overall status
    bot_connected = discord_connected or telegram_connected
    status = "active" if (started and bot_connected) else "inactive"
    
    return {
        "enabled": Config.CONVERSATIONAL_AI_ENABLED,
        "status": status,
        "discord_connected": discord_connected,
        "telegram_connected": telegram_connected
    }
```

### Bot Connectivity Implementation
The bot connectivity is implemented through dedicated classes for each platform:

- **DiscordBot**: Manages connection to Discord using the discord.py library
- **TelegramBot**: Manages connection to Telegram using the python-telegram-bot library

Both implementations include:
- Connection state tracking via the `connected` property
- Graceful shutdown handling with asyncio events
- Error handling and retry mechanisms
- Message sending with proper error handling

**Section sources**
- [core/system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py#L593-L624) - *Updated status reporting*
- [modules/conversational_ai/main.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\main.py) - *Bot connectivity management*
- [modules/conversational_ai/bots/discord_bot.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\bots\discord_bot.py) - *Discord bot implementation*
- [modules/conversational_ai/bots/telegram_bot.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\bots\telegram_bot.py) - *Telegram bot implementation*

## Troubleshooting Guide
Common issues and their solutions when working with the Enhanced Snake Agent:

**Section sources**
- [core/snake_agent_enhanced.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_agent_enhanced.py)
- [core/snake_log_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_log_manager.py#L105-L371)
- [core/snake_threading_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_threading_manager.py#L0-L199)

### Initialization Failures
If the agent fails to initialize:
1. Check `snake_logs/system.log` for error messages
2. Verify that the `snake_logs` directory is writable
3. Ensure required environment variables are set (SNAKE_MAX_THREADS, etc.)
4. Validate configuration parameters using `SnakeAgentConfiguration.validate()`

### File Monitoring Not Working
If file changes are not being detected:
1. Verify that file extensions are in the monitored list (`.py`, `.json`, etc.)
2. Check that directories are not excluded (`.git`, `.venv`, etc.)
3. Ensure the watchdog library is properly installed
4. Review `snake_logs/system.log` for file monitor initialization errors

### High Resource Usage
If CPU or memory usage is excessive:
1. Reduce `max_threads` and `max_processes` in configuration
2. Increase `file_monitor_interval` to reduce polling frequency
3. Check for runaway processes in the process manager
4. Review log files for repeated error loops

### Stuck Tasks
If tasks are not being processed:
1. Check queue sizes using `get_queue_status()` methods
2. Verify that worker threads and processes are active
3. Look for exceptions in log files that might be preventing task completion
4. Ensure the coordination loop is running and not blocked

### Bot Connectivity Issues
If bot connectivity is failing:
1. Check `conversational_ai.log` for connection errors
2. Verify bot tokens are correctly configured in `modules/conversational_ai/config.json`
3. Ensure the bot is enabled in the configuration file
4. Check network connectivity to Discord/Telegram servers
5. Review the bot status using the `get_conversational_ai_status` method

## Conclusion
The Enhanced Snake Agent represents a sophisticated autonomous system for continuous code improvement. Its modular architecture, combining threading and multiprocessing, allows it to efficiently handle both I/O-bound and CPU-intensive tasks. The agent's comprehensive logging, error handling, and state persistence make it robust and reliable for long-running operations. By following the principles of separation of concerns and clear component interfaces, the system remains maintainable and extensible. The implementation demonstrates best practices in concurrent programming, resource management, and system monitoring, making it a powerful tool for automated code enhancement within the RAVANA ecosystem.

**Referenced Files in This Document**   
- [core/snake_agent_enhanced.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_agent_enhanced.py) - *Updated in recent commit d6c6f5d*
- [core/system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py#L593-L624) - *Updated status reporting*
- [modules/conversational_ai/main.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\main.py) - *Bot connectivity management*
- [modules/conversational_ai/bots/discord_bot.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\bots\discord_bot.py) - *Discord bot implementation*
- [modules/conversational_ai/bots/telegram_bot.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\bots\telegram_bot.py) - *Telegram bot implementation*
- [core/snake_data_models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_data_models.py#L375-L412)
- [core/snake_log_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_log_manager.py#L105-L371)
- [core/snake_threading_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_threading_manager.py#L0-L199)
- [core/snake_process_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_process_manager.py#L0-L199)
- [core/snake_file_monitor.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\snake_file_monitor.py#L0-L199)
