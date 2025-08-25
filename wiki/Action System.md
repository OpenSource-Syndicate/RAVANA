# Action System



## Update Summary
**Changes Made**   
- Updated Coding Actions section to reflect enhanced code generation prompt with structured role, task instructions, reasoning framework, output requirements, and safety constraints
- Added new section on Prompt Management System to explain centralized prompt handling
- Enhanced security considerations with additional safety constraints from the enhanced prompt
- Updated referenced files list to include prompt_manager.py, code_generation.json, and llm.py
- Added details about dynamic prompt enhancement with mood adaptation and safety constraints

## Table of Contents
1. [Introduction](#introduction)
2. [Action Base Class and Execute Contract](#action-base-class-and-execute-contract)
3. [Action Registry and Discovery Mechanism](#action-registry-and-discovery-mechanism)
4. [Action Manager Execution Lifecycle](#action-manager-execution-lifecycle)
5. [Built-in Action Types and Use Cases](#built-in-action-types-and-use-cases)
6. [Prompt Management System](#prompt-management-system)
7. [Defining and Registering Custom Actions](#defining-and-registering-custom-actions)
8. [System State and Service Interaction](#system-state-and-service-interaction)
9. [Security Considerations for Action Execution](#security-considerations-for-action-execution)
10. [Troubleshooting Common Issues](#troubleshooting-common-issues)
11. [Conclusion](#conclusion)

## Introduction
The Action System is a core component of the Ravana AGI framework, responsible for executing tasks based on decisions made by the decision engine. It provides a structured, extensible mechanism for defining, registering, and executing actions that the AGI can perform. The system is designed with modularity, safety, and scalability in mind, enabling both built-in and custom actions to be seamlessly integrated. This document provides a comprehensive overview of the action system's architecture, functionality, and best practices.

## Action Base Class and Execute Contract

The `Action` class serves as the abstract base class for all executable actions within the system. It defines a standardized interface that ensures consistency across different types of actions.

```
classDiagram
class Action {
<<abstract>>
+system : AGISystem
+data_service : DataService
+name : str
+description : str
+parameters : List[Dict[str, Any]]
+execute(**kwargs : Any) : Any
+validate_params(params : Dict[str, Any]) : None
+to_dict() : Dict[str, Any]
+to_json() : str
}
Action <|-- WritePythonCodeAction
Action <|-- ExecutePythonFileAction
Action <|-- LogMessageAction
Action <|-- ProposeAndTestInventionAction
Action <|-- ProcessImageAction
Action <|-- BlogPublishAction
Action <|-- CollaborativeTaskAction
```

**Diagram sources**
- [action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L1-L62)

### Key Properties
- **name**: A unique identifier for the action (e.g., "write_python_code")
- **description**: Human-readable explanation of what the action does
- **parameters**: List of dictionaries defining input parameters with name, type, description, and required status

### Execute Method Contract
The `execute` method is an abstract async method that must be implemented by all subclasses. It receives keyword arguments matching the defined parameters and returns the result of the action execution. The method contract requires:
- Parameter validation via `validate_params` before execution
- Asynchronous execution using `async/await` pattern
- Proper error handling and logging
- Return of a dictionary with status information or execution results

### Validation and Serialization
The base class provides built-in validation through `validate_params`, which checks for missing required parameters and unexpected parameters. It also includes `to_dict` and `to_json` methods for serializing action metadata, which is used in LLM prompts and API responses.

**Section sources**
- [action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L1-L62)

## Action Registry and Discovery Mechanism

The `ActionRegistry` is responsible for managing the collection of available actions and providing lookup functionality.

```
classDiagram
class ActionRegistry {
-actions : Dict[str, Action]
+__init__(system : AGISystem, data_service : DataService)
+_register_action(action : Action) : None
+register_action(action : Action) : None
+discover_actions() : None
+get_action(name : str) : Action
+get_all_actions() : List[Action]
+get_action_definitions() : str
}
ActionRegistry --> Action : "contains"
```

**Diagram sources**
- [registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L1-L74)

### Registration Process
The registry initializes with several built-in actions:
- `ProposeAndTestInventionAction`
- `LogMessageAction`
- `WritePythonCodeAction`
- `ExecutePythonFileAction`
- `BlogPublishAction`
- `CollaborativeTaskAction`

These are registered during initialization with their required dependencies (system and data_service). The `CollaborativeTaskAction` is now included as a core built-in action, reflecting its importance in the cross-system task delegation workflow.

### Public Registration Methods
- **register_action**: Public method to register a new action instance
- **_register_action**: Internal method that handles registration with overwrite warnings

### Automatic Discovery
The `discover_actions` method uses Python's `pkgutil.walk_packages` to automatically discover and register all action classes in the `core.actions` package. It:
1. Iterates through all modules in the actions package
2. Imports each module
3. Finds all classes that inherit from `Action` (excluding the base class itself)
4. Instantiates and registers each action
5. Logs warnings for duplicate names and errors for instantiation failures

This discovery mechanism enables plug-and-play extensibility - new actions can be added by simply creating a new file in the actions directory with a properly defined action class.

**Section sources**
- [registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L1-L74)

## Action Manager Execution Lifecycle

The `ActionManager` orchestrates the execution of actions, handling the complete lifecycle from decision parsing to result return.

```
sequenceDiagram
participant DecisionEngine
participant ActionManager
participant ActionRegistry
participant Action
participant DataService
DecisionEngine->>ActionManager : execute_action(decision)
ActionManager->>ActionManager : Parse decision format
alt Raw LLM Response
ActionManager->>ActionManager : Extract JSON from response
ActionManager->>ActionManager : Parse JSON to action_data
else Pre-parsed Action
ActionManager->>ActionManager : Use action_data directly
end
ActionManager->>ActionManager : Validate action_data structure
ActionManager->>ActionRegistry : get_action(action_name)
ActionRegistry-->>ActionManager : Return Action instance
ActionManager->>Action : execute(**action_params)
Action-->>ActionManager : Return execution result
ActionManager->>DataService : save_action_log(success)
ActionManager-->>DecisionEngine : Return result
alt Execution Error
Action-->>ActionManager : Raise exception
ActionManager->>DataService : save_action_log(error)
ActionManager-->>DecisionEngine : Return error object
end
```

**Diagram sources**
- [action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\action_manager.py#L1-L126)

### Decision Parsing
The `execute_action` method handles two decision formats:
1. **Pre-parsed action dictionary**: Contains "action" and "params" keys
2. **Raw LLM response**: Contains "raw_response" with JSON embedded in markdown code blocks

For raw responses, the system extracts the JSON block using string manipulation and parses it. If no valid JSON block is found, it attempts to parse the entire response as JSON.

### Execution Flow
1. Extract action name and parameters from the decision
2. Retrieve the action instance from the registry
3. Log the execution attempt
4. Execute the action with provided parameters
5. Log successful execution to the database
6. Return the result

### Error Handling
The execution lifecycle includes comprehensive error handling:
- **ActionException**: For expected action-related errors (logged with error status)
- **General Exception**: For unexpected errors (logged with full traceback)
- All errors are logged to the database via `save_action_log`

The method returns a standardized error object with an "error" key, ensuring consistent error reporting to the decision engine.

**Section sources**
- [action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\action_manager.py#L1-L126)

## Built-in Action Types and Use Cases

The system provides several built-in action types categorized by functionality.

### Coding Actions
These actions enable the AGI to generate and execute code.

#### WritePythonCodeAction
Generates Python code based on a hypothesis and test plan using the LLM with an enhanced prompt structure.

**Parameters:**
- **file_path**: Where to save the generated code
- **hypothesis**: The concept to test
- **test_plan**: How to test the hypothesis

The action uses a structured prompt template with multiple sections to guide code generation:
- **[ROLE DEFINITION]**: Defines the AI as an expert programmer
- **[CONTEXT]**: Provides the hypothesis and test plan
- **[TASK INSTRUCTIONS]**: Step-by-step process for code generation
- **[REASONING FRAMEWORK]**: Software engineering best practices
- **[OUTPUT REQUIREMENTS]**: Specifications for code quality and format
- **[SAFETY CONSTRAINTS]**: Security and reliability guidelines

The prompt ensures high-quality code generation by requiring:
- Clear, descriptive variable and function names
- Comprehensive inline documentation
- Proper error handling and edge case management
- Efficient algorithms and data structures
- Adherence to Python conventions and best practices
- Confidence score for solution correctness (0.0-1.0)

The action extracts code from markdown code blocks in the LLM response and writes it to the specified file.

#### ExecutePythonFileAction
Executes a Python script and captures its output.

**Parameters:**
- **file_path**: Path to the Python script

Uses `asyncio.create_subprocess_shell` to run the script, capturing stdout and stderr. Returns execution results including return code and output.

**Section sources**
- [coding.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\coding.py#L1-L114)
- [code_generation.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\prompts\code_generation.json#L1-L12)
- [prompt_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\prompt_manager.py#L1-L538)
- [llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py#L1-L1538)

### IO Actions
#### LogMessageAction
Records messages to the console with configurable logging levels.

**Parameters:**
- **message**: Content to log
- **level**: Logging level (info, warning, error)

Appends "[AGI Thought]:" prefix to messages for easy identification in logs.

**Section sources**
- [io.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\io.py#L1-L39)

### Multi-modal Actions
These actions process various media types and are registered by the `EnhancedActionManager`.

#### ProcessImageAction
Analyzes image files using multi-modal services.

#### ProcessAudioAction
Processes and analyzes audio files.

#### AnalyzeDirectoryAction
Analyzes all media files in a directory, optionally recursively.

#### CrossModalAnalysisAction
Performs analysis across multiple content types.

These actions validate file existence before processing and can add results to the knowledge base.

**Section sources**
- [multi_modal.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\multi_modal.py#L1-L78)
- [enhanced_action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\enhanced_action_manager.py#L1-L199)

### Experimental Actions
#### ProposeAndTestInventionAction
Enables the AGI to propose novel ideas and test them through the experimentation engine.

**Parameters:**
- **invention_description**: The novel concept
- **test_plan_suggestion**: How to test it

Frames the invention as a hypothesis and runs it through the advanced experimentation engine, logging results to the database. This action represents the AGI's creative and scientific reasoning capabilities.

**Section sources**
- [experimental.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\experimental.py#L1-L129)

### Blog Actions
New blog-specific actions have been added to support autonomous content creation and publishing.

#### BlogPublishAction
Orchestrates the complete blog publishing workflow, from content generation to platform publication.

```
classDiagram
class BlogPublishAction {
+api_interface : BlogAPIInterface
+content_generator : BlogContentGenerator
+name : str
+description : str
+parameters : List[Dict[str, Any]]
+execute(**kwargs : Any) : Dict[str, Any]
+_log_blog_action(result : Dict[str, Any], action_type : str) : None
+_format_memory_content(result : Dict[str, Any], action_type : str) : str
+test_connection() : Dict[str, Any]
+_calculate_emotional_valence(result : Dict[str, Any], action_type : str) : float
+get_configuration_info() : Dict[str, Any]
}
BlogPublishAction --|> Action : "inherits"
BlogPublishAction --> BlogAPIInterface : "uses"
BlogPublishAction --> BlogContentGenerator : "uses"
```

**Diagram sources**
- [blog.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\blog.py#L18-L382)
- [blog_api.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\blog_api.py)
- [blog_content_generator.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\blog_content_generator.py)

**Parameters:**
- **topic**: Main subject for the blog post (required)
- **style**: Writing style from available options (optional)
- **context**: Additional aspects to focus on (optional)
- **custom_tags**: Tags to include beyond auto-generated ones (optional)
- **dry_run**: If true, generates content without publishing (optional)

The action follows a comprehensive workflow:
1. Validates configuration and parameters
2. Generates content using LLM with memory context
3. Validates API configuration
4. Publishes to the blog platform
5. Logs results to data and memory services

The action includes comprehensive error handling for content generation failures, API errors, and unexpected exceptions. It also provides utility methods like `test_connection()` for API connectivity testing and `get_configuration_info()` for retrieving current blog settings.

**Section sources**
- [blog.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\blog.py#L18-L382)
- [blog_api.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\blog_api.py)
- [blog_content_generator.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\blog_content_generator.py)

### Collaborative Task Actions
New collaborative task actions have been added to support cross-system task delegation and user collaboration.

#### CollaborativeTaskAction
Manages collaborative tasks between RAVANA and users with feedback mechanisms.

```
classDiagram
class CollaborativeTaskAction {
+system : AGISystem
+data_service : DataService
+collaborative_tasks : Dict[str, Any]
+task_feedback : Dict[str, Any]
+name : str
+description : str
+parameters : List[Dict[str, Any]]
+execute(**kwargs : Any) : Dict[str, Any]
+_create_task(title : str, description : str, user_id : str, priority : str, deadline : str) : Dict[str, Any]
+_update_task(task_id : str, title : str, description : str, priority : str, deadline : str) : Dict[str, Any]
+_complete_task(task_id : str) : Dict[str, Any]
+_cancel_task(task_id : str) : Dict[str, Any]
+_provide_feedback(task_id : str, user_id : str, feedback : str, feedback_type : str, rating : int) : Dict[str, Any]
+_request_feedback(task_id : str, user_id : str) : Dict[str, Any]
+get_task(task_id : str) : Dict[str, Any]
+get_user_tasks(user_id : str) : list
+get_task_feedback(task_id : str) : List[Dict[str, Any]]
+get_user_feedback(user_id : str) : List[Dict[str, Any]]
+analyze_collaboration_patterns() : Dict[str, Any]
+_get_most_active_users() : Dict[str, int]
}
CollaborativeTaskAction --|> Action : "inherits"
```

**Diagram sources**
- [collaborative_task.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\collaborative_task.py#L1-L553)

**Parameters:**
- **task_type**: Type of collaborative task (create, update, complete, cancel, provide_feedback, request_feedback)
- **task_id**: Unique identifier for the task (required for update, complete, cancel, provide_feedback)
- **title**: Title of the task (required for create)
- **description**: Detailed description of the task
- **user_id**: User ID to collaborate with
- **priority**: Priority level (low, medium, high, critical)
- **deadline**: Deadline for task completion (ISO format)
- **feedback**: Feedback content (required for provide_feedback)
- **feedback_type**: Type of feedback (positive, negative, suggestion, question)
- **rating**: Numerical rating for the task (1-10)

The action supports a comprehensive workflow for collaborative task management:
1. **Create tasks**: Initiates new collaborative tasks with users
2. **Update tasks**: Modifies existing task details
3. **Complete tasks**: Marks tasks as completed with user notification
4. **Cancel tasks**: Cancels tasks with user notification
5. **Provide feedback**: Collects user feedback on task performance
6. **Request feedback**: Proactively requests feedback from users

The action integrates with the conversational AI system to send notifications and messages to users at key points in the task lifecycle. It maintains in-memory storage for tasks and feedback, with comprehensive logging and history tracking for all collaboration events.

**Section sources**
- [collaborative_task.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\collaborative_task.py#L1-L553)

## Prompt Management System

The system now features a centralized PromptManager that handles all prompt templates and their dynamic enhancement.

```
classDiagram
class PromptManager {
+agi_system : AGISystem
+repository : PromptRepository
+enhancer : PromptEnhancer
+__init__(agi_system : AGISystem)
+_register_default_templates() : None
+get_prompt(template_name : str, context : Dict[str, Any]) : str
+register_prompt_template(name : str, template : str, metadata : Dict[str, Any]) : None
+_post_process_prompt(prompt : str, context : Dict[str, Any]) : str
}
PromptManager --> PromptRepository : "uses"
PromptManager --> PromptEnhancer : "uses"
```

**Diagram sources**
- [prompt_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\prompt_manager.py#L1-L538)

### Centralized Prompt Repository
The PromptManager uses a repository pattern to store and retrieve prompt templates from JSON files in the prompts directory. Each template includes:
- **name**: Unique identifier for the prompt
- **template**: The actual prompt text with placeholders
- **metadata**: Additional information about category, description, and version
- **version**: Version tracking for prompt evolution
- **created_at/updated_at**: Timestamps for version history

### Enhanced Prompt Structure
The coding action now uses a structured prompt with multiple sections that guide the LLM's response:
- **[ROLE DEFINITION]**: Establishes the AI's identity and expertise
- **[CONTEXT]**: Provides specific task details and parameters
- **[TASK INSTRUCTIONS]**: Step-by-step process for completing the task
- **[REASONING FRAMEWORK]**: Methodological approach to problem-solving
- **[OUTPUT REQUIREMENTS]**: Specific format and quality expectations
- **[SAFETY CONSTRAINTS]**: Security and reliability guidelines

### Dynamic Prompt Enhancement
The system applies dynamic enhancements to prompts through post-processing:
- **Mood adaptation**: Adjusts prompt tone based on the AI's emotional state
- **Safety constraints**: Adds context-specific safety requirements
- **Confidence scoring**: Requires the LLM to include confidence scores
- **Risk assessment**: Adds requirements for risk identification and mitigation

### Template Registration and Retrieval
The system automatically loads all JSON files in the prompts directory as templates. Templates can be retrieved by name with context variables that are automatically substituted. The system validates prompts for required sections before use.

**Section sources**
- [prompt_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\prompt_manager.py#L1-L538)
- [code_generation.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\prompts\code_generation.json#L1-L12)

## Defining and Registering Custom Actions

Creating custom actions follows a straightforward process outlined in the developer guide.

### Action Structure Requirements
All custom actions must:
1. Inherit from the `Action` base class
2. Implement the required properties (`name`, `description`, `parameters`)
3. Implement the `execute` method with async functionality

### Step-by-Step Implementation
1. **Create a Python file** in `core/actions/` (e.g., `core/actions/misc.py`)
2. **Define the action class** inheriting from `Action`
3. **Implement required properties** with appropriate metadata
4. **Implement the execute method** with the desired functionality

```python
from core.actions.action import Action
from typing import Any, Dict, List

class HelloWorldAction(Action):
    @property
    def name(self) -> str:
        return "hello_world"

    @property
    def description(self) -> str:
        return "A simple action that prints a greeting."

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "name",
                "type": "string",
                "description": "The name to include in the greeting.",
                "required": True,
            }
        ]

    async def execute(self, **kwargs: Any) -> Any:
        name = kwargs.get("name")
        return f"Hello, {name}!"
```

### Registration Mechanism
Custom actions are automatically discovered and registered when:
- The action class is defined in a module within `core.actions`
- The class inherits from `Action` and is not the base class itself
- The module is importable

No manual registration is required - the `discover_actions` method will automatically find and instantiate the action during system initialization.

**Section sources**
- [DEVELOPER_GUIDE.md](file://c:\Users\ASUS\Documents\GitHub\RAVANA\DEVELOPER_GUIDE.md#L175-L216)
- [action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L1-L62)
- [registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L1-L74)

## System State and Service Interaction

Actions interact with system state and services through dependencies injected during initialization.

### Dependency Injection
The `Action` base class constructor accepts two dependencies:
- **system**: Reference to the main `AGISystem` instance
- **data_service**: Reference to the `DataService` for database operations

These dependencies are passed down from the `ActionManager` through the `ActionRegistry` to all action instances.

### State Access Patterns
Actions access system state through the injected dependencies:
- **AGISystem**: Provides access to other system components like `knowledge_service`
- **DataService**: Enables database operations like logging actions and experiments

For example, the `ProposeAndTestInventionAction` uses `self.data_service.save_experiment_log` to record experiment results, while multi-modal actions use `self.system.knowledge_service.add_knowledge` to update the knowledge base.

### Service Integration
The enhanced action manager demonstrates deeper service integration by:
- Creating a `MultiModalService` instance for media processing
- Using `asyncio.to_thread` to call synchronous service methods without blocking the event loop
- Implementing caching mechanisms to improve performance

This pattern ensures that actions can leverage system services while maintaining proper separation of concerns.

**Section sources**
- [action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L1-L62)
- [action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\action_manager.py#L1-L126)
- [enhanced_action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\enhanced_action_manager.py#L1-L199)

## Security Considerations for Action Execution

The action system incorporates several security measures to prevent misuse and ensure safe execution.

### Code Execution Safety
The `ExecutePythonFileAction` executes code in subprocesses, which provides isolation from the main application process. However, this still represents a potential security risk as executed code has full system access.

The `EnhancedActionManager` implements a 5-minute timeout for action execution using `asyncio.wait_for`, preventing infinite loops or long-running processes from blocking the system.

### Enhanced Safety Constraints
The enhanced code generation prompt includes comprehensive safety constraints:
- **Security vulnerabilities**: Prevents injection, buffer overflows, and other vulnerabilities
- **Resource leaks**: Ensures proper resource management and memory handling
- **Unintended actions**: Validates that code performs only intended operations
- **Input/output validation**: Requires validation of all inputs and outputs
- **Secure coding practices**: Enforces adherence to security best practices

These constraints are embedded in the prompt structure and are applied to all code generation requests.

### Blog-Specific Security
The `BlogPublishAction` introduces new security considerations:
- **API Key Management**: The action uses the enhanced Gemini API key management system with automatic rotation
- **Content Validation**: Generated content is validated before publication
- **Dry Run Mode**: Allows content generation without actual publication for review
- **Configuration Validation**: API configuration is validated before any publication attempts

### Collaboration-Specific Security
The `CollaborativeTaskAction` introduces new security considerations:
- **User ID Validation**: Ensures user IDs are properly validated before task creation
- **Rating Validation**: Validates that ratings are within the 1-10 range
- **Feedback Sanitization**: Should implement input sanitization for feedback content
- **Task Ownership**: Ensures users can only modify tasks they own

### Input Validation
The base `Action` class includes `validate_params` which:
- Checks for missing required parameters
- Rejects unexpected parameters
- Raises `InvalidActionParams` for invalid inputs

This prevents actions from executing with incomplete or malformed data.

### Permission and Access Control
Currently, the system lacks explicit permission controls. Actions have access to:
- Full file system (via file paths)
- Database operations
- System process execution

For production use, additional security layers should be implemented, such as:
- Sandboxed execution environments
- File system access restrictions
- Rate limiting for resource-intensive actions
- Action-specific permission policies

### Self-Modification Risks
The system allows actions to write and execute code, creating potential self-modification capabilities. While this enables powerful autonomous behavior, it also introduces risks:
- Code generation errors could corrupt system functionality
- Malicious LLM output could introduce harmful code
- Recursive self-modification could lead to instability

Mitigation strategies include:
- Code review before execution (not currently implemented)
- Backup and rollback mechanisms
- Execution in isolated environments
- Static analysis of generated code

**Section sources**
- [action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\action_manager.py#L1-L126)
- [enhanced_action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\enhanced_action_manager.py#L1-L199)
- [coding.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\coding.py#L1-L114)
- [blog.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\blog.py#L18-L382)
- [collaborative_task.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\collaborative_task.py#L1-L553)
- [prompt_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\prompt_manager.py#L1-L538)

## Troubleshooting Common Issues

### Registration Failures
**Symptoms:** Action not found in registry, `ActionException` with "Action 'action_name' not found"
**Causes and Solutions:**
- **Class not discovered**: Ensure the action class is in a module within `core.actions` and properly imports the `Action` base class
- **Instantiation error**: Check for exceptions in the `__init__` method; the discovery mechanism logs instantiation errors
- **Name collision**: The registry overwrites actions with duplicate names; check logs for overwrite warnings
- **Missing required properties**: Verify implementation of `name`, `description`, and `parameters` properties

### Execution Timeouts
**Symptoms:** Actions failing with "Action timed out" message
**Causes and Solutions:**
- **Long-running operations**: The `EnhancedActionManager` enforces a 5-minute timeout; optimize action logic or increase timeout if appropriate
- **Blocking operations**: Ensure all I/O operations use async patterns; use `asyncio.to_thread` for synchronous calls
- **Resource constraints**: Check system resources (CPU, memory, disk I/O) that may slow execution

### Permission Issues
**Symptoms:** File not found, permission denied, or subprocess execution failures
**Causes and Solutions:**
- **File access**: Verify the application has read/write permissions to specified file paths
- **Directory existence**: Check that directories exist before writing files
- **Subprocess execution**: Ensure Python is in the system PATH and the executing user has permission to run subprocesses
- **Absolute vs relative paths**: Use absolute paths or ensure relative paths are correct relative to the working directory

### Parameter Validation Errors
**Symptoms:** `InvalidActionParams` exceptions with messages about missing or unexpected parameters
**Causes and Solutions:**
- **Missing required parameters**: Ensure all parameters marked as `required: True` are provided
- **Typos in parameter names**: Double-check parameter names match exactly
- **Extra parameters**: Remove parameters not defined in the action's `parameters` list
- **Type mismatches**: Ensure parameter values match the expected types

### General Debugging Tips
- Check application logs for detailed error messages and stack traces
- Verify action registration by checking the "Available Actions" log output during startup
- Use the `get_action_definitions` method to inspect registered action schemas
- Test actions in isolation when possible
- Monitor the action log database table for execution history and errors

**Section sources**
- [action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\action_manager.py#L1-L126)
- [action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L1-L62)
- [exceptions.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\exceptions.py#L1-L14)
- [enhanced_action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\enhanced_action_manager.py#L1-L199)

## Conclusion
The Action System in the Ravana AGI framework provides a robust, extensible foundation for autonomous behavior. By defining a clear contract through the `Action` base class, implementing a flexible registration mechanism in `ActionRegistry`, and orchestrating execution through `ActionManager`, the system enables both built-in and custom actions to be seamlessly integrated. The architecture supports various action types including coding, IO, multi-modal, experimental, blog-specific, and now collaborative task actions, allowing the AGI to perform diverse tasks. The recent addition of `CollaborativeTaskAction` enhances the system's cross-system task delegation capabilities, enabling end-to-end collaborative workflows with user feedback mechanisms. The enhanced prompt management system with structured templates and dynamic enhancement significantly improves code generation quality and safety. While the system includes basic security measures like input validation and execution timeouts, additional safeguards would be beneficial for production deployment, particularly around code execution and permission controls. The automatic discovery mechanism and clear implementation guidelines make it straightforward to extend the AGI's capabilities with new actions, supporting the system's goal of continuous learning and adaptation.

**Referenced Files in This Document**   
- [action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py) - *Base Action class implementation*
- [registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py) - *ActionRegistry with CollaborativeTaskAction registration*
- [action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\action_manager.py)
- [coding.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\coding.py) - *Enhanced code generation prompt implementation*
- [io.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\io.py)
- [multi_modal.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\multi_modal.py)
- [experimental.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\experimental.py)
- [exceptions.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\exceptions.py)
- [enhanced_action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\enhanced_action_manager.py)
- [blog.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\blog.py) - *Added in recent commit*
- [blog_api.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\blog_api.py) - *Dependency for BlogPublishAction*
- [blog_content_generator.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\blog_content_generator.py) - *Dependency for BlogPublishAction*
- [collaborative_task.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\collaborative_task.py) - *Added in recent commit*
- [DEVELOPER_GUIDE.md](file://c:\Users\ASUS\Documents\GitHub\RAVANA\DEVELOPER_GUIDE.md)
- [prompt_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\prompt_manager.py) - *Prompt management system with enhanced templates*
- [code_generation.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\prompts\code_generation.json) - *Enhanced code generation prompt template*
- [llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py) - *LLM integration and call handling*