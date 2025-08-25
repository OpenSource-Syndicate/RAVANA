# Action Registry



## Update Summary
**Changes Made**   
- Added documentation for the newly integrated CollaborativeTaskAction
- Updated the list of built-in actions to include CollaborativeTaskAction
- Added section sources for newly referenced files
- Updated the Action Registry Overview section to reflect the new action registration
- Enhanced troubleshooting section with additional error scenarios

## Table of Contents
1. [Introduction](#introduction)
2. [Action Registry Overview](#action-registry-overview)
3. [Internal Mapping Structure](#internal-mapping-structure)
4. [Registration Mechanism and Validation](#registration-mechanism-and-validation)
5. [Automatic Action Discovery](#automatic-action-discovery)
6. [Integration with ActionManager](#integration-with-actionmanager)
7. [Registering Custom Actions](#registering-custom-actions)
8. [Troubleshooting Common Issues](#troubleshooting-common-issues)

## Introduction
The Action Registry is a central component of the Ravana AGI system responsible for managing all executable actions. It provides a dynamic registry that allows the system to discover, register, and retrieve actions at runtime. This document provides a comprehensive analysis of the ActionRegistry implementation, its integration with the broader system, and best practices for extending its functionality with custom actions.

## Action Registry Overview

The ActionRegistry serves as a centralized repository for all actions that the AGI can perform. It enables the system to dynamically discover and register action classes, making it easy to extend the AGI's capabilities without modifying core system components.

``mermaid
classDiagram
class ActionRegistry {
+actions : Dict[str, Action]
+__init__(system : AGISystem, data_service : DataService)
+_register_action(action : Action) : None
+register_action(action : Action) : None
+discover_actions() : None
+get_action(name : str) : Action
+get_all_actions() : List[Action]
+get_action_definitions() : str
}
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
ActionRegistry --> Action : "contains"
```

**Diagram sources**
- [core/actions/registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L16-L78)
- [core/actions/action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L6-L62)

**Section sources**
- [core/actions/registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L16-L78)
- [core/actions/action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L6-L62)

## Internal Mapping Structure

The ActionRegistry maintains an internal dictionary that maps action names to their corresponding callable class instances. This structure enables efficient lookup and retrieval of actions by name.

The primary data structure is defined as:

```python
self.actions: Dict[str, Action] = {}
```

This dictionary serves as the core registry, where:
- **Keys**: String identifiers representing the unique name of each action
- **Values**: Instantiated Action objects that can be executed

The mapping structure supports O(1) average-case time complexity for both registration and retrieval operations, making it highly efficient for the AGI system's needs.

``mermaid
flowchart TD
Start([Action Registration]) --> CheckName{"Action name exists?"}
CheckName --> |Yes| LogWarning["Log warning: Action will be overwritten"]
CheckName --> |No| Continue[Continue registration]
LogWarning --> Continue
Continue --> StoreAction["Store action in dictionary: actions[name] = action"]
StoreAction --> End([Registration Complete])
RetrieveStart([Action Retrieval]) --> FindAction["Look up action by name in dictionary"]
FindAction --> ActionExists{"Action exists?"}
ActionExists --> |Yes| ReturnAction["Return action instance"]
ActionExists --> |No| RaiseError["Raise ValueError: Action not found"]
ReturnAction --> End2([Retrieval Complete])
RaiseError --> End2
```

**Diagram sources**
- [core/actions/registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L20-L25)
- [core/actions/registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L45-L50)

**Section sources**
- [core/actions/registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L20-L25)

## Registration Mechanism and Validation

The ActionRegistry implements a robust registration mechanism that handles both explicit and automatic registration of action classes. The system includes validation processes to ensure action integrity and handle edge cases.

### Registration Methods

The registry provides two primary methods for registering actions:

1. **Direct Registration**: Using the `register_action()` method to register individual action instances
2. **Bulk Registration**: Using the `discover_actions()` method to automatically discover and register all actions in the core.actions package

The registration process includes validation for duplicate entries:

```python
def _register_action(self, action: Action) -> None:
    if action.name in self.actions:
        logger.warning(f"Action '{action.name}' is already registered. Overwriting.")
    self.actions[action.name] = action
```

When a duplicate action name is detected, the system logs a warning and overwrites the existing entry, ensuring that the most recent action definition takes precedence.

### Error Handling for Invalid Entries

The registry handles invalid entries through exception handling during the discovery process:

```python
try:
    instance = obj()
    if instance.name in self.actions:
        logger.warning(f"Action '{instance.name}' is already registered. Overwriting.")
    self.actions[instance.name] = instance
except Exception as e:
    logger.error(f"Failed to instantiate action {obj.__name__}: {e}", exc_info=True)
```

This try-except block ensures that failures in instantiating individual actions do not prevent the registration of other valid actions, maintaining system resilience.

``mermaid
sequenceDiagram
participant Client as "Action Client"
participant Registry as "ActionRegistry"
participant Action as "Action Instance"
Client->>Registry : register_action(action)
Registry->>Registry : Check if action.name exists
alt Action name already exists
Registry->>Registry : Log warning about overwrite
end
Registry->>Registry : Store action in actions dictionary
Registry-->>Client : Registration complete
Client->>Registry : discover_actions()
Registry->>Registry : Walk through core.actions package
Registry->>Registry : Import each module
Registry->>Registry : Find Action subclasses
loop For each Action class
Registry->>Registry : Try to instantiate class
alt Instantiation succeeds
Registry->>Registry : Register the action instance
else Instantiation fails
Registry->>Registry : Log error, continue with next class
end
end
Registry-->>Client : Discovery complete
```

**Diagram sources**
- [core/actions/registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L29-L36)
- [core/actions/registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L37-L56)

**Section sources**
- [core/actions/registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L29-L56)

## Automatic Action Discovery

The ActionRegistry implements an automatic discovery mechanism that scans the core.actions package to find and register all action classes without requiring explicit registration calls.

### Discovery Process

The `discover_actions()` method uses Python's introspection capabilities to automatically detect and register action classes:

```python
def discover_actions(self):
    """Discovers and registers all actions in the 'core.actions' package."""
    actions_package = core.actions
    for _, name, is_pkg in pkgutil.walk_packages(actions_package.__path__, actions_package.__name__ + '.'):
        if not is_pkg:
            module = importlib.import_module(name)
            for _, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, Action) and obj is not Action:
                    try:
                        instance = obj()
                        if instance.name in self.actions:
                            logger.warning(f"Action '{instance.name}' is already registered. Overwriting.")
                        self.actions[instance.name] = instance
                    except Exception as e:
                        logger.error(f"Failed to instantiate action {obj.__name__}: {e}", exc_info=True)
```

The discovery process follows these steps:
1. Traverse all modules in the core.actions package using `pkgutil.walk_packages`
2. Import each non-package module
3. Inspect the module to find all classes that inherit from the Action base class
4. Instantiate each action class and register it in the internal dictionary

### Integration with System Initialization

The automatic discovery is integrated into the system's initialization process through the ActionManager:

```python
class ActionManager:
    def __init__(self, system: 'AGISystem', data_service: 'DataService'):
        self.system = system
        self.data_service = data_service
        self.action_registry = ActionRegistry(system, data_service)
        logger.info(f"ActionManager initialized with {len(self.action_registry.actions)} actions.")
        self.log_available_actions()
```

During initialization, the ActionManager creates an ActionRegistry instance, which automatically registers several built-in actions in its constructor, and then the system can call `discover_actions()` to find additional actions.

``mermaid
flowchart TD
Start([System Startup]) --> CreateRegistry["Create ActionRegistry instance"]
CreateRegistry --> RegisterBuiltIn["Register built-in actions in constructor"]
RegisterBuiltIn --> CreateActionManager["Create ActionManager instance"]
CreateActionManager --> InitializeRegistry["Initialize with ActionRegistry"]
InitializeRegistry --> LogActions["Log available actions"]
LogActions --> CallDiscover["Call discover_actions()"]
CallDiscover --> FindModules["Find all modules in core.actions"]
FindModules --> ImportModule["Import each module"]
ImportModule --> FindClasses["Find Action subclasses"]
FindClasses --> Instantiate["Instantiate action classes"]
Instantiate --> RegisterActions["Register actions in dictionary"]
RegisterActions --> Complete["Discovery Complete"]
```

**Diagram sources**
- [core/actions/registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L37-L56)
- [core/action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\action_manager.py#L14-L20)

**Section sources**
- [core/actions/registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L37-L56)
- [core/action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\action_manager.py#L14-L20)

## Integration with ActionManager

The ActionRegistry is tightly integrated with the ActionManager, which serves as the interface between the decision-making components and the execution of actions.

### Execution Flow

When the DecisionEngine determines an action to perform, the ActionManager uses the ActionRegistry to retrieve and execute the appropriate action:

```python
async def execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
    # ... parsing logic ...
    
    try:
        action_name = action_data.get("action")
        action_params = action_data.get("params", {})

        if not action_name:
            logger.warning("No 'action' key found in the parsed JSON.")
            return {"error": "No action taken: 'action' key missing."}

        action = self.action_registry.get_action(action_name)
        if not action:
            raise ActionException(f"Action '{action_name}' not found.")

        logger.info(f"Executing action '{action_name}' with params: {action_params}")
        result = await action.execute(**action_params)
        # ... logging and return logic ...
```

### Enhanced Action Manager Integration

The EnhancedActionManager extends the base ActionManager to register additional multi-modal actions:

```python
class EnhancedActionManager(ActionManager):
    def __init__(self, agi_system, data_service):
        super().__init__(agi_system, data_service)
        self.multi_modal_service = MultiModalService()
        self.action_cache = {}
        self.parallel_limit = 3
        self.register_enhanced_actions()
        
    def register_enhanced_actions(self):
        """Register new multi-modal actions as Action instances."""
        self.action_registry.register_action(ProcessImageAction(self.system, self.data_service))
        self.action_registry.register_action(ProcessAudioAction(self.system, self.data_service))
        self.action_registry.register_action(AnalyzeDirectoryAction(self.system, self.data_service))
        self.action_registry.register_action(CrossModalAnalysisAction(self.system, self.data_service))
```

This demonstrates how the ActionRegistry can be extended with specialized actions while maintaining the same interface.

``mermaid
sequenceDiagram
participant DecisionEngine as "Decision Engine"
participant ActionManager as "ActionManager"
participant Registry as "ActionRegistry"
participant Action as "Action Instance"
participant System as "AGISystem"
DecisionEngine->>ActionManager : execute_action(decision)
ActionManager->>ActionManager : Parse decision JSON
ActionManager->>Registry : get_action(action_name)
Registry->>Registry : Look up action in dictionary
Registry-->>ActionManager : Return action instance
ActionManager->>Action : execute(**params)
Action->>System : Access system resources
Action-->>ActionManager : Return execution result
ActionManager->>ActionManager : Log action execution
ActionManager-->>DecisionEngine : Return result
```

**Diagram sources**
- [core/action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\action_manager.py#L60-L126)
- [core/enhanced_action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\enhanced_action_manager.py#L20-L45)

**Section sources**
- [core/action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\action_manager.py#L60-L126)
- [core/enhanced_action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\enhanced_action_manager.py#L20-L45)

## Registering Custom Actions

Developers can extend the AGI's capabilities by creating custom actions that inherit from the Action base class.

### Built-in Action Examples

The system includes several built-in actions registered during initialization:

- **ProposeAndTestInventionAction**: For proposing and testing new inventions
- **LogMessageAction**: For logging messages to the system
- **WritePythonCodeAction**: For writing Python code files
- **ExecutePythonFileAction**: For executing Python files
- **BlogPublishAction**: For publishing blog posts
- **CollaborativeTaskAction**: For managing collaborative tasks between RAVANA and users with feedback mechanisms

These are registered in the ActionRegistry constructor:

```python
def __init__(self,
             system: 'AGISystem',
             data_service: 'DataService'
             ) -> None:
    self.actions: Dict[str, Action] = {}
    self._register_action(ProposeAndTestInventionAction(system, data_service))
    self._register_action(LogMessageAction(system, data_service))
    self._register_action(WritePythonCodeAction(system, data_service))
    self._register_action(ExecutePythonFileAction(system, data_service))
    self._register_action(BlogPublishAction(system, data_service))
    self._register_action(CollaborativeTaskAction(system, data_service))
```

### Creating Custom Actions

To create a custom action, follow these steps as documented in the DEVELOPER_GUIDE.md:

1. Create a new Python file in the `core/actions/` directory
2. Define a class that inherits from `core.actions.action.Action`
3. Implement the required properties and methods

Example of a custom "hello_world" action:

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

Once created, the action will be automatically discovered and registered when the `discover_actions()` method is called, as it scans all modules in the `core.actions` package for classes that inherit from the Action base class.

``mermaid
flowchart TD
Start([Create Custom Action]) --> CreateFile["Create Python file in core/actions/"]
CreateFile --> DefineClass["Define class inheriting from Action"]
DefineClass --> ImplementName["Implement name property"]
ImplementName --> ImplementDescription["Implement description property"]
ImplementDescription --> ImplementParameters["Implement parameters property"]
ImplementParameters --> ImplementExecute["Implement execute method"]
ImplementExecute --> SaveFile["Save the file"]
SaveFile --> SystemStart["System startup or discovery"]
SystemStart --> DiscoverActions["Call discover_actions()"]
DiscoverActions --> FindModule["Find the new module"]
FindModule --> FindClass["Find Action subclass"]
FindClass --> Instantiate["Instantiate the action"]
Instantiate --> Register["Register in ActionRegistry"]
Register --> Available["Action available for use"]
```

**Diagram sources**
- [docs_archive/DEVELOPER_GUIDE.md](file://c:\Users\ASUS\Documents\GitHub\RAVANA\docs_archive\DEVELOPER_GUIDE.md#L175-L216)
- [core/actions/registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L37-L56)

**Section sources**
- [docs_archive/DEVELOPER_GUIDE.md](file://c:\Users\ASUS\Documents\GitHub\RAVANA\docs_archive\DEVELOPER_GUIDE.md#L175-L216)
- [core/actions/registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L37-L56)
- [core/actions/collaborative_task.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\collaborative_task.py#L6-L552)

## Troubleshooting Common Issues

This section addresses common issues developers may encounter when working with the ActionRegistry and provides guidance for resolution.

### Import-Time Registration Failures

**Issue**: Actions are not being discovered during the import process.

**Causes and Solutions**:
- **Circular imports**: Ensure that action modules do not create circular import dependencies. Use lazy imports or refactor code structure.
- **Missing __init__.py**: Verify that all directories in the `core/actions/` hierarchy contain an `__init__.py` file to be recognized as packages.
- **Syntax errors**: Check for syntax errors in action modules, as these will prevent successful import and discovery.

### Namespace Conflicts

**Issue**: Multiple actions with the same name cause overwriting.

**Behavior**: When duplicate action names are detected, the system logs a warning and overwrites the existing entry:

```python
if action.name in self.actions:
    logger.warning(f"Action '{action.name}' is already registered. Overwriting.")
```

**Best Practices**:
- Use unique, descriptive names for custom actions
- Follow a consistent naming convention (e.g., verb_noun pattern)
- Check existing action names before creating new ones using `get_all_actions()`

### Dynamic Reloading Limitations

**Issue**: The current implementation does not support hot-reloading of actions during runtime.

**Current Behavior**: Actions are registered during system initialization and discovery. Changes to action code require a system restart to take effect.

**Workarounds**:
- Implement a custom reload mechanism that calls `discover_actions()` again
- Use the `register_action()` method to add new actions at runtime
- Design actions to be configurable rather than requiring code changes

### Common Error Scenarios

| Error Scenario | Symptoms | Resolution |
|----------------|----------|------------|
| Invalid action parameters | `InvalidActionParams` exception during execution | Validate parameters using the action's `validate_params()` method before execution |
| Action not found | `ValueError` when calling `get_action()` | Verify the action name is correct and the action is properly registered |
| Action instantiation failure | Error logged during discovery process | Check that the action class can be instantiated with the required parameters |
| Module import failure | Action not discovered, import error in logs | Verify module path and dependencies |
| Collaborative task action failure | Error in conversational AI integration | Verify system references and user ID parameters |

``mermaid
flowchart TD
Start([Troubleshooting]) --> CheckLogs["Check system logs for error messages"]
CheckLogs --> IdentifyIssue["Identify the specific issue"]
subgraph ImportFailures
IdentifyIssue --> ImportError{"Import error?"}
ImportError --> |Yes| CheckSyntax["Check for syntax errors"]
CheckSyntax --> CheckInit["Verify __init__.py files"]
CheckInit --> CheckCircular["Check for circular imports"]
CheckCircular --> ResolveImport["Resolve import issues"]
end
subgraph NamespaceConflicts
IdentifyIssue --> DuplicateName{"Duplicate action name?"}
DuplicateName --> |Yes| CheckExisting["Check existing action names"]
CheckExisting --> UseUnique["Use unique name"]
UseUnique --> RegisterNew["Register with unique name"]
end
subgraph ExecutionErrors
IdentifyIssue --> ExecutionError{"Execution error?"}
ExecutionError --> |Yes| ValidateParams["Validate action parameters"]
ValidateParams --> CheckRequirements["Check required parameters"]
CheckRequirements --> ProvideAll["Provide all required parameters"]
end
ResolveImport --> TestFix["Test the fix"]
RegisterNew --> TestFix
ProvideAll --> TestFix
TestFix --> VerifySuccess{"Issue resolved?"}
VerifySuccess --> |Yes| Complete["Troubleshooting complete"]
VerifySuccess --> |No| Repeat["Repeat troubleshooting process"]
```

**Section sources**
- [core/actions/registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py#L29-L56)
- [core/actions/action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L30-L43)
- [core/action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\action_manager.py#L89-L126)
- [core/actions/collaborative_task.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\collaborative_task.py#L6-L552)

**Referenced Files in This Document**   
- [core/actions/registry.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\registry.py) - *Updated in recent commit with CollaborativeTaskAction registration*
- [core/actions/action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py) - *Base Action class definition*
- [core/action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\action_manager.py) - *ActionManager integration*
- [core/enhanced_action_manager.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\enhanced_action_manager.py) - *Enhanced action manager with multi-modal actions*
- [docs_archive/DEVELOPER_GUIDE.md](file://c:\Users\ASUS\Documents\GitHub\RAVANA\docs_archive\DEVELOPER_GUIDE.md) - *Developer guide for adding new actions*
- [core/system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py) - *AGISystem component*
- [core/actions/collaborative_task.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\collaborative_task.py) - *Added in recent commit for conversational AI integration*