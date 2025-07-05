import importlib
import inspect
import pkgutil
from typing import Dict, List, Type
import logging

from core.actions.action import Action
import core.actions
from core.actions.experimental import ProposeAndTestInventionAction
from core.actions.io import LogMessageAction
from core.actions.coding import WritePythonCodeAction, ExecutePythonFileAction

logger = logging.getLogger(__name__)

class ActionRegistry:
    def __init__(self,
                 system: 'AGISystem',
                 data_service: 'DataService'
                 ) -> None:
        self.actions: Dict[str, Action] = {}
        self._register_action(ProposeAndTestInventionAction(system, data_service))
        self._register_action(LogMessageAction(system, data_service))
        self._register_action(WritePythonCodeAction(system, data_service))
        self._register_action(ExecutePythonFileAction(system, data_service))

    def _register_action(self, action: Action) -> None:
        if action.name in self.actions:
            logger.warning(f"Action '{action.name}' is already registered. Overwriting.")
        self.actions[action.name] = action

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

    def get_action(self, name: str) -> Action:
        action = self.actions.get(name)
        if not action:
            raise ValueError(f"Action '{name}' not found.")
        return action

    def get_all_actions(self) -> List[Action]:
        return list(self.actions.values())

    def get_action_definitions(self) -> str:
        """Returns a formatted string of all action definitions for the LLM prompt."""
        if not self.actions:
            return "No actions available."

        output = "Available Actions:\n"
        for action in self.actions.values():
            output += f"- {action.name}:\n"
            output += f"  Description: {action.description}\n"
            output += "  Parameters:\n"
            if action.parameters:
                for param in action.parameters:
                    output += f"    - {param['name']} ({param['type']}): {param['description']}\n"
            else:
                output += "    - None\n"
        return output 