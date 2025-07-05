from abc import ABC, abstractmethod
from typing import Any, Dict, List
import json

from core.actions.exceptions import InvalidActionParams

class Action(ABC):
    def __init__(self, system: 'AGISystem', data_service: 'DataService'):
        self.system = system
        self.data_service = data_service
    """
    An abstract base class for actions that the AGI can perform.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the action."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A description of what the action does."""
        pass

    @property
    @abstractmethod
    def parameters(self) -> List[Dict[str, Any]]:
        """A list of parameters that the action accepts."""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Executes the action with the given parameters."""
        pass

    def validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validates the given parameters against the action's defined parameters.
        Raises InvalidActionParams if the parameters are invalid.
        """
        required_params = {p['name'] for p in self.parameters if p.get('required', False)}
        provided_params = set(params.keys())

        missing_params = required_params - provided_params
        if missing_params:
            raise InvalidActionParams(f"Missing required parameters: {', '.join(missing_params)}")

        extra_params = provided_params - {p['name'] for p in self.parameters}
        if extra_params:
            raise InvalidActionParams(f"Unexpected parameters: {', '.join(extra_params)}")

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the action."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_json(self) -> str:
        """Returns a JSON string representing the action's schema."""
        return json.dumps(self.to_dict(), indent=2) 