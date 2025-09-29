class ActionError(Exception):
    """Base exception for action-related errors."""


class InvalidActionError(ActionError):
    """Raised when an action is not found in the registry."""


class InvalidActionParams(ActionError):
    """Raised when the parameters for an action are invalid."""


class ActionException(Exception):
    """Custom exception for action execution errors."""
