class ActionError(Exception):
    """Base exception for action-related errors."""
    pass

class InvalidActionError(ActionError):
    """Raised when an action is not found in the registry."""
    pass

class InvalidActionParams(ActionError):
    """Raised when the parameters for an action are invalid."""
    pass

class ActionException(Exception):
    """Custom exception for action execution errors."""
    pass 