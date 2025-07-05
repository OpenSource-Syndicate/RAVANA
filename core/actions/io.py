import logging
from typing import Any, Dict, List
from core.actions.action import Action

logger = logging.getLogger(__name__)

class LogMessageAction(Action):
    @property
    def name(self) -> str:
        return "log_message"

    @property
    def description(self) -> str:
        return "Logs a message to the console. Useful for recording thoughts, observations, or decisions."

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "message",
                "type": "string",
                "description": "The message to log.",
                "required": True,
            },
            {
                "name": "level",
                "type": "string",
                "description": "The logging level (e.g., 'info', 'warning', 'error'). Defaults to 'info'.",
                "required": False,
            },
        ]

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        message = kwargs.get("message")
        level = kwargs.get("level", "info").lower()

        log_func = getattr(logger, level, logger.info)
        log_func(f"[AGI Thought]: {message}")

        return {"status": "success", "message": f"Message logged with level '{level}'."} 