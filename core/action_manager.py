import json
import logging
from typing import Any, Dict

from core.actions.exceptions import ActionError
from core.actions.registry import ActionRegistry

logger = logging.getLogger(__name__)

class ActionManager:
    def __init__(self, agi_system, data_service):
        self.agi_system = agi_system
        self.action_registry = ActionRegistry()
        self.data_service = data_service
        logger.info(f"ActionManager initialized with {len(self.action_registry.get_all_actions())} actions.")
        logger.info(self.action_registry.get_action_definitions())

    async def execute_action(self, decision: Dict[str, Any]) -> Any:
        """
        Parses the decision from the LLM, validates it, and executes the chosen action.
        """
        raw_response = decision.get("raw_response", "")
        if not raw_response:
            logger.warning("Decision engine did not provide a raw_response.")
            return "No action taken: empty response."

        try:
            # Find the JSON block in the raw response
            json_start = raw_response.find("```json")
            json_end = raw_response.rfind("```")

            if json_start == -1 or json_end == -1 or json_start >= json_end:
                logger.warning("No valid JSON block found in the LLM's response.")
                # Fallback: try to parse the whole string
                try:
                    action_data = json.loads(raw_response)
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON from response: {raw_response}")
                    return f"No action taken: could not parse response."
            else:
                json_str = raw_response[json_start + 7:json_end].strip()
                try:
                    action_data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode extracted JSON: {json_str}. Error: {e}")
                    return f"No action taken: could not parse JSON block."

            action_name = action_data.get("action")
            action_params = action_data.get("params", {})

            if not action_name:
                logger.warning("No 'action' key found in the parsed JSON.")
                return "No action taken: 'action' key missing."

            action = self.action_registry.get_action(action_name)
            action.validate_params(action_params)
            
            logger.info(f"Executing action '{action_name}' with params: {action_params}")
            result = await action.execute(**action_params)
            logger.info(f"Action '{action_name}' executed successfully.")
            
            # Log the successful action
            self.data_service.save_action_log(action_name, action_params, "success", result)
            
            return result

        except ActionError as e:
            logger.error(f"Action execution failed: {e}", exc_info=True)
            # Log the failed action
            action_name = locals().get('action_name', 'unknown')
            action_params = locals().get('action_params', {})
            self.data_service.save_action_log(action_name, action_params, "failure", str(e))
            return f"Action failed: {e}"
        except Exception as e:
            logger.error(f"An unexpected error occurred during action execution: {e}", exc_info=True)
            # Log the failed action
            action_name = locals().get('action_name', 'unknown')
            action_params = locals().get('action_params', {})
            self.data_service.save_action_log(action_name, action_params, "failure", str(e))
            return f"An unexpected error occurred: {e}" 