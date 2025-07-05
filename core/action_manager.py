import json
import logging
from typing import Any, Dict
import asyncio

from core.actions.exceptions import ActionError, ActionException
from core.actions.registry import ActionRegistry

logger = logging.getLogger(__name__)

class ActionManager:
    def __init__(self, system: 'AGISystem', data_service: 'DataService'):
        self.system = system
        self.data_service = data_service
        self.action_registry = ActionRegistry(system, data_service)
        logger.info(f"ActionManager initialized with {len(self.action_registry.actions)} actions.")
        self.log_available_actions()

    def log_available_actions(self):
        logger.info("Available Actions:")
        for action_name, action in self.action_registry.actions.items():
            description = action.description
            params = action.parameters
            logger.info(f"- {action_name}:")
            logger.info(f"  Description: {description}")
            if params:
                logger.info("  Parameters:")
                for param in params:
                    param_name = param.get('name')
                    param_type = param.get('type')
                    param_desc = param.get('description', '')
                    logger.info(f"    - {param_name} ({param_type}): {param_desc}")
            else:
                logger.info("  Parameters: None")
        logger.info("")

    async def execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
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
            if not action:
                raise ActionException(f"Action '{action_name}' not found.")

            try:
                logger.info(f"Executing action '{action_name}' with params: {action_params}")
                result = await action.execute(**action_params)
                logger.info(f"Action '{action_name}' executed successfully.")
                
                # Log the action to the database
                await asyncio.to_thread(
                    self.data_service.save_action_log,
                    action_name,
                    action_params,
                    'success',
                    str(result) # Convert result to string for logging
                )
                return result
            except ActionException as e:
                logger.error(f"Action '{action_name}' failed: {e}")
                await asyncio.to_thread(
                    self.data_service.save_action_log,
                    action_name,
                    action_params,
                    'error',
                    str(e)
                )
                return {"error": str(e)}
            except Exception as e:
                logger.error(f"An unexpected error occurred during execution of action '{action_name}': {e}", exc_info=True)
                await asyncio.to_thread(
                    self.data_service.save_action_log,
                    action_name,
                    action_params,
                    'error',
                    f"Unexpected error: {e}"
                )
                return {"error": f"An unexpected error occurred: {e}"}

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