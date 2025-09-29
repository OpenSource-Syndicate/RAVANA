import json
import logging
import os
from typing import Any, Dict, List, TYPE_CHECKING
from pathlib import Path
import asyncio

from core.actions.exceptions import ActionError, ActionException
from core.actions.registry import ActionRegistry
from services.multi_modal_service import MultiModalService
from core.actions.multi_modal import (
    ProcessImageAction,
    ProcessAudioAction,
    AnalyzeDirectoryAction,
    CrossModalAnalysisAction
)

if TYPE_CHECKING:
    from core.system import AGISystem
    from services.data_service import DataService

logger = logging.getLogger(__name__)


class ActionManager:
    def __init__(self, system: 'AGISystem', data_service: 'DataService'):
        self.system = system
        self.data_service = data_service
        self.action_registry = ActionRegistry(system, data_service)
        
        # Enhanced features
        self.multi_modal_service = MultiModalService()
        self.action_cache = {}
        self.parallel_limit = 3  # Max parallel actions
        
        # Register enhanced actions
        self.register_enhanced_actions()
        
        logger.info(
            f"ActionManager initialized with {len(self.action_registry.actions)} actions.")
        self.log_available_actions()

    def register_enhanced_actions(self):
        """Register new multi-modal actions as Action instances."""
        self.action_registry.register_action(
            ProcessImageAction(self.system, self.data_service))
        self.action_registry.register_action(
            ProcessAudioAction(self.system, self.data_service))
        self.action_registry.register_action(
            AnalyzeDirectoryAction(self.system, self.data_service))
        self.action_registry.register_action(
            CrossModalAnalysisAction(self.system, self.data_service))

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
                    logger.info(
                        f"    - {param_name} ({param_type}): {param_desc}")
            else:
                logger.info("  Parameters: None")
        logger.info("")

    async def execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses the decision from the LLM, validates it, and executes the chosen action.
        This method can handle both a raw LLM response and a pre-defined action dictionary.
        """
        action_name = "unknown"
        action_params = {}
        action_data = {}

        # Case 1: The decision is already a parsed action dictionary
        if "action" in decision and "params" in decision:
            action_data = decision

        # Case 2: The decision is a raw response from the LLM
        elif "raw_response" in decision:
            raw_response = decision.get("raw_response", "")
            if not raw_response:
                logger.warning(
                    "Decision engine provided an empty raw_response.")
                return {"error": "No action taken: empty response."}

            try:
                # Find the JSON block in the raw response
                json_start = raw_response.find("```json")
                json_end = raw_response.rfind("```")

                if json_start == -1 or json_end == -1 or json_start >= json_end:
                    logger.warning(
                        "No valid JSON block found in the LLM's response. Trying to parse the whole string.")
                    action_data = json.loads(raw_response)
                else:
                    json_str = raw_response[json_start + 7:json_end].strip()
                    action_data = json.loads(json_str)

            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to decode JSON from response: {raw_response}. Error: {e}")
                # Try to extract action using regex as fallback
                action_match = re.search(
                    r'"action"\s*:\s*"([^"]+)"', raw_response)
                if action_match:
                    action_name = action_match.group(1)
                    # Try to extract params
                    params_match = re.search(
                        r'"params"\s*:\s*({[^}]+})', raw_response)
                    if params_match:
                        try:
                            action_params = json.loads(params_match.group(1))
                        except:
                            action_params = {}
                    return await self._execute_action_with_fallback(action_name, action_params)
                return {"error": "No action taken: could not parse response."}

        else:
            logger.error(f"Invalid decision format: {decision}")
            return {"error": "No action taken: invalid decision format."}

        try:
            action_name = action_data.get("action")
            action_params = action_data.get("params", {})

            if not action_name:
                logger.warning("No 'action' key found in the parsed JSON.")
                return {"error": "No action taken: 'action' key missing."}

            return await self._execute_action_with_fallback(action_name, action_params)

        except Exception as e:
            logger.error(
                f"An unexpected error occurred during execution of action '{action_name}': {e}", exc_info=True)
            await asyncio.to_thread(
                self.data_service.save_action_log,
                action_name,
                action_params,
                'error',
                f"Unexpected error: {e}"
            )
            return {"error": f"An unexpected error occurred: {e}"}

    async def _execute_action_with_fallback(self, action_name: str, action_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with comprehensive error handling and fallback mechanisms."""
        try:
            action = self.action_registry.get_action(action_name)
            if not action:
                # Try to find similar action names as fallback
                similar_actions = self._find_similar_actions(action_name)
                if similar_actions:
                    logger.warning(
                        f"Action '{action_name}' not found. Similar actions: {similar_actions}")
                    # Try the first similar action as fallback
                    action_name = similar_actions[0]
                    action = self.action_registry.get_action(action_name)

                if not action:
                    raise ActionException(f"Action '{action_name}' not found.")

            logger.info(
                f"Executing action '{action_name}' with params: {action_params}")

            # Add timeout to prevent hanging actions
            try:
                result = await asyncio.wait_for(
                    action.execute(**action_params),
                    timeout=300.0  # 5 minute timeout
                )
                logger.info(f"Action '{action_name}' executed successfully.")

                # Log the action to the database
                await asyncio.to_thread(
                    self.data_service.save_action_log,
                    action_name,
                    action_params,
                    'success',
                    str(result)  # Convert result to string for logging
                )
                return result
            except asyncio.TimeoutError:
                logger.error(
                    f"Action '{action_name}' timed out after 5 minutes")
                raise ActionException(f"Action '{action_name}' timed out")

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

    def _find_similar_actions(self, action_name: str) -> List[str]:
        """Find actions with similar names to the requested action."""
        similar_actions = []
        action_name_lower = action_name.lower()

        for registered_action_name in self.action_registry.actions.keys():
            # Check for exact substring match or fuzzy match
            if (action_name_lower in registered_action_name.lower() or
                    registered_action_name.lower() in action_name_lower):
                similar_actions.append(registered_action_name)

        return similar_actions

    async def execute_action_enhanced(self, decision: dict) -> Any:
        """Enhanced action execution with better error handling and caching."""
        try:
            action_name = decision.get('action', 'unknown')
            params = decision.get('params', {})

            # Check cache for repeated actions (except for dynamic actions)
            non_cacheable = {'log_message',
                             'get_current_time', 'generate_random'}
            cache_key = f"{action_name}_{hash(str(params))}"

            if action_name not in non_cacheable and cache_key in self.action_cache:
                logger.info(f"Using cached result for action: {action_name}")
                return self.action_cache[cache_key]

            # Execute action with timeout
            result = await asyncio.wait_for(
                self.execute_action(decision),
                timeout=300  # 5 minute timeout
            )

            # Cache successful results for cacheable actions
            if (action_name not in non_cacheable and
                result and not isinstance(result, Exception) and
                    not str(result).startswith("Error")):
                self.action_cache[cache_key] = result

            return result

        except asyncio.TimeoutError:
            logger.error(f"Action {action_name} timed out")
            return {"error": "Action timed out", "action": action_name}
        except Exception as e:
            logger.error(f"Enhanced action execution failed: {e}")
            return {"error": str(e), "action": action_name}

    async def execute_parallel_actions(self, decisions: List[dict]) -> List[Any]:
        """Execute multiple actions in parallel with concurrency limit."""
        if not decisions:
            return []

        semaphore = asyncio.Semaphore(self.parallel_limit)

        async def execute_with_semaphore(decision):
            async with semaphore:
                return await self.execute_action_enhanced(decision)

        tasks = [execute_with_semaphore(decision) for decision in decisions]

        # Execute with error handling
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results to handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Action {i} failed with exception: {result}")
                    processed_results.append({
                        "error": str(result),
                        "action_index": i,
                        "success": False
                    })
                else:
                    processed_results.append(result)

            logger.info(
                f"Executed {len(decisions)} actions in parallel with {len([r for r in processed_results if not r.get('error')])} successful")
            return processed_results

        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            return [{"error": str(e), "success": False} for _ in decisions]

    async def execute_action_with_retry(self, decision: dict, max_retries: int = 3) -> Any:
        """Execute an action with retry logic."""
        last_exception = None

        for attempt in range(max_retries):
            try:
                result = await self.execute_action_enhanced(decision)

                # If successful, return result
                if not result.get("error"):
                    return result

                # If it's a retryable error, continue
                error_msg = result.get("error", "").lower()
                if any(keyword in error_msg for keyword in ["timeout", "network", "connection", "retry"]):
                    logger.warning(
                        f"Action failed with retryable error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                    last_exception = result.get("error")
                    if attempt < max_retries - 1:  # Don't sleep on last attempt
                        # Exponential backoff
                        await asyncio.sleep(2 ** attempt)
                    continue

                # Non-retryable error, return immediately
                return result

            except Exception as e:
                logger.error(
                    f"Action execution failed (attempt {attempt + 1}/{max_retries}): {e}")
                last_exception = str(e)
                if attempt < max_retries - 1:  # Don't sleep on last attempt
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        # All retries exhausted
        return {
            "error": f"Action failed after {max_retries} attempts. Last error: {last_exception}",
            "success": False
        }

    async def process_image_action(self, image_path: str, analysis_prompt: str = None) -> dict:
        """Action to process and analyze an image."""
        try:
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}

            result = await self.multi_modal_service.process_image(
                image_path,
                analysis_prompt or "Analyze this image in detail"
            )

            # Add to knowledge base if successful
            if result.get('success', False):
                try:
                    await asyncio.to_thread(
                        self.system.knowledge_service.add_knowledge,
                        content=result['description'],
                        source="image_analysis",
                        category="visual_content"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to add image analysis to knowledge: {e}")

            return result

        except Exception as e:
            logger.error(f"Image processing action failed: {e}")
            return {"error": str(e), "success": False}

    async def process_audio_action(self, audio_path: str, analysis_prompt: str = None) -> dict:
        """Action to process and analyze an audio file."""
        try:
            if not os.path.exists(audio_path):
                return {"error": f"Audio file not found: {audio_path}"}

            result = await self.multi_modal_service.process_audio(
                audio_path,
                analysis_prompt or "Describe and analyze this audio"
            )

            # Add to knowledge base if successful
            if result.get('success', False):
                try:
                    await asyncio.to_thread(
                        self.system.knowledge_service.add_knowledge,
                        content=result['description'],
                        source="audio_analysis",
                        category="audio_content"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to add audio analysis to knowledge: {e}")

            return result

        except Exception as e:
            logger.error(f"Audio processing action failed: {e}")
            return {"error": str(e), "success": False}

    async def analyze_directory_action(self, directory_path: str, recursive: bool = False) -> dict:
        """Action to analyze all media files in a directory."""
        try:
            if not os.path.exists(directory_path):
                return {"error": f"Directory not found: {directory_path}"}

            results = await self.multi_modal_service.process_directory(
                directory_path,
                recursive
            )

            # Generate summary
            summary = await self.multi_modal_service.generate_content_summary(results)

            # Add summary to knowledge base
            try:
                await asyncio.to_thread(
                    self.system.knowledge_service.add_knowledge,
                    content=summary,
                    source="directory_analysis",
                    category="batch_analysis"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to add directory analysis to knowledge: {e}")

            return {
                "success": True,
                "directory": directory_path,
                "files_processed": len(results),
                "results": results,
                "summary": summary
            }

        except Exception as e:
            logger.error(f"Directory analysis action failed: {e}")
            return {"error": str(e), "success": False}

    async def cross_modal_analysis_action(self, content_paths: List[str], analysis_prompt: str = None) -> dict:
        """Action to perform cross-modal analysis on multiple files."""
        try:
            if not content_paths:
                return {"error": "No content paths provided"}

            # Process each file
            processed_content = []
            for path in content_paths:
                if not os.path.exists(path):
                    logger.warning(f"File not found: {path}")
                    continue

                ext = Path(path).suffix.lower()
                if ext in self.multi_modal_service.supported_image_formats:
                    result = await self.multi_modal_service.process_image(path)
                elif ext in self.multi_modal_service.supported_audio_formats:
                    result = await self.multi_modal_service.process_audio(path)
                else:
                    logger.warning(f"Unsupported file format: {path}")
                    continue

                processed_content.append(result)

            if not processed_content:
                return {"error": "No valid content could be processed"}

            # Perform cross-modal analysis
            cross_modal_result = await self.multi_modal_service.cross_modal_analysis(
                processed_content,
                analysis_prompt
            )

            # Add to knowledge base
            if cross_modal_result.get('success', False):
                try:
                    await asyncio.to_thread(
                        self.system.knowledge_service.add_knowledge,
                        content=cross_modal_result['analysis'],
                        source="cross_modal_analysis",
                        category="multi_modal_insights"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to add cross-modal analysis to knowledge: {e}")

            return {
                "success": True,
                "content_processed": len(processed_content),
                "cross_modal_analysis": cross_modal_result,
                "individual_results": processed_content
            }

        except Exception as e:
            logger.error(f"Cross-modal analysis action failed: {e}")
            return {"error": str(e), "success": False}

    def clear_cache(self, max_size: int = 100):
        """Clear action cache if it gets too large."""
        if len(self.action_cache) > max_size:
            # Keep only the most recent entries
            items = list(self.action_cache.items())
            self.action_cache = dict(items[-max_size//2:])
            logger.info(
                f"Cleared action cache, kept {len(self.action_cache)} entries")

    async def get_action_statistics(self) -> dict:
        """Get statistics about action usage."""
        try:
            total_actions = len(self.action_registry.actions)
            cached_actions = len(self.action_cache)

            # Get available actions
            available_actions = list(self.action_registry.actions.keys())

            return {
                "total_registered_actions": total_actions,
                "cached_results": cached_actions,
                "available_actions": available_actions,
                "multi_modal_supported": True,
                "parallel_limit": self.parallel_limit
            }

        except Exception as e:
            logger.error(f"Failed to get action statistics: {e}")
            return {"error": str(e)}
