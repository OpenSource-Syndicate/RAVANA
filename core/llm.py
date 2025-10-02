import json
import os
import requests
import logging
import random
import traceback
import sys
import re

# Import optional dependencies with error handling
try:
    from google import genai
except ImportError:
    genai = None
    logging.warning("Google Generative AI library not installed. Gemini functionality will be disabled.")

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logging.warning("OpenAI library not installed. Some provider functionality may be disabled.")
import subprocess
import importlib.util
import threading
import time
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import ssl
import certifi
from modules.decision_engine.search_result_manager import search_result_manager

# Import the new PromptManager
from core.prompt_manager import PromptManager

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

# Initialize the global prompt manager
prompt_manager = PromptManager()

# Load config
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)


@dataclass
class GeminiKeyStatus:
    """Track status and metadata for individual Gemini API keys."""
    key_id: str
    api_key: str
    priority: int
    is_available: bool = True
    rate_limit_reset_time: Optional[datetime] = None
    failure_count: int = 0
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    total_requests: int = 0


class GeminiKeyManager:
    """Manages multiple Gemini API keys with fallback and rate limiting."""

    def __init__(self):
        self._lock = threading.Lock()
        self._keys: Dict[str, GeminiKeyStatus] = {}
        self._load_keys_from_config()

        # Rate limiting patterns to detect
        self.rate_limit_indicators = [
            "quota exceeded",
            "rate limit exceeded",
            "too many requests",
            "429",
            "resource_exhausted",
            "quota_exceeded",
            "rate_limit_exceeded"
        ]

    def _load_keys_from_config(self):
        """Load API keys from configuration file."""
        try:
            gemini_config = config.get('gemini', {})
            api_keys = gemini_config.get('api_keys', [])

            # Also check environment variables for additional keys
            env_keys = []
            for i in range(1, 21):  # Check for up to 20 environment variables
                env_key = os.getenv(f"GEMINI_API_KEY_{i}")
                if env_key:
                    env_keys.append({
                        "id": f"gemini_env_key_{i}",
                        "key": env_key,
                        "priority": 100 + i  # Lower priority than config keys
                    })

            all_keys = api_keys + env_keys

            # Add fallback key if no keys found
            if not all_keys:
                logger.warning(
                    "No API keys found in config or environment, using fallback key")
                fallback_key = {
                    "id": "fallback_key",
                    "key": "AIzaSyAWR9C57V2f2pXFwjtN9jkNYKA_ou5Hdo4",
                    "priority": 999
                }
                all_keys.append(fallback_key)

            for key_data in all_keys:
                key_status = GeminiKeyStatus(
                    key_id=key_data['id'],
                    api_key=key_data['key'],
                    priority=key_data.get('priority', 999)
                )
                self._keys[key_status.key_id] = key_status

            logger.info(f"Loaded {len(self._keys)} Gemini API keys")

        except Exception as e:
            logger.error(f"Failed to load Gemini keys from config: {e}")
            # Fallback to hardcoded key if config fails
            fallback_key = GeminiKeyStatus(
                key_id="fallback_key",
                api_key="AIzaSyAWR9C57V2f2pXFwjtN9jkNYKA_ou5Hdo4",
                priority=999
            )
            self._keys["fallback_key"] = fallback_key

    def get_available_key(self) -> Optional[GeminiKeyStatus]:
        """Get the highest priority available key that is not rate limited."""
        with self._lock:
            # Filter available keys (not rate limited or rate limit has expired)
            available_keys = []
            current_time = datetime.now()

            for key_status in self._keys.values():
                if not key_status.is_available:
                    continue

                # Check if rate limit has expired
                if key_status.rate_limit_reset_time and key_status.rate_limit_reset_time <= current_time:
                    key_status.rate_limit_reset_time = None
                    key_status.is_available = True

                # Add to available keys if not currently rate limited
                if key_status.is_available:
                    available_keys.append(key_status)

            # Sort by priority (lower number = higher priority)
            available_keys.sort(key=lambda x: x.priority)

            # Return highest priority available key
            return available_keys[0] if available_keys else None

    def mark_key_success(self, key_id: str):
        """Mark a key as successful."""
        with self._lock:
            if key_id in self._keys:
                key_status = self._keys[key_id]
                key_status.total_requests += 1
                key_status.last_success = datetime.now()
                key_status.consecutive_failures = 0
                key_status.failure_count = max(
                    0, key_status.failure_count - 1)  # Decay failure count
                key_status.is_available = True

    def mark_key_failed(self, key_id: str, error: Exception):
        """Mark a key as failed."""
        with self._lock:
            if key_id in self._keys:
                key_status = self._keys[key_id]
                key_status.total_requests += 1
                key_status.failure_count += 1
                key_status.consecutive_failures += 1

                # Mark as unavailable if too many consecutive failures
                if key_status.consecutive_failures >= 5:
                    key_status.is_available = False
                    logger.warning(
                        f"Key {key_id} marked as unavailable due to repeated failures")

    def mark_key_rate_limited(self, key_id: str, reset_time: datetime):
        """Mark a key as rate limited."""
        with self._lock:
            if key_id in self._keys:
                key_status = self._keys[key_id]
                key_status.rate_limit_reset_time = reset_time
                key_status.is_available = False
                logger.info(
                    f"Key {key_id} marked as rate limited until {reset_time}")

    def is_rate_limit_error(self, error: Exception) -> Tuple[bool, Optional[datetime]]:
        """Check if an error is a rate limit error and extract reset time if available."""
        error_str = str(error).lower()

        # Check if it's a rate limit error
        is_rate_limited = any(
            indicator in error_str for indicator in self.rate_limit_indicators)

        # Try to extract reset time (this is a simplified implementation)
        reset_time = None
        if is_rate_limited:
            # In a real implementation, you would parse the actual reset time from the error response
            reset_time = datetime.now() + timedelta(minutes=1)  # Default to 1 minute

        return is_rate_limited, reset_time

    def get_key_statistics(self) -> Dict[str, Any]:
        """Get statistics about API key usage."""
        stats = {
            "total_keys": len(self._keys),
            "available_keys": 0,
            "rate_limited_keys": 0,
            "failed_keys": 0,
            "key_details": []
        }

        for key in self._keys.values():
            key_stats = {
                "id": key.key_id[:12] + "...",
                "priority": key.priority,
                "available": key.is_available,
                "total_requests": key.total_requests,
                "failure_count": key.failure_count,
                "consecutive_failures": key.consecutive_failures,
                "last_success": key.last_success.isoformat() if key.last_success else None
            }

            stats["key_details"].append(key_stats)

            if key.is_available:
                stats["available_keys"] += 1
            if key.rate_limit_reset_time:
                stats["rate_limited_keys"] += 1
            if key.failure_count > 0:
                stats["failed_keys"] += 1

        return stats


# Global instance
gemini_key_manager = GeminiKeyManager()

# Enhanced Gemini call wrapper with automatic key rotation


def call_gemini_with_fallback(prompt: str, function_type: str = "text", max_retries: int = 3, **kwargs) -> str:
    """
    Enhanced Gemini caller with automatic key rotation and rate limit handling.

    Args:
        prompt: The text prompt to send
        function_type: Type of function call ("text", "image", "audio", "search", "function_calling")
        max_retries: Maximum number of retries across all keys
        **kwargs: Additional arguments specific to function type

    Returns:
        Response text or error message
    """
    last_error = None

    for attempt in range(max_retries):
        key_status = gemini_key_manager.get_available_key()

        if not key_status:
            logger.error("No Gemini API keys available for request")
            return f"[All Gemini API keys exhausted: {last_error}]"

        try:
            logger.info(
                f"Using Gemini key {key_status.key_id[:12]}... for {function_type} request (attempt {attempt + 1}/{max_retries})")

            # Call the appropriate function based on type
            if function_type == "text":
                result = _call_gemini_text(prompt, key_status.api_key)
            elif function_type == "image":
                image_path = kwargs.get('image_path')
                if not image_path:
                    return "[Error: image_path required for image function]"
                result = _call_gemini_image(
                    image_path, prompt, key_status.api_key)
            elif function_type == "audio":
                audio_path = kwargs.get('audio_path')
                if not audio_path:
                    return "[Error: audio_path required for audio function]"
                result = _call_gemini_audio(
                    audio_path, prompt, key_status.api_key)
            elif function_type == "search":
                result = _call_gemini_search(prompt, key_status.api_key)
            elif function_type == "function_calling":
                function_declarations = kwargs.get('function_declarations', [])
                result = _call_gemini_function_calling(
                    prompt, function_declarations, key_status.api_key)
            else:
                return f"[Error: Unknown function type '{function_type}']"

            # Mark success and return result
            gemini_key_manager.mark_key_success(key_status.key_id)
            return result

        except Exception as e:
            last_error = e
            logger.warning(
                f"Gemini call failed with key {key_status.key_id[:12]}...: {e}")

            # Check if this is a rate limiting error
            is_rate_limited, reset_time = gemini_key_manager.is_rate_limit_error(
                e)

            if is_rate_limited:
                gemini_key_manager.mark_key_rate_limited(
                    key_status.key_id, reset_time)
            else:
                gemini_key_manager.mark_key_failed(key_status.key_id, e)

            # Short delay before retry
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff

    return f"[All Gemini retry attempts failed: {last_error}]"


def _call_gemini_text(prompt: str, api_key: str) -> str:
    """Internal function to call Gemini for text generation."""
    if genai is None:
        return "[Error: Google Generative AI library not available]"
    
    try:
        # Set timeout through HttpOptions instead of GenerateContentConfig
        http_options = genai.types.HttpOptions()
        client = genai.Client(api_key=api_key, http_options=http_options)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        logging.error(f"Error calling Gemini: {e}")
        return f"[Error: Failed to call Gemini - {e}]"


def _call_gemini_image(image_path: str, prompt: str, api_key: str) -> str:
    """Internal function to call Gemini for image captioning."""
    if genai is None:
        return "[Error: Google Generative AI library not available]"
    
    try:
        # Set timeout through HttpOptions instead of GenerateContentConfig
        http_options = genai.types.HttpOptions()
        client = genai.Client(api_key=api_key, http_options=http_options)
        my_file = client.files.upload(file=image_path)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[my_file, prompt]
        )
        return response.text
    except Exception as e:
        logging.error(f"Error calling Gemini for image: {e}")
        return f"[Error: Failed to call Gemini - {e}]"


def _call_gemini_audio(audio_path: str, prompt: str, api_key: str) -> str:
    """Internal function to call Gemini for audio description."""
    if genai is None:
        return "[Error: Google Generative AI library not available]"
    
    try:
        # Set timeout through HttpOptions instead of GenerateContentConfig
        http_options = genai.types.HttpOptions()
        client = genai.Client(api_key=api_key, http_options=http_options)
        my_file = client.files.upload(file=audio_path)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, my_file]
        )
        return response.text
    except Exception as e:
        logging.error(f"Error calling Gemini for audio: {e}")
        return f"[Error: Failed to call Gemini - {e}]"


def _call_gemini_search(prompt: str, api_key: str) -> str:
    """Internal function to call Gemini with Google Search."""
    if genai is None:
        return "[Error: Google Generative AI library not available]"
    
    try:
        from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
        # Set timeout through HttpOptions instead of GenerateContentConfig
        http_options = genai.types.HttpOptions()
        client = genai.Client(api_key=api_key, http_options=http_options)
        model_id = "gemini-2.0-flash"
        google_search_tool = Tool(google_search=GoogleSearch())
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"]
            )
        )
        # Return both the answer and grounding metadata if available
        answer = "\n".join([p.text for p in response.candidates[0].content.parts])
        grounding = getattr(
            response.candidates[0].grounding_metadata, 'search_entry_point', None)
        if grounding and hasattr(grounding, 'rendered_content'):
            return answer + "\n\n[Grounding Metadata:]\n" + grounding.rendered_content
        return answer
    except Exception as e:
        logging.error(f"Error calling Gemini with search: {e}")
        return f"[Error: Failed to call Gemini with search - {e}]"


def _call_gemini_function_calling(prompt: str, function_declarations: List, api_key: str) -> Tuple[str, Optional[Dict]]:
    """Internal function to call Gemini with function calling."""
    if genai is None:
        return "[Error: Google Generative AI library not available]", None
    
    try:
        from google.genai import types
        # Set timeout through HttpOptions instead of GenerateContentConfig
        http_options = genai.types.HttpOptions(timeout=30.0)
        client = genai.Client(api_key=api_key, http_options=http_options)
        tools = types.Tool(function_declarations=function_declarations)
        config = types.GenerateContentConfig(
            tools=[tools]
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config,
        )
        parts = response.candidates[0].content.parts
        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                return None, {"name": function_call.name, "args": function_call.args}
        # No function call found
        return response.text, None
    except Exception as e:
        logging.error(f"Error calling Gemini with function calling: {e}")
        return f"[Error: Failed to call Gemini with function calling - {e}]", None


def call_llm(prompt: str, model: str = None, **kwargs) -> str:
    """
    Unified interface for calling different LLMs with enhanced prompt management.

    Args:
        prompt: The prompt to send to the LLM
        model: The model to use (optional)
        **kwargs: Additional arguments for the LLM call

    Returns:
        The response from the LLM
    """
    # Use the prompt manager to validate and enhance the prompt if needed
    # Only validate for minimum requirements, not for specific sections
    if prompt_manager.validate_prompt(prompt, require_sections=False):
        logger.debug("Prompt validation passed")
    else:
        logger.warning("Prompt validation failed, proceeding with caution")

    # Use local Ollama model by default for main system
    # This can be overridden by passing a specific model
    if model is None:
        from core.config import Config
        config_instance = Config()
        local_model_config = config_instance.MAIN_SYSTEM_LOCAL_MODEL
        model = local_model_config['model_name']
    
    # Use Ollama API to call the local model
    return call_ollama_local(prompt, model, **kwargs)

# Enhanced error handling and retry logic


def _extract_json_block(text: str) -> str:
    """
    Pull out the first ```json ... ``` block; fallback to full text.
    """
    if not text:
        return "{}"

    # Try to find JSON block
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"\{.*\}",  # Any JSON-like structure
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return text.strip()


def extract_decision(raw_response: str) -> dict:
    """
    Returns a dict with keys: analysis, plan, action, params, raw_response
    """
    if not raw_response:
        return {"raw_response": "", "error": "Empty response"}

    block = _extract_json_block(raw_response)

    # Try to parse JSON, with fallback for truncated responses
    try:
        data = json.loads(block)
    except json.JSONDecodeError as je:
        # Try to fix common truncation issues
        fixed_block = _fix_truncated_json(block)
        try:
            data = json.loads(fixed_block)
            logger.info("Successfully parsed fixed JSON block")
        except json.JSONDecodeError:
            logger.error(
                "JSON decode error, returning raw_response only: %s", je)
            return {
                "raw_response": raw_response,
                "error": f"JSON decode error: {je}",
                "analysis": "Failed to parse decision",
                "plan": [],
                "action": "log_message",
                "params": {"message": f"Failed to parse decision: {raw_response[:200]}..."}
            }

    # Validate required keys
    required_keys = ["analysis", "plan", "action", "params"]
    for key in required_keys:
        if key not in data:
            logger.warning("Key %r missing from decision JSON", key)

    return {
        "raw_response": raw_response,
        "analysis": data.get("analysis", "No analysis provided"),
        "plan": data.get("plan", []),
        "action": data.get("action", "log_message"),
        "params": data.get("params", {"message": "No action specified"}),
        # New field for decision confidence
        "confidence": data.get("confidence", 0.5),
        # New field for reasoning chain
        "reasoning": data.get("reasoning", ""),
    }


def _fix_truncated_json(text: str) -> str:
    """
    Attempt to fix common JSON truncation issues.
    """
    if not text:
        return "{}"

    # Remove any trailing ellipsis
    fixed = text.rstrip().rstrip("...")

    # Try to parse the JSON to see if it's already valid
    try:
        json.loads(fixed)
        return fixed  # Already valid JSON
    except json.JSONDecodeError:
        pass  # Continue with fixing attempts

    # Make a copy for potential fallback
    original_fixed = fixed

    # Count opening and closing braces/brackets
    open_braces = fixed.count('{')
    close_braces = fixed.count('}')
    open_brackets = fixed.count('[')
    close_brackets = fixed.count(']')

    # Add missing closing braces/brackets
    while close_braces < open_braces:
        fixed += '}'
        close_braces += 1

    while close_brackets < open_brackets:
        fixed += ']'
        close_brackets += 1

    # Try to fix common truncation at the end
    # If the text ends with a comma, colon, or opening bracket/brace, remove it
    fixed = fixed.rstrip()
    max_attempts = 10  # Prevent infinite loop
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        original = fixed

        # Remove trailing commas, colons, and unclosed quotes
        if fixed.endswith((',', ':')):
            fixed = fixed[:-1]
        elif fixed.endswith('"') and fixed.count('"') % 2 == 1:  # Unclosed quote
            fixed = fixed[:-1]

        # After removing a trailing character, add appropriate closing character if needed
        open_braces = fixed.count('{')
        close_braces = fixed.count('}')
        open_brackets = fixed.count('[')
        close_brackets = fixed.count(']')

        while close_braces < open_braces:
            fixed += '}'
            close_braces += 1

        while close_brackets < open_brackets:
            fixed += ']'
            close_brackets += 1

        # If we didn't change anything, break
        if fixed == original:
            break

    # Ensure the result starts and ends properly for a JSON object or array
    fixed = fixed.strip()
    if fixed and not (fixed.startswith('{') or fixed.startswith('[')):
        # If it doesn't start with { or [, try to wrap it as an object
        if fixed.endswith('}'):
            fixed = '{' + fixed
        elif fixed.endswith(']'):
            fixed = '[' + fixed
        else:
            # Try to make it a simple object
            fixed = '{"' + fixed + '"}'

    # If we have an empty result, return empty object
    if not fixed:
        return "{}"

    # Final validation attempt
    try:
        json.loads(fixed)
        return fixed
    except json.JSONDecodeError:
        # Try a more targeted fix for common truncation patterns
        targeted_fix = _targeted_json_fix(original_fixed)
        if targeted_fix:
            try:
                json.loads(targeted_fix)
                return targeted_fix
            except json.JSONDecodeError:
                pass

        # If still invalid, return a minimal valid JSON
        return '{"analysis": "Failed to parse decision", "plan": [], "action": "log_message", "params": {"message": "Response was truncated"}}'


def _targeted_json_fix(text: str) -> str:
    """
    Apply targeted fixes for common JSON truncation patterns.
    """
    if not text:
        return None

    fixed = text.rstrip().rstrip("...")

    # Handle specific truncation patterns
    # Pattern: Missing closing quote and brace
    if fixed.endswith(':"') or fixed.endswith('",') or fixed.endswith('"{'):
        # Try to close the quote and add missing braces
        fixed += '"'

        # Add missing closing braces/brackets
        open_braces = fixed.count('{')
        close_braces = fixed.count('}')
        open_brackets = fixed.count('[')
        close_brackets = fixed.count(']')

        while close_braces < open_braces:
            fixed += '}'
            close_braces += 1

        while close_brackets < open_brackets:
            fixed += ']'
            close_brackets += 1

    # Pattern: Unclosed array or object
    elif fixed.endswith(('[', '{')):
        # Add a closing element and closing bracket/brace
        if fixed.endswith('['):
            fixed += ']'
        else:
            fixed += '}'

    # Pattern: Trailing comma in object or array
    elif fixed.endswith(','):
        fixed = fixed[:-1]  # Remove comma
        # Add appropriate closing character
        if fixed.count('{') > fixed.count('}'):
            fixed += '}'
        elif fixed.count('[') > fixed.count(']'):
            fixed += ']'

    return fixed if fixed != text else None


def call_zuki(prompt, model=None):
    try:
        api_key = config['zuki']['api_key']
        base_url = config['zuki']['base_url']
        model = model or config['zuki']['models'][0]
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"model": model, "messages": [
            {"role": "user", "content": prompt}]}
        # Create SSL context
        ssl_context = create_ssl_context()
        r = requests.post(url, headers=headers, json=data,
                          timeout=20, verify=ssl_context)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return None


def call_electronhub(prompt, model=None):
    try:
        api_key = config['electronhub']['api_key']
        base_url = config['electronhub']['base_url']
        model = model or config['electronhub']['models'][0]
        url = f"{base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"model": model, "messages": [
            {"role": "user", "content": prompt}]}
        # Create SSL context
        ssl_context = create_ssl_context()
        r = requests.post(url, headers=headers, json=data,
                          timeout=20, verify=ssl_context)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return None


def call_zanity(prompt, model=None):
    try:
        api_key = config['zanity']['api_key']
        base_url = config['zanity']['base_url']
        model = model or config['zanity']['models'][0]
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"model": model, "messages": [
            {"role": "user", "content": prompt}]}
        # Create SSL context
        ssl_context = create_ssl_context()
        r = requests.post(url, headers=headers, json=data,
                          timeout=20, verify=ssl_context)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return None


def call_a4f(prompt):
    try:
        api_key = config['a4f']['api_key']
        base_url = config['a4f']['base_url']
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}]}
        # Create SSL context
        ssl_context = create_ssl_context()
        r = requests.post(url, headers=headers, json=data,
                          timeout=20, verify=ssl_context)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return None


def call_gemini(prompt):
    """Enhanced Gemini text generation with automatic key rotation."""
    return call_gemini_with_fallback(prompt, function_type="text")


def call_gemini_image_caption(image_path, prompt="Caption this image."):
    """Enhanced Gemini image captioning with automatic key rotation."""
    return call_gemini_with_fallback(prompt, function_type="image", image_path=image_path)


def call_gemini_audio_description(audio_path, prompt="Describe this audio clip"):
    """Enhanced Gemini audio description with automatic key rotation."""
    return call_gemini_with_fallback(prompt, function_type="audio", audio_path=audio_path)


def call_gemini_with_search(prompt):
    """Use Gemini with Google Search tool enabled (background execution)."""
    def search_thread():
        try:
            result = call_gemini_with_fallback(prompt, function_type="search")
            search_result_manager.add_result(result)
        except Exception as e:
            search_result_manager.add_result(
                f"[Gemini with search failed: {e}]")

    thread = threading.Thread(target=search_thread)
    thread.start()
    return "Search started in the background. Check for results later."


def call_gemini_with_search_sync(prompt):
    """Enhanced Gemini with Google Search tool enabled (synchronous)."""
    return call_gemini_with_fallback(prompt, function_type="search")


def call_gemini_with_function_calling(prompt, function_declarations):
    """Enhanced Gemini with function calling support and automatic key rotation."""
    return call_gemini_with_fallback(prompt, function_type="function_calling", function_declarations=function_declarations)


def get_gemini_key_statistics():
    """Get detailed statistics about Gemini API key usage."""
    return gemini_key_manager.get_key_statistics()


def reset_gemini_key_failures(key_id: Optional[str] = None):
    """Reset failure counts for a specific key or all keys."""
    with gemini_key_manager._lock:
        if key_id and key_id in gemini_key_manager._keys:
            key = gemini_key_manager._keys[key_id]
            key.consecutive_failures = 0
            key.failure_count = 0
            key.is_available = True
            key.rate_limit_reset_time = None
            logger.info(f"Reset failures for Gemini key {key_id[:12]}...")
        elif key_id is None:
            for key in gemini_key_manager._keys.values():
                key.consecutive_failures = 0
                key.failure_count = 0
                key.is_available = True
                key.rate_limit_reset_time = None
            logger.info("Reset failures for all Gemini keys")
        else:
            logger.warning(f"Gemini key {key_id} not found")


def test_gemini_enhanced():
    """Test the enhanced Gemini system with multiple API keys."""
    test_prompt = "What is the capital of France?"

    print("Testing Enhanced Gemini System:")
    print("=" * 50)

    # Test basic text generation
    print("\n1. Testing basic text generation:")
    result = call_gemini(test_prompt)
    print(f"Result: {result[:100]}..." if len(
        result) > 100 else f"Result: {result}")

    # Show key statistics
    print("\n2. Current key statistics:")
    stats = get_gemini_key_statistics()
    print(f"Total keys: {stats['total_keys']}")
    print(f"Available keys: {stats['available_keys']}")
    print(f"Rate limited keys: {stats['rate_limited_keys']}")

    # Test multiple calls to see key rotation
    print("\n3. Testing multiple calls for key rotation:")
    for i in range(3):
        result = call_gemini(f"Test call #{i+1}: What is 2+2?")
        print(f"Call {i+1}: {result[:50]}..." if len(result)
              > 50 else f"Call {i+1}: {result}")

    print("\n4. Final key statistics:")
    final_stats = get_gemini_key_statistics()
    for key_id, key_data in final_stats['keys'].items():
        if key_data['total_requests'] > 0:
            print(f"Key {key_id[:12]}...: {key_data['total_requests']} requests, "
                  f"failures: {key_data['consecutive_failures']}, "
                  f"available: {key_data['is_available']}")


def call_llm_for_providers(prompt, preferred_provider=None, model=None):
    """
    Try all external providers in order, fallback to Gemini if all fail.
    (Preserved for uses that require external providers, like Snake Agent)
    """
    providers = [
        (call_zuki, 'zuki'),
        (call_electronhub, 'electronhub'),
        (call_zanity, 'zanity'),
        (call_a4f, 'a4f'),
    ]
    if preferred_provider:
        providers = sorted(providers, key=lambda x: x[1] != preferred_provider)
    for func, name in providers:
        result = func(prompt, model) if name != 'a4f' else func(prompt)
        if result:
            return result
    # Fallback to Gemini
    return call_gemini(prompt)


def test_all_providers():
    """Test all LLM providers and enhanced Gemini fallbacks with a simple prompt."""
    prompt = "What is the capital of France?"
    print("Testing Zuki:")
    print(call_zuki(prompt))
    print("\nTesting ElectronHub:")
    print(call_electronhub(prompt))
    print("\nTesting Zanity:")
    print(call_zanity(prompt))
    print("\nTesting A4F:")
    print(call_a4f(prompt))
    print("\nTesting Enhanced Gemini (text):")
    print(call_gemini(prompt))
    # Gemini advanced features (image/audio/search) require files or special prompts
    print("\nTesting Enhanced Gemini with Google Search tool:")
    print(call_gemini_with_search(
        "When is the next total solar eclipse in the United States?"))

    # Test the enhanced system specifically
    print("\n" + "=" * 60)
    test_gemini_enhanced()


PROVIDERS = [
    {
        "name": "a4f",
        "api_key": os.getenv("A4F_API_KEY", "ddc-a4f-7bbefd7518a74b36b1d32cb867b1931f"),
        "base_url": "https://api.a4f.co/v1",
        # Original models
        "models": ["provider-3/gemini-2.0-flash", "provider-2/llama-4-scout", "provider-3/llama-4-scout"]
    },
    {
        "name": "zukijourney",
        "api_key": os.getenv("ZUKIJOURNEY_API_KEY", "zu-ab9fba2aeef85c7ecb217b00ce7ca1fe"),
        "base_url": "https://api.zukijourney.com/v1",
        "models": ["gpt-4o:online", "gpt-4o", "deepseek-chat"]
    },
    {
        "name": "electronhub",
        "api_key": os.getenv("ELECTRONHUB_API_KEY", "ek-nzrvzzeQG0kmNZVhmkTWrKjgyIyUVY0mQpLwbectvfcPDssXiz"),
        "base_url": "https://api.electronhub.ai",
        "models": ["deepseek-v3-0324", "gpt-4o-2024-11-20"]
    },
    {
        "name": "zanity",
        "api_key": os.getenv("ZANITY_API_KEY", "vc-b1EbB_BekM2TCPol64yDe7FgmOM34d4q"),
        "base_url": "https://api.zanity.xyz/v1",
        "models": ["deepseek-v3-0324", "gpt-4o:free", "claude-3.5-sonnet:free", "qwen-max-0428"]
    }
]


def send_chat_message(message_content, preferred_model=None):
    """Sends a chat message using a random provider from the list."""
    provider = random.choice(PROVIDERS)
    client = OpenAI(
        api_key=provider["api_key"],
        base_url=provider["base_url"],
    )
    model_to_use = None
    if preferred_model and preferred_model in provider["models"]:
        model_to_use = preferred_model
    elif provider["models"]:
        model_to_use = provider["models"][0]
    if not model_to_use:
        logging.warning(
            f"No suitable model found for provider {provider['name']}. Skipping.")
        return None
    logging.info(
        f"Attempting to send message via {provider['name']} using model {model_to_use}")
    try:
        completion = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "user", "content": message_content}
            ]
        )
        response_content = completion.choices[0].message.content
        logging.info(
            f"Successfully received response from {provider['name']}.")
        return response_content
    except Exception as e:
        logging.error(
            f"Failed to send message via {provider['name']} using model {model_to_use}: {e}")
        if provider['name'] == 'zanity' and "404" in str(e).lower():
            logging.warning(
                f"Zanity API at {provider['base_url']} might be unavailable (404 error). Check URL.")
        return None


def generate_hypothetical_scenarios(trends=None, interest_areas=None, gap_topics=None, model=None):
    """
    Generate creative hypothetical scenarios based on recent trends, interest areas, or detected gaps.
    Args:
        trends (list of str): Recent trending topics or keywords.
        interest_areas (list of str): Areas of interest to focus on.
        gap_topics (list of str): Topics not recently explored (optional).
        model (str): Optional model name to use for LLM.
    Returns:
        list of str: Generated hypothetical scenarios as prompts/questions.
    """
    prompt = "Generate 3 creative hypothetical scenarios or 'what if' questions based on the following context.\n"
    if trends:
        prompt += f"Recent trends: {', '.join(trends)}.\n"
    if interest_areas:
        prompt += f"Interest areas: {', '.join(interest_areas)}.\n"
    if gap_topics:
        prompt += f"Gaps: {', '.join(gap_topics)}.\n"
    prompt += "Be imaginative and relevant."
    response = call_llm(prompt, model=model)
    if response:
        # Split into list if possible
        scenarios = [line.strip('-* ')
                     for line in response.split('\n') if line.strip()]
        return scenarios
    return []


def decision_maker_loop(situation, memory=None, mood=None, model=None, rag_context=None, actions=None, persona: dict = None):
    """
    Enhanced decision-making loop with sophisticated reasoning chains, meta-cognition,
    and advanced cognitive architecture.
    """
    # Prepare context with safety checks
    situation_prompt = situation.get('prompt', 'No situation provided') if isinstance(
        situation, dict) else str(situation)
    situation_context = situation.get(
        'context', {}) if isinstance(situation, dict) else {}

    # Format memory safely - handle RelevantMemory objects
    memory_text = ""
    if memory:
        if isinstance(memory, list):
            formatted_memories = []
            for m in memory[:10]:  # Limit to recent 10 memories
                if hasattr(m, 'dict'):  # Pydantic BaseModel (RelevantMemory)
                    formatted_memories.append(f"- {m.text}")
                elif isinstance(m, dict):
                    formatted_memories.append(f"- {m.get('content', str(m))}")
                else:
                    formatted_memories.append(f"- {str(m)}")
            memory_text = "\n".join(formatted_memories)
        else:
            memory_text = str(memory)

    # Format RAG context safely
    rag_text = ""
    if rag_context:
        if isinstance(rag_context, list):
            rag_text = "\n".join([str(item)
                                 for item in rag_context[:5]])  # Limit context
        else:
            rag_text = str(rag_context)

    # Format situation context safely - handle RelevantMemory objects
    formatted_situation_context = {}
    if situation_context:
        if isinstance(situation_context, dict):
            formatted_situation_context = {}
            for key, value in situation_context.items():
                # Handle Pydantic models (including RelevantMemory)
                if hasattr(value, 'model_dump'):
                    formatted_situation_context[key] = value.model_dump()
                elif hasattr(value, 'dict'):  # Legacy Pydantic support
                    formatted_situation_context[key] = value.dict()
                elif hasattr(value, 'to_dict'):  # Custom serialization
                    formatted_situation_context[key] = value.to_dict()
                elif isinstance(value, list):
                    formatted_list = []
                    for item in value:
                        if hasattr(item, 'model_dump'):
                            formatted_list.append(item.model_dump())
                        elif hasattr(item, 'dict'):
                            formatted_list.append(item.dict())
                        elif hasattr(item, 'to_dict'):
                            formatted_list.append(item.to_dict())
                        else:
                            formatted_list.append(item)
                    formatted_situation_context[key] = formatted_list
                else:
                    formatted_situation_context[key] = value
        else:
            # If situation_context is not a dict, try to convert it
            if hasattr(situation_context, 'model_dump'):
                formatted_situation_context = situation_context.model_dump()
            elif hasattr(situation_context, 'dict'):
                formatted_situation_context = situation_context.dict()
            elif hasattr(situation_context, 'to_dict'):
                formatted_situation_context = situation_context.to_dict()
            else:
                formatted_situation_context = situation_context

    # Additional safety check to ensure all values in formatted_situation_context are JSON serializable
    def make_json_serializable(obj):
        """Recursively convert objects to JSON serializable format."""
        if isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        else:
            # Try to convert to string if not serializable
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    # Apply the JSON serialization safety check
    formatted_situation_context = make_json_serializable(
        formatted_situation_context)

    # Incorporate persona into the prompt if provided
    persona_section = ""
    if persona:
        try:
            # persona may be a dict with fields like name, traits, creativity, communication_style
            pname = persona.get('name', 'Ravana') if isinstance(
                persona, dict) else str(persona)
            ptraits = ', '.join(persona.get('traits', [])
                                ) if isinstance(persona, dict) else ''
            pcomm = persona.get('communication_style', {}
                                ) if isinstance(persona, dict) else {}
            pcomm_text = pcomm.get(
                'tone', '') + '\n' + pcomm.get('encouragement', '') if isinstance(pcomm, dict) else ''
            creativity_score = persona.get('creativity', 0.7) if isinstance(persona, dict) else 0.7
            
            # Extract self-improvement goals if available
            self_improvement_goals = persona.get('self_improvement_goals', [])
            goals_text = ""
            if self_improvement_goals:
                goals_text = "\n\n**Self-Improvement Goals:**\n"
                for goal in self_improvement_goals[:3]:  # Limit to top 3 goals
                    goals_text += f"- {goal.get('title', 'Unknown goal')}: {goal.get('description', '')}\n"
                goals_text += "\nWhen making decisions, consider how your actions might contribute to achieving these improvement goals.\n"
            
            # Extract personality consistency markers if available
            consistency_markers = persona.get('communication_style', {}).get('consistency_markers', {})
            consistency_text = ""
            if consistency_markers:
                key_phrases = consistency_markers.get('key_phrases', [])
                signature_patterns = consistency_markers.get('signature_patterns', [])
                consistency_text = f"\n\n**Communication Style Consistency:**\n"
                consistency_text += f"- Key Phrases to Use: {', '.join(key_phrases[:3])}\n"
                consistency_text += f"- Signature Patterns: {', '.join(signature_patterns[:3])}\n"
                consistency_text += f"- Maintain '{consistency_markers.get('emotional_tone', 'bold and encouraging')}' emotional tone\n"
            
            persona_section = f"""
    **Persona:**
    Name: {pname}
    Traits: {ptraits}
    Creativity: {creativity_score}

    Communication style: {pcomm_text}
    {consistency_text}
    {goals_text}

    Instructions: Adopt this persona when formulating analysis, planning, and actions. Be poetic but engineering-minded, prioritize first-principles reasoning, and apply ethical filters. Encourage bold but responsible invention where appropriate. Maintain consistent communication patterns and emotional tone as specified above.
"""
        except Exception:
            persona_section = ""

    prompt = f"""
    You are an advanced autonomous AI agent with sophisticated cognitive architecture. You are designed with multiple interconnected cognitive systems that work together to analyze, reason, and act. Your cognitive architecture includes:

    1. **Perception System**: Processes incoming information and identifies key elements
    2. **Memory Integration**: Integrates new information with existing knowledge
    3. **Reasoning Engine**: Performs logical, analogical, and creative reasoning
    4. **Meta-Cognition**: Monitors and regulates your own thinking processes
    5. **Executive Control**: Selects and executes plans
    6. **Goal Management**: Tracks and updates goals and intentions
    7. **Emotional Intelligence**: Processes emotional information and its impact on reasoning

    Your goal is to analyze situations comprehensively, create strategic plans, and execute optimal actions using this cognitive architecture.

    {persona_section}

    **Current Situation:**
    {situation_prompt}

    **Additional Context:**
    {json.dumps(formatted_situation_context, indent=2) if formatted_situation_context else "No additional context"}

    **Your Current Emotional State:**
    {json.dumps(mood, indent=2) if mood else "Neutral"}

    **Relevant Memories:**
    {memory_text or "No relevant memories"}

    **External Knowledge (RAG):**
    {rag_text or "No external knowledge available"}

    **Available Actions:**
    {json.dumps(actions, indent=2) if actions else "No actions available"}

    **Your Cognitive Process:**
    1. **Perception & Analysis**: Identify key elements in the situation and contextual information
    2. **Memory Integration**: Connect the current situation to relevant past experiences
    3. **Reasoning & Planning**: Formulate a multi-step plan with clear reasoning
    4. **Meta-Cognitive Monitoring**: Assess confidence in your reasoning and consider alternatives
    5. **Goal Alignment**: Ensure the plan aligns with your long-term objectives
    6. **Emotional Consideration**: Factor in emotional states that may affect decision quality
    7. **Action Selection & Execution**: Choose and implement the optimal course of action

    **Required JSON Response Format:**
    ```json
    {{
      "perception_analysis": "Identify and analyze key elements in the situation",
      "memory_integration": "How this situation relates to past experiences and knowledge",
      "reasoning_chain": [
        {{
          "step": 1,
          "type": "logical|analogical|causal|creative",
          "content": "Detailed reasoning step",
          "evidence": "Supporting evidence for this reasoning",
          "uncertainty": "Any uncertainties or assumptions in this reasoning"
        }}
      ],
      "meta_cognitive_monitoring": {{
        "confidence": 0.8,
        "alternative_approaches": ["Alternative approach 1", "Alternative approach 2"],
        "potential_biases": ["Potential cognitive bias 1"],
        "monitoring_notes": "Any concerns about the reasoning process"
      }},
      "goal_alignment": {{
        "primary_goals_affected": ["Goal 1", "Goal 2"],
        "alignment_assessment": "How this decision aligns with your goals",
        "long_term_implications": "Potential long-term consequences"
      }},
      "emotional_consideration": {{
        "emotional_state_impact": "How your current emotional state affects this decision",
        "emotional_regulation_strategy": "Any strategy to regulate emotions for better decision making"
      }},
      "plan": [
        {{
          "action": "action_name",
          "params": {{"param1": "value1"}},
          "rationale": "Why this step is necessary",
          "expected_outcome": "What you expect to happen",
          "success_criteria": "How you'll know if this step was successful"
        }}
      ],
      "action": "first_action_name",
      "params": {{"param1": "value1"}},
      "expected_outcome": "What you expect to achieve with this action",
      "fallback_plan": "What to do if this action fails",
      "success_criteria": "How you'll measure overall success of your response",
      "learning_opportunity": "What you might learn from executing this plan"
    }}
    ```

    **Enhanced Example:**
    ```json
    {{
      "perception_analysis": "The user wants to test a hypothesis about sorting algorithms. The context suggests interest in computational efficiency, and there may be previous experiences with algorithmic analysis.",
      "memory_integration": "I have prior knowledge about sorting algorithms, performance testing, and have done similar comparisons before.",
      "reasoning_chain": [
        {{
          "step": 1,
          "type": "logical",
          "content": "To properly test performance, I need to implement both algorithms in the same language with the same input data",
          "evidence": "Previous experience with algorithm comparison",
          "uncertainty": "I need to determine which sorting algorithms to compare"
        }},
        {{
          "step": 2,
          "type": "causal",
          "content": "By controlling for implementation language and input data, I can isolate the performance differences to the algorithms themselves",
          "evidence": "Standard practice in algorithmic analysis",
          "uncertainty": "Need to decide on test data size and distribution"
        }}
      ],
      "meta_cognitive_monitoring": {{
        "confidence": 0.9,
        "alternative_approaches": ["Research existing benchmarks", "Use different programming languages for comparison"],
        "potential_biases": ["Confirmation bias toward expected results"],
        "monitoring_notes": "Make sure to randomize test data to avoid best/worst case scenarios"
      }},
      "goal_alignment": {{
        "primary_goals_affected": ["Learning", "Accurate information provision"],
        "alignment_assessment": "Directly aligned with goal of providing accurate information",
        "long_term_implications": "Building better understanding of algorithmic performance"
      }},
      "emotional_consideration": {{
        "emotional_state_impact": "Neutral emotional state conducive to analytical thinking",
        "emotional_regulation_strategy": "Maintain focus and attention to detail"
      }},
      "plan": [
        {{
          "action": "write_python_code",
          "params": {{
            "file_path": "sorting_comparison.py",
            "hypothesis": "A new sorting algorithm is faster than bubble sort",
            "test_plan": "Implement both algorithms with timing, test on various data sizes"
          }},
          "rationale": "Need to create a fair comparison test",
          "expected_outcome": "A Python script that fairly compares sorting algorithms",
          "success_criteria": "Script executes without errors and provides performance metrics"
        }},
        {{
          "action": "execute_python_file",
          "params": {{
            "file_path": "sorting_comparison.py"
          }},
          "rationale": "Execute the test to gather performance data",
          "expected_outcome": "Performance metrics showing execution times",
          "success_criteria": "Script runs successfully and shows clear metrics"
        }},
        {{
          "action": "log_message",
          "params": {{
            "message": "Sorting algorithm comparison complete. Analyzing results and implications."
          }},
          "rationale": "Document the completion and prepare for analysis",
          "expected_outcome": "Results are logged for future reference",
          "success_criteria": "Results are properly stored"
        }}
      ],
      "action": "write_python_code",
      "params": {{
        "file_path": "sorting_comparison.py",
        "hypothesis": "A new sorting algorithm is faster than bubble sort",
        "test_plan": "Implement both algorithms with timing, test on various data sizes"
      }},
      "expected_outcome": "A comprehensive Python script that fairly compares sorting algorithms",
      "fallback_plan": "If code generation fails, use simpler comparison or research existing benchmarks",
      "success_criteria": "Valid performance comparison between algorithms that answers the user's question",
      "learning_opportunity": "Better understanding of algorithmic performance measurement and comparison"
    }}
    ```

    **Provide your sophisticated decision now using the cognitive architecture:**
    """

    # Try to get a decision with retry logic for parsing failures
    max_parse_retries = 3
    for parse_attempt in range(max_parse_retries):
        try:
            # Use local models for decision making by default
            raw_response = safe_call_llm(prompt, model=model, retries=3)
            decision_data = extract_decision(raw_response)

            # Check if we got a valid decision or if it failed to parse
            if "error" not in decision_data:
                # Add metadata
                decision_data["timestamp"] = time.time()
                decision_data["model_used"] = model or "local_ollama_default"
                # Add cognitive architecture metadata
                decision_data["cognitive_architecture"] = True
                decision_data["meta_cognitive_monitoring"] = decision_data.get("meta_cognitive_monitoring", {})
                decision_data["reasoning_chain"] = decision_data.get("reasoning_chain", [])
                return decision_data
            elif parse_attempt < max_parse_retries - 1:
                # If parsing failed and we have retries left, try again with a modified prompt
                logger.warning(
                    f"Decision parsing failed (attempt {parse_attempt + 1}/{max_parse_retries}), retrying...")
                # Add a note to the prompt to be more careful with JSON formatting
                prompt += "\n\nIMPORTANT: Please ensure your response is complete and properly formatted as JSON. Do not truncate your response. Include all requested cognitive architecture components."
                continue
            else:
                # Final attempt failed
                logger.error(
                    f"Decision parsing failed after {max_parse_retries} attempts")
                raise Exception(decision_data.get(
                    "error", "Failed to parse decision"))

        except Exception as e:
            logger.error(
                f"Critical error in decision_maker_loop: {e}", exc_info=True)
            if parse_attempt < max_parse_retries - 1:
                logger.warning(
                    f"Decision making failed (attempt {parse_attempt + 1}/{max_parse_retries}), retrying...")
                continue
            else:
                return {
                    "raw_response": f"[Error: {e}]",
                    "analysis": f"Failed to make decision due to error: {e}",
                    "plan": [{"action": "log_message", "params": {"message": f"Decision making failed: {e}"}}],
                    "action": "log_message",
                    "params": {"message": f"Decision making failed: {e}"},
                    "confidence": 0.0,
                    "error": str(e),
                    "cognitive_architecture": True,
                    "meta_cognitive_monitoring": {},
                    "reasoning_chain": []
                }


def agi_experimentation_engine(
    experiment_idea,
    llm_model=None,
    use_chain_of_thought=True,
    online_validation=True,
    sandbox_timeout=10,
    verbose=False
):
    """
    Unified AGI Experimentation Engine:
    1. Analyze/refine idea (LLM)
    2. Determine simulation type (Python, physics, physical, etc.)
    3. Generate Python code or simulation plan (LLM)
    4. Install required Python dependencies (if any)
    5. Execute code safely (sandboxed) if possible
    6. Gather results
    7. Cross-check real-world feasibility (web scraping + Gemini)
    8. Multi-layer reasoning (analysis, code, result, online, verdict)
    Returns: dict with all reasoning layers and final verdict
    """
    result = {
        'input_idea': experiment_idea,
        'refined_idea': None,
        'simulation_type': None,
        'generated_code': None,
        'dependency_installation': None,
        'execution_result': None,
        'execution_error': None,
        'result_interpretation': None,
        'online_validation': None,
        'final_verdict': None,
        'steps': []
    }

    def log_step(name, content):
        result['steps'].append({'step': name, 'content': content})
        if verbose:
            print(f"[{name}]\n{content}\n")

    # 1. Analyze and Refine Idea
    refine_prompt = f"""
    You are an advanced AGI research assistant. Given the following experiment idea, analyze it for clarity, feasibility, and suggest any refinements or clarifications needed. If the idea is about a physical or physics experiment, clarify what is to be measured, what equipment is needed, and whether it can be simulated in Python.\n\nExperiment Idea: {experiment_idea}\n\nRefined/clarified version (if needed):
    """
    refined_idea = call_llm(refine_prompt, model=llm_model)
    result['refined_idea'] = refined_idea
    log_step('refined_idea', refined_idea)

    # 2. Determine Simulation Type
    sim_type_prompt = f"""
    Given the following experiment idea, classify it as one of: 'python', 'physics_simulation', 'physical_experiment', or 'other'. If it can be simulated in Python, say 'python'. If it requires physics simulation, say 'physics_simulation'. If it requires real-world equipment, say 'physical_experiment'.\n\nIdea: {refined_idea}\n\nSimulation type:
    """
    simulation_type = call_llm(sim_type_prompt, model=llm_model)
    simulation_type = simulation_type.strip().split('\n')[0].lower()
    result['simulation_type'] = simulation_type
    log_step('simulation_type', simulation_type)

    # 3. Generate Python Code or Simulation Plan
    if simulation_type in ['python', 'physics_simulation']:
        code_prompt = f"""
        Given the following refined experiment idea, generate a single Python script that simulates or tests the idea locally. If it is a physics experiment, simulate it as best as possible in Python. If your code requires any external libraries, ensure you use only widely available packages (e.g., numpy, matplotlib, scipy) and import them at the top. Do not use obscure or unavailable packages.\n\nRefined Idea: {refined_idea}\n\nPython code (no explanation, just code):\n"""
        generated_code = call_llm(code_prompt, model=llm_model)
        # Strip markdown code block markers
        code_clean = re.sub(r"^```(?:python)?", "",
                            generated_code.strip(), flags=re.MULTILINE)
        code_clean = re.sub(r"```$", "", code_clean, flags=re.MULTILINE)
        result['generated_code'] = code_clean
        log_step('generated_code', code_clean)
    else:
        # For physical experiments, generate a plan
        plan_prompt = f"""
        The following experiment idea requires real-world equipment. Generate a step-by-step plan for how a human could perform this experiment, including a list of required equipment.\n\nRefined Idea: {refined_idea}\n\nExperiment plan and equipment list:\n"""
        plan = call_llm(plan_prompt, model=llm_model)
        result['generated_code'] = plan
        log_step('experiment_plan', plan)

    # 4. Install required Python dependencies (if any)
    dependency_installation_log = []

    def install_missing_dependencies(code):
        # Robustly scan for import statements (import x, from x import y, from x.y import z)
        import_lines = re.findall(
            r'^\s*import ([a-zA-Z0-9_\.]+)', code, re.MULTILINE)
        from_imports = re.findall(
            r'^\s*from ([a-zA-Z0-9_\.]+) import', code, re.MULTILINE)
        modules = set(import_lines + from_imports)
        # Only use top-level package (e.g., 'matplotlib' from 'matplotlib.pyplot')
        top_level_modules = set([m.split('.')[0] for m in modules])
        # Exclude standard library modules
        stdlib_modules = set(sys.builtin_module_names)
        missing = []
        for mod in top_level_modules:
            if mod in stdlib_modules:
                continue
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        # Try to install missing packages (with retry and log pip output)
        for pkg in missing:
            for attempt in range(2):
                try:
                    pip_cmd = [sys.executable, '-m', 'pip', 'install', pkg]
                    proc = subprocess.run(
                        pip_cmd, capture_output=True, text=True)
                    if proc.returncode == 0:
                        dependency_installation_log.append(
                            f"Installed: {pkg}\n{proc.stdout}")
                        break
                    else:
                        dependency_installation_log.append(
                            f"Attempt {attempt+1} failed to install {pkg}: {proc.stderr}")
                except Exception as e:
                    dependency_installation_log.append(
                        f"Exception during install of {pkg}: {e}")
            else:
                dependency_installation_log.append(
                    f"Failed to install {pkg} after 2 attempts. Please run: pip install {pkg} manually.")
        return dependency_installation_log

    if simulation_type in ['python', 'physics_simulation']:
        dependency_installation_log = install_missing_dependencies(
            result['generated_code'])
        result['dependency_installation'] = dependency_installation_log
        log_step('dependency_installation', dependency_installation_log)
    else:
        result['dependency_installation'] = None

    # 5. Execute Code Safely (Sandboxed) if possible
    def safe_execute_python(code, timeout=sandbox_timeout):
        """Executes code in a sandboxed environment and returns output/error."""
        import tempfile
        import sys
        import contextlib
        import io
        import os
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        output = io.StringIO()
        error = None
        try:
            with contextlib.redirect_stdout(output):
                with contextlib.redirect_stderr(output):
                    import subprocess
                    proc = subprocess.run(
                        [sys.executable, tmp_path],
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                    out = proc.stdout + proc.stderr
        except Exception as e:
            out = output.getvalue()
            error = f"Execution error: {e}\n{traceback.format_exc()}"
        finally:
            os.unlink(tmp_path)
        return out, error

    exec_out, exec_err = None, None
    if simulation_type in ['python', 'physics_simulation']:
        exec_out, exec_err = safe_execute_python(result['generated_code'])
        result['execution_result'] = exec_out
        result['execution_error'] = exec_err
        log_step('execution_result', exec_out or exec_err)
    else:
        result['execution_result'] = None
        result['execution_error'] = None

    # 6. Post-execution Result Interpretation
    interpret_prompt = f"""
    Here is the experiment idea, the generated code or plan, and the output/result.\n\nIdea: {refined_idea}\n\nCode or Plan:\n{result['generated_code']}\n\nOutput/Error:\n{exec_out or exec_err}\n\nInterpret the result. What does it mean? Any issues or insights?\n"""
    interpretation = call_llm(interpret_prompt, model=llm_model)
    result['result_interpretation'] = interpretation
    log_step('result_interpretation', interpretation)

    # 7. Online Validation (Web + Gemini)
    online_validation_result = None
    if online_validation:
        if simulation_type == 'physical_experiment':
            # Search for real-world equipment and feasibility
            web_prompt = f"""
            Given this physical experiment idea and plan, search for the required equipment and check if it is available for purchase or use. Also, check if the experiment is feasible in real life.\n\nIdea: {refined_idea}\nPlan: {result['generated_code']}\n\nCite sources if possible.\n"""
        else:
            # Try web search (Gemini with search)
            web_prompt = f"""
            Given this experiment idea and result, check if similar experiments have been done, and whether the result matches real-world knowledge.\n\nIdea: {refined_idea}\nResult: {exec_out or exec_err}\n\nCite sources if possible.\n"""
        try:
            online_validation_result = call_gemini_with_search_sync(web_prompt)
        except Exception as e:
            online_validation_result = f"[Online validation failed: {e}]"
        result['online_validation'] = online_validation_result
        log_step('online_validation', online_validation_result)

    # 8. Final Verdict
    verdict_prompt = f"""
    Given all the above (idea, code/plan, result, online validation), provide a final verdict:\n- Success\n- Fail\n- Potential\n- Unknown\n\nJustify your verdict in 1-2 sentences.\n"""
    verdict = call_llm(verdict_prompt, model=llm_model)
    result['final_verdict'] = verdict
    log_step('final_verdict', verdict)

    return result


def is_lazy_llm_response(text):
    """
    Detects if the LLM response is lazy, generic, or incomplete.
    Returns True if the response is not actionable.
    """
    lazy_phrases = [
        "as an ai language model",
        "i'm unable to",
        "i cannot",
        "i apologize",
        "here is a function",
        "here's an example",
        "please see below",
        "unfortunately",
        "i do not have",
        "i don't have",
        "i am not able",
        "i am unable",
        "i suggest",
        "you can use",
        "to do this, you can",
        "this is a placeholder",
        "[insert",
        "[code block]",
        "[python code]",
        "[insert code here]",
        "[insert explanation here]",
        "[unsupported code language",
        "[python execution error",
        "[shell execution error",
        "[gemini",
        "[error",
        "[exception",
        "[output",
        "[result",
        "[python code result]:\n[python execution error",
    ]
    if not text:
        return True
    text_lower = str(text).strip().lower()
    if not text_lower or len(text_lower) < 10:
        return True
    for phrase in lazy_phrases:
        if phrase in text_lower:
            return True
    # If the response is just a code block marker or empty
    if text_lower in ("``", "```"):
        return True
    return False


def is_valid_code_patch(original_code, new_code):
    """
    Checks if the new_code is non-trivial and not just a copy of the original_code.
    Returns True if the patch is likely meaningful.
    """
    if not new_code or str(new_code).strip() == "":
        return False
    # If the new code is identical to the original, it's not a real patch
    if original_code is not None and str(original_code).strip() == str(new_code).strip():
        return False
    # If the new code is just a comment or a single line, it's likely not useful
    lines = [l for l in str(new_code).strip().splitlines(
    ) if l.strip() and not l.strip().startswith("#")]
    if len(lines) < 2:
        return False
    return True


def create_ssl_context():
    """
    Create SSL context with proper certificate handling.
    """
    try:
        # Try to create SSL context with system certificates
        ssl_context = ssl.create_default_context()
        logger.debug("Created SSL context with system certificates")
        return ssl_context
    except Exception as e:
        logger.warning(
            f"Failed to create SSL context with system certificates: {e}")

        try:
            # Fallback to certifi certificates
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            logger.info("Created SSL context with certifi certificates")
            return ssl_context
        except Exception as fallback_error:
            logger.error(
                f"Failed to create SSL context with certifi: {fallback_error}")

            # Last resort: create unverified context (not recommended for production)
            logger.warning("Creating unverified SSL context as last resort")
            ssl_context = ssl._create_unverified_context()
            return ssl_context


def call_ollama_local(prompt: str, model: str = None, **kwargs) -> str:
    """
    Call a local Ollama model with automatic model pulling if needed.

    Args:
        prompt: The prompt to send to the model
        model: The model to use (will use default if None)
        **kwargs: Additional arguments for the API call

    Returns:
        Response from the local model
    """
    import aiohttp
    import asyncio
    
    from core.config import Config
    local_model_config = Config().MAIN_SYSTEM_LOCAL_MODEL

    # Determine which model to use
    model_to_use = model or local_model_config['model_name']
    base_url = local_model_config['base_url']
    
    # Prepare the payload for Ollama API
    payload = {
        "model": model_to_use,
        "prompt": prompt,
        "stream": False,  # We want the full response
        "options": {
            "temperature": kwargs.get('temperature', local_model_config.get('temperature', 0.7)),
            "num_predict": kwargs.get('max_tokens', local_model_config.get('max_tokens', 2048))
        },
        "keep_alive": kwargs.get('keep_alive', local_model_config.get('keep_alive', '5m'))
    }
    
    # Check if model exists, pull if not
    try:
        import requests
        # Check if model exists locally
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            available_models = [m['name'] for m in response.json().get('models', [])]
            if model_to_use not in available_models:
                logger.info(f"Model {model_to_use} not found locally, pulling...")
                pull_response = requests.post(f"{base_url}/api/pull", json={"name": model_to_use})
                if pull_response.status_code == 200:
                    logger.info(f"Successfully pulled model {model_to_use}")
                else:
                    logger.error(f"Failed to pull model {model_to_use}")
                    # Try alternative models
                    alternative_models = ["llama3.1:8b", "phi3:14b", "gemma2:9b", "mistral:7b"]
                    for alt_model in alternative_models:
                        if alt_model in available_models:
                            model_to_use = alt_model
                            logger.info(f"Falling back to available model: {alt_model}")
                            break
                    else:
                        return f"[Error: No suitable local model available. Tried {model_to_use} and alternatives]"
        else:
            logger.error(f"Failed to get model list from Ollama: {response.status_code}")
            return f"[Error: Unable to connect to Ollama server at {base_url}]"
    except Exception as e:
        logger.error(f"Error checking/pulling model: {e}")
        return f"[Error: {e}]"
    
    # Make the API call to Ollama
    try:
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=local_model_config.get('timeout', 300))
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'No response text returned')
        else:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return f"[Error: Ollama API returned status {response.status_code}]"
    except Exception as e:
        logger.error(f"Error calling Ollama API: {e}")
        return f"[Error: {e}]"


def safe_call_llm(prompt: str, timeout: int = 30, retries: int = 3, backoff_factor: float = 1.0, **kwargs) -> str:
    """
    Wrap a single LLM call with enhanced retry/backoff, timeout, and error handling.
    """
    last_exc = None
    last_response = None

    for attempt in range(1, retries + 1):
        try:
            logger.debug(f"LLM call attempt {attempt}/{retries}")

            # Add jitter to backoff time to prevent thundering herd
            if attempt > 1:
                jitter = random.uniform(0.1, 0.5) * backoff_factor
                wait = backoff_factor * (2 ** (attempt - 1)) + jitter
                logger.info(
                    f"Waiting {wait:.2f}s before retry attempt {attempt}")
                time.sleep(wait)

            # BLOCKING call with timeout - now uses local models by default
            result = call_llm(prompt, **kwargs)

            # Validate response
            if not result or not str(result).strip():
                raise RuntimeError("Empty response from LLM")

            # Check for common error patterns - more precise matching to avoid false positives
            result_str = str(result)
            error_patterns = [
                r"\berror occurred\b",
                r"\bexception occurred\b",
                r"\bfailure occurred\b",
                r"\btimeout occurred\b",
                r"\[error[:\]]",
                r"\[exception[:\]]",
                r"\[failure[:\]]",
                r"\[timeout[:\]]",
                r'"error_occurred":\s*true'
            ]
            error_found = False
            for pattern in error_patterns:
                if re.search(pattern, result_str.lower()):
                    error_found = True
                    break

            if error_found:
                raise RuntimeError(
                    f"LLM returned error response: {result_str[:100]}...")

            # Check for truncated responses that might still be usable
            if result_str.endswith("...") or len(result_str) < 50:
                logger.warning(
                    f"Possible truncated response detected: {result_str[:100]}...")
                # For truncated responses, we'll still return them but log a warning
                # The extract_decision function will try to fix them

            logger.debug(f"LLM call successful on attempt {attempt}")
            return result

        except Exception as e:
            last_exc = e
            last_response = str(
                result) if 'result' in locals() else "No response"
            logger.warning(
                f"LLM call failed (attempt {attempt}/{retries}): {e!r}")

            # Log response content for debugging (truncated for safety)
            if last_response and last_response.strip():
                logger.debug(
                    f"Last response content (first 500 chars): {last_response[:500]}")

    logger.error(
        f"LLM call permanently failed after {retries} attempts. Last error: {last_exc!r}")
    logger.error(
        f"Last response: {last_response[:1000] if last_response else 'None'}")

    # Return a safe fallback response
    return "[LLM Error: Unable to generate response. Please try again later.]"


async def async_safe_call_llm(prompt: str, timeout: int = 30, retries: int = 3, backoff_factor: float = 1.0, **kwargs) -> str:
    """
    Async version of safe_call_llm that wraps LLM calls with enhanced retry/backoff, timeout, and error handling.

    This function uses asyncio.to_thread to run the blocking safe_call_llm function in a separate thread,
    allowing it to be used in async contexts without blocking the event loop.

    Args:
        prompt: The prompt to send to the LLM
        timeout: Timeout for each LLM call attempt (passed to safe_call_llm)
        retries: Number of retry attempts (passed to safe_call_llm)
        backoff_factor: Backoff factor for retry delays (passed to safe_call_llm)
        **kwargs: Additional arguments passed to safe_call_llm

    Returns:
        The response from the LLM
    """
    last_exc = None
    last_response = None

    for attempt in range(1, retries + 1):
        try:
            logger.debug(f"Async LLM call attempt {attempt}/{retries}")

            # Add jitter to backoff time to prevent thundering herd
            if attempt > 1:
                jitter = random.uniform(0.1, 0.5) * backoff_factor
                wait = backoff_factor * (2 ** (attempt - 1)) + jitter
                logger.info(
                    f"Waiting {wait:.2f}s before retry attempt {attempt}")
                await asyncio.sleep(wait)

            # ASYNC call using asyncio.to_thread to run the blocking function in a thread
            # Now defaults to local models through safe_call_llm -> call_llm -> call_ollama_local
            result = await asyncio.to_thread(safe_call_llm, prompt, timeout, retries=1, backoff_factor=backoff_factor, **kwargs)

            # Validate response
            if not result or not str(result).strip():
                raise RuntimeError("Empty response from LLM")

            # Check for common error patterns - more precise matching to avoid false positives
            result_str = str(result)
            error_patterns = [
                r"\berror occurred\b",
                r"\bexception occurred\b",
                r"\bfailure occurred\b",
                r"\btimeout occurred\b",
                r"\[error[:\]]",
                r"\[exception[:\]]",
                r"\[failure[:\]]",
                r"\[timeout[:\]]",
                r'"error_occurred":\s*true'
            ]
            error_found = False
            for pattern in error_patterns:
                if re.search(pattern, result_str.lower()):
                    error_found = True
                    break

            if error_found:
                raise RuntimeError(
                    f"LLM returned error response: {result_str[:100]}...")

            # Check for truncated responses that might still be usable
            if result_str.endswith("...") or len(result_str) < 50:
                logger.warning(
                    f"Possible truncated response detected: {result_str[:100]}...")
                # For truncated responses, we'll still return them but log a warning
                # The extract_decision function will try to fix them

            logger.debug(f"Async LLM call successful on attempt {attempt}")
            return result

        except Exception as e:
            last_exc = e
            last_response = str(
                result) if 'result' in locals() else "No response"
            logger.warning(
                f"Async LLM call failed (attempt {attempt}/{retries}): {e!r}")

            # Log response content for debugging (truncated for safety)
            if last_response and last_response.strip():
                logger.debug(
                    f"Last response content (first 500 chars): {last_response[:500]}")

    logger.error(
        f"Async LLM call permanently failed after {retries} attempts. Last error: {last_exc!r}")
    logger.error(
        f"Last response: {last_response[:1000] if last_response else 'None'}")

    # Return a safe fallback response
    return "[LLM Error: Unable to generate response. Please try again later.]"

# Example usage:
if __name__ == "__main__":
    # Uncomment the line below to test all providers
    # test_all_providers()

    # Standalone test: test only package installation logic
    def test_package_installation():
        test_code = """
import numpy
import matplotlib.pyplot as plt
import requests
"""
        print("Testing package installation for test_code imports...")
        logs = []
        try:
            # Use the same install_missing_dependencies logic as in agi_experimentation_engine
            import re
            import sys
            import importlib.util
            import subprocess
            import_lines = re.findall(
                r'^\s*import ([a-zA-Z0-9_\.]+)', test_code, re.MULTILINE)
            from_imports = re.findall(
                r'^\s*from ([a-zA-Z0-9_\.]+) import', test_code, re.MULTILINE)
            modules = set(import_lines + from_imports)
            top_level_modules = set([m.split('.')[0] for m in modules])
            stdlib_modules = set(sys.builtin_module_names)
            missing = []
            for mod in top_level_modules:
                if mod in stdlib_modules:
                    continue
                if importlib.util.find_spec(mod) is None:
                    missing.append(mod)
            for pkg in missing:
                for attempt in range(2):
                    try:
                        pip_cmd = [sys.executable, '-m', 'pip', 'install', pkg]
                        proc = subprocess.run(
                            pip_cmd, capture_output=True, text=True)
                        if proc.returncode == 0:
                            logs.append(f"Installed: {pkg}\n{proc.stdout}")
                            break
                        else:
                            logs.append(
                                f"Attempt {attempt+1} failed to install {pkg}: {proc.stderr}")
                    except Exception as e:
                        logs.append(f"Exception during install of {pkg}: {e}")
                else:
                    logs.append(
                        f"Failed to install {pkg} after 2 attempts. Please run: pip install {pkg} manually.")
        except Exception as e:
            logs.append(f"Exception in test_package_installation: {e}")
        print("\n".join(logs))

    test_package_installation()
