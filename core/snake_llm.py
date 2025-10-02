"""
Snake Agent LLM Interface

This module provides specialized LLM interfaces for the Snake Agent system,
supporting dual models for coding and reasoning tasks using Ollama.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List
import aiohttp
import requests
from core.config import Config

logger = logging.getLogger(__name__)


class SnakeConfigValidator:
    """Validates Snake Agent configuration on startup"""

    @staticmethod
    def validate_ollama_connection() -> bool:
        """Verify Ollama server is accessible"""
        try:
            response = requests.get(
                f"{Config.SNAKE_OLLAMA_BASE_URL}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Cannot connect to Ollama: {e}")
            return False

    @staticmethod
    def validate_electronhub_connection() -> bool:
        """Verify electronhub API is accessible"""
        try:
            headers = {
                "Authorization": f"Bearer {Config.SNAKE_CODING_MODEL.get('api_key', Config.SNAKE_REASONING_MODEL.get('api_key', ''))}",
                "Content-Type": "application/json"
            }
            response = requests.get(
                f"{Config.SNAKE_CODING_MODEL.get('base_url', Config.SNAKE_REASONING_MODEL.get('base_url', 'https://api.electronhub.ai'))}/v1/models", 
                headers=headers, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Cannot connect to electronhub: {e}")
            return False

    @staticmethod
    def validate_models_available() -> tuple[bool, bool, List[str]]:
        """Check if required models are pulled (for Ollama) or accessible (for other providers)"""
        # Determine which provider each model uses
        coding_provider = Config.SNAKE_CODING_MODEL.get('provider', 'ollama').lower()
        reasoning_provider = Config.SNAKE_REASONING_MODEL.get('provider', 'ollama').lower()
        
        coding_available = False
        reasoning_available = False
        available_models = []
        
        if coding_provider == 'ollama':
            try:
                response = requests.get(f"{Config.SNAKE_OLLAMA_BASE_URL}/api/tags", timeout=5)
                if response.status_code == 200:
                    models_data = response.json()
                    available_models.extend([m['name'] for m in models_data.get('models', [])])
                    coding_available = Config.SNAKE_CODING_MODEL['model_name'] in available_models
            except Exception as e:
                logger.debug(f"Error checking coding models: {e}")
        else:
            # For other providers, assume model is available if API is accessible
            try:
                headers = {"Authorization": f"Bearer {Config.SNAKE_CODING_MODEL.get('api_key', '')}", "Content-Type": "application/json"}
                response = requests.get(
                    f"{Config.SNAKE_CODING_MODEL.get('base_url', 'https://api.electronhub.ai')}/v1/models", 
                    headers=headers, timeout=10)
                coding_available = response.status_code == 200
            except Exception as e:
                logger.debug(f"Error checking electronhub coding model: {e}")
        
        if reasoning_provider == 'ollama':
            try:
                response = requests.get(f"{Config.SNAKE_OLLAMA_BASE_URL}/api/tags", timeout=5)
                if response.status_code == 200:
                    models_data = response.json()
                    available_models.extend([m['name'] for m in models_data.get('models', []) if m['name'] not in available_models])
                    reasoning_available = Config.SNAKE_REASONING_MODEL['model_name'] in available_models
            except Exception as e:
                logger.debug(f"Error checking reasoning models: {e}")
        else:
            # For other providers, assume model is available if API is accessible
            try:
                headers = {"Authorization": f"Bearer {Config.SNAKE_REASONING_MODEL.get('api_key', '')}", "Content-Type": "application/json"}
                response = requests.get(
                    f"{Config.SNAKE_REASONING_MODEL.get('base_url', 'https://api.electronhub.ai')}/v1/models", 
                    headers=headers, timeout=10)
                reasoning_available = response.status_code == 200
            except Exception as e:
                logger.debug(f"Error checking electronhub reasoning model: {e}")

        return coding_available, reasoning_available, available_models

    @staticmethod
    def get_startup_report() -> Dict[str, Any]:
        """Generate configuration status report"""
        ollama_connected = SnakeConfigValidator.validate_ollama_connection()
        electronhub_connected = SnakeConfigValidator.validate_electronhub_connection()
        coding_available, reasoning_available, available_models = SnakeConfigValidator.validate_models_available()

        return {
            "ollama_connected": ollama_connected,
            "electronhub_connected": electronhub_connected,
            "coding_model_available": coding_available,
            "reasoning_model_available": reasoning_available,
            "available_models": available_models,
            "config_valid": (ollama_connected or electronhub_connected) and coding_available and reasoning_available
        }


class OllamaClient:
    """Dedicated Ollama client for Snake Agent"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config['base_url']
        self.model_name = config['model_name']

    async def pull_model_if_needed(self) -> bool:
        """Ensure model is available for the provider (only required for Ollama)"""
        try:
            # For Ollama providers, check if model exists
            provider = self.config.get('provider', 'ollama').lower()
            
            if provider == 'ollama':
                # Ollama-specific model checking
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/api/tags") as response:
                        if response.status == 200:
                            models_data = await response.json()
                            available_models = [m['name']
                                                for m in models_data.get('models', [])]

                            if self.model_name not in available_models:
                                logger.info(
                                    f"Model {self.model_name} not found. Attempting to pull...")
                                # Pull the model
                                pull_payload = {"name": self.model_name}
                                async with session.post(f"{self.base_url}/api/pull", json=pull_payload) as pull_response:
                                    if pull_response.status == 200:
                                        # Monitor pull progress
                                        async for line in pull_response.content:
                                            if line:
                                                try:
                                                    progress = json.loads(line)
                                                    if progress.get("status") == "success":
                                                        logger.info(
                                                            f"Successfully pulled model {self.model_name}")
                                                        return True
                                                except json.JSONDecodeError:
                                                    continue
                                    else:
                                        logger.error(
                                            f"Failed to pull model {self.model_name}")
                                        return False
                            else:
                                logger.info(
                                    f"Model {self.model_name} is already available")
                                return True
            else:
                # For other providers like electronhub, models don't need to be "pulled"
                # Just check that the API key is set and the service is accessible
                if self.config.get('api_key'):
                    logger.info(f"Model {self.model_name} configured for {provider} provider (no pull needed)")
                    return True
                else:
                    logger.warning(f"No API key provided for {provider} model {self.model_name}")
                    return False
        except Exception as e:
            logger.error(f"Error checking/pulling model: {e}")
            return False

        return False


class SnakeLLMInterface:
    """Base interface for Snake Agent LLM interactions"""

    def __init__(self, config: Dict[str, Any], model_type: str):
        self.config = config
        self.model_type = model_type  # 'coding' or 'reasoning'
        self.client = OllamaClient(config)
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the LLM interface and ensure model availability"""
        if self._initialized:
            return True

        try:
            # Determine provider type
            provider = self.config.get('provider', 'ollama').lower()
            
            if provider == 'ollama':
                # For Ollama, verify connection to Ollama server
                if not SnakeConfigValidator.validate_ollama_connection():
                    logger.warning(
                        f"Cannot connect to Ollama at {self.config['base_url']}, trying fallback models...")
                    # Try fallback model if available
                    if 'fallback_model' in self.config and self.config['fallback_model']:
                        logger.info(
                            f"Using fallback model: {self.config['fallback_model']}")
                        self.config['model_name'] = self.config['fallback_model']
                        # Update base URL for fallback if needed
                        if self.config.get('fallback_provider') == 'ollama':
                            self.config['base_url'] = Config.SNAKE_OLLAMA_BASE_URL
                    else:
                        raise ConnectionError(
                            f"Cannot connect to Ollama at {self.config['base_url']}")
                else:
                    # Ensure model is available
                    model_available = await self.client.pull_model_if_needed()
                    if not model_available:
                        logger.warning(
                            f"Model {self.config['model_name']} not available, trying fallback...")
                        # Try fallback model if available
                        if 'fallback_model' in self.config and self.config['fallback_model']:
                            self.config['model_name'] = self.config['fallback_model']
                            model_available = await self.client.pull_model_if_needed()
                            if not model_available:
                                raise RuntimeError(
                                    f"Main model {self.config['model_name']} and fallback not available")
                        else:
                            raise RuntimeError(
                                f"Model {self.config['model_name']} not available")
            else:
                # For other providers (electronhub, etc.), just check connectivity
                try:
                    # Just verify that the API key is set and we can potentially connect
                    if not self.config.get('api_key'):
                        logger.warning(f"No API key provided for {provider} provider")
                        # Try fallback if available
                        if 'fallback_model' in self.config and self.config['fallback_model']:
                            logger.info(f"Switching to fallback provider/model: {self.config['fallback_model']}")
                            # Switch to fallback provider
                            self.config['provider'] = self.config['fallback_provider']
                            self.config['model_name'] = self.config['fallback_model']
                            self.config['base_url'] = Config.SNAKE_OLLAMA_BASE_URL
                            provider = self.config['provider']
                        else:
                            raise RuntimeError(f"No API key provided for {provider} provider and no fallback available")
                    else:
                        logger.info(f"Successfully initialized {provider} provider for model {self.config['model_name']}")
                except Exception as e:
                    logger.warning(f"Error initializing {provider} provider: {e}, trying fallback...")
                    if 'fallback_model' in self.config and self.config['fallback_model']:
                        logger.info(f"Switching to fallback provider/model: {self.config['fallback_model']}")
                        # Switch to fallback provider
                        self.config['provider'] = self.config['fallback_provider']
                        self.config['model_name'] = self.config['fallback_model']
                        self.config['base_url'] = Config.SNAKE_OLLAMA_BASE_URL
                        provider = self.config['provider']
                    else:
                        raise

            self._initialized = True
            logger.info(
                f"Snake {self.model_type} LLM interface initialized successfully for {provider} provider")
            return True

        except Exception as e:
            logger.error(
                f"Failed to initialize Snake {self.model_type} LLM: {e}")
            return False

    async def _call_ollama(self, prompt: str, unlimited: bool = None) -> str:
        """Make async call to LLM API with support for both Ollama and OpenAI-compatible APIs"""
        if not self._initialized:
            await self.initialize()

        # Determine provider type to decide API format
        provider = self.config.get('provider', 'ollama').lower()
        
        # Determine if unlimited mode should be used
        use_unlimited = unlimited if unlimited is not None else self.config.get(
            'unlimited_mode', False)

        # Get the logger instance if available from the parent system for interaction logging
        try:
            log_manager = getattr(self, 'log_manager', None)
        except:
            log_manager = None
            
        try:
            timeout = aiohttp.ClientTimeout(total=self.config['timeout'])
            async with aiohttp.ClientSession(timeout=timeout) as session:
                start_time = time.time()
                
                # Log the prompt being sent
                if log_manager:
                    await log_manager.log_interaction(
                        interaction_type=f"{self.model_type}_prompt",
                        prompt=prompt,
                        metadata={
                            "provider": provider,
                            "model": self.config['model_name'],
                            "unlimited": use_unlimited
                        }
                    )
                
                # Use different API formats based on provider
                if provider in ['electronhub', 'zuki', 'zanity', 'gemini'] or 'openai' in provider:
                    # OpenAI-compatible API format
                    payload = {
                        "model": self.config['model_name'],
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.config['temperature'],
                        "stream": False
                    }
                    
                    # Add max_tokens if specified
                    if self.config.get('max_tokens'):
                        payload["max_tokens"] = self.config['max_tokens']
                    
                    headers = {
                        "Authorization": f"Bearer {self.config.get('api_key', '')}",
                        "Content-Type": "application/json"
                    }
                    
                    api_endpoint = f"{self.config['base_url']}/v1/chat/completions"
                    async with session.post(api_endpoint, json=payload, headers=headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                        else:
                            error_text = await response.text()
                            raise Exception(
                                f"OpenAI-compatible API error {response.status}: {error_text}")
                else:
                    # Ollama API format (default)
                    # Set num_predict based on unlimited mode
                    if use_unlimited or self.config.get('max_tokens') is None:
                        num_predict = -1  # Ollama unlimited tokens
                        logger.debug(
                            f"Using unlimited token generation for {self.model_type} model")
                    else:
                        num_predict = self.config['max_tokens']
                        logger.debug(
                            f"Using limited tokens ({num_predict}) for {self.model_type} model")

                    payload = {
                        "model": self.config['model_name'],
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.config['temperature'],
                            "num_predict": num_predict
                        },
                        "keep_alive": self.config['keep_alive']
                    }
                    
                    async with session.post(f"{self.config['base_url']}/api/generate", json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            response_text = result.get('response', '')
                        else:
                            error_text = await response.text()
                            raise Exception(
                                f"Ollama API error {response.status}: {error_text}")

                # Log response metrics
                elapsed_time = time.time() - start_time
                response_length = len(response_text)
                logger.debug(
                    f"LLM response: {response_length} chars in {elapsed_time:.2f}s (provider: {provider})")

                # Log the response
                if log_manager:
                    await log_manager.log_interaction(
                        interaction_type=f"{self.model_type}_response",
                        prompt=prompt,
                        response=response_text,
                        metadata={
                            "provider": provider,
                            "model": self.config['model_name'],
                            "response_length": response_length,
                            "elapsed_time": elapsed_time
                        }
                    )

                return response_text

        except asyncio.TimeoutError:
            # Log timeout error
            if log_manager:
                log_manager.log_error_with_traceback(
                    TimeoutError(f"LLM request timed out after {self.config['timeout']} seconds ({self.config['timeout']//60} minutes)"),
                    f"LLM timeout error for {self.model_type}",
                    {"provider": provider, "model": self.config['model_name'], "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt}
                )
            raise TimeoutError(
                f"LLM request timed out after {self.config['timeout']} seconds ({self.config['timeout']//60} minutes)")
        except Exception as e:
            # Log other errors
            if log_manager:
                log_manager.log_error_with_traceback(
                    e, 
                    f"Error calling LLM API for {self.model_type}",
                    {"provider": provider, "model": self.config['model_name'], "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt}
                )
            logger.error(f"Error calling LLM API: {e}")
            raise


class SnakeCodingLLM(SnakeLLMInterface):
    """Specialized LLM interface for coding tasks"""

    def __init__(self, log_manager=None):
        # If the configured provider exists in repository providers config,
        # prefer a provider-selected model (e.g., electronhub) for coding tasks.
        conf = Config.SNAKE_CODING_MODEL or {}
        provider = conf.get('provider')
        if provider and provider in Config.PROVIDERS_CONFIG:
            try:
                prov_model = Config.get_provider_model(provider, role='coding')
                if prov_model and prov_model.get('model_name'):
                    # Ensure API key is included from the original config
                    prov_model['api_key'] = conf.get('api_key', '')
                    super().__init__(prov_model, 'coding')
                    self.log_manager = log_manager
                    return
            except Exception as e:
                logger.warning(f"Failed to get provider model for coding: {e}")

        # Fallback to configured coding model
        try:
            super().__init__(Config.SNAKE_CODING_MODEL, 'coding')
        except Exception as e:
            logger.error(
                f"Failed to initialize coding LLM with configured model: {e}")
            # Final fallback to a simple Ollama model that's more likely to be available
            fallback_config = {
                'provider': 'ollama',
                'model_name': 'gpt-oss:20b',
                'base_url': Config.SNAKE_OLLAMA_BASE_URL,
                'api_key': '',  # Ollama doesn't need API key
                'temperature': 0.1,
                'max_tokens': None,
                'unlimited_mode': True,
                'chunk_size': 4096,
                'timeout': 300,
                'keep_alive': '10m',
                'fallback_provider': 'ollama',
                'fallback_model': 'gpt-oss:20b'
            }
            super().__init__(fallback_config, 'coding')
        self.log_manager = log_manager

    async def analyze_code(self, code_content: str, analysis_type: str = "general") -> str:
        """Analyze code with specialized prompts for coding model"""
        prompt = self._build_code_analysis_prompt(code_content, analysis_type)
        # Use unlimited tokens for comprehensive code analysis
        return await self._call_ollama(prompt, unlimited=True)

    async def generate_improvement(self, analysis_result: str, code_content: str) -> str:
        """Generate code improvements based on analysis"""
        prompt = self._build_improvement_prompt(analysis_result, code_content)
        # Use unlimited tokens for detailed improvement suggestions
        return await self._call_ollama(prompt, unlimited=True)

    async def review_code_safety(self, code_content: str) -> str:
        """Review code for potential safety issues"""
        prompt = self._build_safety_review_prompt(code_content)
        # Use unlimited tokens for thorough safety analysis
        return await self._call_ollama(prompt, unlimited=True)

    def _build_code_analysis_prompt(self, code_content: str, analysis_type: str) -> str:
        """Build specialized code analysis prompt"""
        base_prompt = f"""
You are a specialized code analysis AI focused on improving code quality, performance, and maintainability.

Analysis Type: {analysis_type}

Code to analyze:
```python
{code_content}
```

Please provide a detailed analysis including:
1. Code quality assessment
2. Performance bottlenecks
3. Potential improvements
4. Architecture suggestions
5. Best practice recommendations

Focus on actionable improvements that would enhance the RAVANA AGI system.
"""
        return base_prompt.strip()

    def _build_improvement_prompt(self, analysis_result: str, code_content: str) -> str:
        """Build improvement generation prompt"""
        prompt = f"""
You are a specialized code improvement AI. Based on the analysis provided, generate specific code improvements.

Previous Analysis:
{analysis_result}

Original Code:
```python
{code_content}
```

Please provide:
1. Specific code improvements (with code examples)
2. Refactoring suggestions
3. Performance optimizations
4. Error handling improvements
5. Documentation enhancements

Output should include actual code snippets for proposed changes.
"""
        return prompt.strip()

    def _build_safety_review_prompt(self, code_content: str) -> str:
        """Build safety review prompt"""
        prompt = f"""
You are a security and safety expert reviewing code for potential issues.

Code to review:
```python
{code_content}
```

Please analyze for:
1. Security vulnerabilities
2. Potential system damage
3. Resource leaks
4. Infinite loops or blocking operations
5. File system safety
6. Network security issues

Provide a safety score (0-10) and detailed explanation.
"""
        return prompt.strip()


class SnakeReasoningLLM(SnakeLLMInterface):
    """Specialized LLM interface for reasoning tasks"""

    def __init__(self, log_manager=None):
        conf = Config.SNAKE_REASONING_MODEL or {}
        provider = conf.get('provider')
        if provider and provider in Config.PROVIDERS_CONFIG:
            try:
                prov_model = Config.get_provider_model(
                    provider, role='reasoning')
                if prov_model and prov_model.get('model_name'):
                    # Ensure API key is included from the original config
                    prov_model['api_key'] = conf.get('api_key', '')
                    super().__init__(prov_model, 'reasoning')
                    self.log_manager = log_manager
                    return
            except Exception as e:
                logger.warning(
                    f"Failed to get provider model for reasoning: {e}")

        # Fallback to configured reasoning model
        try:
            super().__init__(Config.SNAKE_REASONING_MODEL, 'reasoning')
        except Exception as e:
            logger.error(
                f"Failed to initialize reasoning LLM with configured model: {e}")
            # Final fallback to a simple Ollama model that's more likely to be available
            fallback_config = {
                'provider': 'ollama',
                'model_name': 'deepseek-r1:8b',
                'base_url': Config.SNAKE_OLLAMA_BASE_URL,
                'api_key': '',  # Ollama doesn't need API key
                'temperature': 0.3,
                'max_tokens': None,
                'unlimited_mode': True,
                'chunk_size': 2048,
                'timeout': 300,
                'keep_alive': '10m',
                'fallback_provider': 'ollama',
                'fallback_model': 'deepseek-r1:8b'
            }
            super().__init__(fallback_config, 'reasoning')
        self.log_manager = log_manager

    async def evaluate_safety(self, proposed_change: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate safety and impact of proposed changes"""
        prompt = self._build_safety_evaluation_prompt(proposed_change)
        # Use unlimited tokens for comprehensive safety evaluation
        response = await self._call_ollama(prompt, unlimited=True)
        return self._parse_safety_response(response)

    async def plan_communication(self, findings: Dict[str, Any]) -> Dict[str, Any]:
        """Plan communication strategy with RAVANA"""
        prompt = self._build_communication_prompt(findings)
        # Use unlimited tokens for detailed communication planning
        response = await self._call_ollama(prompt, unlimited=True)
        return self._parse_communication_response(response)

    async def analyze_system_impact(self, change_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential system-wide impact of changes"""
        prompt = self._build_impact_analysis_prompt(change_proposal)
        # Use unlimited tokens for thorough impact analysis
        response = await self._call_ollama(prompt, unlimited=True)
        return self._parse_impact_response(response)

    def _build_safety_evaluation_prompt(self, proposed_change: Dict[str, Any]) -> str:
        """Build safety evaluation prompt"""
        prompt = f"""
You are a safety evaluator for an AGI system. Evaluate the safety and impact of the proposed change.

Proposed Change:
{json.dumps(proposed_change, indent=2)}

Please evaluate:
1. Safety risk level (LOW/MEDIUM/HIGH)
2. Potential system impact
3. Rollback difficulty
4. Dependencies affected
5. Testing requirements
6. Approval recommendation

Provide response in JSON format with clear reasoning.
"""
        return prompt.strip()

    def _build_communication_prompt(self, findings: Dict[str, Any]) -> str:
        """Build communication planning prompt"""
        prompt = f"""
You are a communication planner for an autonomous agent system. Plan how to communicate findings to the main RAVANA system.

Findings:
{json.dumps(findings, indent=2)}

Please provide:
1. Communication priority (LOW/MEDIUM/HIGH/CRITICAL)
2. Message summary
3. Detailed technical explanation
4. Recommended action
5. Timeline for implementation
6. Risk assessment

Format response as JSON.
"""
        return prompt.strip()

    def _build_impact_analysis_prompt(self, change_proposal: Dict[str, Any]) -> str:
        """Build system impact analysis prompt"""
        prompt = f"""
You are a system analyst evaluating the impact of proposed changes on the RAVANA AGI system.

Change Proposal:
{json.dumps(change_proposal, indent=2)}

Analyze:
1. Affected components
2. Performance implications
3. Stability risks
4. Integration challenges
5. Testing strategy
6. Rollout recommendations

Provide detailed JSON response with impact assessment.
"""
        return prompt.strip()

    def _parse_safety_response(self, response: str) -> Dict[str, Any]:
        """Parse safety evaluation response"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback parsing
            return {
                "risk_level": "HIGH",
                "reasoning": response,
                "approval_recommended": False,
                "error": "Failed to parse JSON response"
            }

    def _parse_communication_response(self, response: str) -> Dict[str, Any]:
        """Parse communication planning response"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "priority": "LOW",
                "message": response,
                "error": "Failed to parse JSON response"
            }

    def _parse_impact_response(self, response: str) -> Dict[str, Any]:
        """Parse impact analysis response"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "impact_level": "UNKNOWN",
                "analysis": response,
                "error": "Failed to parse JSON response"
            }


# Factory functions for easy instantiation
async def create_snake_coding_llm(log_manager=None) -> SnakeCodingLLM:
    """Create and initialize coding LLM interface"""
    llm = SnakeCodingLLM(log_manager)
    await llm.initialize()
    return llm


async def create_snake_reasoning_llm(log_manager=None) -> SnakeReasoningLLM:
    """Create and initialize reasoning LLM interface"""
    llm = SnakeReasoningLLM(log_manager)
    await llm.initialize()
    return llm
