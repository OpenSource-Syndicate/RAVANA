"""
Snake Agent LLM Interface

This module provides specialized LLM interfaces for the Snake Agent system,
supporting dual models for coding and reasoning tasks using Ollama.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional, List
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
            response = requests.get(f"{Config.SNAKE_OLLAMA_BASE_URL}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return False
    
    @staticmethod
    def validate_models_available() -> tuple[bool, bool, List[str]]:
        """Check if required models are pulled"""
        try:
            response = requests.get(f"{Config.SNAKE_OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                available_models = [m['name'] for m in response.json().get('models', [])]
                coding_available = Config.SNAKE_CODING_MODEL['model_name'] in available_models
                reasoning_available = Config.SNAKE_REASONING_MODEL['model_name'] in available_models
                return coding_available, reasoning_available, available_models
        except Exception as e:
            logger.error(f"Error checking models: {e}")
        return False, False, []
    
    @staticmethod
    def get_startup_report() -> Dict[str, Any]:
        """Generate configuration status report"""
        ollama_connected = SnakeConfigValidator.validate_ollama_connection()
        coding_available, reasoning_available, available_models = SnakeConfigValidator.validate_models_available()
        
        return {
            "ollama_connected": ollama_connected,
            "coding_model_available": coding_available,
            "reasoning_model_available": reasoning_available,
            "available_models": available_models,
            "config_valid": ollama_connected and coding_available and reasoning_available
        }


class OllamaClient:
    """Dedicated Ollama client for Snake Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config['base_url']
        self.model_name = config['model_name']
    
    async def pull_model_if_needed(self) -> bool:
        """Ensure model is available locally, pull if necessary"""
        try:
            # Check if model exists
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        models_data = await response.json()
                        available_models = [m['name'] for m in models_data.get('models', [])]
                        
                        if self.model_name not in available_models:
                            logger.info(f"Model {self.model_name} not found. Attempting to pull...")
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
                                                    logger.info(f"Successfully pulled model {self.model_name}")
                                                    return True
                                            except json.JSONDecodeError:
                                                continue
                                else:
                                    logger.error(f"Failed to pull model {self.model_name}")
                                    return False
                        else:
                            logger.info(f"Model {self.model_name} is already available")
                            return True
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
            # Verify Ollama connection
            if not SnakeConfigValidator.validate_ollama_connection():
                raise ConnectionError(f"Cannot connect to Ollama at {self.config['base_url']}")
            
            # Ensure model is available
            model_available = await self.client.pull_model_if_needed()
            if not model_available:
                raise RuntimeError(f"Model {self.config['model_name']} not available")
            
            self._initialized = True
            logger.info(f"Snake {self.model_type} LLM interface initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Snake {self.model_type} LLM: {e}")
            return False
    
    async def _call_ollama(self, prompt: str, unlimited: bool = None) -> str:
        """Make async call to Ollama API with optional unlimited token generation"""
        if not self._initialized:
            await self.initialize()
        
        # Determine if unlimited mode should be used
        use_unlimited = unlimited if unlimited is not None else self.config.get('unlimited_mode', False)
        
        # Set num_predict based on unlimited mode
        if use_unlimited or self.config.get('max_tokens') is None:
            num_predict = -1  # Ollama unlimited tokens
            logger.debug(f"Using unlimited token generation for {self.model_type} model")
        else:
            num_predict = self.config['max_tokens']
            logger.debug(f"Using limited tokens ({num_predict}) for {self.model_type} model")
        
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
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.config['timeout'])
            async with aiohttp.ClientSession(timeout=timeout) as session:
                start_time = time.time()
                async with session.post(f"{self.config['base_url']}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get('response', '')
                        
                        # Log response metrics
                        elapsed_time = time.time() - start_time
                        response_length = len(response_text)
                        logger.debug(f"LLM response: {response_length} chars in {elapsed_time:.2f}s (unlimited: {use_unlimited})")
                        
                        return response_text
                    else:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error {response.status}: {error_text}")
                        
        except asyncio.TimeoutError:
            raise TimeoutError(f"Ollama request timed out after {self.config['timeout']} seconds")
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise


class SnakeCodingLLM(SnakeLLMInterface):
    """Specialized LLM interface for coding tasks"""
    
    def __init__(self):
        super().__init__(Config.SNAKE_CODING_MODEL, 'coding')
    
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
    
    def __init__(self):
        super().__init__(Config.SNAKE_REASONING_MODEL, 'reasoning')
    
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
async def create_snake_coding_llm() -> SnakeCodingLLM:
    """Create and initialize coding LLM interface"""
    llm = SnakeCodingLLM()
    await llm.initialize()
    return llm


async def create_snake_reasoning_llm() -> SnakeReasoningLLM:
    """Create and initialize reasoning LLM interface"""
    llm = SnakeReasoningLLM()
    await llm.initialize()
    return llm