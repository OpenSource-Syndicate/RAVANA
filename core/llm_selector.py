"""
Intelligent LLM Selector for Ravana AGI System
Handles multiple LLM providers with priority for electronhub and gemini
"""
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Enumeration of LLM providers."""
    ELECTRONHUB = "electronhub"
    GEMINI = "gemini"
    ZUKI = "zuki"
    ZANITY = "zanity"
    A4F = "a4f"
    LOCAL = "local"


class LLMTask(Enum):
    """Enumeration of different LLM tasks."""
    REASONING = "reasoning"
    CODING = "coding"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    MULTIMODAL = "multimodal"
    QUICK_RESPONSE = "quick_response"
    DETAILED_RESPONSE = "detailed_response"


@dataclass
class ProviderSpec:
    """Specification for an LLM provider."""
    name: str
    priority: int  # Higher number = higher priority
    primary_model: str
    model_categories: Dict[str, List[str]]
    api_key_required: bool
    rate_limit: Optional[Dict[str, int]] = None
    fallback_providers: Optional[List[str]] = None


class IntelligentLLMSelector:
    """Intelligently selects the best LLM provider based on task and context."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_specs = self._initialize_provider_specs()
        self.provider_status = {}  # Track provider status and availability
        self.provider_usage_stats = {}  # Track usage for load balancing
        self._initialize_provider_status()

    def _initialize_provider_specs(self) -> Dict[str, ProviderSpec]:
        """Initialize provider specifications based on config."""
        specs = {}

        # ElectronHub - Highest priority
        electronhub_models = self.config.get(
            'electronhub', {}).get('models', [])
        specs['electronhub'] = ProviderSpec(
            name='electronhub',
            priority=10,
            primary_model='deepseek-r1:free',  # High-quality reasoning model
            model_categories={
                'reasoning': [m for m in electronhub_models if 'reasoner' in m or 'r1' in m],
                'coding': [m for m in electronhub_models if 'coder' in m or 'gpt-oss' in m],
                'analysis': electronhub_models,
                'detailed_response': [m for m in electronhub_models if 'gpt-oss' in m or 'deepseek' in m]
            },
            api_key_required=True,
            rate_limit={'requests_per_minute': 60},
            fallback_providers=['zuki', 'zanity', 'gemini']
        )

        # Gemini - High priority
        gemini_models = ['gemini-1.5-pro', 'gemini-1.5-flash',
                         'gemini-2.0-flash-exp', 'gemini-exp-1206']
        specs['gemini'] = ProviderSpec(
            name='gemini',
            priority=9,
            primary_model='gemini-1.5-pro',
            model_categories={
                'reasoning': ['gemini-1.5-pro', 'gemini-exp-1206'],
                'multimodal': ['gemini-1.5-pro', 'gemini-1.5-flash'],
                'analysis': ['gemini-1.5-pro', 'gemini-exp-1206'],
                'creative': ['gemini-exp-1206']
            },
            api_key_required=True,
            rate_limit={'requests_per_minute': 1500},  # Gemini has high limits
            fallback_providers=['electronhub', 'zuki', 'zanity']
        )

        # Zuki - Medium priority
        zuki_models = self.config.get('zuki', {}).get('models', [])
        specs['zuki'] = ProviderSpec(
            name='zuki',
            priority=7,
            primary_model='gpt-4o:online',
            model_categories={
                'reasoning': [m for m in zuki_models if 'gpt' in m or 'reasoner' in m],
                'analysis': zuki_models,
                'multimodal': [m for m in zuki_models if 'gpt' in m or 'vision' in m]
            },
            api_key_required=True,
            rate_limit={'requests_per_minute': 3000},
            fallback_providers=['zanity', 'gemini', 'electronhub']
        )

        # Zanity - Medium priority
        zanity_models = self.config.get('zanity', {}).get('models', [])
        specs['zanity'] = ProviderSpec(
            name='zanity',
            priority=6,
            primary_model='deepseek-r1',
            model_categories={
                'reasoning': [m for m in zanity_models if 'reasoner' in m or 'deepseek' in m],
                'analysis': zanity_models,
                'quick_response': [m for m in zanity_models if 'free' in m]
            },
            api_key_required=True,
            rate_limit={'requests_per_minute': 1000},
            fallback_providers=['gemini', 'electronhub', 'zuki']
        )

        # A4F - Lower priority but reliable
        specs['a4f'] = ProviderSpec(
            name='a4f',
            priority=5,
            primary_model='default',
            model_categories={
                'analysis': ['default_model'],
                'quick_response': ['default_model']
            },
            api_key_required=True,
            rate_limit={'requests_per_minute': 500},
            fallback_providers=['zuki', 'zanity', 'gemini']
        )

        # Local as fallback
        specs['local'] = ProviderSpec(
            name='local',
            priority=1,  # Lowest priority
            primary_model='ollama/llama3.2',
            model_categories={
                'quick_response': ['ollama/llama3.2', 'ollama/mistral'],
                'local_processing': ['ollama/llama3.2']
            },
            api_key_required=False,
            fallback_providers=None  # No fallback from local
        )

        return specs

    def _initialize_provider_status(self):
        """Initialize provider status tracking."""
        for provider_name in self.provider_specs.keys():
            self.provider_status[provider_name] = {
                'available': True,
                'last_error': None,
                'error_count': 0,
                'last_used': None
            }
            self.provider_usage_stats[provider_name] = {
                'requests_count': 0,
                'success_count': 0,
                'error_count': 0,
                'last_request_time': None
            }

    def analyze_task_requirements(self,
                                  task: LLMTask,
                                  content: str = "",
                                  response_quality_needed: float = 0.7,
                                  response_speed_needed: float = 0.5) -> Dict[str, Any]:
        """Analyze task requirements to determine optimal provider."""
        analysis = {
            'required_capabilities': [],
            'priority_factors': {
                'quality_weight': 0.0,
                'speed_weight': 0.0,
                'cost_weight': 0.0,
                'reliability_weight': 0.0
            }
        }

        # Determine required capabilities based on task
        if task == LLMTask.MULTIMODAL:
            analysis['required_capabilities'].append('vision')
        elif task == LLMTask.CODING:
            analysis['required_capabilities'].append('code')
        elif task == LLMTask.REASONING:
            analysis['required_capabilities'].append('reasoning')
        elif task == LLMTask.QUICK_RESPONSE:
            analysis['required_capabilities'].append('speed')

        # Determine priority factors
        if response_quality_needed > 0.8:
            analysis['priority_factors']['quality_weight'] = 1.0
        else:
            analysis['priority_factors']['quality_weight'] = response_quality_needed

        if response_speed_needed > 0.7:
            analysis['priority_factors']['speed_weight'] = 1.0
        else:
            analysis['priority_factors']['speed_weight'] = response_speed_needed

        # Cost is a factor when speed isn't critical
        if response_speed_needed < 0.6:
            analysis['priority_factors']['cost_weight'] = 0.8
        else:
            analysis['priority_factors']['cost_weight'] = 0.3

        # Always important
        analysis['priority_factors']['reliability_weight'] = 0.9

        return analysis

    def get_available_providers(self, task: LLMTask = None) -> List[str]:
        """Get list of currently available providers."""
        available = []
        for provider_name, status in self.provider_status.items():
            if status['available']:
                specs = self.provider_specs.get(provider_name)
                if specs:
                    # For specific tasks, ensure provider has relevant models
                    if task:
                        category = task.value
                        models = specs.model_categories.get(category, [])
                        if models or category == 'quick_response':  # Quick response can use any
                            available.append(provider_name)
                    else:
                        available.append(provider_name)
        return available

    def select_best_provider(self,
                             task: LLMTask,
                             content: str = "",
                             response_quality_needed: float = 0.7,
                             response_speed_needed: float = 0.5,
                             specific_model: str = None) -> Optional[Dict[str, str]]:
        """
        Select the best LLM provider based on task requirements.

        Args:
            task: The type of task to be performed
            content: Content that may influence model selection
            response_quality_needed: Required quality level (0-1)
            response_speed_needed: Required speed level (0-1)
            specific_model: Specific model to use (overrides selection)

        Returns:
            Dictionary with provider information or None if no provider available
        """
        if specific_model:
            # If specific model requested, try to find provider that has it
            for provider_name, specs in self.provider_specs.items():
                if not self.provider_status[provider_name]['available']:
                    continue
                # Check if model exists in this provider
                for category_models in specs.model_categories.values():
                    if specific_model in category_models:
                        return {
                            'provider': provider_name,
                            'model': specific_model,
                            'config': self.config.get(provider_name, {})
                        }

        # Analyze task requirements
        requirements = self.analyze_task_requirements(
            task, content, response_quality_needed, response_speed_needed
        )

        # Get available providers for this task
        available_providers = self.get_available_providers(task)
        if not available_providers:
            logger.warning(f"No providers available for task: {task.value}")
            return None

        # Score providers based on requirements and priorities
        best_provider = None
        best_score = -1

        for provider_name in available_providers:
            specs = self.provider_specs[provider_name]
            score = 0

            # Priority-based scoring
            score += specs.priority * 100  # Base priority score

            # Quality requirements
            quality_factor = requirements['priority_factors']['quality_weight']
            if quality_factor > 0.6 and provider_name in ['electronhub', 'zuki', 'gemini']:
                score += quality_factor * 50  # High quality providers

            # Speed requirements
            speed_factor = requirements['priority_factors']['speed_weight']
            if speed_factor > 0.7 and provider_name in ['zanity', 'a4f', 'gemini']:
                score += speed_factor * 30  # Fast providers

            # Reliability factor
            reliability_factor = requirements['priority_factors']['reliability_weight']
            error_count = self.provider_status[provider_name]['error_count']
            # Lower error count = higher reliability
            reliability_score = max(0, 100 - error_count * 10)
            score += reliability_factor * reliability_score

            # Cost factor (lower priority providers may be cheaper)
            cost_factor = requirements['priority_factors']['cost_weight']
            if cost_factor > 0.5 and provider_name in ['zanity', 'a4f']:
                score += cost_factor * 20

            # Task-specific model availability
            task_category = task.value
            provider_models = specs.model_categories.get(task_category, [])
            if provider_models:
                score += 30  # Bonus for having task-specific models
            elif task_category == 'quick_response':
                score += 20  # Most providers can handle quick responses

            # Apply usage load balancing - penalize heavily used providers
            usage_stats = self.provider_usage_stats[provider_name]
            recent_requests = usage_stats.get('requests_count', 0)
            if recent_requests > 100:  # Arbitrary threshold
                # More requests = lower score
                load_factor = min(1.0, 100.0 / recent_requests)
                score *= load_factor

            if score > best_score:
                best_score = score
                best_provider = provider_name

        if best_provider:
            # Select the best model for this task from the selected provider
            specs = self.provider_specs[best_provider]
            task_category = task.value
            available_models = specs.model_categories.get(task_category, [])

            # Choose model based on requirements
            if available_models:
                best_model = available_models[0]  # Take first suitable model
                # For quality tasks, try to pick higher quality models
                if task in [LLMTask.REASONING, LLMTask.ANALYSIS] and response_quality_needed > 0.8:
                    high_quality_models = [
                        m for m in available_models if 'r1' in m or 'pro' in m or 'gpt-4' in m]
                    if high_quality_models:
                        best_model = high_quality_models[0]
            else:
                # If no specific models for this task, use primary model or any available
                best_model = specs.primary_model
                if not best_model and self.config.get(best_provider, {}).get('models'):
                    best_model = self.config[best_provider]['models'][0]

            # Update usage statistics
            self.provider_usage_stats[best_provider]['requests_count'] += 1
            self.provider_usage_stats[best_provider]['last_request_time'] = time.time(
            )

            return {
                'provider': best_provider,
                'model': best_model,
                'config': self.config.get(best_provider, {}),
                'score': best_score
            }

        return None

    def mark_provider_success(self, provider_name: str):
        """Mark a provider as successful for future selection."""
        if provider_name in self.provider_status:
            self.provider_status[provider_name]['error_count'] = 0
            self.provider_status[provider_name]['available'] = True
            self.provider_status[provider_name]['last_error'] = None

            # Update success statistics
            self.provider_usage_stats[provider_name]['success_count'] += 1

    def mark_provider_error(self, provider_name: str, error: str = None):
        """Mark a provider as having an error."""
        if provider_name in self.provider_status:
            self.provider_status[provider_name]['error_count'] += 1
            self.provider_usage_stats[provider_name]['error_count'] += 1

            if self.provider_status[provider_name]['error_count'] > 5:  # Too many errors
                self.provider_status[provider_name]['available'] = False
                logger.warning(
                    f"Provider {provider_name} marked as unavailable due to errors")

            self.provider_status[provider_name]['last_error'] = error

    def get_fallback_provider(self, failed_provider: str) -> Optional[str]:
        """Get the next best provider if the current one fails."""
        specs = self.provider_specs.get(failed_provider)
        if not specs or not specs.fallback_providers:
            # If no fallbacks specified, try providers in priority order
            available = self.get_available_providers()
            for provider in sorted(available,
                                   key=lambda x: self.provider_specs[x].priority,
                                   reverse=True):
                if provider != failed_provider:
                    return provider
            return None

        # Use specified fallback providers
        for fallback in specs.fallback_providers:
            if (fallback in self.provider_status and
                    self.provider_status[fallback]['available']):
                return fallback

        return None


# Global instance for shared use
llm_selector = None


def initialize_llm_selector(config: Dict[str, Any]) -> IntelligentLLMSelector:
    """Initialize the LLM selector with the provided config."""
    global llm_selector
    llm_selector = IntelligentLLMSelector(config)
    return llm_selector


def get_llm_selector() -> Optional[IntelligentLLMSelector]:
    """Get the global LLM selector instance."""
    return llm_selector
