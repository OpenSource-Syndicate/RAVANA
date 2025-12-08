"""
Simplified Configuration System for RAVANA AGI

This module provides a simplified configuration system that reduces complexity
while maintaining essential functionality for the RAVANA AGI system.
"""

import os
import json
from typing import Dict, Any, Optional


class SimpleConfig:
    """
    Simplified configuration class for RAVANA AGI system.
    """
    
    def __init__(self):
        # Basic system settings
        self.system_name = os.environ.get("SYSTEM_NAME", "RAVANA")
        self.database_url = os.environ.get("DATABASE_URL", "sqlite:///ravana_agi.db")
        self.log_level = os.environ.get("LOG_LEVEL", "INFO")
        
        # Core operational settings
        self.curiosity_chance = float(os.environ.get("CURIOSITY_CHANCE", 0.4))
        self.reflection_chance = float(os.environ.get("REFLECTION_CHANCE", 0.15))
        self.loop_sleep_duration = int(os.environ.get("LOOP_SLEEP_DURATION", 7))
        self.max_iterations = int(os.environ.get("MAX_ITERATIONS", 15))
        
        # Model configuration - simplified to one primary model
        self.primary_model = {
            'provider': os.environ.get("PRIMARY_MODEL_PROVIDER", "ollama"),
            'model_name': os.environ.get("PRIMARY_MODEL_NAME", "llama3.1:8b"),
            'base_url': os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            'temperature': float(os.environ.get("MODEL_TEMPERATURE", "0.7")),
            'max_tokens': int(os.environ.get("MODEL_MAX_TOKENS", "2048")),
            'timeout': int(os.environ.get("MODEL_TIMEOUT", "300"))
        }
        
        # API keys - simplified to essential providers
        self.api_keys = {
            'electronhub': os.environ.get("ELECTRONHUB_API_KEY", "ek-sVvxMYfdFQ0Kl6Aj2tmV7b8n5v0Y0sDHVsOUZWyx2vbs0AbuAc"),
            'zuki': os.environ.get("ZUKI_API_KEY", "985160dfa1fd499fd12af708d16552e37a8c6f77cbfb50ae400e3ff33fbd791bc7b3b82625379a1f5ca7568f1ee04eb81a0f8c06f0ba6c276d3dddfe13e9c18d"),
            'gemini': os.environ.get("GEMINI_API_KEY", "AIzaSyBW-aVU-x7JCjBJVVKjPGUacups0-GBHvQ")
        }
        
        # Memory and embedding settings
        self.embedding_model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedding_use_cuda = self._str_to_bool(os.environ.get("EMBEDDING_USE_CUDA", "True"))
        
        # Snake Agent settings - simplified
        self.snake_agent_enabled = self._str_to_bool(os.environ.get("SNAKE_AGENT_ENABLED", "True"))
        self.snake_agent_interval = int(os.environ.get("SNAKE_AGENT_INTERVAL", 180))
        
        # Shutdown settings
        self.shutdown_timeout = int(os.environ.get("SHUTDOWN_TIMEOUT", 30))
        self.graceful_shutdown_enabled = self._str_to_bool(os.environ.get("GRACEFUL_SHUTDOWN_ENABLED", "True"))
        
        # Blog settings
        self.blog_enabled = self._str_to_bool(os.environ.get("BLOG_ENABLED", "True"))
        self.blog_api_url = os.environ.get("BLOG_API_URL", "https://ravana-blog.netlify.app/api/publish")
        
        # Conversational AI settings
        self.conversational_ai_enabled = self._str_to_bool(os.environ.get("CONVERSATIONAL_AI_ENABLED", "True"))
        
        # Background task intervals
        self.data_collection_interval = int(os.environ.get("DATA_COLLECTION_INTERVAL", 1800))
        self.event_detection_interval = int(os.environ.get("EVENT_DETECTION_INTERVAL", 300))
        self.knowledge_compression_interval = int(os.environ.get("KNOWLEDGE_COMPRESSION_INTERVAL", 7200))
        
        # Persona settings
        self.persona_name = os.environ.get("PERSONA_NAME", "Ravana")
        self.persona_creativity = float(os.environ.get("PERSONA_CREATIVITY", 0.7))

    def _str_to_bool(self, value: str) -> bool:
        """Convert string to boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)

    def get_config_dict(self) -> Dict[str, Any]:
        """Return configuration as a dictionary."""
        return {
            'system_name': self.system_name,
            'database_url': self.database_url,
            'log_level': self.log_level,
            'curiosity_chance': self.curiosity_chance,
            'reflection_chance': self.reflection_chance,
            'loop_sleep_duration': self.loop_sleep_duration,
            'max_iterations': self.max_iterations,
            'primary_model': self.primary_model,
            'api_keys': self.api_keys,
            'embedding_model': self.embedding_model,
            'embedding_use_cuda': self.embedding_use_cuda,
            'snake_agent_enabled': self.snake_agent_enabled,
            'snake_agent_interval': self.snake_agent_interval,
            'shutdown_timeout': self.shutdown_timeout,
            'graceful_shutdown_enabled': self.graceful_shutdown_enabled,
            'blog_enabled': self.blog_enabled,
            'blog_api_url': self.blog_api_url,
            'conversational_ai_enabled': self.conversational_ai_enabled,
            'data_collection_interval': self.data_collection_interval,
            'event_detection_interval': self.event_detection_interval,
            'knowledge_compression_interval': self.knowledge_compression_interval,
            'persona_name': self.persona_name,
            'persona_creativity': self.persona_creativity
        }

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        config_dict = self.get_config_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)

    def load_from_file(self, filepath: str) -> None:
        """Load configuration from a JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Update instance attributes from the loaded config
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def validate(self) -> tuple[bool, list]:
        """Validate the configuration and return (is_valid, list_of_errors)."""
        errors = []
        
        # Validate ranges for numeric values
        try:
            if not 0 <= self.curiosity_chance <= 1:
                errors.append("CURIOSITY_CHANCE must be between 0 and 1")
            if not 0 <= self.reflection_chance <= 1:
                errors.append("REFLECTION_CHANCE must be between 0 and 1")
            if self.loop_sleep_duration < 0:
                errors.append("LOOP_SLEEP_DURATION must be non-negative")
            if self.max_iterations <= 0:
                errors.append("MAX_ITERATIONS must be positive")
        except (TypeError, ValueError):
            errors.append("Some configuration values have incorrect types")
        
        # Validate primary model structure
        if not isinstance(self.primary_model, dict):
            errors.append("PRIMARY_MODEL must be a dictionary")
        else:
            required_model_keys = ['model_name', 'base_url', 'temperature']
            for key in required_model_keys:
                if key not in self.primary_model:
                    errors.append(f"PRIMARY_MODEL missing required key: {key}")
        
        return len(errors) == 0, errors


# Singleton instance for global access
_simple_config_instance = None


def get_config() -> SimpleConfig:
    """Get the singleton configuration instance."""
    global _simple_config_instance
    if _simple_config_instance is None:
        _simple_config_instance = SimpleConfig()
    return _simple_config_instance