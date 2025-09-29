"""
Standardized, simplified configuration for RAVANA AGI system
"""

import os
import json
from typing import Optional


class StandardConfig:
    """
    Simplified, standardized configuration for RAVANA AGI system
    """

    # Database Configuration
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///ravana_agi.db")

    # Logging Configuration
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.environ.get("LOG_FORMAT", "TEXT")

    # Core System Settings
    LOOP_SLEEP_DURATION = int(os.environ.get("LOOP_SLEEP_DURATION", 7))
    ERROR_SLEEP_DURATION = int(os.environ.get("ERROR_SLEEP_DURATION", 30))
    MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", 15))
    RESEARCH_TASK_TIMEOUT = int(os.environ.get("RESEARCH_TASK_TIMEOUT", 1200))

    # AI Model Configuration
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_USE_CUDA = os.environ.get(
        "EMBEDDING_USE_CUDA", "True").lower() in ("true", "1", "yes")

    # Emotional Intelligence Settings
    POSITIVE_MOODS = [
        'Confident', 'Curious', 'Reflective', 'Excited',
        'Content', 'Optimistic', 'Creative', 'Focussed'
    ]
    NEGATIVE_MOODS = [
        'Frustrated', 'Stuck', 'Low Energy', 'Bored',
        'Overwhelmed', 'Confused', 'Anxious'
    ]
    EMOTIONAL_PERSONA = "Adaptive"

    # Experimentation Settings
    CURIOSITY_CHANCE = float(os.environ.get("CURIOSITY_CHANCE", 0.4))
    REFLECTION_CHANCE = float(os.environ.get("REFLECTION_CHANCE", 0.15))

    # Memory Settings
    MEMORY_CONTEXT_DAYS = int(os.environ.get("MEMORY_CONTEXT_DAYS", 30))

    # Personality Settings
    PERSONA_NAME = os.environ.get("PERSONA_NAME", "Ravana")
    PERSONA_ORIGIN = os.environ.get("PERSONA_ORIGIN", "Ancient Sri Lanka")
    PERSONA_CREATIVITY = float(os.environ.get("PERSONA_CREATIVITY", 0.7))

    # Enhanced Features
    SNAKE_AGENT_ENABLED = os.environ.get(
        "SNAKE_AGENT_ENABLED", "True").lower() in ("true", "1", "yes")
    CONVERSATIONAL_AI_ENABLED = os.environ.get(
        "CONVERSATIONAL_AI_ENABLED", "True").lower() in ("true", "1", "yes")
    BLOG_ENABLED = os.environ.get(
        "BLOG_ENABLED", "True").lower() in ("true", "1", "yes")

    # RSS Feed URLs
    FEED_URLS = [
        "http://rss.cnn.com/rss/cnn_latest.rss",
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://www.reddit.com/r/worldnews/.rss",
        "https://techcrunch.com/feed/",
        "https://www.npr.org/rss/rss.php?id=1001",
    ]

    # Invention settings
    # seconds between invention attempts
    INVENTION_INTERVAL = int(os.environ.get("INVENTION_INTERVAL", 7200))

    # Shutdown Configuration
    SHUTDOWN_TIMEOUT = int(os.environ.get("SHUTDOWN_TIMEOUT", 30))  # seconds
    GRACEFUL_SHUTDOWN_ENABLED = bool(os.environ.get(
        "GRACEFUL_SHUTDOWN_ENABLED", "True").lower() in ["true", "1", "yes"])
    STATE_PERSISTENCE_ENABLED = bool(os.environ.get(
        "STATE_PERSISTENCE_ENABLED", "True").lower() in ["true", "1", "yes"])
    SHUTDOWN_STATE_FILE = os.environ.get(
        "SHUTDOWN_STATE_FILE", "shutdown_state.json")
    FORCE_SHUTDOWN_AFTER = int(os.environ.get(
        "FORCE_SHUTDOWN_AFTER", 60))  # seconds

    # Task Intervals
    DATA_COLLECTION_INTERVAL = int(os.environ.get(
        "DATA_COLLECTION_INTERVAL", 1800))  # Reduced to 30 minutes
    EVENT_DETECTION_INTERVAL = int(os.environ.get(
        "EVENT_DETECTION_INTERVAL", 300))  # Reduced to 5 minutes
    KNOWLEDGE_COMPRESSION_INTERVAL = int(os.environ.get(
        # Increased to 2 hours for better processing
        "KNOWLEDGE_COMPRESSION_INTERVAL", 7200))

    # Snake Agent Configuration
    # 3 minutes default for better responsiveness
    SNAKE_AGENT_INTERVAL = int(os.environ.get("SNAKE_AGENT_INTERVAL", 180))
    SNAKE_OLLAMA_BASE_URL = os.environ.get(
        "SNAKE_OLLAMA_BASE_URL", "http://localhost:11434")

    # AI Provider Configuration for Snake Agent
    SNAKE_PROVIDER_BASE_URL = os.environ.get(
        "SNAKE_PROVIDER_BASE_URL", "https://api.electronhub.ai")  # Prioritize electronhub
    SNAKE_PROVIDER_TIMEOUT = int(os.environ.get(
        "SNAKE_PROVIDER_TIMEOUT", 120))  # 2 minutes for API calls
    SNAKE_PROVIDER_KEEP_ALIVE = os.environ.get(
        "SNAKE_PROVIDER_KEEP_ALIVE", "10m")

    # Alternative Model Options
    SNAKE_AVAILABLE_CODING_MODELS = [
        "gpt-oss-20b:free",      # electronhub free model
        "gpt-oss-120b:free",     # electronhub free model
        "qwen3-coder-480b-a35b-instruct:free",  # electronhub free model
        "deepseek-r1:free",      # electronhub free model
        "deepseek-r1-0528:free",  # electronhub free model
        "deepseek-coder:6.7b",
        "deepseek-coder:1.3b",
        "codellama:7b",
        "codellama:13b",
        "starcoder2:3b",
        "starcoder2:7b"
    ]

    SNAKE_AVAILABLE_REASONING_MODELS = [
        "deepseek-r1:free",      # electronhub free model
        "deepseek-r1-0528:free",  # electronhub free model
        "llama-3.3-70b-instruct:free",  # electronhub free model
        "qwen3-next-80b-a3b-instruct:free",  # electronhub free model
        "deepseek-v3-0324",      # high-quality reasoning
        "gpt-4o:online",         # zuki premium
        "llama3.1:8b",
        "llama3.1:70b",
        "qwen2.5:7b",
        "qwen2.5:14b",
        "mistral:7b",
        "gemma2:9b",
        "claude-3.5-sonnet:free"  # zanity free model
    ]

    # Snake Agent Safety Configuration
    SNAKE_SANDBOX_TIMEOUT = int(os.environ.get(
        "SNAKE_SANDBOX_TIMEOUT", 60))  # seconds
    SNAKE_MAX_FILE_SIZE = int(os.environ.get(
        "SNAKE_MAX_FILE_SIZE", 1048576))  # 1MB
    SNAKE_BLACKLIST_PATHS = os.environ.get("SNAKE_BLACKLIST_PATHS", "").split(
        ",") if os.environ.get("SNAKE_BLACKLIST_PATHS") else []
    SNAKE_APPROVAL_REQUIRED = bool(os.environ.get(
        "SNAKE_APPROVAL_REQUIRED", "True").lower() in ["true", "1", "yes"])

    # Communication Configuration
    SNAKE_COMM_CHANNEL = os.environ.get("SNAKE_COMM_CHANNEL", "memory_service")
    SNAKE_COMM_PRIORITY_THRESHOLD = float(
        os.environ.get("SNAKE_COMM_PRIORITY_THRESHOLD", "0.8"))

    # Enhanced Snake Agent Configuration
    SNAKE_ENHANCED_MODE = bool(os.environ.get(
        "SNAKE_ENHANCED_MODE", "True").lower() in ["true", "1", "yes"])
    # Increased for better concurrency
    SNAKE_MAX_THREADS = int(os.environ.get("SNAKE_MAX_THREADS", "12"))
    # Increased for better concurrency
    SNAKE_MAX_PROCESSES = int(os.environ.get("SNAKE_MAX_PROCESSES", "6"))
    # Increased for better analysis
    SNAKE_ANALYSIS_THREADS = int(os.environ.get("SNAKE_ANALYSIS_THREADS", "4"))
    SNAKE_MONITOR_INTERVAL = float(os.environ.get(
        "SNAKE_MONITOR_INTERVAL", "1.0"))  # Reduced for faster monitoring
    SNAKE_PERF_MONITORING = bool(os.environ.get(
        "SNAKE_PERF_MONITORING", "True").lower() in ["true", "1", "yes"])
    SNAKE_AUTO_RECOVERY = bool(os.environ.get(
        "SNAKE_AUTO_RECOVERY", "True").lower() in ["true", "1", "yes"])
    SNAKE_LOG_RETENTION_DAYS = int(os.environ.get(
        "SNAKE_LOG_RETENTION_DAYS", "60"))  # Increased retention

    # Threading and Multiprocessing Limits
    # Increased to 10 minutes for complex tasks
    SNAKE_TASK_TIMEOUT = float(os.environ.get("SNAKE_TASK_TIMEOUT", "600.0"))
    SNAKE_HEARTBEAT_INTERVAL = float(os.environ.get(
        "SNAKE_HEARTBEAT_INTERVAL", "5.0"))  # Reduced for better responsiveness
    # Increased for better throughput
    SNAKE_MAX_QUEUE_SIZE = int(os.environ.get("SNAKE_MAX_QUEUE_SIZE", "2000"))
    # Reduced to 30 minutes for better resource management
    SNAKE_CLEANUP_INTERVAL = float(
        os.environ.get("SNAKE_CLEANUP_INTERVAL", "1800.0"))

    # Peek prioritizer
    SNAKE_USE_PEEK_PRIORITIZER = bool(os.environ.get(
        # Enabled by default
        "SNAKE_USE_PEEK_PRIORITIZER", "True").lower() in ["true", "1", "yes"])

    # Blog Integration Configuration
    BLOG_API_URL = os.environ.get(
        "RAVANA_BLOG_API_URL", "https://ravana-blog.netlify.app/api/publish")
    BLOG_AUTH_TOKEN = os.environ.get(
        "RAVANA_BLOG_AUTH_TOKEN", "ravana_secret_token_2024")

    # Content Generation Settings
    BLOG_DEFAULT_STYLE = os.environ.get("BLOG_DEFAULT_STYLE", "technical")
    BLOG_MAX_CONTENT_LENGTH = int(os.environ.get(
        "BLOG_MAX_CONTENT_LENGTH", "1000000"))  # Effectively unlimited
    # Reduced for more frequent posts
    BLOG_MIN_CONTENT_LENGTH = int(
        os.environ.get("BLOG_MIN_CONTENT_LENGTH", "300"))
    BLOG_AUTO_TAGGING_ENABLED = bool(os.environ.get(
        "BLOG_AUTO_TAGGING_ENABLED", "True").lower() in ["true", "1", "yes"])
    # Increased for better categorization
    BLOG_MAX_TAGS = int(os.environ.get("BLOG_MAX_TAGS", "15"))

    # Publishing Behavior
    BLOG_AUTO_PUBLISH_ENABLED = bool(os.environ.get("BLOG_AUTO_PUBLISH_ENABLED", "True").lower() in [
                                     "true", "1", "yes"])  # Enabled by default
    BLOG_REQUIRE_APPROVAL = bool(os.environ.get("BLOG_REQUIRE_APPROVAL", "False").lower() in [
                                 "true", "1", "yes"])  # Disabled for faster publishing
    BLOG_PUBLISH_FREQUENCY_HOURS = int(os.environ.get(
        "BLOG_PUBLISH_FREQUENCY_HOURS", "12"))  # Increased frequency

    # API Communication Settings
    BLOG_TIMEOUT_SECONDS = int(os.environ.get(
        "BLOG_TIMEOUT_SECONDS", "60"))  # Increased for reliability
    # Increased for better reliability
    BLOG_RETRY_ATTEMPTS = int(os.environ.get("BLOG_RETRY_ATTEMPTS", "5"))
    BLOG_RETRY_BACKOFF_FACTOR = float(os.environ.get(
        "BLOG_RETRY_BACKOFF_FACTOR", "1.5"))  # Reduced for faster retries
    # Increased for better handling
    BLOG_MAX_RETRY_DELAY = int(os.environ.get("BLOG_MAX_RETRY_DELAY", "120"))

    # Content Quality Settings
    BLOG_CONTENT_STYLES = ["technical", "casual", "academic", "creative",
                           "philosophical", "analytical", "insightful", "explanatory"]
    BLOG_MEMORY_CONTEXT_DAYS = int(os.environ.get(
        "BLOG_MEMORY_CONTEXT_DAYS", "14"))  # Increased for better context
    BLOG_INCLUDE_MOOD_CONTEXT = bool(os.environ.get(
        "BLOG_INCLUDE_MOOD_CONTEXT", "True").lower() in ["true", "1", "yes"])

    # Conversational AI Configuration
    CONVERSATIONAL_AI_START_DELAY = int(os.environ.get(
        "CONVERSATIONAL_AI_START_DELAY", 2))  # Reduced for faster startup

    # Additional configuration for compatibility
    # Changed to quality for better embeddings
    EMBEDDING_MODEL_TYPE = os.environ.get("EMBEDDING_MODEL_TYPE", "quality")
    # cuda, cpu, mps, or None for auto
    EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", None)

    # Enhanced Shutdown Configuration
    SHUTDOWN_HEALTH_CHECK_ENABLED = bool(os.environ.get(
        "SHUTDOWN_HEALTH_CHECK_ENABLED", "True").lower() in ["true", "1", "yes"])
    SHUTDOWN_BACKUP_ENABLED = bool(os.environ.get(
        "SHUTDOWN_BACKUP_ENABLED", "True").lower() in ["true", "1", "yes"])
    # Increased for better reliability
    SHUTDOWN_BACKUP_COUNT = int(os.environ.get("SHUTDOWN_BACKUP_COUNT", 10))
    SHUTDOWN_STATE_VALIDATION_ENABLED = bool(os.environ.get(
        "SHUTDOWN_STATE_VALIDATION_ENABLED", "True").lower() in ["true", "1", "yes"])
    SHUTDOWN_VALIDATION_ENABLED = bool(os.environ.get(
        "SHUTDOWN_VALIDATION_ENABLED", "True").lower() in ["true", "1", "yes"])
    SHUTDOWN_COMPRESSION_ENABLED = bool(os.environ.get(
        "SHUTDOWN_COMPRESSION_ENABLED", "True").lower() in ["true", "1", "yes"])
    COMPONENT_PREPARE_TIMEOUT = float(os.environ.get(
        "COMPONENT_PREPARE_TIMEOUT", 15.0))  # Increased for complex components
    COMPONENT_SHUTDOWN_TIMEOUT = float(os.environ.get(
        "COMPONENT_SHUTDOWN_TIMEOUT", 25.0))  # Increased for thorough shutdown

    # Memory Service Shutdown Configuration
    MEMORY_SERVICE_SHUTDOWN_TIMEOUT = int(os.environ.get(
        "MEMORY_SERVICE_SHUTDOWN_TIMEOUT", 25))  # Increased for complete memory save
    POSTGRES_CONNECTION_TIMEOUT = int(os.environ.get(
        "POSTGRES_CONNECTION_TIMEOUT", 15))  # Increased for stable connections
    CHROMADB_PERSIST_ON_SHUTDOWN = bool(os.environ.get(
        "CHROMADB_PERSIST_ON_SHUTDOWN", "True").lower() in ["true", "1", "yes"])
    TEMP_FILE_CLEANUP_ENABLED = bool(os.environ.get(
        "TEMP_FILE_CLEANUP_ENABLED", "True").lower() in ["true", "1", "yes"])

    # Resource Cleanup Configuration
    ACTION_CACHE_PERSIST = bool(os.environ.get(
        "ACTION_CACHE_PERSIST", "True").lower() in ["true", "1", "yes"])
    RESOURCE_CLEANUP_TIMEOUT = int(os.environ.get(
        "RESOURCE_CLEANUP_TIMEOUT", 20))  # Increased for thorough cleanup
    DATABASE_CLEANUP_TIMEOUT = int(os.environ.get(
        "DATABASE_CLEANUP_TIMEOUT", 25))  # Increased for complete cleanup

    # Snake Agent Graceful Shutdown Integration
    SNAKE_SHUTDOWN_TIMEOUT = int(os.environ.get(
        "SNAKE_SHUTDOWN_TIMEOUT", 30))  # seconds
    SNAKE_STATE_PERSISTENCE = bool(os.environ.get(
        "SNAKE_STATE_PERSISTENCE", "True").lower() in ["true", "1", "yes"])

    # Provider Configuration
    PROVIDERS_CONFIG = {}
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                PROVIDERS_CONFIG = json.load(f)
    except Exception:
        PROVIDERS_CONFIG = {}

    @classmethod
    def get_provider_config(cls, provider_name: str) -> Optional[dict]:
        """Get specific provider configuration"""
        return cls.PROVIDERS_CONFIG.get(provider_name)

    @classmethod
    def get_default_provider(cls) -> str:
        """Get the default provider"""
        return "electronhub"  # As this was working in our tests

    @classmethod
    def get_default_models(cls) -> dict:
        """Get default model configuration"""
        return {
            "coding": "gpt-oss-20b:free",
            "reasoning": "deepseek-r1:free"
        }
