import os
import json


class Config:
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///ravana_agi.db")
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.environ.get("LOG_FORMAT", "TEXT")
    FEED_URLS = [
        "http://rss.cnn.com/rss/cnn_latest.rss",
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://www.reddit.com/r/worldnews/.rss",
        "https://techcrunch.com/feed/",
        "https://www.npr.org/rss/rss.php?id=1001",
    ]

    # Autonomous Loop Settings - Optimized for better performance
    # Increased for more exploration
    CURIOSITY_CHANCE = float(os.environ.get("CURIOSITY_CHANCE", 0.4))
    # Increased for better learning
    REFLECTION_CHANCE = float(os.environ.get("REFLECTION_CHANCE", 0.15))
    # Reduced for more frequent operations
    LOOP_SLEEP_DURATION = int(os.environ.get("LOOP_SLEEP_DURATION", 7))
    # Reduced for faster recovery
    ERROR_SLEEP_DURATION = int(os.environ.get("ERROR_SLEEP_DURATION", 30))
    # Increased for more experimentation
    MAX_EXPERIMENT_LOOPS = int(os.environ.get("MAX_EXPERIMENT_LOOPS", 15))
    # Increased for more thorough processing
    MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", 15))
    # Increased to 20 minutes for complex research
    RESEARCH_TASK_TIMEOUT = int(os.environ.get("RESEARCH_TASK_TIMEOUT", 1200))

    # Emotional Intelligence Settings - Enhanced
    POSITIVE_MOODS = ['Confident', 'Curious', 'Reflective',
                      'Excited', 'Content', 'Optimistic', 'Creative', 'Focussed']
    NEGATIVE_MOODS = ['Frustrated', 'Stuck', 'Low Energy',
                      'Bored', 'Overwhelmed', 'Confused', 'Anxious']
    EMOTIONAL_PERSONA = "Adaptive"  # Changed to adaptive for better responses

    # Model Settings - Optimized for performance
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    # Changed to quality for better embeddings
    EMBEDDING_MODEL_TYPE = os.environ.get("EMBEDDING_MODEL_TYPE", "quality")
    EMBEDDING_USE_CUDA = os.environ.get("EMBEDDING_USE_CUDA", "True").lower() in [
        "true", "1", "yes"]  # Enabled by default
    # cuda, cpu, mps, or None for auto
    EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", None)

    # Background Task Intervals (in seconds) - Optimized for responsiveness
    DATA_COLLECTION_INTERVAL = int(os.environ.get(
        "DATA_COLLECTION_INTERVAL", 1800))  # Reduced to 30 minutes
    EVENT_DETECTION_INTERVAL = int(os.environ.get(
        "EVENT_DETECTION_INTERVAL", 300))  # Reduced to 5 minutes
    KNOWLEDGE_COMPRESSION_INTERVAL = int(os.environ.get(
        # Increased to 2 hours for better processing
        "KNOWLEDGE_COMPRESSION_INTERVAL", 7200))
    # Personality / Invention settings
    PERSONA_NAME = os.environ.get("PERSONA_NAME", "Ravana")
    PERSONA_ORIGIN = os.environ.get("PERSONA_ORIGIN", "Ancient Sri Lanka")
    PERSONA_CREATIVITY = float(os.environ.get("PERSONA_CREATIVITY", 0.7))
    # seconds between invention attempts
    INVENTION_INTERVAL = int(os.environ.get("INVENTION_INTERVAL", 7200))

    # Graceful Shutdown Configuration
    SHUTDOWN_TIMEOUT = int(os.environ.get("SHUTDOWN_TIMEOUT", 30))  # seconds
    GRACEFUL_SHUTDOWN_ENABLED = bool(os.environ.get(
        "GRACEFUL_SHUTDOWN_ENABLED", "True").lower() in ["true", "1", "yes"])
    STATE_PERSISTENCE_ENABLED = bool(os.environ.get(
        "STATE_PERSISTENCE_ENABLED", "True").lower() in ["true", "1", "yes"])
    SHUTDOWN_STATE_FILE = os.environ.get(
        "SHUTDOWN_STATE_FILE", "shutdown_state.json")
    FORCE_SHUTDOWN_AFTER = int(os.environ.get(
        "FORCE_SHUTDOWN_AFTER", 60))  # seconds

    # Enhanced Shutdown Configuration - Optimized
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

    # Memory Service Shutdown Configuration - Optimized
    MEMORY_SERVICE_SHUTDOWN_TIMEOUT = int(os.environ.get(
        "MEMORY_SERVICE_SHUTDOWN_TIMEOUT", 25))  # Increased for complete memory save
    POSTGRES_CONNECTION_TIMEOUT = int(os.environ.get(
        "POSTGRES_CONNECTION_TIMEOUT", 15))  # Increased for stable connections
    CHROMADB_PERSIST_ON_SHUTDOWN = bool(os.environ.get(
        "CHROMADB_PERSIST_ON_SHUTDOWN", "True").lower() in ["true", "1", "yes"])
    TEMP_FILE_CLEANUP_ENABLED = bool(os.environ.get(
        "TEMP_FILE_CLEANUP_ENABLED", "True").lower() in ["true", "1", "yes"])

    # Resource Cleanup Configuration - Optimized
    ACTION_CACHE_PERSIST = bool(os.environ.get(
        "ACTION_CACHE_PERSIST", "True").lower() in ["true", "1", "yes"])
    RESOURCE_CLEANUP_TIMEOUT = int(os.environ.get(
        "RESOURCE_CLEANUP_TIMEOUT", 20))  # Increased for thorough cleanup
    DATABASE_CLEANUP_TIMEOUT = int(os.environ.get(
        "DATABASE_CLEANUP_TIMEOUT", 25))  # Increased for complete cleanup

    # Snake Agent Configuration - Enhanced performance settings
    SNAKE_AGENT_ENABLED = bool(os.environ.get(
        "SNAKE_AGENT_ENABLED", "True").lower() in ["true", "1", "yes"])
    # 3 minutes default for better responsiveness
    SNAKE_AGENT_INTERVAL = int(os.environ.get("SNAKE_AGENT_INTERVAL", 180))
    SNAKE_OLLAMA_BASE_URL = os.environ.get(
        "SNAKE_OLLAMA_BASE_URL", "http://localhost:11434")

    # AI Provider Configuration for Snake Agent - Prioritizing electronhub and gemini
    SNAKE_PROVIDER_BASE_URL = os.environ.get(
        "SNAKE_PROVIDER_BASE_URL", "https://api.electronhub.ai")  # Prioritize electronhub
    SNAKE_PROVIDER_TIMEOUT = int(os.environ.get(
        "SNAKE_PROVIDER_TIMEOUT", 120))  # 2 minutes for API calls
    SNAKE_PROVIDER_KEEP_ALIVE = os.environ.get(
        "SNAKE_PROVIDER_KEEP_ALIVE", "10m")

    # Dual LLM Models for Snake Agent (AI Provider-based with fallback to Ollama)
    SNAKE_CODING_MODEL = {
        # Use electronhub by default
        "provider": os.environ.get("SNAKE_CODING_PROVIDER", "electronhub"),
        # Use free models from electronhub
        "model_name": os.environ.get("SNAKE_CODING_MODEL", "gpt-oss-20b:free"),
        # Use electronhub as default
        "base_url": os.environ.get("SNAKE_PROVIDER_BASE_URL", "https://api.electronhub.ai"),
        "api_key": os.environ.get("SNAKE_ELECTRONHUB_API_KEY", "ek-sVvxMYfdFQ0Kl6Aj2tmV7b8n5v0Y0sDHVsOUZWyx2vbs0AbuAc"),
        "temperature": float(os.environ.get("SNAKE_CODING_TEMPERATURE", "0.1")),
        "max_tokens": None if os.environ.get("SNAKE_CODING_MAX_TOKENS", "unlimited").lower() in ["unlimited", "none", "-1"] else int(os.environ.get("SNAKE_CODING_MAX_TOKENS", "4096")),
        "unlimited_mode": bool(os.environ.get("SNAKE_UNLIMITED_MODE", "True").lower() in ["true", "1", "yes"]),
        "chunk_size": int(os.environ.get("SNAKE_CHUNK_SIZE", "4096")),
        # Better timeout for API calls
        "timeout": int(os.environ.get("SNAKE_PROVIDER_TIMEOUT", 120)),
        "keep_alive": os.environ.get("SNAKE_PROVIDER_KEEP_ALIVE", "10m"),
        "fallback_provider": "ollama",  # Fallback to local if needed
        "fallback_model": "deepseek-coder:6.7b"
    }

    SNAKE_REASONING_MODEL = {
        # Use electronhub by default
        "provider": os.environ.get("SNAKE_REASONING_PROVIDER", "electronhub"),
        # Use free models from electronhub
        "model_name": os.environ.get("SNAKE_REASONING_MODEL", "deepseek-r1:free"),
        # Use electronhub as default
        "base_url": os.environ.get("SNAKE_PROVIDER_BASE_URL", "https://api.electronhub.ai"),
        "api_key": os.environ.get("SNAKE_ELECTRONHUB_API_KEY", "ek-sVvxMYfdFQ0Kl6Aj2tmV7b8n5v0Y0sDHVsOUZWyx2vbs0AbuAc"),
        "temperature": float(os.environ.get("SNAKE_REASONING_TEMPERATURE", "0.3")),
        "max_tokens": None if os.environ.get("SNAKE_REASONING_MAX_TOKENS", "unlimited").lower() in ["unlimited", "none", "none", "-1"] else int(os.environ.get("SNAKE_REASONING_MAX_TOKENS", "2048")),
        "unlimited_mode": bool(os.environ.get("SNAKE_UNLIMITED_MODE", "True").lower() in ["true", "1", "yes"]),
        "chunk_size": int(os.environ.get("SNAKE_CHUNK_SIZE", "2048")),
        # Better timeout for API calls
        "timeout": int(os.environ.get("SNAKE_PROVIDER_TIMEOUT", 120)),
        "keep_alive": os.environ.get("SNAKE_PROVIDER_KEEP_ALIVE", "10m"),
        "fallback_provider": "ollama",  # Fallback to local if needed
        "fallback_model": "llama3.1:8b"
    }

    # Alternative Model Options prioritizing electronhub and gemini (user can override via environment variables)
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

    # Snake Agent Graceful Shutdown Integration (extends existing shutdown config)
    SNAKE_SHUTDOWN_TIMEOUT = int(os.environ.get(
        "SNAKE_SHUTDOWN_TIMEOUT", 30))  # seconds
    SNAKE_STATE_PERSISTENCE = bool(os.environ.get(
        "SNAKE_STATE_PERSISTENCE", "True").lower() in ["true", "1", "yes"])

    # Enhanced Snake Agent Configuration - Optimized for Performance
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

    # Threading and Multiprocessing Limits - Optimized
    # Increased to 10 minutes for complex tasks
    SNAKE_TASK_TIMEOUT = float(os.environ.get("SNAKE_TASK_TIMEOUT", "600.0"))
    SNAKE_HEARTBEAT_INTERVAL = float(os.environ.get(
        "SNAKE_HEARTBEAT_INTERVAL", "5.0"))  # Reduced for better responsiveness
    # Increased for better throughput
    SNAKE_MAX_QUEUE_SIZE = int(os.environ.get("SNAKE_MAX_QUEUE_SIZE", "2000"))
    # Reduced to 30 minutes for better resource management
    SNAKE_CLEANUP_INTERVAL = float(
        os.environ.get("SNAKE_CLEANUP_INTERVAL", "1800.0"))

    # Peek prioritizer: enable lightweight peek+scoring to choose files before full analysis
    SNAKE_USE_PEEK_PRIORITIZER = bool(os.environ.get(
        # Enabled by default
        "SNAKE_USE_PEEK_PRIORITIZER", "True").lower() in ["true", "1", "yes"])

    # Load provider models from repository-level core/config.json if present.
    # This file is intended to list hosted providers (electronhub, zuki, etc.)
    PROVIDERS_CONFIG = {}
    try:
        _providers_path = os.path.join(
            os.path.dirname(__file__), 'config.json')
        if os.path.exists(_providers_path):
            with open(_providers_path, 'r', encoding='utf-8') as _pf:
                PROVIDERS_CONFIG = json.load(_pf)
    except Exception:
        PROVIDERS_CONFIG = {}

    @staticmethod
    def get_provider_model(provider_name: str, role: str = 'reasoning') -> dict:
        """Return a model configuration dict for the given provider and role.

        The selection is heuristic: it inspects the provider's "models" list
        and picks the first model whose name matches role-specific keywords.
        Falls back to the first listed model if nothing matches.
        The returned dict has keys: provider, model_name, base_url, temperature, timeout, keep_alive.
        """
        prov = Config.PROVIDERS_CONFIG.get(
            provider_name) if Config.PROVIDERS_CONFIG else None
        if not prov:
            return {}

        models = prov.get('models', [])
        # Normalize model names to string, handle objects
        norm_models = []
        for m in models:
            if isinstance(m, dict):
                name = m.get('name') or m.get('model') or m.get('model_name')
                if name:
                    norm_models.append(name)
            elif isinstance(m, str):
                norm_models.append(m)

        role = (role or 'reasoning').lower()
        # simple keyword maps
        role_keywords = {
            'coding': ['coder', 'code', 'codellama', 'starcoder', 'coding', 'coder-'],
            'reasoning': ['reason', 'gpt', 'deepseek', 'llama', 'qwen', 'gpt-oss', 'gpt-4o', 'mistral'],
            'multimodal': ['vision', 'image', 'multimodal']
        }

        keywords = role_keywords.get(role, role_keywords['reasoning'])

        chosen = None
        for nm in norm_models:
            lname = nm.lower()
            for kw in keywords:
                if kw in lname:
                    chosen = nm
                    break
            if chosen:
                break

        if not chosen and norm_models:
            chosen = norm_models[0]

        # Build a returned config merging provider base info
        result = {
            'provider': provider_name,
            'model_name': chosen,
            'base_url': prov.get('base_url') or prov.get('baseUrl') or prov.get('endpoint') or '',
            'temperature': float(os.environ.get('SNAKE_CODING_TEMPERATURE', '0.1')),
            'max_tokens': None,
            'unlimited_mode': True,
            'chunk_size': int(os.environ.get('SNAKE_CHUNK_SIZE', '4096')),
            'timeout': int(os.environ.get('SNAKE_OLLAMA_TIMEOUT', 3000)),
            'keep_alive': os.environ.get('SNAKE_OLLAMA_KEEP_ALIVE', '10m')
        }

        return result

    # Blog Integration Configuration - Optimized
    BLOG_ENABLED = bool(os.environ.get("RAVANA_BLOG_ENABLED",
                        "True").lower() in ["true", "1", "yes"])
    BLOG_API_URL = os.environ.get(
        "RAVANA_BLOG_API_URL", "https://ravana-blog.netlify.app/api/publish")
    BLOG_AUTH_TOKEN = os.environ.get(
        "RAVANA_BLOG_AUTH_TOKEN", "ravana_secret_token_2024")

    # Content Generation Settings - Enhanced
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

    # Publishing Behavior - Optimized
    BLOG_AUTO_PUBLISH_ENABLED = bool(os.environ.get("BLOG_AUTO_PUBLISH_ENABLED", "True").lower() in [
                                     "true", "1", "yes"])  # Enabled by default
    BLOG_REQUIRE_APPROVAL = bool(os.environ.get("BLOG_REQUIRE_APPROVAL", "False").lower() in [
                                 "true", "1", "yes"])  # Disabled for faster publishing
    BLOG_PUBLISH_FREQUENCY_HOURS = int(os.environ.get(
        "BLOG_PUBLISH_FREQUENCY_HOURS", "12"))  # Increased frequency

    # API Communication Settings - Optimized
    BLOG_TIMEOUT_SECONDS = int(os.environ.get(
        "BLOG_TIMEOUT_SECONDS", "60"))  # Increased for reliability
    # Increased for better reliability
    BLOG_RETRY_ATTEMPTS = int(os.environ.get("BLOG_RETRY_ATTEMPTS", "5"))
    BLOG_RETRY_BACKOFF_FACTOR = float(os.environ.get(
        "BLOG_RETRY_BACKOFF_FACTOR", "1.5"))  # Reduced for faster retries
    # Increased for better handling
    BLOG_MAX_RETRY_DELAY = int(os.environ.get("BLOG_MAX_RETRY_DELAY", "120"))

    # Content Quality Settings - Enhanced
    BLOG_CONTENT_STYLES = ["technical", "casual", "academic", "creative",
                           "philosophical", "analytical", "insightful", "explanatory"]
    BLOG_MEMORY_CONTEXT_DAYS = int(os.environ.get(
        "BLOG_MEMORY_CONTEXT_DAYS", "14"))  # Increased for better context
    BLOG_INCLUDE_MOOD_CONTEXT = bool(os.environ.get(
        "BLOG_INCLUDE_MOOD_CONTEXT", "True").lower() in ["true", "1", "yes"])

    # Conversational AI Configuration - Optimized
    CONVERSATIONAL_AI_ENABLED = bool(os.environ.get(
        "CONVERSATIONAL_AI_ENABLED", "True").lower() in ["true", "1", "yes"])
    CONVERSATIONAL_AI_START_DELAY = int(os.environ.get(
        "CONVERSATIONAL_AI_START_DELAY", 2))  # Reduced for faster startup
