import os
import json


class Config:
    # Use a class variable to hold the singleton instance
    _instance = None

    def __new__(cls):
        # Create a single instance of the class
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance for testing purposes."""
        cls._instance = None
    
    def _str_to_bool(self, value):
        """Convert string to boolean in a more robust way."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value)

    def __init__(self):
        # Only initialize once per instance
        if hasattr(self, 'initialized'):
            return

        self.DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///ravana_agi.db")
        self.LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
        self.LOG_FORMAT = os.environ.get("LOG_FORMAT", "TEXT")
        self.FEED_URLS = [
            "http://rss.cnn.com/rss/cnn_latest.rss",
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://www.reddit.com/r/worldnews/.rss",
            "https://techcrunch.com/feed/",
            "https://www.npr.org/rss/rss.php?id=1001",
        ]

        # Autonomous Loop Settings - Optimized for better performance
        # Increased for more exploration
        self.CURIOSITY_CHANCE = float(os.environ.get("CURIOSITY_CHANCE", 0.4))
        # Increased for better learning
        self.REFLECTION_CHANCE = float(os.environ.get("REFLECTION_CHANCE", 0.15))
        # Reduced for more frequent operations
        self.LOOP_SLEEP_DURATION = int(os.environ.get("LOOP_SLEEP_DURATION", 7))
        # Reduced for faster recovery
        self.ERROR_SLEEP_DURATION = int(os.environ.get("ERROR_SLEEP_DURATION", 30))
        # Increased for more experimentation
        self.MAX_EXPERIMENT_LOOPS = int(os.environ.get("MAX_EXPERIMENT_LOOPS", 15))
        # Increased for more thorough processing
        self.MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", 15))
        # Increased to 20 minutes for complex research
        self.RESEARCH_TASK_TIMEOUT = int(os.environ.get("RESEARCH_TASK_TIMEOUT", 1200))

        # Emotional Intelligence Settings - Enhanced
        self.POSITIVE_MOODS = ['Confident', 'Curious', 'Reflective',
                          'Excited', 'Content', 'Optimistic', 'Creative', 'Focussed']
        self.NEGATIVE_MOODS = ['Frustrated', 'Stuck', 'Low Energy',
                          'Bored', 'Overwhelmed', 'Confused', 'Anxious']
        self.EMOTIONAL_PERSONA = "Adaptive"  # Changed to adaptive for better responses

        # Model Settings - Optimized for performance
        self.EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        # Changed to quality for better embeddings
        self.EMBEDDING_MODEL_TYPE = os.environ.get("EMBEDDING_MODEL_TYPE", "quality")
        self.EMBEDDING_USE_CUDA = self._str_to_bool(os.environ.get("EMBEDDING_USE_CUDA", "True"))  # Enabled by default
        # cuda, cpu, mps, or None for auto
        self.EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", None)

        # Background Task Intervals (in seconds) - Optimized for responsiveness
        self.DATA_COLLECTION_INTERVAL = int(os.environ.get(
            "DATA_COLLECTION_INTERVAL", 1800))  # Reduced to 30 minutes
        self.EVENT_DETECTION_INTERVAL = int(os.environ.get(
            "EVENT_DETECTION_INTERVAL", 300))  # Reduced to 5 minutes
        self.KNOWLEDGE_COMPRESSION_INTERVAL = int(os.environ.get(
            # Increased to 2 hours for better processing
            "KNOWLEDGE_COMPRESSION_INTERVAL", 7200))
        # Personality / Invention settings
        self.PERSONA_NAME = os.environ.get("PERSONA_NAME", "Ravana")
        self.PERSONA_ORIGIN = os.environ.get("PERSONA_ORIGIN", "Ancient Sri Lanka")
        self.PERSONA_CREATIVITY = float(os.environ.get("PERSONA_CREATIVITY", 0.7))
        # seconds between invention attempts
        self.INVENTION_INTERVAL = int(os.environ.get("INVENTION_INTERVAL", 7200))

        # Graceful Shutdown Configuration
        self.SHUTDOWN_TIMEOUT = int(os.environ.get("SHUTDOWN_TIMEOUT", 30))  # seconds
        self.GRACEFUL_SHUTDOWN_ENABLED = self._str_to_bool(os.environ.get("GRACEFUL_SHUTDOWN_ENABLED", "True"))
        self.STATE_PERSISTENCE_ENABLED = self._str_to_bool(os.environ.get("STATE_PERSISTENCE_ENABLED", "True"))
        self.SHUTDOWN_STATE_FILE = os.environ.get(
            "SHUTDOWN_STATE_FILE", "shutdown_state.json")
        self.FORCE_SHUTDOWN_AFTER = int(os.environ.get(
            "FORCE_SHUTDOWN_AFTER", 60))  # seconds

        # Enhanced Shutdown Configuration - Optimized
        self.SHUTDOWN_HEALTH_CHECK_ENABLED = self._str_to_bool(os.environ.get("SHUTDOWN_HEALTH_CHECK_ENABLED", "True"))
        self.SHUTDOWN_BACKUP_ENABLED = self._str_to_bool(os.environ.get("SHUTDOWN_BACKUP_ENABLED", "True"))
        # Increased for better reliability
        self.SHUTDOWN_BACKUP_COUNT = int(os.environ.get("SHUTDOWN_BACKUP_COUNT", 10))
        self.SHUTDOWN_STATE_VALIDATION_ENABLED = self._str_to_bool(os.environ.get("SHUTDOWN_STATE_VALIDATION_ENABLED", "True"))
        self.SHUTDOWN_VALIDATION_ENABLED = self._str_to_bool(os.environ.get("SHUTDOWN_VALIDATION_ENABLED", "True"))
        self.SHUTDOWN_COMPRESSION_ENABLED = self._str_to_bool(os.environ.get("SHUTDOWN_COMPRESSION_ENABLED", "True"))
        self.COMPONENT_PREPARE_TIMEOUT = float(os.environ.get(
            "COMPONENT_PREPARE_TIMEOUT", 15.0))  # Increased for complex components
        self.COMPONENT_SHUTDOWN_TIMEOUT = float(os.environ.get(
            "COMPONENT_SHUTDOWN_TIMEOUT", 25.0))  # Increased for thorough shutdown

        # Memory Service Shutdown Configuration - Optimized
        self.MEMORY_SERVICE_SHUTDOWN_TIMEOUT = int(os.environ.get(
            "MEMORY_SERVICE_SHUTDOWN_TIMEOUT", 25))  # Increased for complete memory save
        self.POSTGRES_CONNECTION_TIMEOUT = int(os.environ.get(
            "POSTGRES_CONNECTION_TIMEOUT", 15))  # Increased for stable connections
        self.CHROMADB_PERSIST_ON_SHUTDOWN = self._str_to_bool(os.environ.get("CHROMADB_PERSIST_ON_SHUTDOWN", "True"))
        self.TEMP_FILE_CLEANUP_ENABLED = self._str_to_bool(os.environ.get("TEMP_FILE_CLEANUP_ENABLED", "True"))

        # Resource Cleanup Configuration - Optimized
        self.ACTION_CACHE_PERSIST = self._str_to_bool(os.environ.get("ACTION_CACHE_PERSIST", "True"))
        self.RESOURCE_CLEANUP_TIMEOUT = int(os.environ.get(
            "RESOURCE_CLEANUP_TIMEOUT", 20))  # Increased for thorough cleanup
        self.DATABASE_CLEANUP_TIMEOUT = int(os.environ.get(
            "DATABASE_CLEANUP_TIMEOUT", 25))  # Increased for complete cleanup

        # Snake Agent Configuration - Enhanced performance settings
        self.SNAKE_AGENT_ENABLED = self._str_to_bool(os.environ.get("SNAKE_AGENT_ENABLED", "True"))
        # 3 minutes default for better responsiveness
        self.SNAKE_AGENT_INTERVAL = int(os.environ.get("SNAKE_AGENT_INTERVAL", 180))
        self.SNAKE_OLLAMA_BASE_URL = os.environ.get(
            "SNAKE_OLLAMA_BASE_URL", "http://localhost:11434")

        # AI Provider Configuration for Snake Agent - Prioritizing electronhub and gemini
        self.SNAKE_PROVIDER_BASE_URL = os.environ.get(
            "SNAKE_PROVIDER_BASE_URL", "https://api.electronhub.ai")  # Prioritize electronhub
        self.SNAKE_PROVIDER_TIMEOUT = int(os.environ.get(
            "SNAKE_PROVIDER_TIMEOUT", 120))  # 2 minutes for API calls
        self.SNAKE_PROVIDER_KEEP_ALIVE = os.environ.get(
            "SNAKE_PROVIDER_KEEP_ALIVE", "10m")

        # Dual LLM Models for Snake Agent (AI Provider-based with fallback to Ollama)
        self.SNAKE_CODING_MODEL = {
            # Use electronhub by default
            "provider": os.environ.get("SNAKE_CODING_PROVIDER", "electronhub"),
            # Use free models from electronhub
            "model_name": os.environ.get("SNAKE_CODING_MODEL", "gpt-oss-20b:free"),
            # Use electronhub as default
            "base_url": os.environ.get("SNAKE_PROVIDER_BASE_URL", "https://api.electronhub.ai"),
            "api_key": os.environ.get("SNAKE_ELECTRONHUB_API_KEY", "ek-sVvxMYfdFQ0Kl6Aj2tmV7b8n5v0Y0sDHVsOUZWyx2vbs0AbuAc"),
            "temperature": float(os.environ.get("SNAKE_CODING_TEMPERATURE", "0.1")),
            "max_tokens": None if os.environ.get("SNAKE_CODING_MAX_TOKENS", "unlimited").lower() in ["unlimited", "none", "none", "-1"] else int(os.environ.get("SNAKE_CODING_MAX_TOKENS", "4096")),
            "unlimited_mode": self._str_to_bool(os.environ.get("SNAKE_UNLIMITED_MODE", "True")),
            "chunk_size": int(os.environ.get("SNAKE_CHUNK_SIZE", "4096")),
            # Better timeout for API calls
            "timeout": int(os.environ.get("SNAKE_PROVIDER_TIMEOUT", 120)),
            "keep_alive": os.environ.get("SNAKE_PROVIDER_KEEP_ALIVE", "10m"),
            "fallback_provider": "ollama",  # Fallback to local if needed
            "fallback_model": "deepseek-coder:6.7b"
        }

        self.SNAKE_REASONING_MODEL = {
            # Use electronhub by default
            "provider": os.environ.get("SNAKE_REASONING_PROVIDER", "electronhub"),
            # Use free models from electronhub
            "model_name": os.environ.get("SNAKE_REASONING_MODEL", "deepseek-r1:free"),
            # Use electronhub as default
            "base_url": os.environ.get("SNAKE_PROVIDER_BASE_URL", "https://api.electronhub.ai"),
            "api_key": os.environ.get("SNAKE_ELECTRONHUB_API_KEY", "ek-sVvxMYfdFQ0Kl6Aj2tmV7b8n5v0Y0sDHVsOUZWyx2vbs0AbuAc"),
            "temperature": float(os.environ.get("SNAKE_REASONING_TEMPERATURE", "0.3")),
            "max_tokens": None if os.environ.get("SNAKE_REASONING_MAX_TOKENS", "unlimited").lower() in ["unlimited", "none", "none", "-1"] else int(os.environ.get("SNAKE_REASONING_MAX_TOKENS", "2048")),
            "unlimited_mode": self._str_to_bool(os.environ.get("SNAKE_UNLIMITED_MODE", "True")),
            "chunk_size": int(os.environ.get("SNAKE_CHUNK_SIZE", "2048")),
            # Better timeout for API calls
            "timeout": int(os.environ.get("SNAKE_PROVIDER_TIMEOUT", 120)),
            "keep_alive": os.environ.get("SNAKE_PROVIDER_KEEP_ALIVE", "10m"),
            "fallback_provider": "ollama",  # Fallback to local if needed
            "fallback_model": "llama3.1:8b"
        }

        # Alternative Model Options prioritizing electronhub and gemini (user can override via environment variables)
        self.SNAKE_AVAILABLE_CODING_MODELS = [
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

        self.SNAKE_AVAILABLE_REASONING_MODELS = [
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
        self.SNAKE_SANDBOX_TIMEOUT = int(os.environ.get(
            "SNAKE_SANDBOX_TIMEOUT", 60))  # seconds
        self.SNAKE_MAX_FILE_SIZE = int(os.environ.get(
            "SNAKE_MAX_FILE_SIZE", 1048576))  # 1MB
        self.SNAKE_BLACKLIST_PATHS = os.environ.get("SNAKE_BLACKLIST_PATHS", "").split(
            ",") if os.environ.get("SNAKE_BLACKLIST_PATHS") else []
        self.SNAKE_APPROVAL_REQUIRED = self._str_to_bool(os.environ.get("SNAKE_APPROVAL_REQUIRED", "True"))

        # Snake Agent Safety Configuration
        self.SNAKE_SANDBOX_TIMEOUT = int(os.environ.get(
            "SNAKE_SANDBOX_TIMEOUT", 60))  # seconds
        self.SNAKE_MAX_FILE_SIZE = int(os.environ.get(
            "SNAKE_MAX_FILE_SIZE", 1048576))  # 1MB
        self.SNAKE_BLACKLIST_PATHS = os.environ.get("SNAKE_BLACKLIST_PATHS", "").split(
            ",") if os.environ.get("SNAKE_BLACKLIST_PATHS") else []
        self.SNAKE_APPROVAL_REQUIRED = self._str_to_bool(os.environ.get("SNAKE_APPROVAL_REQUIRED", "True"))

        # Communication Configuration
        self.SNAKE_COMM_CHANNEL = os.environ.get("SNAKE_COMM_CHANNEL", "memory_service")
        self.SNAKE_COMM_PRIORITY_THRESHOLD = float(
            os.environ.get("SNAKE_COMM_PRIORITY_THRESHOLD", "0.8"))

        # Snake Agent Graceful Shutdown Integration (extends existing shutdown config)
        self.SNAKE_SHUTDOWN_TIMEOUT = int(os.environ.get(
            "SNAKE_SHUTDOWN_TIMEOUT", 30))  # seconds
        self.SNAKE_STATE_PERSISTENCE = self._str_to_bool(os.environ.get("SNAKE_STATE_PERSISTENCE", "True"))

        # Enhanced Snake Agent Configuration - Optimized for Performance
        self.SNAKE_ENHANCED_MODE = self._str_to_bool(os.environ.get("SNAKE_ENHANCED_MODE", "True"))
        # Increased for better concurrency
        self.SNAKE_MAX_THREADS = int(os.environ.get("SNAKE_MAX_THREADS", "12"))
        # Increased for better concurrency
        self.SNAKE_MAX_PROCESSES = int(os.environ.get("SNAKE_MAX_PROCESSES", "6"))
        # Increased for better analysis
        self.SNAKE_ANALYSIS_THREADS = int(os.environ.get("SNAKE_ANALYSIS_THREADS", "4"))
        self.SNAKE_MONITOR_INTERVAL = float(os.environ.get(
            "SNAKE_MONITOR_INTERVAL", "1.0"))  # Reduced for faster monitoring
        self.SNAKE_PERF_MONITORING = self._str_to_bool(os.environ.get("SNAKE_PERF_MONITORING", "True"))
        self.SNAKE_AUTO_RECOVERY = self._str_to_bool(os.environ.get("SNAKE_AUTO_RECOVERY", "True"))
        self.SNAKE_LOG_RETENTION_DAYS = int(os.environ.get(
            "SNAKE_LOG_RETENTION_DAYS", "60"))  # Increased retention

        # Threading and Multiprocessing Limits - Optimized
        # Increased to 10 minutes for complex tasks
        self.SNAKE_TASK_TIMEOUT = float(os.environ.get("SNAKE_TASK_TIMEOUT", "600.0"))
        self.SNAKE_HEARTBEAT_INTERVAL = float(os.environ.get(
            "SNAKE_HEARTBEAT_INTERVAL", "5.0"))  # Reduced for better responsiveness
        # Increased for better throughput
        self.SNAKE_MAX_QUEUE_SIZE = int(os.environ.get("SNAKE_MAX_QUEUE_SIZE", "2000"))
        # Reduced to 30 minutes for better resource management
        self.SNAKE_CLEANUP_INTERVAL = float(
            os.environ.get("SNAKE_CLEANUP_INTERVAL", "1800.0"))

        # Peek prioritizer: enable lightweight peek+scoring to choose files before full analysis
        self.SNAKE_USE_PEEK_PRIORITIZER = self._str_to_bool(os.environ.get("SNAKE_USE_PEEK_PRIORITIZER", "True"))

        # Load provider models from repository-level core/config.json if present.
        # This file is intended to list hosted providers (electronhub, zuki, etc.)
        self.PROVIDERS_CONFIG = {}
        try:
            _providers_path = os.path.join(
                os.path.dirname(__file__), 'config.json')
            if os.path.exists(_providers_path):
                with open(_providers_path, 'r', encoding='utf-8') as _pf:
                    self.PROVIDERS_CONFIG = json.load(_pf)
        except Exception:
            self.PROVIDERS_CONFIG = {}

        # Local model configuration for main system (to replace external providers)
        self.MAIN_SYSTEM_LOCAL_MODEL = {
            'provider': 'ollama',
            'model_name': os.environ.get("MAIN_MODEL_NAME", "llama3.1:8b"),  # Use local model by default
            'base_url': os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            'temperature': float(os.environ.get("MAIN_MODEL_TEMPERATURE", "0.7")),
            'max_tokens': int(os.environ.get("MAIN_MODEL_MAX_TOKENS", "2048")) if os.environ.get("MAIN_MODEL_MAX_TOKENS", "2048") != "none" else None,
            'timeout': int(os.environ.get("MAIN_MODEL_TIMEOUT", "300")),  # 5 minutes
            'keep_alive': os.environ.get("MAIN_MODEL_KEEP_ALIVE", "5m")
        }

        # Blog Integration Configuration - Optimized
        self.BLOG_ENABLED = self._str_to_bool(os.environ.get("RAVANA_BLOG_ENABLED", "True"))
        self.BLOG_API_URL = os.environ.get(
            "RAVANA_BLOG_API_URL", "https://ravana-blog.netlify.app/api/publish")
        self.BLOG_AUTH_TOKEN = os.environ.get(
            "RAVANA_BLOG_AUTH_TOKEN", "ravana_secret_token_2024")

        # Content Generation Settings - Enhanced
        self.BLOG_DEFAULT_STYLE = os.environ.get("BLOG_DEFAULT_STYLE", "technical")
        self.BLOG_MAX_CONTENT_LENGTH = int(os.environ.get(
            "BLOG_MAX_CONTENT_LENGTH", "1000000"))  # Effectively unlimited
        # Reduced for more frequent posts
        self.BLOG_MIN_CONTENT_LENGTH = int(
            os.environ.get("BLOG_MIN_CONTENT_LENGTH", "300"))
        self.BLOG_AUTO_TAGGING_ENABLED = self._str_to_bool(os.environ.get("BLOG_AUTO_TAGGING_ENABLED", "True"))
        # Increased for better categorization
        self.BLOG_MAX_TAGS = int(os.environ.get("BLOG_MAX_TAGS", "15"))

        # Publishing Behavior - Optimized
        self.BLOG_AUTO_PUBLISH_ENABLED = self._str_to_bool(os.environ.get("BLOG_AUTO_PUBLISH_ENABLED", "True"))  # Enabled by default
        self.BLOG_REQUIRE_APPROVAL = self._str_to_bool(os.environ.get("BLOG_REQUIRE_APPROVAL", "False"))  # Disabled for faster publishing
        self.BLOG_PUBLISH_FREQUENCY_HOURS = int(os.environ.get(
            "BLOG_PUBLISH_FREQUENCY_HOURS", "12"))  # Increased frequency

        # API Communication Settings - Optimized
        self.BLOG_TIMEOUT_SECONDS = int(os.environ.get(
            "BLOG_TIMEOUT_SECONDS", "60"))  # Increased for reliability
        # Increased for better reliability
        self.BLOG_RETRY_ATTEMPTS = int(os.environ.get("BLOG_RETRY_ATTEMPTS", "5"))
        self.BLOG_RETRY_BACKOFF_FACTOR = float(os.environ.get(
            "BLOG_RETRY_BACKOFF_FACTOR", "1.5"))  # Reduced for faster retries
        # Increased for better handling
        self.BLOG_MAX_RETRY_DELAY = int(os.environ.get("BLOG_MAX_RETRY_DELAY", "120"))

        # Content Quality Settings - Enhanced
        self.BLOG_CONTENT_STYLES = ["technical", "casual", "academic", "creative",
                               "philosophical", "analytical", "insightful", "explanatory"]
        self.BLOG_MEMORY_CONTEXT_DAYS = int(os.environ.get(
            "BLOG_MEMORY_CONTEXT_DAYS", "14"))  # Increased for better context
        self.BLOG_INCLUDE_MOOD_CONTEXT = self._str_to_bool(os.environ.get("BLOG_INCLUDE_MOOD_CONTEXT", "True"))

        # Conversational AI Configuration - Optimized
        self.CONVERSATIONAL_AI_ENABLED = self._str_to_bool(os.environ.get("CONVERSATIONAL_AI_ENABLED", "True"))
        self.CONVERSATIONAL_AI_START_DELAY = int(os.environ.get(
            "CONVERSATIONAL_AI_START_DELAY", 2))  # Reduced for faster startup
        
        # Snake Agent Enhanced Logging Configuration
        self.SNAKE_LOG_ERRORS_TO_FILE = self._str_to_bool(os.environ.get("SNAKE_LOG_ERRORS_TO_FILE", "True"))
        self.SNAKE_LOG_INTERACTIONS_TO_JSON = self._str_to_bool(os.environ.get("SNAKE_LOG_INTERACTIONS_TO_JSON", "True"))
        self.SNAKE_LOG_ERROR_FILE = os.environ.get(
            "SNAKE_LOG_ERROR_FILE", "snake_logs/snake_errors.log")
        self.SNAKE_LOG_INTERACTIONS_FILE = os.environ.get(
            "SNAKE_LOG_INTERACTIONS_FILE", "snake_logs/interactions.json")
        self.SNAKE_LOG_MAX_FILE_SIZE = int(os.environ.get(
            "SNAKE_LOG_MAX_FILE_SIZE", 10485760))  # 10MB default
        self.SNAKE_LOG_BACKUP_COUNT = int(os.environ.get(
            "SNAKE_LOG_BACKUP_COUNT", 5))
        
        # Mark as initialized
        self.initialized = True

    def __getattr__(self, name):
        """Provide default values for missing configuration attributes."""
        # Define default values for known config attributes
        defaults = {
            'MAIN_SYSTEM_LOCAL_MODEL': {
                'provider': 'ollama',
                'model_name': os.environ.get("MAIN_MODEL_NAME", "llama3.1:8b"),
                'base_url': os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                'temperature': float(os.environ.get("MAIN_MODEL_TEMPERATURE", "0.7")),
                'max_tokens': int(os.environ.get("MAIN_MODEL_MAX_TOKENS", "2048")) if os.environ.get("MAIN_MODEL_MAX_TOKENS", "2048") != "none" else None,
                'timeout': int(os.environ.get("MAIN_MODEL_TIMEOUT", "300")),
                'keep_alive': os.environ.get("MAIN_MODEL_KEEP_ALIVE", "5m")
            },
            'CURIOSITY_CHANCE': float(os.environ.get("CURIOSITY_CHANCE", 0.4)),
            'REFLECTION_CHANCE': float(os.environ.get("REFLECTION_CHANCE", 0.15)),
            'LOOP_SLEEP_DURATION': int(os.environ.get("LOOP_SLEEP_DURATION", 7)),
            'ERROR_SLEEP_DURATION': int(os.environ.get("ERROR_SLEEP_DURATION", 30)),
            'MAX_EXPERIMENT_LOOPS': int(os.environ.get("MAX_EXPERIMENT_LOOPS", 15)),
            'MAX_ITERATIONS': int(os.environ.get("MAX_ITERATIONS", 15)),
            'RESEARCH_TASK_TIMEOUT': int(os.environ.get("RESEARCH_TASK_TIMEOUT", 1200)),
            'EMBEDDING_MODEL': os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            'EMBEDDING_MODEL_TYPE': os.environ.get("EMBEDDING_MODEL_TYPE", "quality"),
            'EMBEDDING_USE_CUDA': self._str_to_bool(os.environ.get("EMBEDDING_USE_CUDA", "True")),
            'EMBEDDING_DEVICE': os.environ.get("EMBEDDING_DEVICE", None),
            'DATA_COLLECTION_INTERVAL': int(os.environ.get("DATA_COLLECTION_INTERVAL", 1800)),
            'EVENT_DETECTION_INTERVAL': int(os.environ.get("EVENT_DETECTION_INTERVAL", 300)),
            'KNOWLEDGE_COMPRESSION_INTERVAL': int(os.environ.get("KNOWLEDGE_COMPRESSION_INTERVAL", 7200)),
            'PERSONA_NAME': os.environ.get("PERSONA_NAME", "Ravana"),
            'PERSONA_ORIGIN': os.environ.get("PERSONA_ORIGIN", "Ancient Sri Lanka"),
            'PERSONA_CREATIVITY': float(os.environ.get("PERSONA_CREATIVITY", 0.7)),
            'INVENTION_INTERVAL': int(os.environ.get("INVENTION_INTERVAL", 7200)),
            'SHUTDOWN_TIMEOUT': int(os.environ.get("SHUTDOWN_TIMEOUT", 30)),
            'GRACEFUL_SHUTDOWN_ENABLED': self._str_to_bool(os.environ.get("GRACEFUL_SHUTDOWN_ENABLED", "True")),
            'STATE_PERSISTENCE_ENABLED': self._str_to_bool(os.environ.get("STATE_PERSISTENCE_ENABLED", "True")),
            'SHUTDOWN_STATE_FILE': os.environ.get("SHUTDOWN_STATE_FILE", "shutdown_state.json"),
            'FORCE_SHUTDOWN_AFTER': int(os.environ.get("FORCE_SHUTDOWN_AFTER", 60)),
            'SHUTDOWN_HEALTH_CHECK_ENABLED': self._str_to_bool(os.environ.get("SHUTDOWN_HEALTH_CHECK_ENABLED", "True")),
            'SHUTDOWN_BACKUP_ENABLED': self._str_to_bool(os.environ.get("SHUTDOWN_BACKUP_ENABLED", "True")),
            'SHUTDOWN_BACKUP_COUNT': int(os.environ.get("SHUTDOWN_BACKUP_COUNT", 10)),
            'SHUTDOWN_STATE_VALIDATION_ENABLED': self._str_to_bool(os.environ.get("SHUTDOWN_STATE_VALIDATION_ENABLED", "True")),
            'SHUTDOWN_VALIDATION_ENABLED': self._str_to_bool(os.environ.get("SHUTDOWN_VALIDATION_ENABLED", "True")),
            'SHUTDOWN_COMPRESSION_ENABLED': self._str_to_bool(os.environ.get("SHUTDOWN_COMPRESSION_ENABLED", "True")),
            'COMPONENT_PREPARE_TIMEOUT': float(os.environ.get("COMPONENT_PREPARE_TIMEOUT", 15.0)),
            'COMPONENT_SHUTDOWN_TIMEOUT': float(os.environ.get("COMPONENT_SHUTDOWN_TIMEOUT", 25.0)),
            'MEMORY_SERVICE_SHUTDOWN_TIMEOUT': int(os.environ.get("MEMORY_SERVICE_SHUTDOWN_TIMEOUT", 25)),
            'POSTGRES_CONNECTION_TIMEOUT': int(os.environ.get("POSTGRES_CONNECTION_TIMEOUT", 15)),
            'CHROMADB_PERSIST_ON_SHUTDOWN': self._str_to_bool(os.environ.get("CHROMADB_PERSIST_ON_SHUTDOWN", "True")),
            'TEMP_FILE_CLEANUP_ENABLED': self._str_to_bool(os.environ.get("TEMP_FILE_CLEANUP_ENABLED", "True")),
            'ACTION_CACHE_PERSIST': self._str_to_bool(os.environ.get("ACTION_CACHE_PERSIST", "True")),
            'RESOURCE_CLEANUP_TIMEOUT': int(os.environ.get("RESOURCE_CLEANUP_TIMEOUT", 20)),
            'DATABASE_CLEANUP_TIMEOUT': int(os.environ.get("DATABASE_CLEANUP_TIMEOUT", 25)),
            'SNAKE_AGENT_ENABLED': self._str_to_bool(os.environ.get("SNAKE_AGENT_ENABLED", "True")),
            'SNAKE_AGENT_INTERVAL': int(os.environ.get("SNAKE_AGENT_INTERVAL", 180)),
            'SNAKE_OLLAMA_BASE_URL': os.environ.get("SNAKE_OLLAMA_BASE_URL", "http://localhost:11434"),
            'SNAKE_MAX_FILE_SIZE': int(os.environ.get("SNAKE_MAX_FILE_SIZE", 1048576)),
            'SNAKE_COMM_CHANNEL': os.environ.get("SNAKE_COMM_CHANNEL", "memory_service"),
            'SNAKE_APPROVAL_REQUIRED': self._str_to_bool(os.environ.get("SNAKE_APPROVAL_REQUIRED", "True")),
            'SNAKE_SANDBOX_TIMEOUT': int(os.environ.get("SNAKE_SANDBOX_TIMEOUT", 60)),
            'SNAKE_BLACKLIST_PATHS': os.environ.get("SNAKE_BLACKLIST_PATHS", "").split(",") if os.environ.get("SNAKE_BLACKLIST_PATHS") else [],
            'SNAKE_COMM_PRIORITY_THRESHOLD': float(os.environ.get("SNAKE_COMM_PRIORITY_THRESHOLD", "0.8")),
            'BLOG_ENABLED': self._str_to_bool(os.environ.get("RAVANA_BLOG_ENABLED", "True")),
            'BLOG_API_URL': os.environ.get("RAVANA_BLOG_API_URL", "https://ravana-blog.netlify.app/api/publish"),
            'BLOG_AUTH_TOKEN': os.environ.get("RAVANA_BLOG_AUTH_TOKEN", "ravana_secret_token_2024"),
            'CONVERSATIONAL_AI_ENABLED': self._str_to_bool(os.environ.get("CONVERSATIONAL_AI_ENABLED", "True")),
            'SNAKE_LOG_ERRORS_TO_FILE': self._str_to_bool(os.environ.get("SNAKE_LOG_ERRORS_TO_FILE", "True")),
            'SNAKE_LOG_INTERACTIONS_TO_JSON': self._str_to_bool(os.environ.get("SNAKE_LOG_INTERACTIONS_TO_JSON", "True")),
            'SNAKE_LOG_ERROR_FILE': os.environ.get("SNAKE_LOG_ERROR_FILE", "snake_logs/snake_errors.log"),
            'SNAKE_LOG_INTERACTIONS_FILE': os.environ.get("SNAKE_LOG_INTERACTIONS_FILE", "snake_logs/interactions.json"),
            'SNAKE_LOG_MAX_FILE_SIZE': int(os.environ.get("SNAKE_LOG_MAX_FILE_SIZE", 10485760)),
            'SNAKE_LOG_BACKUP_COUNT': int(os.environ.get("SNAKE_LOG_BACKUP_COUNT", 5)),
        }
        
        if name in defaults:
            # Set the default value and return it
            setattr(self, name, defaults[name])
            return getattr(self, name)
        else:
            # For other missing attributes, raise AttributeError to maintain normal behavior
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def validate_config(self) -> tuple[bool, list]:
        """
        Validates the configuration and returns (is_valid, list_of_errors).
        """
        errors = []
        
        # Check critical attributes
        critical_attrs = [
            'DATABASE_URL', 
            'MAIN_SYSTEM_LOCAL_MODEL', 
            'EMBEDDING_MODEL',
            'CURIOSITY_CHANCE',
            'REFLECTION_CHANCE',
            'LOOP_SLEEP_DURATION',
            'MAX_ITERATIONS'
        ]
        
        for attr in critical_attrs:
            if not hasattr(self, attr):
                errors.append(f"Missing critical configuration attribute: {attr}")
            elif getattr(self, attr) is None:
                errors.append(f"Critical configuration attribute {attr} is None")
        
        # Validate ranges for numeric values
        try:
            if not 0 <= getattr(self, 'CURIOSITY_CHANCE', -1) <= 1:
                errors.append("CURIOSITY_CHANCE must be between 0 and 1")
            if not 0 <= getattr(self, 'REFLECTION_CHANCE', -1) <= 1:
                errors.append("REFLECTION_CHANCE must be between 0 and 1")
            if getattr(self, 'LOOP_SLEEP_DURATION', -1) < 0:
                errors.append("LOOP_SLEEP_DURATION must be non-negative")
            if getattr(self, 'MAX_ITERATIONS', -1) <= 0:
                errors.append("MAX_ITERATIONS must be positive")
        except (TypeError, ValueError):
            errors.append("Some configuration values have incorrect types")
        
        # Validate MAIN_SYSTEM_LOCAL_MODEL structure
        main_model = getattr(self, 'MAIN_SYSTEM_LOCAL_MODEL', {})
        if not isinstance(main_model, dict):
            errors.append("MAIN_SYSTEM_LOCAL_MODEL must be a dictionary")
        else:
            required_model_keys = ['model_name', 'base_url', 'temperature']
            for key in required_model_keys:
                if key not in main_model:
                    errors.append(f"MAIN_SYSTEM_LOCAL_MODEL missing required key: {key}")
        
        return len(errors) == 0, errors

    def get_provider_model(self, provider_name: str, role: str = 'reasoning') -> dict:
        """Return a model configuration dict for the given provider and role.

        The selection is heuristic: it inspects the provider's "models" list
        and picks the first model whose name matches role-specific keywords.
        Falls back to the first listed model if nothing matches.
        The returned dict has keys: provider, model_name, base_url, temperature, timeout, keep_alive.
        """
        # Get the instance to access PROVIDERS_CONFIG
        instance = self if hasattr(self, 'PROVIDERS_CONFIG') else Config()
        prov = instance.PROVIDERS_CONFIG.get(
            provider_name) if instance.PROVIDERS_CONFIG else None
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