import os

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

    # Autonomous Loop Settings
    CURIOSITY_CHANCE = float(os.environ.get("CURIOSITY_CHANCE", 0.3))
    REFLECTION_CHANCE = float(os.environ.get("REFLECTION_CHANCE", 0.1))
    LOOP_SLEEP_DURATION = int(os.environ.get("LOOP_SLEEP_DURATION", 10))
    ERROR_SLEEP_DURATION = int(os.environ.get("ERROR_SLEEP_DURATION", 60))
    MAX_EXPERIMENT_LOOPS = int(os.environ.get("MAX_EXPERIMENT_LOOPS", 10))
    MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", 10))
    RESEARCH_TASK_TIMEOUT = int(os.environ.get("RESEARCH_TASK_TIMEOUT", 600)) # 10 minutes

    # Emotional Intelligence Settings
    POSITIVE_MOODS = ['Confident', 'Curious', 'Reflective', 'Excited', 'Content']
    NEGATIVE_MOODS = ['Frustrated', 'Stuck', 'Low Energy', 'Bored']
    EMOTIONAL_PERSONA = "Optimistic"

    # Model Settings
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Background Task Intervals (in seconds)
    DATA_COLLECTION_INTERVAL = int(os.environ.get("DATA_COLLECTION_INTERVAL", 3600))
    EVENT_DETECTION_INTERVAL = int(os.environ.get("EVENT_DETECTION_INTERVAL", 600))
    KNOWLEDGE_COMPRESSION_INTERVAL = int(os.environ.get("KNOWLEDGE_COMPRESSION_INTERVAL", 3600)) 
    # Personality / Invention settings
    PERSONA_NAME = os.environ.get("PERSONA_NAME", "Ravana")
    PERSONA_ORIGIN = os.environ.get("PERSONA_ORIGIN", "Ancient Sri Lanka")
    PERSONA_CREATIVITY = float(os.environ.get("PERSONA_CREATIVITY", 0.7))
    INVENTION_INTERVAL = int(os.environ.get("INVENTION_INTERVAL", 7200))  # seconds between invention attempts

    # Graceful Shutdown Configuration
    SHUTDOWN_TIMEOUT = int(os.environ.get("SHUTDOWN_TIMEOUT", 30))  # seconds
    GRACEFUL_SHUTDOWN_ENABLED = bool(os.environ.get("GRACEFUL_SHUTDOWN_ENABLED", "True").lower() in ["true", "1", "yes"])
    STATE_PERSISTENCE_ENABLED = bool(os.environ.get("STATE_PERSISTENCE_ENABLED", "True").lower() in ["true", "1", "yes"])
    SHUTDOWN_STATE_FILE = os.environ.get("SHUTDOWN_STATE_FILE", "shutdown_state.json")
    FORCE_SHUTDOWN_AFTER = int(os.environ.get("FORCE_SHUTDOWN_AFTER", 60))  # seconds
    
    # Memory Service Shutdown Configuration
    MEMORY_SERVICE_SHUTDOWN_TIMEOUT = int(os.environ.get("MEMORY_SERVICE_SHUTDOWN_TIMEOUT", 15))
    POSTGRES_CONNECTION_TIMEOUT = int(os.environ.get("POSTGRES_CONNECTION_TIMEOUT", 10))
    CHROMADB_PERSIST_ON_SHUTDOWN = bool(os.environ.get("CHROMADB_PERSIST_ON_SHUTDOWN", "True").lower() in ["true", "1", "yes"])
    TEMP_FILE_CLEANUP_ENABLED = bool(os.environ.get("TEMP_FILE_CLEANUP_ENABLED", "True").lower() in ["true", "1", "yes"])
    
    # Resource Cleanup Configuration
    ACTION_CACHE_PERSIST = bool(os.environ.get("ACTION_CACHE_PERSIST", "True").lower() in ["true", "1", "yes"])
    RESOURCE_CLEANUP_TIMEOUT = int(os.environ.get("RESOURCE_CLEANUP_TIMEOUT", 10))
    DATABASE_CLEANUP_TIMEOUT = int(os.environ.get("DATABASE_CLEANUP_TIMEOUT", 15))
    
    # Snake Agent Configuration
    SNAKE_AGENT_ENABLED = bool(os.environ.get("SNAKE_AGENT_ENABLED", "True").lower() in ["true", "1", "yes"])
    SNAKE_AGENT_INTERVAL = int(os.environ.get("SNAKE_AGENT_INTERVAL", 300))  # 5 minutes default
    
    # Ollama Configuration for Snake Agent
    SNAKE_OLLAMA_BASE_URL = os.environ.get("SNAKE_OLLAMA_BASE_URL", "http://localhost:11434")
    SNAKE_OLLAMA_TIMEOUT = int(os.environ.get("SNAKE_OLLAMA_TIMEOUT", 120))  # seconds
    SNAKE_OLLAMA_KEEP_ALIVE = os.environ.get("SNAKE_OLLAMA_KEEP_ALIVE", "5m")
    
    # Dual LLM Models for Snake Agent (Ollama-based)
    SNAKE_CODING_MODEL = {
        "provider": "ollama",
        "model_name": os.environ.get("SNAKE_CODING_MODEL", "gpt-oss:20b"),
        "base_url": os.environ.get("SNAKE_OLLAMA_BASE_URL", "http://localhost:11434"),
        "temperature": float(os.environ.get("SNAKE_CODING_TEMPERATURE", "0.1")),
        "max_tokens": None if os.environ.get("SNAKE_CODING_MAX_TOKENS", "unlimited").lower() in ["unlimited", "none", "-1"] else int(os.environ.get("SNAKE_CODING_MAX_TOKENS", "4096")),
        "unlimited_mode": bool(os.environ.get("SNAKE_UNLIMITED_MODE", "True").lower() in ["true", "1", "yes"]),
        "chunk_size": int(os.environ.get("SNAKE_CHUNK_SIZE", "4096")),
        "timeout": int(os.environ.get("SNAKE_OLLAMA_TIMEOUT", 300)),  # Extended for longer responses
        "keep_alive": os.environ.get("SNAKE_OLLAMA_KEEP_ALIVE", "10m")  # Extended keep alive
    }
    
    SNAKE_REASONING_MODEL = {
        "provider": "ollama",
        "model_name": os.environ.get("SNAKE_REASONING_MODEL", "deepseek-r1:7b"),
        "base_url": os.environ.get("SNAKE_OLLAMA_BASE_URL", "http://localhost:11434"),
        "temperature": float(os.environ.get("SNAKE_REASONING_TEMPERATURE", "0.3")),
        "max_tokens": None if os.environ.get("SNAKE_REASONING_MAX_TOKENS", "unlimited").lower() in ["unlimited", "none", "-1"] else int(os.environ.get("SNAKE_REASONING_MAX_TOKENS", "2048")),
        "unlimited_mode": bool(os.environ.get("SNAKE_UNLIMITED_MODE", "True").lower() in ["true", "1", "yes"]),
        "chunk_size": int(os.environ.get("SNAKE_CHUNK_SIZE", "2048")),
        "timeout": int(os.environ.get("SNAKE_OLLAMA_TIMEOUT", 300)),  # Extended for longer responses
        "keep_alive": os.environ.get("SNAKE_OLLAMA_KEEP_ALIVE", "10m")  # Extended keep alive
    }
    
    # Alternative Model Options (user can override via environment variables)
    SNAKE_AVAILABLE_CODING_MODELS = [
        "deepseek-coder:6.7b",
        "deepseek-coder:1.3b",
        "codellama:7b", 
        "codellama:13b",
        "starcoder2:3b",
        "starcoder2:7b"
    ]
    
    SNAKE_AVAILABLE_REASONING_MODELS = [
        "llama3.1:8b",
        "llama3.1:70b",
        "qwen2.5:7b",
        "qwen2.5:14b",
        "mistral:7b",
        "gemma2:9b"
    ]
    
    # Snake Agent Safety Configuration
    SNAKE_SANDBOX_TIMEOUT = int(os.environ.get("SNAKE_SANDBOX_TIMEOUT", 60))  # seconds
    SNAKE_MAX_FILE_SIZE = int(os.environ.get("SNAKE_MAX_FILE_SIZE", 1048576))  # 1MB
    SNAKE_BLACKLIST_PATHS = os.environ.get("SNAKE_BLACKLIST_PATHS", "").split(",") if os.environ.get("SNAKE_BLACKLIST_PATHS") else []
    SNAKE_APPROVAL_REQUIRED = bool(os.environ.get("SNAKE_APPROVAL_REQUIRED", "True").lower() in ["true", "1", "yes"])
    
    # Communication Configuration
    SNAKE_COMM_CHANNEL = os.environ.get("SNAKE_COMM_CHANNEL", "memory_service")
    SNAKE_COMM_PRIORITY_THRESHOLD = float(os.environ.get("SNAKE_COMM_PRIORITY_THRESHOLD", "0.8"))
    
    # Snake Agent Graceful Shutdown Integration (extends existing shutdown config)
    SNAKE_SHUTDOWN_TIMEOUT = int(os.environ.get("SNAKE_SHUTDOWN_TIMEOUT", 30))  # seconds
    SNAKE_STATE_PERSISTENCE = bool(os.environ.get("SNAKE_STATE_PERSISTENCE", "True").lower() in ["true", "1", "yes"])
    
    # Enhanced Snake Agent Configuration
    SNAKE_ENHANCED_MODE = bool(os.environ.get("SNAKE_ENHANCED_MODE", "True").lower() in ["true", "1", "yes"])
    SNAKE_MAX_THREADS = int(os.environ.get("SNAKE_MAX_THREADS", "8"))
    SNAKE_MAX_PROCESSES = int(os.environ.get("SNAKE_MAX_PROCESSES", "4"))
    SNAKE_ANALYSIS_THREADS = int(os.environ.get("SNAKE_ANALYSIS_THREADS", "3"))
    SNAKE_MONITOR_INTERVAL = float(os.environ.get("SNAKE_MONITOR_INTERVAL", "2.0"))  # seconds
    SNAKE_PERF_MONITORING = bool(os.environ.get("SNAKE_PERF_MONITORING", "True").lower() in ["true", "1", "yes"])
    SNAKE_AUTO_RECOVERY = bool(os.environ.get("SNAKE_AUTO_RECOVERY", "True").lower() in ["true", "1", "yes"])
    SNAKE_LOG_RETENTION_DAYS = int(os.environ.get("SNAKE_LOG_RETENTION_DAYS", "30"))
    
    # Threading and Multiprocessing Limits
    SNAKE_TASK_TIMEOUT = float(os.environ.get("SNAKE_TASK_TIMEOUT", "300.0"))  # 5 minutes
    SNAKE_HEARTBEAT_INTERVAL = float(os.environ.get("SNAKE_HEARTBEAT_INTERVAL", "10.0"))  # seconds
    SNAKE_MAX_QUEUE_SIZE = int(os.environ.get("SNAKE_MAX_QUEUE_SIZE", "1000"))
    SNAKE_CLEANUP_INTERVAL = float(os.environ.get("SNAKE_CLEANUP_INTERVAL", "3600.0"))  # 1 hour

