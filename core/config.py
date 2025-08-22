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

