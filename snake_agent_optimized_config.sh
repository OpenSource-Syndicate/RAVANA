# Optimized Snake Agent Configuration
# These environment variables provide optimal settings for peak Snake Agent performance

# Threading and multiprocessing configuration
export SNAKE_MAX_THREADS=16
export SNAKE_MAX_PROCESSES=8
export SNAKE_ANALYSIS_THREADS=6

# Performance and monitoring settings
export SNAKE_PERF_MONITORING=True
export SNAKE_AUTO_RECOVERY=True
export SNAKE_ENHANCED_MODE=True

# Resource management settings
export SNAKE_TASK_TIMEOUT=900.0  # 15 minutes
export SNAKE_MAX_QUEUE_SIZE=5000  # Increased for better buffering
export SNAKE_HEARTBEAT_INTERVAL=2.0  # More frequent monitoring

# Logging settings
export SNAKE_LOG_RETENTION_DAYS=90

# VLTM (Very Long-Term Memory) settings
export SNAKE_VLTM_ENABLED=True
export SNAKE_VLTM_STORAGE_DIR=optimized_snake_vltm_storage

# File monitoring settings for better efficiency
export SNAKE_MONITOR_INTERVAL=0.5  # Faster monitoring

# Enable peek prioritizer for efficient file selection
export SNAKE_USE_PEEK_PRIORITIZER=True

echo "Snake Agent optimized configuration loaded"