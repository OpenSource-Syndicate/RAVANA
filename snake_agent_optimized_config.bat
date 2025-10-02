@echo off
REM Optimized Snake Agent Configuration for Windows
REM These environment variables provide optimal settings for peak Snake Agent performance

REM Threading and multiprocessing configuration
set SNAKE_MAX_THREADS=16
set SNAKE_MAX_PROCESSES=8
set SNAKE_ANALYSIS_THREADS=6

REM Performance and monitoring settings
set SNAKE_PERF_MONITORING=True
set SNAKE_AUTO_RECOVERY=True
set SNAKE_ENHANCED_MODE=True

REM Resource management settings
set SNAKE_TASK_TIMEOUT=900.0  REM 15 minutes
set SNAKE_MAX_QUEUE_SIZE=5000  REM Increased for better buffering
set SNAKE_HEARTBEAT_INTERVAL=2.0  REM More frequent monitoring

REM Logging settings
set SNAKE_LOG_RETENTION_DAYS=90

REM VLTM (Very Long-Term Memory) settings
set SNAKE_VLTM_ENABLED=True
set SNAKE_VLTM_STORAGE_DIR=optimized_snake_vltm_storage

REM File monitoring settings for better efficiency
set SNAKE_MONITOR_INTERVAL=0.5  REM Faster monitoring

REM Enable peek prioritizer for efficient file selection
set SNAKE_USE_PEEK_PRIORITIZER=True

echo Snake Agent optimized configuration loaded