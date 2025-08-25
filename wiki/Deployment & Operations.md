# Deployment & Operations



## Update Summary
**Changes Made**   
- Added documentation for singleton pattern implementation in ConversationalAI class
- Updated Bot Connectivity Verification section with instance tracking details
- Added new section for User Platform Preference Tracking
- Enhanced section sources to reflect new files and functionality
- Added references to DiscordBot and TelegramBot instance tracking mechanisms
- Updated startup command documentation to reflect singleton behavior

## Table of Contents
1. [Runtime Requirements](#runtime-requirements)
2. [Dependency Installation](#dependency-installation)
3. [Configuration Setup](#configuration-setup)
4. [Startup Commands](#startup-commands)
5. [Monitoring & Health Checks](#monitoring--health-checks)
6. [Logging Configuration](#logging-configuration)
7. [Backup & Recovery Procedures](#backup--recovery-procedures)
8. [Performance Tuning](#performance-tuning)
9. [Scaling Considerations](#scaling-considerations)
10. [Failure Recovery](#failure-recovery)
11. [Security Hardening](#security-hardening)
12. [Operational Runbooks](#operational-runbooks)
13. [Bot Connectivity Verification](#bot-connectivity-verification)
14. [User Platform Preference Tracking](#user-platform-preference-tracking)

## Runtime Requirements

The RAVANA system is a Python-based artificial general intelligence (AGI) framework designed for autonomous reasoning, memory consolidation, and multi-modal processing. It requires a modern computing environment with sufficient resources to support concurrent AI inference, database operations, and background services.

### CPU & Memory Requirements

- **Minimum**: 4-core CPU, 8GB RAM
- **Recommended**: 8-core CPU, 16GB+ RAM
- **High-performance**: 16-core CPU, 32GB+ RAM (for large-scale knowledge compression and physics simulations)

The system runs multiple concurrent processes including:
- Main AGI loop
- Memory server (separate process)
- Background data fetching and event detection
- Multi-modal processing (image/audio)

### GPU Requirements

- **Optional but recommended**: NVIDIA GPU with CUDA support (RTX 3060 or higher)
- Required for: 
  - Local LLM inference (if not using cloud APIs)
  - FAISS semantic search acceleration
  - Image/audio processing
- Minimum VRAM: 8GB
- Supported frameworks: CUDA, cuDNN

### Storage Requirements

- **System files**: ~500MB
- **Database**: Variable, scales with usage (initial: 100MB)
- **FAISS index**: ~100MB–1GB depending on knowledge base size
- **Temporary files**: Up to 2GB in `/tmp/agi_multimodal`
- **Recommended**: SSD storage for optimal database and index performance

**Section sources**
- [core/config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L1-L50)
- [services/knowledge_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\knowledge_service.py#L1-L30)
- [services/memory_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\memory_service.py#L1-L20)

## Dependency Installation

### Python Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip
```

### Core Dependencies (via pyproject.toml)

```bash
# Install all dependencies
pip install -e .

# Or install manually from requirements
pip install \
  sqlmodel \
  "sentence-transformers" \
  faiss-cpu \
  psutil \
  requests \
  python-dotenv \
  numpy \
  "openai" \
  "google-generativeai" \
  "pydantic>=2.0"
```

### Optional GPU-Accelerated Dependencies

```bash
# For GPU support (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

### External Service Dependencies

- **Google Gemini API**: Required for multi-modal processing
  - Set `GEMINI_API_KEY` in environment
- **OpenAI API**: Optional fallback for LLM calls
  - Set `OPENAI_API_KEY` in environment
- **PostgreSQL (optional)**: For production database
  - Default: SQLite (file-based)

**Section sources**
- [pyproject.toml](file://c:\Users\ASUS\Documents\GitHub\RAVANA\pyproject.toml#L1-L50)
- [services/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\multi_modal_service.py#L1-L20)
- [core/llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py#L1-L30)

## Configuration Setup

### Environment Variables

Create `.env` file in project root:

```env
# API Keys
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here

# Database
DATABASE_URL=sqlite:///./ravana.db
# For PostgreSQL: DATABASE_URL=postgresql://user:pass@localhost/ravana

# Memory Service
MEMORY_SERVER_HOST=localhost
MEMORY_SERVER_PORT=8001
MEMORY_SERVICE_SHUTDOWN_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/ravana.log

# Data Feeds
FEED_URLS=https://example.com/feed1.xml,https://example.com/feed2.json
```

### Database Initialization

```python
from database.engine import create_db_and_tables

create_db_and_tables()
```

This creates all required tables based on `database/models.py`.

### Configuration File (config.json)

```json
{
  "debug_mode": false,
  "auto_save_interval": 300,
  "max_memory_entries": 10000,
  "consolidation_threshold": 100,
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_provider": "gemini"
}
```

**Section sources**
- [core/config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py#L1-L50)
- [database/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\database\models.py#L1-L56)
- [database/engine.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\database\engine.py#L1-L20)

## Startup Commands

### Autonomous Mode (Main AGI Loop)

```bash
python main.py
```

This starts the full AGI system with:
- Continuous data ingestion
- Decision engine
- Memory management
- Autonomous action execution

### Physics Testing Mode

```bash
# Run all physics experiments
python run_physics_tests.py

# Interactive physics CLI
python physics_cli.py
```

These modes are used for validating physical reasoning capabilities and running controlled experiments.

### Single Task Mode

```bash
# Execute specific action via CLI
python main.py --action coding --params '{"task": "implement quicksort"}'

# Process multi-modal directory
python main.py --mode multimodal --dir ./input_files
```

### Service-Specific Startup

```bash
# Start only memory service
python -m modules.episodic_memory.main

# Start data fetching service
python -c "from services.data_service import DataService; from database.engine import engine; ds = DataService(engine, ['https://rss.example.com']); ds.fetch_and_save_articles()"
```

### Conversational AI Module Startup

```bash
# Start conversational AI module in standalone mode
python launch_conversational_ai.py

# Verify bot connectivity before starting
python launch_conversational_ai.py --verify-bots
```

The `--verify-bots` flag checks the connectivity of configured bot platforms (Discord and Telegram) and exits with appropriate status code. The ConversationalAI class implements a singleton pattern, ensuring only one instance can be created and initialized in the system.

**Section sources**
- [main.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\main.py#L1-L100)
- [run_physics_tests.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\run_physics_tests.py#L1-L20)
- [physics_cli.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\physics_cli.py#L1-L15)
- [launch_conversational_ai.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\launch_conversational_ai.py#L1-L236) - *Updated in recent commit*
- [modules/conversational_ai/main.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\main.py#L1-L483) - *Modified in recent commit*

## Monitoring & Health Checks

### Service Health Endpoints

| Service | Health Endpoint | Process Detection |
|-------|----------------|-------------------|
| Memory Server | `http://localhost:8001/health` | Checks for `memory.py` process |
| Main System | None (internal) | PID file or process name |

### Health Check Implementation

```python
# Example health check for memory service
async def check_memory_service_health():
    import requests
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        return response.status_code == 200
    except:
        return False
```

### Monitoring Metrics

The system exposes the following operational metrics:

- **MemoryService.get_memory_statistics()**
  - `status`: operational/error
  - `total_memories`: count of stored memories
  - `last_consolidation`: timestamp
  - `memory_server_status`: running/not running

- **Database health**
  - Connection pool status
  - Table row counts
  - Index integrity

### Prometheus Integration (Recommended)

Add metrics exporter to expose for Prometheus:

```python
from prometheus_client import start_http_server, Counter, Gauge

# Example metric
MEMORY_COUNT = Gauge('ravana_memory_count', 'Number of stored memories')
```

**Section sources**
- [services/memory_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\memory_service.py#L150-L167)
- [services/data_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\data_service.py#L1-L20)

## Logging Configuration

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: Normal operation events
- **WARNING**: Potential issues
- **ERROR**: Recoverable errors
- **CRITICAL**: System failures

### Log Output

Logs are written to:
- **Console**: Real-time monitoring
- **File**: `logs/ravana.log` (rotated daily)
- **Structured format**: JSON when `LOG_JSON=true`

### Log Categories

| Logger | Purpose |
|-------|--------|
| `ravana` | Main system events |
| `MemoryService` | Memory operations |
| `DataService` | Data fetching and storage |
| `KnowledgeService` | Knowledge compression and retrieval |
| `MultiModalService` | Image/audio processing |

### Example Log Entry

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "MemoryService",
  "event": "Memory server detected, attempting graceful shutdown...",
  "context": {"pid": 1234}
}
```

**Section sources**
- [services/memory_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\memory_service.py#L1-L10)
- [services/knowledge_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\knowledge_service.py#L1-L10)

## Backup & Recovery Procedures

### Database Backup

```bash
# SQLite backup
cp ravana.db ravana.db.backup.$(date +%Y%m%d_%H%M%S)

# With compression
tar -czf ravana_db_$(date +%Y%m%d).tar.gz ravana.db
```

For production PostgreSQL:

```bash
pg_dump -U ravana_user -h localhost ravana_db > ravana_backup.sql
```

### Knowledge Index Backup

```bash
# FAISS index and ID map
cp knowledge_index.faiss knowledge_index.faiss.backup
cp knowledge_id_map.pkl knowledge_id_map.pkl.backup

# Compressed backup
tar -czf knowledge_backup_$(date +%Y%m%d).tar.gz knowledge_index.faiss knowledge_id_map.pkl
```

### Automated Backup Script

```bash
#!/bin/bash
BACKUP_DIR="/backups/ravana/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

cp ravana.db $BACKUP_DIR/
cp knowledge_index.faiss $BACKUP_DIR/
cp knowledge_id_map.pkl $BACKUP_DIR/

# Keep last 7 days
find /backups/ravana -type d -name "202*" | sort | head -n -7 | xargs rm -rf
```

### Recovery Procedure

1. Stop RAVANA services
2. Replace database and index files
3. Restart services
4. Verify integrity via health checks

**Section sources**
- [database/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\database\models.py#L1-L56)
- [services/knowledge_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\knowledge_service.py#L1-L30)
- [services/memory_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\memory_service.py#L1-L20)

## Performance Tuning

### Database Optimization

- **Indexing**: Ensure proper indexes on timestamp fields
- **Connection pooling**: Use SQLAlchemy pool settings for PostgreSQL
- **Vacuuming**: Regular `VACUUM` for SQLite databases

### Knowledge Service Tuning

- **FAISS Index Type**: Use `IndexIVFFlat` for large datasets (>100K entries)
- **Embedding Model**: Switch to larger models (`all-mpnet-base-v2`) for better accuracy
- **Batch Processing**: Process summaries in batches during compression

### Memory Management

- **Consolidation Frequency**: Adjust based on activity level
- **Memory Pruning**: Implement TTL-based cleanup for old memories
- **Vector Store**: Consider switching to dedicated vector database (Pinecone, Weaviate) at scale

### Caching Strategy

- **Query Results**: Cache frequent knowledge queries
- **LLM Responses**: Cache deterministic LLM calls
- **Embeddings**: Cache frequently used text embeddings

**Section sources**
- [services/knowledge_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\knowledge_service.py#L50-L100)
- [database/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\database\models.py#L1-L56)

## Scaling Considerations

### Horizontal Scaling

The system can be scaled by separating services:

| Component | Scalability |
|---------|-------------|
| Main AGI Loop | Single instance (stateful) |
| Memory Service | Can run as separate microservice |
| Data Service | Can be distributed |
| Knowledge Service | Can be containerized |
| Multi-Modal Service | Horizontally scalable |

### Containerization (Docker)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -e .

CMD ["python", "main.py"]
```

### Kubernetes Deployment (Recommended for Production)

Deploy components as separate pods with:
- Resource limits/requests
- Liveness and readiness probes
- Persistent volumes for database and indexes
- Secret management for API keys

### Load Distribution

- **Data ingestion**: Distribute feed processing
- **Multi-modal processing**: Queue-based worker pool
- **Knowledge compression**: Scheduled batch jobs

**Section sources**
- [services/data_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\data_service.py#L1-L20)
- [services/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\multi_modal_service.py#L1-L30)

## Failure Recovery

### Process Crash Recovery

- **MemoryService**: Automatically detects and shuts down orphaned memory server processes
- **Main System**: Use process manager (systemd, PM2) for auto-restart
- **Data Service**: Idempotent operations prevent duplication

### Data Corruption Handling

- **Database**: Use transactions and regular integrity checks
- **FAISS Index**: Recreate from database if corrupted
- **Backups**: Restore from last known good state

### Graceful Shutdown

```python
# Uses Shutdownable interface
await memory_service.prepare_shutdown()
await memory_service.shutdown(timeout=30)
```

Ensures:
- Memory server shutdown
- Database transaction completion
- Log flushing
- State persistence

### Disaster Recovery Plan

1. **Immediate**: Switch to backup instance
2. **Short-term**: Restore from latest backup
3. **Long-term**: Rebuild knowledge index from logs

**Section sources**
- [services/memory_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\memory_service.py#L100-L150)
- [core/shutdown_coordinator.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\shutdown_coordinator.py#L1-L20)

## Security Hardening

### API Key Protection

- Store keys in environment variables or secret manager
- Never commit to version control
- Use short-lived keys when possible

### Input Validation

- Validate all external inputs
- Sanitize file paths in multi-modal service
- Limit upload sizes

### File System Security

- Restrict permissions on configuration files
- Secure temporary directory (`/tmp/agi_multimodal`)
- Regular cleanup of temp files via `cleanup_temp_files()`

### Network Security

- Bind services to localhost when possible
- Use firewall rules to restrict access
- Enable TLS for external endpoints

### Audit Logging

- Log all security-relevant events
- Monitor for suspicious activity
- Regular log reviews

**Section sources**
- [services/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\multi_modal_service.py#L1-L50)
- [core/llm.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\llm.py#L1-L20)

## Operational Runbooks

### Normal Restart Procedure

1. Send SIGTERM to main process
2. Wait for graceful shutdown (30s timeout)
3. Verify all services terminated
4. Start process again

```bash
pkill -f main.py
sleep 5
python main.py
```

### Emergency Stop

```bash
pkill -9 -f memory.py
pkill -9 -f main.py
```

Use only if process is unresponsive.

### Update Procedure

1. Pull latest code
2. Backup database and indexes
3. Install new dependencies
4. Test in staging environment
5. Deploy with rolling restart

### Incident Response

| Incident | Response |
|--------|---------|
| High CPU usage | Check for runaway processes, restart memory service |
| Database corruption | Restore from backup, rebuild indexes |
| API key exhaustion | Rotate keys, check for leaks |
| Memory leak | Restart service, analyze logs |
| Failed health check | Investigate logs, restore from backup if needed |

### Routine Maintenance

- **Daily**: Check logs, verify backups
- **Weekly**: Database optimization, cleanup temp files
- **Monthly**: Security audit, dependency updates

**Section sources**
- [services/memory_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\memory_service.py#L100-L150)
- [services/data_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\data_service.py#L1-L20)
- [main.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\main.py#L1-L20)

## Bot Connectivity Verification

### Purpose

The bot connectivity verification feature allows administrators to test the connectivity of configured bot platforms (Discord and Telegram) before starting the full conversational AI system. This helps identify configuration issues such as invalid tokens or network connectivity problems.

### Verification Command

```bash
python launch_conversational_ai.py --verify-bots
```

This command:
- Loads the conversational AI configuration
- Tests connectivity for each enabled bot platform
- Returns exit code 0 if all enabled bots are connected successfully
- Returns exit code 1 if any enabled bot fails to connect
- Outputs detailed results to stdout

### Expected Output

```
=== Bot Verification Results ===
Discord: CONNECTED
Telegram: FAILED
  Message: Telegram bot did not connect in time
```

### Configuration Requirements

Bot connectivity verification relies on the configuration in `modules/conversational_ai/config.json`:

```json
{
  "discord_token": "your_discord_bot_token",
  "telegram_token": "your_telegram_bot_token",
  "platforms": {
    "discord": {
      "enabled": true,
      "command_prefix": "!"
    },
    "telegram": {
      "enabled": true,
      "command_prefix": "/"
    }
  }
}
```

### Integration with Deployment Workflows

The verification command can be integrated into deployment pipelines:

```bash
# In CI/CD pipeline
python launch_conversational_ai.py --verify-bots
if [ $? -ne 0 ]; then
    echo "Bot verification failed, aborting deployment"
    exit 1
fi
# Proceed with deployment
```

### Troubleshooting

Common issues and solutions:

- **Token not found**: Verify that tokens are correctly specified in `config.json`
- **Network connectivity**: Ensure the server has outbound internet access
- **Firewall restrictions**: Check if firewall rules block connections to Discord/Telegram APIs
- **Invalid tokens**: Regenerate bot tokens from the respective developer portals

### Instance Tracking Implementation

The DiscordBot and TelegramBot classes implement class-level tracking to prevent multiple instances from running concurrently. Each bot class has `_instance_started` and `_active_instance` class variables that track the global state of bot instances. When a bot is started, it checks these class variables to ensure no other instance is already running. This prevents conflicts and resource contention in production environments.

**Section sources**
- [launch_conversational_ai.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\launch_conversational_ai.py#L1-L236) - *Updated in recent commit*
- [modules/conversational_ai/main.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\main.py#L1-L483) - *Modified in recent commit*
- [modules/conversational_ai/config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\config.json#L1-L28) - *Configuration for conversational AI module*
- [modules/conversational_ai/bots/discord_bot.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\bots\discord_bot.py#L1-L226) - *Added instance tracking in recent commit*
- [modules/conversational_ai/bots/telegram_bot.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\bots\telegram_bot.py#L1-L228) - *Added instance tracking in recent commit*

## User Platform Preference Tracking

### Purpose

The system tracks user platform preferences to enable intelligent message routing and personalized interactions. When a user interacts with the conversational AI through a specific platform (Discord or Telegram), this preference is recorded and used for future communications.

### Implementation Details

The user platform preference tracking is implemented through the following components:

1. **UserPlatformProfile**: A dataclass that stores user platform information including:
   - `user_id`: Unique identifier of the user
   - `last_platform`: The last platform used (discord/telegram)
   - `platform_user_id`: Platform-specific user identifier
   - `preferences`: User preferences dictionary
   - `last_interaction`: Timestamp of last interaction

2. **UserProfileManager**: Manages the storage and retrieval of user profiles, persisting them across sessions.

3. **_track_user_platform method**: Automatically called whenever a user sends a message, updating their platform preference.

### Automatic Tracking

The system automatically tracks platform usage through the `process_user_message` method in the ConversationalAI class:

```python
def _track_user_platform(self, user_id: str, platform: str):
    """Track the user's platform preference."""
    try:
        profile = UserPlatformProfile(
            user_id=user_id,
            last_platform=platform,
            platform_user_id=user_id,
            preferences={},
            last_interaction=datetime.now()
        )
        self.user_profile_manager.set_user_platform_profile(user_id, profile)
    except Exception as e:
        logger.error(f"Error tracking user platform for user {user_id}: {e}")
```

### Intelligent Message Routing

When sending messages to users, the system uses their stored platform preference:

```python
async def send_message_to_user(self, user_id: str, message: str, platform: str = None):
    """Send a message to a user through the appropriate platform."""
    try:
        if not platform:
            # Try to get the user's last used platform from their profile
            profile = self.user_profile_manager.get_user_platform_profile(user_id)
            if profile:
                platform = profile.last_platform
        # Send message through the determined platform
        # ...
    except Exception as e:
        logger.error(f"Error sending message to user {user_id}: {e}")
```

### Configuration

No additional configuration is required for user platform preference tracking. The feature is enabled by default and works automatically with the conversational AI module.

### Use Cases

1. **Follow-up messages**: The system can send follow-up messages through the user's preferred platform
2. **Task notifications**: When a task is completed, notifications are sent through the user's last-used platform
3. **Cross-platform continuity**: Users can switch between platforms while maintaining their conversation context

**Section sources**
- [modules/conversational_ai/main.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\main.py#L402-L435) - *User platform tracking implementation*
- [modules/conversational_ai/communication/data_models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\communication\data_models.py#L57-L76) - *UserPlatformProfile data model*
- [modules/conversational_ai/main.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\main.py#L433-L461) - *Intelligent message routing*

**Referenced Files in This Document**   
- [main.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\main.py) - *Updated in recent commit*
- [launch_conversational_ai.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\launch_conversational_ai.py) - *Updated in recent commit*
- [core/config.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\config.py)
- [core/system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py)
- [database/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\database\models.py)
- [services/data_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\data_service.py)
- [services/knowledge_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\knowledge_service.py)
- [services/memory_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\memory_service.py)
- [services/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\multi_modal_service.py)
- [physics_cli.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\physics_cli.py)
- [run_physics_tests.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\run_physics_tests.py)
- [pyproject.toml](file://c:\Users\ASUS\Documents\GitHub\RAVANA\pyproject.toml)
- [modules/conversational_ai/main.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\main.py) - *Modified in recent commit*
- [modules/conversational_ai/config.json](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\config.json) - *Configuration for conversational AI module*
- [modules/conversational_ai/bots/discord_bot.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\bots\discord_bot.py) - *Added instance tracking in recent commit*
- [modules/conversational_ai/bots/telegram_bot.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\bots\telegram_bot.py) - *Added instance tracking in recent commit*
- [modules/conversational_ai/communication/data_models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\conversational_ai\communication\data_models.py) - *User platform tracking models*