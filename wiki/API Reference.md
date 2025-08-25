# API Reference



## Update Summary
**Changes Made**   
- Updated Episodic Memory Client section with new methods for audio/image upload, advanced search, and batch processing
- Added new section for Multi-modal Memory Service with comprehensive API documentation
- Added new section for Memory Models with detailed data structure documentation
- Added new sections for PostgreSQL Store and Embedding Service implementations
- Updated Table of Contents to reflect new sections and organization
- Enhanced source tracking with specific file references and annotations

## Table of Contents
1. [AGISystem Class](#agisystem-class)
2. [Action Abstract Base Class](#action-abstract-base-class)
3. [Service APIs](#service-apis)
   - [Data Service](#data-service)
   - [Knowledge Service](#knowledge-service)
   - [Memory Service](#memory-service)
   - [Multi-modal Service](#multi-modal-service)
4. [Module-specific APIs](#module-specific-apis)
   - [Episodic Memory Client](#episodic-memory-client)
   - [Reflection System](#reflection-system)
5. [Multi-modal Memory Service](#multi-modal-memory-service)
6. [Memory Models](#memory-models)
7. [PostgreSQL Store](#postgresql-store)
8. [Embedding Service](#embedding-service)
9. [Initialization and Lifecycle Management](#initialization-and-lifecycle-management)
10. [Thread Safety and Async Usage](#thread-safety-and-async-usage)
11. [Error Handling](#error-handling)
12. [Usage Examples](#usage-examples)

## AGISystem Class

The `AGISystem` class is the central orchestrator of the Ravana AGI system, managing the integration of various modules, services, and decision-making processes. It implements an autonomous loop that continuously processes situations, makes decisions, executes actions, and reflects on outcomes.

### Methods

#### `__init__(engine)`
Initializes the AGI system with core components and services.

**:Parameters**
- `engine`: Database engine instance for persistence operations

**:Attributes**
- `data_service`: DataService instance for data operations
- `knowledge_service`: KnowledgeService instance for knowledge management
- `memory_service`: MemoryService instance for memory operations
- `action_manager`: EnhancedActionManager for action execution
- `shared_state`: SharedState object for cross-component state
- `personality`: Personality instance influencing behavior

**:Exceptions**
- None explicitly raised, but depends on underlying service initialization

**:Usage**
```python
from sqlalchemy import create_engine
from core.system import AGISystem

engine = create_engine("sqlite:///ravana.db")
agi = AGISystem(engine)
```

**Section sources**
- [core/system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py#L1-L625)

#### `run_autonomous_loop()`
Starts the main autonomous loop of the AGI system, which continuously processes iterations.

**:Returns**
- None (runs indefinitely until stopped)

**:Exceptions**
- Logs critical errors but continues execution with extended sleep intervals

**:Usage**
```python
import asyncio

async def main():
    await agi.run_autonomous_loop()

asyncio.run(main())
```

**Section sources**
- [core/system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py#L520-L550)

#### `run_single_task(prompt: str)`
Executes a single task specified by a prompt, running multiple iterations if needed.

**:Parameters**
- `prompt`: String describing the task to be performed

**:Returns**
- None (modifies internal state and executes actions)

**:Usage**
```python
await agi.run_single_task("Research quantum computing advancements")
```

**Section sources**
- [core/system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py#L552-L585)

#### `stop()`
Gracefully stops the AGI system and all background tasks.

**:Returns**
- None

**:Usage**
```python
await agi.stop()
```

**Section sources**
- [core/system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py#L100-L120)

#### `get_recent_events(time_limit_seconds: int = 3600)`
Retrieves recent events from the database within a specified time window.

**:Parameters**
- `time_limit_seconds`: Number of seconds in the past to include (default: 3600)

**:Returns**
- List of Event objects from the database

**:Usage**
```python
events = await agi.get_recent_events(7200)  # Get events from last 2 hours
```

**Section sources**
- [core/system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py#L490-L518)

## Action Abstract Base Class

The `Action` class is an abstract base class that defines the interface for all actions that the AGI system can perform. All concrete actions must inherit from this class and implement its abstract methods.

### Methods

#### `name` (property)
Returns the name of the action.

**:Returns**
- String representing the action name

**Section sources**
- [core/actions/action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L15-L18)

#### `description` (property)
Returns a description of what the action does.

**:Returns**
- String describing the action's purpose

**Section sources**
- [core/actions/action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L20-L23)

#### `parameters` (property)
Returns a list of parameters that the action accepts.

**:Returns**
- List of dictionaries, each containing parameter metadata (name, type, required, etc.)

**:Example Return**
```python
[
    {
        "name": "query",
        "type": "string",
        "required": True,
        "description": "Search query string"
    }
]
```

**Section sources**
- [core/actions/action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L25-L28)

#### `execute(**kwargs: Any)`
Abstract method that executes the action with the given parameters.

**:Parameters**
- `**kwargs`: Arbitrary keyword arguments matching the action's parameters

**:Returns**
- Any: Result of the action execution

**:Exceptions**
- Must be implemented by subclasses

**Section sources**
- [core/actions/action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L30-L33)

#### `validate_params(params: Dict[str, Any])`
Validates the given parameters against the action's defined parameters.

**:Parameters**
- `params`: Dictionary of parameter names and values

**:Exceptions**
- `InvalidActionParams`: Raised when required parameters are missing or unexpected parameters are provided

**:Usage**
```python
try:
    action.validate_params({"query": "AI research"})
    # Parameters are valid
except InvalidActionParams as e:
    print(f"Invalid parameters: {e}")
```

**Section sources**
- [core/actions/action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L35-L52)

#### `to_dict()`
Returns a dictionary representation of the action.

**:Returns**
- Dictionary containing name, description, and parameters

**Section sources**
- [core/actions/action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L54-L60)

#### `to_json()`
Returns a JSON string representing the action's schema.

**:Returns**
- String containing JSON-formatted action schema

**Section sources**
- [core/actions/action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py#L62-L66)

## Service APIs

### Data Service

The `DataService` class handles data persistence operations, including articles, events, and logging.

#### `__init__(engine, feed_urls, embedding_model=None, sentiment_classifier=None)`
Initializes the data service with database connection and processing models.

**:Parameters**
- `engine`: Database engine instance
- `feed_urls`: List of RSS feed URLs to monitor
- `embedding_model`: Optional sentence transformer model for embeddings
- `sentiment_classifier`: Optional pipeline for sentiment analysis

**Section sources**
- [services/data_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\data_service.py#L8-L15)

#### `fetch_and_save_articles()`
Fetches articles from configured RSS feeds and saves new ones to the database.

**:Returns**
- Integer: Number of new articles saved

**:Usage**
```python
num_saved = await asyncio.to_thread(data_service.fetch_and_save_articles)
```

**Section sources**
- [services/data_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\data_service.py#L17-L38)

#### `detect_and_save_events()`
Analyzes recent articles to detect and save significant events.

**:Returns**
- Integer: Number of events detected and saved

**:Usage**
```python
num_events = await asyncio.to_thread(data_service.detect_and_save_events)
```

**Section sources**
- [services/data_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\data_service.py#L40-L63)

#### `save_action_log(action_name, params, status, result)`
Persists a record of an executed action to the database.

**:Parameters**
- `action_name`: String name of the action
- `params`: Dictionary of action parameters
- `status`: String status ("success" or "error")
- `result`: Any result data from action execution

**Section sources**
- [services/data_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\data_service.py#L65-L77)

#### `save_mood_log(mood_vector)`
Saves the current mood vector to the database.

**:Parameters**
- `mood_vector`: Dictionary representing the current emotional state

**Section sources**
- [services/data_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\data_service.py#L79-L87)

#### `save_situation_log(situation)`
Saves a generated situation and returns its database ID.

**:Parameters**
- `situation`: Dictionary containing situation details

**:Returns**
- Integer: Database ID of the saved situation

**Section sources**
- [services/data_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\data_service.py#L89-L104)

#### `save_decision_log(situation_id, raw_response)`
Saves a decision made by the AGI to the database.

**:Parameters**
- `situation_id`: Integer ID of the associated situation
- `raw_response`: String containing the raw LLM response

**Section sources**
- [services/data_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\data_service.py#L106-L117)

#### `save_experiment_log(hypothesis, *args)`
Saves experiment results to the database with flexible calling conventions.

**:Parameters**
- `hypothesis`: String describing the experiment hypothesis
- `*args`: Either a single dict of results, or three arguments (test_plan, final_verdict, execution_result)

**:Exceptions**
- `TypeError`: Raised when arguments don't match expected patterns

**:Usage**
```python
# Style 1: Dictionary of results
data_service.save_experiment_log("AI creativity improves with reflection", {"findings": "Positive correlation observed"})

# Style 2: Individual components
data_service.save_experiment_log(
    "Memory consolidation improves recall",
    "Tested with 100 memory queries",
    "Supported",
    {"accuracy": 0.92}
)
```

**Section sources**
- [services/data_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\data_service.py#L119-L156)

### Knowledge Service

The `KnowledgeService` manages the storage, retrieval, and compression of knowledge.

#### `__init__(engine, embedding_model=None)`
Initializes the knowledge service with database connection and embedding model.

**:Parameters**
- `engine`: Database engine instance
- `embedding_model`: Optional SentenceTransformer instance (defaults to 'all-MiniLM-L6-v2')

**:Notes**
- Automatically initializes FAISS index for semantic search if available
- Loads existing index from disk or creates new one

**Section sources**
- [services/knowledge_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\knowledge_service.py#L17-L25)

#### `add_knowledge(content, source="unknown", category="misc")`
Adds new knowledge by summarizing content and saving it with metadata.

**:Parameters**
- `content`: String content to be added
- `source`: String source identifier (default: "unknown")
- `category`: String category (default: "misc")

**:Returns**
- Dictionary containing:
  - `timestamp`: ISO format timestamp
  - `summary`: Generated summary text
  - `source`: Source identifier
  - `category`: Category
  - `duplicate`: Boolean indicating if content already existed
  - `id`: Database ID (if new)

**:Exceptions**
- `ValueError`: When no content is provided
- Logs and re-raises any other exceptions

**:Usage**
```python
result = await asyncio.to_thread(
    knowledge_service.add_knowledge,
    "Recent advances in quantum computing have enabled...",
    source="research_paper",
    category="science"
)
```

**Section sources**
- [services/knowledge_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\knowledge_service.py#L85-L160)

#### `get_knowledge_by_category(category, limit=10)`
Retrieves knowledge entries by category.

**:Parameters**
- `category`: String category to filter by
- `limit`: Maximum number of results to return (default: 10)

**:Returns**
- List of dictionaries containing knowledge entry details

**Section sources**
- [services/knowledge_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\knowledge_service.py#L162-L178)

#### `get_recent_knowledge(hours=24, limit=20)`
Retrieves recently added knowledge entries.

**:Parameters**
- `hours`: Number of hours in the past to include (default: 24)
- `limit`: Maximum number of results (default: 20)

**:Returns**
- List of knowledge entry dictionaries

**Section sources**
- [services/knowledge_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\knowledge_service.py#L180-L198)

#### `search_knowledge(query, limit=10)`
Performs text search in knowledge summaries.

**:Parameters**
- `query`: Search query string
- `limit`: Maximum results to return (default: 10)

**:Returns**
- List of dictionaries with knowledge entries and relevance scores

**:Notes**
- Uses simple LIKE search; could be enhanced with full-text search
- Includes relevance_score based on keyword matching

**Section sources**
- [services/knowledge_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\knowledge_service.py#L200-L220)

#### `compress_and_save_knowledge()`
Compresses recent knowledge into a summary and saves it.

**:Returns**
- Dictionary containing the generated summary

**:Exceptions**
- Logs and re-raises any exceptions

**:Usage**
```python
summary = await asyncio.to_thread(knowledge_service.compress_and_save_knowledge)
```

**Section sources**
- [services/knowledge_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\knowledge_service.py#L222-L256)

### Memory Service

The `MemoryService` provides an interface to the episodic memory system.

#### `get_relevant_memories(query_text)`
Retrieves memories relevant to a query.

**:Parameters**
- `query_text`: String query to find relevant memories

**:Returns**
- Awaitable that resolves to a response object with relevant memories

**:Usage**
```python
response = await memory_service.get_relevant_memories("vacation plans")
for memory in response.relevant_memories:
    print(f"{memory.text} (similarity: {memory.similarity})")
```

**Section sources**
- [services/memory_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\memory_service.py#L7-L10)

#### `save_memories(memories)`
Saves a list of memories to persistent storage.

**:Parameters**
- `memories`: List of memory strings or objects

**:Returns**
- Awaitable (completes when save operation finishes)

**:Usage**
```python
await memory_service.save_memories(["I planned a trip to Hawaii", "I enjoy hiking"])
```

**Section sources**
- [services/memory_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\memory_service.py#L12-L14)

#### `extract_memories(user_input, ai_output)`
Extracts memories from user-AI interaction.

**:Parameters**
- `user_input`: String containing user message
- `ai_output`: String containing AI response

**:Returns**
- Awaitable resolving to an object with extracted memories

**:Usage**
```python
result = await memory_service.extract_memories(
    "I'm planning a vacation to Hawaii", 
    "That sounds wonderful!"
)
```

**Section sources**
- [services/memory_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\memory_service.py#L16-L18)

#### `consolidate_memories()`
Performs memory consolidation to optimize retrieval.

**:Returns**
- Awaitable resolving to consolidation results

**:Usage**
```python
result = await memory_service.consolidate_memories()
```

**Section sources**
- [services/memory_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\memory_service.py#L20-L24)

### Multi-modal Service

The `MultiModalService` handles processing of images, audio, and cross-modal analysis.

#### `__init__()`
Initializes the multi-modal service with supported formats and temporary directory.

**:Notes**
- Creates temporary directory for processing files
- Defines supported image and audio formats

**Section sources**
- [services/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\multi_modal_service.py#L25-L32)

#### `process_image(image_path, prompt="Analyze this image in detail")`
Processes an image file and returns detailed analysis.

**:Parameters**
- `image_path`: String path to the image file
- `prompt`: Optional custom prompt for analysis (default: "Analyze this image in detail")

**:Returns**
- Dictionary containing:
  - `type`: "image"
  - `path`: Original path
  - `format`: File extension
  - `size_bytes`: File size
  - `description`: AI-generated description
  - `analysis_prompt`: Prompt used
  - `success`: Boolean indicating success
  - `error`: Error message if unsuccessful

**:Exceptions**
- `FileNotFoundError`: When image file doesn't exist
- `ValueError`: When file format is unsupported

**:Usage**
```python
result = await multi_modal_service.process_image("/path/to/photo.jpg")
if result["success"]:
    print(result["description"])
```

**Section sources**
- [services/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\multi_modal_service.py#L34-L98)

#### `process_audio(audio_path, prompt="Describe and analyze this audio")`
Processes an audio file and returns analysis.

**:Parameters**
- `audio_path`: String path to the audio file
- `prompt`: Optional custom prompt for analysis (default: "Describe and analyze this audio")

**:Returns**
- Dictionary with similar structure to process_image result

**:Exceptions**
- `FileNotFoundError`: When audio file doesn't exist
- `ValueError`: When file format is unsupported

**Section sources**
- [services/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\multi_modal_service.py#L100-L158)

#### `cross_modal_analysis(content_list, analysis_prompt=None)`
Performs analysis across multiple content types.

**:Parameters**
- `content_list`: List of processed content objects (from process_image/process_audio)
- `analysis_prompt`: Optional custom analysis prompt

**:Returns**
- Dictionary containing cross-modal analysis

**:Usage**
```python
analysis = await multi_modal_service.cross_modal_analysis([
    image_result, 
    audio_result
])
```

**Section sources**
- [services/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\multi_modal_service.py#L160-L230)

#### `generate_content_summary(processed_content)`
Generates a comprehensive summary of multi-modal content.

**:Parameters**
- `processed_content`: List of processed content results

**:Returns**
- String summary of all content

**Section sources**
- [services/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\multi_modal_service.py#L232-L290)

#### `process_directory(directory_path, recursive=False)`
Processes all supported files in a directory.

**:Parameters**
- `directory_path`: Path to directory to process
- `recursive`: Whether to include subdirectories (default: False)

**:Returns**
- List of processing results for each file

**Section sources**
- [services/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\multi_modal_service.py#L292-L345)

#### `cleanup_temp_files(max_age_hours=24)`
Cleans up temporary files older than specified age.

**:Parameters**
- `max_age_hours`: Maximum age in hours (default: 24)

**:Notes**
- Runs synchronously (not async)
- Used for maintenance

**Section sources**
- [services/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\multi_modal_service.py#L347-L350)

## Module-specific APIs

### Episodic Memory Client

The episodic memory client provides direct access to the memory database API.

#### `extract_memories(user_input, ai_output)`
Calls the /extract_memories/ endpoint to extract memories from conversation.

**:Parameters**
- `user_input`: String user message
- `ai_output`: String AI response

**:Returns**
- Dictionary with extracted memories or None on failure

**:Usage**
```python
result = extract_memories("I love hiking in the mountains", "That sounds invigorating!")
if result and 'memories' in result:
    print(f"Extracted: {result['memories']}")
```

**Section sources**
- [modules/episodic_memory/client.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\client.py#L45-L50)

#### `save_memories(memories_list, memory_type='long-term')`
Saves a list of memories to the server.

**:Parameters**
- `memories_list`: List of memory strings
- `memory_type`: String type of memory (default: 'long-term')

**:Returns**
- Dictionary with save response or None on failure

**Section sources**
- [modules/episodic_memory/client.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\client.py#L52-L57)

#### `get_relevant_memories(query_text, top_n=5, similarity_threshold=0.7)`
Retrieves memories relevant to a query.

**:Parameters**
- `query_text`: Search query
- `top_n`: Maximum number of results (default: 5)
- `similarity_threshold`: Minimum similarity score (default: 0.7)

**:Returns**
- Dictionary with relevant memories or None on failure

**Section sources**
- [modules/episodic_memory/client.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\client.py#L59-L66)

#### `health_check()`
Checks the health of the memory database API.

**:Returns**
- Dictionary with health status or None on failure

**:Usage**
```python
health = health_check()
if health and health.get("status") == "ok":
    print("Memory API is healthy")
```

**Section sources**
- [modules/episodic_memory/client.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\client.py#L68-L71)

#### `upload_audio_file(file_path, context=None, extract_text=True)`
Uploads an audio file and processes it into memory.

**:Parameters**
- `file_path`: Path to the audio file
- `context`: Optional context for transcription
- `extract_text`: Whether to extract text from audio

**:Returns**
- Dictionary with processing result or None on failure

**:Usage**
```python
result = upload_audio_file("meeting_recording.mp3", context="Team meeting about project timeline")
if result and result["success"]:
    print(f"Audio processed with transcript: {result['transcript']}")
```

**Section sources**
- [modules/episodic_memory/client.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\client.py#L73-L80)

#### `upload_image_file(file_path, description=None)`
Uploads an image file and processes it into memory.

**:Parameters**
- `file_path`: Path to the image file
- `description`: Optional description of the image

**:Returns**
- Dictionary with processing result or None on failure

**:Usage**
```python
result = upload_image_file("vacation_photo.jpg", description="Sunset at the beach")
if result and result["success"]:
    print(f"Image processed with description: {result['description']}")
```

**Section sources**
- [modules/episodic_memory/client.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\client.py#L82-L89)

#### `advanced_search(query, content_types=None, memory_types=None, search_mode="hybrid")`
Performs advanced search with multiple filtering options.

**:Parameters**
- `query`: Search query string
- `content_types`: List of content types to include
- `memory_types`: List of memory types to include
- `search_mode`: Search mode ("text", "vector", "hybrid")

**:Returns**
- Dictionary with search results or None on failure

**:Usage**
```python
results = advanced_search(
    "beach vacation", 
    content_types=["image", "text"], 
    memory_types=["episodic"],
    search_mode="hybrid"
)
if results:
    for result in results["results"]:
        print(f"Found: {result['content_text']} (score: {result['similarity_score']})")
```

**Section sources**
- [modules/episodic_memory/client.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\client.py#L91-L100)

#### `batch_process_files(file_paths, content_types=None)`
Processes multiple files in batch.

**:Parameters**
- `file_paths`: List of file paths to process
- `content_types`: Optional list of content types corresponding to files

**:Returns**
- Dictionary with batch processing results or None on failure

**:Usage**
```python
results = batch_process_files([
    "notes.txt", 
    "meeting_recording.mp3", 
    "project_diagram.jpg"
])
if results:
    print(f"Processed {results['successful_count']} of {results['total_processed']} files")
```

**Section sources**
- [modules/episodic_memory/client.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\client.py#L102-L111)

### Reflection System

The `ReflectionModule` enables self-reflection capabilities.

#### `__init__(agi_system)`
Initializes the reflection module with a reference to the AGI system.

**:Parameters**
- `agi_system`: Reference to the main AGISystem instance

**Section sources**
- [modules/reflection_module.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\reflection_module.py#L10-L12)

#### `reflect_on_experiment(experiment_results)`
Analyzes experiment results and generates insights.

**:Parameters**
- `experiment_results`: Dictionary containing experiment data including 'hypothesis' and 'findings'

**:Notes**
- Automatically adds generated insights to the knowledge base
- Uses "reflection" as source and "insight" as category

**Section sources**
- [modules/reflection_module.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\reflection_module.py#L14-L33)

#### `reflect(shared_state)`
Performs general reflection based on the system's state.

**:Parameters**
- `shared_state`: SharedState object containing mood history and other state

**:Notes**
- Currently focuses on mood history analysis
- Placeholder for more sophisticated reflection logic

**Section sources**
- [modules/reflection_module.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\reflection_module.py#L35-L46)

## Multi-modal Memory Service

The `MultiModalMemoryService` is the main orchestration class for the multi-modal memory system, integrating PostgreSQL storage, embeddings, Whisper audio processing, and advanced search capabilities.

### Methods

#### `__init__(database_url, text_model_name="all-MiniLM-L6-v2", whisper_model_size="base", device=None)`
Initializes the multi-modal memory service with all required components.

**:Parameters**
- `database_url`: PostgreSQL connection URL
- `text_model_name`: SentenceTransformer model name for text embeddings
- `whisper_model_size`: Whisper model size for audio processing
- `device`: Device to use ("cpu", "cuda", "auto")

**:Usage**
```python
service = MultiModalMemoryService(
    database_url="postgresql://user:pass@localhost:5432/ravana",
    text_model_name="all-MiniLM-L6-v2",
    whisper_model_size="base"
)
await service.initialize()
```

**Section sources**
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py#L50-L85)

#### `initialize()`
Initializes all service components and establishes database connections.

**:Returns**
- Awaitable that completes when initialization is finished

**:Exceptions**
- Raises exception if initialization fails

**:Usage**
```python
try:
    await service.initialize()
    print("Service initialized successfully")
except Exception as e:
    print(f"Initialization failed: {e}")
```

**Section sources**
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py#L90-L105)

#### `close()`
Closes all service components gracefully and releases resources.

**:Returns**
- Awaitable that completes when shutdown is finished

**:Usage**
```python
await service.close()
```

**Section sources**
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py#L107-L165)

#### `process_text_memory(text, memory_type="episodic", tags=None, emotional_valence=None)`
Processes and stores text memory with embeddings.

**:Parameters**
- `text`: Text content to store
- `memory_type`: Type of memory ("episodic", "semantic", "procedural")
- `tags`: Optional list of tags
- `emotional_valence`: Emotional valence (-1.0 to 1.0)

**:Returns**
- Awaitable resolving to the stored MemoryRecord

**:Usage**
```python
record = await service.process_text_memory(
    "I enjoyed the concert last night",
    memory_type="episodic",
    tags=["entertainment", "music"],
    emotional_valence=0.8
)
```

**Section sources**
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py#L167-L215)

#### `process_audio_memory(audio_path, context=None, memory_type="episodic", tags=None)`
Processes and stores audio memory with Whisper transcription and embeddings.

**:Parameters**
- `audio_path`: Path to the audio file
- `context`: Optional context for transcription
- `memory_type`: Type of memory
- `tags`: Optional list of tags

**:Returns**
- Awaitable resolving to the stored MemoryRecord

**:Usage**
```python
record = await service.process_audio_memory(
    "meeting_recording.mp3",
    context="Team meeting about project timeline",
    tags=["work", "meetings"]
)
```

**Section sources**
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py#L217-L275)

#### `process_image_memory(image_path, description=None, memory_type="episodic", tags=None)`
Processes and stores image memory with metadata and embeddings.

**:Parameters**
- `image_path`: Path to the image file
- `description`: Optional image description
- `memory_type`: Type of memory
- `tags`: Optional list of tags

**:Returns**
- Awaitable resolving to the stored MemoryRecord

**:Usage**
```python
record = await service.process_image_memory(
    "vacation_photo.jpg",
    description="Sunset at the beach",
    tags=["vacation", "nature"]
)
```

**Section sources**
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py#L277-L335)

#### `extract_memories_from_conversation(request)`
Extracts memories from a conversation using LLM analysis.

**:Parameters**
- `request`: ConversationRequest object with user_input, ai_output, and context

**:Returns**
- Awaitable resolving to MemoriesList object

**:Usage**
```python
request = ConversationRequest(
    user_input="I'm planning a trip to Japan next spring",
    ai_output="That sounds like an amazing adventure!",
    context="Travel planning conversation"
)
memories = await service.extract_memories_from_conversation(request)
```

**Section sources**
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py#L337-L415)

#### `save_extracted_memories(memories_list)`
Saves a list of extracted memories to the database.

**:Parameters**
- `memories_list`: MemoriesList object containing memories to save

**:Returns**
- Awaitable resolving to list of saved MemoryRecord objects

**:Usage**
```python
saved_records = await service.save_extracted_memories(memories_list)
print(f"Saved {len(saved_records)} memories")
```

**Section sources**
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py#L417-L455)

#### `search_memories(request)`
Searches memories using advanced search capabilities.

**:Parameters**
- `request`: SearchRequest object with query and search parameters

**:Returns**
- Awaitable resolving to SearchResponse object

**:Usage**
```python
request = SearchRequest(
    query="beach vacation",
    content_types=[ContentType.IMAGE, ContentType.TEXT],
    memory_types=[MemoryType.EPISODIC],
    search_mode=SearchMode.HYBRID,
    limit=10,
    similarity_threshold=0.7
)
response = await service.search_memories(request)
```

**Section sources**
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py#L457-L475)

#### `find_similar_memories(memory_id, limit=10, similarity_threshold=0.7)`
Finds memories similar to a given memory.

**:Parameters**
- `memory_id`: UUID of the reference memory
- `limit`: Maximum number of similar memories to return
- `similarity_threshold`: Minimum similarity score (0.0 to 1.0)

**:Returns**
- Awaitable resolving to list of similar MemoryRecord objects

**:Usage**
```python
similar_memories = await service.find_similar_memories(
    memory_id="a1b2c3d4-e5f6-7890-1234-567890abcdef",
    limit=5,
    similarity_threshold=0.8
)
```

**Section sources**
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py#L477-L515)

#### `batch_process_files(request)`
Processes multiple files in batch with parallel processing.

**:Parameters**
- `request`: BatchProcessRequest object with file paths and processing options

**:Returns**
- Awaitable resolving to BatchProcessResult object

**:Usage**
```python
request = BatchProcessRequest(
    file_paths=["notes.txt", "meeting.mp3", "diagram.jpg"],
    parallel_processing=True,
    max_workers=4
)
result = await service.batch_process_files(request)
```

**Section sources**
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py#L517-L605)

#### `get_memory_statistics()`
Retrieves comprehensive statistics about the memory system.

**:Returns**
- Awaitable resolving to MemoryStatistics object

**:Usage**
```python
stats = await service.get_memory_statistics()
print(f"Total memories: {stats.total_memories}")
print(f"Storage size: {stats.storage_size_mb:.2f} MB")
```

**Section sources**
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py#L607-L645)

#### `consolidate_memories(memory_ids=None, max_memories=50)`
Consolidates memories to optimize storage and retrieval.

**:Parameters**
- `memory_ids`: Optional list of specific memory IDs to consolidate
- `max_memories`: Maximum number of memories to process

**:Returns**
- Awaitable resolving to dictionary with consolidation results

**:Usage**
```python
result = await service.consolidate_memories(
    memory_ids=["a1b2c3d4-e5f6-7890-1234-567890abcdef"],
    max_memories=100
)
```

**Section sources**
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py#L647-L675)

#### `health_check()`
Performs a comprehensive health check of the memory system.

**:Returns**
- Awaitable resolving to dictionary with health status

**:Usage**
```python
health = await service.health_check()
if health["status"] == "healthy":
    print("Memory system is healthy")
else:
    print(f"Memory system status: {health['status']}")
```

**Section sources**
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py#L677-L725)

## Memory Models

This section documents the data models used in the multi-modal memory system, defining the structure of memory records, search requests, and responses.

### Core Data Models

#### `MemoryRecord`
Main memory record model containing all memory data and metadata.

**:Attributes**
- `id`: Optional UUID for the memory record
- `content_type`: ContentType enum value (TEXT, AUDIO, IMAGE, VIDEO)
- `content_text`: Optional text content
- `content_metadata`: Dictionary of additional metadata
- `file_path`: Optional path to associated file
- `text_embedding`: Optional list of floats for text embedding
- `image_embedding`: Optional list of floats for image embedding
- `audio_embedding`: Optional list of floats for audio embedding
- `unified_embedding`: Optional list of floats for combined embedding
- `created_at`: Optional datetime of creation
- `last_accessed`: Optional datetime of last access
- `access_count`: Integer count of accesses
- `memory_type`: MemoryType enum value (EPISODIC, SEMANTIC, PROCEDURAL)
- `emotional_valence`: Optional float (-1.0 to 1.0) for emotional valence
- `confidence_score`: Float (0.0 to 1.0) for confidence in memory
- `tags`: List of string tags
- `audio_metadata`: Optional AudioMetadata object
- `image_metadata`: Optional ImageMetadata object
- `video_metadata`: Optional VideoMetadata object

**Section sources**
- [modules/episodic_memory/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\models.py#L51-L150)

#### `SearchRequest`
Request model for memory search operations with advanced filtering.

**:Attributes**
- `query`: Search query string (1-1000 characters)
- `content_types`: Optional list of ContentType values to filter by
- `memory_types`: Optional list of MemoryType values to filter by
- `search_mode`: SearchMode enum value (TEXT, VECTOR, HYBRID)
- `limit`: Integer limit on results (1-100)
- `similarity_threshold`: Float threshold for similarity (0.0-1.0)
- `include_metadata`: Boolean to include metadata in results
- `tags`: Optional list of tags to filter by
- `query_content_type`: Optional ContentType for cross-modal search
- `target_content_types`: Optional list of target types for cross-modal search
- `created_after`: Optional datetime filter for creation date
- `created_before`: Optional datetime filter for creation date

**Section sources**
- [modules/episodic_memory/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\models.py#L152-L190)

#### `SearchResponse`
Response model for search operations containing results and metadata.

**:Attributes**
- `results`: List of SearchResult objects
- `total_found`: Integer count of total matching memories
- `search_time_ms`: Integer search duration in milliseconds
- `search_mode`: SearchMode enum value used
- `query_metadata`: Dictionary of additional query metadata

**Section sources**
- [modules/episodic_memory/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\models.py#L230-L240)

#### `ConversationRequest`
Request model for memory extraction from conversations.

**:Attributes**
- `user_input`: User message text (required)
- `ai_output`: AI response text (required)
- `context`: Optional context for extraction
- `extract_emotions`: Boolean to extract emotional content
- `memory_type`: MemoryType enum value for extracted memories

**Section sources**
- [modules/episodic_memory/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\models.py#L242-L250)

#### `MemoriesList`
List of extracted memories with metadata.

**:Attributes**
- `memories`: List of memory text strings
- `memory_type`: MemoryType enum value for all memories
- `confidence_scores`: Optional list of confidence scores
- `emotional_valences`: Optional list of emotional valences

**Section sources**
- [modules/episodic_memory/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\models.py#L252-L258)

### Content Type Models

#### `AudioMetadata`
Metadata specific to audio content.

**:Attributes**
- `transcript`: Optional transcription text
- `language_code`: Optional language code (e.g., "en")
- `confidence_scores`: Dictionary of confidence scores
- `duration_seconds`: Optional duration in seconds
- `audio_features`: Dictionary of audio analysis features
- `sample_rate`: Optional sample rate in Hz
- `channels`: Optional number of audio channels

**Section sources**
- [modules/episodic_memory/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\models.py#L11-L25)

#### `ImageMetadata`
Metadata specific to image content.

**:Attributes**
- `width`: Optional image width in pixels
- `height`: Optional image height in pixels
- `object_detections`: Dictionary of detected objects
- `scene_description`: Optional scene description
- `image_hash`: Optional perceptual hash
- `color_palette`: List of dominant colors
- `image_features`: Dictionary of image analysis features

**Section sources**
- [modules/episodic_memory/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\models.py#L27-L40)

#### `VideoMetadata`
Metadata specific to video content.

**:Attributes**
- `duration_seconds`: Optional duration in seconds
- `frame_rate`: Optional frame rate in fps
- `width`: Optional video width in pixels
- `height`: Optional video height in pixels
- `video_features`: Dictionary of video analysis features
- `thumbnail_path`: Optional path to thumbnail image

**Section sources**
- [modules/episodic_memory/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\models.py#L42-L50)

### Enumerations

#### `ContentType`
Enumeration of supported content types.

**:Values**
- `TEXT`: Text content
- `AUDIO`: Audio content
- `IMAGE`: Image content
- `VIDEO`: Video content

**Section sources**
- [modules/episodic_memory/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\models.py#L1-L5)

#### `MemoryType`
Enumeration of memory types.

**:Values**
- `EPISODIC`: Episodic memories (personal experiences)
- `SEMANTIC`: Semantic memories (facts and knowledge)
- `PROCEDURAL`: Procedural memories (skills and procedures)

**Section sources**
- [modules/episodic_memory/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\models.py#L7-L10)

#### `SearchMode`
Enumeration of search modes.

**:Values**
- `TEXT`: Text-based search
- `VECTOR`: Vector similarity search
- `HYBRID`: Hybrid search combining text and vector

**Section sources**
- [modules/episodic_memory/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\models.py#L42-L45)

## PostgreSQL Store

The `PostgreSQLStore` class handles database operations for the multi-modal memory system with pgvector support for similarity search.

### Methods

#### `__init__(database_url, pool_size=10, max_connections=20)`
Initializes the PostgreSQL store with connection parameters.

**:Parameters**
- `database_url`: PostgreSQL connection URL
- `pool_size`: Minimum connection pool size
- `max_connections`: Maximum connection pool size

**:Exceptions**
- Raises ImportError if AsyncPG is not available

**Section sources**
- [modules/episodic_memory/postgresql_store.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\postgresql_store.py#L50-L65)

#### `initialize()`
Initializes the database connection pool.

**:Returns**
- Awaitable that completes when initialization is finished

**:Exceptions**
- Raises exception if initialization fails

**:Usage**
```python
await store.initialize()
```

**Section sources**
- [modules/episodic_memory/postgresql_store.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\postgresql_store.py#L67-L85)

#### `close()`
Closes the database connection pool.

**:Returns**
- Awaitable that completes when closure is finished

**:Usage**
```python
await store.close()
```

**Section sources**
- [modules/episodic_memory/postgresql_store.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\postgresql_store.py#L87-L95)

#### `save_memory_record(memory_record)`
Saves a memory record to the database.

**:Parameters**
- `memory_record`: MemoryRecord object to save

**:Returns**
- Awaitable resolving to the saved MemoryRecord

**:Exceptions**
- Raises exception if save fails

**:Usage**
```python
saved_record = await store.save_memory_record(memory_record)
```

**Section sources**
- [modules/episodic_memory/postgresql_store.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\postgresql_store.py#L97-L155)

#### `get_memory_record(memory_id)`
Retrieves a memory record by ID.

**:Parameters**
- `memory_id`: UUID of the memory record

**:Returns**
- Awaitable resolving to MemoryRecord or None if not found

**:Usage**
```python
record = await store.get_memory_record("a1b2c3d4-e5f6-7890-1234-567890abcdef")
if record:
    print(f"Found memory: {record.content_text}")
```

**Section sources**
- [modules/episodic_memory/postgresql_store.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\postgresql_store.py#L305-L325)

#### `vector_search(embedding, embedding_type="text", limit=10, similarity_threshold=0.7, content_types=None)`
Performs vector similarity search.

**:Parameters**
- `embedding`: List of floats representing the query embedding
- `embedding_type`: Type of embedding ("text", "image", "audio", "unified")
- `limit`: Maximum number of results to return
- `similarity_threshold`: Minimum similarity score (0.0-1.0)
- `content_types`: Optional list of ContentType values to filter by

**:Returns**
- Awaitable resolving to list of (MemoryRecord, similarity_score) tuples

**:Usage**
```python
results = await store.vector_search(
    embedding=[0.1, 0.2, 0.3, ...],
    embedding_type="text",
    limit=5,
    similarity_threshold=0.8
)
```

**Section sources**
- [modules/episodic_memory/postgresql_store.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\postgresql_store.py#L391-L455)

#### `text_search(query_text, limit=10, content_types=None)`
Performs full-text search.

**:Parameters**
- `query_text`: Search query string
- `limit`: Maximum number of results to return
- `content_types`: Optional list of ContentType values to filter by

**:Returns**
- Awaitable resolving to list of (MemoryRecord, relevance_score) tuples

**:Usage**
```python
results = await store.text_search("beach vacation", limit=10)
```

**Section sources**
- [modules/episodic_memory/postgresql_store.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\postgresql_store.py#L457-L505)

#### `delete_memory_record(memory_id)`
Deletes a memory record by ID.

**:Parameters**
- `memory_id`: UUID of the memory record to delete

**:Returns**
- Awaitable resolving to boolean (True if deleted, False if not found)

**:Usage**
```python
deleted = await store.delete_memory_record("a1b2c3d4-e5f6-7890-1234-567890abcdef")
if deleted:
    print("Memory deleted successfully")
```

**Section sources**
- [modules/episodic_memory/postgresql_store.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\postgresql_store.py#L507-L525)

#### `get_memory_statistics()`
Retrieves comprehensive statistics about the memory system.

**:Returns**
- Awaitable resolving to dictionary with statistics

**:Usage**
```python
stats = await store.get_memory_statistics()
print(f"Total memories: {stats['total_memories']}")
```

**Section sources**
- [modules/episodic_memory/postgresql_store.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\postgresql_store.py#L527-L575)

#### `cleanup_old_memories(days_old=30, keep_minimum=1000)`
Cleans up old, rarely accessed memories.

**:Parameters**
- `days_old`: Age threshold in days
- `keep_minimum`: Minimum number of memories to keep

**:Returns**
- Awaitable resolving to integer count of deleted memories

**:Usage**
```python
deleted_count = await store.cleanup_old_memories(days_old=60, keep_minimum=500)
print(f"Cleaned up {deleted_count} old memories")
```

**Section sources**
- [modules/episodic_memory/postgresql_store.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\postgresql_store.py#L577-L590)

## Embedding Service

The `EmbeddingService` class handles the generation of embeddings for multi-modal content, supporting text, image, audio, and unified embeddings.

### Methods

#### `__init__(text_model_name="all-MiniLM-L6-v2", device=None, cache_size=1000)`
Initializes the embedding service with configuration parameters.

**:Parameters**
- `text_model_name`: Name of the sentence transformer model
- `device`: Device to use ("cpu", "cuda", "auto")
- `cache_size`: Size of the embedding cache

**:Exceptions**
- Raises ImportError if transformers dependencies are not available

**:Usage**
```python
service = EmbeddingService(
    text_model_name="all-MiniLM-L6-v2",
    device="cuda",
    cache_size=2000
)
```

**Section sources**
- [modules/episodic_memory/embedding_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\embedding_service.py#L100-L125)

#### `generate_text_embedding(text)`
Generates embedding for text content.

**:Parameters**
- `text`: Text string to embed

**:Returns**
- Awaitable resolving to list of floats (embedding values)

**:Usage**
```python
embedding = await service.generate_text_embedding("This is a sample text")
```

**Section sources**
- [modules/episodic_memory/embedding_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\embedding_service.py#L127-L165)

#### `generate_image_embedding(image_path)`
Generates embedding for image content.

**:Parameters**
- `image_path`: Path to the image file

**:Returns**
- Awaitable resolving to list of floats (embedding values)

**:Usage**
```python
embedding = await service.generate_image_embedding("photo.jpg")
```

**Section sources**
- [modules/episodic_memory/embedding_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\embedding_service.py#L167-L245)

#### `generate_audio_embedding(audio_features)`
Generates embedding for audio content from extracted features.

**:Parameters**
- `audio_features`: Dictionary of audio analysis features

**:Returns**
- Awaitable resolving to list of floats (embedding values)

**:Usage**
```python
embedding = await service.generate_audio_embedding({
    "mfcc": {"mean": [...], "std": [...]},
    "spectral_centroid": {"mean": 1000.0, "std": 200.0}
})
```

**Section sources**
- [modules/episodic_memory/embedding_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\embedding_service.py#L247-L335)

#### `generate_unified_embedding(memory_record)`
Generates unified embedding combining all available modalities.

**:Parameters**
- `memory_record`: MemoryRecord object with various embeddings

**:Returns**
- Awaitable resolving to list of floats (unified embedding values)

**:Usage**
```python
unified_embedding = await service.generate_unified_embedding(memory_record)
```

**Section sources**
- [modules/episodic_memory/embedding_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\embedding_service.py#L337-L415)

#### `generate_embeddings(memory_record)`
Generates all relevant embeddings for a memory record.

**:Parameters**
- `memory_record`: MemoryRecord object to process

**:Returns**
- Awaitable resolving to MemoryRecord with embeddings populated

**:Usage**
```python
processed_record = await service.generate_embeddings(memory_record)
```

**Section sources**
- [modules/episodic_memory/embedding_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\embedding_service.py#L417-L475)

## Initialization and Lifecycle Management

The AGI system follows a specific initialization and lifecycle pattern to ensure proper setup and graceful shutdown.

### Initialization Pattern

```python
from sqlalchemy import create_engine
from core.system import AGISystem

# Create database engine
engine = create_engine("sqlite:///ravana.db")

# Initialize AGI system
agi = AGISystem(engine)

# System is now ready for use
```

The initialization process:
1. Creates all service instances (data, knowledge, memory)
2. Initializes modules (situation generator, emotional intelligence, etc.)
3. Sets up shared state and behavior modifiers
4. Prepares background tasks (not started yet)

### Lifecycle Management

The AGI system supports both autonomous and task-based operation modes:

#### Autonomous Mode
```python
import asyncio

async def main():
    try:
        await agi.run_autonomous_loop()
    except KeyboardInterrupt:
        await agi.stop()

asyncio.run(main())
```

#### Single Task Mode
```python
await agi.run_single_task("Research renewable energy technologies")
await agi.stop()
```

#### Graceful Shutdown
```python
await agi.stop()  # Cancels background tasks and closes resources
```

The system uses an asyncio.Event for shutdown signaling and properly cancels all background tasks.

**Section sources**
- [core/system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py#L1-L625)

## Thread Safety and Async Usage

The AGI system is designed for asynchronous operation with careful consideration of thread safety.

### Async/Await Usage

Most methods are coroutines and must be awaited:

```python
# Correct usage
await agi.run_iteration()
await agi._handle_curiosity()

# Incorrect usage (will return coroutine object)
agi.run_iteration()  # Missing await
```

### Thread Safety Considerations

- Database operations use `asyncio.to_thread()` for synchronous calls
- The `EnhancedActionManager` handles action execution in thread pool
- Shared state modifications are coordinated through the main event loop
- Background tasks are properly managed and canceled on shutdown

### Background Tasks

The system manages several background tasks:
- Data collection (RSS feeds)
- Event detection
- Knowledge compression
- Memory consolidation
- Invention tracking

These tasks are automatically started in `run_autonomous_loop()` and properly cleaned up in `stop()`.

**Section sources**
- [core/system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py#L520-L550)

## Error Handling

The system implements comprehensive error handling at multiple levels.

### Exception Types

- `InvalidActionParams`: Raised when action parameters are invalid
- `ValueError`: Used for validation errors (e.g., empty content)
- `TypeError`: For incorrect argument types
- `FileNotFoundError`: When files don't exist
- `json.JSONDecodeError`: When parsing JSON fails

### Error Handling Patterns

#### Service-Level Error Handling
```python
try:
    result = await self.memory_service.extract_memories(interaction_summary, "")
    if result and result.memories:
        await self.memory_service.save_memories(result.memories)
except Exception as e:
    logger.error(f"Failed during memorization: {e}", exc_info=True)
```

#### Background Task Error Handling
```python
while not self._shutdown.is_set():
    try:
        # Task logic here
        pass
    except asyncio.CancelledError:
        break
    except Exception as e:
        logger.error(f"Error in task: {e}", exc_info=True)
```

#### External API Error Handling
The episodic memory client implements retry logic:
- Maximum of 3 retries for connection errors
- 1-second delay between retries
- Comprehensive error reporting

**Section sources**
- [core/system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py#L100-L625)
- [modules/episodic_memory/client.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\client.py#L1-L154)

## Usage Examples

### Basic System Initialization and Operation

```python
from sqlalchemy import create_engine
from core.system import AGISystem
import asyncio

async def main():
    # Initialize system
    engine = create_engine("sqlite:///ravana.db")
    agi = AGISystem(engine)
    
    try:
        # Run autonomous loop
        await agi.run_autonomous_loop()
    except KeyboardInterrupt:
        print("Shutting down...")
        await agi.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Executing a Single Task

```python
async def execute_research_task():
    agi = AGISystem(engine)
    
    # Run a specific task
    await agi.run_single_task("Investigate the impact of climate change on coastal cities")
    
    # Get recent events
    events = await agi.get_recent_events(3600)
    print(f"Found {len(events)} recent events")
    
    # Shutdown
    await agi.stop()

asyncio.run(execute_research_task())
```

### Using the Multi-modal Service

```python
async def analyze_media():
    multi_modal = MultiModalService()
    
    # Process an image
    image_result = await multi_modal.process_image("photo.jpg")
    
    # Process an audio file
    audio_result = await multi_modal.process_audio("recording.mp3")
    
    # Cross-modal analysis
    analysis = await multi_modal.cross_modal_analysis([
        image_result, 
        audio_result
    ])
    
    # Generate summary
    summary = await multi_modal.generate_content_summary([
        image_result, 
        audio_result
    ])
    
    print(summary)

asyncio.run(analyze_media())
```

### Working with the Memory System

```python
# Direct client usage
def manage_memories():
    # Extract memories from conversation
    result = extract_memories(
        "I'm planning a trip to Japan next spring", 
        "That sounds like an amazing adventure!"
    )
    
    if result and 'memories' in result:
        # Save extracted memories
        save_response = save_memories(result['memories'])
        
        # Retrieve relevant memories later
        relevant = get_relevant_memories("travel plans", top_n=3)
        for mem in relevant['relevant_memories']:
            print(f"Found: {mem['text']}")

manage_memories()
```

### Adding Knowledge

```python
async def add_research_knowledge():
    knowledge_service = KnowledgeService(engine)
    
    research_content = """
    Recent studies show that neural network pruning can reduce model size 
    by up to 90% with minimal accuracy loss. This technique involves 
    removing redundant weights and neurons from trained models.
    """
    
    result = await asyncio.to_thread(
        knowledge_service.add_knowledge,
        content=research_content,
        source="research_paper",
        category="machine_learning"
    )
    
    if not result.get('duplicate'):
        print(f"Added new knowledge with ID: {result['id']}")
    else:
        print("Knowledge already existed in database")

asyncio.run(add_research_knowledge())
```

### Advanced Memory Operations

```python
async def advanced_memory_operations():
    # Initialize multi-modal memory service
    service = MultiModalMemoryService(
        database_url="postgresql://user:pass@localhost:5432/ravana"
    )
    await service.initialize()
    
    # Process different types of memories
    text_record = await service.process_text_memory(
        "I learned about quantum computing today",
        tags=["learning", "science"]
    )
    
    audio_record = await service.process_audio_memory(
        "lecture_recording.mp3",
        context="Physics lecture on quantum mechanics"
    )
    
    image_record = await service.process_image_memory(
        "quantum_diagram.jpg",
        description="Diagram explaining quantum entanglement"
    )
    
    # Advanced search
    search_request = SearchRequest(
        query="quantum computing",
        content_types=[ContentType.TEXT, ContentType.AUDIO, ContentType.IMAGE],
        search_mode=SearchMode.HYBRID,
        limit=5
    )
    search_response = await service.search_memories(search_request)
    
    # Display results
    for result in search_response.results:
        print(f"Found: {result.memory_record.content_text[:100]}... "
              f"(score: {result.similarity_score:.3f})")
    
    # Batch processing
    batch_request = BatchProcessRequest(
        file_paths=["notes1.txt", "notes2.txt", "lecture.mp3"],
        parallel_processing=True
    )
    batch_result = await service.batch_process_files(batch_request)
    
    print(f"Processed {batch_result.successful_count}/{batch_result.total_processed} files")
    
    # Cleanup
    await service.close()

asyncio.run(advanced_memory_operations())
```

**Section sources**
- [core/system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py#L1-L625)
- [services/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\multi_modal_service.py#L1-L350)
- [modules/episodic_memory/client.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\client.py#L1-L154)
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py#L1-L657)
- [modules/episodic_memory/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\models.py#L1-L251)

**Referenced Files in This Document**   
- [core/system.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\system.py) - *Updated in recent commit*
- [core/actions/action.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\core\actions\action.py)
- [services/data_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\data_service.py)
- [services/knowledge_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\knowledge_service.py)
- [services/memory_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\memory_service.py)
- [services/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\services\multi_modal_service.py)
- [modules/episodic_memory/client.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\client.py) - *Enhanced with new features*
- [modules/episodic_memory/multi_modal_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\multi_modal_service.py) - *Added in recent commit*
- [modules/episodic_memory/models.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\models.py) - *Added in recent commit*
- [modules/episodic_memory/postgresql_store.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\postgresql_store.py) - *Added in recent commit*
- [modules/episodic_memory/embedding_service.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\episodic_memory\embedding_service.py) - *Added in recent commit*
- [modules/reflection_module.py](file://c:\Users\ASUS\Documents\GitHub\RAVANA\modules\reflection_module.py)