# Memory Service API

## Overview

The Memory Service API provides programmatic access to memory storage and retrieval operations within RAVANA AGI. This API enables modules and external systems to store, query, and manage both episodic and semantic memories through standardized interfaces.

## Authentication

All API endpoints require authentication through API keys:

```
Authorization: Bearer YOUR_API_KEY
```

## Error Handling

API responses follow standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 500: Internal Server Error

Error responses include JSON body with error details:
```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": "Additional error information"
}
```

## Memory Operations

### Store Memory

```
POST /api/v1/memory
```

Store a new memory record in the appropriate memory system.

**Request Body:**
```json
{
  "type": "episodic",
  "content": {
    "event": "User asked about machine learning",
    "context": "Educational conversation",
    "response": "Provided definition of machine learning"
  },
  "metadata": {
    "timestamp": "2023-01-01T10:30:00Z",
    "importance": 0.7,
    "emotional_context": {
      "happiness": 0.8,
      "curiosity": 0.9
    }
  },
  "tags": ["education", "machine-learning", "user-interaction"]
}
```

**Response:**
```json
{
  "id": "generated_memory_id",
  "type": "episodic",
  "timestamp": "2023-01-01T10:30:00Z",
  "status": "success"
}
```

### Retrieve Memory

```
GET /api/v1/memory/{id}
```

Retrieve specific memory by ID.

**Parameters:**
- `id` (path): Unique identifier of the memory record

**Response:**
```json
{
  "id": "memory_id",
  "type": "episodic",
  "content": {
    "event": "User asked about machine learning",
    "context": "Educational conversation",
    "response": "Provided definition of machine learning"
  },
  "metadata": {
    "timestamp": "2023-01-01T10:30:00Z",
    "importance": 0.7,
    "emotional_context": {
      "happiness": 0.8,
      "curiosity": 0.9
    }
  },
  "tags": ["education", "machine-learning", "user-interaction"],
  "access_count": 3,
  "last_accessed": "2023-01-02T15:45:00Z"
}
```

### Update Memory

```
PUT /api/v1/memory/{id}
```

Update existing memory record.

**Parameters:**
- `id` (path): Unique identifier of the memory record

**Request Body:**
```json
{
  "content": {
    "additional_notes": "User showed particular interest in deep learning"
  },
  "tags": ["education", "machine-learning", "user-interaction", "deep-learning"]
}
```

**Response:**
```json
{
  "id": "memory_id",
  "updated": "2023-01-01T11:00:00Z",
  "status": "success"
}
```

### Delete Memory

```
DELETE /api/v1/memory/{id}
```

Delete memory record by ID.

**Parameters:**
- `id` (path): Unique identifier of the memory record

**Response:**
```json
{
  "id": "memory_id",
  "deleted": "2023-01-01T11:00:00Z",
  "status": "success"
}
```

## Memory Search and Retrieval

### Search Memories

```
POST /api/v1/memory/search
```

Search memories using semantic and contextual queries.

**Request Body:**
```json
{
  "query": "machine learning discussions",
  "type": "episodic",
  "context": {
    "time_range": {
      "start": "2023-01-01T00:00:00Z",
      "end": "2023-01-31T23:59:59Z"
    },
    "emotional_state": {
      "happiness": {"min": 0.5},
      "curiosity": {"min": 0.7}
    }
  },
  "tags": ["education", "machine-learning"],
  "limit": 10,
  "min_similarity": 0.7
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "memory_id1",
      "type": "episodic",
      "content": {
        "event": "User asked about machine learning",
        "response": "Provided definition of machine learning"
      },
      "similarity_score": 0.85,
      "relevance_score": 0.92
    },
    {
      "id": "memory_id2",
      "type": "episodic",
      "content": {
        "event": "Discussion about neural networks",
        "response": "Explained deep learning concepts"
      },
      "similarity_score": 0.78,
      "relevance_score": 0.85
    }
  ],
  "total": 2,
  "query": "machine learning discussions"
}
```

### Contextual Memory Retrieval

```
POST /api/v1/memory/context
```

Retrieve memories based on current context.

**Request Body:**
```json
{
  "context": {
    "current_topic": "artificial intelligence",
    "emotional_state": {
      "curiosity": 0.8,
      "happiness": 0.7
    },
    "recent_activities": ["user_question", "research_task"]
  },
  "type": "episodic",
  "limit": 5
}
```

**Response:**
```json
{
  "context": "artificial intelligence discussion",
  "memories": [
    {
      "id": "ai_memory_id",
      "content": {
        "event": "Previous AI discussion",
        "key_points": ["definitions", "applications", "ethics"]
      },
      "contextual_relevance": 0.88
    }
  ]
}
```

### Recent Memories

```
GET /api/v1/memory/recent
```

Retrieve recently accessed or created memories.

**Parameters:**
- `type` (query): Type of memory (episodic/semantic) (optional)
- `limit` (query): Maximum number of results (default: 10)

**Response:**
```json
{
  "memories": [
    {
      "id": "recent_memory_id",
      "type": "episodic",
      "content": {
        "event": "Latest user interaction",
        "timestamp": "2023-01-01T15:30:00Z"
      },
      "accessed": "2023-01-01T15:35:00Z"
    }
  ],
  "total": 1
}
```

## Memory System Operations

### Get Memory Statistics

```
GET /api/v1/memory/stats
```

Retrieve statistics about memory usage and distribution.

**Response:**
```json
{
  "total_memories": 12500,
  "episodic_count": 10500,
  "semantic_count": 2000,
  "storage_usage": {
    "episodic": "2.5GB",
    "semantic": "0.8GB"
  },
  "access_patterns": {
    "daily_accesses": 1250,
    "most_accessed": "user_interactions",
    "average_retention": "45 days"
  }
}
```

### Memory Consolidation

```
POST /api/v1/memory/consolidate
```

Trigger memory consolidation process.

**Request Body:**
```json
{
  "type": "episodic_to_semantic",
  "strategy": "abstract_and_summarize",
  "parameters": {
    "min_importance": 0.7,
    "time_threshold": "30d"
  }
}
```

**Response:**
```json
{
  "job_id": "consolidation_job_123",
  "status": "started",
  "estimated_completion": "2023-01-01T12:00:00Z"
}
```

### Check Consolidation Status

```
GET /api/v1/memory/consolidate/{job_id}
```

Check the status of a memory consolidation job.

**Parameters:**
- `job_id` (path): ID of the consolidation job

**Response:**
```json
{
  "job_id": "consolidation_job_123",
  "status": "completed",
  "progress": 100,
  "results": {
    "processed_memories": 150,
    "created_knowledge": 12,
    "compressed_memories": 45
  },
  "completed": "2023-01-01T11:45:00Z"
}
```

## Memory Types

### Episodic Memory Operations

```
POST /api/v1/memory/episodic
```

Store an episodic memory (personal experience).

**Request Body:**
```json
{
  "event": "User interaction session",
  "participants": ["user", "ravana"],
  "location": "chat_interface",
  "duration": "15m",
  "outcome": "successful_information_exchange",
  "emotional_trajectory": [
    {"time": "00:00", "happiness": 0.6, "curiosity": 0.7},
    {"time": "07:30", "happiness": 0.8, "curiosity": 0.9},
    {"time": "15:00", "happiness": 0.7, "curiosity": 0.8}
  ]
}
```

### Semantic Memory Operations

```
POST /api/v1/memory/semantic
```

Store a semantic memory (general knowledge).

**Request Body:**
```json
{
  "concept": "Machine Learning",
  "definition": "A method of data analysis that automates analytical model building.",
  "category": "Artificial Intelligence",
  "applications": ["image_recognition", "natural_language_processing", "prediction"],
  "related_concepts": ["Deep Learning", "Neural Networks", "Data Science"]
}
```

## Batch Operations

### Batch Store Memories

```
POST /api/v1/memory/batch
```

Store multiple memories in a single request.

**Request Body:**
```json
{
  "memories": [
    {
      "type": "episodic",
      "content": {"event": "Event 1"}
    },
    {
      "type": "semantic",
      "content": {"concept": "Concept 1"}
    }
  ]
}
```

**Response:**
```json
{
  "inserted": 2,
  "ids": ["id1", "id2"],
  "status": "success"
}
```

## Performance and Monitoring

### Health Check

```
GET /api/v1/memory/health
```

Check the health status of the memory service.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": "24h",
  "systems": {
    "episodic": "healthy",
    "semantic": "healthy"
  },
  "storage": {
    "episodic": "2.5GB/5GB",
    "semantic": "0.8GB/2GB"
  }
}
```

### Metrics

```
GET /api/v1/memory/metrics
```

Retrieve performance metrics for the memory service.

**Response:**
```json
{
  "queries_per_second": 75.3,
  "average_response_time": 28.4,
  "error_rate": 0.002,
  "cache_hit_rate": 0.82,
  "consolidation_jobs": 5,
  "memory_growth_rate": "150/day"
}
```

## Rate Limiting

The Memory Service API implements rate limiting to ensure fair usage:
- 1000 requests per minute per API key
- 10000 requests per hour per API key

Exceeding rate limits will result in a 429 (Too Many Requests) response.