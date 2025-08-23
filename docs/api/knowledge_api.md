# Knowledge Service API

## Overview

The Knowledge Service API provides programmatic access to knowledge management and semantic search operations within RAVANA AGI. This API enables modules and external systems to store, query, and manage structured knowledge through standardized interfaces.

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

## Knowledge Base Operations

### Store Knowledge

```
POST /api/v1/knowledge
```

Store structured knowledge in the knowledge base.

**Request Body:**
```json
{
  "concept": "Artificial Intelligence",
  "definition": "A branch of computer science dealing with the simulation of intelligent behavior in computers.",
  "category": "Computer Science",
  "relationships": [
    {
      "type": "related_to",
      "target": "Machine Learning"
    },
    {
      "type": "subcategory_of",
      "target": "Computer Science"
    }
  ],
  "metadata": {
    "source": "Wikipedia",
    "confidence": 0.95
  }
}
```

**Response:**
```json
{
  "id": "generated_id",
  "timestamp": "2023-01-01T00:00:00Z",
  "status": "success"
}
```

### Retrieve Knowledge

```
GET /api/v1/knowledge/{id}
```

Retrieve specific knowledge by ID.

**Parameters:**
- `id` (path): Unique identifier of the knowledge record

**Response:**
```json
{
  "id": "knowledge_id",
  "concept": "Artificial Intelligence",
  "definition": "A branch of computer science dealing with the simulation of intelligent behavior in computers.",
  "category": "Computer Science",
  "relationships": [
    {
      "type": "related_to",
      "target": "Machine Learning",
      "target_id": "ml_id"
    }
  ],
  "metadata": {
    "source": "Wikipedia",
    "confidence": 0.95,
    "created": "2023-01-01T00:00:00Z",
    "updated": "2023-01-01T00:00:00Z"
  }
}
```

### Update Knowledge

```
PUT /api/v1/knowledge/{id}
```

Update existing knowledge record.

**Parameters:**
- `id` (path): Unique identifier of the knowledge record

**Request Body:**
```json
{
  "definition": "An updated definition of artificial intelligence.",
  "additional_relationships": [
    {
      "type": "related_to",
      "target": "Deep Learning"
    }
  ]
}
```

**Response:**
```json
{
  "id": "knowledge_id",
  "updated": "2023-01-01T00:00:00Z",
  "status": "success"
}
```

### Delete Knowledge

```
DELETE /api/v1/knowledge/{id}
```

Delete knowledge record by ID.

**Parameters:**
- `id` (path): Unique identifier of the knowledge record

**Response:**
```json
{
  "id": "knowledge_id",
  "deleted": "2023-01-01T00:00:00Z",
  "status": "success"
}
```

## Semantic Search

### Search Knowledge

```
POST /api/v1/knowledge/search
```

Perform semantic search across the knowledge base.

**Request Body:**
```json
{
  "query": "What is machine learning?",
  "context": "Computer science education",
  "limit": 10,
  "min_confidence": 0.7
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "ml_knowledge_id",
      "concept": "Machine Learning",
      "definition": "A method of data analysis that automates analytical model building.",
      "confidence": 0.92,
      "relevance_score": 0.88
    },
    {
      "id": "ai_knowledge_id",
      "concept": "Artificial Intelligence",
      "definition": "A branch of computer science dealing with the simulation of intelligent behavior.",
      "confidence": 0.89,
      "relevance_score": 0.75
    }
  ],
  "total": 2,
  "query": "What is machine learning?",
  "processed_query": "machine learning definition computer science"
}
```

### Find Related Knowledge

```
GET /api/v1/knowledge/{id}/related
```

Find knowledge concepts related to a specific concept.

**Parameters:**
- `id` (path): Unique identifier of the knowledge record
- `relationship_type` (query): Type of relationship to filter (optional)
- `limit` (query): Maximum number of results (default: 100)

**Response:**
```json
{
  "concept": "Machine Learning",
  "related": [
    {
      "id": "dl_knowledge_id",
      "concept": "Deep Learning",
      "relationship_type": "subtype_of",
      "confidence": 0.95
    },
    {
      "id": "ai_knowledge_id",
      "concept": "Artificial Intelligence",
      "relationship_type": "subcategory_of",
      "confidence": 0.92
    }
  ]
}
```

## Ontology Management

### List Ontologies

```
GET /api/v1/knowledge/ontologies
```

List all available ontologies.

**Response:**
```json
{
  "ontologies": [
    {
      "name": "ravana_core",
      "version": "1.0.0",
      "concepts": 1250,
      "relationships": 3400
    },
    {
      "name": "computer_science",
      "version": "2.1.0",
      "concepts": 850,
      "relationships": 2100
    }
  ]
}
```

### Get Ontology Details

```
GET /api/v1/knowledge/ontologies/{name}
```

Retrieve detailed information about a specific ontology.

**Parameters:**
- `name` (path): Name of the ontology

**Response:**
```json
{
  "name": "ravana_core",
  "version": "1.0.0",
  "description": "Core ontology for RAVANA AGI system",
  "categories": ["concepts", "entities", "relationships"],
  "concepts": [
    {
      "name": "ArtificialIntelligence",
      "properties": ["definition", "category", "applications"],
      "relationships": ["related_to", "subtype_of"]
    }
  ],
  "statistics": {
    "total_concepts": 1250,
    "total_relationships": 3400,
    "last_updated": "2023-01-01T00:00:00Z"
  }
}
```

### Validate Knowledge

```
POST /api/v1/knowledge/validate
```

Validate knowledge structure and consistency.

**Request Body:**
```json
{
  "concept": "New Concept",
  "definition": "Definition of the new concept",
  "category": "Test Category",
  "relationships": [
    {
      "type": "related_to",
      "target": "Existing Concept"
    }
  ]
}
```

**Response:**
```json
{
  "valid": true,
  "issues": [],
  "suggestions": [
    "Consider adding more specific relationships",
    "Definition could be more detailed"
  ],
  "confidence_score": 0.85
}
```

## Graph Operations

### Get Knowledge Graph

```
GET /api/v1/knowledge/graph
```

Retrieve a portion of the knowledge graph.

**Parameters:**
- `center` (query): Central concept for the graph (optional)
- `depth` (query): Depth of relationships to include (default: 2)
- `limit` (query): Maximum number of nodes (default: 100)

**Response:**
```json
{
  "nodes": [
    {
      "id": "ai_id",
      "label": "Artificial Intelligence",
      "category": "Concept"
    },
    {
      "id": "ml_id",
      "label": "Machine Learning",
      "category": "Concept"
    }
  ],
  "edges": [
    {
      "source": "ml_id",
      "target": "ai_id",
      "type": "subcategory_of",
      "confidence": 0.92
    }
  ],
  "center": "ai_id"
}
```

### Find Shortest Path

```
GET /api/v1/knowledge/path
```

Find the shortest path between two concepts in the knowledge graph.

**Parameters:**
- `from` (query): Starting concept ID or name
- `to` (query): Ending concept ID or name

**Response:**
```json
{
  "path": [
    {
      "id": "ml_id",
      "concept": "Machine Learning"
    },
    {
      "id": "ai_id",
      "concept": "Artificial Intelligence"
    },
    {
      "id": "cs_id",
      "concept": "Computer Science"
    }
  ],
  "relationships": [
    "subcategory_of",
    "subcategory_of"
  ],
  "length": 2
}
```

## Batch Operations

### Batch Store Knowledge

```
POST /api/v1/knowledge/batch
```

Store multiple knowledge records in a single request.

**Request Body:**
```json
{
  "knowledge": [
    {
      "concept": "Concept 1",
      "definition": "Definition of concept 1"
    },
    {
      "concept": "Concept 2",
      "definition": "Definition of concept 2"
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
GET /api/v1/knowledge/health
```

Check the health status of the knowledge service.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": "24h",
  "ontology": "ravana_core_v1.0.0",
  "concepts": 1250,
  "relationships": 3400
}
```

### Metrics

```
GET /api/v1/knowledge/metrics
```

Retrieve performance metrics for the knowledge service.

**Response:**
```json
{
  "queries_per_second": 45.2,
  "average_response_time": 45.7,
  "error_rate": 0.005,
  "cache_hit_rate": 0.78,
  "graph_complexity": 0.65
}
```

## Rate Limiting

The Knowledge Service API implements rate limiting to ensure fair usage:
- 500 requests per minute per API key
- 5000 requests per hour per API key

Exceeding rate limits will result in a 429 (Too Many Requests) response.