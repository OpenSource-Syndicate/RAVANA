# Data Service API

## Overview

The Data Service API provides programmatic access to data storage and retrieval operations within RAVANA AGI. This API enables modules and external systems to store, query, and manage data through standardized interfaces.

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

## Data Operations

### Store Data

```
POST /api/v1/data/{collection}
```

Store data in a specific collection.

**Parameters:**
- `collection` (path): Name of the data collection
- `data` (body): JSON object containing data to store

**Request Body:**
```json
{
  "field1": "value1",
  "field2": "value2",
  "nested_object": {
    "field3": "value3"
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

### Retrieve Data

```
GET /api/v1/data/{collection}/{id}
```

Retrieve specific data by ID from a collection.

**Parameters:**
- `collection` (path): Name of the data collection
- `id` (path): Unique identifier of the data record

**Response:**
```json
{
  "id": "record_id",
  "data": {
    "field1": "value1",
    "field2": "value2"
  },
  "metadata": {
    "created": "2023-01-01T00:00:00Z",
    "updated": "2023-01-01T00:00:00Z"
  }
}
```

### Query Data

```
POST /api/v1/data/{collection}/query
```

Query data using flexible query parameters.

**Parameters:**
- `collection` (path): Name of the data collection

**Request Body:**
```json
{
  "filter": {
    "field1": "value1",
    "field2": {"$gt": 100}
  },
  "sort": [{"field": "timestamp", "order": "desc"}],
  "limit": 100,
  "offset": 0
}
```

**Response:**
```json
{
  "results": [
    {
      "id": "record_id1",
      "data": {"field1": "value1", "field2": 150},
      "metadata": {"created": "2023-01-01T00:00:00Z"}
    },
    {
      "id": "record_id2",
      "data": {"field1": "value1", "field2": 200},
      "metadata": {"created": "2023-01-02T00:00:00Z"}
    }
  ],
  "total": 2,
  "limit": 100,
  "offset": 0
}
```

### Update Data

```
PUT /api/v1/data/{collection}/{id}
```

Update existing data record.

**Parameters:**
- `collection` (path): Name of the data collection
- `id` (path): Unique identifier of the data record

**Request Body:**
```json
{
  "field1": "updated_value1",
  "new_field": "new_value"
}
```

**Response:**
```json
{
  "id": "record_id",
  "updated": "2023-01-01T00:00:00Z",
  "status": "success"
}
```

### Delete Data

```
DELETE /api/v1/data/{collection}/{id}
```

Delete data record by ID.

**Parameters:**
- `collection` (path): Name of the data collection
- `id` (path): Unique identifier of the data record

**Response:**
```json
{
  "id": "record_id",
  "deleted": "2023-01-01T00:00:00Z",
  "status": "success"
}
```

## Collection Management

### List Collections

```
GET /api/v1/data/collections
```

List all available data collections.

**Response:**
```json
{
  "collections": [
    {"name": "users", "count": 1250},
    {"name": "sessions", "count": 42},
    {"name": "logs", "count": 100000}
  ]
}
```

### Create Collection

```
POST /api/v1/data/collections
```

Create a new data collection.

**Request Body:**
```json
{
  "name": "new_collection",
  "schema": {
    "field1": "string",
    "field2": "integer",
    "field3": "datetime"
  }
}
```

**Response:**
```json
{
  "name": "new_collection",
  "created": "2023-01-01T00:00:00Z",
  "status": "success"
}
```

### Delete Collection

```
DELETE /api/v1/data/collections/{name}
```

Delete an entire collection and all its data.

**Parameters:**
- `name` (path): Name of the collection to delete

**Response:**
```json
{
  "name": "collection_name",
  "deleted": "2023-01-01T00:00:00Z",
  "status": "success"
}
```

## Batch Operations

### Batch Store

```
POST /api/v1/data/{collection}/batch
```

Store multiple records in a single request.

**Parameters:**
- `collection` (path): Name of the data collection

**Request Body:**
```json
{
  "records": [
    {"field1": "value1", "field2": "value2"},
    {"field1": "value3", "field2": "value4"}
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

### Batch Update

```
PUT /api/v1/data/{collection}/batch
```

Update multiple records in a single request.

**Parameters:**
- `collection` (path): Name of the data collection

**Request Body:**
```json
{
  "updates": [
    {"id": "id1", "data": {"field1": "new_value1"}},
    {"id": "id2", "data": {"field2": "new_value2"}}
  ]
}
```

**Response:**
```json
{
  "updated": 2,
  "status": "success"
}
```

## Search Operations

### Full-text Search

```
GET /api/v1/data/{collection}/search
```

Perform full-text search across collection data.

**Parameters:**
- `collection` (path): Name of the data collection
- `q` (query): Search query string
- `fields` (query): Comma-separated list of fields to search
- `limit` (query): Maximum number of results (default: 100)

**Response:**
```json
{
  "results": [
    {
      "id": "record_id",
      "data": {"field1": "matching value", "field2": "other data"},
      "score": 0.95
    }
  ],
  "total": 1,
  "query": "search terms"
}
```

## Performance and Monitoring

### Health Check

```
GET /api/v1/data/health
```

Check the health status of the data service.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": "24h",
  "connections": 5,
  "collections": 10
}
```

### Metrics

```
GET /api/v1/data/metrics
```

Retrieve performance metrics for the data service.

**Response:**
```json
{
  "requests_per_second": 125.5,
  "average_response_time": 15.2,
  "error_rate": 0.01,
  "storage_usage": "45.2GB",
  "cache_hit_rate": 0.85
}
```

## Rate Limiting

The Data Service API implements rate limiting to ensure fair usage:
- 1000 requests per minute per API key
- 10000 requests per hour per API key

Exceeding rate limits will result in a 429 (Too Many Requests) response.