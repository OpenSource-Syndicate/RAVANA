# Multi-modal Service API

## Overview

The Multi-modal Service API provides programmatic access to media processing and cross-modal analysis operations within RAVANA AGI. This API enables modules and external systems to process, analyze, and integrate multiple data types including text, images, and audio through standardized interfaces.

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

## Media Processing

### Process Text

```
POST /api/v1/media/text/process
```

Process and analyze text content.

**Request Body:**
```json
{
  "content": "Artificial intelligence is transforming the world through machine learning and deep neural networks.",
  "operations": ["sentiment_analysis", "entity_extraction", "summarization"],
  "language": "en"
}
```

**Response:**
```json
{
  "id": "text_processing_id",
  "content": "Artificial intelligence is transforming the world through machine learning and deep neural networks.",
  "analysis": {
    "sentiment": {
      "polarity": 0.7,
      "subjectivity": 0.6
    },
    "entities": [
      {
        "text": "Artificial intelligence",
        "type": "CONCEPT",
        "confidence": 0.95
      },
      {
        "text": "machine learning",
        "type": "TECHNOLOGY",
        "confidence": 0.92
      }
    ],
    "summary": "AI is changing the world via machine learning and neural networks."
  },
  "processing_time": 0.45
}
```

### Process Image

```
POST /api/v1/media/image/process
```

Process and analyze image content.

**Request Body:**
```json
{
  "image_data": "base64_encoded_image_data",
  "operations": ["object_detection", "scene_classification", "text_extraction"],
  "format": "jpeg"
}
```

**Response:**
```json
{
  "id": "image_processing_id",
  "analysis": {
    "objects": [
      {
        "label": "person",
        "confidence": 0.95,
        "bounding_box": {"x": 100, "y": 150, "width": 200, "height": 300}
      },
      {
        "label": "car",
        "confidence": 0.87,
        "bounding_box": {"x": 300, "y": 200, "width": 150, "height": 100}
      }
    ],
    "scene": {
      "type": "urban_street",
      "confidence": 0.89
    },
    "text": "Street Sign: Main Street",
    "colors": ["#3366CC", "#FFFFFF", "#000000"]
  },
  "processing_time": 1.25
}
```

### Process Audio

```
POST /api/v1/media/audio/process
```

Process and analyze audio content.

**Request Body:**
```json
{
  "audio_data": "base64_encoded_audio_data",
  "operations": ["speech_recognition", "speaker_diarization", "emotion_analysis"],
  "format": "wav",
  "sample_rate": 16000
}
```

**Response:**
```json
{
  "id": "audio_processing_id",
  "analysis": {
    "transcript": "Hello, how are you doing today? I'm interested in learning about artificial intelligence.",
    "speakers": [
      {
        "id": "speaker_1",
        "segments": [
          {"start": 0.0, "end": 3.5, "text": "Hello, how are you doing today?"}
        ]
      },
      {
        "id": "speaker_2",
        "segments": [
          {"start": 3.5, "end": 7.2, "text": "I'm interested in learning about artificial intelligence."}
        ]
      }
    ],
    "emotions": [
      {"time": 1.0, "emotion": "happiness", "confidence": 0.85},
      {"time": 5.0, "emotion": "curiosity", "confidence": 0.92}
    ]
  },
  "processing_time": 2.75
}
```

## Cross-modal Operations

### Cross-modal Analysis

```
POST /api/v1/multimodal/analyze
```

Perform analysis across multiple media types.

**Request Body:**
```json
{
  "media": [
    {
      "type": "text",
      "content": "A beautiful sunset over the ocean"
    },
    {
      "type": "image",
      "image_data": "base64_encoded_image_data"
    }
  ],
  "analysis_type": "content_alignment",
  "operations": ["semantic_similarity", "concept_mapping"]
}
```

**Response:**
```json
{
  "id": "cross_modal_analysis_id",
  "results": {
    "semantic_similarity": 0.87,
    "concept_alignment": {
      "text_concepts": ["sunset", "ocean", "beautiful"],
      "image_concepts": ["sunset", "water", "horizon"],
      "common_concepts": ["sunset"],
      "alignment_score": 0.75
    },
    "cross_modal_insights": [
      "Both media sources reference a sunset scene",
      "Text emphasizes aesthetic qualities while image shows spatial relationships"
    ]
  },
  "processing_time": 3.15
}
```

### Multi-modal Fusion

```
POST /api/v1/multimodal/fuse
```

Fuse information from multiple modalities into a unified representation.

**Request Body:**
```json
{
  "inputs": [
    {
      "type": "text",
      "content": "Technical presentation about neural networks"
    },
    {
      "type": "image",
      "image_data": "base64_encoded_image_data"
    },
    {
      "type": "audio",
      "audio_data": "base64_encoded_audio_data"
    }
  ],
  "fusion_strategy": "semantic_integration"
}
```

**Response:**
```json
{
  "id": "fusion_id",
  "fused_representation": {
    "main_topic": "Neural Networks",
    "key_points": [
      "Architecture overview",
      "Training process",
      "Applications in computer vision"
    ],
    "presentation_style": "technical",
    "audience_engagement": "moderate",
    "confidence": 0.89
  },
  "modality_contributions": {
    "text": 0.4,
    "image": 0.35,
    "audio": 0.25
  },
  "processing_time": 4.25
}
```

## Media Storage and Retrieval

### Store Media

```
POST /api/v1/media
```

Store media content for later retrieval.

**Request Body:**
```json
{
  "type": "image",
  "content": "base64_encoded_image_data",
  "metadata": {
    "source": "user_upload",
    "context": "educational_material",
    "tags": ["machine_learning", "neural_networks", "diagram"]
  },
  "format": "png"
}
```

**Response:**
```json
{
  "id": "media_id",
  "type": "image",
  "stored": "2023-01-01T10:30:00Z",
  "size": "2.5MB",
  "status": "success"
}
```

### Retrieve Media

```
GET /api/v1/media/{id}
```

Retrieve stored media content.

**Parameters:**
- `id` (path): Unique identifier of the media record

**Response:**
```json
{
  "id": "media_id",
  "type": "image",
  "content": "base64_encoded_image_data",
  "metadata": {
    "source": "user_upload",
    "context": "educational_material",
    "tags": ["machine_learning", "neural_networks", "diagram"],
    "created": "2023-01-01T10:30:00Z"
  },
  "format": "png",
  "size": "2.5MB"
}
```

### Search Media

```
GET /api/v1/media/search
```

Search for media based on metadata and content analysis.

**Parameters:**
- `query` (query): Search query terms
- `type` (query): Media type filter (text/image/audio)
- `tags` (query): Comma-separated list of tags
- `limit` (query): Maximum number of results (default: 50)

**Response:**
```json
{
  "results": [
    {
      "id": "media_id1",
      "type": "image",
      "preview": "base64_encoded_thumbnail",
      "tags": ["machine_learning", "neural_networks"],
      "relevance": 0.92
    },
    {
      "id": "media_id2",
      "type": "text",
      "preview": "First few sentences of text content...",
      "tags": ["machine_learning", "tutorial"],
      "relevance": 0.85
    }
  ],
  "total": 2,
  "query": "machine learning"
}
```

## Media Transformation

### Convert Media Format

```
POST /api/v1/media/convert
```

Convert media from one format to another.

**Request Body:**
```json
{
  "source_id": "media_id",
  "target_format": "jpeg",
  "quality": 0.85,
  "resize": {
    "width": 800,
    "height": 600
  }
}
```

**Response:**
```json
{
  "id": "converted_media_id",
  "type": "image",
  "format": "jpeg",
  "size": "1.2MB",
  "converted": "2023-01-01T11:00:00Z"
}
```

### Extract Features

```
POST /api/v1/media/extract
```

Extract specific features from media content.

**Request Body:**
```json
{
  "media_id": "media_id",
  "features": ["color_palette", "dominant_objects", "key_phrases"],
  "parameters": {
    "color_count": 5,
    "object_confidence_threshold": 0.8
  }
}
```

**Response:**
```json
{
  "id": "extraction_id",
  "features": {
    "color_palette": ["#FF0000", "#00FF00", "#0000FF"],
    "dominant_objects": [
      {"label": "person", "confidence": 0.95},
      {"label": "car", "confidence": 0.87}
    ],
    "key_phrases": ["sunset", "ocean", "horizon"]
  },
  "processing_time": 1.75
}
```

## Batch Operations

### Batch Process Media

```
POST /api/v1/media/batch/process
```

Process multiple media items in a single request.

**Request Body:**
```json
{
  "items": [
    {
      "type": "text",
      "content": "Text content 1"
    },
    {
      "type": "image",
      "image_data": "base64_encoded_image_data"
    }
  ],
  "operations": ["sentiment_analysis", "object_detection"]
}
```

**Response:**
```json
{
  "processed": 2,
  "results": [
    {
      "id": "result_id1",
      "type": "text",
      "analysis": {"sentiment": 0.7}
    },
    {
      "id": "result_id2",
      "type": "image",
      "analysis": {"objects": ["person", "car"]}
    }
  ],
  "status": "success"
}
```

## Performance and Monitoring

### Health Check

```
GET /api/v1/media/health
```

Check the health status of the multi-modal service.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": "24h",
  "processors": {
    "text": "healthy",
    "image": "healthy",
    "audio": "healthy"
  },
  "storage": "15.2GB/50GB"
}
```

### Metrics

```
GET /api/v1/media/metrics
```

Retrieve performance metrics for the multi-modal service.

**Response:**
```json
{
  "processing_rate": {
    "text": 125.5,
    "image": 45.2,
    "audio": 25.7
  },
  "average_processing_time": {
    "text": 0.35,
    "image": 1.85,
    "audio": 3.25
  },
  "error_rate": 0.003,
  "storage_usage": "15.2GB/50GB",
  "cache_hit_rate": 0.72
}
```

## Supported Formats and Limitations

### Text Processing
- Supported formats: Plain text, JSON, XML
- Maximum size: 1MB
- Supported languages: en, es, fr, de, it, pt, ru, zh, ja, ko

### Image Processing
- Supported formats: JPEG, PNG, GIF, BMP, TIFF
- Maximum resolution: 4K (3840×2160)
- Maximum size: 10MB

### Audio Processing
- Supported formats: WAV, MP3, FLAC, OGG
- Maximum duration: 30 minutes
- Maximum size: 50MB

## Rate Limiting

The Multi-modal Service API implements rate limiting to ensure fair usage:
- 200 requests per minute per API key
- 2000 requests per hour per API key

Exceeding rate limits will result in a 429 (Too Many Requests) response.