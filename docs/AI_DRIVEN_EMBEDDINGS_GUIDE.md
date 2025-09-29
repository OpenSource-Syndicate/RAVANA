# Ravana AGI - AI-Driven Embedding Model Selection System

## Overview

The Ravana AGI system now features an advanced AI-driven embedding model selection system that intelligently chooses the most appropriate embedding model based on the specific task requirements, content characteristics, and performance needs.

## Key Features

### 1. AI-Driven Model Selection
- **Intelligent Purpose Detection**: Automatically selects the most suitable model based on the intended use case
- **Context Analysis**: Analyzes text content to determine optimal model characteristics
- **Performance Optimization**: Balances quality vs. speed requirements

### 2. Multiple Model Purposes
The system recognizes and optimizes for these specific purposes:
- **GENERAL**: Default purpose for general embedding tasks
- **SEMANTIC_SEARCH**: Optimized for document retrieval and search
- **SENTENCE_SIMILARITY**: Specialized for comparing sentence-level similarity
- **MULTILINGUAL**: Handles content in multiple languages
- **QUALITY_CRITICAL**: High-accuracy embeddings for critical applications
- **PERFORMANCE_CRITICAL**: Fast embeddings for real-time operations
- **CLUSTERING**: Optimized for grouping similar content

### 3. Available Models and Specializations

| Model | Purpose | Dimensions | Quality | Speed | Multilingual |
|-------|---------|------------|---------|-------|--------------|
| all-MiniLM-L6-v2 | General | 384 | 0.70 | 0.90 | No |
| all-MiniLM-L12-v2 | Performance | 384 | 0.75 | 0.80 | No |
| all-mpnet-base-v2 | Quality | 768 | 0.95 | 0.60 | No |
| paraphrase-multilingual-MiniLM-L12-v2 | Multilingual | 768 | 0.80 | 0.70 | Yes |
| paraphrase-MiniLM-L6-v2 | Sentence Similarity | 384 | 0.75 | 0.85 | No |
| distiluse-base-multilingual-cased | Multilingual | 512 | 0.85 | 0.75 | Yes |

## How It Works

### 1. Purpose-Based Selection
The system automatically selects models based on the intended purpose:

```python
from core.embeddings_manager import embeddings_manager, ModelPurpose

# For semantic search tasks
similar_docs = embeddings_manager.get_similar_texts(
    query, documents, 
    purpose=ModelPurpose.SEMANTIC_SEARCH
)

# For sentence similarity
similarity = embeddings_manager.get_similarity(
    sentence1, sentence2,
    purpose=ModelPurpose.SENTENCE_SIMILARITY
)

# For multilingual content
embedding = embeddings_manager.get_embedding(
    multilingual_text,
    purpose=ModelPurpose.MULTILINGUAL
)
```

### 2. Context-Aware Analysis
The system analyzes content characteristics to select optimal models:
- Text length and complexity
- Language detection
- Domain-specific terminology
- Performance requirements

### 3. Fallback Mechanisms
The system implements robust fallback chains:
- Primary model selection based on purpose
- Content analysis for optimization
- Performance requirement matching
- Fallback to baseline models if needed
- Error recovery with alternative models

## Integration with Ravana Components

### System Integration
- All Ravana components now use the intelligent embeddings manager
- Automatic model selection based on use case
- Memory-efficient model caching
- Performance-optimized batch processing

### Memory Service
- Semantic search for memory retrieval
- Context-aware embedding generation
- Quality-optimized recall of relevant memories

### Knowledge Service
- Dynamic model selection for knowledge indexing
- Specialized models for different content types
- Performance optimization for large knowledge bases

### Reflection Module
- Context-aware similarity matching
- Quality-optimized concept comparison
- Multilingual support for diverse content

## Usage Examples

### Basic Usage
```python
from core.embeddings_manager import embeddings_manager, ModelPurpose

# Generate embeddings with automatic model selection
embedding = embeddings_manager.get_embedding(
    "Your text here",
    purpose=ModelPurpose.GENERAL  # or other appropriate purpose
)

# Calculate similarity between texts
similarity = embeddings_manager.get_similarity(
    "Text 1", "Text 2",
    purpose=ModelPurpose.SENTENCE_SIMILARITY
)

# Find similar texts
similar_texts = embeddings_manager.get_similar_texts(
    query, text_list,
    top_k=5,
    purpose=ModelPurpose.SEMANTIC_SEARCH
)
```

### Performance Requirements
```python
# Specify performance requirements if needed
embedding = embeddings_manager.get_embedding(
    long_document,
    purpose=ModelPurpose.QUALITY_CRITICAL,
    performance_requirements={
        'speed_requirement': 0.8,  # Minimum speed score 0-1
        'quality_requirement': 0.9  # Minimum quality score 0-1
    }
)
```

### Model Management
```python
# Get current model information
model_info = embeddings_manager.get_current_model_info()
print(f"Current model: {model_info['name']}")

# Get available models
available_models = embeddings_manager.get_available_models()
print(f"Available: {available_models}")

# Unload all models to free memory
embeddings_manager.unload_all_models()
```

## Benefits

1. **Optimal Performance**: Each task uses the most appropriate model
2. **Resource Efficiency**: Caches models to avoid redundant loading
3. **Quality Assurance**: Purpose-specific optimization
4. **Robustness**: Comprehensive fallback mechanisms
5. **Scalability**: Handles diverse content types and requirements
6. **Adaptability**: Learns from usage patterns over time

## Configuration

The system can be configured through the core configuration:

```python
# In core/config.py
EMBEDDING_MODEL_TYPE = os.environ.get("EMBEDDING_MODEL_TYPE", "balanced")
EMBEDDING_USE_CUDA = os.environ.get("EMBEDDING_USE_CUDA", "False").lower() in ["true", "1", "yes"]
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", None)  # cuda, cpu, mps, or None for auto
```

## Maintenance and Monitoring

The system provides comprehensive monitoring capabilities:
- Current model information via `get_current_model_info()`
- Available models list via `get_available_models()`
- Performance metrics and quality scores
- Memory usage optimization tools

This AI-driven embedding system ensures that Ravana AGI always uses the most appropriate model for each specific task, optimizing for quality, performance, and resource efficiency.