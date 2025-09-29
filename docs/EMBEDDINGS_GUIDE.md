# Ravana AGI - Local Embedding Models Guide

This document explains how to use and configure local embedding models in the Ravana AGI system.

## Overview

The Ravana AGI system now uses a sophisticated local embedding system based on Sentence Transformers. The system is managed through the `EmbeddingsManager` class which provides:

- Multiple model options optimized for different needs
- Automatic device selection (CPU, CUDA, MPS)
- Memory efficient loading and unloading
- Performance optimized batch processing
- Similarity calculations and text matching

## Available Embedding Models

The system offers several embedding models optimized for different use cases:

### Model Types:
1. **balanced** (`all-MiniLM-L6-v2`) - Good balance of speed and quality (default)
2. **quality** (`all-mpnet-base-v2`) - Highest quality embeddings
3. **performance** (`all-MiniLM-L12-v2`) - Good performance and quality
4. **multilingual** (`paraphrase-multilingual-MiniLM-L12-v2`) - For multilingual content
5. **large** (`sentence-transformers/all-mpnet-base-v2`) - Highest quality for complex tasks

### Configuration Options:

You can configure the embedding model through environment variables or directly in code:

```python
# In core/config.py, you can set:

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # Specific model name
EMBEDDING_MODEL_TYPE = os.environ.get("EMBEDDING_MODEL_TYPE", "balanced")  # Model type preference
EMBEDDING_USE_CUDA = os.environ.get("EMBEDDING_USE_CUDA", "False")  # Use CUDA if available
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", None)  # Explicit device (cuda, cpu, mps)
```

## Using the Embeddings Manager

### Basic Usage:

```python
from core.embeddings_manager import embeddings_manager

# Initialize the model (optional - done automatically on first use)
embeddings_manager.initialize_model(preference='quality')

# Generate embeddings for single text
embedding = embeddings_manager.get_embedding("Your text here")

# Generate embeddings for multiple texts
texts = ["text1", "text2", "text3"]
embeddings = embeddings_manager.get_embedding(texts)

# Calculate similarity between two texts
similarity = embeddings_manager.get_similarity("text1", "text2")

# Find similar texts
similar_texts = embeddings_manager.get_similar_texts(
    query="query text", 
    texts=["candidate1", "candidate2", "candidate3"], 
    top_k=2
)
```

## Model Initialization in Ravana System

The main Ravana system is automatically configured to use the enhanced embeddings manager. The configuration is loaded from the `Config` class.

## Performance Tips

1. **Model Selection**: Use 'balanced' for general purpose, 'performance' for speed, 'quality' for accuracy
2. **Batch Processing**: Process multiple texts together for better performance
3. **Device Selection**: Use CUDA when available for faster processing
4. **Memory Management**: The system automatically manages memory and can unload models when not needed

## Integration with Existing Code

The embeddings manager is fully integrated with:
- Memory service for knowledge retrieval
- Data service for content analysis
- Reflection module for context matching
- Knowledge service for concept similarity

## Environment Variables

You can customize the embedding behavior with these environment variables:

```bash
# Set the embedding model type
export EMBEDDING_MODEL_TYPE=quality

# Enable CUDA usage if available
export EMBEDDING_USE_CUDA=true

# Set specific device
export EMBEDDING_DEVICE=cuda

# Set specific model (overrides type)
export EMBEDDING_MODEL=all-mpnet-base-v2
```

## Best Practices

1. Initialize the embeddings manager early in your application lifecycle
2. Use the same instance across the application to avoid redundant loading
3. Choose the appropriate model type based on your performance/quality requirements
4. Monitor memory usage, especially with larger models
5. Use batch processing when possible for better performance