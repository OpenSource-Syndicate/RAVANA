#!/usr/bin/env python3
"""
Test Sentence Transformers Module
This script tests the sentence_transformers module to diagnose any issues.
"""

import os
import sys
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentence_transformers_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SentenceTransformersTest")

def test_import():
    """Test if the sentence_transformers module can be imported."""
    logger.info("Testing import of sentence_transformers...")
    
    try:
        import sentence_transformers
        logger.info(f"sentence_transformers imported successfully. Version: {sentence_transformers.__version__}")
        return True
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Attempting to install sentence_transformers...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
            import sentence_transformers
            logger.info(f"sentence_transformers installed and imported successfully. Version: {sentence_transformers.__version__}")
            return True
        except Exception as e:
            logger.error(f"Failed to install sentence_transformers: {e}")
            return False
    except Exception as e:
        logger.error(f"Error testing import: {e}")
        return False

def test_model_loading():
    """Test if a sentence transformer model can be loaded."""
    logger.info("Testing model loading...")
    
    try:
        import sentence_transformers
        
        # Try to load a small model first
        logger.info("Loading a small sentence transformer model...")
        start_time = time.time()
        model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        end_time = time.time()
        
        logger.info(f"Model loaded successfully in {end_time - start_time:.2f} seconds")
        
        # Check the device
        device = model.device
        logger.info(f"Model is using device: {device}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def test_embedding_generation():
    """Test if embeddings can be generated."""
    logger.info("Testing embedding generation...")
    
    try:
        import sentence_transformers
        import numpy as np
        
        # Load a small model
        model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings for a simple sentence
        sentences = ["This is a test sentence."]
        logger.info(f"Generating embeddings for: {sentences}")
        
        start_time = time.time()
        embeddings = model.encode(sentences)
        end_time = time.time()
        
        logger.info(f"Embeddings generated successfully in {end_time - start_time:.2f} seconds")
        logger.info(f"Embedding shape: {embeddings.shape}")
        logger.info(f"Embedding sample (first 5 values): {embeddings[0][:5]}")
        
        return True
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return False

def test_similarity_calculation():
    """Test if similarity between sentences can be calculated."""
    logger.info("Testing similarity calculation...")
    
    try:
        import sentence_transformers
        import numpy as np
        from sentence_transformers.util import cos_sim
        
        # Load a small model
        model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings for two sentences
        sentences = ["This is a test sentence.", "This is a similar test sentence."]
        logger.info(f"Calculating similarity between: {sentences}")
        
        embeddings = model.encode(sentences)
        similarity = cos_sim(embeddings[0], embeddings[1])
        
        logger.info(f"Similarity calculated successfully: {similarity.item():.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return False

def test_gpu_availability():
    """Test if GPU is available for sentence_transformers."""
    logger.info("Testing GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Found {torch.cuda.device_count()} device(s).")
            for i in range(torch.cuda.device_count()):
                logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            logger.info("CUDA is not available. Using CPU.")
            return False
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting sentence_transformers module tests")
    
    # Test import
    import_ok = test_import()
    logger.info(f"Import test: {'PASSED' if import_ok else 'FAILED'}")
    
    if not import_ok:
        logger.error("Cannot proceed with tests as sentence_transformers could not be imported")
        return 1
    
    # Test GPU availability
    gpu_available = test_gpu_availability()
    logger.info(f"GPU availability test: {'GPU AVAILABLE' if gpu_available else 'USING CPU'}")
    
    # Test model loading
    model_ok = test_model_loading()
    logger.info(f"Model loading test: {'PASSED' if model_ok else 'FAILED'}")
    
    if not model_ok:
        logger.error("Cannot proceed with remaining tests as model could not be loaded")
        return 1
    
    # Test embedding generation
    embedding_ok = test_embedding_generation()
    logger.info(f"Embedding generation test: {'PASSED' if embedding_ok else 'FAILED'}")
    
    # Test similarity calculation
    similarity_ok = test_similarity_calculation()
    logger.info(f"Similarity calculation test: {'PASSED' if similarity_ok else 'FAILED'}")
    
    # Overall result
    if import_ok and model_ok and embedding_ok and similarity_ok:
        logger.info("All tests PASSED")
        print("\n✅ sentence_transformers module is working correctly!")
        return 0
    else:
        logger.error("Some tests FAILED")
        print("\n❌ sentence_transformers module has issues. Check the log for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 