import logging
from typing import Optional
import torch
from transformers import pipeline

from core.embeddings_manager import embeddings_manager, ModelPurpose

logger = logging.getLogger(__name__)


def create_sentiment_classifier():
    """
    Creates a sentiment classifier pipeline with proper error handling for meta tensor issues.
    This function addresses the 'NotImplementedError: Cannot copy out of meta tensor' error
    by explicitly managing the device assignment instead of using device_map='auto'.
    """
    try:
        # Determine the appropriate device
        device = 0 if torch.cuda.is_available() else -1
        
        # Create the pipeline without device_map to avoid meta tensor issues
        classifier = pipeline(
            'sentiment-analysis',
            model='cardiffnlp/twitter-roberta-base-sentiment-latest',
            device=device
        )
        
        logger.info("✅ Sentiment classifier loaded successfully")
        return classifier
    except Exception as e:
        logger.error(f"❌ Failed to create sentiment classifier: {e}")
        # Fallback: try with CPU if GPU fails
        try:
            logger.info("Attempting fallback to CPU device...")
            classifier = pipeline(
                'sentiment-analysis',
                model='cardiffnlp/twitter-roberta-base-sentiment-latest',
                device=-1  # CPU
            )
            logger.info("✅ Sentiment classifier loaded successfully on CPU as fallback")
            return classifier
        except Exception as fallback_error:
            logger.error(f"❌ Fallback also failed: {fallback_error}")
            raise


class ModelManager:
    """Centralized model manager to prevent redundant model loading."""

    _instance = None
    _sentiment_classifier = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def initialize_models(self):
        """Initialize all models with proper error handling."""
        if self._initialized:
            logger.info("Models already initialized, skipping initialization")
            return

        try:
            # Initialize sentiment classifier
            if self._sentiment_classifier is None:
                logger.info("Loading sentiment classifier")
                self._sentiment_classifier = create_sentiment_classifier()

            self._initialized = True
            logger.info("✅ All models initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize models: {e}")
            raise

    @property
    def embedding_model(self):
        """Get the embedding model instance from embeddings manager."""
        return embeddings_manager

    @property
    def sentiment_classifier(self) -> Optional:
        """Get the sentiment classifier instance."""
        return self._sentiment_classifier


# Global instance
model_manager = ModelManager()
