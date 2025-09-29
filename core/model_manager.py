import logging
from typing import Optional
from transformers import pipeline

from core.embeddings_manager import embeddings_manager, ModelPurpose

logger = logging.getLogger(__name__)


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
                self._sentiment_classifier = pipeline('sentiment-analysis')
                logger.info("✅ Sentiment classifier loaded successfully")

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
