import logging
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline

logger = logging.getLogger(__name__)

class ModelManager:
    """Centralized model manager to prevent redundant model loading."""
    
    _instance = None
    _embedding_model = None
    _sentiment_classifier = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def initialize_models(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize all models with proper error handling."""
        if self._initialized:
            logger.info("Models already initialized, skipping initialization")
            return
        
        try:
            # Initialize embedding model
            if self._embedding_model is None:
                logger.info(f"Loading embedding model: {embedding_model_name}")
                self._embedding_model = SentenceTransformer(embedding_model_name)
                logger.info(f"✅ Embedding model '{embedding_model_name}' loaded successfully")
            
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
    def embedding_model(self) -> Optional[SentenceTransformer]:
        """Get the embedding model instance."""
        return self._embedding_model
    
    @property
    def sentiment_classifier(self) -> Optional:
        """Get the sentiment classifier instance."""
        return self._sentiment_classifier

# Global instance
model_manager = ModelManager()