"""
Embeddings Manager for Ravana AGI System
Handles local embedding models with optimized performance and resource management
"""
import logging
import numpy as np
from typing import List, Union, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelPurpose(Enum):
    """Enumeration of different purposes for embedding models."""
    GENERAL = "general"
    SEMANTIC_SEARCH = "semantic_search"
    TEXT_CLASSIFICATION = "text_classification"
    SENTENCE_SIMILARITY = "sentence_similarity"
    CLUSTERING = "clustering"
    MULTILINGUAL = "multilingual"
    PERFORMANCE_CRITICAL = "performance_critical"
    QUALITY_CRITICAL = "quality_critical"


@dataclass
class ModelSpec:
    """Specification for an embedding model."""
    name: str
    purpose: ModelPurpose
    dimension: int
    speed_score: float  # Higher is faster (0-1 scale)
    quality_score: float  # Higher is better quality (0-1 scale)
    memory_usage: str  # 'low', 'medium', 'high'
    multilingual: bool


class IntelligentModelSelector:
    """Intelligently selects the best embedding model based on context and requirements."""

    def __init__(self):
        self.model_specs = {
            'all-MiniLM-L6-v2': ModelSpec(
                name='all-MiniLM-L6-v2',
                purpose=ModelPurpose.GENERAL,
                dimension=384,
                speed_score=0.9,
                quality_score=0.7,
                memory_usage='low',
                multilingual=False
            ),
            'all-MiniLM-L12-v2': ModelSpec(
                name='all-MiniLM-L12-v2',
                purpose=ModelPurpose.PERFORMANCE_CRITICAL,
                dimension=384,
                speed_score=0.8,
                quality_score=0.75,
                memory_usage='medium',
                multilingual=False
            ),
            'all-mpnet-base-v2': ModelSpec(
                name='all-mpnet-base-v2',
                purpose=ModelPurpose.QUALITY_CRITICAL,
                dimension=768,
                speed_score=0.6,
                quality_score=0.95,
                memory_usage='high',
                multilingual=False
            ),
            'paraphrase-multilingual-MiniLM-L12-v2': ModelSpec(
                name='paraphrase-multilingual-MiniLM-L12-v2',
                purpose=ModelPurpose.MULTILINGUAL,
                dimension=768,
                speed_score=0.7,
                quality_score=0.8,
                memory_usage='medium',
                multilingual=True
            ),
            'paraphrase-MiniLM-L6-v2': ModelSpec(
                name='paraphrase-MiniLM-L6-v2',
                purpose=ModelPurpose.SENTENCE_SIMILARITY,
                dimension=384,
                speed_score=0.85,
                quality_score=0.75,
                memory_usage='low',
                multilingual=False
            ),
            'distiluse-base-multilingual-cased': ModelSpec(
                name='distiluse-base-multilingual-cased',
                purpose=ModelPurpose.MULTILINGUAL,
                dimension=512,
                speed_score=0.75,
                quality_score=0.85,
                memory_usage='medium',
                multilingual=True
            )
        }

    def analyze_text_content(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """Analyze text content to determine optimal model characteristics."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        content_analysis = {
            'multilingual': False,
            'avg_length': 0,
            'total_chars': 0,
            'num_texts': len(texts)
        }

        # Check for multilingual content (simplified - in practice this would use more robust detection)
        total_chars = 0
        for text in texts:
            total_chars += len(text)
            # Check for non-ASCII characters that might indicate non-English content
            if any(ord(char) > 127 for char in text):
                content_analysis['multilingual'] = True

        content_analysis['avg_length'] = total_chars / max(1, len(texts))
        content_analysis['total_chars'] = total_chars

        return content_analysis

    def select_best_model(self,
                          purpose: ModelPurpose,
                          text: Union[str, List[str]] = None,
                          performance_requirements: Dict[str, float] = None) -> str:
        """
        Select the best model based on purpose, content, and requirements.

        Args:
            purpose: The intended purpose of the embeddings
            text: Text content to analyze for model selection
            performance_requirements: Dict with keys like 'speed_requirement' (0-1), 'quality_requirement' (0-1)

        Returns:
            Best model name for the given requirements
        """
        # Analyze content if provided
        content_analysis = self.analyze_text_content(text) if text else {}

        # Start with models matching the primary purpose
        candidate_specs = [
            spec for spec in self.model_specs.values()
            if spec.purpose == purpose or spec.purpose == ModelPurpose.GENERAL
        ]

        # If multilingual content detected, prefer multilingual models
        if content_analysis.get('multilingual', False):
            multilingual_specs = [
                spec for spec in candidate_specs if spec.multilingual]
            if multilingual_specs:
                candidate_specs = multilingual_specs

        # Apply performance requirements if provided
        if performance_requirements:
            min_speed = performance_requirements.get('speed_requirement', 0)
            min_quality = performance_requirements.get(
                'quality_requirement', 0)

            candidate_specs = [
                spec for spec in candidate_specs
                if spec.speed_score >= min_speed and spec.quality_score >= min_quality
            ]

        # If no candidates left, fall back to general models
        if not candidate_specs:
            candidate_specs = list(self.model_specs.values())

        # Score candidates by relevance to purpose and content
        best_model = None
        best_score = -1

        for spec in candidate_specs:
            score = 0

            # Higher score for purpose match
            if spec.purpose == purpose:
                score += 50
            elif spec.purpose == ModelPurpose.GENERAL:
                score += 10

            # Adjust score based on multilingual requirement
            if content_analysis.get('multilingual', False) and spec.multilingual:
                score += 30
            elif not content_analysis.get('multilingual', False) and not spec.multilingual:
                # Prefer non-multilingual for English content
                score += 10

            # Balance of speed and quality based on content length
            content_length = content_analysis.get('total_chars', 0)
            if content_length > 10000:  # Long content - prioritize speed
                score += spec.speed_score * 20
                score += spec.quality_score * 10
            else:  # Short content - balance both
                score += spec.speed_score * 15
                score += spec.quality_score * 15

            if score > best_score:
                best_score = score
                best_model = spec.name

        # Return the best model or fall back to default
        return best_model or 'all-MiniLM-L6-v2'


class EmbeddingsManager:
    """Centralized manager for embedding models with AI-driven model selection and fallback mechanisms."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingsManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not hasattr(self, '_initialized'):
            self._current_model = None
            self._current_model_name = None
            self._model_cache = {}  # Cache loaded models
            self._model_selector = IntelligentModelSelector()
            self._initialized = True
            self._fallback_chain = [
                'all-mpnet-base-v2',  # High quality first
                'all-MiniLM-L12-v2',  # Good performance
                'all-MiniLM-L6-v2',   # Reliable baseline
                'paraphrase-MiniLM-L6-v2',  # Specialized for similarity
            ]

    def _load_model(self, model_name: str, device: str = None) -> SentenceTransformer:
        """Load a model, either from cache or by downloading."""
        if model_name in self._model_cache:
            logger.debug(f"Using cached model: {model_name}")
            return self._model_cache[model_name]

        # Determine device automatically if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        try:
            logger.info(
                f"Loading embedding model: {model_name} on device: {device}")
            model = SentenceTransformer(model_name, device=device)

            # Cache the model
            self._model_cache[model_name] = model
            logger.debug(f"Model {model_name} loaded and cached")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None

    def _ensure_model_loaded(self, model_name: str, device: str = None) -> bool:
        """Ensure the specified model is loaded."""
        if model_name not in self._model_cache:
            model = self._load_model(model_name, device)
            if model is None:
                return False
        return True

    def select_and_load_model(self,
                              purpose: ModelPurpose,
                              text: Union[str, List[str]] = None,
                              performance_requirements: Dict[str,
                                                             float] = None,
                              fallback_to_general: bool = True) -> bool:
        """
        Select and load the best embedding model based on AI-driven analysis.

        Args:
            purpose: The intended purpose of the embeddings
            text: Text content to analyze for model selection
            performance_requirements: Dict with performance requirements
            fallback_to_general: Whether to fall back to general purpose if specific model fails

        Returns:
            True if a model was successfully loaded, False otherwise
        """
        # Select the best model based on purpose and content
        target_model = self._model_selector.select_best_model(
            purpose, text, performance_requirements
        )

        logger.info(
            f"Selected model '{target_model}' for purpose: {purpose.value}")

        # Only reload if different from current model
        if self._current_model_name != target_model:
            # Try to load the target model
            if self._load_model(target_model):
                self._current_model = self._model_cache[target_model]
                self._current_model_name = target_model
                logger.info(f"Successfully loaded model: {target_model}")
                return True
            else:
                logger.warning(
                    f"Failed to load selected model: {target_model}")

                # Try fallback models
                if fallback_to_general:
                    logger.info("Trying fallback models...")
                    for fallback_model in self._fallback_chain:
                        if fallback_model != target_model:
                            if self._load_model(fallback_model):
                                self._current_model = self._model_cache[fallback_model]
                                self._current_model_name = fallback_model
                                logger.info(
                                    f"Loaded fallback model: {fallback_model}")
                                return True

                logger.error("All model loading attempts failed")
                return False
        else:
            logger.debug(
                f"Model {target_model} already loaded, using existing instance")
            return True

    def get_embedding(self,
                      text: Union[str, List[str]],
                      purpose: ModelPurpose = ModelPurpose.GENERAL,
                      normalize: bool = True,
                      batch_size: int = 32,
                      performance_requirements: Dict[str, float] = None) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings with AI-driven model selection.

        Args:
            text: Input text or list of texts
            purpose: Purpose for which embeddings are needed
            normalize: Whether to normalize embeddings
            batch_size: Batch size for processing multiple texts
            performance_requirements: Dict with performance requirements

        Returns:
            Embedding vector(s) as numpy array(s)
        """
        # Select and load the appropriate model
        if not self.select_and_load_model(purpose, text, performance_requirements):
            raise RuntimeError("Failed to load any embedding model")

        # Ensure current model is set
        if self._current_model is None:
            raise RuntimeError("No embedding model is loaded")

        # Ensure input is list for batch processing
        if isinstance(text, str):
            texts = [text]
            is_single = True
        else:
            texts = text
            is_single = False

        try:
            # Generate embeddings
            embeddings = self._current_model.encode(
                texts,
                normalize_embeddings=normalize,
                batch_size=batch_size,
                convert_to_numpy=True
            )

            # Return appropriate format
            if is_single:
                return embeddings[0]  # Return single embedding as 1D array
            else:
                return embeddings  # Return multiple embeddings as 2D array

        except Exception as e:
            logger.error(
                f"Error generating embeddings with {self._current_model_name}: {e}")
            # Try with fallback model
            if self._current_model_name != 'all-MiniLM-L6-v2':
                logger.info("Falling back to all-MiniLM-L6-v2...")
                if self._load_model('all-MiniLM-L6-v2'):
                    self._current_model = self._model_cache['all-MiniLM-L6-v2']
                    self._current_model_name = 'all-MiniLM-L6-v2'

                    embeddings = self._current_model.encode(
                        texts,
                        normalize_embeddings=normalize,
                        batch_size=batch_size,
                        convert_to_numpy=True
                    )

                    if is_single:
                        return embeddings[0]
                    else:
                        return embeddings

            raise

    def get_similarity(self,
                       text1: str,
                       text2: str,
                       purpose: ModelPurpose = ModelPurpose.SENTENCE_SIMILARITY) -> float:
        """
        Calculate cosine similarity between two texts with intelligent model selection.

        Args:
            text1: First text
            text2: Second text
            purpose: Purpose for similarity calculation (defaults to SENTENCE_SIMILARITY)

        Returns:
            Cosine similarity score between 0 and 1
        """
        # Use the appropriate purpose for similarity calculation
        emb1 = self.get_embedding(text1, purpose=purpose)
        emb2 = self.get_embedding(text2, purpose=purpose)

        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / \
            (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

    def get_similar_texts(self,
                          query: str,
                          texts: List[str],
                          top_k: int = 5,
                          purpose: ModelPurpose = ModelPurpose.SEMANTIC_SEARCH) -> List[tuple]:
        """
        Find most similar texts to a query with intelligent model selection.

        Args:
            query: Query text
            texts: List of candidate texts
            top_k: Number of top similar texts to return
            purpose: Purpose for similarity search

        Returns:
            List of tuples (text, similarity_score) sorted by similarity
        """
        # Select and load appropriate model
        all_texts = [query] + texts
        if not self.select_and_load_model(purpose, all_texts):
            raise RuntimeError("Failed to load any embedding model")

        query_embedding = self.get_embedding(query)
        text_embeddings = self.get_embedding(texts)

        # Calculate similarities
        similarities = []
        for i, text_emb in enumerate(text_embeddings):
            similarity = np.dot(query_embedding, text_emb) / \
                (np.linalg.norm(query_embedding) * np.linalg.norm(text_emb))
            similarities.append((texts[i], float(similarity)))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def unload_all_models(self):
        """Unload all loaded models to free memory."""
        if self._model_cache:
            for model_name in list(self._model_cache.keys()):
                del self._model_cache[model_name]
            self._current_model = None
            self._current_model_name = None
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("All embedding models unloaded and memory freed")

    def unload_model(self, model_name: str):
        """Unload a specific model from cache."""
        if model_name in self._model_cache:
            del self._model_cache[model_name]
            if self._current_model_name == model_name:
                self._current_model = None
                self._current_model_name = None
            logger.info(f"Model {model_name} unloaded from cache")

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self._model_selector.model_specs.keys())

    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        if self._current_model_name and self._current_model_name in self._model_selector.model_specs:
            spec = self._model_selector.model_specs[self._current_model_name]
            return {
                'name': self._current_model_name,
                'purpose': spec.purpose.value,
                'dimension': spec.dimension,
                'speed_score': spec.speed_score,
                'quality_score': spec.quality_score,
                'memory_usage': spec.memory_usage,
                'multilingual': spec.multilingual
            }
        return {'name': self._current_model_name, 'info': 'Model loaded but not in spec'}


# Global instance for shared use
embeddings_manager = EmbeddingsManager()
