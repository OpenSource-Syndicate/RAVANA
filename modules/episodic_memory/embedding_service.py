"""
Enhanced EmbeddingService for multi-modal memory system.
Handles text, image, audio, and unified embeddings for cross-modal retrieval.
"""

import logging
import asyncio
import hashlib
import numpy as np
from typing import Dict, Any, Optional, List
import time

try:
    from sentence_transformers import SentenceTransformer
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    torch = None
    transforms = None
    Image = None
    logging.warning(f"Transformers dependencies not available: {e}")

from .models import MemoryRecord, ContentType
from .whisper_processor import WhisperAudioProcessor

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Simple in-memory cache for embeddings to avoid recomputation."""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}

    def _get_key(self, content: str, model_name: str) -> str:
        """Generate cache key from content and model."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{model_name}:{content_hash}"

    def get(self, content: str, model_name: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        key = self._get_key(content, model_name)
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def put(self, content: str, model_name: str, embedding: List[float]):
        """Store embedding in cache."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.access_times.keys(),
                             key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        key = self._get_key(content, model_name)
        self.cache[key] = embedding
        self.access_times[key] = time.time()

    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_times.clear()


class EmbeddingService:
    """
    Enhanced embedding service supporting text, image, audio, and unified embeddings.
    """

    def __init__(self,
                 text_model_name: str = "all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 cache_size: int = 1000):
        """
        Initialize the embedding service.

        Args:
            text_model_name: Name of the sentence transformer model
            device: Device to use ("cpu", "cuda", "auto")
            cache_size: Size of embedding cache
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers dependencies not available. Install with: pip install sentence-transformers torch torchvision Pillow")

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.text_model_name = text_model_name
        self.cache = EmbeddingCache(cache_size)

        # Model instances (lazy loaded)
        self.text_model = None
        self.whisper_processor = None

        # Embedding dimensions
        self.text_embedding_dim = 384   # all-MiniLM-L6-v2
        self.image_embedding_dim = 512  # CLIP ViT-B/32
        self.audio_embedding_dim = 512  # Audio features -> compressed
        self.unified_embedding_dim = 1024  # Combined embeddings

        logger.info(f"Initialized EmbeddingService with device={self.device}")

    def _load_text_model(self):
        """Lazy load text embedding model."""
        if self.text_model is None:
            try:
                logger.info(f"Loading text model: {self.text_model_name}")
                self.text_model = SentenceTransformer(
                    self.text_model_name, device=self.device)
                logger.info(f"Text model loaded successfully on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load text model: {e}")
                raise

    def _load_whisper_processor(self):
        """Lazy load Whisper processor for audio."""
        if self.whisper_processor is None:
            try:
                self.whisper_processor = WhisperAudioProcessor(
                    device=self.device)
                logger.info("Whisper processor loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper processor: {e}")
                raise

    async def generate_text_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text content.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        if not text or not text.strip():
            return [0.0] * self.text_embedding_dim

        # Check cache first
        cached = self.cache.get(text, self.text_model_name)
        if cached is not None:
            return cached

        self._load_text_model()

        try:
            # Generate embedding
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self.text_model.encode(
                    text, convert_to_tensor=False, normalize_embeddings=True)
            )

            # Convert to list and cache
            embedding_list = embedding.tolist()
            self.cache.put(text, self.text_model_name, embedding_list)

            return embedding_list

        except Exception as e:
            logger.error(f"Text embedding generation failed: {e}")
            return [0.0] * self.text_embedding_dim

    async def generate_image_embedding(self, image_path: str) -> List[float]:
        """
        Generate embedding for image content using CLIP.

        Args:
            image_path: Path to image file

        Returns:
            List of embedding values
        """
        try:
            # For now, we'll use a simplified approach
            # In production, you would use CLIP or similar models

            # Load and process image
            image = Image.open(image_path).convert('RGB')

            # Extract basic image features as a placeholder
            # This would be replaced with actual CLIP embeddings
            features = self._extract_basic_image_features(image)

            # Pad or truncate to desired dimension
            if len(features) < self.image_embedding_dim:
                features.extend(
                    [0.0] * (self.image_embedding_dim - len(features)))
            else:
                features = features[:self.image_embedding_dim]

            return features

        except Exception as e:
            logger.error(
                f"Image embedding generation failed for {image_path}: {e}")
            return [0.0] * self.image_embedding_dim

    def _extract_basic_image_features(self, image: Image.Image) -> List[float]:
        """
        Extract basic image features as placeholder for CLIP embeddings.
        In production, this would be replaced with actual CLIP model.
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)

            # Extract basic statistics
            features = []

            # Color statistics
            for channel in range(3):  # RGB
                channel_data = img_array[:, :, channel].flatten()
                features.extend([
                    float(np.mean(channel_data)),
                    float(np.std(channel_data)),
                    float(np.median(channel_data)),
                    float(np.percentile(channel_data, 25)),
                    float(np.percentile(channel_data, 75))
                ])

            # Image dimensions
            features.extend([
                float(image.width),
                float(image.height),
                float(image.width * image.height)  # Area
            ])

            # Histogram features (simplified)
            hist, _ = np.histogram(img_array.flatten(),
                                   bins=32, range=(0, 256))
            hist_normalized = hist / np.sum(hist)
            features.extend(hist_normalized.tolist())

            return features

        except Exception as e:
            logger.error(f"Basic image feature extraction failed: {e}")
            return [0.0] * 50  # Fallback

    async def generate_audio_embedding(self, audio_features: Dict[str, Any]) -> List[float]:
        """
        Generate embedding for audio content from extracted features.

        Args:
            audio_features: Audio features from WhisperAudioProcessor

        Returns:
            List of embedding values
        """
        try:
            features = []

            # Extract numerical features from audio analysis
            if "mfcc" in audio_features:
                mfcc_data = audio_features["mfcc"]
                if "mean" in mfcc_data:
                    features.extend(mfcc_data["mean"])
                if "std" in mfcc_data:
                    features.extend(mfcc_data["std"])

            if "spectral_centroid" in audio_features:
                sc = audio_features["spectral_centroid"]
                features.extend([sc.get("mean", 0.0), sc.get("std", 0.0)])

            if "zero_crossing_rate" in audio_features:
                zcr = audio_features["zero_crossing_rate"]
                features.extend([zcr.get("mean", 0.0), zcr.get("std", 0.0)])

            if "rms_energy" in audio_features:
                rms = audio_features["rms_energy"]
                features.extend([rms.get("mean", 0.0), rms.get("std", 0.0)])

            if "spectral_rolloff" in audio_features:
                rolloff = audio_features["spectral_rolloff"]
                features.extend(
                    [rolloff.get("mean", 0.0), rolloff.get("std", 0.0)])

            # Add scalar features
            features.append(audio_features.get("tempo", 0.0))
            features.append(audio_features.get("beat_count", 0.0))

            # Add chroma features if available
            if "chroma" in audio_features:
                chroma_data = audio_features["chroma"]
                if "mean" in chroma_data:
                    features.extend(chroma_data["mean"])

            # Pad or truncate to desired dimension
            if len(features) < self.audio_embedding_dim:
                features.extend(
                    [0.0] * (self.audio_embedding_dim - len(features)))
            else:
                features = features[:self.audio_embedding_dim]

            return features

        except Exception as e:
            logger.error(f"Audio embedding generation failed: {e}")
            return [0.0] * self.audio_embedding_dim

    async def generate_unified_embedding(self, memory_record: MemoryRecord) -> List[float]:
        """
        Generate unified embedding combining all available modalities.

        Args:
            memory_record: Memory record with various embeddings

        Returns:
            Unified embedding
        """
        try:
            unified = []

            # Combine embeddings with weights
            text_weight = 0.4
            image_weight = 0.3
            audio_weight = 0.3

            # Text embedding (weighted)
            if memory_record.text_embedding:
                text_emb = np.array(memory_record.text_embedding) * text_weight
                unified.extend(text_emb.tolist())
            else:
                unified.extend(
                    [0.0] * int(self.unified_embedding_dim * text_weight))

            # Image embedding (weighted)
            if memory_record.image_embedding:
                image_emb = np.array(
                    memory_record.image_embedding) * image_weight
                # Take first N elements to fit unified dimension
                unified.extend(
                    image_emb[:int(self.unified_embedding_dim * image_weight)].tolist())
            else:
                unified.extend(
                    [0.0] * int(self.unified_embedding_dim * image_weight))

            # Audio embedding (weighted)
            if memory_record.audio_embedding:
                audio_emb = np.array(
                    memory_record.audio_embedding) * audio_weight
                # Take first N elements to fit unified dimension
                unified.extend(
                    audio_emb[:int(self.unified_embedding_dim * audio_weight)].tolist())
            else:
                unified.extend(
                    [0.0] * int(self.unified_embedding_dim * audio_weight))

            # Ensure exact dimension
            if len(unified) < self.unified_embedding_dim:
                unified.extend(
                    [0.0] * (self.unified_embedding_dim - len(unified)))
            else:
                unified = unified[:self.unified_embedding_dim]

            # Normalize the unified embedding
            unified_array = np.array(unified)
            norm = np.linalg.norm(unified_array)
            if norm > 0:
                unified_array = unified_array / norm

            return unified_array.tolist()

        except Exception as e:
            logger.error(f"Unified embedding generation failed: {e}")
            return [0.0] * self.unified_embedding_dim

    async def generate_embeddings(self, memory_record: MemoryRecord) -> MemoryRecord:
        """
        Generate all relevant embeddings for a memory record.

        Args:
            memory_record: Memory record to process

        Returns:
            Memory record with embeddings populated
        """
        try:
            # Generate text embedding
            if memory_record.content_text:
                memory_record.text_embedding = await self.generate_text_embedding(
                    memory_record.content_text
                )

            # Generate content-type specific embeddings
            if memory_record.content_type == ContentType.IMAGE and memory_record.file_path:
                memory_record.image_embedding = await self.generate_image_embedding(
                    memory_record.file_path
                )

            elif memory_record.content_type == ContentType.AUDIO:
                if memory_record.audio_metadata and memory_record.audio_metadata.audio_features:
                    memory_record.audio_embedding = await self.generate_audio_embedding(
                        memory_record.audio_metadata.audio_features
                    )

                # Also generate text embedding from transcript
                if memory_record.audio_metadata and memory_record.audio_metadata.transcript:
                    memory_record.text_embedding = await self.generate_text_embedding(
                        memory_record.audio_metadata.transcript
                    )

            # Generate unified embedding
            memory_record.unified_embedding = await self.generate_unified_embedding(memory_record)

            logger.info(
                f"Generated embeddings for {memory_record.content_type} content")
            return memory_record

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        try:
            if not embedding1 or not embedding2:
                return 0.0

            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)

            # Compute cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)

            # Ensure result is in [0, 1] range
            return max(0.0, min(1.0, (similarity + 1) / 2))

        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0

    async def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch for efficiency.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        self._load_text_model()

        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return [[0.0] * self.text_embedding_dim] * len(texts)

            # Generate embeddings in batch
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.text_model.encode(
                    valid_texts, convert_to_tensor=False, normalize_embeddings=True)
            )

            # Convert to list format
            return [emb.tolist() for emb in embeddings]

        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return [[0.0] * self.text_embedding_dim] * len(texts)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        return {
            "cache_size": len(self.cache.cache),
            "max_size": self.cache.max_size,
            "hit_rate": getattr(self.cache, '_hits', 0) / max(1, getattr(self.cache, '_requests', 1))
        }

    def clear_cache(self):
        """Clear embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")

    def cleanup(self):
        """Clean up resources."""
        if self.text_model is not None:
            del self.text_model
            self.text_model = None

        if self.whisper_processor is not None:
            self.whisper_processor.cleanup()
            self.whisper_processor = None

        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.cache.clear()
        logger.info("EmbeddingService cleanup completed")
