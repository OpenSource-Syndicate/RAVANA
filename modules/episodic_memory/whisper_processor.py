"""
WhisperAudioProcessor for multi-modal memory system.
Handles audio transcription, feature extraction, and analysis using OpenAI Whisper.
"""

import os
import logging
import asyncio
import tempfile
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np

try:
    import whisper
    import librosa
    import soundfile as sf
    import torch
    WHISPER_AVAILABLE = True
except ImportError as e:
    WHISPER_AVAILABLE = False
    whisper = None
    librosa = None
    sf = None
    torch = None
    logging.warning(f"Whisper/audio dependencies not available: {e}")

from .models import AudioMetadata

logger = logging.getLogger(__name__)

class WhisperAudioProcessor:
    """
    Audio processor using OpenAI Whisper for transcription and librosa for feature extraction.
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize the Whisper audio processor.
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device to use ("cpu", "cuda", "auto")
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper dependencies not available. Install with: pip install openai-whisper librosa soundfile")
        
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.supported_formats = {'.wav', '.mp3', '.m4a', '.ogg', '.flac', '.aac'}
        
        # Audio processing parameters
        self.sample_rate = 16000  # Whisper's expected sample rate
        self.max_duration = 300   # Maximum audio duration in seconds (5 minutes)
        
        logger.info(f"Initialized WhisperAudioProcessor with model_size={model_size}, device={self.device}")
    
    def _load_model(self):
        """Lazy load Whisper model to save memory."""
        if self.model is None:
            try:
                logger.info(f"Loading Whisper model: {self.model_size}")
                self.model = whisper.load_model(self.model_size, device=self.device)
                logger.info(f"Successfully loaded Whisper model on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
    
    async def process_audio(self, audio_path: str, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Process audio file with Whisper transcription and feature extraction.
        
        Args:
            audio_path: Path to audio file
            custom_prompt: Optional prompt to guide transcription
            
        Returns:
            Dict containing transcription and analysis results
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            file_ext = Path(audio_path).suffix.lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported audio format: {file_ext}")
            
            # Load and validate audio
            audio_info = await self._get_audio_info(audio_path)
            if audio_info["duration"] > self.max_duration:
                logger.warning(f"Audio duration {audio_info['duration']:.1f}s exceeds maximum {self.max_duration}s")
            
            # Preprocess audio if needed
            processed_audio_path = await self._preprocess_audio(audio_path)
            
            # Transcribe with Whisper
            transcription_result = await self._transcribe_audio(processed_audio_path, custom_prompt)
            
            # Extract audio features
            audio_features = await self._extract_audio_features(processed_audio_path)
            
            # Clean up temporary files
            if processed_audio_path != audio_path:
                os.unlink(processed_audio_path)
            
            # Combine results
            result = {
                "transcript": transcription_result["text"],
                "language": transcription_result["language"],
                "segments": transcription_result.get("segments", []),
                "confidence": self._calculate_overall_confidence(transcription_result),
                "audio_features": audio_features,
                "duration": audio_info["duration"],
                "sample_rate": audio_info["sample_rate"],
                "channels": audio_info["channels"],
                "file_size": os.path.getsize(audio_path),
                "processing_info": {
                    "model_size": self.model_size,
                    "device": self.device,
                    "custom_prompt_used": custom_prompt is not None
                }
            }
            
            logger.info(f"Successfully processed audio: {audio_path} (duration: {audio_info['duration']:.1f}s)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process audio {audio_path}: {e}")
            raise
    
    async def _get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """Get basic audio file information."""
        try:
            # Use librosa to get audio info without loading the full file
            duration = librosa.get_duration(path=audio_path)
            
            # Get more detailed info using soundfile
            with sf.SoundFile(audio_path) as f:
                sample_rate = f.samplerate
                channels = f.channels
                frames = f.frames
            
            return {
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "frames": frames
            }
        except Exception as e:
            logger.warning(f"Could not get audio info for {audio_path}: {e}")
            return {"duration": 0, "sample_rate": 16000, "channels": 1, "frames": 0}
    
    async def _preprocess_audio(self, audio_path: str) -> str:
        """
        Preprocess audio file to ensure compatibility with Whisper.
        Returns path to processed file (may be the same as input).
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Check if preprocessing is needed
            needs_processing = (
                sr != self.sample_rate or
                len(y.shape) > 1 or  # Multi-channel
                len(y) / sr > self.max_duration
            )
            
            if not needs_processing:
                return audio_path
            
            # Resample to Whisper's expected sample rate
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            
            # Convert to mono if stereo
            if len(y.shape) > 1:
                y = librosa.to_mono(y)
            
            # Trim to maximum duration
            if len(y) / self.sample_rate > self.max_duration:
                y = y[:int(self.max_duration * self.sample_rate)]
                logger.warning(f"Audio trimmed to {self.max_duration} seconds")
            
            # Save preprocessed audio to temporary file
            temp_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            sf.write(temp_path, y, self.sample_rate)
            
            logger.info(f"Preprocessed audio saved to: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return audio_path  # Return original path if preprocessing fails
    
    async def _transcribe_audio(self, audio_path: str, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe audio using Whisper."""
        self._load_model()
        
        try:
            # Prepare Whisper options
            options = {
                "language": None,  # Auto-detect language
                "task": "transcribe",
                "word_timestamps": True,
                "temperature": 0.0,  # Deterministic output
                "no_speech_threshold": 0.6,
                "logprob_threshold": -1.0,
                "compression_ratio_threshold": 2.4
            }
            
            if custom_prompt:
                options["initial_prompt"] = custom_prompt
            
            # Run transcription
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: self.model.transcribe(audio_path, **options)
            )
            
            logger.info(f"Transcription completed. Language: {result.get('language', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise
    
    async def _extract_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract audio features for embedding generation and analysis."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract various audio features
            features = {}
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["spectral_centroid"] = {
                "mean": float(np.mean(spectral_centroids)),
                "std": float(np.std(spectral_centroids)),
                "median": float(np.median(spectral_centroids))
            }
            
            # MFCCs (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features["mfcc"] = {
                "mean": mfccs.mean(axis=1).tolist(),
                "std": mfccs.std(axis=1).tolist()
            }
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features["chroma"] = {
                "mean": chroma.mean(axis=1).tolist(),
                "std": chroma.std(axis=1).tolist()
            }
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features["zero_crossing_rate"] = {
                "mean": float(np.mean(zcr)),
                "std": float(np.std(zcr))
            }
            
            # Tempo and beat tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features["tempo"] = float(tempo)
            features["beat_count"] = len(beats)
            
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            features["rms_energy"] = {
                "mean": float(np.mean(rms)),
                "std": float(np.std(rms))
            }
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features["spectral_rolloff"] = {
                "mean": float(np.mean(rolloff)),
                "std": float(np.std(rolloff))
            }
            
            logger.info("Audio feature extraction completed")
            return features
            
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            return {}
    
    def _calculate_overall_confidence(self, transcription_result: Dict[str, Any]) -> float:
        """Calculate overall confidence score from Whisper result."""
        try:
            if "segments" in transcription_result and transcription_result["segments"]:
                # Calculate average confidence from segments
                confidences = []
                for segment in transcription_result["segments"]:
                    if "avg_logprob" in segment:
                        # Convert log probability to confidence (0-1)
                        confidence = min(1.0, max(0.0, np.exp(segment["avg_logprob"])))
                        confidences.append(confidence)
                
                if confidences:
                    return float(np.mean(confidences))
            
            # Fallback: use simple heuristics
            text_length = len(transcription_result.get("text", "").strip())
            if text_length == 0:
                return 0.0
            elif text_length < 10:
                return 0.6
            else:
                return 0.8
                
        except Exception as e:
            logger.warning(f"Could not calculate confidence: {e}")
            return 0.5
    
    def create_audio_metadata(self, processing_result: Dict[str, Any]) -> AudioMetadata:
        """Create AudioMetadata object from processing result."""
        return AudioMetadata(
            transcript=processing_result.get("transcript"),
            language_code=processing_result.get("language"),
            confidence_scores={
                "overall": processing_result.get("confidence", 0.5),
                "segments": [
                    s.get("avg_logprob", 0.0) for s in processing_result.get("segments", [])
                ]
            },
            duration_seconds=processing_result.get("duration"),
            audio_features=processing_result.get("audio_features", {}),
            sample_rate=processing_result.get("sample_rate"),
            channels=processing_result.get("channels", 1)
        )
    
    async def batch_process_audio(self, audio_paths: List[str], 
                                 custom_prompts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Process multiple audio files in batch."""
        results = []
        prompts = custom_prompts or [None] * len(audio_paths)
        
        for i, audio_path in enumerate(audio_paths):
            try:
                result = await self.process_audio(audio_path, prompts[i])
                results.append({
                    "path": audio_path,
                    "success": True,
                    "result": result
                })
            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
                results.append({
                    "path": audio_path,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info("WhisperAudioProcessor cleanup completed")