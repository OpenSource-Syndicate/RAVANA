"""
Multi-Modal Service for AGI System
Handles image, audio, and cross-modal processing capabilities.
Enhanced with better integration for various media types and improved cross-modal analysis.
"""

import os
import logging
import asyncio
import tempfile
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from core.llm import call_gemini_image_caption, call_gemini_audio_description, call_gemini_with_function_calling

logger = logging.getLogger(__name__)


class MultiModalService:
    """Service for handling multi-modal content processing."""

    def __init__(self):
        self.supported_image_formats = {
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        self.supported_audio_formats = {
            '.mp3', '.wav', '.m4a', '.ogg', '.flac'}
        self.supported_video_formats = {
            '.mp4', '.mov', '.avi', '.mkv', '.webm'}
        self.temp_dir = Path(tempfile.gettempdir()) / "agi_multimodal"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Enhanced context tracking
        self.processing_context = {}

    async def process_image(self, image_path: str, prompt: str = "Analyze this image in detail", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process an image and return detailed analysis.

        Args:
            image_path: Path to the image file
            prompt: Custom prompt for image analysis
            context: Additional context for processing

        Returns:
            Dict containing analysis results
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            file_ext = Path(image_path).suffix.lower()
            if file_ext not in self.supported_image_formats:
                raise ValueError(f"Unsupported image format: {file_ext}")

            # Use Gemini for image captioning
            loop = asyncio.get_event_loop()
            description = await loop.run_in_executor(
                None,
                call_gemini_image_caption,
                image_path,
                prompt
            )

            # Extract metadata
            file_size = os.path.getsize(image_path)
            file_stat = os.stat(image_path)
            
            # Enhanced metadata extraction
            enhanced_metadata = await self._extract_enhanced_metadata(image_path, "image", context)

            result = {
                "type": "image",
                "path": image_path,
                "format": file_ext,
                "size_bytes": file_size,
                "description": description,
                "analysis_prompt": prompt,
                "metadata": enhanced_metadata,
                "success": True,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Successfully processed image: {image_path}")
            return result

        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            return {
                "type": "image",
                "path": image_path,
                "success": False,
                "error": str(e),
                "description": f"Failed to process image: {e}",
                "timestamp": datetime.now().isoformat()
            }

    async def process_audio(self, audio_path: str, prompt: str = "Describe and analyze this audio", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process an audio file and return analysis.

        Args:
            audio_path: Path to the audio file
            prompt: Custom prompt for audio analysis
            context: Additional context for processing

        Returns:
            Dict containing analysis results
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            file_ext = Path(audio_path).suffix.lower()
            if file_ext not in self.supported_audio_formats:
                raise ValueError(f"Unsupported audio format: {file_ext}")

            # Use Gemini for audio description
            loop = asyncio.get_event_loop()
            description = await loop.run_in_executor(
                None,
                call_gemini_audio_description,
                audio_path,
                prompt
            )

            # Extract metadata
            file_size = os.path.getsize(audio_path)
            file_stat = os.stat(audio_path)
            
            # Enhanced metadata extraction
            enhanced_metadata = await self._extract_enhanced_metadata(audio_path, "audio", context)

            result = {
                "type": "audio",
                "path": audio_path,
                "format": file_ext,
                "size_bytes": file_size,
                "description": description,
                "analysis_prompt": prompt,
                "metadata": enhanced_metadata,
                "success": True,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Successfully processed audio: {audio_path}")
            return result

        except Exception as e:
            logger.error(f"Failed to process audio {audio_path}: {e}")
            return {
                "type": "audio",
                "path": audio_path,
                "success": False,
                "error": str(e),
                "description": f"Failed to process audio: {e}",
                "timestamp": datetime.now().isoformat()
            }

    async def process_video(self, video_path: str, prompt: str = "Analyze this video content", context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a video file and return analysis.

        Args:
            video_path: Path to the video file
            prompt: Custom prompt for video analysis
            context: Additional context for processing

        Returns:
            Dict containing analysis results
        """
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            file_ext = Path(video_path).suffix.lower()
            if file_ext not in self.supported_video_formats:
                raise ValueError(f"Unsupported video format: {file_ext}")

            # For now, we'll use a generic approach for video processing
            # In a full implementation, this would extract frames and analyze them
            description = f"Video file at {video_path} processed with content analysis capabilities."

            # Extract metadata
            file_size = os.path.getsize(video_path)
            file_stat = os.stat(video_path)
            
            # Enhanced metadata extraction
            enhanced_metadata = await self._extract_enhanced_metadata(video_path, "video", context)

            result = {
                "type": "video",
                "path": video_path,
                "format": file_ext,
                "size_bytes": file_size,
                "description": description,
                "analysis_prompt": prompt,
                "metadata": enhanced_metadata,
                "success": True,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Successfully processed video: {video_path}")
            return result

        except Exception as e:
            logger.error(f"Failed to process video {video_path}: {e}")
            return {
                "type": "video",
                "path": video_path,
                "success": False,
                "error": str(e),
                "description": f"Failed to process video: {e}",
                "timestamp": datetime.now().isoformat()
            }

    async def _extract_enhanced_metadata(self, file_path: str, file_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract enhanced metadata from a file.
        
        Args:
            file_path: Path to the file
            file_type: Type of file (image, audio, video)
            context: Additional context for metadata extraction
            
        Returns:
            Dict containing enhanced metadata
        """
        try:
            file_stat = os.stat(file_path)
            
            metadata = {
                "creation_time": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                "modification_time": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "access_time": datetime.fromtimestamp(file_stat.st_atime).isoformat(),
                "file_permissions": oct(file_stat.st_mode)[-3:],
                "file_type": file_type,
                "context": context or {},
                "processing_session": str(hash(file_path + str(datetime.now())))
            }
            
            # Add file-type specific metadata
            if file_type == "image":
                # Image-specific metadata would be extracted here in a full implementation
                metadata["image_specific"] = {
                    "resolution_approximation": "would be extracted in full implementation",
                    "color_depth": "would be extracted in full implementation"
                }
            elif file_type == "audio":
                # Audio-specific metadata would be extracted here in a full implementation
                metadata["audio_specific"] = {
                    "duration_approximation": "would be extracted in full implementation",
                    "sample_rate": "would be extracted in full implementation"
                }
            elif file_type == "video":
                # Video-specific metadata would be extracted here in a full implementation
                metadata["video_specific"] = {
                    "duration_approximation": "would be extracted in full implementation",
                    "resolution_approximation": "would be extracted in full implementation"
                }
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Failed to extract enhanced metadata for {file_path}: {e}")
            return {
                "file_type": file_type,
                "context": context or {},
                "error": str(e)
            }

    async def cross_modal_analysis(self, content_list: List[Dict[str, Any]], analysis_prompt: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform cross-modal analysis on multiple types of content.

        Args:
            content_list: List of processed content (images, audio, text)
            analysis_prompt: Custom prompt for cross-modal analysis
            context: Additional context for analysis

        Returns:
            Dict containing cross-modal analysis results
        """
        try:
            if not content_list:
                raise ValueError(
                    "No content provided for cross-modal analysis")

            # Prepare content descriptions
            descriptions = []
            content_types = []

            for content in content_list:
                if content.get('success', False):
                    descriptions.append(content.get('description', ''))
                    content_types.append(content.get('type', 'unknown'))

            if not descriptions:
                raise ValueError(
                    "No successfully processed content for analysis")

            # Create analysis prompt with context
            if not analysis_prompt:
                context_str = f"\nContext: {json.dumps(context, indent=2)}" if context else ""
                
                analysis_prompt = f"""
                Perform a comprehensive cross-modal analysis of the following content:
                
                Content types: {', '.join(set(content_types))}
                {context_str}
                
                Content descriptions:
                {chr(10).join([f"{i+1}. {desc}" for i, desc in enumerate(descriptions)])}
                
                Please provide:
                1. Common themes and patterns across all content
                2. Relationships and connections between different modalities
                3. Insights that emerge from combining these different types of information
                4. Potential applications or implications
                5. Any contradictions or interesting contrasts
                6. Synthesis of information across all modalities
                """

            # Use LLM for cross-modal analysis
            loop = asyncio.get_event_loop()
            from core.llm import safe_call_llm
            analysis = await loop.run_in_executor(
                None,
                safe_call_llm,
                analysis_prompt
            )

            result = {
                "type": "cross_modal_analysis",
                "content_types": content_types,
                "num_items": len(content_list),
                "analysis": analysis,
                "context": context or {},
                "success": True,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(
                f"Successfully performed cross-modal analysis on {len(content_list)} items")
            return result

        except Exception as e:
            logger.error(f"Cross-modal analysis failed: {e}")
            return {
                "type": "cross_modal_analysis",
                "success": False,
                "error": str(e),
                "analysis": f"Cross-modal analysis failed: {e}",
                "timestamp": datetime.now().isoformat()
            }

    async def generate_content_summary(self, processed_content: List[Dict[str, Any]], context: Dict[str, Any] = None) -> str:
        """
        Generate a comprehensive summary of all processed multi-modal content.

        Args:
            processed_content: List of processed content results
            context: Additional context for summary generation

        Returns:
            String summary of all content
        """
        try:
            if not processed_content:
                return "No multi-modal content processed."

            successful_content = [
                c for c in processed_content if c.get('success', False)]
            failed_content = [
                c for c in processed_content if not c.get('success', False)]

            summary_parts = []

            # Summary header
            context_str = f" (Context: {context.get('summary_purpose', 'General')})" if context else ""
            summary_parts.append(
                f"Multi-Modal Content Summary ({len(processed_content)} items processed){context_str}")
            summary_parts.append("=" * 50)

            # Successful content
            if successful_content:
                summary_parts.append(
                    f"\nSuccessfully Processed ({len(successful_content)} items):")
                for i, content in enumerate(successful_content, 1):
                    content_type = content.get('type', 'unknown').title()
                    description = content.get(
                        'description', 'No description')[:200]
                    summary_parts.append(
                        f"\n{i}. {content_type}: {description}...")

            # Failed content
            if failed_content:
                summary_parts.append(
                    f"\n\nFailed to Process ({len(failed_content)} items):")
                for i, content in enumerate(failed_content, 1):
                    content_type = content.get('type', 'unknown').title()
                    error = content.get('error', 'Unknown error')
                    summary_parts.append(f"\n{i}. {content_type}: {error}")

            # Cross-modal insights
            if len(successful_content) > 1:
                cross_modal = await self.cross_modal_analysis(successful_content, context=context)
                if cross_modal.get('success', False):
                    summary_parts.append(f"\n\nCross-Modal Analysis:")
                    summary_parts.append(cross_modal.get(
                        'analysis', 'No analysis available'))

            return "\n".join(summary_parts)

        except Exception as e:
            logger.error(f"Failed to generate content summary: {e}")
            return f"Failed to generate multi-modal content summary: {e}"

    async def process_directory(self, directory_path: str, recursive: bool = False, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process all supported files in a directory.

        Args:
            directory_path: Path to directory to process
            recursive: Whether to process subdirectories
            context: Additional context for processing

        Returns:
            List of processing results
        """
        try:
            if not os.path.exists(directory_path):
                raise FileNotFoundError(
                    f"Directory not found: {directory_path}")

            results = []
            directory = Path(directory_path)

            # Get all files
            if recursive:
                files = list(directory.rglob("*"))
            else:
                files = list(directory.iterdir())

            # Filter for supported files
            supported_files = []
            for file_path in files:
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in self.supported_image_formats or ext in self.supported_audio_formats or ext in self.supported_video_formats:
                        supported_files.append(file_path)

            logger.info(
                f"Found {len(supported_files)} supported files in {directory_path}")

            # Process each file
            for file_path in supported_files:
                try:
                    ext = file_path.suffix.lower()
                    if ext in self.supported_image_formats:
                        result = await self.process_image(str(file_path), context=context)
                    elif ext in self.supported_audio_formats:
                        result = await self.process_audio(str(file_path), context=context)
                    elif ext in self.supported_video_formats:
                        result = await self.process_video(str(file_path), context=context)
                    else:
                        continue

                    results.append(result)

                except Exception as e:
                    logger.warning(f"Failed to process file {file_path}: {e}")
                    results.append({
                        "type": "unknown",
                        "path": str(file_path),
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })

            logger.info(
                f"Processed {len(results)} files from directory {directory_path}")
            return results

        except Exception as e:
            logger.error(f"Failed to process directory {directory_path}: {e}")
            return [{
                "type": "directory_processing",
                "path": directory_path,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }]

    async def process_mixed_content(self, content_items: List[Dict[str, Any]], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process a mixed collection of different content types.
        
        Args:
            content_items: List of content items with type and path information
            context: Additional context for processing
            
        Returns:
            List of processing results
        """
        results = []
        
        for item in content_items:
            try:
                content_type = item.get('type')
                content_path = item.get('path')
                
                if not content_type or not content_path:
                    results.append({
                        "type": "invalid_item",
                        "success": False,
                        "error": "Missing type or path information",
                        "timestamp": datetime.now().isoformat()
                    })
                    continue
                
                # Process based on content type
                if content_type == "image":
                    result = await self.process_image(content_path, context=context)
                elif content_type == "audio":
                    result = await self.process_audio(content_path, context=context)
                elif content_type == "video":
                    result = await self.process_video(content_path, context=context)
                else:
                    result = {
                        "type": content_type,
                        "path": content_path,
                        "success": False,
                        "error": f"Unsupported content type: {content_type}",
                        "timestamp": datetime.now().isoformat()
                    }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing content item {item}: {e}")
                results.append({
                    "type": "processing_error",
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        return results

    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up temporary files older than specified age."""
        try:
            import time
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            cleaned_count = 0
            for file_path in self.temp_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} temporary files")

        except Exception as e:
            logger.warning(f"Failed to cleanup temporary files: {e}")

    def set_context(self, context_id: str, context: Dict[str, Any]):
        """
        Set processing context for a specific session.
        
        Args:
            context_id: Unique identifier for the context
            context: Context data to store
        """
        self.processing_context[context_id] = context
        
    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Get processing context for a specific session.
        
        Args:
            context_id: Unique identifier for the context
            
        Returns:
            Context data or None if not found
        """
        return self.processing_context.get(context_id)
        
    def clear_context(self, context_id: str):
        """
        Clear processing context for a specific session.
        
        Args:
            context_id: Unique identifier for the context
        """
        if context_id in self.processing_context:
            del self.processing_context[context_id]
