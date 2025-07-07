"""
Multi-Modal Service for AGI System
Handles image, audio, and cross-modal processing capabilities.
"""

import os
import logging
import asyncio
import tempfile
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from modules.decision_engine.llm import call_gemini_image_caption, call_gemini_audio_description, call_gemini_with_function_calling

logger = logging.getLogger(__name__)

class MultiModalService:
    """Service for handling multi-modal content processing."""
    
    def __init__(self):
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        self.supported_audio_formats = {'.mp3', '.wav', '.m4a', '.ogg', '.flac'}
        self.temp_dir = Path(tempfile.gettempdir()) / "agi_multimodal"
        self.temp_dir.mkdir(exist_ok=True)
        
    async def process_image(self, image_path: str, prompt: str = "Analyze this image in detail") -> Dict[str, Any]:
        """
        Process an image and return detailed analysis.
        
        Args:
            image_path: Path to the image file
            prompt: Custom prompt for image analysis
            
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
            
            result = {
                "type": "image",
                "path": image_path,
                "format": file_ext,
                "size_bytes": file_size,
                "description": description,
                "analysis_prompt": prompt,
                "success": True,
                "error": None
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
                "description": f"Failed to process image: {e}"
            }
    
    async def process_audio(self, audio_path: str, prompt: str = "Describe and analyze this audio") -> Dict[str, Any]:
        """
        Process an audio file and return analysis.
        
        Args:
            audio_path: Path to the audio file
            prompt: Custom prompt for audio analysis
            
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
            
            result = {
                "type": "audio",
                "path": audio_path,
                "format": file_ext,
                "size_bytes": file_size,
                "description": description,
                "analysis_prompt": prompt,
                "success": True,
                "error": None
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
                "description": f"Failed to process audio: {e}"
            }
    
    async def cross_modal_analysis(self, content_list: List[Dict[str, Any]], analysis_prompt: str = None) -> Dict[str, Any]:
        """
        Perform cross-modal analysis on multiple types of content.
        
        Args:
            content_list: List of processed content (images, audio, text)
            analysis_prompt: Custom prompt for cross-modal analysis
            
        Returns:
            Dict containing cross-modal analysis results
        """
        try:
            if not content_list:
                raise ValueError("No content provided for cross-modal analysis")
            
            # Prepare content descriptions
            descriptions = []
            content_types = []
            
            for content in content_list:
                if content.get('success', False):
                    descriptions.append(content.get('description', ''))
                    content_types.append(content.get('type', 'unknown'))
            
            if not descriptions:
                raise ValueError("No successfully processed content for analysis")
            
            # Create analysis prompt
            if not analysis_prompt:
                analysis_prompt = f"""
                Perform a comprehensive cross-modal analysis of the following content:
                
                Content types: {', '.join(set(content_types))}
                
                Content descriptions:
                {chr(10).join([f"{i+1}. {desc}" for i, desc in enumerate(descriptions)])}
                
                Please provide:
                1. Common themes and patterns across all content
                2. Relationships and connections between different modalities
                3. Insights that emerge from combining these different types of information
                4. Potential applications or implications
                5. Any contradictions or interesting contrasts
                """
            
            # Use LLM for cross-modal analysis
            loop = asyncio.get_event_loop()
            from modules.decision_engine.llm import safe_call_llm
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
                "success": True,
                "error": None
            }
            
            logger.info(f"Successfully performed cross-modal analysis on {len(content_list)} items")
            return result
            
        except Exception as e:
            logger.error(f"Cross-modal analysis failed: {e}")
            return {
                "type": "cross_modal_analysis",
                "success": False,
                "error": str(e),
                "analysis": f"Cross-modal analysis failed: {e}"
            }
    
    async def generate_content_summary(self, processed_content: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive summary of all processed multi-modal content.
        
        Args:
            processed_content: List of processed content results
            
        Returns:
            String summary of all content
        """
        try:
            if not processed_content:
                return "No multi-modal content processed."
            
            successful_content = [c for c in processed_content if c.get('success', False)]
            failed_content = [c for c in processed_content if not c.get('success', False)]
            
            summary_parts = []
            
            # Summary header
            summary_parts.append(f"Multi-Modal Content Summary ({len(processed_content)} items processed)")
            summary_parts.append("=" * 50)
            
            # Successful content
            if successful_content:
                summary_parts.append(f"\nSuccessfully Processed ({len(successful_content)} items):")
                for i, content in enumerate(successful_content, 1):
                    content_type = content.get('type', 'unknown').title()
                    description = content.get('description', 'No description')[:200]
                    summary_parts.append(f"\n{i}. {content_type}: {description}...")
            
            # Failed content
            if failed_content:
                summary_parts.append(f"\n\nFailed to Process ({len(failed_content)} items):")
                for i, content in enumerate(failed_content, 1):
                    content_type = content.get('type', 'unknown').title()
                    error = content.get('error', 'Unknown error')
                    summary_parts.append(f"\n{i}. {content_type}: {error}")
            
            # Cross-modal insights
            if len(successful_content) > 1:
                cross_modal = await self.cross_modal_analysis(successful_content)
                if cross_modal.get('success', False):
                    summary_parts.append(f"\n\nCross-Modal Analysis:")
                    summary_parts.append(cross_modal.get('analysis', 'No analysis available'))
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate content summary: {e}")
            return f"Failed to generate multi-modal content summary: {e}"
    
    async def process_directory(self, directory_path: str, recursive: bool = False) -> List[Dict[str, Any]]:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to directory to process
            recursive: Whether to process subdirectories
            
        Returns:
            List of processing results
        """
        try:
            if not os.path.exists(directory_path):
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
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
                    if ext in self.supported_image_formats or ext in self.supported_audio_formats:
                        supported_files.append(file_path)
            
            logger.info(f"Found {len(supported_files)} supported files in {directory_path}")
            
            # Process each file
            for file_path in supported_files:
                try:
                    ext = file_path.suffix.lower()
                    if ext in self.supported_image_formats:
                        result = await self.process_image(str(file_path))
                    elif ext in self.supported_audio_formats:
                        result = await self.process_audio(str(file_path))
                    else:
                        continue
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to process file {file_path}: {e}")
                    results.append({
                        "type": "unknown",
                        "path": str(file_path),
                        "success": False,
                        "error": str(e)
                    })
            
            logger.info(f"Processed {len(results)} files from directory {directory_path}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to process directory {directory_path}: {e}")
            return [{
                "type": "directory_processing",
                "path": directory_path,
                "success": False,
                "error": str(e)
            }]
    
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
