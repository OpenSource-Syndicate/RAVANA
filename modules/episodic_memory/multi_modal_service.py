"""
MultiModalMemoryService - Main orchestration class for the multi-modal memory system.
Integrates all components: PostgreSQL storage, embeddings, Whisper, and search engine.
"""

import logging
import asyncio
import tempfile
import os
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
import uuid

from .models import (
    MemoryRecord, ContentType, MemoryType, SearchRequest, SearchResponse,
    ConversationRequest, MemoriesList, AudioMetadata, ImageMetadata,
    ProcessingResult, BatchProcessRequest, BatchProcessResult,
    MemoryStatistics, PerformanceMetrics
)
from .postgresql_store import PostgreSQLStore
from .embedding_service import EmbeddingService
from .whisper_processor import WhisperAudioProcessor
from .search_engine import AdvancedSearchEngine
from core.config import Config

logger = logging.getLogger(__name__)

class MultiModalMemoryService:
    """
    Main service class for multi-modal memory operations.
    Orchestrates storage, embeddings, audio processing, and search.
    """
    
    def __init__(self, 
                 database_url: str,
                 text_model_name: str = "all-MiniLM-L6-v2",
                 whisper_model_size: str = "base",
                 device: Optional[str] = None):
        """
        Initialize the multi-modal memory service.
        
        Args:
            database_url: PostgreSQL connection URL
            text_model_name: SentenceTransformer model name
            whisper_model_size: Whisper model size
            device: Device to use ("cpu", "cuda", "auto")
        """
        self.database_url = database_url
        self.device = device
        
        # Initialize components
        self.postgres_store = PostgreSQLStore(database_url)
        self.embedding_service = EmbeddingService(
            text_model_name=text_model_name,
            device=device
        )
        self.whisper_processor = WhisperAudioProcessor(
            model_size=whisper_model_size,
            device=device
        )
        self.search_engine = AdvancedSearchEngine(
            self.postgres_store,
            self.embedding_service,
            self.whisper_processor
        )
        
        # Service state
        self.initialized = False
        self.start_time = datetime.utcnow()
        
        logger.info("Initialized MultiModalMemoryService")
    
    async def initialize(self):
        """Initialize all service components."""
        try:
            await self.postgres_store.initialize()
            self.initialized = True
            logger.info("MultiModalMemoryService initialization completed")
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            raise
    
    async def close(self):
        """Close all service components gracefully."""
        logger.info("Initiating MultiModalMemoryService shutdown...")
        
        try:
            # Close PostgreSQL store first (database connections)
            if hasattr(self, 'postgres_store'):
                await asyncio.wait_for(
                    self.postgres_store.close(), 
                    timeout=Config.POSTGRES_CONNECTION_TIMEOUT
                )
                logger.info("PostgreSQL store closed")
        except asyncio.TimeoutError:
            logger.warning("PostgreSQL store close exceeded timeout")
        except Exception as e:
            logger.error(f"Error closing PostgreSQL store: {e}")
        
        try:
            # Clean up embedding service
            if hasattr(self, 'embedding_service'):
                self.embedding_service.cleanup()
                logger.info("Embedding service cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up embedding service: {e}")
        
        try:
            # Clean up Whisper processor
            if hasattr(self, 'whisper_processor'):
                self.whisper_processor.cleanup()
                logger.info("Whisper processor cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up whisper processor: {e}")
        
        try:
            # Clean up temporary files if enabled
            if Config.TEMP_FILE_CLEANUP_ENABLED:
                await self._cleanup_temp_files()
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
        
        self.initialized = False
        logger.info("MultiModalMemoryService shutdown completed")
    
    async def process_text_memory(self, 
                                text: str,
                                memory_type: MemoryType = MemoryType.EPISODIC,
                                tags: Optional[List[str]] = None,
                                emotional_valence: Optional[float] = None) -> MemoryRecord:
        """
        Process and store text memory.
        
        Args:
            text: Text content
            memory_type: Type of memory
            tags: Optional tags
            emotional_valence: Emotional valence (-1 to 1)
            
        Returns:
            Stored memory record
        """
        try:
            # Create memory record
            memory_record = MemoryRecord(
                content_type=ContentType.TEXT,
                content_text=text,
                memory_type=memory_type,
                tags=tags or [],
                emotional_valence=emotional_valence,
                created_at=datetime.utcnow()
            )
            
            # Generate embeddings
            memory_record = await self.embedding_service.generate_embeddings(memory_record)
            
            # Save to database
            saved_record = await self.postgres_store.save_memory_record(memory_record)
            
            logger.info(f"Processed text memory: {saved_record.id}")
            return saved_record
            
        except Exception as e:
            logger.error(f"Text memory processing failed: {e}")
            raise
    
    async def process_audio_memory(self, 
                                 audio_path: str,
                                 context: Optional[str] = None,
                                 memory_type: MemoryType = MemoryType.EPISODIC,
                                 tags: Optional[List[str]] = None) -> MemoryRecord:
        """
        Process and store audio memory with Whisper transcription.
        
        Args:
            audio_path: Path to audio file
            context: Optional context for transcription
            memory_type: Type of memory
            tags: Optional tags
            
        Returns:
            Stored memory record with audio metadata
        """
        try:
            # Process audio with Whisper
            audio_result = await self.whisper_processor.process_audio(audio_path, context)
            
            # Create audio metadata
            audio_metadata = self.whisper_processor.create_audio_metadata(audio_result)
            
            # Create memory record
            memory_record = MemoryRecord(
                content_type=ContentType.AUDIO,
                content_text=audio_result.get("transcript"),
                file_path=audio_path,
                memory_type=memory_type,
                tags=tags or [],
                confidence_score=audio_result.get("confidence", 0.8),
                audio_metadata=audio_metadata,
                created_at=datetime.utcnow()
            )
            
            # Generate embeddings
            memory_record = await self.embedding_service.generate_embeddings(memory_record)
            
            # Save to database
            saved_record = await self.postgres_store.save_memory_record(memory_record)
            
            logger.info(f"Processed audio memory: {saved_record.id}")
            return saved_record
            
        except Exception as e:
            logger.error(f"Audio memory processing failed: {e}")
            raise
    
    async def process_image_memory(self, 
                                 image_path: str,
                                 description: Optional[str] = None,
                                 memory_type: MemoryType = MemoryType.EPISODIC,
                                 tags: Optional[List[str]] = None) -> MemoryRecord:
        """
        Process and store image memory.
        
        Args:
            image_path: Path to image file
            description: Optional image description
            memory_type: Type of memory
            tags: Optional tags
            
        Returns:
            Stored memory record with image metadata
        """
        try:
            # Create basic image metadata
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
            
            image_metadata = ImageMetadata(
                width=width,
                height=height,
                scene_description=description
            )
            
            # Create memory record
            memory_record = MemoryRecord(
                content_type=ContentType.IMAGE,
                content_text=description,
                file_path=image_path,
                memory_type=memory_type,
                tags=tags or [],
                image_metadata=image_metadata,
                created_at=datetime.utcnow()
            )
            
            # Generate embeddings
            memory_record = await self.embedding_service.generate_embeddings(memory_record)
            
            # Save to database
            saved_record = await self.postgres_store.save_memory_record(memory_record)
            
            logger.info(f"Processed image memory: {saved_record.id}")
            return saved_record
            
        except Exception as e:
            logger.error(f"Image memory processing failed: {e}")
            raise
    
    async def extract_memories_from_conversation(self, request: ConversationRequest) -> MemoriesList:
        """
        Extract memories from conversation using LLM.
        
        Args:
            request: Conversation request
            
        Returns:
            List of extracted memories
        """
        try:
            # Import LLM function
            from core.llm import call_llm
            
            # Prepare extraction prompt
            prompt = f"""
            You are a memory extraction module for an AGI. Your task is to analyze a conversation 
            and identify key pieces of information to be stored in the AGI's long-term memory.

            Focus on extracting:
            - Key facts and information
            - User preferences and characteristics
            - Important goals, plans, or intentions
            - Notable events or experiences
            - Emotional context if relevant

            Guidelines:
            - Each memory should be a single, self-contained statement
            - Keep memories concise (under 30 words)
            - Prefer information that is likely to be relevant long-term
            - Do not store transitory conversational details
            - Output as a JSON object with a "memories" array

            Conversation:
            User: {request.user_input}
            AI: {request.ai_output}
            {f"Context: {request.context}" if request.context else ""}

            Extract memories as JSON:
            """
            
            # Call LLM
            llm_response = await asyncio.to_thread(call_llm, prompt)
            
            # Parse response
            import json
            import re
            
            try:
                # Extract JSON from response
                match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    memories = parsed.get("memories", [])
                else:
                    # Fallback: split by lines
                    memories = [line.strip() for line in llm_response.split('\n') 
                              if line.strip() and not line.startswith('#')]
            except json.JSONDecodeError:
                memories = [line.strip() for line in llm_response.split('\n') 
                          if line.strip() and not line.startswith('#')]
            
            return MemoriesList(
                memories=memories,
                memory_type=request.memory_type
            )
            
        except Exception as e:
            logger.error(f"Memory extraction failed: {e}")
            return MemoriesList(memories=[], memory_type=request.memory_type)
    
    async def save_extracted_memories(self, memories_list: MemoriesList) -> List[MemoryRecord]:
        """
        Save extracted memories to the database.
        
        Args:
            memories_list: List of memory texts
            
        Returns:
            List of saved memory records
        """
        saved_records = []
        
        for memory_text in memories_list.memories:
            if not memory_text.strip():
                continue
                
            try:
                memory_record = await self.process_text_memory(
                    text=memory_text,
                    memory_type=memories_list.memory_type
                )
                saved_records.append(memory_record)
            except Exception as e:
                logger.error(f"Failed to save memory '{memory_text}': {e}")
        
        logger.info(f"Saved {len(saved_records)} memories")
        return saved_records
    
    async def search_memories(self, request: SearchRequest) -> SearchResponse:
        """
        Search memories using the advanced search engine.
        
        Args:
            request: Search request
            
        Returns:
            Search response with results
        """
        return await self.search_engine.search(request)
    
    async def find_similar_memories(self, 
                                  memory_id: uuid.UUID,
                                  limit: int = 10,
                                  similarity_threshold: float = 0.7) -> List[MemoryRecord]:
        """
        Find memories similar to a given memory.
        
        Args:
            memory_id: Reference memory ID
            limit: Maximum results
            similarity_threshold: Minimum similarity
            
        Returns:
            List of similar memory records
        """
        try:
            # Get the reference memory
            reference_memory = await self.postgres_store.get_memory_record(memory_id)
            if not reference_memory:
                logger.warning(f"Reference memory not found: {memory_id}")
                return []
            
            # Find similar memories
            search_results = await self.search_engine.find_similar_memories(
                reference_memory, limit, similarity_threshold
            )
            
            return [result.memory_record for result in search_results]
            
        except Exception as e:
            logger.error(f"Similar memories search failed: {e}")
            return []
    
    async def batch_process_files(self, request: BatchProcessRequest) -> BatchProcessResult:
        """
        Process multiple files in batch.
        
        Args:
            request: Batch processing request
            
        Returns:
            Batch processing results
        """
        start_time = datetime.utcnow()
        results = []
        successful_count = 0
        failed_count = 0
        
        # Determine processing function for each file
        tasks = []
        for i, file_path in enumerate(request.file_paths):
            content_type = request.content_types[i] if request.content_types else self._detect_content_type(file_path)
            task = self._create_processing_task(file_path, content_type)
            tasks.append(task)
        
        # Process files
        if request.parallel_processing:
            # Process in parallel with limited concurrency
            semaphore = asyncio.Semaphore(request.max_workers)
            
            async def process_with_semaphore(task):
                async with semaphore:
                    return await task
            
            parallel_tasks = [process_with_semaphore(task) for task in tasks]
            results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
        else:
            # Process sequentially
            for task in tasks:
                result = await task
                results.append(result)
        
        # Process results
        processing_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processing_results.append(ProcessingResult(
                    memory_record=None,
                    processing_time_ms=0,
                    success=False,
                    error_message=str(result)
                ))
                failed_count += 1
            else:
                processing_results.append(result)
                if result.success:
                    successful_count += 1
                else:
                    failed_count += 1
        
        total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return BatchProcessResult(
            results=processing_results,
            total_processed=len(request.file_paths),
            successful_count=successful_count,
            failed_count=failed_count,
            total_time_ms=int(total_time)
        )
    
    def _detect_content_type(self, file_path: str) -> ContentType:
        """Detect content type from file extension."""
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.txt', '.md', '.json']:
            return ContentType.TEXT
        elif ext in ['.wav', '.mp3', '.m4a', '.ogg', '.flac']:
            return ContentType.AUDIO
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            return ContentType.IMAGE
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return ContentType.VIDEO
        else:
            return ContentType.TEXT  # Default
    
    async def _create_processing_task(self, file_path: str, content_type: ContentType):
        """Create processing task for a file."""
        start_time = datetime.utcnow()
        
        try:
            if content_type == ContentType.TEXT:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                memory_record = await self.process_text_memory(content)
            elif content_type == ContentType.AUDIO:
                memory_record = await self.process_audio_memory(file_path)
            elif content_type == ContentType.IMAGE:
                memory_record = await self.process_image_memory(file_path)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ProcessingResult(
                memory_record=memory_record,
                processing_time_ms=int(processing_time),
                success=True
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ProcessingResult(
                memory_record=None,
                processing_time_ms=int(processing_time),
                success=False,
                error_message=str(e)
            )
    
    async def get_memory_statistics(self) -> MemoryStatistics:
        """Get comprehensive memory statistics."""
        try:
            db_stats = await self.postgres_store.get_memory_statistics()
            
            # Get recent memories
            recent_query = """
                SELECT * FROM memory_records 
                ORDER BY created_at DESC 
                LIMIT 10
            """
            # This would need to be implemented in postgres_store
            
            # Get most accessed memories
            accessed_query = """
                SELECT * FROM memory_records 
                ORDER BY access_count DESC 
                LIMIT 10
            """
            
            return MemoryStatistics(
                total_memories=db_stats.get("total_memories", 0),
                by_content_type=db_stats.get("content_types", {}),
                by_memory_type={},  # Would be calculated from DB
                storage_size_mb=0.0,  # Would be calculated
                avg_confidence_score=db_stats.get("avg_confidence", 0.0),
                most_accessed_memories=[],  # Would be populated
                recent_additions=[],  # Would be populated
                consolidation_stats={}
            )
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return MemoryStatistics(
                total_memories=0,
                by_content_type={},
                by_memory_type={},
                storage_size_mb=0.0,
                avg_confidence_score=0.0,
                most_accessed_memories=[],
                recent_additions=[],
                consolidation_stats={}
            )
    
    async def consolidate_memories(self, 
                                 memory_ids: Optional[List[uuid.UUID]] = None,
                                 max_memories: int = 50) -> Dict[str, Any]:
        """
        Consolidate memories using LLM-based approach.
        
        Args:
            memory_ids: Specific memories to consolidate
            max_memories: Maximum memories to process
            
        Returns:
            Consolidation results
        """
        try:
            # This would implement the consolidation logic from the original memory.py
            logger.info("Memory consolidation would be implemented here")
            return {"status": "consolidation_placeholder"}
            
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            # Check database connection
            db_stats = await self.postgres_store.get_memory_statistics()
            db_connected = bool(db_stats)
            
            # Check embedding service
            test_embedding = await self.embedding_service.generate_text_embedding("test")
            embedding_ready = len(test_embedding) > 0
            
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            return {
                "status": "healthy" if db_connected and embedding_ready else "degraded",
                "database_connected": db_connected,
                "embedding_service_ready": embedding_ready,
                "memory_count": db_stats.get("total_memories", 0),
                "uptime_seconds": int(uptime),
                "initialized": self.initialized
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self.initialized
            }
    
    async def _cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        try:
            import tempfile
            temp_dirs = [
                Path(tempfile.gettempdir()) / "ravana_audio",
                Path(tempfile.gettempdir()) / "ravana_images"
            ]
            
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    file_count = 0
                    for file_path in temp_dir.iterdir():
                        try:
                            if file_path.is_file():
                                file_path.unlink()
                                file_count += 1
                        except Exception as e:
                            logger.warning(f"Could not remove temp file {file_path}: {e}")
                    
                    if file_count > 0:
                        logger.info(f"Cleaned up {file_count} temporary files from {temp_dir}")
                    
                    # Try to remove empty directory
                    try:
                        if not any(temp_dir.iterdir()):
                            temp_dir.rmdir()
                    except OSError:
                        pass  # Directory not empty or already removed
                        
        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")