"""
Enhanced Action Manager with multi-modal support and parallel execution.
"""

import asyncio
import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from core.action_manager import ActionManager
from services.multi_modal_service import MultiModalService
from core.actions.multi_modal import (
    ProcessImageAction,
    ProcessAudioAction,
    AnalyzeDirectoryAction,
    CrossModalAnalysisAction
)

logger = logging.getLogger(__name__)

class EnhancedActionManager(ActionManager):
    """Enhanced action manager with multi-modal capabilities."""
    
    def __init__(self, agi_system, data_service):
        super().__init__(agi_system, data_service)
        self.multi_modal_service = MultiModalService()
        self.action_cache = {}
        self.parallel_limit = 3  # Max parallel actions
        self.register_enhanced_actions()
        
    def register_enhanced_actions(self):
        """Register new multi-modal actions as Action instances."""
        self.action_registry.register_action(ProcessImageAction(self.system, self.data_service))
        self.action_registry.register_action(ProcessAudioAction(self.system, self.data_service))
        self.action_registry.register_action(AnalyzeDirectoryAction(self.system, self.data_service))
        self.action_registry.register_action(CrossModalAnalysisAction(self.system, self.data_service))
        
    async def execute_action_enhanced(self, decision: dict) -> Any:
        """Enhanced action execution with better error handling and caching."""
        try:
            action_name = decision.get('action', 'unknown')
            params = decision.get('params', {})
            
            # Check cache for repeated actions (except for dynamic actions)
            non_cacheable = {'log_message', 'get_current_time', 'generate_random'}
            cache_key = f"{action_name}_{hash(str(params))}"
            
            if action_name not in non_cacheable and cache_key in self.action_cache:
                logger.info(f"Using cached result for action: {action_name}")
                return self.action_cache[cache_key]
            
            # Execute action with timeout
            result = await asyncio.wait_for(
                self.execute_action(decision),
                timeout=300  # 5 minute timeout
            )
            
            # Cache successful results for cacheable actions
            if (action_name not in non_cacheable and 
                result and not isinstance(result, Exception) and 
                not str(result).startswith("Error")):
                self.action_cache[cache_key] = result
                
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Action {action_name} timed out")
            return {"error": "Action timed out", "action": action_name}
        except Exception as e:
            logger.error(f"Enhanced action execution failed: {e}")
            return {"error": str(e), "action": action_name}
    
    async def execute_parallel_actions(self, decisions: List[dict]) -> List[Any]:
        """Execute multiple actions in parallel with concurrency limit."""
        if not decisions:
            return []
            
        semaphore = asyncio.Semaphore(self.parallel_limit)
        
        async def execute_with_semaphore(decision):
            async with semaphore:
                return await self.execute_action_enhanced(decision)
        
        tasks = [execute_with_semaphore(decision) for decision in decisions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"Executed {len(decisions)} actions in parallel")
        return results
    
    async def process_image_action(self, image_path: str, analysis_prompt: str = None) -> dict:
        """Action to process and analyze an image."""
        try:
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}
            
            result = await self.multi_modal_service.process_image(
                image_path, 
                analysis_prompt or "Analyze this image in detail"
            )
            
            # Add to knowledge base if successful
            if result.get('success', False):
                try:
                    await asyncio.to_thread(
                        self.system.knowledge_service.add_knowledge,
                        content=result['description'],
                        source="image_analysis",
                        category="visual_content"
                    )
                except Exception as e:
                    logger.warning(f"Failed to add image analysis to knowledge: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Image processing action failed: {e}")
            return {"error": str(e), "success": False}
    
    async def process_audio_action(self, audio_path: str, analysis_prompt: str = None) -> dict:
        """Action to process and analyze an audio file."""
        try:
            if not os.path.exists(audio_path):
                return {"error": f"Audio file not found: {audio_path}"}
            
            result = await self.multi_modal_service.process_audio(
                audio_path,
                analysis_prompt or "Describe and analyze this audio"
            )
            
            # Add to knowledge base if successful
            if result.get('success', False):
                try:
                    await asyncio.to_thread(
                        self.system.knowledge_service.add_knowledge,
                        content=result['description'],
                        source="audio_analysis", 
                        category="audio_content"
                    )
                except Exception as e:
                    logger.warning(f"Failed to add audio analysis to knowledge: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Audio processing action failed: {e}")
            return {"error": str(e), "success": False}
    
    async def analyze_directory_action(self, directory_path: str, recursive: bool = False) -> dict:
        """Action to analyze all media files in a directory."""
        try:
            if not os.path.exists(directory_path):
                return {"error": f"Directory not found: {directory_path}"}
            
            results = await self.multi_modal_service.process_directory(
                directory_path, 
                recursive
            )
            
            # Generate summary
            summary = await self.multi_modal_service.generate_content_summary(results)
            
            # Add summary to knowledge base
            try:
                await asyncio.to_thread(
                    self.system.knowledge_service.add_knowledge,
                    content=summary,
                    source="directory_analysis",
                    category="batch_analysis"
                )
            except Exception as e:
                logger.warning(f"Failed to add directory analysis to knowledge: {e}")
            
            return {
                "success": True,
                "directory": directory_path,
                "files_processed": len(results),
                "results": results,
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Directory analysis action failed: {e}")
            return {"error": str(e), "success": False}
    
    async def cross_modal_analysis_action(self, content_paths: List[str], analysis_prompt: str = None) -> dict:
        """Action to perform cross-modal analysis on multiple files."""
        try:
            if not content_paths:
                return {"error": "No content paths provided"}
            
            # Process each file
            processed_content = []
            for path in content_paths:
                if not os.path.exists(path):
                    logger.warning(f"File not found: {path}")
                    continue
                
                ext = Path(path).suffix.lower()
                if ext in self.multi_modal_service.supported_image_formats:
                    result = await self.multi_modal_service.process_image(path)
                elif ext in self.multi_modal_service.supported_audio_formats:
                    result = await self.multi_modal_service.process_audio(path)
                else:
                    logger.warning(f"Unsupported file format: {path}")
                    continue
                
                processed_content.append(result)
            
            if not processed_content:
                return {"error": "No valid content could be processed"}
            
            # Perform cross-modal analysis
            cross_modal_result = await self.multi_modal_service.cross_modal_analysis(
                processed_content,
                analysis_prompt
            )
            
            # Add to knowledge base
            if cross_modal_result.get('success', False):
                try:
                    await asyncio.to_thread(
                        self.system.knowledge_service.add_knowledge,
                        content=cross_modal_result['analysis'],
                        source="cross_modal_analysis",
                        category="multi_modal_insights"
                    )
                except Exception as e:
                    logger.warning(f"Failed to add cross-modal analysis to knowledge: {e}")
            
            return {
                "success": True,
                "content_processed": len(processed_content),
                "cross_modal_analysis": cross_modal_result,
                "individual_results": processed_content
            }
            
        except Exception as e:
            logger.error(f"Cross-modal analysis action failed: {e}")
            return {"error": str(e), "success": False}
    
    def clear_cache(self, max_size: int = 100):
        """Clear action cache if it gets too large."""
        if len(self.action_cache) > max_size:
            # Keep only the most recent entries
            items = list(self.action_cache.items())
            self.action_cache = dict(items[-max_size//2:])
            logger.info(f"Cleared action cache, kept {len(self.action_cache)} entries")
    
    async def get_action_statistics(self) -> dict:
        """Get statistics about action usage."""
        try:
            total_actions = len(self.action_registry.actions)
            cached_actions = len(self.action_cache)
            
            # Get available actions
            available_actions = list(self.action_registry.actions.keys())
            
            return {
                "total_registered_actions": total_actions,
                "cached_results": cached_actions,
                "available_actions": available_actions,
                "multi_modal_supported": True,
                "parallel_limit": self.parallel_limit
            }
            
        except Exception as e:
            logger.error(f"Failed to get action statistics: {e}")
            return {"error": str(e)}
