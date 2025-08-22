from modules.episodic_memory.memory import (
    extract_memories_api,
    save_memories,
    get_relevant_memories_api,
    consolidate_memories_api,
)
import asyncio
import logging
import subprocess
import signal
import psutil
from core.config import Config

logger = logging.getLogger(__name__)

class MemoryService:
    def __init__(self):
        self.memory_server_process = None
        self.memory_server_host = "localhost"
        self.memory_server_port = 8001
    
    async def get_relevant_memories(self, query_text: str):
        return await get_relevant_memories_api({"query_text": query_text})

    async def save_memories(self, memories):
        await asyncio.to_thread(save_memories, memories)

    async def extract_memories(self, user_input: str, ai_output: str):
        return await extract_memories_api({"user_input": user_input, "ai_output": ai_output})

    async def consolidate_memories(self):
        from modules.episodic_memory.memory import ConsolidateRequest
        return await consolidate_memories_api(ConsolidateRequest())
    
    async def cleanup(self):
        """Clean up memory service resources during shutdown."""
        logger.info("Cleaning up memory service...")
        
        try:
            # If we're running memory service in a separate process, shut it down
            await self._shutdown_memory_server()
            
        except Exception as e:
            logger.error(f"Error during memory service cleanup: {e}")
        
        logger.info("Memory service cleanup completed")
    
    async def _shutdown_memory_server(self):
        """Shutdown memory server if running as separate process."""
        try:
            # Check if memory server is running on expected port
            import requests
            try:
                response = requests.get(
                    f"http://{self.memory_server_host}:{self.memory_server_port}/health",
                    timeout=2
                )
                if response.status_code == 200:
                    logger.info("Memory server detected, attempting graceful shutdown...")
                    
                    # Try to find the memory server process
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            if proc.info['cmdline'] and any('memory.py' in cmd for cmd in proc.info['cmdline']):
                                logger.info(f"Found memory server process (PID: {proc.pid})")
                                
                                # Send SIGTERM for graceful shutdown
                                proc.terminate()
                                
                                # Wait for process to shutdown gracefully
                                try:
                                    proc.wait(timeout=Config.MEMORY_SERVICE_SHUTDOWN_TIMEOUT)
                                    logger.info("Memory server shut down gracefully")
                                except psutil.TimeoutExpired:
                                    logger.warning("Memory server didn't shutdown gracefully, forcing...")
                                    proc.kill()
                                
                                break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    
            except requests.RequestException:
                # Memory server not running or not accessible
                logger.info("Memory server not accessible or not running")
                
        except Exception as e:
            logger.error(f"Error shutting down memory server: {e}") 