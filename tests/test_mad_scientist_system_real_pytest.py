#!/usr/bin/env python3
"""
Pytest-compatible test script for the Mad Scientist System in RAVANA AGI using real system components.
This script tests the core functionality of the mad scientist components with real services
but mocked LLM responses to prevent hanging and ensure reliability.
"""
import asyncio
import sys
import os
import logging
from unittest.mock import patch

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pytest
from modules.mad_scientist_system import MadScientistSystem
from modules.mad_scientist_impossible_projects import MadScientistModule
from modules.innovation_publishing_system import InnovationPublishingSystem
from core.config import Config
from core.enhanced_memory_service import enhanced_memory_service
from services.knowledge_service import KnowledgeService
from database.engine import get_engine
from core.shutdown_coordinator import ShutdownCoordinator, ShutdownPriority


class OptimizedBlogScheduler:
    """Optimized blog scheduler for testing with real system components."""
    
    async def register_learning_event(self, **kwargs):
        print(f"Blog scheduler registered: {kwargs.get('topic', 'Unknown topic')}")
        # Add a small delay to mimic real processing without hanging
        await asyncio.sleep(0.05)
        return True


class RealAGISystem:
    """Real AGI system for testing (using actual services rather than mocks)."""
    
    def __init__(self):
        self.config = Config()
        self.memory_service = enhanced_memory_service
        self.knowledge_service = KnowledgeService(get_engine())
        self.background_tasks = set()  # Track background tasks for cleanup
        self._shutdown = asyncio.Event()  # Shutdown event
        
        # Initialize shutdown coordinator
        self.shutdown_coordinator = ShutdownCoordinator(self)
        
        # Register cleanup tasks with shutdown coordinator
        self.shutdown_coordinator.register_cleanup_handler(self.cleanup_resources, is_async=True)
        self.shutdown_coordinator.register_component(self, ShutdownPriority.MEDIUM, is_async=True)
    
    async def cleanup_resources(self):
        """Cleanup function for resources used during testing."""
        print("Cleaning up resources...")
        
        # Cancel any remaining background tasks
        for task in self.background_tasks.copy():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass  # Expected when cancelling tasks
        
        print("Resource cleanup completed.")


async def run_with_timeout(coro, timeout_seconds: int, default_return=None):
    """Run a coroutine with a timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        print(f"Operation timed out after {timeout_seconds} seconds")
        return default_return
    except asyncio.CancelledError:
        print(f"Operation was cancelled after {timeout_seconds} seconds")
        return default_return


@pytest.mark.asyncio
async def test_impossible_projects_module(mock_llm_responses):
    """Test the impossible projects module with real components."""
    print("Testing Impossible Projects Module with real components...")
    
    # Initialize with real system components
    real_agi = RealAGISystem()
    blog_scheduler = OptimizedBlogScheduler()
    
    impossible_projects = MadScientistModule(real_agi, blog_scheduler)
    
    # Test generating an impossible project with timeout
    project = await run_with_timeout(
        impossible_projects.generate_impossible_project("artificial general intelligence"),
        timeout_seconds=10
    )
    
    assert project is not None, "Project should be generated"
    print(f"Generated impossible project: {project.name}")
    print(f"Description: {project.description}")
    print(f"Impossibility reason: {project.impossibility_reason}")
    print(f"Initial approach: {project.initial_approach}")
    
    # Test attempting the impossible project with timeout
    result = await run_with_timeout(
        impossible_projects.attempt_impossible_project(project.id),
        timeout_seconds=20
    )
    
    assert result is not None, "Project attempt should complete"
    print(f"Attempt result: {result}")
    
    print("Impossible Projects Module test completed.\n")
    
    # Cleanup after the test
    await real_agi.cleanup_resources()


@pytest.mark.asyncio
async def test_innovation_publishing_system(mock_llm_responses):
    """Test the innovation publishing system with real components."""
    print("Testing Innovation Publishing System with real components...")
    
    # Initialize with real system components
    real_agi = RealAGISystem()
    blog_scheduler = OptimizedBlogScheduler()
    
    publisher = InnovationPublishingSystem(real_agi, blog_scheduler)
    
    # Test identifying an innovation for publication with timeout
    publication = await run_with_timeout(
        publisher.identify_innovation_for_publication(
            discovery_type="breakthrough",
            discovery_content="Developed a new approach to consciousness transfer",
            context="In research on AI self-modeling"
        ),
        timeout_seconds=10
    )
    
    assert publication is not None, "Publication should be identified"
    print(f"Identified publication: {publication.title}")
    print(f"Importance: {publication.importance}")
    print(f"Tags: {publication.tags}")
    
    print("Innovation Publishing System test completed.\n")
    
    # Cleanup after the test
    await real_agi.cleanup_resources()


@pytest.mark.asyncio
async def test_mad_scientist_system_integration(mock_llm_responses):
    """Test the full mad scientist system integration with real components."""
    print("Testing Mad Scientist System Integration with real components...")
    
    # Initialize with real system components
    real_agi = RealAGISystem()
    blog_scheduler = OptimizedBlogScheduler()
    
    mad_scientist_system = MadScientistSystem(real_agi, blog_scheduler)
    
    # Test running a mad scientist cycle with timeout
    result = await run_with_timeout(
        mad_scientist_system.run_mad_scientist_cycle(
            domain="quantum computing",
            focus_area="quantum consciousness"
        ),
        timeout_seconds=30
    )
    
    assert result is not None, "Mad scientist cycle should complete"
    print(f"Mad scientist cycle result: {result}")
    
    # Get metrics
    metrics = mad_scientist_system.get_mad_scientist_metrics()
    print(f"Mad scientist metrics: {metrics}")
    
    print("Mad Scientist System Integration test completed.\n")
    
    # Cleanup after the test
    await real_agi.cleanup_resources()


@pytest.mark.asyncio
async def test_mad_scientist_extended_session(mock_llm_responses):
    """Test the extended mad scientist session with real components."""
    print("Testing Extended Mad Scientist Session with real components...")
    
    # Initialize with real system components
    real_agi = RealAGISystem()
    blog_scheduler = OptimizedBlogScheduler()
    
    mad_scientist_system = MadScientistSystem(real_agi, blog_scheduler)
    
    # Test running an extended session with timeout
    result = await run_with_timeout(
        mad_scientist_system.run_extended_mad_scientist_session(
            domain="machine learning",
            focus_area="self-improving systems",
            cycles=2  # Reduced to 2 cycles to prevent long execution
        ),
        timeout_seconds=60  # 1 minute timeout
    )
    
    assert result is not None, "Extended session should complete"
    print(f"Extended mad scientist session completed {len(result) if result else 0} cycles")
    
    print("Extended Mad Scientist Session test completed.\n")
    
    # Cleanup after the test
    await real_agi.cleanup_resources()


if __name__ == "__main__":
    # Configure logging to avoid excessive output
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests with pytest
    pytest.main([__file__, "-v"])