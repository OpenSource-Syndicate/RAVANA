#!/usr/bin/env python3
"""
Test script for the Mad Scientist System in RAVANA AGI.
This script tests the core functionality of the mad scientist components.
"""
import asyncio
import sys
import os
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from modules.mad_scientist_system import MadScientistSystem
from modules.mad_scientist_impossible_projects import MadScientistModule
from modules.innovation_publishing_system import InnovationPublishingSystem
from core.config import Config


class MockBlogScheduler:
    """Mock blog scheduler for testing."""
    
    async def register_learning_event(self, **kwargs):
        print(f"Mock blog scheduler would register: {kwargs.get('topic', 'Unknown topic')}")
        return True


class MockAGISystem:
    """Mock AGI system for testing."""
    
    def __init__(self):
        self.config = Config()
        from core.enhanced_memory_service import enhanced_memory_service
        self.memory_service = enhanced_memory_service
        from services.knowledge_service import KnowledgeService
        from database.engine import get_engine
        self.knowledge_service = KnowledgeService(get_engine())


def mock_async_safe_call_llm(prompt: str, timeout: int = 30, retries: int = 3, backoff_factor: float = 1.0, **kwargs) -> str:
    """
    Mock implementation of async_safe_call_llm that returns predictable responses for testing
    without requiring an external LLM service.
    """
    if "impossible project" in prompt.lower() or "generate a specific" in prompt.lower():
        return """
        {
            "name": "Test Impossible Project",
            "description": "A project that tests core functionality",
            "impossibility_reason": "Based on current limitations",
            "initial_approach": "Use mock implementation",
            "risk_level": 0.5
        }
        """
    elif "design a detailed implementation plan" in prompt.lower():
        return """
        {
            "steps": ["Step 1: Initialize", "Step 2: Process", "Step 3: Complete"],
            "expected_failures": ["Resource constraints"],
            "potential_alternatives": ["Optimize resource usage", "Reduce scope"],
            "required_resources": ["Computation", "Knowledge"]
        }
        """
    elif "simulate the execution of this step" in prompt.lower():
        return """
        {
            "success": false,
            "result": "Step executed with partial success",
            "failure_reason": "Mock implementation limitation",
            "unexpected_insights": ["Consider alternative approach"]
        }
        """
    elif "analyze the failure of this impossible project" in prompt.lower():
        return """
        {
            "failure_summary": "Project failed due to fundamental limitations",
            "insights_gained": ["Learned about limitations", "Discovered alternative paths"],
            "new_impossibility_understanding": "Confirmed core assumptions",
            "alternative_approaches_hinted": ["Modify constraints", "Change approach"],
            "related_applications": ["Other AGI projects"]
        }
        """
    elif "identify alternative approaches" in prompt.lower():
        return "1. Change the approach entirely\\n2. Reduce scope\\n3. Use different resources"
    elif "assess the importance and publishability" in prompt.lower():
        return """
        {
            "importance": 0.7,
            "title": "Test Discovery",
            "summary": "A discovery made during testing",
            "tags": ["test", "discovery", "agi"]
        }
        """
    elif "create detailed, publication-ready content" in prompt.lower():
        return "# Test Publication\\n\\nThis is detailed content for testing."
    else:
        # Default response for any other prompts
        return "Mock response for: " + prompt[:100] + "..."


async def test_impossible_projects_module():
    """Test the impossible projects module."""
    print("Testing Impossible Projects Module...")
    
    # Patch the async_safe_call_llm function to use our mock
    with patch('modules.mad_scientist_impossible_projects.async_safe_call_llm', mock_async_safe_call_llm):
        mock_agi = MockAGISystem()
        blog_scheduler = MockBlogScheduler()
        
        impossible_projects = MadScientistModule(mock_agi, blog_scheduler)
        
        # Test generating an impossible project
        project = await impossible_projects.generate_impossible_project("artificial general intelligence")
        print(f"Generated impossible project: {project.name}")
        print(f"Description: {project.description}")
        print(f"Impossibility reason: {project.impossibility_reason}")
        print(f"Initial approach: {project.initial_approach}")
        
        # Test attempting the impossible project
        result = await impossible_projects.attempt_impossible_project(project.id)
        print(f"Attempt result: {result}")
    
    print("Impossible Projects Module test completed.\n")


async def test_innovation_publishing_system():
    """Test the innovation publishing system."""
    print("Testing Innovation Publishing System...")
    
    # Patch the async_safe_call_llm function to use our mock
    with patch('modules.innovation_publishing_system.async_safe_call_llm', mock_async_safe_call_llm), \
         patch('modules.mad_scientist_impossible_projects.async_safe_call_llm', mock_async_safe_call_llm):
        mock_agi = MockAGISystem()
        blog_scheduler = MockBlogScheduler()
        
        publisher = InnovationPublishingSystem(mock_agi, blog_scheduler)
        
        # Test identifying an innovation for publication
        publication = await publisher.identify_innovation_for_publication(
            discovery_type="breakthrough",
            discovery_content="Developed a new approach to consciousness transfer",
            context="In research on AI self-modeling"
        )
        
        if publication:
            print(f"Identified publication: {publication.title}")
            print(f"Importance: {publication.importance}")
            print(f"Tags: {publication.tags}")
        else:
            print("No significant innovation identified for publication")
    
    print("Innovation Publishing System test completed.\n")


async def test_mad_scientist_system_integration():
    """Test the full mad scientist system integration."""
    print("Testing Mad Scientist System Integration...")
    
    # Patch both modules to use our mock
    with patch('modules.mad_scientist_impossible_projects.async_safe_call_llm', mock_async_safe_call_llm), \
         patch('modules.innovation_publishing_system.async_safe_call_llm', mock_async_safe_call_llm), \
         patch('core.llm.async_safe_call_llm', mock_async_safe_call_llm):
        mock_agi = MockAGISystem()
        blog_scheduler = MockBlogScheduler()
        
        mad_scientist_system = MadScientistSystem(mock_agi, blog_scheduler)
        
        # Test running a mad scientist cycle
        result = await mad_scientist_system.run_mad_scientist_cycle(
            domain="quantum computing",
            focus_area="quantum consciousness"
        )
        
        print(f"Mad scientist cycle result: {result}")
        
        # Get metrics
        metrics = mad_scientist_system.get_mad_scientist_metrics()
        print(f"Mad scientist metrics: {metrics}")
    
    print("Mad Scientist System Integration test completed.\n")


async def main():
    """Run all tests."""
    print("Starting Mad Scientist System tests...\n")
    
    # Run tests with LLM mocking to prevent hanging
    with patch('modules.mad_scientist_impossible_projects.async_safe_call_llm', mock_async_safe_call_llm), \
         patch('modules.innovation_publishing_system.async_safe_call_llm', mock_async_safe_call_llm):
        await test_impossible_projects_module()
        await test_innovation_publishing_system()
        await test_mad_scientist_system_integration()
    
    print("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())