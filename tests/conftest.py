"""Configuration for pytest in RAVANA AGI system."""
import sys
import os
import pytest
import asyncio
from unittest.mock import patch

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the event loop for all async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm_responses():
    """Mock LLM responses to prevent actual API calls during testing."""
    async def mock_async_safe_call_llm(prompt: str, timeout: int = 30, retries: int = 3, backoff_factor: float = 1.0, **kwargs) -> str:
        """Async mock implementation that returns predictable responses for faster testing."""
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
                "importance": 0.8,
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

    with patch('modules.mad_scientist_impossible_projects.async_safe_call_llm', mock_async_safe_call_llm), \
         patch('modules.innovation_publishing_system.async_safe_call_llm', mock_async_safe_call_llm), \
         patch('core.llm.async_safe_call_llm', mock_async_safe_call_llm):
        yield mock_async_safe_call_llm