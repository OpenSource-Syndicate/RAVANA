"""Integration tests for RAVANA AGI system."""

import pytest
import asyncio
from core.system import AGISystem
from database.engine import get_engine


@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete system workflows."""

    @pytest.fixture
    async def running_system(self, test_engine):
        """Create and initialize a running AGI system."""
        system = AGISystem(test_engine)
        await system.initialize_components()
        yield system
        # Cleanup
        if hasattr(system, 'session'):
            system.session.close()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_iteration_cycle(self, running_system):
        """Test a complete iteration cycle."""
        # Run one iteration
        await running_system.run_iteration()
        
        # Verify state was updated
        assert running_system.shared_state is not None
        assert running_system.shared_state.mood is not None

    @pytest.mark.asyncio
    async def test_memory_knowledge_integration(self, running_system, sample_memory_data):
        """Test integration between memory and knowledge services."""
        # Save memories
        memories = [
            {'content': mem, 'type': 'episodic'}
            for mem in sample_memory_data
        ]
        await running_system.memory_service.save_memories(memories)
        
        # Add knowledge
        knowledge_result = running_system.knowledge_service.add_knowledge(
            "Integration test knowledge",
            "test",
            "integration"
        )
        
        assert knowledge_result is not None
        assert 'summary' in knowledge_result

    @pytest.mark.asyncio
    async def test_decision_action_workflow(self, running_system):
        """Test decision making to action execution workflow."""
        # Generate a situation
        situation = await running_system._generate_situation()
        assert situation is not None
        
        # Make a decision
        await running_system._retrieve_memories(situation['prompt'])
        decision = await running_system._make_decision(situation)
        
        assert decision is not None
        assert 'action' in decision

    @pytest.mark.asyncio
    async def test_emotional_intelligence_integration(self, running_system, sample_action_output):
        """Test emotional intelligence integration with system."""
        initial_mood = running_system.shared_state.mood.copy()
        
        # Process action through emotional intelligence
        running_system.emotional_intelligence.process_action_result(sample_action_output)
        
        # Update system mood
        running_system.shared_state.mood = running_system.emotional_intelligence.get_mood_vector()
        
        # Verify mood was updated
        assert running_system.shared_state.mood != initial_mood

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_background_task_startup(self, running_system):
        """Test background task initialization."""
        # Start background tasks
        tasks = [
            asyncio.create_task(running_system.data_collection_task()),
            asyncio.create_task(running_system.event_detection_task())
        ]
        
        # Let them run briefly
        await asyncio.sleep(2)
        
        # Cancel tasks
        for task in tasks:
            task.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Test passed if no exceptions
        assert True
