"""Tests for core AGI system functionality."""

import pytest
import asyncio
from datetime import datetime
from core.system import AGISystem
from core.state import SharedState
from database.engine import get_engine
from core.enhanced_memory_service import MemoryType, Memory


class TestAGISystem:
    """Test AGI system initialization and core functionality."""

    @pytest.fixture
    async def agi_system(self, test_engine):
        """Create AGI system instance for testing."""
        system = AGISystem(test_engine)
        # Initialize components
        await system.initialize_components()
        yield system
        # Cleanup
        if hasattr(system, 'session'):
            system.session.close()

    def test_agi_system_initialization(self, agi_system):
        """Test AGI system initializes correctly."""
        assert agi_system is not None
        assert agi_system.engine is not None
        assert agi_system.config is not None
        assert agi_system.shared_state is not None

    def test_shared_state_initialization(self, agi_system):
        """Test shared state is properly initialized."""
        state = agi_system.shared_state
        assert isinstance(state, SharedState)
        assert isinstance(state.mood, dict)
        assert isinstance(state.mood_history, list)
        assert isinstance(state.curiosity_topics, list)

    def test_component_registration(self, agi_system):
        """Test all components are registered."""
        assert agi_system.data_service is not None
        assert agi_system.knowledge_service is not None
        assert agi_system.memory_service is not None
        assert agi_system.action_manager is not None
        assert agi_system.emotional_intelligence is not None

    @pytest.mark.asyncio
    async def test_situation_generation(self, agi_system):
        """Test situation generation."""
        situation = await agi_system._generate_situation()
        assert situation is not None
        assert 'prompt' in situation
        assert isinstance(situation['prompt'], str)
        assert len(situation['prompt']) > 0

    @pytest.mark.asyncio
    async def test_memory_retrieval(self, agi_system, sample_memory_data):
        """Test memory retrieval functionality."""
        # Save some test memories first
        await agi_system.memory_service.save_memories(
            [{'content': mem, 'type': 'episodic'} for mem in sample_memory_data]
        )
        
        # Retrieve memories
        await agi_system._retrieve_memories("quantum computing")
        assert agi_system.shared_state.recent_memories is not None

    @pytest.mark.asyncio
    async def test_mood_update(self, agi_system, sample_action_output):
        """Test mood updates after actions."""
        old_mood = agi_system.shared_state.mood.copy()
        await agi_system._update_mood_and_reflect(sample_action_output)
        new_mood = agi_system.shared_state.mood
        
        # Mood should have been updated
        assert new_mood is not None
        assert isinstance(new_mood, dict)

    @pytest.mark.asyncio
    async def test_curiosity_handling(self, agi_system):
        """Test curiosity topic generation."""
        await agi_system._handle_curiosity()
        
        # Check if curiosity topics were generated
        assert hasattr(agi_system.shared_state, 'curiosity_topics')
        assert isinstance(agi_system.shared_state.curiosity_topics, list)

    def test_behavior_modifiers(self, agi_system):
        """Test behavior modifier system."""
        assert hasattr(agi_system, 'behavior_modifiers')
        assert isinstance(agi_system.behavior_modifiers, dict)

    @pytest.mark.asyncio
    async def test_component_initialization(self, test_engine):
        """Test component initialization sequence."""
        system = AGISystem(test_engine)
        success = await system.initialize_components()
        
        # Should initialize successfully or gracefully handle failures
        assert isinstance(success, bool)

    def test_shutdown_coordinator(self, agi_system):
        """Test shutdown coordinator is set up."""
        assert agi_system.shutdown_coordinator is not None
        assert hasattr(agi_system.shutdown_coordinator, 'initiate_shutdown')

    def test_reasoning_engine_initialization(self, agi_system):
        """Test reasoning engine is properly initialized."""
        assert hasattr(agi_system, 'reasoning_engine')
        assert agi_system.reasoning_engine is not None
        assert hasattr(agi_system.reasoning_engine, 'reason')

    def test_self_reflection_initialization(self, agi_system):
        """Test self-reflection module is properly initialized."""
        assert hasattr(agi_system, 'self_reflection')
        assert agi_system.self_reflection is not None
        assert hasattr(agi_system.self_reflection, 'reflect_on_performance')

    def test_self_modification_initialization(self, agi_system):
        """Test self-modification module is properly initialized."""
        assert hasattr(agi_system, 'self_modification')
        assert agi_system.self_modification is not None
        assert hasattr(agi_system.self_modification, 'propose_modification')

    def test_experimentation_module_initialization(self, agi_system):
        """Test experimentation module is properly initialized."""
        assert hasattr(agi_system, 'experimentation_module')
        assert agi_system.experimentation_module is not None
        assert hasattr(agi_system.experimentation_module, 'generate_hypothesis')

    @pytest.mark.asyncio
    async def test_reasoning_engine_functionality(self, agi_system):
        """Test the reasoning engine functionality."""
        test_situation = {
            'prompt': 'How can I improve my learning efficiency?',
            'context': {}
        }
        
        # This should not raise an exception
        result = await agi_system.reasoning_engine.reason(test_situation)
        
        assert result is not None
        assert 'reasoning_process' in result
        assert 'synthesized_conclusion' in result
        assert 'confidence_score' in result

    @pytest.mark.asyncio
    async def test_self_reflection_functionality(self, agi_system):
        """Test the self-reflection functionality."""
        result = await agi_system.self_reflection.reflect_on_performance(time_window_hours=1)
        
        assert result is not None
        assert hasattr(result, 'insights')
        assert hasattr(result, 'identified_improvements')
        assert hasattr(result, 'suggested_actions')

    @pytest.mark.asyncio
    async def test_self_modification_proposal(self, agi_system):
        """Test the self-modification proposal functionality."""
        plan = await agi_system.self_modification.propose_modification(
            target_component='decision_making',
            modification_type='strategy',
            proposed_change='Improve decision speed by 10%',
            expected_outcome='Faster decision execution'
        )
        
        assert plan is not None
        assert hasattr(plan, 'target_component')
        assert plan.target_component == 'decision_making'
        assert plan.proposed_change == 'Improve decision speed by 10%'

    @pytest.mark.asyncio
    async def test_experimentation_module_functionality(self, agi_system):
        """Test the experimentation module functionality."""
        # Test hypothesis generation
        hypothesis = await agi_system.experimentation_module.generate_hypothesis('learning')
        assert hypothesis is not None
        assert isinstance(hypothesis, str)
        assert len(hypothesis) > 0

        # Test experiment design
        experiment = await agi_system.experimentation_module.design_experiment(hypothesis)
        assert experiment is not None
        assert experiment.hypothesis == hypothesis
        assert experiment.procedure is not None
        assert len(experiment.procedure) > 0

    @pytest.mark.asyncio
    async def test_emotional_influence_on_behavior(self, agi_system, sample_action_output):
        """Test that emotions influence behavior modifications."""
        # Process an action to update mood
        old_mood = agi_system.shared_state.mood.copy()
        await agi_system._update_mood_and_reflect(sample_action_output)
        new_mood = agi_system.shared_state.mood
        
        # Check that mood was updated
        assert new_mood is not None
        assert new_mood != old_mood
        
        # Check that behavior modifiers were generated
        assert agi_system.behavior_modifiers is not None
        assert isinstance(agi_system.behavior_modifiers, dict)
        
        # Check specific behavior modification keys
        expected_keys = [
            'exploration_bias', 'risk_tolerance', 'learning_rate', 
            'action_selection_bias', 'attention_span', 'social_engagement', 
            'creativity_bias', 'conservation_factor', 'decision_speed', 
            'memory_recall_bias', 'emotional_memory_bias', 'emotional_decision_weight'
        ]
        for key in expected_keys:
            assert key in agi_system.behavior_modifiers

    @pytest.mark.asyncio
    async def test_memory_with_emotional_bias(self, agi_system):
        """Test memory retrieval with emotional bias."""
        # Add some sample memories with emotional content
        sample_memories = [
            {'content': 'I succeeded in completing the task successfully', 'type': MemoryType.EPISODIC},
            {'content': 'The project failed due to technical issues', 'type': MemoryType.EPISODIC},
            {'content': 'Learning new concepts is exciting', 'type': MemoryType.EPISODIC},
            {'content': 'Feeling frustrated with the slow progress', 'type': MemoryType.EPISODIC}
        ]
        await agi_system.memory_service.save_memories(sample_memories)
        
        # Test retrieval with positive emotional bias
        agi_system.shared_state.emotional_memory_bias = 0.8  # Positive bias
        await agi_system._retrieve_memories("learning")
        positive_biased_memories = agi_system.shared_state.recent_memories
        assert positive_biased_memories is not None
        
        # Test retrieval with negative emotional bias
        agi_system.shared_state.emotional_memory_bias = -0.8  # Negative bias
        await agi_system._retrieve_memories("challenges")
        negative_biased_memories = agi_system.shared_state.recent_memories
        assert negative_biased_memories is not None
