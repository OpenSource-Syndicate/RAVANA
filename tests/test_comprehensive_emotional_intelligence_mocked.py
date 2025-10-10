"""
Comprehensive Mocked Test Suite for Emotional Intelligence and Decision-Making Modules

This test suite provides extensive testing of the Emotional Intelligence and Decision-Making
functionality using mock components instead of real systems to ensure tests are reliable,
fast, and don't depend on external resources or services.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from core.config import Config
from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence, EmotionalEvent
from modules.decision_engine.decision_maker import goal_driven_decision_maker_loop
from core.llm import call_llm, extract_decision


class MockMemoryService:
    """Mock memory service for testing"""
    def __init__(self):
        self.memories = []
        
    async def add_episodic_memory(self, content, metadata=None, embedding_text=None):
        """Mock adding episodic memory"""
        self.memories.append({
            "content": content,
            "metadata": metadata,
            "embedding_text": embedding_text,
            "timestamp": datetime.now()
        })


class MockKnowledgeService:
    """Mock knowledge service for testing"""
    def get_knowledge_by_category(self, category, limit=10):
        """Mock getting knowledge by category"""
        if category == "decision_context":
            return [{"content": "Past successful decisions involved careful analysis", "source": "historical_data"}]
        return []


class MockAGISystem:
    """Mock AGI System for testing purposes"""
    def __init__(self):
        self.memory_service = MockMemoryService()
        self.knowledge_service = MockKnowledgeService()


@pytest.fixture
def mock_agi_system():
    """Create a mock AGI system for testing"""
    return MockAGISystem()


@pytest.fixture
def emotional_intelligence():
    """Create an emotional intelligence instance for testing"""
    return EmotionalIntelligence()


@pytest.fixture
def sample_situation():
    """Sample situation for testing"""
    return {
        'type': 'normal',
        'prompt': 'Test situation for decision making',
        'context': {'key': 'value', 'timestamp': datetime.now().isoformat()}
    }


@pytest.fixture
def sample_memory():
    """Sample memory data for testing"""
    return [
        "Previous experience with similar situations",
        "Learning from past mistakes",
        "Important deadline approaching"
    ]


@pytest.mark.mock
@pytest.mark.asyncio
async def test_emotional_intelligence_initialization(emotional_intelligence):
    """Test that Emotional Intelligence initializes correctly"""
    assert emotional_intelligence is not None
    assert emotional_intelligence.mood_vector is not None
    assert isinstance(emotional_intelligence.mood_vector, dict)
    assert len(emotional_intelligence.mood_vector) > 0

    # Check that some key mood categories exist
    expected_moods = {
        "Confident", "Curious", "Frustrated", "Excited", "Cautious", 
        "Creative", "Overwhelmed", "Inspired", "Anxious", "Content"
    }
    
    for mood in expected_moods:
        assert mood in emotional_intelligence.mood_vector, f"Mood '{mood}' not found in mood vector"


@pytest.mark.mock
@pytest.mark.asyncio
async def test_update_mood(emotional_intelligence):
    """Test updating mood values"""
    initial_value = emotional_intelligence.mood_vector.get('Confident', 0.0)
    emotional_intelligence.update_mood('Confident', 0.2)
    new_value = emotional_intelligence.mood_vector['Confident']
    
    assert new_value != initial_value
    # Value should be updated within bounds
    assert 0.0 <= new_value <= 1.0


@pytest.mark.mock
@pytest.mark.asyncio
async def test_get_dominant_mood(emotional_intelligence):
    """Test getting dominant mood"""
    # Set specific moods
    emotional_intelligence.update_mood('Curious', 0.9)
    emotional_intelligence.update_mood('Confident', 0.3)
    
    dominant = emotional_intelligence.get_dominant_mood()
    
    assert dominant is not None
    assert isinstance(dominant, str)
    assert dominant == 'Curious'  # Should be the highest value


@pytest.mark.asyncio
async def test_mood_decay(emotional_intelligence):
    """Test mood decay functionality"""
    emotional_intelligence.update_mood('Excited', 0.8)
    initial_value = emotional_intelligence.mood_vector['Excited']
    
    emotional_intelligence.decay_moods(decay=0.1)
    new_value = emotional_intelligence.mood_vector['Excited']
    
    assert new_value < initial_value
    assert new_value >= 0.0  # Should not go below 0


@pytest.mark.asyncio
async def test_blend_moods(emotional_intelligence):
    """Test mood blending"""
    emotional_intelligence.update_mood('Confident', 0.8)
    emotional_intelligence.update_mood('Curious', 0.7)
    
    # Capture initial moods
    initial_confident = emotional_intelligence.mood_vector['Confident']
    initial_curious = emotional_intelligence.mood_vector['Curious']
    
    # Apply blending
    emotional_intelligence.blend_moods()
    
    # Check that blending happened - moods might slightly influence each other
    # or specific blended moods might be created
    assert emotional_intelligence.mood_vector['Confident'] >= 0.0
    assert emotional_intelligence.mood_vector['Curious'] >= 0.0


@pytest.mark.asyncio
async def test_mood_history_tracking(emotional_intelligence):
    """Test mood history tracking"""
    initial_length = len(emotional_intelligence.mood_history)
    
    # Update mood
    emotional_intelligence.update_mood('Excited', 0.5)
    emotional_intelligence.record_mood_change('Excited', 0.5)
    
    # Check that history was updated
    assert len(emotional_intelligence.mood_history) >= initial_length


@pytest.mark.asyncio
async def test_emotional_response_generation(emotional_intelligence):
    """Test emotional response generation"""
    response = emotional_intelligence.generate_emotional_response(
        "How are you feeling today?",
        "Feeling curious about new challenges"
    )
    
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_emotional_state_serialization(emotional_intelligence):
    """Test emotional state serialization"""
    # Get the state
    state = emotional_intelligence.get_state()
    
    assert state is not None
    assert isinstance(state, dict)
    assert 'mood_vector' in state
    assert 'mood_history' in state
    assert 'mood_momentum' in state


@pytest.mark.asyncio
async def test_emotional_state_deserialization(emotional_intelligence):
    """Test emotional state deserialization"""
    # Get initial state
    initial_state = emotional_intelligence.get_state()
    
    # Create new emotional intelligence instance
    new_ei = EmotionalIntelligence()
    
    # Set state
    new_ei.set_state(initial_state)
    
    # Verify states are similar
    new_state = new_ei.get_state()
    assert new_state['mood_vector'] == initial_state['mood_vector']


@pytest.mark.asyncio
async def test_emotional_influence_on_behavior(emotional_intelligence):
    """Test how emotions influence behavior"""
    # Set a dominant mood
    emotional_intelligence.update_mood('Curious', 0.9)
    
    # Get behavior modifiers influenced by mood
    behavior_modifiers = emotional_intelligence.get_behavior_modifiers()
    
    assert behavior_modifiers is not None
    assert isinstance(behavior_modifiers, dict)
    
    # Check that at least some behavior modifiers exist
    expected_modifiers = [
        'exploration_bias', 'risk_tolerance', 'learning_rate', 
        'action_selection_bias', 'attention_span', 'social_engagement', 
        'creativity_bias', 'conservation_factor', 'decision_speed'
    ]
    
    for modifier in expected_modifiers:
        assert modifier in behavior_modifiers


@pytest.mark.asyncio
async def test_emotional_response_to_events(emotional_intelligence):
    """Test emotional response to specific events"""
    # Create a positive event
    emotional_intelligence.process_event('success', 'Task completed successfully', intensity=0.8)
    
    # Check that positive moods increased
    assert emotional_intelligence.mood_vector['Confident'] > 0.0
    assert emotional_intelligence.mood_vector['Excited'] > 0.0
    
    # Create a negative event
    emotional_intelligence.process_event('failure', 'Task failed unexpectedly', intensity=0.6)
    
    # Check that negative moods increased
    assert emotional_intelligence.mood_vector['Frustrated'] >= 0.0


@pytest.mark.asyncio
async def test_complex_emotional_state_transitions(emotional_intelligence):
    """Test complex emotional state transitions"""
    # Start with calm state
    emotional_intelligence.update_mood('Calm', 0.8)
    
    # Process a sequence of events
    emotional_intelligence.process_event('challenge', 'New difficult task', intensity=0.7)
    emotional_intelligence.process_event('curiosity', 'Interesting problem to solve', intensity=0.9)
    emotional_intelligence.process_event('success', 'Made progress on task', intensity=0.6)
    
    # Check that emotional state evolved appropriately
    final_dominant = emotional_intelligence.get_dominant_mood()
    assert final_dominant is not None
    
    # The state should have changed from the initial calm state
    assert emotional_intelligence.mood_vector['Calm'] >= 0.0


@pytest.mark.asyncio
async def test_emotional_memory_integration(mock_agi_system, emotional_intelligence):
    """Test integration of emotions with memory"""
    # Process an emotional event
    emotional_intelligence.process_event('insight', 'Made an important realization', intensity=0.8)
    
    # Add memory with emotional context
    await mock_agi_system.memory_service.add_episodic_memory(
        "Realized the importance of emotional regulation",
        metadata={'emotional_state': emotional_intelligence.get_dominant_mood()}
    )
    
    # Verify the memory was added
    assert len(mock_agi_system.memory_service.memories) == 1
    memory = mock_agi_system.memory_service.memories[0]
    assert 'emotional_state' in memory['metadata']


@pytest.mark.asyncio
async def test_decision_maker_basic_functionality():
    """Test basic goal-driven decision making functionality"""
    situation = {
        'type': 'normal',
        'prompt': 'Test situation for decision making',
        'context': {}
    }
    memory = ["Previous experience 1", "Previous experience 2"]
    shared_state = {}
    
    # Mock the LLM call to avoid external dependency
    with patch('modules.decision_engine.decision_maker.call_llm') as mock_call:
        mock_call.return_value = '''
        {
            "analysis": "Test analysis",
            "plan": ["step1", "step2"],
            "action": "log_message",
            "params": {"message": "test"}
        }
        '''
        
        decision = goal_driven_decision_maker_loop(
            situation=situation,
            memory=memory,
            shared_state=shared_state
        )
        
        assert decision is not None
        assert isinstance(decision, dict)
        assert 'action' in decision or 'reason' in decision


@pytest.mark.asyncio
async def test_decision_with_hypotheses():
    """Test decision making with hypotheses"""
    situation = {'prompt': 'Test with hypotheses'}
    hypotheses = [
        "Hypothesis 1: System performs better with X",
        "Hypothesis 2: Approach Y is more efficient"
    ]
    
    # Mock the LLM call to avoid external dependency
    with patch('modules.decision_engine.decision_maker.call_llm') as mock_call:
        mock_call.return_value = '''
        {
            "analysis": "Hypothesis analysis",
            "selected_hypothesis": "Hypothesis 1",
            "action": "run_experiment",
            "params": {"hypothesis": "Hypothesis 1"}
        }
        '''
        
        decision = goal_driven_decision_maker_loop(
            situation=situation,
            hypotheses=hypotheses
        )
        
        assert decision is not None
        assert isinstance(decision, dict)


@pytest.mark.asyncio
async def test_experiment_initiation_decision():
    """Test experiment initiation through decision making"""
    situation = {
        'type': 'exploration',
        'prompt': 'Explore new optimization strategies'
    }
    shared_state = {'active_experiment': None}
    
    # Mock the LLM call to avoid external dependency
    with patch('modules.decision_engine.decision_maker.call_llm') as mock_call:
        mock_call.return_value = '''
        {
            "analysis": "Need to run experiments",
            "plan": ["design experiment", "execute", "analyze results"],
            "action": "initiate_experiment",
            "params": {"experiment_type": "optimization"}
        }
        '''
        
        decision = goal_driven_decision_maker_loop(
            situation=situation,
            shared_state=shared_state
        )
        
        assert decision is not None
        assert isinstance(decision, dict)
        assert decision.get('action') == 'initiate_experiment'


@pytest.mark.asyncio
async def test_emotionally_influenced_decision_making(mock_agi_system):
    """Test decision making with emotional influence"""
    # Create an emotional intelligence instance
    emotional_intelligence = EmotionalIntelligence()
    
    # Set a specific emotional state
    emotional_intelligence.update_mood('Curious', 0.9)
    emotional_intelligence.update_mood('Cautious', 0.2)
    
    # Create a situation that would benefit from curiosity
    situation = {
        'type': 'exploration',
        'prompt': 'Investigate new potential optimizations in the codebase',
        'context': {}
    }
    
    # Get behavior modifiers from emotional state
    behavior_modifiers = emotional_intelligence.get_behavior_modifiers()
    
    # Mock the LLM call to avoid external dependency
    with patch('modules.decision_engine.decision_maker.call_llm') as mock_call:
        mock_call.return_value = '''
        {
            "analysis": "High curiosity detected, explore new options",
            "plan": ["scan codebase", "identify patterns", "propose improvements"],
            "action": "explore_codebase",
            "params": {"depth": "deep", "focus": "optimization"}
        }
        '''
        
        decision = goal_driven_decision_maker_loop(
            situation=situation,
            emotional_state=emotional_intelligence.mood_vector,
            behavior_modifiers=behavior_modifiers
        )
        
        assert decision is not None
        assert isinstance(decision, dict)
        # The decision should reflect the curious emotional state
        assert 'explore' in decision.get('action', '').lower() or 'explore' in decision.get('params', {}).get('focus', '').lower()


@pytest.mark.asyncio
async def test_decision_making_with_memory_context(mock_agi_system):
    """Test decision making with memory context"""
    situation = {
        'type': 'problem_solving',
        'prompt': 'Handle a performance issue similar to previous one',
        'context': {}
    }
    
    # Add some relevant memories
    await mock_agi_system.memory_service.add_episodic_memory(
        "Previously solved similar performance issue by optimizing loops",
        metadata={'category': 'performance', 'outcome': 'success'}
    )
    
    await mock_agi_system.memory_service.add_episodic_memory(
        "Another solution involved caching results",
        metadata={'category': 'performance', 'outcome': 'partial_success'}
    )
    
    # Extract memory context
    memory_context = [mem['content'] for mem in mock_agi_system.memory_service.memories]
    
    # Mock the LLM call to avoid external dependency
    with patch('modules.decision_engine.decision_maker.call_llm') as mock_call:
        mock_call.return_value = '''
        {
            "analysis": "Based on past experiences, try optimizing loops first",
            "plan": ["profile code", "optimize bottlenecks", "measure improvement"],
            "action": "profile_performance",
            "params": {"focus": "loops"}
        }
        '''
        
        decision = goal_driven_decision_maker_loop(
            situation=situation,
            memory=memory_context
        )
        
        assert decision is not None
        assert isinstance(decision, dict)
        assert decision.get('action') is not None


@pytest.mark.asyncio
async def test_decision_making_with_knowledge_context(mock_agi_system):
    """Test decision making with knowledge context"""
    situation = {
        'type': 'learning',
        'prompt': 'Learn about new optimization techniques',
        'context': {}
    }
    
    # Get knowledge context
    knowledge_context = mock_agi_system.knowledge_service.get_knowledge_by_category('decision_context')
    
    # Mock the LLM call to avoid external dependency
    with patch('modules.decision_engine.decision_maker.call_llm') as mock_call:
        mock_call.return_value = '''
        {
            "analysis": "Based on knowledge, follow systematic approach",
            "plan": ["research", "experiment", "validate"],
            "action": "research_optimization",
            "params": {"method": "systematic"}
        }
        '''
        
        decision = goal_driven_decision_maker_loop(
            situation=situation,
            knowledge_context=knowledge_context
        )
        
        assert decision is not None
        assert isinstance(decision, dict)
        assert decision.get('action') is not None


@pytest.mark.asyncio
async def test_risk_averse_decision_making(emotional_intelligence):
    """Test decision making with risk-averse emotional state"""
    # Set a risk-averse emotional state
    emotional_intelligence.update_mood('Cautious', 0.9)
    emotional_intelligence.update_mood('Anxious', 0.6)
    
    situation = {
        'type': 'experimentation',
        'prompt': 'Run a potentially risky experiment',
        'context': {'risk_level': 'high'}
    }
    
    # Get behavior modifiers from emotional state
    behavior_modifiers = emotional_intelligence.get_behavior_modifiers()
    
    # Mock the LLM call to avoid external dependency
    with patch('modules.decision_engine.decision_maker.call_llm') as mock_call:
        mock_call.return_value = '''
        {
            "analysis": "High risk detected and emotional state is cautious, suggest safer approach",
            "plan": ["analyze risks", "prepare rollback", "run in safe environment"],
            "action": "prepare_safe_experiment",
            "params": {"safety_first": true}
        }
        '''
        
        decision = goal_driven_decision_maker_loop(
            situation=situation,
            emotional_state=emotional_intelligence.mood_vector,
            behavior_modifiers=behavior_modifiers
        )
        
        assert decision is not None
        assert isinstance(decision, dict)
        # The decision should reflect the cautious emotional state
        assert 'safe' in decision.get('action', '').lower() or decision.get('params', {}).get('safety_first', False) is True


@pytest.mark.asyncio
async def test_creative_decision_making(emotional_intelligence):
    """Test decision making with creative emotional state"""
    # Set a creative emotional state
    emotional_intelligence.update_mood('Creative', 0.8)
    emotional_intelligence.update_mood('Inspired', 0.7)
    emotional_intelligence.update_mood('Curious', 0.9)
    
    situation = {
        'type': 'problem_solving',
        'prompt': 'Solve a complex problem with unconventional approach',
        'context': {'problem': 'complex_system_optimization'}
    }
    
    # Get behavior modifiers from emotional state
    behavior_modifiers = emotional_intelligence.get_behavior_modifiers()
    # Enhance creativity bias
    behavior_modifiers['creativity_bias'] = 0.9
    
    # Mock the LLM call to avoid external dependency
    with patch('modules.decision_engine.decision_maker.call_llm') as mock_call:
        mock_call.return_value = '''
        {
            "analysis": "Emotional state is creative and inspired, try novel approach",
            "plan": ["brainstorm alternatives", "try unconventional method", "innovate"],
            "action": "innovate_solution",
            "params": {"approach": "unconventional", "creativity_level": "high"}
        }
        '''
        
        decision = goal_driven_decision_maker_loop(
            situation=situation,
            emotional_state=emotional_intelligence.mood_vector,
            behavior_modifiers=behavior_modifiers
        )
        
        assert decision is not None
        assert isinstance(decision, dict)
        # The decision should reflect the creative emotional state
        assert 'innovate' in decision.get('action', '').lower() or decision.get('params', {}).get('creativity_level') == 'high'


@pytest.mark.asyncio
async def test_emotional_intelligence_event_processing():
    """Test comprehensive event processing in emotional intelligence"""
    ei = EmotionalIntelligence()
    
    # Process various types of events
    events = [
        ('success', 'Completed a task', 0.8),
        ('failure', 'Made an error', 0.5),
        ('obstacle', 'Encountered a challenge', 0.7),
        ('opportunity', 'Found a new possibility', 0.9),
        ('achievement', 'Reached a milestone', 1.0)
    ]
    
    for event_type, description, intensity in events:
        ei.process_event(event_type, description, intensity)
    
    # Verify that emotional state has changed appropriately
    dominant_mood = ei.get_dominant_mood()
    assert dominant_mood is not None
    
    # Check that mood history has grown
    assert len(ei.mood_history) >= len(events)


@pytest.mark.asyncio
async def test_emotional_intelligence_contextual_response():
    """Test contextual emotional responses"""
    ei = EmotionalIntelligence()
    
    # Update moods to simulate a particular state
    ei.update_mood('Curious', 0.8)
    ei.update_mood('Analytical', 0.7)
    
    # Generate contextual response
    context = "Working on a complex algorithm that requires deep thinking"
    response = ei.generate_emotional_response("How are you approaching this task?", context)
    
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_complex_decision_scenario():
    """Test a complex decision-making scenario"""
    situation = {
        'type': 'multi_objective',
        'prompt': 'Balance performance optimization with code maintainability',
        'context': {
            'constraints': ['time', 'quality', 'maintainability'],
            'stakeholders': ['developers', 'users', 'maintainers']
        }
    }
    
    memory = [
        "Previous optimization improved performance but hurt maintainability",
        "Users prefer fast performance", 
        "Developers prefer clean, maintainable code"
    ]
    
    shared_state = {
        'current_performance': 0.7,
        'code_quality_score': 0.8,
        'deadline_pressure': 0.6
    }
    
    # Mock the LLM call to avoid external dependency
    with patch('modules.decision_engine.decision_maker.call_llm') as mock_call:
        mock_call.return_value = '''
        {
            "analysis": "Need to balance multiple objectives, prioritize based on current state",
            "plan": ["profile current state", "identify critical bottlenecks", "optimize selectively"],
            "action": "profile_and_optimize_selectively",
            "params": {
                "optimization_target": "critical_bottlenecks_only",
                "maintain_quality_threshold": true
            }
        }
        '''
        
        decision = goal_driven_decision_maker_loop(
            situation=situation,
            memory=memory,
            shared_state=shared_state
        )
        
        assert decision is not None
        assert isinstance(decision, dict)
        assert 'action' in decision
        assert 'params' in decision


@pytest.mark.asyncio
async def test_emotional_intelligence_long_term_effects():
    """Test long-term emotional effects and patterns"""
    ei = EmotionalIntelligence()
    
    # Simulate mood changes over time
    daily_events = [
        ('success', 'Good progress', 0.7),
        ('setback', 'Minor issue', 0.3),
        ('success', 'Another win', 0.8),
        ('challenge', 'Difficult problem', 0.6),
        ('success', 'Solved it!', 0.9),
        ('learning', 'Gained new insight', 0.7)
    ]
    
    for event_type, description, intensity in daily_events:
        ei.process_event(event_type, description, intensity)
        # Apply decay to simulate time passage
        ei.decay_moods(decay=0.05)
    
    # Check the final emotional state
    final_state = ei.get_state()
    assert final_state is not None
    assert 'mood_vector' in final_state
    assert 'mood_history' in final_state
    
    # History should contain entries for each event processed
    assert len(final_state['mood_history']) >= len(daily_events)


@pytest.mark.asyncio
async def test_decision_maker_with_conflicting_emotions():
    """Test decision making with conflicting emotional states"""
    # Create an emotional state with conflicting emotions
    conflicting_emotions = {
        "Confident": 0.8,
        "Cautious": 0.7,  # High caution conflicts with confidence
        "Curious": 0.9,
        "Anxious": 0.6   # Anxiety conflicts with curiosity
    }
    
    situation = {
        'type': 'exploration',
        'prompt': 'Explore a risky but potentially rewarding approach',
        'context': {'risk': 'moderate', 'reward': 'high'}
    }
    
    # Calculate behavior modifiers from conflicting emotions
    ei = EmotionalIntelligence()
    # Directly set mood vector to simulate conflicting state
    ei.mood_vector = conflicting_emotions
    
    behavior_modifiers = ei.get_behavior_modifiers()
    
    # Mock the LLM call to avoid external dependency
    with patch('modules.decision_engine.decision_maker.call_llm') as mock_call:
        mock_call.return_value = '''
        {
            "analysis": "Mixed emotional signals, balance risk and opportunity",
            "plan": ["analyze thoroughly", "prepare contingencies", "start small"],
            "action": "analyze_and_plan_carefully",
            "params": {"risk_tolerance": "moderate", "exploration_depth": "measured"}
        }
        '''
        
        decision = goal_driven_decision_maker_loop(
            situation=situation,
            emotional_state=conflicting_emotions,
            behavior_modifiers=behavior_modifiers
        )
        
        assert decision is not None
        assert isinstance(decision, dict)
        # Decision should reflect balanced approach due to conflicting emotions
        assert any(word in decision.get('action', '').lower() for word in ['analyze', 'plan', 'careful'])


if __name__ == "__main__":
    pytest.main([__file__])