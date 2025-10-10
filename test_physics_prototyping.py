"""
Test script for the new Physics Prototyping System integration with Mad Scientist.
"""
import pytest
import pytest_asyncio
import asyncio
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from modules.physics_prototyping_system import PhysicsPrototypingSystem
from modules.physics_analysis_system import PhysicsDomain
from modules.mad_scientist_system import MadScientistSystem


class MockBlogScheduler:
    async def register_learning_event(self, **kwargs):
        print(f"Blog event registered: {kwargs.get('topic', 'Unknown topic')}")
        return True


@pytest_asyncio.fixture
async def mock_blog_scheduler():
    """Create a mock blog scheduler for testing."""
    return MockBlogScheduler()


@pytest_asyncio.fixture
async def physics_prototyper(mock_blog_scheduler):
    """Create a physics prototyper instance with mock dependencies."""
    prototyper = PhysicsPrototypingSystem(None, mock_blog_scheduler)
    
    # Set up a mock AGI system
    prototyper.agi_system = type('MockAGISystem', (), {
        'memory_service': type('MockMemoryService', (), {
            'retrieve_relevant_memories': lambda query, top_k=5: []
        })(),
        'knowledge_service': type('MockKnowledgeService', (), {
            'add_knowledge': lambda content, source, category: None
        })()
    })()
    
    return prototyper


@pytest.mark.asyncio
async def test_basic_physics_simulation(physics_prototyper):
    """Test the basic physics simulation functionality."""
    result = await physics_prototyper.prototype_physics_experiment(
        "Test hypothesis: How does a ball fall under gravity?",
        PhysicsDomain.MECHANICS
    )
    
    assert result.success is True
    assert len(result.data) > 0
    assert result.simulation_type is not None
    assert result.analysis is not None


@pytest.mark.asyncio
async def test_quantum_mechanics_simulation(physics_prototyper):
    """Test quantum mechanics simulation."""
    from modules.physics_prototyping_system import SimulationType
    
    result = await physics_prototyper.prototype_physics_experiment(
        "Quantum tunneling through a potential barrier",
        PhysicsDomain.QUANTUM_MECHANICS
    )
    
    assert result.success is True
    assert len(result.data) > 0
    # Quantum mechanics experiments often default to theoretical physics
    assert result.simulation_type in [SimulationType.QUANTUM_SIMULATION, 
                                    SimulationType.THEORETICAL_PHYSICS]


@pytest.mark.asyncio
async def test_impossible_physics_simulation(physics_prototyper):
    """Test simulation of impossible physics concepts."""
    result = await physics_prototyper.prototype_physics_experiment(
        "Perpetual motion machine that violates conservation of energy",
        PhysicsDomain.THERMODYNAMICS
    )
    
    assert result.success is True
    # Impossible physics experiments should typically be refuted or inconclusive
    assert result.analysis.get('support') in ['refute', 'inconclusive', 'impossible']


@pytest.mark.asyncio
async def test_prototyping_metrics(physics_prototyper):
    """Test that metrics are properly calculated."""
    # First run a simulation to generate data
    await physics_prototyper.prototype_physics_experiment(
        "Test hypothesis: How does a ball fall under gravity?",
        PhysicsDomain.MECHANICS
    )
    
    metrics = await physics_prototyper.get_prototyping_metrics()
    
    assert 'total_simulations' in metrics
    assert 'success_rate' in metrics
    assert metrics['total_simulations'] >= 1
    assert 0.0 <= metrics['success_rate'] <= 1.0


@pytest.mark.asyncio
async def test_mad_scientist_integration(mock_blog_scheduler):
    """Test the integration with the mad scientist system."""
    # Create the physics prototyping system that would be part of the AGI system
    physics_prototyper = PhysicsPrototypingSystem(None, mock_blog_scheduler)
    
    # Set up a mock AGI system with the prototyper
    agi_system = type('MockAGISystem', (), {
        'blog_scheduler': mock_blog_scheduler,
        'memory_service': type('MockMemoryService', (), {
            'retrieve_relevant_memories': lambda query, top_k=5: []
        })(),
        'knowledge_service': type('MockKnowledgeService', (), {
            'add_knowledge': lambda content, source, category: None
        })(),
        # Add the physics prototyping system that we'll test integration with
        'physics_prototyping_system': physics_prototyper
    })()
    
    # Set the agi_system reference in the physics prototyper
    physics_prototyper.agi_system = agi_system
    
    # Initialize the mad scientist system
    mad_scientist = MadScientistSystem(agi_system, mock_blog_scheduler)
    
    # Verify the mad scientist system has a reference to the physics prototyper
    assert hasattr(mad_scientist, 'physics_prototyper')
    
    # Verify it's the same instance as the AGI system
    assert mad_scientist.physics_prototyper is physics_prototyper
    
    # Test that it can call the physics prototyping system
    result = await mad_scientist.physics_prototyper.prototype_physics_experiment(
        "Simple test: object falling under gravity",
        PhysicsDomain.MECHANICS
    )
    
    assert result.success is True
    assert result.simulation_type is not None
    
    # Test metrics integration
    metrics = await mad_scientist.get_mad_scientist_metrics()
    assert 'physics_prototyping_metrics' in metrics
    assert metrics['physics_prototyping_metrics'] is not None