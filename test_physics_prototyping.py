"""
Test script for the new Physics Prototyping System integration with Mad Scientist.
"""
import asyncio
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from core.system import AGISystem
from core.config import Config
from modules.physics_prototyping_system import PhysicsPrototypingSystem
from modules.physics_analysis_system import PhysicsDomain


async def test_physics_prototyping():
    """Test the physics prototyping system directly."""
    print("Testing Physics Prototyping System...")
    
    # Create a mock AGI system (simplified version for testing)
    class MockBlogScheduler:
        async def register_learning_event(self, **kwargs):
            print(f"Blog event registered: {kwargs.get('topic', 'Unknown topic')}")
            return True
    
    mock_blog_scheduler = MockBlogScheduler()
    
    # Initialize the physics prototyping system
    agi_system = None  # Will be set after initialization
    prototyper = PhysicsPrototypingSystem(None, mock_blog_scheduler)
    
    # For this test, we'll set the agi_system after creation to avoid circular dependencies
    prototyper.agi_system = type('MockAGISystem', (), {
        'memory_service': type('MockMemoryService', (), {
            'retrieve_relevant_memories': lambda query, top_k=5: []
        })(),
        'knowledge_service': type('MockKnowledgeService', (), {
            'add_knowledge': lambda content, source, category: None
        })()
    })()
    
    print("\n1. Testing basic physics simulation...")
    try:
        # Test with a simple hypothesis
        result = await prototyper.prototype_physics_experiment(
            "Test hypothesis: How does a ball fall under gravity?",
            PhysicsDomain.MECHANICS
        )
        print(f"   Success: {result.success}")
        print(f"   Analysis: {result.analysis.get('conclusion', 'No conclusion')[:100]}...")
        print(f"   Data points: {len(result.data)}")
        print("   [PASS] Physics prototyping test passed")
    except Exception as e:
        print(f"   [FAIL] Physics prototyping test failed: {e}")
        return False
    
    print("\n2. Testing with a quantum mechanics hypothesis...")
    try:
        result = await prototyper.prototype_physics_experiment(
            "Quantum tunneling through a potential barrier",
            PhysicsDomain.QUANTUM_MECHANICS
        )
        print(f"   Success: {result.success}")
        print(f"   Simulation type: {result.simulation_type.value}")
        print(f"   Data points: {len(result.data)}")
        print("   [PASS] Quantum mechanics test passed")
    except Exception as e:
        print(f"   [FAIL] Quantum mechanics test failed: {e}")
        return False
    
    print("\n3. Testing with an impossible physics hypothesis...")
    try:
        result = await prototyper.prototype_physics_experiment(
            "Perpetual motion machine that violates conservation of energy",
            PhysicsDomain.THERMODYNAMICS
        )
        print(f"   Success: {result.success}")
        print(f"   Analysis support: {result.analysis.get('support', 'Unknown')}")
        print(f"   Key observations: {result.analysis.get('key_observations', [])[:2]}")
        print("   [PASS] Impossible physics test passed")
    except Exception as e:
        print(f"   [FAIL] Impossible physics test failed: {e}")
        return False
    
    print("\n4. Testing metrics...")
    try:
        metrics = await prototyper.get_prototyping_metrics()
        print(f"   Total simulations: {metrics['total_simulations']}")
        print(f"   Success rate: {metrics['success_rate']:.2%}")
        print("   [PASS] Metrics test passed")
    except Exception as e:
        print(f"   [FAIL] Metrics test failed: {e}")
        return False
    
    return True


async def test_mad_scientist_integration():
    """Test the integration with the mad scientist system."""
    print("\n\nTesting Mad Scientist System Integration...")
    
    class MockBlogScheduler:
        async def register_learning_event(self, **kwargs):
            print(f"Blog event registered: {kwargs.get('topic', 'Unknown topic')}")
            return True
    
    mock_blog_scheduler = MockBlogScheduler()
    
    try:
        # Create the physics prototyping system that would be part of the AGI system
        from modules.physics_prototyping_system import PhysicsPrototypingSystem
        physics_prototyper = PhysicsPrototypingSystem(None, mock_blog_scheduler)
        
        # Since we're not running in full AGI context, set up a mock AGI system with the prototyper
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
        
        # Initialize just the mad scientist system
        from modules.mad_scientist_system import MadScientistSystem
        mad_scientist = MadScientistSystem(agi_system, mock_blog_scheduler)
        
        # Check if physics prototyper is properly set to the same instance as the AGI system
        if hasattr(mad_scientist, 'physics_prototyper'):
            print("   [PASS] Mad Scientist system has physics prototyper")
            
            # Verify it's the same instance as the AGI system
            if mad_scientist.physics_prototyper is physics_prototyper:
                print("   [PASS] Mad Scientist system is using the same physics prototyper instance as AGI system")
            else:
                print("   [FAIL] Mad Scientist system is not using the same physics prototyper instance")
                return False
            
            # Test that it can call the physics prototyping system
            try:
                result = await mad_scientist.physics_prototyper.prototype_physics_experiment(
                    "Simple test: object falling under gravity",
                    PhysicsDomain.MECHANICS
                )
                print(f"   [PASS] Integration test successful: {result.success}")
                print(f"   Simulation type: {result.simulation_type.value}")
            except Exception as e:
                print(f"   [FAIL] Integration test failed: {e}")
                return False
        else:
            print("   [FAIL] Mad Scientist system missing physics prototyper")
            return False
            
        # Test metrics integration (async method)
        try:
            metrics = await mad_scientist.get_mad_scientist_metrics()
            if 'physics_prototyping_metrics' in metrics:
                print("   [PASS] Physics prototyping metrics integrated in mad scientist system")
            else:
                print("   [FAIL] Physics prototyping metrics not found in mad scientist system")
                return False
        except Exception as e:
            print(f"   [FAIL] Metrics integration test failed: {e}")
            return False
            
    except Exception as e:
        print(f"   [FAIL] Mad Scientist integration test failed: {e}")
        return False
    
    return True


async def main():
    """Run all tests."""
    print("Starting Physics Prototyping System Tests...")
    
    # Test the physics prototyping system
    physics_test_passed = await test_physics_prototyping()
    
    # Test integration with mad scientist system
    mad_scientist_test_passed = await test_mad_scientist_integration()
    
    print(f"\n\nTest Results:")
    print(f"Physics Prototyping: {'[PASS]' if physics_test_passed else '[FAIL]'}")
    print(f"Mad Scientist Integration: {'[PASS]' if mad_scientist_test_passed else '[FAIL]'}")
    
    if physics_test_passed and mad_scientist_test_passed:
        print("\n[SUCCESS] All tests passed! The physics prototyping system is working correctly.")
        return True
    else:
        print("\n[ERROR] Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)