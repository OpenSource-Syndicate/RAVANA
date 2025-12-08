"""
Test script to validate consciousness features in the RAVANA AGI system
"""
import asyncio
import logging
from datetime import datetime
from core.consciousness_core import ConsciousnessCore
from core.state import SharedState
from core.system import AGISystem
from database.engine import create_db_and_tables, get_engine


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_consciousness_core():
    """Test the core consciousness functionality"""
    logger.info("Testing consciousness core functionality...")
    
    # Create a mock AGI system for testing
    class MockAGISystem:
        def __init__(self):
            self.shared_state = SharedState({'Confident': 0.5, 'Curious': 0.3})
    
    mock_agi_system = MockAGISystem()
    
    # Initialize consciousness core
    consciousness_core = ConsciousnessCore(mock_agi_system)
    
    # Test 1: Process cognitive inputs
    test_inputs = {
        'memory_1': {'type': 'semantic', 'content': 'Knowledge about consciousness', 'priority': 0.8},
        'memory_2': {'type': 'episodic', 'content': 'Previous experience with attention', 'priority': 0.6},
        'goal_1': {'type': 'goal', 'content': 'Improve decision making', 'priority': 0.9},
        'emotion_1': {'type': 'emotion', 'content': 'Curious about new concepts', 'priority': 0.7}
    }
    
    logger.info("Testing cognitive input processing...")
    processed_inputs = await consciousness_core.process_cognitive_inputs(test_inputs)
    logger.info(f"Processed inputs: {list(processed_inputs.keys())}")
    
    # Test 2: Get consciousness state
    logger.info("Testing consciousness state retrieval...")
    consciousness_state = consciousness_core.get_consciousness_state()
    logger.info(f"Attention focus: {consciousness_state.attention_focus}")
    logger.info(f"Self model keys: {list(consciousness_state.self_model.keys())}")
    logger.info(f"Experience stream length: {len(consciousness_state.experience_stream)}")
    
    # Test 3: Introspection
    logger.info("Testing introspection...")
    introspection_result = await consciousness_core.introspect()
    logger.info(f"Introspection result keys: {list(introspection_result.keys())}")
    
    # Test 4: Experience reflection
    logger.info("Testing experience reflection...")
    sample_experience = {
        'timestamp': datetime.now(),
        'components': {'task': 'decision making', 'outcome': 'successful'},
        'coherence': 0.8,
        'context': {'mood': 'Confident', 'focus': 'decision'},
        'subjective_perspective': {'self_relevance': 0.9, 'valence': 0.7}
    }
    reflection_result = await consciousness_core.reflect_on_experience(sample_experience)
    logger.info(f"Reflection learning extracted: {reflection_result.get('learning_extracted', [])}")
    
    logger.info("Consciousness core functionality test completed.")
    return True


async def test_consciousness_integration():
    """Test consciousness integration with AGI system"""
    logger.info("Testing consciousness integration with AGI system...")
    
    # Create database engine
    engine = get_engine()
    
    try:
        # Initialize AGI system
        agi_system = AGISystem(engine)
        
        # Initialize components
        initialization_success = await agi_system.initialize_components()
        if not initialization_success:
            logger.error("AGI system initialization failed")
            return False
        
        logger.info("AGI system initialized successfully with consciousness features")
        
        # Test consciousness integration
        consciousness_state = await agi_system.consciousness_integration.get_consciousness_state()
        logger.info(f"Consciousness state retrieved: {consciousness_state}")
        
        # Test processing with consciousness
        test_inputs = {
            'test_memory': {'content': 'Test memory for consciousness processing', 'type': 'semantic'},
            'test_goal': {'content': 'Test goal for attention', 'type': 'goal'}
        }
        
        consciousness_output = await agi_system.consciousness_integration.process_with_consciousness(test_inputs)
        logger.info(f"Consciousness processing output: {list(consciousness_output.keys())}")
        
        # Test reflection with consciousness
        sample_experience = {
            'action_output': 'Test action completed',
            'situation_context': {'task': 'testing', 'complexity': 'low'},
            'mood_before': {'Curious': 0.5},
            'mood_after': {'Satisfied': 0.6},
            'timestamp': datetime.now()
        }
        
        reflection_result = await agi_system.consciousness_integration.reflect_on_experience(sample_experience)
        logger.info(f"Consciousness-aware reflection result: {reflection_result}")
        
        logger.info("Consciousness integration test completed successfully.")
        return True
        
    except Exception as e:
        logger.error(f"Error in consciousness integration test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if 'agi_system' in locals():
            try:
                await agi_system.stop("test_cleanup")
            except:
                pass


async def test_attention_mechanisms():
    """Test attention mechanisms specifically"""
    logger.info("Testing attention mechanisms...")
    
    # Create a mock AGI system for testing
    class MockAGISystem:
        def __init__(self):
            self.shared_state = SharedState({'Confident': 0.5, 'Curious': 0.3})
    
    mock_agi_system = MockAGISystem()
    
    # Initialize consciousness core
    consciousness_core = ConsciousnessCore(mock_agi_system)
    
    # Test attention controller directly
    attention_controller = consciousness_core.attention_system
    
    # Test inputs with different salience levels
    test_inputs = {
        'high_priority_task': {'type': 'task', 'urgency': 'high', 'importance': 'critical'},
        'medium_priority_info': {'type': 'information', 'urgency': 'medium', 'importance': 'moderate'},
        'low_priority_memory': {'type': 'memory', 'urgency': 'low', 'importance': 'trivia'},
        'self_related_goal': {'type': 'goal', 'urgency': 'high', 'importance': 'self-improvement', 'self_relevant': True}
    }
    
    # Calculate salience scores
    salience_scores = await attention_controller.calculate_salience_batch(test_inputs)
    logger.info(f"Salience scores: {salience_scores}")
    
    # Update attention focus
    new_focus = await attention_controller.update_attention_focus(salience_scores)
    logger.info(f"New attention focus: {new_focus}")
    
    # Check if self-related items got higher priority (if implementation considers self-relevance)
    if 'self_related_goal' in salience_scores:
        logger.info(f"Self-related goal salience: {salience_scores['self_related_goal']}")
    
    logger.info("Attention mechanisms test completed.")
    return True


async def test_global_workspace():
    """Test global workspace functionality"""
    logger.info("Testing global workspace...")
    
    # Create a mock AGI system for testing
    class MockAGISystem:
        def __init__(self):
            self.shared_state = SharedState({'Confident': 0.5, 'Curious': 0.3})
    
    mock_agi_system = MockAGISystem()
    
    # Initialize consciousness core
    consciousness_core = ConsciousnessCore(mock_agi_system)
    
    # Test global workspace directly
    global_workspace = consciousness_core.global_workspace
    
    # Register a test module
    global_workspace.register_module('test_module', 'read_write')
    
    # Broadcast some content
    content1_success = await global_workspace.broadcast_content(
        content_id='important_info_1',
        content={'data': 'Critical information', 'type': 'priority'},
        source_module='test_module',
        priority=0.9
    )
    
    content2_success = await global_workspace.broadcast_content(
        content_id='regular_info_1',
        content={'data': 'Regular information', 'type': 'normal'},
        source_module='test_module',
        priority=0.5
    )
    
    logger.info(f"Broadcast content 1 success: {content1_success}")
    logger.info(f"Broadcast content 2 success: {content2_success}")
    
    # Get prioritized content
    prioritized_content = global_workspace.get_prioritized_content('test_module', max_items=5)
    logger.info(f"Prioritized content: {prioritized_content}")
    
    # Check workspace contents
    workspace_state = global_workspace.workspace_contents
    logger.info(f"Workspace contents count: {len(workspace_state)}")
    
    logger.info("Global workspace test completed.")
    return True


async def run_all_tests():
    """Run all consciousness feature tests"""
    logger.info("Starting consciousness features validation tests...")
    
    tests = [
        ("Consciousness Core", test_consciousness_core),
        ("Attention Mechanisms", test_attention_mechanisms),
        ("Global Workspace", test_global_workspace),
        ("Consciousness Integration", test_consciousness_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        try:
            result = await test_func()
            results.append((test_name, result))
            logger.info(f"{test_name} test: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            logger.error(f"{test_name} test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("CONSCIOUSNESS FEATURES VALIDATION SUMMARY:")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All consciousness feature tests PASSED!")
        return True
    else:
        logger.info("❌ Some consciousness feature tests FAILED!")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)