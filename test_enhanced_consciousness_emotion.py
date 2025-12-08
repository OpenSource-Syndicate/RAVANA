"""
Simple test script to validate the enhanced consciousness and emotional systems
"""
import asyncio
from datetime import datetime
from core.enhanced_consciousness_core import EnhancedConsciousnessCore
from modules.emotional_intellegence.enhanced_emotional_intellegence import EnhancedEmotionalIntelligence
from core.consciousness_emotion_integrator import ConsciousnessEmotionIntegrator


async def test_enhanced_consciousness():
    """Test the enhanced consciousness system"""
    print("Testing Enhanced Consciousness System...")
    
    # Since we don't have the full AGI system, we'll create a mock system
    class MockAgiSystem:
        def __init__(self):
            pass

    mock_system = MockAgiSystem()
    
    # Initialize the enhanced consciousness core with mock system
    consciousness_core = EnhancedConsciousnessCore(mock_system)
    
    # Prepare test inputs
    test_inputs = {
        'information_1': {'type': 'fact', 'content': 'The sky is blue', 'priority': 0.6},
        'information_2': {'type': 'goal', 'content': 'Learn new concepts', 'priority': 0.8},
        'information_3': {'type': 'memory', 'content': 'Previous learning', 'priority': 0.4}
    }
    
    # Prepare test context
    test_context = {
        'current_goals': ['learning', 'exploration'],
        'cognitive_load': 0.4,
        'is_focusing': True,
        'emotional_state': {'valence': 0.2, 'arousal': 0.5},
        'stress_level': 0.2
    }
    
    # Process with dynamic consciousness
    result = consciousness_core.process_with_dynamic_consciousness(test_inputs, test_context)
    
    print(f"Consciousness processing result keys: {list(result.keys())}")
    print(f"Attended inputs: {list(result['attended_inputs'].keys())}")
    print(f"Consciousness metrics: {result['consciousness_metrics']}")
    
    return True


async def test_enhanced_emotional_intelligence():
    """Test the enhanced emotional intelligence system"""
    print("\nTesting Enhanced Emotional Intelligence System...")
    
    # Initialize the enhanced emotional intelligence system
    ei = EnhancedEmotionalIntelligence()
    
    # Create a test action result
    action_result = {
        'success': True,
        'outcome_quality': 0.8,
        'performance': 0.75,
        'importance': 0.9,
        'unexpected': False
    }
    
    # Create context
    context = {
        'cognitive_load': 0.3,
        'emotional_relevance': 0.7
    }
    
    # Process with enhanced emotional system
    result = ei.process_action_result_enhanced(action_result, context)
    
    print(f"Emotional processing result keys: {list(result.keys())}")
    print(f"Emotional impact: {result['emotional_impact']}")
    print(f"Regulated impact: {result['regulated_impact']}")
    print(f"Current mood profile: {result['mood_changes']}")
    
    # Test emotional regulation
    print("\nTesting emotional regulation...")
    regulation_result = ei.regulate_emotion("Neutral", 0.8)
    print(f"Regulation result: {regulation_result}")
    
    # Test emotional resilience metrics
    resilience_metrics = ei.get_emotional_resilience_metrics()
    print(f"Resilience metrics: {resilience_metrics}")
    
    return True


async def test_consciousness_emotion_integration():
    """Test the integration between consciousness and emotional systems"""
    print("\nTesting Consciousness-Emotion Integration...")
    
    # Create mock AGI system
    class MockAgiSystem:
        def __init__(self):
            pass

    mock_system = MockAgiSystem()
    
    # Initialize both systems
    consciousness_core = EnhancedConsciousnessCore(mock_system)
    emotional_intelligence = EnhancedEmotionalIntelligence()
    
    # Create integrator
    integrator = ConsciousnessEmotionIntegrator(consciousness_core, emotional_intelligence)
    
    # Prepare test inputs
    test_inputs = {
        'task_input': {'type': 'reasoning', 'content': 'Solve complex problem', 'priority': 0.9},
        'memory_input': {'type': 'information', 'content': 'Relevant knowledge', 'priority': 0.7}
    }
    
    # Prepare context
    context = {
        'current_goals': ['problem_solving', 'learning'],
        'cognitive_load': 0.6,
        'emotional_state': {'valence': 0.3, 'arousal': 0.6}
    }
    
    # Test integrated processing
    result = integrator.integrate_processing_with_emotions(test_inputs, context)
    
    print(f"Integration result keys: {list(result.keys())}")
    print(f"Integration metrics: {result['integration_metrics']}")
    print(f"Consciousness output keys: {list(result['consciousness_output'].keys())}")
    print(f"Emotional output keys: {list(result['emotional_output'].keys())}")
    
    # Test emotional awareness activation
    awareness_result = integrator.activate_emotional_awareness(depth=0.8)
    print(f"Emotional awareness result: {awareness_result}")
    
    # Test dynamic balance recommendations
    recommendations = integrator.get_dynamic_balance_recommendations()
    print(f"Balance recommendations: {recommendations}")
    
    return True


async def run_all_tests():
    """Run all tests"""
    print("Starting enhanced consciousness and emotion system tests...\n")
    
    tests = [
        ("Enhanced Consciousness", test_enhanced_consciousness),
        ("Enhanced Emotional Intelligence", test_enhanced_emotional_intelligence),
        ("Consciousness-Emotion Integration", test_consciousness_emotion_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name} test...")
        try:
            result = await test_func()
            results.append((test_name, result))
            print(f"{test_name} test: {'PASSED' if result else 'FAILED'}\n")
        except Exception as e:
            print(f"{test_name} test failed with error: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("="*60)
    print("TEST SUMMARY:")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All enhanced consciousness and emotion system tests PASSED!")
        return True
    else:
        print("❌ Some tests FAILED!")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)