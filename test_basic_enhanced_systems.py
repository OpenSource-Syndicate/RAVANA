"""
Test script for consciousness and emotional systems with error correction
"""
import asyncio
import os
import json
from datetime import datetime
from core.enhanced_consciousness_core import EnhancedConsciousnessCore


def create_dummy_config_files():
    """Create dummy config files for testing"""
    config_dir = "modules/emotional_intellegence/"
    os.makedirs(config_dir, exist_ok=True)
    
    # Create dummy config file
    config_path = os.path.join(config_dir, "config.json")
    if not os.path.exists(config_path):
        config_data = {
            "emotional_intelligence_config": {
                "triggers": {},
                "behavior_influences": {},
                "mood_dynamics": {
                    "momentum_factor": 0.7,
                    "damping_factor": 0.9
                }
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
    
    # Create dummy persona file
    persona_path = os.path.join(config_dir, "persona.json")
    if not os.path.exists(persona_path):
        persona_data = {
            "personas": {
                "Balanced": {
                    "mood_multipliers": {},
                    "adaptation_rate": 0.1
                }
            },
            "default_persona": "Balanced"
        }
        with open(persona_path, 'w') as f:
            json.dump(persona_data, f)


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
    if 'consciousness_metrics' in result:
        print(f"Consciousness metrics: {result['consciousness_metrics']}")
    
    return True


async def test_original_emotional_intelligence():
    """Test the original emotional intelligence system"""
    print("\nTesting Original Emotional Intelligence System...")
    
    # Create dummy config files first
    create_dummy_config_files()
    
    try:
        from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence
        
        # Initialize the emotional intelligence system
        ei = EmotionalIntelligence()
        
        # Test basic functionality
        print(f"Initial mood vector keys: {list(ei.mood_vector.keys())[:5]}...")  # Show first 5 keys
        
        # Test process_action_natural
        action_output = "The agent successfully completed the task and learned something new"
        ei.process_action_natural(action_output)
        
        print(f"Current mood after action: {ei.get_mood_vector()}")
        
        # Test influence behavior
        behavior_modifiers = ei.influence_behavior()
        print(f"Behavior modifiers: {list(behavior_modifiers.keys())}")
        
        print("Original EI test completed successfully")
        return True
    
    except Exception as e:
        print(f"Original EI test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_basic_tests():
    """Run basic tests for the enhanced systems"""
    print("Starting basic tests for consciousness and emotional systems...\n")
    
    tests = [
        ("Enhanced Consciousness", test_enhanced_consciousness),
        ("Original Emotional Intelligence", test_original_emotional_intelligence)
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
            import traceback
            traceback.print_exc()
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
        print("🎉 All tests PASSED!")
        return True
    else:
        print("❌ Some tests FAILED!")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_basic_tests())
    exit(0 if success else 1)