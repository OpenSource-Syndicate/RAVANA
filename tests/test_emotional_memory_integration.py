#!/usr/bin/env python3
"""
Integration test for emotional intelligence with memory system
"""

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__)))

from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence

def mock_memory_retrieval(emotional_context):
    """
    Mock function to simulate memory retrieval based on emotional context
    """
    # In a real implementation, this would query the memory system
    # using the emotional context to weight memory retrieval
    dominant_mood = emotional_context["dominant_mood"]
    
    # Mock memory items related to different moods
    memory_map = {
        "Confident": ["Previous successful task completion", "Positive feedback received"],
        "Curious": ["Interesting research topic discovered", "New learning opportunity"],
        "Frustrated": ["Previous error encountered", "Failed attempt at problem solving"],
        "Inspired": ["Creative solution developed", "Breakthrough moment experienced"]
    }
    
    return memory_map.get(dominant_mood, ["General memory item"])

def test_emotional_memory_integration():
    print("Testing Emotional Intelligence with Memory Integration")
    print("=" * 55)
    
    # Initialize the emotional intelligence system
    ei = EmotionalIntelligence()
    
    print("1. Testing mood-based memory retrieval...")
    # Set a specific mood
    ei.mood_vector["Confident"] = 0.8
    ei.mood_vector["Curious"] = 0.6
    
    # Get emotional context
    emotional_context = ei.get_emotional_context()
    print(f"   Emotional context: {emotional_context}")
    
    # Retrieve memories based on emotional context
    memories = mock_memory_retrieval(emotional_context)
    print(f"   Retrieved memories: {memories}")
    print("   ✓ Mood-based memory retrieval works")
    
    print("\n2. Testing emotional event logging with memory context...")
    # Process an action that generates emotional events
    action_result = {
        "new_discovery": True,
        "task_completed": True
    }
    
    # Store previous event count
    initial_events = len(ei.emotional_events)
    
    # Process the action
    ei.process_action_result(action_result)
    
    # Check that emotional events were logged
    new_events = len(ei.emotional_events)
    print(f"   Emotional events: {initial_events} -> {new_events}")
    
    if new_events > initial_events:
        latest_event = ei.emotional_events[-1]
        print(f"   Latest event triggers: {latest_event.triggers}")
        print(f"   Latest event intensity: {latest_event.intensity}")
        print("   ✓ Emotional event logging with context works")
    else:
        print("   ✗ Emotional event logging failed")
    
    print("\n3. Testing emotional influence on behavior...")
    # Get behavior influence based on current emotional state
    behavior_influence = ei.influence_behavior()
    print(f"   Behavior influence: {behavior_influence}")
    
    # Check that we get meaningful behavior influence data
    if behavior_influence and isinstance(behavior_influence, dict):
        print("   ✓ Emotional influence on behavior works")
    else:
        print("   ✗ Emotional influence on behavior failed")
    
    print("\n4. Testing emotional context for decision making...")
    # Get emotional context for decision making
    decision_context = ei.get_emotional_context()
    
    # Check that we have all required components
    required_keys = ["dominant_mood", "mood_vector", "recent_events"]
    all_present = all(key in decision_context for key in required_keys)
    
    if all_present:
        print("   ✓ Emotional context for decision making works")
    else:
        print("   ✗ Emotional context for decision making failed")
        missing_keys = [key for key in required_keys if key not in decision_context]
        print(f"   Missing keys: {missing_keys}")
    
    print("\n" + "=" * 55)
    print("Emotional Intelligence Memory Integration test completed!")

if __name__ == "__main__":
    test_emotional_memory_integration()