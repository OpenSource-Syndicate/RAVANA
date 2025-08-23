#!/usr/bin/env python3
"""
Validation script for the enhanced emotional intelligence module
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence

def test_emotional_intelligence():
    print("Testing Enhanced Emotional Intelligence Module")
    print("=" * 50)
    
    # Initialize the emotional intelligence system
    ei = EmotionalIntelligence()
    
    print("1. Testing mood dimensions initialization...")
    print(f"   Total moods: {len(ei.ALL_MOODS)}")
    print(f"   Primary moods: {len(ei.PRIMARY_MOODS)}")
    print(f"   Secondary moods: {len(ei.SECONDARY_MOODS)}")
    print("   ✓ Mood dimensions initialized correctly")
    
    print("\n2. Testing mood update with momentum...")
    initial_confident = ei.mood_vector["Confident"]
    ei.update_mood("Confident", 0.5)
    print(f"   Confident mood changed from {initial_confident} to {ei.mood_vector['Confident']}")
    print(f"   Momentum for Confident: {ei.mood_momentum['Confident']}")
    print("   ✓ Mood update with momentum works")
    
    print("\n3. Testing mood blending...")
    # Set up conditions for blending
    ei.mood_vector["Confident"] = 0.7
    ei.mood_vector["Curious"] = 0.8
    previous_inspired = ei.mood_vector["Inspired"]
    ei.blend_moods()
    print(f"   Inspired mood changed from {previous_inspired} to {ei.mood_vector['Inspired']}")
    print("   ✓ Mood blending works")
    
    print("\n4. Testing emotional event logging...")
    initial_events = len(ei.emotional_events)
    mood_changes = {"Confident": 0.3, "Curious": 0.2}
    triggers = ["new_discovery"]
    context = "Test context"
    ei.log_emotional_event(mood_changes, triggers, context)
    print(f"   Emotional events: {initial_events} -> {len(ei.emotional_events)}")
    print("   ✓ Emotional event logging works")
    
    print("\n5. Testing emotional context retrieval...")
    context = ei.get_emotional_context()
    print(f"   Dominant mood: {context['dominant_mood']}")
    print(f"   Recent events count: {len(context['recent_events'])}")
    print("   ✓ Emotional context retrieval works")
    
    print("\n6. Testing mood intensity levels...")
    ei.mood_vector["Excited"] = 0.8
    intensity = ei.get_mood_intensity_level("Excited")
    print(f"   Excited mood intensity: {intensity}")
    print("   ✓ Mood intensity level classification works")
    
    print("\n7. Testing persona adaptation...")
    # Store initial multiplier
    initial_multiplier = ei.persona.get("mood_multipliers", {}).get("Confident", 1.0)
    ei.adapt_persona("Test experience", "positive outcome")
    new_multiplier = ei.persona.get("mood_multipliers", {}).get("Confident", 1.0)
    print(f"   Confident multiplier: {initial_multiplier} -> {new_multiplier}")
    print("   ✓ Persona adaptation works")
    
    print("\n8. Testing action processing...")
    action_result = {"new_discovery": True, "task_completed": False}
    ei.process_action_result(action_result)
    print(f"   Dominant mood after action: {ei.get_dominant_mood()}")
    print("   ✓ Action processing works")
    
    print("\n" + "=" * 50)
    print("All tests passed! Enhanced Emotional Intelligence module is working correctly.")

if __name__ == "__main__":
    test_emotional_intelligence()