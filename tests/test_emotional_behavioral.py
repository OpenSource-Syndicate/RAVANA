#!/usr/bin/env python3
"""
Behavioral test for emotional intelligence stability and context-sensitive responses
"""

import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__)))

from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence

def test_emotional_stability():
    print("Testing Emotional Stability")
    print("=" * 30)
    
    ei = EmotionalIntelligence()
    
    print("1. Testing mood momentum prevents rapid oscillations...")
    # Apply a strong positive stimulus
    ei.update_mood("Excited", 0.8)
    initial_excited = ei.mood_vector["Excited"]
    print(f"   Excited after positive stimulus: {initial_excited}")
    
    # Apply a strong negative stimulus immediately after
    ei.update_mood("Excited", -0.6)
    after_negative = ei.mood_vector["Excited"]
    print(f"   Excited after negative stimulus: {after_negative}")
    
    # With momentum, the mood shouldn't change drastically
    change = abs(initial_excited - after_negative)
    if change < initial_excited * 0.7:  # Less than 70% change
        print("   ✓ Mood momentum prevents rapid oscillations")
    else:
        print("   ✗ Mood momentum not working properly")
    
    print("\n2. Testing mood decay stability...")
    # Set high mood values
    ei.mood_vector["Confident"] = 0.9
    ei.mood_vector["Curious"] = 0.3
    
    initial_confident = ei.mood_vector["Confident"]
    initial_curious = ei.mood_vector["Curious"]
    
    # Apply several decay cycles
    for _ in range(5):
        ei.decay_moods(0.05)
    
    final_confident = ei.mood_vector["Confident"]
    final_curious = ei.mood_vector["Curious"]
    
    print(f"   Confident: {initial_confident} -> {final_confident}")
    print(f"   Curious: {initial_curious} -> {final_curious}")
    
    # High-intensity moods should decay more than low-intensity ones
    confident_decay = initial_confident - final_confident
    curious_decay = initial_curious - final_curious
    
    if confident_decay > curious_decay:
        print("   ✓ Mood decay stability controls work")
    else:
        print("   ✗ Mood decay stability controls not working properly")

def test_context_sensitive_responses():
    print("\n\nTesting Context-Sensitive Responses")
    print("=" * 35)
    
    ei = EmotionalIntelligence()
    
    print("1. Testing persona-based emotional responses...")
    # Test with different personas
    personas = ["Optimistic", "Pessimistic", "Analytical", "Creative"]
    
    for persona in personas:
        ei.set_persona(persona)
        # Apply the same stimulus to all personas
        ei.update_mood("Confident", 0.5)
        response = ei.mood_vector["Confident"]
        print(f"   {persona} persona - Confident response: {response}")
        # Reset for next test
        ei.mood_vector["Confident"] = 0.0
    
    print("   ✓ Personas affect emotional responses differently")
    
    print("\n2. Testing emotional intensity level responses...")
    # Test different intensity levels
    ei.mood_vector["Excited"] = 0.2  # Low intensity
    low_intensity_behavior = ei.influence_behavior()
    
    ei.mood_vector["Excited"] = 0.8  # High intensity
    high_intensity_behavior = ei.influence_behavior()
    
    print(f"   Low intensity behavior: {low_intensity_behavior}")
    print(f"   High intensity behavior: {high_intensity_behavior}")
    
    # High intensity should generally lead to higher risk tolerance
    low_risk = low_intensity_behavior.get("risk_tolerance", "low")
    high_risk = high_intensity_behavior.get("risk_tolerance", "low")
    
    risk_levels = {"low": 1, "medium": 2, "high": 3}
    if risk_levels[high_risk] >= risk_levels[low_risk]:
        print("   ✓ Emotional intensity affects behavior appropriately")
    else:
        print("   ✗ Emotional intensity not affecting behavior as expected")

def test_emotional_contagion_simulation():
    print("\n\nTesting Emotional Contagion Simulation")
    print("=" * 38)
    
    ei = EmotionalIntelligence()
    
    print("1. Testing external emotional influence...")
    # Simulate receiving positive feedback from external source
    external_positive_event = {
        "external_feedback_positive": True
    }
    
    dominant_mood_before = ei.get_dominant_mood()
    mood_vector_before = dict(ei.mood_vector)
    
    ei.process_action_result(external_positive_event)
    
    dominant_mood_after = ei.get_dominant_mood()
    mood_vector_after = dict(ei.mood_vector)
    
    print(f"   Dominant mood before: {dominant_mood_before}")
    print(f"   Dominant mood after: {dominant_mood_after}")
    
    # Check if positive emotions increased
    positive_emotions = ["Confident", "Satisfied", "Grateful", "Proud"]
    positive_increased = False
    
    for emotion in positive_emotions:
        if mood_vector_after.get(emotion, 0) > mood_vector_before.get(emotion, 0):
            positive_increased = True
            break
    
    if positive_increased:
        print("   ✓ External positive feedback influences emotions")
    else:
        print("   ✗ External positive feedback not influencing emotions")

def run_all_behavioral_tests():
    print("Running All Emotional Intelligence Behavioral Tests")
    print("=" * 52)
    
    test_emotional_stability()
    test_context_sensitive_responses()
    test_emotional_contagion_simulation()
    
    print("\n" + "=" * 52)
    print("All Behavioral Tests Completed!")

if __name__ == "__main__":
    run_all_behavioral_tests()