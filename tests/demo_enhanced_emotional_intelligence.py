#!/usr/bin/env python3
"""
Demonstration of Enhanced Emotional Intelligence Features
"""

from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))


def demonstrate_enhanced_features():
    print("Enhanced Emotional Intelligence System Demonstration")
    print("=" * 52)

    # Initialize the enhanced emotional intelligence system
    ei = EmotionalIntelligence()

    print("1. Expanded Mood Dimensions")
    print("-" * 28)
    print(f"Total moods available: {len(ei.ALL_MOODS)}")
    print(f"Primary moods: {len(ei.PRIMARY_MOODS)}")
    print(f"Secondary moods: {len(ei.SECONDARY_MOODS)}")
    print(f"Sample moods: {ei.ALL_MOODS[:10]}...")
    print()

    print("2. Advanced Mood Dynamics")
    print("-" * 25)

    # Demonstrate momentum
    print("Momentum Effect:")
    ei.update_mood("Excited", 0.7)
    print(
        f"  Excited mood after positive stimulus: {ei.mood_vector['Excited']:.3f}")
    print(f"  Momentum for Excited: {ei.mood_momentum['Excited']:.3f}")

    # Demonstrate blending
    print("\nMood Blending:")
    ei.mood_vector["Confident"] = 0.8
    ei.mood_vector["Curious"] = 0.7
    previous_inspired = ei.mood_vector["Inspired"]
    ei.blend_moods()
    print(f"  Inspired mood before blending: {previous_inspired:.3f}")
    print(f"  Inspired mood after blending: {ei.mood_vector['Inspired']:.3f}")
    print()

    print("3. Emotional Memory Integration")
    print("-" * 31)

    # Log an emotional event
    mood_changes = {"Confident": 0.3, "Curious": 0.2}
    triggers = ["new_discovery", "task_completed"]
    context = "Successfully completed a complex learning task"
    ei.log_emotional_event(mood_changes, triggers, context)

    # Retrieve emotional context
    emotional_context = ei.get_emotional_context()
    print(f"Dominant mood: {emotional_context['dominant_mood']}")
    print(f"Recent events count: {len(emotional_context['recent_events'])}")
    if emotional_context['recent_events']:
        latest_event = emotional_context['recent_events'][-1]
        print(f"Latest event triggers: {latest_event['triggers']}")
    print()

    print("4. Enhanced Persona System")
    print("-" * 24)

    # Show different personas
    personas = ["Optimistic", "Pessimistic",
                "Analytical", "Creative", "Empathetic"]
    print("Persona-based responses to the same stimulus:")
    for persona in personas:
        ei.set_persona(persona)
        # Apply the same stimulus to all personas
        ei.mood_vector["Confident"] = 0.0  # Reset
        ei.update_mood("Confident", 0.5)
        response = ei.mood_vector["Confident"]
        print(f"  {persona:12}: {response:.3f}")
    print()

    print("5. Emotional Intensity Levels")
    print("-" * 28)

    emotions_to_test = ["Low Energy", "Curious", "Excited"]
    for emotion in emotions_to_test:
        # Test low intensity
        ei.mood_vector[emotion] = 0.2
        low_level = ei.get_mood_intensity_level(emotion)

        # Test high intensity
        ei.mood_vector[emotion] = 0.8
        high_level = ei.get_mood_intensity_level(emotion)

        print(f"{emotion:12} - Low: {low_level:6} | High: {high_level:6}")
    print()

    print("6. Context-Sensitive Emotional Responses")
    print("-" * 40)

    # Process different types of actions
    actions = [
        {"new_discovery": True},
        {"error_occurred": True},
        {"milestone_achieved": True},
        {"external_feedback_positive": True}
    ]

    ei.set_persona("Balanced")
    for i, action in enumerate(actions):
        print(f"Action {i+1}: {list(action.keys())[0]}")
        ei.process_action_result(action)
        dominant = ei.get_dominant_mood()
        intensity_level = ei.get_mood_intensity_level(dominant)
        behavior = ei.influence_behavior()
        print(f"  Dominant mood: {dominant} ({intensity_level} intensity)")
        print(f"  Behavior influence: risk_tolerance={behavior.get('risk_tolerance', 'N/A')}, "
              f"exploration_tendency={behavior.get('exploration_tendency', 'N/A')}")
        print()

    print("7. Emotional Adaptation")
    print("-" * 20)

    # Show persona adaptation
    print("Persona adaptation through experience:")
    initial_multiplier = ei.persona.get(
        "mood_multipliers", {}).get("Confident", 1.0)
    print(f"  Initial Confident multiplier: {initial_multiplier:.3f}")

    # Simulate positive experiences
    for _ in range(3):
        ei.adapt_persona("Successfully completed task", "positive outcome")

    new_multiplier = ei.persona.get(
        "mood_multipliers", {}).get("Confident", 1.0)
    print(
        f"  Confident multiplier after positive experiences: {new_multiplier:.3f}")
    print(f"  Change: {new_multiplier - initial_multiplier:+.3f}")
    print()

    print("=" * 52)
    print("Demonstration Complete!")
    print("The enhanced emotional intelligence system now supports:")
    print("• Expanded mood dimensions with 40+ emotions")
    print("• Advanced mood dynamics with momentum and blending")
    print("• Emotional memory integration for context-aware responses")
    print("• Enhanced persona system with adaptation capabilities")
    print("• Emotional intensity level classification")
    print("• Context-sensitive emotional responses")


if __name__ == "__main__":
    demonstrate_enhanced_features()
