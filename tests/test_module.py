import asyncio
import sys
import os
import pytest

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the real conversational AI modules using the correct paths
from modules.conversational_ai.main import ConversationalAI
from modules.conversational_ai.emotional_intelligence.conversational_ei import ConversationalEmotionalIntelligence


def test_response_generation():
    """Test that responses are contextually relevant and not generic spam."""
    ei = ConversationalEmotionalIntelligence()

    # Test with different emotional states
    emotional_context = {
        "dominant_mood": "Curious",
        "mood_vector": {"Curious": 0.8, "Engaged": 0.6},
        "recent_events": [],
        "detected_interests": ["technology"]
    }

    response = ei.generate_response(
        "What is artificial intelligence?", emotional_context)

    # Assert response is not generic spam
    assert "I understand" not in response, f"Response contains spam phrase: {response}"
    assert "How can I help you further" not in response, f"Response contains spam phrase: {response}"
    assert len(response) > 20, f"Response too short: {response}"


def test_interest_detection():
    """Test that interests are properly detected."""
    ei = ConversationalEmotionalIntelligence()

    interests = ei._detect_user_interests(
        "I love programming and machine learning")
    assert "technology" in interests, f"Technology interest not detected: {interests}"

    interests = ei._detect_user_interests(
        "Let's discuss philosophy and consciousness")
    assert "philosophy" in interests, f"Philosophy interest not detected: {interests}"


def test_fallback_responses():
    """Test that fallback responses are appropriate."""
    ei = ConversationalEmotionalIntelligence()

    # Test question fallback
    response = ei._generate_fallback_response(
        "What time is it?", {"dominant_mood": "Curious"})
    assert "question" in response.lower() or "tell me more" in response.lower(
    ), f"Inappropriate question fallback: {response}"

    # Test statement fallback
    response = ei._generate_fallback_response(
        "I like technology", {"dominant_mood": "Curious"})
    assert "fascinating" in response.lower() or "dive deeper" in response.lower(
    ), f"Inappropriate statement fallback: {response}"


@pytest.mark.asyncio
async def test_conversational_ai():
    """Test the conversational AI module."""
    # Initialize the conversational AI
    conversational_ai = ConversationalAI()

    # Test user profile management
    user_profile = conversational_ai.user_profile_manager.get_user_profile(
        "test_user_123", "discord")

    # Test emotional intelligence
    emotional_context = conversational_ai.emotional_intelligence.process_user_message(
        "Hello, how are you?",
        {"user_id": "test_user_123"}
    )

    # Test response generation
    response = conversational_ai.emotional_intelligence.generate_response(
        "Hello, how are you?",
        emotional_context
    )

    # Test memory interface
    context = conversational_ai.memory_interface.get_context(
        "test_user_123")