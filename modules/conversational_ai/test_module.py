import asyncio
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try different import approaches
try:
    from conversational_ai.main import ConversationalAI
except ImportError:
    try:
        from modules.conversational_ai.main import ConversationalAI
    except ImportError:
        # Direct import
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from modules.conversational_ai.main import ConversationalAI

# Import the emotional intelligence module directly for testing
try:
    from conversational_ai.emotional_intelligence.conversational_ei import ConversationalEmotionalIntelligence
except ImportError:
    from modules.conversational_ai.emotional_intelligence.conversational_ei import ConversationalEmotionalIntelligence

def test_response_generation():
    """Test that responses are contextually relevant and not generic spam."""
    print("Testing Response Generation")
    
    ei = ConversationalEmotionalIntelligence()
    
    # Test with different emotional states
    emotional_context = {
        "dominant_mood": "Curious",
        "mood_vector": {"Curious": 0.8, "Engaged": 0.6},
        "recent_events": [],
        "detected_interests": ["technology"]
    }
    
    response = ei.generate_response("What is artificial intelligence?", emotional_context)
    
    # Assert response is not generic spam
    assert "I understand" not in response, f"Response contains spam phrase: {response}"
    assert "How can I help you further" not in response, f"Response contains spam phrase: {response}"
    assert len(response) > 20, f"Response too short: {response}"
    print(f"✓ Response generation working: {response[:100]}...")

def test_interest_detection():
    """Test that interests are properly detected."""
    print("Testing Interest Detection")
    
    ei = ConversationalEmotionalIntelligence()
    
    interests = ei._detect_user_interests("I love programming and machine learning")
    assert "technology" in interests, f"Technology interest not detected: {interests}"
    
    interests = ei._detect_user_interests("Let's discuss philosophy and consciousness")
    assert "philosophy" in interests, f"Philosophy interest not detected: {interests}"
    
    print("✓ Interest detection working")

def test_fallback_responses():
    """Test that fallback responses are appropriate."""
    print("Testing Fallback Responses")
    
    ei = ConversationalEmotionalIntelligence()
    
    # Test question fallback
    response = ei._generate_fallback_response("What time is it?", {"dominant_mood": "Curious"})
    assert "question" in response.lower() or "tell me more" in response.lower(), f"Inappropriate question fallback: {response}"
    
    # Test statement fallback
    response = ei._generate_fallback_response("I like technology", {"dominant_mood": "Curious"})
    assert "fascinating" in response.lower() or "dive deeper" in response.lower(), f"Inappropriate statement fallback: {response}"
    
    print("✓ Fallback responses working")

async def test_conversational_ai():
    """Test the conversational AI module."""
    print("Testing Conversational AI Module")
    
    # Initialize the conversational AI
    try:
        conversational_ai = ConversationalAI()
        print("✓ Conversational AI initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing Conversational AI: {e}")
        return
    
    # Test user profile management
    try:
        user_profile = conversational_ai.user_profile_manager.get_user_profile("test_user_123", "discord")
        print("✓ User profile management working")
    except Exception as e:
        print(f"✗ Error with user profile management: {e}")
    
    # Test emotional intelligence
    try:
        emotional_context = conversational_ai.emotional_intelligence.process_user_message(
            "Hello, how are you?", 
            {"user_id": "test_user_123"}
        )
        print("✓ Emotional intelligence processing working")
    except Exception as e:
        print(f"✗ Error with emotional intelligence processing: {e}")
    
    # Test response generation
    try:
        response = conversational_ai.emotional_intelligence.generate_response(
            "Hello, how are you?", 
            emotional_context
        )
        print(f"✓ Response generation working: {response}")
    except Exception as e:
        print(f"✗ Error with response generation: {e}")
    
    # Test memory interface
    try:
        context = conversational_ai.memory_interface.get_context("test_user_123")
        print("✓ Memory interface working")
    except Exception as e:
        print(f"✗ Error with memory interface: {e}")
    
    print("Test completed!")

if __name__ == "__main__":
    # Run unit tests first
    try:
        test_response_generation()
        test_interest_detection()
        test_fallback_responses()
        print("\nAll unit tests passed!\n")
    except Exception as e:
        print(f"Unit test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Run integration test
    asyncio.run(test_conversational_ai())