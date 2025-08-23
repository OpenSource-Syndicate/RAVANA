#!/usr/bin/env python3
"""
Comprehensive test to verify that the conversational AI spam issue is fully resolved.
This test specifically checks that the system no longer generates generic spam responses
like "I understand. How can I help you further with this?".
"""

import sys
import os
import traceback

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the emotional intelligence module directly for testing
try:
    from conversational_ai.emotional_intelligence.conversational_ei import ConversationalEmotionalIntelligence
except ImportError:
    from modules.conversational_ai.emotional_intelligence.conversational_ei import ConversationalEmotionalIntelligence


def test_spam_response_prevention():
    """Test that the system no longer generates generic spam responses."""
    print("Testing Spam Response Prevention")
    
    ei = ConversationalEmotionalIntelligence()
    
    # Test cases that previously would have triggered spam responses
    test_cases = [
        # Technical questions
        ("What is artificial intelligence?", {"dominant_mood": "Curious", "detected_interests": ["technology"]}),
        ("How does machine learning work?", {"dominant_mood": "Curious", "detected_interests": ["technology"]}),
        
        # Philosophical questions
        ("What is the meaning of consciousness?", {"dominant_mood": "Curious", "detected_interests": ["philosophy"]}),
        ("How do we define reality?", {"dominant_mood": "Curious", "detected_interests": ["philosophy"]}),
        
        # Casual conversation
        ("Hello, how are you?", {"dominant_mood": "Curious", "detected_interests": []}),
        ("I'm doing well today", {"dominant_mood": "Confident", "detected_interests": []}),
        
        # Edge cases that might trigger fallback responses
        ("", {"dominant_mood": "Curious", "detected_interests": []}),  # Empty message
        ("...", {"dominant_mood": "Bored", "detected_interests": []}),  # Minimal message
        ("?", {"dominant_mood": "Confused", "detected_interests": []}),  # Just a question mark
    ]
    
    # Spam phrases that should never appear in responses
    spam_phrases = [
        "I understand. How can I help you further with this?",
        "I understand",
        "How can I help you further",
        "Let me know if you need anything else",
        "Is there anything else I can assist you with?",
        "Feel free to ask me anything",
        "I'm here to help you with any questions",
    ]
    
    all_tests_passed = True
    
    for i, (message, emotional_context) in enumerate(test_cases, 1):
        try:
            response = ei.generate_response(message, emotional_context)
            
            # Check if response is empty or too short
            if not response or len(response.strip()) < 5:
                print(f"✗ Test {i} FAILED: Response is empty or too short: '{response}'")
                all_tests_passed = False
                continue
                
            # Check if response contains any spam phrases
            response_lower = response.lower()
            contains_spam = False
            found_spam_phrase = ""
            
            for spam_phrase in spam_phrases:
                if spam_phrase.lower() in response_lower:
                    contains_spam = True
                    found_spam_phrase = spam_phrase
                    break
            
            if contains_spam:
                print(f"✗ Test {i} FAILED: Response contains spam phrase '{found_spam_phrase}': {response}")
                all_tests_passed = False
            else:
                print(f"✓ Test {i} PASSED: {response[:100]}{'...' if len(response) > 100 else ''}")
                
        except Exception as e:
            print(f"✗ Test {i} FAILED with exception: {e}")
            traceback.print_exc()
            all_tests_passed = False
    
    return all_tests_passed


def test_fallback_quality():
    """Test that fallback responses are meaningful and not generic."""
    print("\nTesting Fallback Response Quality")
    
    ei = ConversationalEmotionalIntelligence()
    
    # Test fallback responses directly
    fallback_cases = [
        ("What time is it?", {"dominant_mood": "Curious"}),
        ("I like technology", {"dominant_mood": "Curious"}),
        ("This is frustrating", {"dominant_mood": "Frustrated"}),
        ("I'm bored", {"dominant_mood": "Bored"}),
    ]
    
    all_tests_passed = True
    
    for i, (message, emotional_context) in enumerate(fallback_cases, 1):
        try:
            response = ei._generate_fallback_response(message, emotional_context)
            
            # Check if response is empty or too short
            if not response or len(response.strip()) < 10:
                print(f"✗ Fallback Test {i} FAILED: Response is empty or too short: '{response}'")
                all_tests_passed = False
                continue
                
            # Check if response is meaningful (not just generic)
            generic_indicators = [
                "I understand", 
                "help you further", 
                "anything else", 
                "feel free",
                "here to help"
            ]
            
            is_generic = any(indicator.lower() in response.lower() for indicator in generic_indicators)
            
            if is_generic:
                print(f"✗ Fallback Test {i} FAILED: Response is too generic: {response}")
                all_tests_passed = False
            else:
                print(f"✓ Fallback Test {i} PASSED: {response}")
                
        except Exception as e:
            print(f"✗ Fallback Test {i} FAILED with exception: {e}")
            traceback.print_exc()
            all_tests_passed = False
    
    return all_tests_passed


def main():
    """Run all tests to verify the spam fix."""
    print("Running Comprehensive Spam Fix Verification Tests")
    print("=" * 60)
    
    # Run spam response prevention tests
    spam_test_passed = test_spam_response_prevention()
    
    # Run fallback quality tests
    fallback_test_passed = test_fallback_quality()
    
    print("\n" + "=" * 60)
    if spam_test_passed and fallback_test_passed:
        print("🎉 ALL TESTS PASSED! The conversational AI spam issue has been successfully resolved.")
        print("The system now generates meaningful, contextually relevant responses instead of generic spam.")
        return 0
    else:
        print("❌ SOME TESTS FAILED! The conversational AI spam issue may not be fully resolved.")
        return 1


if __name__ == "__main__":
    sys.exit(main())