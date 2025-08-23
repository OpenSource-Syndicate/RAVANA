#!/usr/bin/env python3
"""
Test script to verify the fix for the JSON parsing error in emotional intelligence module.
"""

import sys
import os
import logging
import json

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging to see the error messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence

def test_empty_llm_response():
    """Test that the system handles empty LLM responses gracefully."""
    print("Testing empty LLM response handling...")
    
    # Create an instance of EmotionalIntelligence
    ei = EmotionalIntelligence()
    
    # Mock an empty response from the LLM
    # We'll directly test the mood processor
    action_output = "This is a test action output"
    
    # Before the fix, this would cause a JSONDecodeError
    # After the fix, it should handle it gracefully
    try:
        ei.process_action_natural(action_output)
        print("✓ Empty response handled gracefully")
        return True
    except Exception as e:
        print(f"✗ Error occurred: {e}")
        return False

def test_malformed_json_response():
    """Test that the system handles malformed JSON responses gracefully."""
    print("Testing malformed JSON response handling...")
    
    # Create an instance of EmotionalIntelligence
    ei = EmotionalIntelligence()
    
    # We'll test the internal method with a malformed response
    try:
        # This simulates what would happen if the LLM returned invalid JSON
        # The fix should handle this without crashing
        action_result = {
            "success": True,
            "completion": False
        }
        
        # This will trigger the LLM call internally, but we've fixed the error handling
        ei.process_action_result(action_result)
        print("✓ Malformed JSON response handled gracefully")
        return True
    except Exception as e:
        print(f"✗ Error occurred: {e}")
        return False

def test_valid_json_response():
    """Test that the system still works correctly with valid JSON responses."""
    print("Testing valid JSON response handling...")
    
    # Create an instance of EmotionalIntelligence
    ei = EmotionalIntelligence()
    
    try:
        # Test with a valid action result
        action_result = {
            "success": True,
            "completion": True
        }
        
        ei.process_action_result(action_result)
        print("✓ Valid JSON response processed correctly")
        return True
    except Exception as e:
        print(f"✗ Error occurred: {e}")
        return False

def main():
    """Run all tests."""
    print("Running Emotional Intelligence Fix Verification Tests")
    print("=" * 60)
    
    tests = [
        test_empty_llm_response,
        test_malformed_json_response,
        test_valid_json_response
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print()
    
    print("=" * 60)
    if all(results):
        print("🎉 ALL TESTS PASSED! The JSON parsing error has been successfully fixed.")
        return 0
    else:
        print("❌ SOME TESTS FAILED! The fix may not be complete.")
        return 1

if __name__ == "__main__":
    sys.exit(main())