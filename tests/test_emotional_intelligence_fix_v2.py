#!/usr/bin/env python3
"""
Test script to verify the improved fix for the JSON parsing error in emotional intelligence module.
"""

from modules.emotional_intellegence.emotional_intellegence import EmotionalIntelligence
from modules.emotional_intellegence.mood_processor import MoodProcessor
import sys
import os
import logging

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging to see the error messages
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_json_extraction():
    """Test the JSON extraction functionality with various response formats."""
    print("Testing JSON extraction functionality...")

    # Create an instance of EmotionalIntelligence
    ei = EmotionalIntelligence()
    mood_processor = MoodProcessor(ei)

    # Test cases with different response formats
    test_cases = [
        # Valid JSON response
        ('{"test": true}', {"test": True}),

        # JSON in markdown code blocks
        ('```json\n{"test": true}\n```', {"test": True}),
        ('```\n{"test": true}\n```', {"test": True}),

        # JSON with extra text
        ('Here is the JSON: {"test": true}', {"test": True}),

        # Empty response
        ('', {}),

        # Whitespace only
        ('   \n  \t  ', {}),

        # Invalid JSON
        ('This is not JSON', {}),

        # Complex JSON response like the one from the LLM
        ('```json\n{\n  "new_discovery": false,\n  "task_completed": false,\n  "error_occurred": false,\n  "repetition_detected": false,\n  "inactivity": false,\n  "milestone_achieved": false,\n  "external_feedback_positive": false,\n  "external_feedback_negative": false,\n  "resource_limitation": false,\n  "conflict_detected": false\n}\n```',
         {"new_discovery": False, "task_completed": False, "error_occurred": False, "repetition_detected": False, "inactivity": False, "milestone_achieved": False, "external_feedback_positive": False, "external_feedback_negative": False, "resource_limitation": False, "conflict_detected": False}),
    ]

    results = []
    for i, (response, expected) in enumerate(test_cases):
        try:
            result = mood_processor._extract_json_from_response(response)
            if result == expected:
                print(f"✓ Test case {i+1} passed")
                results.append(True)
            else:
                print(
                    f"✗ Test case {i+1} failed: expected {expected}, got {result}")
                results.append(False)
        except Exception as e:
            print(f"✗ Test case {i+1} failed with exception: {e}")
            results.append(False)

    return all(results)


def test_process_action_natural():
    """Test the process_action_natural method with a mock LLM response."""
    print("Testing process_action_natural method...")

    # Create an instance of EmotionalIntelligence
    ei = EmotionalIntelligence()

    try:
        # Test with a valid action output
        action_output = "This is a test action output"
        ei.process_action_natural(action_output)
        print("✓ process_action_natural handled correctly")
        return True
    except Exception as e:
        print(f"✗ process_action_natural failed with exception: {e}")
        return False


def test_process_action_result():
    """Test the process_action_result method."""
    print("Testing process_action_result method...")

    # Create an instance of EmotionalIntelligence
    ei = EmotionalIntelligence()

    try:
        # Test with a valid action result
        action_result = {
            "success": True,
            "completion": True
        }
        ei.process_action_result(action_result)
        print("✓ process_action_result handled correctly")
        return True
    except Exception as e:
        print(f"✗ process_action_result failed with exception: {e}")
        return False


def main():
    """Run all tests."""
    print("Running Emotional Intelligence Improved Fix Verification Tests")
    print("=" * 70)

    tests = [
        test_json_extraction,
        test_process_action_natural,
        test_process_action_result
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

    print("=" * 70)
    if all(results):
        print(
            "🎉 ALL TESTS PASSED! The improved JSON parsing error fix is working correctly.")
        return 0
    else:
        print("❌ SOME TESTS FAILED! The fix may need further improvements.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
