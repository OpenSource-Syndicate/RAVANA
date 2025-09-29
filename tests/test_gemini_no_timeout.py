#!/usr/bin/env python3
"""
Test script to verify Gemini API calls work without timeouts.
"""

import sys
import os

# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))


def test_gemini_no_timeout():
    """Test that Gemini calls work without timeout configurations."""
    print("Testing Gemini API calls without timeout...")

    try:
        from core.llm import call_gemini
        print("✅ Successfully imported Gemini functions")

        # Test basic functionality
        result = call_gemini("What is 2+2?")

        if result and not result.startswith("[") and "4" in result:
            print("✅ Basic Gemini call successful")
            print(
                f"   Result: {result[:100]}{'...' if len(result) > 100 else ''}")
        else:
            print(f"⚠️ Basic call result: {result}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_gemini_no_timeout()
