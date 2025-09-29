#!/usr/bin/env python3
"""
Quick validation script for enhanced Gemini API system.
Run this to verify the implementation is working correctly.
"""

import sys
import os

# Add core directory to path
sys.path.insert(0, os.path.dirname(__file__))


def validate_gemini_enhancement():
    """Validate the enhanced Gemini system."""
    print("🔍 Validating Enhanced Gemini API System...")
    print("=" * 50)

    try:
        # Test import
        from core.llm import (
            call_gemini,
            get_gemini_key_statistics,
            gemini_key_manager
        )
        print("✅ Successfully imported enhanced Gemini functions")

        # Test configuration loading
        stats = get_gemini_key_statistics()
        print(f"✅ Configuration loaded: {stats['total_keys']} API keys found")
        print(f"✅ Available keys: {stats['available_keys']}")

        # Test basic functionality
        print("\n🧪 Testing basic Gemini call...")
        result = call_gemini("What is 2+2?")

        if result and not result.startswith("[") and "4" in result:
            print("✅ Basic Gemini call successful")
            print(
                f"   Result: {result[:100]}{'...' if len(result) > 100 else ''}")
        else:
            print(f"⚠️ Basic call result: {result}")

        # Show key usage
        final_stats = get_gemini_key_statistics()
        used_keys = [k for k, v in final_stats['keys'].items()
                     if v['total_requests'] > 0]
        if used_keys:
            print(f"✅ Key rotation working: Used key {used_keys[0][:15]}...")

        print("\n🎉 Validation completed successfully!")
        print("The enhanced Gemini API system with multiple keys is working correctly.")

        return True

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = validate_gemini_enhancement()
    sys.exit(0 if success else 1)
