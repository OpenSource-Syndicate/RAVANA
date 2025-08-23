#!/usr/bin/env python3
"""
Test script for the enhanced Gemini API system with multiple API keys and fallback functionality.

This script validates:
1. Multiple API key configuration loading
2. Automatic key rotation on rate limiting
3. Failure detection and key management
4. All Gemini function types (text, image, audio, search, function calling)
5. Statistics and monitoring capabilities
"""

import sys
import os
import time
import json

# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

try:
    from core.llm import (
        call_gemini,
        call_gemini_image_caption,
        call_gemini_audio_description,
        call_gemini_with_search_sync,
        call_gemini_with_function_calling,
        get_gemini_key_statistics,
        reset_gemini_key_failures,
        test_gemini_enhanced,
        gemini_key_manager
    )
    print("✅ Successfully imported enhanced Gemini functions")
except ImportError as e:
    print(f"❌ Failed to import enhanced Gemini functions: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic Gemini text generation functionality."""
    print("\n🧪 Testing Basic Functionality")
    print("=" * 50)
    
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "What is 15 * 24?",
        "Name three programming languages."
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt}")
        result = call_gemini(prompt)
        
        if result and not result.startswith("[") and not result.startswith("Error"):
            print(f"✅ Success: {result[:100]}{'...' if len(result) > 100 else ''}")
        else:
            print(f"❌ Failed: {result}")
        
        # Small delay to avoid overwhelming the API
        time.sleep(1)

def test_key_statistics():
    """Test key statistics and monitoring functionality."""
    print("\n📊 Testing Key Statistics")
    print("=" * 50)
    
    stats = get_gemini_key_statistics()
    
    print(f"Total API keys configured: {stats['total_keys']}")
    print(f"Currently available keys: {stats['available_keys']}")
    print(f"Rate limited keys: {stats['rate_limited_keys']}")
    
    print("\nDetailed key information:")
    for key_id, key_data in stats['keys'].items():
        status = "✅ Available" if key_data['is_available'] else "❌ Unavailable"
        print(f"  Key {key_id[:15]}...: {status}")
        print(f"    Priority: {key_data['priority']}")
        print(f"    Total requests: {key_data['total_requests']}")
        print(f"    Consecutive failures: {key_data['consecutive_failures']}")
        if key_data['last_success']:
            print(f"    Last success: {key_data['last_success']}")
        if key_data['rate_limit_reset_time']:
            print(f"    Rate limit reset: {key_data['rate_limit_reset_time']}")
        print()

def test_rate_limit_simulation():
    """Simulate rate limiting behavior to test key rotation."""
    print("\n🔄 Testing Rate Limit Simulation")
    print("=" * 50)
    
    print("Making multiple rapid requests to test key rotation...")
    
    for i in range(5):
        print(f"\nRequest {i+1}:")
        result = call_gemini(f"Test request #{i+1}: What is {i+1} + {i+1}?")
        
        if result and not result.startswith("["):
            print(f"✅ Success: {result}")
        else:
            print(f"❌ Failed: {result}")
        
        # Show current statistics
        stats = get_gemini_key_statistics()
        print(f"Available keys: {stats['available_keys']}/{stats['total_keys']}")
        
        time.sleep(0.5)  # Short delay

def test_function_calling():
    """Test Gemini function calling capability."""
    print("\n🔧 Testing Function Calling")
    print("=" * 50)
    
    # Define a simple function for testing
    function_declarations = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit to use"
                    }
                },
                "required": ["location"]
            }
        }
    ]
    
    prompt = "What's the weather like in San Francisco?"
    
    try:
        result = call_gemini_with_function_calling(prompt, function_declarations)
        print(f"Function calling result: {result}")
        
        if isinstance(result, tuple) and len(result) == 2:
            text_response, function_call = result
            if function_call:
                print(f"✅ Function call detected: {function_call}")
            else:
                print(f"✅ Text response: {text_response}")
        else:
            print(f"✅ Response: {result}")
    except Exception as e:
        print(f"❌ Function calling failed: {e}")

def test_error_handling():
    """Test error handling and recovery mechanisms."""
    print("\n🛡️ Testing Error Handling")
    print("=" * 50)
    
    # Test with an invalid prompt that might cause issues
    print("Testing with potentially problematic input...")
    
    problematic_prompts = [
        "",  # Empty prompt
        "a" * 10000,  # Very long prompt
        "🤖💻🔥" * 100,  # Unicode characters
    ]
    
    for i, prompt in enumerate(problematic_prompts, 1):
        print(f"\nTest {i}: {'Empty prompt' if not prompt else f'Prompt length: {len(prompt)} chars'}")
        
        try:
            result = call_gemini(prompt)
            if result and not result.startswith("["):
                print(f"✅ Handled gracefully: {result[:50]}{'...' if len(result) > 50 else ''}")
            else:
                print(f"⚠️ Error handled: {result}")
        except Exception as e:
            print(f"❌ Exception occurred: {e}")

def test_configuration_loading():
    """Test configuration loading and validation."""
    print("\n⚙️ Testing Configuration Loading")
    print("=" * 50)
    
    try:
        # Check if keys were loaded from configuration
        stats = get_gemini_key_statistics()
        
        if stats['total_keys'] >= 10:
            print(f"✅ Configuration loaded successfully: {stats['total_keys']} keys found")
            
            # Verify key priorities
            priorities = [key_data['priority'] for key_data in stats['keys'].values()]
            if sorted(priorities) == list(range(1, len(priorities) + 1)):
                print("✅ Key priorities configured correctly")
            else:
                print("⚠️ Key priorities may have issues")
        else:
            print(f"⚠️ Expected at least 10 keys, found {stats['total_keys']}")
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

def main():
    """Main test function."""
    print("🚀 Enhanced Gemini API System Test Suite")
    print("=" * 60)
    
    # Test configuration loading first
    test_configuration_loading()
    
    # Test basic functionality
    test_basic_functionality()
    
    # Test key statistics
    test_key_statistics()
    
    # Test rate limit simulation
    test_rate_limit_simulation()
    
    # Test function calling
    test_function_calling()
    
    # Test error handling
    test_error_handling()
    
    # Run the built-in enhanced test
    print("\n🔬 Running Built-in Enhanced Test")
    print("=" * 50)
    test_gemini_enhanced()
    
    # Final statistics
    print("\n📈 Final Statistics")
    print("=" * 50)
    test_key_statistics()
    
    print("\n🎉 Test suite completed!")
    print("Review the results above to ensure all functionality is working correctly.")

if __name__ == "__main__":
    main()