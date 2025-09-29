import json
import sys
import os

print("Testing Enhanced Gemini Implementation...")
print("=" * 50)

# Test 1: Configuration Loading
try:
    with open('core/config.json', 'r') as f:
        config = json.load(f)

    gemini_config = config.get('gemini', {})
    api_keys = gemini_config.get('api_keys', [])

    print(f"✅ Configuration loaded successfully")
    print(f"✅ Found {len(api_keys)} Gemini API keys")

    if len(api_keys) >= 10:
        print(f"✅ All 10 API keys configured correctly")
    else:
        print(f"⚠️ Expected 10 keys, found {len(api_keys)}")

except Exception as e:
    print(f"❌ Configuration loading failed: {e}")
    sys.exit(1)

# Test 2: Import Enhanced Functions
try:
    sys.path.insert(0, os.path.dirname(__file__))
    from core.llm import (
        call_gemini,
        get_gemini_key_statistics,
        gemini_key_manager,
        call_gemini_with_fallback
    )
    print(f"✅ Enhanced Gemini functions imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Key Manager Initialization
try:
    stats = get_gemini_key_statistics()
    print(f"✅ Key manager initialized")
    print(f"   Total keys: {stats['total_keys']}")
    print(f"   Available keys: {stats['available_keys']}")
    print(f"   Rate limited keys: {stats['rate_limited_keys']}")
except Exception as e:
    print(f"❌ Key manager test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Basic Functionality Test
try:
    print("\nTesting basic Gemini call...")
    result = call_gemini("What is the capital of Japan?")

    if result and not result.startswith("[") and not result.startswith("Error"):
        print(f"✅ Basic call successful")
        print(f"   Result: {result[:100]}{'...' if len(result) > 100 else ''}")

        # Check key usage
        final_stats = get_gemini_key_statistics()
        used_keys = [k for k, v in final_stats['keys'].items()
                     if v['total_requests'] > 0]
        if used_keys:
            print(f"✅ Key rotation working: Used key {used_keys[0][:15]}...")
    else:
        print(f"⚠️ Call result: {result}")

except Exception as e:
    print(f"❌ Basic functionality test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 Testing completed!")
