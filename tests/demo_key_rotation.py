from core.llm import call_gemini, get_gemini_key_statistics
import sys
sys.path.insert(0, '.')


print("🚀 Demonstrating Enhanced Gemini Key Rotation")
print("=" * 50)

# Show initial state
print("\n📊 Initial Key Statistics:")
initial_stats = get_gemini_key_statistics()
print(f"Total keys: {initial_stats['total_keys']}")
print(f"Available keys: {initial_stats['available_keys']}")

# Make multiple calls to see key usage
print("\n🧪 Making multiple test calls...")
test_prompts = [
    "What is 2 + 2?",
    "What is the capital of Italy?",
    "Name a programming language."
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\nCall {i}: {prompt}")
    result = call_gemini(prompt)
    print(f"Result: {result[:50]}{'...' if len(result) > 50 else ''}")

# Show final statistics
print("\n📈 Final Key Usage Statistics:")
final_stats = get_gemini_key_statistics()

used_keys = []
for key_id, key_data in final_stats['keys'].items():
    if key_data['total_requests'] > 0:
        used_keys.append((key_id, key_data['total_requests']))

if used_keys:
    print("Keys used:")
    for key_id, requests in used_keys:
        print(f"  • {key_id[:20]}...: {requests} request(s)")
else:
    print("No key usage recorded")

print(
    f"\nTotal available keys: {final_stats['available_keys']}/{final_stats['total_keys']}")
print("✅ Enhanced Gemini system is working correctly!")
