import requests
import json
import time
from typing import Optional, Dict, Any, List

BASE_URL = "http://localhost:8000"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

def make_request(method: str, endpoint: str, json_data: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Optional[Dict[str, Any]]:
    """Helper function to make HTTP requests with retries and better error handling"""
    url = f"{BASE_URL}/{endpoint}"
    headers = {"Content-Type": "application/json"}
    
    for attempt in range(MAX_RETRIES):
        try:
            if method.lower() == "get":
                response = requests.get(url, headers=headers, timeout=timeout)
            else:
                response = requests.post(url, headers=headers, json=json_data, timeout=timeout)
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Connection error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Failed to connect to server after {MAX_RETRIES} attempts: {e}")
                return None
        except requests.exceptions.Timeout as e:
            if attempt < MAX_RETRIES - 1:
                print(f"Request timeout (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"Request timed out after {MAX_RETRIES} attempts: {e}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    print(f"Response content: {e.response.json()}")
                except json.JSONDecodeError:
                    print(f"Response content: {e.response.text}")
            return None
    return None

def extract_memories(user_input: str, ai_output: str) -> Optional[Dict[str, Any]]:
    """Calls the /extract_memories/ endpoint with separate user and AI messages."""
    payload = {"user_input": user_input, "ai_output": ai_output}
    return make_request("post", "extract_memories/", json_data=payload)

def save_memories(memories_list: List[str], memory_type: str = 'long-term') -> Optional[Dict[str, Any]]:
    """Saves a list of memories to the server with a specified type."""
    data = {"memories": memories_list, "type": memory_type}
    return make_request("POST", "/save_memories/", json_data=data)

def get_relevant_memories(query_text: str, top_n: int = 5, similarity_threshold: float = 0.7) -> Optional[Dict[str, Any]]:
    """Calls the /get_relevant_memories/ endpoint."""
    payload = {
        "query_text": query_text,
        "top_n": top_n,
        "similarity_threshold": similarity_threshold
    }
    return make_request("post", "get_relevant_memories/", json_data=payload)

def health_check() -> Optional[Dict[str, Any]]:
    """Calls the /health endpoint."""
    return make_request("get", "health", timeout=10)

if __name__ == "__main__":
    print("--- MemoryDB Client Test Script ---")

    # 0. Health Check
    print("\n0. Checking API health...")
    health = health_check()
    if health and health.get("status") == "ok":
        print(f"API Health: {health['status']} - {health.get('message', '')}")
    else:
        print(f"Health check failed or API not healthy: {health}")
        print("Please ensure the FastAPI server (main.py) is running.")
        exit()

    # 1. Extract memories
    print("\n1. Extracting memories...")
    user1 = "I'm planning a vacation to Hawaii next month. And then Japan"
    ai1 = "That sounds wonderful!"
    extracted_data1 = extract_memories(user1, ai1)
    memories1 = []
    if extracted_data1 and 'memories' in extracted_data1:
        memories1 = extracted_data1['memories']
        print(f"Extracted from user: '{user1[:30]}...', ai: '{ai1[:30]}...': {memories1}")
    else:
        print(f"Failed to extract memories from user: '{user1[:30]}...', ai: '{ai1[:30]}...'. Response: {extracted_data1}")

    user2 = "My favorite hobby is hiking and I enjoy reading science fiction and i like to analyze every detail."
    ai2 = "Interesting!"
    extracted_data2 = extract_memories(user2, ai2)
    memories2 = []
    if extracted_data2 and 'memories' in extracted_data2:
        memories2 = extracted_data2['memories']
        print(f"Extracted from user: '{user2[:30]}...', ai: '{ai2[:30]}...': {memories2}")
    else:
        print(f"Failed to extract memories from user: '{user2[:30]}...', ai: '{ai2[:30]}...'. Response: {extracted_data2}")

    all_extracted_memories = memories1 + memories2

    # 2. Save memories
    if all_extracted_memories:
        print("\n2. Saving all extracted memories...")
        save_response = save_memories(all_extracted_memories)
        if save_response:
            print(f"Save response: {save_response}")
        else:
            print("Failed to save memories.")
    else:
        print("\n2. No memories extracted to save.")

    # 3. Get relevant memories
    print("\n3. Retrieving relevant memories...")
    queries = [
        {"text": "What are the vacation plans?", "top_n": 2, "threshold": 0.6},
        {"text": "What are the user's hobbies?", "top_n": 3, "threshold": 0.5},
        {"text": "Does the user like books?", "top_n": 1, "threshold": 0.5}
    ]

    for q_params in queries:
        query_text = q_params["text"]
        print(f"\nQuerying for: '{query_text}' (top_n={q_params['top_n']}, threshold={q_params['threshold']})")
        relevant_data = get_relevant_memories(query_text, top_n=q_params['top_n'], similarity_threshold=q_params['threshold'])
        if relevant_data and 'relevant_memories' in relevant_data:
            if relevant_data['relevant_memories']:
                for mem in relevant_data['relevant_memories']:
                    print(f"  - '{mem['text']}' (Similarity: {mem['similarity']:.3f})")
            else:
                print("  No relevant memories found above the threshold.")
        else:
            print(f"  Could not retrieve relevant memories. Response: {relevant_data}")

    # Example of a query that might not find highly similar results
    print("\nQuerying for something potentially unrelated: 'What's the weather like?'")
    irrelevant_query = "What's the weather like?"
    relevant_data_irrelevant = get_relevant_memories(irrelevant_query, top_n=2, similarity_threshold=0.1) # Low threshold
    if relevant_data_irrelevant and 'relevant_memories' in relevant_data_irrelevant:
        if relevant_data_irrelevant['relevant_memories']:
            for mem in relevant_data_irrelevant['relevant_memories']:
                print(f"  - '{mem['text']}' (Similarity: {mem['similarity']:.3f})")
        else:
            print("  No relevant memories found even with low threshold.")
    else:
        print(f"  Could not retrieve relevant memories. Response: {relevant_data_irrelevant}")


    print("\n--- End of MemoryDB Client Test Script ---")