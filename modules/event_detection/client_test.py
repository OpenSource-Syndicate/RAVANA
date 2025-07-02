import requests
import json

# URL of the running FastAPI application
URL = "http://127.0.0.1:8001/process/"

# Sample data for testing
sample_data = {
    "texts": [
        "Massive solar flare expected to hit Earth tomorrow.",
        "New study shows coffee can improve memory.",
        "Scientists are amazed by the recent solar activity.",
        "The government announced new tax cuts for small businesses.",
        "Another report on solar flares causing potential power outages.",
        "This is some offensive content that should be filtered out.",
        "Local sports team wins the championship.",
        "Experts warn about the impact of the incoming solar storm.",
        "I really hate this, it's terrible.",
        "Researchers find a link between caffeine and alertness."
    ]
}

def run_test():
    """
    Sends a POST request to the /process/ endpoint and prints the response.
    """
    print("Sending request to:", URL)
    print("Payload:", json.dumps(sample_data, indent=2))
    
    try:
        response = requests.post(URL, json=sample_data)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        print("\n--- Test Successful ---")
        print("Status Code:", response.status_code)
        print("Response JSON:", json.dumps(response.json(), indent=2))
        
    except requests.exceptions.RequestException as e:
        print("\n--- Test Failed ---")
        print(f"An error occurred: {e}")

def test_empty_request():
    """
    Tests the endpoint with an empty list of texts.
    """
    print("\n--- Testing empty request ---")
    try:
        response = requests.post(URL, json={"texts": []})
        print("Status Code:", response.status_code)
        print("Response JSON:", json.dumps(response.json(), indent=2))
        if response.status_code == 400:
            print("Successfully caught empty request error.")
        else:
            print("Failed to catch empty request error.")
            
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Make sure the FastAPI server in main.py is running before executing this script.
    run_test()
    test_empty_request()