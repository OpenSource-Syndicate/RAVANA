import json
import os
import requests
from google import genai
from openai import OpenAI
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

# Load config
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Gemini fallback API key
GEMINI_API_KEY = "AIzaSyAWR9C57V2f2pXFwjtN9jkNYKA_ou5Hdo4"

def call_zuki(prompt, model=None):
    try:
        api_key = config['zuki']['api_key']
        base_url = config['zuki']['base_url']
        model = model or config['zuki']['models'][0]
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        r = requests.post(url, headers=headers, json=data, timeout=20)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return None

def call_electronhub(prompt, model=None):
    try:
        api_key = config['electronhub']['api_key']
        base_url = config['electronhub']['base_url']
        model = model or config['electronhub']['models'][0]
        url = f"{base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        r = requests.post(url, headers=headers, json=data, timeout=20)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return None

def call_zanity(prompt, model=None):
    try:
        api_key = config['zanity']['api_key']
        base_url = config['zanity']['base_url']
        model = model or config['zanity']['models'][0]
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        r = requests.post(url, headers=headers, json=data, timeout=20)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return None

def call_a4f(prompt):
    try:
        api_key = config['a4f']['api_key']
        base_url = config['a4f']['base_url']
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}]}
        r = requests.post(url, headers=headers, json=data, timeout=20)
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    except Exception as e:
        return None

def call_gemini(prompt):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=prompt
        )
        return response.text
    except Exception as e:
        return f"[Gemini fallback failed: {e}]"

def call_gemini_image_caption(image_path, prompt="Caption this image."):
    """Send an image and prompt to Gemini for captioning."""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        my_file = client.files.upload(file=image_path)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[my_file, prompt],
        )
        return response.text
    except Exception as e:
        return f"[Gemini image captioning failed: {e}]"

def call_gemini_audio_description(audio_path, prompt="Describe this audio clip"):
    """Send an audio file and prompt to Gemini for description."""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        my_file = client.files.upload(file=audio_path)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, my_file],
        )
        return response.text
    except Exception as e:
        return f"[Gemini audio description failed: {e}]"

def call_gemini_with_search(prompt):
    """Use Gemini with Google Search tool enabled."""
    try:
        from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
        client = genai.Client(api_key=GEMINI_API_KEY)
        model_id = "gemini-2.0-flash"
        google_search_tool = Tool(google_search=GoogleSearch())
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            )
        )
        # Return both the answer and grounding metadata if available
        answer = "\n".join([p.text for p in response.candidates[0].content.parts])
        grounding = getattr(response.candidates[0].grounding_metadata, 'search_entry_point', None)
        if grounding and hasattr(grounding, 'rendered_content'):
            return answer + "\n\n[Grounding Metadata:]\n" + grounding.rendered_content
        return answer
    except Exception as e:
        return f"[Gemini with search failed: {e}]"

def call_gemini_with_function_calling(prompt, function_declarations):
    """
    Call Gemini with function calling support. Returns a tuple (response_text, function_call_dict or None).
    function_declarations: list of function declaration dicts as per Gemini API.
    """
    try:
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        tools = types.Tool(function_declarations=function_declarations)
        config = types.GenerateContentConfig(tools=[tools])
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=config,
        )
        parts = response.candidates[0].content.parts
        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                function_call = part.function_call
                return None, {"name": function_call.name, "args": function_call.args}
        # No function call found
        return response.text, None
    except Exception as e:
        return f"[Gemini function calling failed: {e}]", None

def call_llm(prompt, preferred_provider=None, model=None):
    """
    Try all providers in order, fallback to Gemini if all fail.
    """
    providers = [
        (call_zuki, 'zuki'),
        (call_electronhub, 'electronhub'),
        (call_zanity, 'zanity'),
        (call_a4f, 'a4f'),
    ]
    if preferred_provider:
        providers = sorted(providers, key=lambda x: x[1] != preferred_provider)
    for func, name in providers:
        result = func(prompt, model) if name != 'a4f' else func(prompt)
        if result:
            return result
    # Fallback to Gemini
    return call_gemini(prompt)

def test_all_providers():
    """Test all LLM providers and Gemini fallbacks with a simple prompt."""
    prompt = "What is the capital of France?"
    print("Testing Zuki:")
    print(call_zuki(prompt))
    print("\nTesting ElectronHub:")
    print(call_electronhub(prompt))
    print("\nTesting Zanity:")
    print(call_zanity(prompt))
    print("\nTesting A4F:")
    print(call_a4f(prompt))
    print("\nTesting Gemini (text):")
    print(call_gemini(prompt))
    # Gemini advanced features (image/audio/search) require files or special prompts
    print("\nTesting Gemini with Google Search tool:")
    print(call_gemini_with_search("When is the next total solar eclipse in the United States?"))

PROVIDERS = [
    {
        "name": "a4f",
        "api_key": os.getenv("A4F_API_KEY", "ddc-a4f-7bbefd7518a74b36b1d32cb867b1931f"),
        "base_url": "https://api.a4f.co/v1",
        "models": ["provider-3/gemini-2.0-flash", "provider-2/llama-4-scout", "provider-3/llama-4-scout"] # Original models
    },
    {
        "name": "zukijourney",
        "api_key": os.getenv("ZUKIJOURNEY_API_KEY", "zu-ab9fba2aeef85c7ecb217b00ce7ca1fe"),
        "base_url": "https://api.zukijourney.com/v1",
        "models": ["gpt-4o:online", "gpt-4o", "deepseek-chat"]
    },
    {
        "name": "electronhub",
        "api_key": os.getenv("ELECTRONHUB_API_KEY", "ek-nzrvzzeQG0kmNZVhmkTWrKjgyIyUVY0mQpLwbectvfcPDssXiz"),
        "base_url": "https://api.electronhub.ai",
        "models": ["deepseek-v3-0324", "gpt-4o-2024-11-20"]
    },
    {
        "name": "zanity",
        "api_key": os.getenv("ZANITY_API_KEY", "vc-b1EbB_BekM2TCPol64yDe7FgmOM34d4q"),
        "base_url": "https://api.zanity.xyz/v1",
        "models": ["deepseek-v3-0324", "gpt-4o:free", "claude-3.5-sonnet:free", "qwen-max-0428"]
    }
]

def send_chat_message(message_content, preferred_model=None):
    """Sends a chat message using a random provider from the list."""
    provider = random.choice(PROVIDERS)
    client = OpenAI(
        api_key=provider["api_key"],
        base_url=provider["base_url"],
    )
    model_to_use = None
    if preferred_model and preferred_model in provider["models"]:
        model_to_use = preferred_model
    elif provider["models"]:
        model_to_use = provider["models"][0]
    if not model_to_use:
        logging.warning(f"No suitable model found for provider {provider['name']}. Skipping.")
        return None
    logging.info(f"Attempting to send message via {provider['name']} using model {model_to_use}")
    try:
        completion = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "user", "content": message_content}
            ]
        )
        response_content = completion.choices[0].message.content
        logging.info(f"Successfully received response from {provider['name']}.")
        return response_content
    except Exception as e:
        logging.error(f"Failed to send message via {provider['name']} using model {model_to_use}: {e}")
        if provider['name'] == 'zanity' and "404" in str(e).lower():
            logging.warning(f"Zanity API at {provider['base_url']} might be unavailable (404 error). Check URL.")
        return None


# Example usage:
if __name__ == "__main__":
    # Uncomment the line below to test all providers
    test_all_providers()