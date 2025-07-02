import json
import os
import requests
from google import genai
from openai import OpenAI
import logging
import random
from datetime import datetime, timezone
import subprocess
import re

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

def call_llm(prompt, preferred_provider=None, model=None, tools=None, tool_choice=None):
    """
    Try all providers in order, fallback to Gemini if all fail.
    Added support for tool calling with 'tools' and 'tool_choice' parameters.
    """
    # Special handling for tool calling
    if tools:
        # Try to use providers with tool calling support
        try:
            # Try OpenAI compatible APIs
            provider = 'zanity'  # Prefer provider with function calling support
            api_key, base_url = None, None
            for key, value in config.items():
                if isinstance(value, dict) and 'name' in value and value.get('name', '').lower() == provider:
                    api_key = value.get('api_key')
                    base_url = value.get('base_url')
                    break
            
            if api_key and base_url:
                client = OpenAI(api_key=api_key, base_url=base_url)
                model_name = model or "gpt-4o"
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    tools=tools,
                    tool_choice=tool_choice
                )
                
                # Parse tool calls
                if response.choices[0].message.tool_calls:
                    tool_calls = []
                    for tool_call in response.choices[0].message.tool_calls:
                        tool_calls.append({
                            "id": tool_call.id,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        })
                    return {
                        "content": response.choices[0].message.content,
                        "tool_calls": tool_calls
                    }
                else:
                    return response.choices[0].message.content
                    
        except Exception as e:
            print(f"Tool calling error: {str(e)}. Falling back to standard LLM call.")
    
    # Standard LLM call without tool calling
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

def extract_code_blocks(text):
    """Extract code blocks from text. Returns a list of (language, code) tuples."""
    code_blocks = re.findall(r'```(\w+)?\n([\s\S]*?)```', text)
    return [(lang.strip() if lang else '', code.strip()) for lang, code in code_blocks]

def execute_code_block(lang, code):
    """Execute a code block in the specified language and return output or error as a string (no prefix)."""
    import io
    import contextlib
    if lang.lower() == 'python':
        def try_exec(code_to_run):
            shared_dict = {}
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exec(code_to_run, shared_dict, shared_dict)
            output = stdout.getvalue().strip()
            if not output:
                last_value = None
                for k, v in shared_dict.items():
                    if not callable(v) and not hasattr(v, '__module__'):
                        last_value = v
                if last_value is not None:
                    output = str(last_value)
                else:
                    output = 'Python code executed successfully.'
            return output
        try:
            return try_exec(code)
        except NameError as e:
            # Try to auto-inject missing imports for common modules
            missing = []
            msg = str(e)
            for mod in ("subprocess", "os", "sys"):
                if f"name '{mod}' is not defined" in msg and f"import {mod}" not in code:
                    missing.append(f"import {mod}")
            if missing:
                code_with_imports = '\n'.join(missing) + '\n' + code
                try:
                    return try_exec(code_with_imports)
                except Exception as e2:
                    return f'[Python execution error after import fix: {e2}]'
            return f'[Python execution error: {e}]'
        except Exception as e:
            return f'[Python execution error: {e}]'
    elif lang.lower() in ('sh', 'bash', 'shell'):  # shell
        try:
            result = subprocess.run(code, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                output = result.stdout.strip() or 'Shell code executed successfully.'
                return output
            else:
                return f'[Shell execution error: {result.stderr.strip()}]'
        except Exception as e:
            return f'[Shell execution error: {e}]'
    else:
        return f'[Unsupported code language: {lang}]'

# Helper to select best model and provider from config

def get_best_model_and_provider(phase):
    """
    Select the best model and provider for a given phase (planning, execution, reflection).
    Returns (api_key, base_url, model_name)
    """
    # Model preferences by phase
    phase_prefs = {
        'planning': ["gpt-4o", "deepseek-v3-0324", "deepseek-r1", "llama-4-scout"],
        'execution': ["gpt-4o", "deepseek-v3-0324", "deepseek-r1", "llama-4-scout"],
        'reflection': ["gpt-4o", "claude-3.5-sonnet", "claude-3.7-sonnet", "deepseek-v3-0324", "deepseek-r1"]
    }
    prefs = phase_prefs.get(phase, [])
    # Providers to search
    provider_keys = ["zuki", "electronhub", "zanity", "a4f"]
    for provider_key in provider_keys:
        prov = config.get(provider_key)
        if not prov:
            continue
        models = prov.get("models", [])
        # models can be list of dicts or strings
        model_names = [m["name"] if isinstance(m, dict) else m for m in models]
        for pref in prefs:
            for m in model_names:
                if pref in m:
                    api_key = prov.get("api_key")
                    base_url = prov.get("base_url")
                    return api_key, base_url, m
    # fallback: first available model
    for provider_key in provider_keys:
        prov = config.get(provider_key)
        if not prov:
            continue
        models = prov.get("models", [])
        model_names = [m["name"] if isinstance(m, dict) else m for m in models]
        if model_names:
            api_key = prov.get("api_key")
            base_url = prov.get("base_url")
            return api_key, base_url, model_names[0]
    # fallback: None
    return None, None, None

def run_langchain_reflection(task_summary, outcome=None):
    """
    Chain Planning → Execution → Reflection using direct OpenAI-compatible API calls.
    If the plan contains Python or shell code, execute it and use the output as the outcome.
    Returns a dict with timestamp, task_summary, plan, outcome, and reflection.
    Uses best model for each phase from config.json.
    Enhanced prompt engineering for code execution.
    """
    # 1. Planning phase
    planning_api_key, planning_base_url, planning_model = get_best_model_and_provider('planning')
    planning_prompt = (
        "You are an expert Python developer and AI agent. "
        "Given the following task, create a step-by-step plan. "
        "If code is needed, include a single, complete, and self-contained code block that can be executed as-is. "
        "Do not reference undefined functions or variables. "
        "Always include all function and variable definitions needed for the code to run. "
        "If the task is a calculation, print the result.\n"
        f"Task: {task_summary}\nPlan:"
    )
    planning_client = OpenAI(api_key=planning_api_key, base_url=planning_base_url)
    plan_response = planning_client.chat.completions.create(
        model=planning_model,
        messages=[{"role": "user", "content": planning_prompt}]
    )
    plan = plan_response.choices[0].message.content

    # 2. Execution phase: look for code blocks and execute if found
    code_blocks = extract_code_blocks(plan)
    execution_results = []
    if code_blocks:
        for lang, code in code_blocks:
            # Special handling: If the task is shell-related and the code is Python that just runs a shell command, extract and run as shell
            is_shell_task = 'shell' in task_summary.lower() or 'list files' in task_summary.lower()
            if lang.lower() == 'python' and is_shell_task:
                import re
                # Try to extract the shell command from subprocess.run([...])
                match = re.search(r"subprocess\.run\(\[([^\]]+)\]", code)
                if match:
                    cmd_args = match.group(1)
                    # Convert to a list of strings
                    cmd_list = [s.strip().strip("'\"") for s in cmd_args.split(',')]
                    # On Windows, use 'dir', on Unix use 'ls'
                    import os
                    if os.name == 'nt':
                        if 'ls' in cmd_list:
                            cmd_list = ['cmd', '/c', 'dir']
                    else:
                        if 'dir' in cmd_list:
                            cmd_list = ['ls', '-la']
                    import subprocess
                    try:
                        result = subprocess.run(cmd_list, capture_output=True, text=True, timeout=30)
                        if result.returncode == 0:
                            output = result.stdout.strip() or 'Shell code executed successfully.'
                        else:
                            output = f'[Shell execution error: {result.stderr.strip()}]'
                    except Exception as e:
                        output = f'[Shell execution error: {e}]'
                    execution_results.append(f"[Shell code result]:\n{output}")
                    continue  # Skip normal Python execution for this block
            # Normal code execution
            result = execute_code_block(lang, code)
            if lang.lower() == 'python':
                execution_results.append(f"[Python code result]:\n{result}")
            elif lang.lower() in ('sh', 'bash', 'shell'):
                execution_results.append(f"[Shell code result]:\n{result}")
            else:
                execution_results.append(result)
        outcome = '\n'.join(execution_results)
        if len(code_blocks) == 1 and 'summary' in task_summary.lower() and '[Python code result]:' in outcome:
            outcome = outcome.replace('[Python code result]:\n', '', 1)
    else:
        # Fallback: use LLM to describe the outcome
        execution_api_key, execution_base_url, execution_model = get_best_model_and_provider('execution')
        execution_prompt = (
            "You are an AI agent executing the following plan. "
            "Describe the outcome in detail.\n"
            f"Plan: {plan}\nOutcome:"
        )
        execution_client = OpenAI(api_key=execution_api_key, base_url=execution_base_url)
        execution_response = execution_client.chat.completions.create(
            model=execution_model,
            messages=[{"role": "user", "content": execution_prompt}]
        )
        outcome = outcome or execution_response.choices[0].message.content

    # 3. Reflection phase
    reflection_api_key, reflection_base_url, reflection_model = get_best_model_and_provider('reflection')
    reflection_prompt = (
        "You are an AI agent journaling after a major task. "
        "Given the following summary and outcome, answer these questions in a structured way:\n"
        "1. What worked?\n2. What failed?\n3. What surprised you?\n4. What do you still need to learn?\n"
        f"Task Summary: {task_summary}\nOutcome: {outcome}\n"
        "Respond in a clear, numbered format."
    )
    reflection_client = OpenAI(api_key=reflection_api_key, base_url=reflection_base_url)
    reflection_response = reflection_client.chat.completions.create(
        model=reflection_model,
        messages=[{"role": "user", "content": reflection_prompt}]
    )
    reflection = reflection_response.choices[0].message.content

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_summary": task_summary,
        "plan": plan,
        "outcome": outcome,
        "reflection": reflection
    }
    from reflection_db import save_reflection
    save_reflection(entry)
    return entry

def is_lazy_llm_response(text):
    """
    Detects if the LLM response is lazy, generic, or incomplete.
    Returns True if the response is not actionable.
    """
    lazy_phrases = [
        "as an ai language model",
        "i'm unable to",
        "i cannot",
        "i apologize",
        "here is a function",
        "here's an example",
        "please see below",
        "unfortunately",
        "i do not have",
        "i don't have",
        "i am not able",
        "i am unable",
        "i suggest",
        "you can use",
        "to do this, you can",
        "this is a placeholder",
        "[insert",
        "[code block]",
        "[python code]",
        "[insert code here]",
        "[insert explanation here]",
        "[unsupported code language",
        "[python execution error",
        "[shell execution error",
        "[gemini",
        "[error",
        "[exception",
        "[output",
        "[result",
        "[python code result]:\n[python execution error",
    ]
    text_lower = text.strip().lower()
    if not text_lower or len(text_lower) < 10:
        return True
    for phrase in lazy_phrases:
        if phrase in text_lower:
            return True
    # If the response is just a code block marker or empty
    if text_lower in ("```", "```"):
        return True
    return False

def is_valid_code_patch(original_code, new_code):
    """
    Checks if the new_code is non-trivial and not just a copy of the original_code.
    Returns True if the patch is likely meaningful.
    """
    if not new_code or new_code.strip() == "":
        return False
    # If the new code is identical to the original, it's not a real patch
    if original_code.strip() == new_code.strip():
        return False
    # If the new code is just a comment or a single line, it's likely not useful
    lines = [l for l in new_code.strip().splitlines() if l.strip() and not l.strip().startswith("#")]
    if len(lines) < 2:
        return False
    return True

# Example usage:
if __name__ == "__main__":
    # Uncomment the line below to test all providers
    test_all_providers()