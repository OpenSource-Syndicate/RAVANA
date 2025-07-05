import json
import os
import requests
from google import genai
from openai import OpenAI
import logging
import random
import tempfile
import traceback
import sys
import contextlib
import io
import re
import pkgutil
import subprocess
import importlib.util

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

def generate_hypothetical_scenarios(trends=None, interest_areas=None, gap_topics=None, model=None):
    """
    Generate creative hypothetical scenarios based on recent trends, interest areas, or detected gaps.
    Args:
        trends (list of str): Recent trending topics or keywords.
        interest_areas (list of str): Areas of interest to focus on.
        gap_topics (list of str): Topics not recently explored (optional).
        model (str): Optional model name to use for LLM.
    Returns:
        list of str: Generated hypothetical scenarios as prompts/questions.
    """
    prompt = "Generate 3 creative hypothetical scenarios or 'what if' questions based on the following context.\n"
    if trends:
        prompt += f"Recent trends: {', '.join(trends)}.\n"
    if interest_areas:
        prompt += f"Interest areas: {', '.join(interest_areas)}.\n"
    if gap_topics:
        prompt += f"Gaps: {', '.join(gap_topics)}.\n"
    prompt += "Be imaginative and relevant."
    response = call_llm(prompt, model=model)
    if response:
        # Split into list if possible
        scenarios = [line.strip('-* ') for line in response.split('\n') if line.strip()]
        return scenarios
    return []

def decision_maker_loop(situation, memory=None, mood=None, model=None, chain_of_thought=True, rag_context=None, actions=None):
    """
    The main decision-making loop for the AGI.

    Args:
        situation (dict): The current situation or prompt for the AGI.
        memory (list, optional): A list of relevant memories. Defaults to None.
        mood (dict, optional): The current mood of the AGI. Defaults to None.
        model (str, optional): The LLM model to use. Defaults to None.
        chain_of_thought (bool, optional): Whether to use chain of thought prompting. Defaults to True.
        rag_context (str, optional): Additional context from RAG. Defaults to None.
        actions (str, optional): A formatted string of available actions. Defaults to None.

    Returns:
        dict: A dictionary containing the decision, including the raw response from the LLM.
    """
    # Construct the prompt
    prompt = f"**Situation:**\n{situation['prompt']}\n\n"

    if memory:
        prompt += "**Relevant Memories:**\n"
        for mem in memory:
            prompt += f"- {mem['content']}\n"
        prompt += "\n"

    if mood:
        prompt += f"**Current Mood:**\n{json.dumps(mood, indent=2)}\n\n"
        
    if rag_context:
        prompt += f"**Additional Context:**\n{rag_context}\n\n"

    if actions:
        prompt += f"**Available Actions:**\n{actions}\n\n"

    prompt += """
**Your Task:**
Based on the situation, memories, and your current mood, decide on the best course of action.
1.  **Analyze the situation:** Briefly explain your understanding of the current state.
2.  **Formulate a plan:** Outline a step-by-step plan to address the situation.
3.  **Choose an action:** Select ONE action from the 'Available Actions' list and format your response as a JSON object within a ```json ... ``` block.

**JSON Output Format:**
{
  "action": "action_name",
  "params": {
    "param_name_1": "value_1",
    "param_name_2": "value_2"
  }
}
"""

    if chain_of_thought:
        # This is a simplified representation of CoT.
        # A more complex implementation might involve multiple LLM calls.
        prompt += "\n**Chain of Thought:**\n1. Analyze the situation.\n2. Formulate a plan.\n3. Choose an action and format the output.\n"

    # Call the LLM
    raw_response = call_llm(prompt, model=model)

    return {"raw_response": raw_response}

def agi_experimentation_engine(
    experiment_idea,
    llm_model=None,
    use_chain_of_thought=True,
    online_validation=True,
    sandbox_timeout=10,
    verbose=False
):
    """
    Unified AGI Experimentation Engine:
    1. Analyze/refine idea (LLM)
    2. Determine simulation type (Python, physics, physical, etc.)
    3. Generate Python code or simulation plan (LLM)
    4. Install required Python dependencies (if any)
    5. Execute code safely (sandboxed) if possible
    6. Gather results
    7. Cross-check real-world feasibility (web scraping + Gemini)
    8. Multi-layer reasoning (analysis, code, result, online, verdict)
    Returns: dict with all reasoning layers and final verdict
    """
    result = {
        'input_idea': experiment_idea,
        'refined_idea': None,
        'simulation_type': None,
        'generated_code': None,
        'dependency_installation': None,
        'execution_result': None,
        'execution_error': None,
        'result_interpretation': None,
        'online_validation': None,
        'final_verdict': None,
        'steps': []
    }

    def log_step(name, content):
        result['steps'].append({'step': name, 'content': content})
        if verbose:
            print(f"[{name}]\n{content}\n")

    # 1. Analyze and Refine Idea
    refine_prompt = f"""
    You are an advanced AGI research assistant. Given the following experiment idea, analyze it for clarity, feasibility, and suggest any refinements or clarifications needed. If the idea is about a physical or physics experiment, clarify what is to be measured, what equipment is needed, and whether it can be simulated in Python.\n\nExperiment Idea: {experiment_idea}\n\nRefined/clarified version (if needed):
    """
    refined_idea = call_llm(refine_prompt, model=llm_model)
    result['refined_idea'] = refined_idea
    log_step('refined_idea', refined_idea)

    # 2. Determine Simulation Type
    sim_type_prompt = f"""
    Given the following experiment idea, classify it as one of: 'python', 'physics_simulation', 'physical_experiment', or 'other'. If it can be simulated in Python, say 'python'. If it requires physics simulation, say 'physics_simulation'. If it requires real-world equipment, say 'physical_experiment'.\n\nIdea: {refined_idea}\n\nSimulation type:
    """
    simulation_type = call_llm(sim_type_prompt, model=llm_model)
    simulation_type = simulation_type.strip().split('\n')[0].lower()
    result['simulation_type'] = simulation_type
    log_step('simulation_type', simulation_type)

    # 3. Generate Python Code or Simulation Plan
    if simulation_type in ['python', 'physics_simulation']:
        code_prompt = f"""
        Given the following refined experiment idea, generate a single Python script that simulates or tests the idea locally. If it is a physics experiment, simulate it as best as possible in Python. If your code requires any external libraries, ensure you use only widely available packages (e.g., numpy, matplotlib, scipy) and import them at the top. Do not use obscure or unavailable packages.\n\nRefined Idea: {refined_idea}\n\nPython code (no explanation, just code):\n"""
        generated_code = call_llm(code_prompt, model=llm_model)
        # Strip markdown code block markers
        code_clean = re.sub(r"^```(?:python)?", "", generated_code.strip(), flags=re.MULTILINE)
        code_clean = re.sub(r"```$", "", code_clean, flags=re.MULTILINE)
        result['generated_code'] = code_clean
        log_step('generated_code', code_clean)
    else:
        # For physical experiments, generate a plan
        plan_prompt = f"""
        The following experiment idea requires real-world equipment. Generate a step-by-step plan for how a human could perform this experiment, including a list of required equipment.\n\nRefined Idea: {refined_idea}\n\nExperiment plan and equipment list:\n"""
        plan = call_llm(plan_prompt, model=llm_model)
        result['generated_code'] = plan
        log_step('experiment_plan', plan)

    # 4. Install required Python dependencies (if any)
    dependency_installation_log = []
    def install_missing_dependencies(code):
        # Robustly scan for import statements (import x, from x import y, from x.y import z)
        import_lines = re.findall(r'^\s*import ([a-zA-Z0-9_\.]+)', code, re.MULTILINE)
        from_imports = re.findall(r'^\s*from ([a-zA-Z0-9_\.]+) import', code, re.MULTILINE)
        modules = set(import_lines + from_imports)
        # Only use top-level package (e.g., 'matplotlib' from 'matplotlib.pyplot')
        top_level_modules = set([m.split('.')[0] for m in modules])
        # Exclude standard library modules
        stdlib_modules = set(sys.builtin_module_names)
        missing = []
        for mod in top_level_modules:
            if mod in stdlib_modules:
                continue
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        # Try to install missing packages (with retry and log pip output)
        for pkg in missing:
            for attempt in range(2):
                try:
                    pip_cmd = [sys.executable, '-m', 'pip', 'install', pkg]
                    proc = subprocess.run(pip_cmd, capture_output=True, text=True)
                    if proc.returncode == 0:
                        dependency_installation_log.append(f"Installed: {pkg}\n{proc.stdout}")
                        break
                    else:
                        dependency_installation_log.append(f"Attempt {attempt+1} failed to install {pkg}: {proc.stderr}")
                except Exception as e:
                    dependency_installation_log.append(f"Exception during install of {pkg}: {e}")
            else:
                dependency_installation_log.append(f"Failed to install {pkg} after 2 attempts. Please run: pip install {pkg} manually.")
        return dependency_installation_log

    if simulation_type in ['python', 'physics_simulation']:
        dependency_installation_log = install_missing_dependencies(result['generated_code'])
        result['dependency_installation'] = dependency_installation_log
        log_step('dependency_installation', dependency_installation_log)
    else:
        result['dependency_installation'] = None

    # 5. Execute Code Safely (Sandboxed) if possible
    def safe_execute_python(code, timeout=sandbox_timeout):
        """Executes code in a sandboxed environment and returns output/error."""
        import tempfile, sys, contextlib, io, os
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        output = io.StringIO()
        error = None
        try:
            with contextlib.redirect_stdout(output):
                with contextlib.redirect_stderr(output):
                    import subprocess
                    proc = subprocess.run(
                        [sys.executable, tmp_path],
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                    out = proc.stdout + proc.stderr
        except Exception as e:
            out = output.getvalue()
            error = f"Execution error: {e}\n{traceback.format_exc()}"
        finally:
            os.unlink(tmp_path)
        return out, error

    exec_out, exec_err = None, None
    if simulation_type in ['python', 'physics_simulation']:
        exec_out, exec_err = safe_execute_python(result['generated_code'])
        result['execution_result'] = exec_out
        result['execution_error'] = exec_err
        log_step('execution_result', exec_out or exec_err)
    else:
        result['execution_result'] = None
        result['execution_error'] = None

    # 6. Post-execution Result Interpretation
    interpret_prompt = f"""
    Here is the experiment idea, the generated code or plan, and the output/result.\n\nIdea: {refined_idea}\n\nCode or Plan:\n{result['generated_code']}\n\nOutput/Error:\n{exec_out or exec_err}\n\nInterpret the result. What does it mean? Any issues or insights?\n"""
    interpretation = call_llm(interpret_prompt, model=llm_model)
    result['result_interpretation'] = interpretation
    log_step('result_interpretation', interpretation)

    # 7. Online Validation (Web + Gemini)
    online_validation_result = None
    if online_validation:
        if simulation_type == 'physical_experiment':
            # Search for real-world equipment and feasibility
            web_prompt = f"""
            Given this physical experiment idea and plan, search for the required equipment and check if it is available for purchase or use. Also, check if the experiment is feasible in real life.\n\nIdea: {refined_idea}\nPlan: {result['generated_code']}\n\nCite sources if possible.\n"""
        else:
            # Try web search (Gemini with search)
            web_prompt = f"""
            Given this experiment idea and result, check if similar experiments have been done, and whether the result matches real-world knowledge.\n\nIdea: {refined_idea}\nResult: {exec_out or exec_err}\n\nCite sources if possible.\n"""
        try:
            online_validation_result = call_gemini_with_search(web_prompt)
        except Exception as e:
            online_validation_result = f"[Online validation failed: {e}]"
        result['online_validation'] = online_validation_result
        log_step('online_validation', online_validation_result)

    # 8. Final Verdict
    verdict_prompt = f"""
    Given all the above (idea, code/plan, result, online validation), provide a final verdict:\n- Success\n- Fail\n- Potential\n- Unknown\n\nJustify your verdict in 1-2 sentences.\n"""
    verdict = call_llm(verdict_prompt, model=llm_model)
    result['final_verdict'] = verdict
    log_step('final_verdict', verdict)

    return result

# Example usage:
if __name__ == "__main__":
    # Uncomment the line below to test all providers
    # test_all_providers()

    # Standalone test: test only package installation logic
    def test_package_installation():
        test_code = """
import numpy
import matplotlib.pyplot as plt
import requests
"""
        print("Testing package installation for test_code imports...")
        logs = []
        try:
            # Use the same install_missing_dependencies logic as in agi_experimentation_engine
            import re, sys, importlib.util, subprocess
            import_lines = re.findall(r'^\s*import ([a-zA-Z0-9_\.]+)', test_code, re.MULTILINE)
            from_imports = re.findall(r'^\s*from ([a-zA-Z0-9_\.]+) import', test_code, re.MULTILINE)
            modules = set(import_lines + from_imports)
            top_level_modules = set([m.split('.')[0] for m in modules])
            stdlib_modules = set(sys.builtin_module_names)
            missing = []
            for mod in top_level_modules:
                if mod in stdlib_modules:
                    continue
                if importlib.util.find_spec(mod) is None:
                    missing.append(mod)
            for pkg in missing:
                for attempt in range(2):
                    try:
                        pip_cmd = [sys.executable, '-m', 'pip', 'install', pkg]
                        proc = subprocess.run(pip_cmd, capture_output=True, text=True)
                        if proc.returncode == 0:
                            logs.append(f"Installed: {pkg}\n{proc.stdout}")
                            break
                        else:
                            logs.append(f"Attempt {attempt+1} failed to install {pkg}: {proc.stderr}")
                    except Exception as e:
                        logs.append(f"Exception during install of {pkg}: {e}")
                else:
                    logs.append(f"Failed to install {pkg} after 2 attempts. Please run: pip install {pkg} manually.")
        except Exception as e:
            logs.append(f"Exception in test_package_installation: {e}")
        print("\n".join(logs))

    test_package_installation()