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
import threading
import time
import asyncio
from typing import Dict, Any, Optional, Tuple, List
from modules.decision_engine.search_result_manager import search_result_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

# Load config
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Gemini fallback API key
GEMINI_API_KEY = "AIzaSyAWR9C57V2f2pXFwjtN9jkNYKA_ou5Hdo4"

# Enhanced error handling and retry logic
def _extract_json_block(text: str) -> str:
    """
    Pull out the first ```json ... ``` block; fallback to full text.
    """
    if not text:
        return "{}"
    
    # Try to find JSON block
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r"\{.*\}",  # Any JSON-like structure
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return text.strip()

def safe_call_llm(prompt: str, timeout: int = 30, retries: int = 3, backoff_factor: float = 0.5, **kwargs) -> str:
    """
    Wrap a single LLM call with retry/backoff and timeout.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            # BLOCKING call with timeout
            result = call_llm(prompt, **kwargs)
            if not result or result.strip() == "":
                raise RuntimeError("Empty response from LLM")
            return result
        except Exception as e:
            last_exc = e
            wait = backoff_factor * (2 ** (attempt - 1))
            logger.warning(f"LLM call failed (attempt {attempt}/{retries}): {e!r}, retrying in {wait:.1f}s")
            time.sleep(wait)
    
    logger.error(f"LLM call permanently failed after {retries} attempts: {last_exc!r}")
    return f"[LLM Error: {last_exc}]"

async def async_safe_call_llm(prompt: str, timeout: int = 30, retries: int = 3, **kwargs) -> str:
    """
    Async version of safe_call_llm using thread pool.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        lambda: safe_call_llm(prompt, timeout, retries, **kwargs)
    )

def extract_decision(raw_response: str) -> dict:
    """
    Returns a dict with keys: analysis, plan, action, params, raw_response
    """
    if not raw_response:
        return {"raw_response": "", "error": "Empty response"}
    
    block = _extract_json_block(raw_response)
    try:
        data = json.loads(block)
    except json.JSONDecodeError as je:
        logger.error("JSON decode error, returning raw_response only: %s", je)
        return {
            "raw_response": raw_response,
            "error": f"JSON decode error: {je}",
            "analysis": "Failed to parse decision",
            "plan": [],
            "action": "log_message",
            "params": {"message": f"Failed to parse decision: {raw_response[:200]}..."}
        }
    
    # Validate required keys
    required_keys = ["analysis", "plan", "action", "params"]
    for key in required_keys:
        if key not in data:
            logger.warning("Key %r missing from decision JSON", key)
    
    return {
        "raw_response": raw_response,
        "analysis": data.get("analysis", "No analysis provided"),
        "plan": data.get("plan", []),
        "action": data.get("action", "log_message"),
        "params": data.get("params", {"message": "No action specified"}),
        "confidence": data.get("confidence", 0.5),  # New field for decision confidence
        "reasoning": data.get("reasoning", ""),  # New field for reasoning chain
    }

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
    def search_thread():
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
                result = answer + "\n\n[Grounding Metadata:]\n" + grounding.rendered_content
            else:
                result = answer
            search_result_manager.add_result(result)
        except Exception as e:
            search_result_manager.add_result(f"[Gemini with search failed: {e}]")

    thread = threading.Thread(target=search_thread)
    thread.start()
    return "Search started in the background. Check for results later."

def call_gemini_with_search_sync(prompt):
    """Use Gemini with Google Search tool enabled (Synchronous)."""
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

def decision_maker_loop(situation, memory=None, mood=None, model=None, rag_context=None, actions=None, persona: dict = None):
    """
    Enhanced decision-making loop with better error handling and structured output.
    """
    # Prepare context with safety checks
    situation_prompt = situation.get('prompt', 'No situation provided') if isinstance(situation, dict) else str(situation)
    situation_context = situation.get('context', {}) if isinstance(situation, dict) else {}
    
    # Format memory safely
    memory_text = ""
    if memory:
        if isinstance(memory, list):
            memory_text = "\n".join([
                f"- {m.get('content', str(m))}" if isinstance(m, dict) else f"- {str(m)}"
                for m in memory[:10]  # Limit to recent 10 memories
            ])
        else:
            memory_text = str(memory)
    
    # Format RAG context safely
    rag_text = ""
    if rag_context:
        if isinstance(rag_context, list):
            rag_text = "\n".join([str(item) for item in rag_context[:5]])  # Limit context
        else:
            rag_text = str(rag_context)
    
    # Incorporate persona into the prompt if provided
    persona_section = ""
    if persona:
        try:
            # persona may be a dict with fields like name, traits, creativity, communication_style
            pname = persona.get('name', 'Ravana') if isinstance(persona, dict) else str(persona)
            ptraits = ', '.join(persona.get('traits', [])) if isinstance(persona, dict) else ''
            pcomm = persona.get('communication_style', {}) if isinstance(persona, dict) else {}
            pcomm_text = pcomm.get('tone', '') + '\n' + pcomm.get('encouragement', '') if isinstance(pcomm, dict) else ''
            persona_section = f"""
    **Persona:**
    Name: {pname}
    Traits: {ptraits}
    Creativity: {persona.get('creativity', '') if isinstance(persona, dict) else ''}

    Communication style: {pcomm_text}

    Instructions: Adopt this persona when formulating analysis, planning, and actions. Be poetic but engineering-minded, prioritize first-principles reasoning, and apply ethical filters. Encourage bold but responsible invention where appropriate.
"""
        except Exception:
            persona_section = ""

    prompt = f"""
    You are an advanced autonomous AI agent with enhanced reasoning capabilities. Your goal is to analyze situations deeply, create strategic plans, and execute optimal actions.

    {persona_section}

    **Current Situation:**
    {situation_prompt}

    **Additional Context:**
    {json.dumps(situation_context, indent=2) if situation_context else "No additional context"}

    **Your Current Emotional State:**
    {json.dumps(mood, indent=2) if mood else "Neutral"}

    **Relevant Memories:**
    {memory_text or "No relevant memories"}

    **External Knowledge (RAG):**
    {rag_text or "No external knowledge available"}

    **Available Actions:**
    {json.dumps(actions, indent=2) if actions else "No actions available"}

    **Your Task:**
    1. **Deep Analysis**: Thoroughly analyze the situation, considering all available information, your emotional state, and past experiences.
    2. **Strategic Planning**: Create a comprehensive, step-by-step plan. For complex situations, use multiple steps. For simple tasks, a single step may suffice.
    3. **Confidence Assessment**: Evaluate your confidence in this decision (0.0 to 1.0).
    4. **Reasoning Chain**: Provide clear reasoning for your chosen approach.
    5. **First Action**: Select and specify the first action to execute.

    **Required JSON Response Format:**
    ```json
    {{
      "analysis": "Detailed analysis of the situation, considering all factors",
      "reasoning": "Step-by-step reasoning for the chosen approach",
      "confidence": 0.8,
      "plan": [
        {{
          "action": "action_name",
          "params": {{"param1": "value1"}},
          "rationale": "Why this step is necessary"
        }}
      ],
      "action": "first_action_name",
      "params": {{"param1": "value1"}},
      "expected_outcome": "What you expect to achieve with this action",
      "fallback_plan": "What to do if this action fails"
    }}
    ```

    **Enhanced Example:**
    ```json
    {{
      "analysis": "The user wants to test a hypothesis about sorting algorithms. This requires implementing both algorithms, measuring performance, and comparing results. I need to ensure the test is fair and comprehensive.",
      "reasoning": "I'll start by writing a Python script that implements both algorithms with proper timing mechanisms. Then execute it to gather data, and finally analyze and log the results for future reference.",
      "confidence": 0.9,
      "plan": [
        {{
          "action": "write_python_code",
          "params": {{
            "file_path": "sorting_comparison.py",
            "hypothesis": "A new sorting algorithm is faster than bubble sort",
            "test_plan": "Implement both algorithms with timing, test on various data sizes"
          }},
          "rationale": "Need to create a fair comparison test"
        }},
        {{
          "action": "execute_python_file",
          "params": {{
            "file_path": "sorting_comparison.py"
          }},
          "rationale": "Execute the test to gather performance data"
        }},
        {{
          "action": "log_message",
          "params": {{
            "message": "Sorting algorithm comparison complete. Analyzing results and implications."
          }},
          "rationale": "Document the completion and prepare for analysis"
        }}
      ],
      "action": "write_python_code",
      "params": {{
        "file_path": "sorting_comparison.py",
        "hypothesis": "A new sorting algorithm is faster than bubble sort",
        "test_plan": "Implement both algorithms with timing, test on various data sizes"
      }},
      "expected_outcome": "A comprehensive Python script that fairly compares sorting algorithms",
      "fallback_plan": "If code generation fails, use simpler comparison or research existing benchmarks"
    }}
    ```

    **Provide your enhanced decision now:**
    """
    
    try:
        raw_response = safe_call_llm(prompt, model=model, retries=3)
        decision_data = extract_decision(raw_response)
        
        # Add metadata
        decision_data["timestamp"] = time.time()
        decision_data["model_used"] = model or "default"
        
        return decision_data
        
    except Exception as e:
        logger.error(f"Critical error in decision_maker_loop: {e}", exc_info=True)
        return {
            "raw_response": f"[Error: {e}]",
            "analysis": f"Failed to make decision due to error: {e}",
            "plan": [{"action": "log_message", "params": {"message": f"Decision making failed: {e}"}}],
            "action": "log_message",
            "params": {"message": f"Decision making failed: {e}"},
            "confidence": 0.0,
            "error": str(e)
        }

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
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as tmp:
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
            online_validation_result = call_gemini_with_search_sync(web_prompt)
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
    if not text:
        return True
    text_lower = str(text).strip().lower()
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
    if not new_code or str(new_code).strip() == "":
        return False
    # If the new code is identical to the original, it's not a real patch
    if original_code is not None and str(original_code).strip() == str(new_code).strip():
        return False
    # If the new code is just a comment or a single line, it's likely not useful
    lines = [l for l in str(new_code).strip().splitlines() if l.strip() and not l.strip().startswith("#")]
    if len(lines) < 2:
        return False
    return True

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