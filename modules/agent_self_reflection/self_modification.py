import os
import json
import tempfile
import shutil
import difflib
import re
import traceback
from datetime import datetime
from .reflection_db import load_reflections, save_reflection
from .llm import call_llm, is_lazy_llm_response, is_valid_code_patch
from ..decision_engine.planner import plan_from_context

SELF_PATH = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = SELF_PATH
AUDIT_LOG = os.path.join(MODULE_ROOT, 'self_modification_audit.json')
TOOL_REGISTRY = {}

# --- Tool Registration Decorator ---
def register_tool(name, description, parameters):
    def decorator(func):
        TOOL_REGISTRY[name] = {
            'function': func,
            'description': description,
            'parameters': parameters
        }
        return func
    return decorator

# --- Tool: Read File ---
@register_tool(
    name="read_file",
    description="Read the contents of a file",
    parameters={
        "filename": {
            "type": "string",
            "description": "Path to the file relative to module root"
        }
    }
)
def read_file(filename):
    path = os.path.join(MODULE_ROOT, filename)
    if not os.path.exists(path):
        return f"File not found: {filename}"
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

# --- Tool: Edit File ---
@register_tool(
    name="edit_file",
    description="Replace content in a file between specified lines",
    parameters={
        "filename": {
            "type": "string",
            "description": "Path to the file relative to module root"
        },
        "start": {
            "type": "integer",
            "description": "Start line number (1-indexed)"
        },
        "end": {
            "type": "integer",
            "description": "End line number (inclusive)"
        },
        "new_content": {
            "type": "string",
            "description": "New content to replace between start and end lines"
        }
    }
)
def edit_file(filename, start, end, new_content):
    path = os.path.join(MODULE_ROOT, filename)
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if start < 1 or end > len(lines):
        return f"Invalid line range: {start}-{end}. File has {len(lines)} lines."
    
    before = lines[:start-1]
    after = lines[end:]
    new_lines = new_content.splitlines(keepends=True)
    new_content = ''.join(before + new_lines + after)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    return f"Successfully modified {filename} lines {start}-{end}"

# --- Tool: Run Tests ---
@register_tool(
    name="run_tests",
    description="Execute the test suite in a sandbox environment",
    parameters={
        "test_command": {
            "type": "string",
            "description": "Command to run tests (default: 'python test_self_reflection.py')",
            "default": "python test_self_reflection.py"
        }
    }
)
def run_tests(test_command="python test_self_reflection.py"):
    temp_dir = tempfile.mkdtemp()
    try:
        # Copy entire module to temp directory
        shutil.copytree(MODULE_ROOT, temp_dir, dirs_exist_ok=True)
        
        # Execute tests
        import subprocess
        process = subprocess.run(
            test_command.split(),
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=120
        )
        return {
            "success": process.returncode == 0,
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr
        }
    finally:
        shutil.rmtree(temp_dir)

# --- Utility: Log audit trail ---
def log_audit(entry):
    try:
        if os.path.exists(AUDIT_LOG):
            with open(AUDIT_LOG, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
        data.append(entry)
        with open(AUDIT_LOG, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Failed to write audit log: {str(e)}")

# --- Find actionable failures ---
def find_actionable_reflections():
    reflections = load_reflections()
    actionable = []
    for entry in reflections:
        reflection = entry.get('reflection', '').lower()
        # More comprehensive failure indicators
        if any(term in reflection for term in 
              ['fail', 'error', 'bug', 'defect', 'issue', 'crash', 'break', 'problem']):
            actionable.append(entry)
    return actionable

# --- Enhanced bug extraction with tool calling ---
def extract_bug_info(reflection_entry):
    tools_spec = [{
        "name": "log_bug_report",
        "description": "Records a discovered bug in the system",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "File path relative to module root"
                },
                "function": {
                    "type": "string",
                    "description": "Function or class name where bug occurs"
                },
                "summary": {
                    "type": "string",
                    "description": "Concise description of the bug"
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                    "description": "Impact level of the bug"
                }
            },
            "required": ["filename", "function", "summary"]
        }
    }]
    
    prompt = f"""
Analyze the following reflection log and extract bug information. 
If no bug is found, respond with {{"status": "no_bug"}}.

Reflection:
{reflection_entry['reflection']}

Use the log_bug_report tool if a bug is identified.
"""
    response = call_llm(prompt, tools=tools_spec, tool_choice="auto")
    
    try:
        if response and 'tool_calls' in response:
            for call in response['tool_calls']:
                if call['function']['name'] == 'log_bug_report':
                    args = json.loads(call['function']['arguments'])
                    if all(k in args for k in ('filename', 'function', 'summary')):
                        args['severity'] = args.get('severity', 'medium')
                        return args
        return None
    except Exception:
        return None

# --- Extract code block with context ---
def extract_code_block(filename, symbol, context_lines=3):
    path = os.path.join(MODULE_ROOT, filename)
    if not os.path.exists(path):
        return None, None, None
    
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find function/class definition
    pattern = re.compile(rf'^\s*(def|class)\s+{symbol}\b')
    start_index = None
    for i, line in enumerate(lines):
        if pattern.match(line):
            start_index = i
            break
    
    if start_index is None:
        return None, None, None
    
    # Determine indentation level
    indent = len(lines[start_index]) - len(lines[start_index].lstrip())
    end_index = start_index
    
    # Find end of block
    for i in range(start_index + 1, len(lines)):
        if lines[i].strip() == '':
            continue
        current_indent = len(lines[i]) - len(lines[i].lstrip())
        if current_indent <= indent and not lines[i].lstrip().startswith('#'):
            break
        end_index = i
    
    # Add context
    start_index = max(0, start_index - context_lines)
    end_index = min(len(lines) - 1, end_index + context_lines)
    
    code_block = ''.join(lines[start_index:end_index+1])
    return code_block, start_index+1, end_index+1

# --- Generate patch with tool calling ---
def generate_patch(filename, symbol, code_block, start, end, bug_summary):
    tools_spec = [{
        "name": "propose_code_patch",
        "description": "Proposes a fix for identified bugs",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "File to modify"
                },
                "start_line": {
                    "type": "integer",
                    "description": "Start line of code to replace"
                },
                "end_line": {
                    "type": "integer",
                    "description": "End line of code to replace"
                },
                "new_code": {
                    "type": "string",
                    "description": "Fixed code content"
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief explanation of changes"
                }
            },
            "required": ["filename", "start_line", "end_line", "new_code"]
        }
    }]
    
    prompt = f"""
You are an expert Python developer. Fix the bug described below:

## Bug Summary
{bug_summary}

## Original Code
File: {filename}
Lines {start}-{end}:
```python
{code_block}
```

Provide a minimal fix that addresses ONLY the reported issue. Maintain existing:
- Coding style and conventions
- Comments and documentation
- Function signatures unless absolutely necessary

Output ONLY the tool call with the fixed code block.
"""
    response = call_llm(prompt, tools=tools_spec, tool_choice={"type": "function", "function": {"name": "propose_code_patch"}})
    
    try:
        if response and 'tool_calls' in response:
            for call in response['tool_calls']:
                if call['function']['name'] == 'propose_code_patch':
                    args = json.loads(call['function']['arguments'])
                    if all(k in args for k in ('filename', 'start_line', 'end_line', 'new_code')):
                        # Validate line numbers match request
                        if args['start_line'] != start or args['end_line'] != end:
                            print(f"Warning: Line mismatch in patch: requested {start}-{end}, got {args['start_line']}-{args['end_line']}")
                        
                        # Validate code safety
                        if is_lazy_llm_response(args['new_code']):
                            return None
                            
                        return {
                            'filename': args['filename'],
                            'start': args['start_line'],
                            'end': args['end_line'],
                            'new_code': args['new_code'],
                            'explanation': args.get('explanation', '')
                        }
        return None
    except Exception as e:
        print(f"Patch generation error: {str(e)}")
        return None

# --- Test patch with rollback capability ---
def test_patch(filename, start, end, new_code):
    # Create backup
    original_path = os.path.join(MODULE_ROOT, filename)
    backup_path = original_path + ".bak"
    shutil.copy2(original_path, backup_path)
    
    try:
        # Apply patch temporarily
        with open(original_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        new_lines = new_code.splitlines(keepends=True)
        patched_content = ''.join(lines[:start-1] + new_lines + lines[end:])
        
        with open(original_path, 'w', encoding='utf-8') as f:
            f.write(patched_content)
        
        # Run tests
        test_result = run_tests()
        return test_result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    finally:
        # Restore original file
        shutil.move(backup_path, original_path)

# --- Enhanced self-modification workflow ---
def run_self_modification():
    actionable = find_actionable_reflections()
    if not actionable:
        print("No actionable reflections found")
        return
    
    print(f"Found {len(actionable)} actionable reflections")
    
    for entry in actionable:
        print("\n" + "="*80)
        print(f"Processing reflection: {entry.get('id', 'unknown')}")
        
        bug_info = extract_bug_info(entry)
        if not bug_info:
            print("No bug information extracted")
            continue
        
        print(f"Bug found in {bug_info['filename']}:{bug_info['function']} - {bug_info['summary']}")
        
        code_block, start, end = extract_code_block(
            bug_info['filename'], 
            bug_info['function'],
            context_lines=5
        )
        
        if not code_block:
            print("Code block not found")
            continue
        
        patch = generate_patch(
            bug_info['filename'],
            bug_info['function'],
            code_block,
            start,
            end,
            bug_info['summary']
        )
        
        if not patch:
            print("Patch generation failed")
            continue
        
        print(f"Generated patch for lines {patch['start']}-{patch['end']}")
        if patch.get('explanation'):
            print(f"Fix explanation: {patch['explanation']}")
        
        # Validate patch safety
        if is_lazy_llm_response(patch['new_code']):
            print("Rejected lazy LLM response")
            continue
            
        # Test patch
        test_result = test_patch(
            patch['filename'],
            patch['start'],
            patch['end'],
            patch['new_code']
        )
        
        # Create audit entry
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'reflection_id': entry.get('id'),
            'bug_info': bug_info,
            'patch': patch,
            'test_result': test_result,
            'applied': False
        }
        
        # Apply patch if tests pass
        if test_result.get('success'):
            print("Tests passed - applying patch")
            edit_file(
                patch['filename'],
                patch['start'],
                patch['end'],
                patch['new_code']
            )
            audit_entry['applied'] = True
            # Record successful modification in reflection DB
            save_reflection({
                'timestamp': datetime.now().isoformat(),
                'task_summary': 'Self-modification',
                'outcome': f"Applied fix for {bug_info['summary']} in {bug_info['filename']}",
                'reflection': f"1. What worked?\n- Successfully identified and fixed a bug in {bug_info['function']} function.\n2. What failed?\n- N/A\n3. What surprised you?\n- The system was able to self-modify its own code safely.\n4. What do you still need to learn?\n- Continue improving bug detection and automated fixes."
            })
        else:
            print("Tests failed - patch rejected")
            print(f"Test output:\n{test_result.get('stdout', '')}\n{test_result.get('stderr', '')}")
        
        log_audit(audit_entry)

def generate_hypothesis(shared_state: dict) -> str | None:
    """
    Analyzes the agent's recent performance and formulates a testable hypothesis using an LLM.
    """
    # Extract relevant history from shared state
    recent_moods = shared_state.get('mood_history', [])
    recent_actions = shared_state.get('action_history', [])
    recent_reflections = load_reflections()[-5:] # Get last 5 reflections

    if len(recent_actions) < 5:
        print("Not enough action history to generate a hypothesis.")
        return None

    # Prepare context for the LLM
    context = f"""
    Here is a summary of my recent performance:
    - Recent Moods: {json.dumps(recent_moods)}
    - Recent Actions: {json.dumps([a.get('summary', a) for a in recent_actions])}
    - Recent Self-Reflections: {json.dumps(recent_reflections, indent=2)}

    Analyze this data and identify a potential area for improvement.
    Formulate a single, clear, and testable hypothesis about my behavior or performance.
    The hypothesis should propose a cause-and-effect relationship that can be tested through a specific action or scenario.

    Example Hypothesis: "I hypothesize that my planning accuracy decreases when I am in a 'sad' or 'anxious' mood, leading to less effective outcomes."
    Example Hypothesis: "I hypothesize that I am spending too much time on creative tasks and not enough on technical challenges, slowing my skill development."
    
    Return only the string containing the hypothesis.
    """
    
    print("Generating a new hypothesis from agent's recent history...")
    hypothesis = call_llm(context)

    if hypothesis and not is_lazy_llm_response(hypothesis):
        print(f"Generated Hypothesis: {hypothesis}")
        return hypothesis
    
    print("Failed to generate a valid hypothesis.")
    return None

def analyze_experiment_outcome(hypothesis: str, situation_prompt: str, outcome: str) -> dict:
    """
    Analyzes the result of an experiment designed to test a hypothesis.
    Generates a new reflection based on the findings and saves it.
    """
    print(f"Analyzing outcome for hypothesis: {hypothesis}")
    
    prompt = f"""
    You are a research analyst examining the results of an AI's self-experiment.

    ## Hypothesis
    "{hypothesis}"

    ## Experiment
    The AI was presented with the following situation to test the hypothesis:
    "{situation_prompt}"

    ## Outcome
    The result of the AI's action was:
    "{outcome}"

    ## Analysis Task
    1.  **Analyze the Outcome**: Did the outcome support, refute, or was it inconclusive for the hypothesis?
    2.  **Explain Your Reasoning**: Provide a brief, logical explanation for your conclusion.
    3.  **Formulate a New Principle**: Based on your analysis, formulate a new learned principle or an updated belief for the AI. This should be a concise takeaway.

    ## Response Format
    Return a JSON object with the following keys: "conclusion" ("supported", "refuted", "inconclusive"), "reasoning" (string), "new_principle" (string).
    """

    analysis_json = call_llm(prompt)
    
    try:
        analysis = json.loads(analysis_json)
        if all(k in analysis for k in ["conclusion", "reasoning", "new_principle"]):
            print(f"Experiment Conclusion: {analysis['conclusion']} - {analysis['new_principle']}")
            
            # Create and save a new reflection summarizing the experiment
            reflection_text = f"""
            **Experiment Log**
            - **Hypothesis**: {hypothesis}
            - **Conclusion**: The hypothesis was {analysis['conclusion']}.
            - **Reasoning**: {analysis['reasoning']}
            - **Learned Principle**: {analysis['new_principle']}
            """
            
            save_reflection({
                'timestamp': datetime.now().isoformat(),
                'task_summary': f"Conducted experiment to test hypothesis: '{hypothesis[:50]}...'",
                'outcome': f"Experiment was {analysis['conclusion']}. Learned: {analysis['new_principle']}",
                'reflection': reflection_text
            })

            # If a highly ambitious experiment failed, turn it into a long-term goal.
            if analysis.get('conclusion') in ['refuted', 'inconclusive']:
                ambition_prompt = f"""
                Is the following hypothesis highly ambitious, speculative, or related to fundamental concepts like consciousness, cosmology, or theoretical physics?
                Answer with only "yes" or "no".

                Hypothesis: "{hypothesis}"
                """
                ambition_response = call_llm(ambition_prompt).lower().strip()

                if "yes" in ambition_response:
                    print(f"Ambitious experiment failed. Creating a long-term goal.")
                    goal_context = f"Pursue the ambitious goal derived from the failed hypothesis: {hypothesis}. This will involve foundational research and developing novel approaches over a long period."
                    try:
                        new_goal_id = plan_from_context(goal_context, timeframe="lifelong")
                        print(f"Created new long-term goal with ID: {new_goal_id}")
                        
                        # Update the reflection to note the new goal
                        analysis['new_long_term_goal'] = f"Failure has led to a new long-term research goal (ID: {new_goal_id}) to pursue this ambitious concept."
                        
                        # Overwrite the previous reflection with this new context
                        reflection_text += f"\n- **Next Step**: This hypothesis was too ambitious for a direct test. It has been converted into a long-term research goal (ID: {new_goal_id})."
                        save_reflection({
                            'timestamp': datetime.now().isoformat(),
                            'task_summary': f"Failed to validate ambitious hypothesis: '{hypothesis[:50]}...'",
                            'outcome': f"Experiment was {analysis['conclusion']}. Converted to long-term goal.",
                            'reflection': reflection_text
                        })

                    except Exception as e:
                        print(f"Failed to create long-term goal: {e}")

            return analysis
        else:
            print("Analysis from LLM was missing required keys.")
            return {"error": "Invalid analysis format from LLM."}
    except (json.JSONDecodeError, TypeError):
        print(f"Failed to decode analysis from LLM: {analysis_json}")
        return {"error": "Failed to decode analysis from LLM."}

if __name__ == "__main__":
    run_self_modification()

    # Example of how the new functions would be used in the main AGI loop
    print("\n--- Testing Experimentation Loop ---")
    mock_shared_state = {
        "mood_history": ["happy", "neutral", "neutral", "sad", "sad", "sad"],
        "action_history": [
            {"summary": "Wrote a poem", "outcome": "positive feedback"},
            {"summary": "Debugged a file", "outcome": "tests failed"},
            {"summary": "Analyzed trends", "outcome": "found no new events"},
            {"summary": "Reflected on consciousness", "outcome": "no conclusion"},
            {"summary": "Planned a complex task", "outcome": "plan was inefficient"}
        ]
    }
    # 1. Generate a hypothesis
    hypo = generate_hypothesis(mock_shared_state)
    
    if hypo:
        # 2. In the AGI loop, this hypothesis would trigger a specific situation.
        #    For this test, we'll just define a mock situation and outcome.
        mock_situation = "A technical challenge to optimize a sorting algorithm under time pressure."
        mock_outcome = "The agent failed to optimize the algorithm, and its performance was worse than the baseline, confirming its plan was inefficient."
        
        # 3. Analyze the outcome
        analyze_experiment_outcome(hypo, mock_situation, mock_outcome) 