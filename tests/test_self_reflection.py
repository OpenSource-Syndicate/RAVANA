import os
import json
from modules.agent_self_reflection.main import reflect_on_task
from modules.agent_self_reflection.reflection_db import load_reflections


def test_reflect_on_task():
    # Use the real LLM model (no patching)
    entry = reflect_on_task(
        "Test task for reflection sections", "Test outcome for reflection sections")
    print("Reflection Entry:", json.dumps(entry, indent=2))
    reflection = entry.get('reflection', '')
    # Check for all required sections
    assert 'reflection' in entry
    assert "1." in reflection and "2." in reflection and "3." in reflection and "4." in reflection, "Reflection missing required sections."
    print("Reflection output:\n", reflection)


def test_load_reflections():
    reflections = load_reflections()
    print("All Reflections:", json.dumps(reflections, indent=2))
    assert isinstance(reflections, list)


def test_reflection_sections():
    task = "Test planning and execution of a new feature."
    outcome = "Feature was implemented but had some bugs."
    entry = reflect_on_task(task, outcome)
    reflection = entry['reflection']
    assert "1." in reflection and "2." in reflection and "3." in reflection and "4." in reflection, "Reflection missing required sections."
    print("Reflection output:\n", reflection)


# def test_langchain_python_execution():
#     from main import run_langchain_reflection
#     task = "Calculate the factorial of 5 using Python."
#     entry = run_langchain_reflection(task)
#     print("LangChain Python Execution Entry:", json.dumps(entry, indent=2))
#     assert 'plan' in entry and 'outcome' in entry and 'reflection' in entry
#     assert 'Python code result' in entry['outcome']
#     assert '1.' in entry['reflection']


# def test_langchain_shell_execution():
#     from main import run_langchain_reflection
#     task = "List all files in the current directory using shell."
#     entry = run_langchain_reflection(task)
#     print("LangChain Shell Execution Entry:", json.dumps(entry, indent=2))
#     assert 'plan' in entry and 'outcome' in entry and 'reflection' in entry
#     assert 'shell code result' in entry['outcome'].lower(
#     ) or 'sh code result' in entry['outcome'].lower()
#     assert '1.' in entry['reflection']


# def test_langchain_no_code():
#     from main import run_langchain_reflection
#     task = "Write a short summary about the importance of self-reflection."
#     entry = run_langchain_reflection(task)
#     print("LangChain No Code Entry:", json.dumps(entry, indent=2))
#     assert 'plan' in entry and 'outcome' in entry and 'reflection' in entry
#     assert 'code result' not in entry['outcome'].lower()
#     assert '1.' in entry['reflection']


# def test_custom_langchain_reflection():
#     from main import run_langchain_reflection
#     task = "Develop a Python script to sort a list of numbers using bubble sort."
#     outcome = "The script was implemented and sorted the list correctly, but was slower than built-in sort."
#     entry = run_langchain_reflection(task, outcome)
#     print("Custom LangChain Reflection Entry:", json.dumps(entry, indent=2))
#     assert 'plan' in entry and 'outcome' in entry and 'reflection' in entry
#     assert '1.' in entry['reflection']


def test_self_modification_patch():
    import tempfile
    import shutil
    from modules.agent_self_reflection.self_modification import run_self_modification, log_audit, AUDIT_LOG
    # Setup: create a temp copy of the module
    temp_dir = tempfile.mkdtemp()
    try:
        for fname in os.listdir(os.path.dirname(__file__)):
            src = os.path.join(os.path.dirname(__file__), fname)
            dst = os.path.join(temp_dir, fname)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        # Simulate a reflection log with a known bug
        test_reflection = {
            "timestamp": "2025-07-01T00:00:00Z",
            "task_summary": "Test bug in dummy_function",
            "outcome": "Function failed on edge case.",
            "reflection": "1. What failed?\n- The function 'dummy_function' in 'llm.py' does not handle empty input correctly.\n2. ..."
        }
        # Write to temp reflections.json
        reflections_path = os.path.join(temp_dir, 'reflections.json')
        with open(reflections_path, 'w', encoding='utf-8') as f:
            json.dump([test_reflection], f)
        # Patch the loader to use temp file
        import sys
        sys.path.insert(0, temp_dir)
        import importlib
        reflection_db = importlib.import_module('modules.agent_self_reflection.reflection_db')
        reflection_db.REFLECTIONS_FILE = reflections_path
        # Run self-modification (should not patch real code)
        run_self_modification()
        # Check audit log
        audit_path = os.path.join(temp_dir, 'self_modification_audit.json')
        if os.path.exists(audit_path):
            with open(audit_path, 'r', encoding='utf-8') as f:
                audit = json.load(f)
            print("Audit log:", json.dumps(audit, indent=2))
        else:
            print("No audit log generated.")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # test_reflect_on_task()
    # test_load_reflections()
    # test_reflection_sections()
    # test_langchain_python_execution()
    # test_langchain_shell_execution()
    # test_langchain_no_code()
    # test_custom_langchain_reflection()
    test_self_modification_patch()
