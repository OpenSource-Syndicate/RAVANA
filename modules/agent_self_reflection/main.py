import os
import sys
import json
from datetime import datetime, timezone
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../episodic_memory')))
from core.llm import call_llm, run_langchain_reflection
from reflection_prompts import REFLECTION_PROMPT
from reflection_db import save_reflection, load_reflections

def reflect_on_task(task_summary, outcome):
    """Generate a self-reflection using the LLM."""
    prompt = REFLECTION_PROMPT.format(task_summary=task_summary, outcome=outcome)
    reflection = call_llm(prompt)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_summary": task_summary,
        "outcome": outcome,
        "reflection": reflection
    }
    save_reflection(entry)
    return entry

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Self-Reflection & Self-Modification Module")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Reflection command
    reflect_parser = subparsers.add_parser('reflect', help='Generate a reflection')
    reflect_parser.add_argument('--task', type=str, required=True, help='Task summary')
    reflect_parser.add_argument('--outcome', type=str, required=True, help='Outcome description')
    reflect_parser.add_argument('--use-langchain', action='store_true', help='Use LangChain for Planning → Execution → Reflection')
    
    # Self-modification command
    modify_parser = subparsers.add_parser('modify', help='Run self-modification on reflection logs')
    
    args = parser.parse_args()
    
    if args.command == 'reflect':
        if args.use_langchain:
            entry = run_langchain_reflection(args.task, args.outcome)
        else:
            entry = reflect_on_task(args.task, args.outcome)
        print(json.dumps(entry, indent=2))
    elif args.command == 'modify':
        from self_modification import run_self_modification
        run_self_modification()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()