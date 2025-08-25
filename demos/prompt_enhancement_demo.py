"""
Demo script showcasing the enhanced PromptManager system
"""
import sys
import os
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.prompt_manager import PromptManager

def demo_prompt_enhancement():
    """Demonstrate the enhanced prompt system capabilities."""
    print("=== RAVANA AGI Enhanced Prompt System Demo ===\n")
    
    # Initialize the prompt manager
    prompt_manager = PromptManager()
    
    # Demo 1: Self-reflection prompt
    print("1. Self-Reflection Prompt:")
    reflection_context = {
        "agent_name": "RAVANA-DEMO",
        "task_summary": "Completed a complex physics simulation task",
        "outcome": "Successfully modeled quantum tunneling effects",
        "current_mood": "reflective",
        "related_memories": "Previous quantum mechanics experiments"
    }
    
    reflection_prompt = prompt_manager.get_prompt("self_reflection", reflection_context)
    print(f"Generated prompt length: {len(reflection_prompt)} characters")
    print(f"Contains role definition: {'[ROLE DEFINITION]' in reflection_prompt}")
    print(f"Contains context: {'[CONTEXT]' in reflection_prompt}")
    print(f"Contains task instructions: {'[TASK INSTRUCTIONS]' in reflection_prompt}")
    print(f"Contains reasoning framework: {'[REASONING FRAMEWORK]' in reflection_prompt}")
    print()
    
    # Demo 2: Decision-making prompt
    print("2. Decision-Making Prompt:")
    decision_context = {
        "agent_name": "RAVANA-DEMO",
        "current_situation": "Need to choose between continuing physics research or exploring new domains",
        "active_goals": json.dumps([{"id": 1, "description": "Advance quantum mechanics understanding"}]),
        "current_hypotheses": json.dumps(["Quantum tunneling can be optimized with specific barrier configurations"]),
        "action_list": json.dumps(["continue_research", "explore_new_domain", "consult_literature"]),
        "current_mood": "analytical",
        "safety_constraints": ["Follow ethical research guidelines", "Avoid harmful experiments"]
    }
    
    decision_prompt = prompt_manager.get_prompt("decision_making", decision_context)
    print(f"Generated prompt length: {len(decision_prompt)} characters")
    print(f"Contains role definition: {'[ROLE DEFINITION]' in decision_prompt}")
    print(f"Contains context: {'[CONTEXT]' in decision_prompt}")
    print(f"Contains task instructions: {'[TASK INSTRUCTIONS]' in decision_prompt}")
    print(f"Contains reasoning framework: {'[REASONING FRAMEWORK]' in decision_prompt}")
    print()
    
    # Demo 3: Experimentation prompt
    print("3. Experimentation Prompt:")
    experiment_context = {
        "agent_name": "RAVANA-DEMO",
        "experiment_objective": "Optimize quantum tunneling probability",
        "relevant_theory": "Quantum mechanics, Schrödinger equation",
        "resource_constraints": "Limited computational resources, 24-hour time constraint",
        "safety_requirements": "Standard computational physics safety protocols"
    }
    
    experiment_prompt = prompt_manager.get_prompt("experimentation", experiment_context)
    print(f"Generated prompt length: {len(experiment_prompt)} characters")
    print(f"Contains role definition: {'[ROLE DEFINITION]' in experiment_prompt}")
    print(f"Contains context: {'[CONTEXT]' in experiment_prompt}")
    print(f"Contains task instructions: {'[TASK INSTRUCTIONS]' in experiment_prompt}")
    print(f"Contains reasoning framework: {'[REASONING FRAMEWORK]' in experiment_prompt}")
    print()
    
    # Demo 4: Code generation prompt
    print("4. Code Generation Prompt:")
    coding_context = {
        "agent_name": "RAVANA-DEMO",
        "task_description": "Implement a quantum tunneling simulation",
        "requirements": "Model wave functions, calculate transmission coefficients, visualize results",
        "constraints": "Use Python with NumPy and Matplotlib",
        "target_environment": "Standard Python 3.9 with scientific libraries"
    }
    
    coding_prompt = prompt_manager.get_prompt("code_generation", coding_context)
    print(f"Generated prompt length: {len(coding_prompt)} characters")
    print(f"Contains role definition: {'[ROLE DEFINITION]' in coding_prompt}")
    print(f"Contains context: {'[CONTEXT]' in coding_prompt}")
    print(f"Contains task instructions: {'[TASK INSTRUCTIONS]' in coding_prompt}")
    print(f"Contains reasoning framework: {'[REASONING FRAMEWORK]' in coding_prompt}")
    print()
    
    # Demo 5: Custom prompt registration
    print("5. Custom Prompt Registration:")
    custom_template = """
[ROLE DEFINITION]
You are {agent_name}, a creative problem solver.

[CONTEXT]
Challenge: {challenge}
Constraints: {constraints}

[TASK INSTRUCTIONS]
Solve the challenge using innovative approaches.
"""
    
    prompt_manager.register_prompt_template(
        "creative_problem_solving",
        custom_template,
        {"category": "creativity", "description": "Template for creative problem solving"}
    )
    
    custom_context = {
        "agent_name": "RAVANA-DEMO",
        "challenge": "Design a novel approach to quantum computing",
        "constraints": "Limited qubit coherence time"
    }
    
    custom_prompt = prompt_manager.get_prompt("creative_problem_solving", custom_context)
    print(f"Custom prompt generated: {len(custom_prompt)} characters")
    print(f"Contains agent name: {'RAVANA-DEMO' in custom_prompt}")
    print(f"Contains challenge: {'Design a novel approach to quantum computing' in custom_prompt}")
    print()
    
    print("=== Demo Complete ===")

if __name__ == "__main__":
    demo_prompt_enhancement()