import sys
import os
import pprint

# Ensure module-5 is in sys.path for import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm import agi_experimentation_engine

# def test_agi_experimentation():
#     experiment_idea = "Test if sorting a list of numbers in Python using the built-in sorted() function works correctly."
#     result = agi_experimentation_engine(
#         experiment_idea,
#         llm_model=None,  # Use default LLM selection
#         use_chain_of_thought=True,
#         online_validation=False,  # Disable online validation for faster test
#         verbose=True
#     )
#     print("\n=== AGI Experimentation Engine Result: Python Sort Test ===")
#     pprint.pprint(result)

def test_physics_simulation():
    experiment_idea = "Simulate the motion of a simple pendulum using Python and plot its angle over time."
    result = agi_experimentation_engine(
        experiment_idea,
        llm_model=None,
        use_chain_of_thought=True,
        online_validation=False,
        verbose=True
    )
    print("\n=== AGI Experimentation Engine Result: Physics Simulation (Pendulum) ===")
    pprint.pprint(result)

def test_physical_experiment():
    experiment_idea = "Measure the acceleration due to gravity by dropping an object from a known height and timing its fall."
    result = agi_experimentation_engine(
        experiment_idea,
        llm_model=None,
        use_chain_of_thought=True,
        online_validation=True,  # Enable online validation for real-world feasibility
        verbose=True
    )
    print("\n=== AGI Experimentation Engine Result: Physical Experiment (Gravity) ===")
    pprint.pprint(result)

def test_complex_agi_experiment():
    experiment_idea = "Simulate an AI agent learning to balance a pole on a cart (cartpole problem) using reinforcement learning in Python."
    result = agi_experimentation_engine(
        experiment_idea,
        llm_model=None,
        use_chain_of_thought=True,
        online_validation=False,
        verbose=True
    )
    print("\n=== AGI Experimentation Engine Result: Complex AGI Experiment (Cartpole RL) ===")
    pprint.pprint(result)

if __name__ == "__main__":
    test_physics_simulation()
    test_physical_experiment()
    test_complex_agi_experiment() 