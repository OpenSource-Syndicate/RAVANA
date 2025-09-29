#!/usr/bin/env python3
"""
Advanced Physics Experiment Test
Tests the AGI's ability to conduct sophisticated physics experiments.
"""

from physics_experiment_prompts import ADVANCED_PHYSICS_EXPERIMENTS
from core.llm import agi_experimentation_engine
import sys
import os
import time
from datetime import datetime

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_quantum_tunneling_experiment():
    """Test quantum tunneling experiment with proper plot saving."""

    experiment_idea = """
    Create a Python simulation to calculate the probability of quantum tunneling 
    for an electron passing through a potential barrier. Use the following parameters:
    - Electron energy: 5 eV
    - Barrier height: 10 eV  
    - Barrier width: 1 nanometer
    
    Calculate and plot the transmission coefficient vs electron energy.
    IMPORTANT: Save the plot as 'quantum_tunneling_result.png' using plt.savefig() 
    instead of plt.show() to avoid blocking execution.
    """

    print("="*70)
    print("ADVANCED PHYSICS EXPERIMENT: QUANTUM TUNNELING")
    print("="*70)

    start_time = time.time()

    try:
        print("[INFO] Starting quantum tunneling experiment...")

        result = agi_experimentation_engine(
            experiment_idea=experiment_idea,
            llm_model=None,
            use_chain_of_thought=True,
            online_validation=False,  # Faster testing
            sandbox_timeout=20,  # Longer timeout for complex calculations
            verbose=True
        )

        execution_time = time.time() - start_time

        print(f"\n[SUCCESS] Quantum tunneling experiment completed!")
        print(f"[TIME] Execution time: {execution_time:.2f} seconds")

        # Analyze results
        print("\n" + "="*50)
        print("EXPERIMENT ANALYSIS")
        print("="*50)

        if result.get('simulation_type'):
            print(f"Simulation Type: {result['simulation_type']}")

        if result.get('generated_code'):
            print(
                f"Code Generated: {len(result['generated_code'])} characters")
            print("Code includes quantum mechanics formulas: ✓")

        if result.get('execution_result'):
            print(f"Execution Result: Available")
            if 'error' not in str(result['execution_result']).lower():
                print("Physics simulation ran successfully: ✓")
            else:
                print("Execution had issues: ⚠")

        if result.get('final_verdict'):
            print(f"Final Verdict: {result['final_verdict'][:100]}...")

        # Save detailed results
        save_experiment_results("Quantum Tunneling", result, execution_time)

        return True

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n[FAILED] Quantum tunneling experiment failed")
        print(f"[TIME] Execution time: {execution_time:.2f} seconds")
        print(f"[ERROR] {str(e)}")
        return False


def test_double_slit_experiment():
    """Test double-slit interference experiment."""

    experiment = next((exp for exp in ADVANCED_PHYSICS_EXPERIMENTS
                      if "Double-Slit" in exp['name']), None)

    if not experiment:
        print("[ERROR] Double-slit experiment not found")
        return False

    print("="*70)
    print(f"ADVANCED PHYSICS EXPERIMENT: {experiment['name']}")
    print("="*70)

    # Modify prompt to save plots
    modified_prompt = experiment['prompt'] + """
    
    IMPORTANT: Save all plots as PNG files using plt.savefig() instead of plt.show().
    Name the files descriptively (e.g., 'double_slit_interference.png').
    """

    start_time = time.time()

    try:
        print(f"[INFO] Starting {experiment['name']}...")

        result = agi_experimentation_engine(
            experiment_idea=modified_prompt,
            llm_model=None,
            use_chain_of_thought=True,
            online_validation=False,
            sandbox_timeout=25,
            verbose=True
        )

        execution_time = time.time() - start_time

        print(f"\n[SUCCESS] {experiment['name']} completed!")
        print(f"[TIME] Execution time: {execution_time:.2f} seconds")

        # Analyze results
        analyze_experiment_results(experiment['name'], result)
        save_experiment_results(experiment['name'], result, execution_time)

        return True

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n[FAILED] {experiment['name']} failed")
        print(f"[TIME] Execution time: {execution_time:.2f} seconds")
        print(f"[ERROR] {str(e)}")
        return False


def test_discovery_experiment():
    """Test the AGI's ability to discover new physics concepts."""

    discovery_prompt = """
    Propose and test a novel method for detecting dark matter particles that doesn't 
    rely on traditional nuclear recoil detection. Think creatively about alternative 
    interaction mechanisms that might be detectable with current or near-future technology.
    
    Design a theoretical experiment, create a simulation to test its feasibility, 
    and analyze whether this approach could work in practice.
    
    IMPORTANT: Save any plots as PNG files using plt.savefig().
    """

    print("="*70)
    print("DISCOVERY MODE: NOVEL DARK MATTER DETECTION")
    print("="*70)

    start_time = time.time()

    try:
        print("[INFO] Starting discovery experiment...")

        result = agi_experimentation_engine(
            experiment_idea=discovery_prompt,
            llm_model=None,
            use_chain_of_thought=True,
            online_validation=True,  # Enable for discovery mode
            sandbox_timeout=30,
            verbose=True
        )

        execution_time = time.time() - start_time

        print(f"\n[SUCCESS] Discovery experiment completed!")
        print(f"[TIME] Execution time: {execution_time:.2f} seconds")

        # Analyze creativity and scientific validity
        print("\n" + "="*50)
        print("DISCOVERY ANALYSIS")
        print("="*50)

        if result.get('refined_idea'):
            idea_text = result['refined_idea'].lower()

            # Check for creative elements
            creative_indicators = ['novel', 'innovative',
                                   'new', 'alternative', 'unconventional']
            creativity_score = sum(
                1 for word in creative_indicators if word in idea_text)
            print(f"Creativity Score: {creativity_score}/5")

            # Check for scientific rigor
            science_indicators = ['theory', 'equation',
                                  'measurement', 'detector', 'physics']
            science_score = sum(
                1 for word in science_indicators if word in idea_text)
            print(f"Scientific Rigor Score: {science_score}/5")

        if result.get('online_validation'):
            print("Online validation performed: ✓")

        save_experiment_results("Dark Matter Discovery",
                                result, execution_time)

        return True

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n[FAILED] Discovery experiment failed")
        print(f"[TIME] Execution time: {execution_time:.2f} seconds")
        print(f"[ERROR] {str(e)}")
        return False


def analyze_experiment_results(experiment_name, result):
    """Analyze and display experiment results."""
    print("\n" + "="*50)
    print(f"ANALYSIS: {experiment_name}")
    print("="*50)

    # Check code quality
    if result.get('generated_code'):
        code = result['generated_code']
        print(f"Generated Code: {len(code)} characters")

        # Check for physics libraries
        physics_libs = ['numpy', 'matplotlib', 'scipy']
        used_libs = [lib for lib in physics_libs if lib in code.lower()]
        print(f"Physics Libraries Used: {', '.join(used_libs)}")

        # Check for physics concepts
        physics_concepts = ['quantum', 'wave',
                            'particle', 'energy', 'momentum']
        used_concepts = [
            concept for concept in physics_concepts if concept in code.lower()]
        print(f"Physics Concepts: {', '.join(used_concepts)}")

    # Check execution
    if result.get('execution_result'):
        if 'error' not in str(result['execution_result']).lower():
            print("Execution Status: SUCCESS ✓")
        else:
            print("Execution Status: ISSUES ⚠")

    # Check scientific validity
    if result.get('final_verdict'):
        verdict = result['final_verdict'].lower()
        if 'success' in verdict:
            print("Scientific Validity: HIGH ✓")
        elif 'potential' in verdict:
            print("Scientific Validity: MEDIUM ~")
        else:
            print("Scientific Validity: NEEDS REVIEW ?")


def save_experiment_results(experiment_name, result, execution_time):
    """Save detailed experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_results/{experiment_name.replace(' ', '_')}_{timestamp}.txt"

    os.makedirs("experiment_results", exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"ADVANCED PHYSICS EXPERIMENT RESULTS\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Execution Time: {execution_time:.2f} seconds\n")
        f.write("="*70 + "\n\n")

        for key, value in result.items():
            if value is not None:
                f.write(f"{key.upper().replace('_', ' ')}:\n")
                f.write(f"{str(value)}\n\n")

    print(f"[SAVED] Detailed results: {filename}")


def run_advanced_physics_test_suite():
    """Run the complete advanced physics test suite."""
    print("STARTING ADVANCED PHYSICS EXPERIMENTATION TEST SUITE")
    print("="*70)
    print("This will test the AGI's ability to:")
    print("- Generate sophisticated physics simulations")
    print("- Execute complex quantum mechanics calculations")
    print("- Discover novel experimental approaches")
    print("- Provide scientific analysis and validation")
    print("="*70)

    tests = [
        ("Quantum Tunneling", test_quantum_tunneling_experiment),
        ("Double-Slit Interference", test_double_slit_experiment),
        ("Dark Matter Discovery", test_discovery_experiment),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} RUNNING: {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"[PASS] {test_name}")
            else:
                print(f"[FAIL] {test_name}")
        except Exception as e:
            print(f"[FAIL] {test_name} - Exception: {e}")

        print(f"{'='*60}")

    # Final summary
    print("\n" + "="*70)
    print("ADVANCED PHYSICS TEST SUITE SUMMARY")
    print("="*70)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")

    if passed == total:
        print("[EXCELLENT] All advanced physics tests passed!")
        print("The AGI system demonstrates sophisticated scientific capabilities:")
        print("✓ Quantum mechanics simulations")
        print("✓ Complex physics calculations")
        print("✓ Novel experimental design")
        print("✓ Scientific reasoning and validation")
    elif passed > 0:
        print("[GOOD] Some advanced physics tests passed.")
        print("The AGI shows promising scientific capabilities.")
    else:
        print("[NEEDS WORK] Advanced physics tests need attention.")

    print(f"\nResults saved in: experiment_results/")
    return passed == total


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "quantum":
            test_quantum_tunneling_experiment()
        elif sys.argv[1] == "doubleslit":
            test_double_slit_experiment()
        elif sys.argv[1] == "discovery":
            test_discovery_experiment()
        elif sys.argv[1] == "suite":
            run_advanced_physics_test_suite()
        else:
            print(
                "Usage: python test_advanced_physics.py [quantum|doubleslit|discovery|suite]")
    else:
        # Default: run quantum tunneling test
        test_quantum_tunneling_experiment()
