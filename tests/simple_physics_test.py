#!/usr/bin/env python3
"""
Simple Physics Experimentation Test
Tests the AGI experimentation engine directly without full system setup.
"""

from physics_experiment_prompts import ADVANCED_PHYSICS_EXPERIMENTS, DISCOVERY_PROMPTS
from core.llm import agi_experimentation_engine
import sys
import os
import time
import logging
from datetime import datetime

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_single_physics_experiment():
    """Test a single physics experiment using the experimentation engine."""

    # Choose a simple experiment to start with
    experiment = {
        "name": "Quantum Tunneling Simulation",
        "prompt": "Create a Python simulation of quantum tunneling through a rectangular potential barrier. Calculate the transmission probability for an electron with energy 5 eV trying to tunnel through a 10 eV barrier that is 1 nanometer wide.",
        "difficulty": "intermediate"
    }

    print("="*60)
    print(f"TESTING PHYSICS EXPERIMENT: {experiment['name']}")
    print(f"DIFFICULTY: {experiment['difficulty']}")
    print("="*60)

    start_time = time.time()

    try:
        logger.info(f"Starting experiment: {experiment['name']}")

        # Run the experimentation engine
        result = agi_experimentation_engine(
            experiment_idea=experiment['prompt'],
            llm_model=None,  # Use default model
            use_chain_of_thought=True,
            online_validation=True,
            sandbox_timeout=15,  # Shorter timeout for testing
            verbose=True
        )

        execution_time = time.time() - start_time

        print(f"\n✓ EXPERIMENT COMPLETED SUCCESSFULLY")
        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print(f"🧪 Simulation Type: {result.get('simulation_type', 'Unknown')}")
        print(f"📊 Final Verdict: {result.get('final_verdict', 'No verdict')}")

        # Show key results
        if result.get('generated_code'):
            print(f"\n📝 Generated Code Preview:")
            code_preview = result['generated_code'][:300] + "..." if len(
                result['generated_code']) > 300 else result['generated_code']
            print(code_preview)

        if result.get('execution_result'):
            print(f"\n🔬 Execution Result Preview:")
            exec_preview = str(result['execution_result'])[:200] + "..." if len(
                str(result['execution_result'])) > 200 else str(result['execution_result'])
            print(exec_preview)

        if result.get('online_validation'):
            print(f"\n🌐 Online Validation Preview:")
            validation_preview = str(result['online_validation'])[:200] + "..." if len(
                str(result['online_validation'])) > 200 else str(result['online_validation'])
            print(validation_preview)

        # Save detailed results
        save_experiment_results(experiment, result, execution_time)

        return True

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n✗ EXPERIMENT FAILED")
        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print(f"❌ Error: {str(e)}")
        logger.error(f"Experiment failed: {e}")
        return False


def test_discovery_mode():
    """Test the discovery mode with a creative physics prompt."""

    discovery_prompt = "What if we could create a device that uses quantum entanglement to instantly transmit information across any distance, bypassing the speed of light limitation?"

    print("="*60)
    print("TESTING DISCOVERY MODE")
    print("="*60)
    print(f"Prompt: {discovery_prompt}")

    start_time = time.time()

    try:
        logger.info("Starting discovery mode test")

        result = agi_experimentation_engine(
            experiment_idea=discovery_prompt,
            llm_model=None,
            use_chain_of_thought=True,
            online_validation=True,
            sandbox_timeout=15,
            verbose=True
        )

        execution_time = time.time() - start_time

        print(f"\n✓ DISCOVERY TEST COMPLETED")
        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print(
            f"🔬 Refined Idea: {result.get('refined_idea', 'No refinement')[:200]}...")
        print(
            f"📊 Final Assessment: {result.get('final_verdict', 'No verdict')}")

        return True

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"\n✗ DISCOVERY TEST FAILED")
        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print(f"❌ Error: {str(e)}")
        logger.error(f"Discovery test failed: {e}")
        return False


def save_experiment_results(experiment, result, execution_time):
    """Save experiment results to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_results/physics_test_{timestamp}.txt"

    os.makedirs("experiment_results", exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"PHYSICS EXPERIMENT TEST RESULTS\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"="*60 + "\n\n")

        f.write(f"EXPERIMENT: {experiment['name']}\n")
        f.write(f"DIFFICULTY: {experiment['difficulty']}\n")
        f.write(f"EXECUTION TIME: {execution_time:.2f} seconds\n\n")

        f.write(f"ORIGINAL PROMPT:\n{experiment['prompt']}\n\n")

        for key, value in result.items():
            if value is not None:
                f.write(f"{key.upper().replace('_', ' ')}:\n")
                f.write(f"{str(value)}\n\n")

    print(f"📁 Detailed results saved to: {filename}")


def run_quick_test_suite():
    """Run a quick test suite with multiple experiments."""
    print("🚀 STARTING QUICK PHYSICS EXPERIMENTATION TEST SUITE")
    print("="*60)

    tests_passed = 0
    total_tests = 0

    # Test 1: Single Physics Experiment
    print("\n🧪 TEST 1: PHYSICS EXPERIMENT")
    total_tests += 1
    if test_single_physics_experiment():
        tests_passed += 1

    # Test 2: Discovery Mode
    print("\n🔍 TEST 2: DISCOVERY MODE")
    total_tests += 1
    if test_discovery_mode():
        tests_passed += 1

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUITE SUMMARY")
    print("="*60)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {tests_passed/total_tests*100:.1f}%")

    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! The AGI experimentation system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs for details.")

    return tests_passed == total_tests


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "experiment":
            test_single_physics_experiment()
        elif sys.argv[1] == "discovery":
            test_discovery_mode()
        elif sys.argv[1] == "suite":
            run_quick_test_suite()
        else:
            print(
                "Usage: python simple_physics_test.py [experiment|discovery|suite]")
    else:
        # Default: run single experiment
        test_single_physics_experiment()
