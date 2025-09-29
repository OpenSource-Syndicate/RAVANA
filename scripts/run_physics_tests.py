#!/usr/bin/env python3
"""
Simple test runner for physics experiments
Usage:
  python run_physics_tests.py                    # Run full test suite
  python run_physics_tests.py single             # Run single quick test
  python run_physics_tests.py discovery          # Run discovery tests only
  python run_physics_tests.py experiment <name>  # Run specific experiment
"""

import asyncio
import sys
from test_physics_experiments import (
    run_comprehensive_physics_tests,
    run_single_experiment_test,
    PhysicsExperimentTester
)
from physics_experiment_prompts import ADVANCED_PHYSICS_EXPERIMENTS, DISCOVERY_PROMPTS


def print_available_experiments():
    """Print list of available experiments."""
    print("\nAvailable Experiments:")
    print("-" * 50)
    for i, exp in enumerate(ADVANCED_PHYSICS_EXPERIMENTS, 1):
        print(f"{i:2d}. {exp['name']} ({exp['difficulty']})")
    print("-" * 50)


async def run_discovery_tests():
    """Run only discovery mode tests."""
    tester = PhysicsExperimentTester()

    try:
        await tester.setup()
        print("Running Discovery Mode Tests...")

        for i, prompt in enumerate(DISCOVERY_PROMPTS[:5], 1):  # Test first 5
            print(f"\nDiscovery Test {i}/5: {prompt[:80]}...")
            result = await tester.test_discovery_mode(prompt)

            if result['success']:
                print(
                    f"✓ Success - Creativity: {result['creativity_score']}, Plausibility: {result['scientific_plausibility']}")
            else:
                print(f"✗ Failed: {result.get('error', 'Unknown error')}")

        print(f"\nCompleted {len(DISCOVERY_PROMPTS[:5])} discovery tests")

    except Exception as e:
        print(f"Discovery tests failed: {e}")
    finally:
        await tester.cleanup()


def main():
    if len(sys.argv) == 1:
        # Run full test suite
        print("Running comprehensive physics experimentation test suite...")
        asyncio.run(run_comprehensive_physics_tests())

    elif sys.argv[1] == "single":
        # Run single quick test
        print("Running single experiment test...")
        asyncio.run(run_single_experiment_test())

    elif sys.argv[1] == "discovery":
        # Run discovery tests only
        asyncio.run(run_discovery_tests())

    elif sys.argv[1] == "experiment":
        if len(sys.argv) < 3:
            print_available_experiments()
            print("\nUsage: python run_physics_tests.py experiment <experiment_name>")
            return

        experiment_name = sys.argv[2]
        print(f"Running specific experiment: {experiment_name}")
        asyncio.run(run_single_experiment_test(experiment_name))

    elif sys.argv[1] == "list":
        print_available_experiments()

    elif sys.argv[1] == "help":
        print(__doc__)

    else:
        print("Unknown command. Use 'help' for usage information.")


if __name__ == "__main__":
    main()
