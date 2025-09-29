#!/usr/bin/env python3
"""
Physics Experimentation CLI for AGI System
Easy-to-use interface for running physics experiments through the AGI.
"""

from physics_experiment_prompts import ADVANCED_PHYSICS_EXPERIMENTS, DISCOVERY_PROMPTS
import asyncio
import sys
import os

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def show_help():
    """Show help information."""
    print("""
AGI Physics Experimentation CLI

USAGE:
    python physics_cli.py <command> [options]

COMMANDS:
    list                    - Show all available physics experiments
    run <experiment_name>   - Run a specific physics experiment
    discovery              - Run discovery mode with random physics question
    test                   - Run comprehensive physics test suite
    help                   - Show this help message

EXAMPLES:
    python physics_cli.py list
    python physics_cli.py run "Quantum Tunneling"
    python physics_cli.py discovery
    python physics_cli.py test

EXPERIMENT INTEGRATION:
    The AGI system will:
    ✓ Analyze the physics problem scientifically
    ✓ Generate Python simulation code
    ✓ Execute experiments safely in sandbox
    ✓ Create visualizations and plots
    ✓ Provide scientific interpretation
    ✓ Cross-reference with real-world knowledge
    ✓ Store results in memory and knowledge base
    """)


def list_experiments():
    """List all available physics experiments."""
    print("\nAVAILABLE PHYSICS EXPERIMENTS:")
    print("=" * 70)

    # Group by difficulty
    difficulties = ['intermediate', 'advanced', 'expert']

    for difficulty in difficulties:
        experiments = [
            exp for exp in ADVANCED_PHYSICS_EXPERIMENTS if exp['difficulty'] == difficulty]
        if experiments:
            print(f"\n{difficulty.upper()} LEVEL:")
            print("-" * 30)
            for i, exp in enumerate(experiments, 1):
                print(f"{i:2d}. {exp['name']}")
                print(f"    {exp['prompt'][:80]}...")
                print()

    print(f"Total: {len(ADVANCED_PHYSICS_EXPERIMENTS)} experiments available")
    print("\nTo run an experiment: python physics_cli.py run \"<experiment_name>\"")


def show_discovery_prompts():
    """Show available discovery prompts."""
    print("\nDISCOVERY MODE PROMPTS:")
    print("=" * 70)

    for i, prompt in enumerate(DISCOVERY_PROMPTS, 1):
        print(f"{i:2d}. {prompt}")
        print()

    print("Discovery mode will randomly select one of these prompts.")


async def run_experiment(experiment_name):
    """Run a specific physics experiment."""
    print(f"Starting AGI with physics experiment: {experiment_name}")

    # Use the main AGI system
    os.system(
        f'uv run python main.py --physics-experiment "{experiment_name}"')


async def run_discovery():
    """Run discovery mode."""
    print("Starting AGI in discovery mode...")

    # Use the main AGI system
    os.system('uv run python main.py --discovery-mode')


async def run_tests():
    """Run comprehensive physics tests."""
    print("Starting comprehensive physics experimentation tests...")

    # Use the main AGI system
    os.system('uv run python main.py --test-experiments')


def main():
    """Main CLI function."""
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()

    if command == "help" or command == "-h" or command == "--help":
        show_help()

    elif command == "list":
        list_experiments()

    elif command == "discovery-prompts":
        show_discovery_prompts()

    elif command == "run":
        if len(sys.argv) < 3:
            print("Error: Please specify an experiment name")
            print("Usage: python physics_cli.py run \"<experiment_name>\"")
            print("\nUse 'python physics_cli.py list' to see available experiments")
            return

        experiment_name = sys.argv[2]
        asyncio.run(run_experiment(experiment_name))

    elif command == "discovery":
        asyncio.run(run_discovery())

    elif command == "test":
        asyncio.run(run_tests())

    else:
        print(f"Unknown command: {command}")
        print("Use 'python physics_cli.py help' for usage information")


if __name__ == "__main__":
    main()
