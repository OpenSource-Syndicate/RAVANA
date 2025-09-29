"""
RAVANA Experiment Template
Use this as a starting point for your experiments
"""

from core.standard_config import StandardConfig
from database.engine import create_db_and_tables, engine
from core.system import AGISystem
import sys
import os
import asyncio
from datetime import datetime
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


async def run_experiment():
    """Run your experiment here"""
    print(f"[{datetime.now()}] Starting experiment...")

    # Example: Initialize the AGI system for your experiment
    try:
        create_db_and_tables()
        agi_system = AGISystem(engine)
        await agi_system.initialize_components()

        # Your experimental code here
        print("AGI system initialized for experiment")

        # Example: Run a simple task
        result = await agi_system.run_single_task("Analyze the current state of RAVANA capabilities")
        print(f"Task result: {result}")

    except Exception as e:
        print(f"Error during experiment: {e}")

    print(f"[{datetime.now()}] Experiment completed")


def save_results(results, filename=None):
    """Save experiment results to the results directory"""
    if filename is None:
        filename = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    results_path = os.path.join("analysis", "results", filename)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    asyncio.run(run_experiment())
