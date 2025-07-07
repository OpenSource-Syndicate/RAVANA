import asyncio
import logging
import sys
import os
import signal
from pythonjsonlogger import jsonlogger
import argparse

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.system import AGISystem
from database.engine import create_db_and_tables, engine
from core.config import Config
from physics_experiment_prompts import ADVANCED_PHYSICS_EXPERIMENTS, DISCOVERY_PROMPTS
from modules.decision_engine.llm import agi_experimentation_engine

# Logging setup
def setup_logging():
    log_file = 'ravana_agi.log'
    # Overwrite the log file on each run
    if os.path.exists(log_file):
        os.remove(log_file)

    if Config.LOG_FORMAT.upper() == 'JSON':
        formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        
        # Stream handler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        # File handler for file output
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.handlers = [] # Clear existing handlers
        root_logger.addHandler(stream_handler)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(Config.LOG_LEVEL)
        root_logger.propagate = False
    else:
        # For text format, configure basicConfig with both stream and file handlers
        root_logger = logging.getLogger()
        root_logger.handlers = [] # Clear existing handlers
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Stream handler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        
        # File handler for file output
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        root_logger.addHandler(stream_handler)
        root_logger.addHandler(file_handler)
        root_logger.setLevel(Config.LOG_LEVEL)

setup_logging()
logger = logging.getLogger(__name__)

async def run_physics_experiment(agi_system, experiment_name):
    """Run a specific physics experiment through the AGI system."""
    logger.info(f"Running physics experiment: {experiment_name}")
    
    # Find the experiment
    experiment = None
    for exp in ADVANCED_PHYSICS_EXPERIMENTS:
        if experiment_name.lower() in exp['name'].lower():
            experiment = exp
            break
    
    if not experiment:
        logger.error(f"Physics experiment '{experiment_name}' not found")
        print(f"Available experiments:")
        for i, exp in enumerate(ADVANCED_PHYSICS_EXPERIMENTS, 1):
            print(f"{i:2d}. {exp['name']}")
        return
    
    # Create a task prompt for the AGI system
    task_prompt = f"""
    I want you to propose and test this advanced physics experiment:
    
    Experiment: {experiment['name']}
    Difficulty: {experiment['difficulty']}
    
    {experiment['prompt']}
    
    Please use the propose_and_test_invention action to formally submit this experiment 
    to the experimentation system. Make sure to save any plots as PNG files instead 
    of using plt.show() to avoid blocking execution.
    """
    
    logger.info(f"Starting AGI task for {experiment['name']}")
    await agi_system.run_single_task(task_prompt)
    logger.info(f"Physics experiment {experiment['name']} completed")

async def run_discovery_mode(agi_system):
    """Run the AGI in discovery mode to explore novel physics concepts."""
    logger.info("Starting AGI in discovery mode")
    
    import random
    discovery_prompt = random.choice(DISCOVERY_PROMPTS)
    
    task_prompt = f"""
    I want you to explore this fascinating physics question and propose novel experiments:
    
    Discovery Challenge: {discovery_prompt}
    
    Please think creatively and propose innovative experimental approaches that could 
    shed new light on this question. Use the propose_and_test_invention action to 
    formally submit your most promising idea for testing.
    
    Focus on:
    1. Novel experimental approaches
    2. Creative use of existing technology
    3. Theoretical feasibility
    4. Potential for new discoveries
    
    Save any plots as PNG files instead of using plt.show().
    """
    
    logger.info(f"Starting discovery mode with prompt: {discovery_prompt[:100]}...")
    await agi_system.run_single_task(task_prompt)
    logger.info("Discovery mode exploration completed")

async def run_experiment_tests(agi_system):
    """Run a comprehensive test of the physics experimentation system."""
    logger.info("Starting physics experimentation test suite")
    
    test_experiments = [
        "Quantum Tunneling Barrier Analysis",
        "Double-Slit Interference with Variable Parameters", 
        "Extreme Time Dilation Scenarios"
    ]
    
    for exp_name in test_experiments:
        logger.info(f"Testing experiment: {exp_name}")
        try:
            await run_physics_experiment(agi_system, exp_name)
            logger.info(f"✓ {exp_name} completed successfully")
        except Exception as e:
            logger.error(f"✗ {exp_name} failed: {e}")
    
    logger.info("Physics experimentation test suite completed")
    
    # Also test discovery mode
    logger.info("Testing discovery mode...")
    try:
        await run_discovery_mode(agi_system)
        logger.info("✓ Discovery mode test completed successfully")
    except Exception as e:
        logger.error(f"✗ Discovery mode test failed: {e}")
    
    logger.info("All experimentation tests completed")

async def main():
    """Main function to run the AGI system."""
    parser = argparse.ArgumentParser(description="Ravana AGI")
    parser.add_argument("--prompt", type=str, help="Run the AGI with a single prompt and then exit.")
    parser.add_argument("--physics-experiment", type=str, help="Run a specific physics experiment by name.")
    parser.add_argument("--discovery-mode", action="store_true", help="Run in discovery mode to explore novel physics concepts.")
    parser.add_argument("--test-experiments", action="store_true", help="Run physics experimentation test suite.")
    args = parser.parse_args()

    logger.info("Starting Ravana AGI...")
    
    # Create database and tables
    create_db_and_tables()

    # Initialize the AGI system
    agi_system = AGISystem(engine)
    
    # Handle graceful shutdown
    loop = asyncio.get_event_loop()
    
    # Add signal handlers for POSIX-based systems
    if os.name != 'nt':
        loop.add_signal_handler(signal.SIGINT, lambda: asyncio.create_task(agi_system.stop()))
        loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.create_task(agi_system.stop()))
    
    try:
        if args.physics_experiment:
            await run_physics_experiment(agi_system, args.physics_experiment)
        elif args.discovery_mode:
            await run_discovery_mode(agi_system)
        elif args.test_experiments:
            await run_experiment_tests(agi_system)
        elif args.prompt:
            await agi_system.run_single_task(args.prompt)
        else:
            await agi_system.run_autonomous_loop()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Main task interrupted or cancelled.")
    finally:
        logger.info("Shutting down...")
        if not agi_system._shutdown.is_set():
            await agi_system.stop()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit:
        logger.info("Ravana AGI stopped.")