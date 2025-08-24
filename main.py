import asyncio
import logging
import sys
import os
import signal
import io
from pythonjsonlogger import jsonlogger
import argparse
import platform
import threading
from typing import Optional

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.system import AGISystem
from database.engine import create_db_and_tables, engine
from core.config import Config
from physics_experiment_prompts import ADVANCED_PHYSICS_EXPERIMENTS, DISCOVERY_PROMPTS
from core.llm import agi_experimentation_engine

# Logging setup
def setup_logging():
    log_file = 'ravana_agi.log'
    # Overwrite the log file on each run
    if os.path.exists(log_file):
        os.remove(log_file)
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Clear existing handlers

    # Choose formatter based on config
    if Config.LOG_FORMAT.upper() == 'JSON':
        formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    else:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Wrap stdout to ensure UTF-8 encoding and safe error handling on Windows consoles
    safe_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    stream_handler = logging.StreamHandler(stream=safe_stdout)
    stream_handler.setFormatter(formatter)

    # File handler ensures UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(Config.LOG_LEVEL)
    if Config.LOG_FORMAT.upper() == 'JSON':
        root_logger.propagate = False

setup_logging()
logger = logging.getLogger(__name__)

# Global shutdown event for cross-platform signal handling
shutdown_event = asyncio.Event()
agi_system_instance: Optional[AGISystem] = None


def setup_signal_handlers():
    """Set up cross-platform signal handlers for graceful shutdown."""
    try:
        if platform.system() == "Windows":
            # Windows signal handling
            import signal
            
            def windows_signal_handler(signum, frame):
                logger.info(f"🛑 Received signal {signum} on Windows")
                # Set the shutdown event directly instead of calling async code
                shutdown_event.set()
            
            signal.signal(signal.SIGINT, windows_signal_handler)
            signal.signal(signal.SIGTERM, windows_signal_handler)
            
            # Windows-specific: Handle console control events
            try:
                import win32api
                def console_ctrl_handler(ctrl_type):
                    if ctrl_type in (win32api.CTRL_C_EVENT, win32api.CTRL_BREAK_EVENT, 
                                   win32api.CTRL_CLOSE_EVENT, win32api.CTRL_SHUTDOWN_EVENT):
                        logger.info(f"🛑 Received Windows console control event: {ctrl_type}")
                        shutdown_event.set()
                        return True
                    return False
                
                win32api.SetConsoleCtrlHandler(console_ctrl_handler, True)
                logger.info("✅ Windows console control handler registered")
                
            except ImportError:
                logger.info("ℹ️  pywin32 not available, using basic Windows signal handling")
            
            logger.info("✅ Windows signal handlers configured")
            
        else:
            # POSIX signal handling (Linux, macOS, etc.)
            def posix_signal_handler(signum, frame):
                logger.info(f"🛑 Received signal {signum} on POSIX system")
                shutdown_event.set()
            
            # Get current event loop
            loop = asyncio.get_event_loop()
            
            # Add signal handlers to the event loop
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda s=sig: posix_signal_handler(s, None))
            
            # Additional POSIX signals
            try:
                loop.add_signal_handler(signal.SIGHUP, lambda: posix_signal_handler(signal.SIGHUP, None))
                logger.info("✅ POSIX signal handlers configured (SIGINT, SIGTERM, SIGHUP)")
            except (AttributeError, NotImplementedError):
                logger.info("✅ POSIX signal handlers configured (SIGINT, SIGTERM)")
    
    except Exception as e:
        logger.error(f"❌ Error setting up signal handlers: {e}")
        logger.info("⚠️  Continuing without enhanced signal handling")

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
    """Main entry point for the RAVANA AGI system."""
    global agi_system_instance
    
    logger.info("🚀 Starting RAVANA AGI System")
    logger.info(f"🧠 Using model: {Config.EMBEDDING_MODEL}")
    logger.info(f"📊 Log level: {Config.LOG_LEVEL}")
    
    try:
        # Create database and tables
        logger.info("Initializing database...")
        create_db_and_tables()
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return 1
    
    # Set up signal handlers for graceful shutdown
    setup_signal_handlers()
    
    agi_system = None
    try:
        # Initialize the AGI system with improved error handling
        logger.info("Initializing AGI system...")
        agi_system = AGISystem(engine)
        agi_system_instance = agi_system
        
        # Initialize components with better error handling
        initialization_success = await agi_system.initialize_components()
        if not initialization_success:
            logger.error("AGI system initialization failed")
            return 1
            
        logger.info("AGI system initialized successfully")
        
        # Start Snake Agent if enabled
        if Config.SNAKE_AGENT_ENABLED and agi_system.snake_agent:
            try:
                logger.info("Starting Snake Agent...")
                await agi_system.start_snake_agent()
                logger.info("Snake Agent started successfully")
            except Exception as e:
                logger.error(f"Failed to start Snake Agent: {e}")
                # Continue even if Snake Agent fails to start
        
        # Start Conversational AI if enabled
        if Config.CONVERSATIONAL_AI_ENABLED and agi_system.conversational_ai:
            try:
                logger.info("Starting Conversational AI...")
                await agi_system.start_conversational_ai()
                logger.info("Conversational AI started successfully")
            except Exception as e:
                logger.error(f"Failed to start Conversational AI: {e}")
                # Continue even if Conversational AI fails to start
        
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="RAVANA AGI System")
        parser.add_argument("--physics-experiment", type=str, help="Run a specific physics experiment")
        parser.add_argument("--discovery-mode", action="store_true", help="Run in discovery mode")
        parser.add_argument("--test-experiments", action="store_true", help="Run experiment tests")
        parser.add_argument("--single-task", type=str, help="Run a single task")
        
        args = parser.parse_args()
        
        # Handle different run modes
        if args.physics_experiment:
            logger.info(f"Running physics experiment: {args.physics_experiment}")
            await run_physics_experiment(agi_system, args.physics_experiment)
        elif args.discovery_mode:
            logger.info("Running in discovery mode")
            await run_discovery_mode(agi_system)
        elif args.test_experiments:
            logger.info("Running experiment tests")
            await run_experiment_tests(agi_system)
        elif args.single_task:
            logger.info(f"Running single task: {args.single_task}")
            await agi_system.run_single_task(args.single_task)
        else:
            # Start the autonomous loop
            logger.info("Starting autonomous AGI loop...")
            await agi_system.run_autonomous_loop()
            
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Unexpected error in main: {e}", exc_info=True)
        return 1
    finally:
        # Ensure graceful shutdown
        if agi_system:
            logger.info("Initiating graceful shutdown...")
            try:
                await agi_system.stop("system_shutdown")
                logger.info("✅ AGI System shutdown completed")
            except Exception as e:
                logger.error(f"❌ Error during shutdown: {e}", exc_info=True)
                return 1
    
    return 0

if __name__ == "__main__":
    # Run the main async function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
