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
                if agi_system_instance:
                    # Use thread to call async shutdown
                    threading.Thread(
                        target=lambda: asyncio.run(agi_system_instance.stop("signal")),
                        daemon=True
                    ).start()
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
                        if agi_system_instance:
                            threading.Thread(
                                target=lambda: asyncio.run(agi_system_instance.stop("console_event")),
                                daemon=True
                            ).start()
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
            def posix_signal_handler():
                logger.info("🛑 Received shutdown signal on POSIX system")
                if agi_system_instance:
                    asyncio.create_task(agi_system_instance.stop("signal"))
                shutdown_event.set()
            
            # Get current event loop
            loop = asyncio.get_event_loop()
            
            # Add signal handlers to the event loop
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, posix_signal_handler)
            
            # Additional POSIX signals
            try:
                loop.add_signal_handler(signal.SIGHUP, posix_signal_handler)
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
    """Main function to run the AGI system with enhanced shutdown handling."""
    global agi_system_instance
    
    parser = argparse.ArgumentParser(description="Ravana AGI")
    parser.add_argument("--prompt", type=str, help="Run the AGI with a single prompt and then exit.")
    parser.add_argument("--physics-experiment", type=str, help="Run a specific physics experiment by name.")
    parser.add_argument("--discovery-mode", action="store_true", help="Run in discovery mode to explore novel physics concepts.")
    parser.add_argument("--test-experiments", action="store_true", help="Run physics experimentation test suite.")
    parser.add_argument("--skip-state-recovery", action="store_true", help="Skip loading previous state on startup.")
    args = parser.parse_args()

    logger.info("🚀 Starting Ravana AGI System...")
    logger.info(f"💻 Platform: {platform.system()} {platform.release()}")
    logger.info(f"🐍 Python: {sys.version}")
    
    # Create database and tables
    logger.info("🗄 Initializing database...")
    create_db_and_tables()

    # Initialize the AGI system
    logger.info("🧠 Initializing AGI system...")
    agi_system_instance = AGISystem(engine)
    
    # Set up signal handlers after AGI system is initialized
    setup_signal_handlers()
    
    # Log startup configuration
    from core.config import Config
    logger.info(f"⚙️  Graceful shutdown: {'enabled' if Config.GRACEFUL_SHUTDOWN_ENABLED else 'disabled'}")
    logger.info(f"💾 State persistence: {'enabled' if Config.STATE_PERSISTENCE_ENABLED else 'disabled'}")
    logger.info(f"⏱️  Shutdown timeout: {Config.SHUTDOWN_TIMEOUT}s")
    
    try:
        # Run the appropriate mode
        if args.physics_experiment:
            logger.info(f"🔬 Running physics experiment: {args.physics_experiment}")
            await run_physics_experiment(agi_system_instance, args.physics_experiment)
            
        elif args.discovery_mode:
            logger.info("🔍 Running in discovery mode")
            await run_discovery_mode(agi_system_instance)
            
        elif args.test_experiments:
            logger.info("🧪 Running physics experimentation test suite")
            await run_experiment_tests(agi_system_instance)
            
        elif args.prompt:
            logger.info(f"📝 Running single task: {args.prompt[:100]}...")
            await run_single_task_with_shutdown(agi_system_instance, args.prompt)
            
        else:
            logger.info("🔄 Starting autonomous loop")
            await run_autonomous_with_shutdown(agi_system_instance)
            
    except KeyboardInterrupt:
        logger.info("⚡ Keyboard interrupt received")
        await shutdown_agi_system(agi_system_instance, "keyboard_interrupt")
        
    except asyncio.CancelledError:
        logger.info("⚡ Async task cancelled")
        await shutdown_agi_system(agi_system_instance, "task_cancelled")
        
    except Exception as e:
        logger.error(f"❌ Critical error in main: {e}", exc_info=True)
        await shutdown_agi_system(agi_system_instance, "critical_error")
        raise
        
    finally:
        logger.info("📋 Final cleanup in main()")
        if agi_system_instance and not agi_system_instance._shutdown.is_set():
            try:
                await shutdown_agi_system(agi_system_instance, "finally_block")
            except Exception as e:
                logger.error(f"Error in final cleanup: {e}")
        
        logger.info("✅ Ravana AGI shutdown sequence completed")


async def run_single_task_with_shutdown(agi_system: AGISystem, prompt: str):
    """Run a single task with shutdown monitoring."""
    try:
        # Create a task for the single task execution
        task = asyncio.create_task(agi_system.run_single_task(prompt))
        
        # Wait for either task completion or shutdown signal
        done, pending = await asyncio.wait(
            [task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for pending_task in pending:
            pending_task.cancel()
        
        # Check if shutdown was requested
        if shutdown_event.is_set():
            logger.info("🛑 Shutdown requested during single task execution")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
    except Exception as e:
        logger.error(f"Error in single task execution: {e}")
        raise


async def run_autonomous_with_shutdown(agi_system: AGISystem):
    """Run autonomous loop with shutdown monitoring."""
    try:
        # Create a task for the autonomous loop
        loop_task = asyncio.create_task(agi_system.run_autonomous_loop())
        
        # Wait for either loop completion or shutdown signal
        done, pending = await asyncio.wait(
            [loop_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for pending_task in pending:
            pending_task.cancel()
        
        # Check if shutdown was requested
        if shutdown_event.is_set():
            logger.info("🛑 Shutdown requested during autonomous loop")
            # The loop should already be stopping due to the shutdown event being set
            try:
                await asyncio.wait_for(loop_task, timeout=Config.SHUTDOWN_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("⚠️  Autonomous loop did not stop within timeout")
                loop_task.cancel()
                try:
                    await loop_task
                except asyncio.CancelledError:
                    pass
        
    except Exception as e:
        logger.error(f"Error in autonomous loop execution: {e}")
        raise


async def shutdown_agi_system(agi_system: AGISystem, reason: str):
    """Shutdown the AGI system gracefully."""
    if not agi_system:
        logger.warning("No AGI system instance to shutdown")
        return
    
    try:
        logger.info(f"🛑 Initiating AGI system shutdown - Reason: {reason}")
        
        # Check if graceful shutdown is enabled
        from core.config import Config
        if Config.GRACEFUL_SHUTDOWN_ENABLED:
            await agi_system.stop(reason)
        else:
            logger.info("⚡ Graceful shutdown disabled, performing basic shutdown")
            agi_system._shutdown.set()
            
            # Cancel background tasks
            for task in agi_system.background_tasks:
                task.cancel()
            
            # Wait briefly for tasks to complete
            try:
                await asyncio.wait_for(
                    asyncio.gather(*agi_system.background_tasks, return_exceptions=True),
                    timeout=5
                )
            except asyncio.TimeoutError:
                logger.warning("Some background tasks did not complete within timeout")
            
            # Close database session
            if hasattr(agi_system, 'session'):
                agi_system.session.close()
        
        logger.info("✅ AGI system shutdown completed")
        
    except Exception as e:
        logger.error(f"❌ Error during AGI system shutdown: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        logger.info("🚀 Ravana AGI System starting up...")
        asyncio.run(main())
    except SystemExit as e:
        logger.info(f"📊 System exit requested: {e}")
    except KeyboardInterrupt:
        logger.info("⚡ Process interrupted by user")
    except Exception as e:
        logger.critical(f"❌ Critical startup error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("👋 Ravana AGI process terminated.")