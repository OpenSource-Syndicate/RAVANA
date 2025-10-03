from core.llm import agi_experimentation_engine
from scripts.physics_experiment_prompts import ADVANCED_PHYSICS_EXPERIMENTS, DISCOVERY_PROMPTS
from core.config import Config
from database.engine import create_db_and_tables, get_engine

engine = get_engine()
from core.system import AGISystem
import asyncio
import logging
import sys
import os
import io
from pythonjsonlogger import jsonlogger
import argparse
import platform
from typing import Optional

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Logging setup

def setup_logging():
    log_file = 'ravana_agi.log'
    # Overwrite the log file on each run
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except PermissionError:
            # If we can't remove the file (e.g., it's being used by another process on Windows),
            # we'll append to it instead of overwriting
            pass
        except Exception as e:
            # Handle any other unexpected errors
            pass
    root_logger = logging.getLogger()
    root_logger.handlers = []  # Clear existing handlers

    # Access Config instance to ensure it's initialized
    config = Config()
    
    # Choose formatter based on config, with fallback in case LOG_FORMAT is missing
    try:
        log_format = config.LOG_FORMAT.upper()
    except AttributeError:
        # Fallback to default if LOG_FORMAT is missing
        log_format = 'TEXT'
    
    if log_format == 'JSON':
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s')
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Wrap stdout to ensure UTF-8 encoding and safe error handling on Windows consoles
    safe_stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    stream_handler = logging.StreamHandler(stream=safe_stdout)
    stream_handler.setFormatter(formatter)

    # File handler ensures UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    
    # Set logger level with fallback in case LOG_LEVEL is missing
    try:
        log_level = config.LOG_LEVEL
    except AttributeError:
        # Fallback to default if LOG_LEVEL is missing
        log_level = 'INFO'
    root_logger.setLevel(log_level)
    
    if log_format == 'JSON':
        root_logger.propagate = False


setup_logging()
logger = logging.getLogger(__name__)


def pull_required_models():
    """Pull required models before system initialization."""
    logger.info("Pulling required models...")
    
    # Use the main system local model settings from Config
    config = Config()  # Ensure Config is initialized
    local_model_config = config.MAIN_SYSTEM_LOCAL_MODEL
    
    # Check if we can connect to Ollama server
    base_url = local_model_config['base_url']
    model_to_pull = local_model_config['model_name']
    
    try:
        import requests  # Import here to make sure it's available
        # Check if Ollama server is available
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            available_models = [m['name'] for m in response.json().get('models', [])]
            
            # Check if the required model is already available
            if model_to_pull not in available_models:
                logger.info(f"Pulling model: {model_to_pull}")
                pull_response = requests.post(f"{base_url}/api/pull", 
                                            json={"name": model_to_pull}, 
                                            timeout=local_model_config.get('timeout', 300))
                
                if pull_response.status_code == 200:
                    logger.info(f"Successfully pulled model: {model_to_pull}")
                else:
                    logger.error(f"Failed to pull model: {model_to_pull}. Status: {pull_response.status_code}")
            else:
                logger.info(f"Model {model_to_pull} already exists locally")
        else:
            logger.error(f"Failed to connect to Ollama server: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Ollama server at {base_url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error while pulling models: {e}")
    
    # Also check for any additional models that might be needed
    additional_models = [
        "all-MiniLM-L6-v2",  # For embeddings
        "llama3.1:8b",       # Alternative model
        "mistral:7b"         # Alternative model
    ]
    
    try:
        import requests  # Import here to make sure it's available
        # Check if Ollama server is available
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            available_models = [m['name'] for m in response.json().get('models', [])]
            
            for model in additional_models:
                if model not in available_models:
                    logger.info(f"Pulling additional model: {model}")
                    pull_response = requests.post(f"{base_url}/api/pull", 
                                               json={"name": model}, 
                                               timeout=local_model_config.get('timeout', 300))
                    
                    if pull_response.status_code == 200:
                        logger.info(f"Successfully pulled model: {model}")
                    else:
                        logger.error(f"Failed to pull model: {model}. Status: {pull_response.status_code}")
                else:
                    logger.info(f"Model {model} already exists locally")
    except Exception as e:
        logger.error(f"Error checking/pulling additional models: {e}")
    
    logger.info("Model pulling process completed")

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
                    # Create a list of valid control event constants
                    valid_ctrl_events = []

                    # Add the signal constants that are available
                    if hasattr(signal, 'CTRL_C_EVENT'):
                        valid_ctrl_events.append(signal.CTRL_C_EVENT)
                    if hasattr(signal, 'CTRL_BREAK_EVENT'):
                        valid_ctrl_events.append(signal.CTRL_BREAK_EVENT)
                    if hasattr(signal, 'CTRL_CLOSE_EVENT'):
                        valid_ctrl_events.append(signal.CTRL_CLOSE_EVENT)
                    if hasattr(signal, 'CTRL_SHUTDOWN_EVENT'):
                        valid_ctrl_events.append(signal.CTRL_SHUTDOWN_EVENT)

                    # For backward compatibility, also check for win32api constants if they exist
                    # (though they don't in this case)
                    if ctrl_type in valid_ctrl_events:
                        logger.info(
                            f"🛑 Received Windows console control event: {ctrl_type}")
                        shutdown_event.set()
                        return True
                    return False

                win32api.SetConsoleCtrlHandler(console_ctrl_handler, True)
                logger.info("✅ Windows console control handler registered")

            except ImportError:
                logger.info(
                    "ℹ️  pywin32 not available, using basic Windows signal handling")

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
                loop.add_signal_handler(
                    sig, lambda s=sig: posix_signal_handler(s, None))

            # Additional POSIX signals
            try:
                loop.add_signal_handler(
                    signal.SIGHUP, lambda: posix_signal_handler(signal.SIGHUP, None))
                logger.info(
                    "✅ POSIX signal handlers configured (SIGINT, SIGTERM, SIGHUP)")
            except (AttributeError, NotImplementedError):
                logger.info(
                    "✅ POSIX signal handlers configured (SIGINT, SIGTERM)")

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

    logger.info(
        f"Starting discovery mode with prompt: {discovery_prompt[:100]}...")
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

    logger.info("Starting RAVANA AGI System")
    config = Config()  # Ensure Config is initialized
    logger.info(f"Using model: {config.EMBEDDING_MODEL}")
    logger.info(f"📊 Log level: {config.LOG_LEVEL}")

    # Pull required models before initializing the system
    logger.info("📦 Pulling required models before system initialization...")
    pull_required_models()

    try:
        # Create database and tables
        logger.info("Initializing database...")
        create_db_and_tables(engine)
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
        config_instance = Config()  # Get instance of config
        if config_instance.SNAKE_AGENT_ENABLED and agi_system.snake_agent:
            try:
                logger.info("Starting Snake Agent...")
                await agi_system.start_snake_agent()
                logger.info("Snake Agent started successfully")
            except Exception as e:
                logger.error(f"Failed to start Snake Agent: {e}")
                # Continue even if Snake Agent fails to start

        # Start Bots directly if enabled (instead of Conversational AI)
        if config_instance.CONVERSATIONAL_AI_ENABLED:
            try:
                logger.info("Starting bots directly...")
                # Import bot classes
                from modules.conversational_ai.bots.discord_bot import DiscordBot
                from modules.conversational_ai.bots.telegram_bot import TelegramBot
                from modules.conversational_ai.main import ConversationalAI

                # Create conversational AI instance to get config
                ai = ConversationalAI()

                # Store bot instances and tasks globally for cleanup
                global discord_bot_instance, telegram_bot_instance, bot_tasks
                discord_bot_instance = None
                telegram_bot_instance = None
                bot_tasks = []

                # Start Discord bot if enabled
                if ai.config.get("platforms", {}).get("discord", {}).get("enabled", False):
                    try:
                        token = ai.config.get("discord_token")
                        if token:
                            discord_bot_instance = DiscordBot(
                                token=token,
                                command_prefix=ai.config["platforms"]["discord"]["command_prefix"],
                                conversational_ai=ai
                            )

                            # Start Discord bot in a separate task
                            async def discord_runner():
                                try:
                                    await discord_bot_instance.start()
                                except Exception as e:
                                    logger.error(f"Error in Discord bot: {e}")

                            discord_task = asyncio.create_task(
                                discord_runner())
                            bot_tasks.append(discord_task)
                            logger.info("Discord bot task created")
                        else:
                            logger.warning(
                                "Discord token not found, skipping Discord bot")
                    except Exception as e:
                        logger.error(f"Failed to start Discord bot: {e}")

                # Start Telegram bot if enabled
                if ai.config.get("platforms", {}).get("telegram", {}).get("enabled", False):
                    try:
                        token = ai.config.get("telegram_token")
                        if token:
                            telegram_bot_instance = await TelegramBot.get_instance(
                                token=token,
                                command_prefix=ai.config["platforms"]["telegram"]["command_prefix"],
                                conversational_ai=ai
                            )

                            # Start Telegram bot in a separate task
                            async def telegram_runner():
                                try:
                                    await telegram_bot_instance.start()
                                except Exception as e:
                                    logger.error(f"Error in Telegram bot: {e}")

                            telegram_task = asyncio.create_task(
                                telegram_runner())
                            bot_tasks.append(telegram_task)
                            logger.info("Telegram bot task created")
                        else:
                            logger.warning(
                                "Telegram token not found, skipping Telegram bot")
                    except Exception as e:
                        logger.error(f"Failed to start Telegram bot: {e}")

                if bot_tasks:
                    logger.info(
                        f"Bots started successfully ({len(bot_tasks)} bot(s) running)")
                else:
                    logger.info(
                        "No bots were started (none enabled or configured)")

            except Exception as e:
                logger.error(f"Failed to start bots: {e}")
                logger.exception("Full traceback:")
                # Continue even if bots fail to start
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="RAVANA AGI System")
        parser.add_argument("--physics-experiment", type=str,
                            help="Run a specific physics experiment")
        parser.add_argument("--discovery-mode",
                            action="store_true", help="Run in discovery mode")
        parser.add_argument("--test-experiments",
                            action="store_true", help="Run experiment tests")
        parser.add_argument("--single-task", type=str,
                            help="Run a single task")

        args = parser.parse_args()

        # Background task to monitor global shutdown event and trigger AGI system shutdown
        async def monitor_shutdown_event():
            """Monitor the global shutdown event and trigger AGI system shutdown."""
            try:
                while not shutdown_event.is_set():
                    await asyncio.sleep(0.1)  # Check every 100ms

                logger.info(
                    "Global shutdown event detected, triggering AGI system shutdown...")
                if agi_system:
                    await agi_system.stop("global_shutdown_event")
            except Exception as e:
                logger.error(f"Error in shutdown monitor: {e}")

        # Handle different run modes
        if args.physics_experiment:
            logger.info(
                f"Running physics experiment: {args.physics_experiment}")
            # Start shutdown monitor for single task mode
            shutdown_monitor_task = asyncio.create_task(
                monitor_shutdown_event())
            await run_physics_experiment(agi_system, args.physics_experiment)
            # Cancel the shutdown monitor task
            shutdown_monitor_task.cancel()
        elif args.discovery_mode:
            logger.info("Running in discovery mode")
            # Start shutdown monitor for discovery mode
            shutdown_monitor_task = asyncio.create_task(
                monitor_shutdown_event())
            await run_discovery_mode(agi_system)
            # Cancel the shutdown monitor task
            shutdown_monitor_task.cancel()
        elif args.test_experiments:
            logger.info("Running experiment tests")
            # Start shutdown monitor for test experiments mode
            shutdown_monitor_task = asyncio.create_task(
                monitor_shutdown_event())
            await run_experiment_tests(agi_system)
            # Cancel the shutdown monitor task
            shutdown_monitor_task.cancel()
        elif args.single_task:
            logger.info(f"Running single task: {args.single_task}")
            # Start shutdown monitor for single task mode
            shutdown_monitor_task = asyncio.create_task(
                monitor_shutdown_event())
            await agi_system.run_single_task(args.single_task)
            # Cancel the shutdown monitor task
            shutdown_monitor_task.cancel()
        else:
            # Start the autonomous loop
            logger.info("Starting autonomous AGI loop...")
            # Log the availability of new enhanced systems
            if hasattr(agi_system, 'function_calling_system'):
                logger.info("Function calling system is available for advanced operations")
            if hasattr(agi_system, 'physics_analysis_system'):
                logger.info("Physics analysis system is available for scientific calculations")
            if hasattr(agi_system, 'failure_learning_system'):
                logger.info("Failure learning system is available for continuous improvement")
            # Start shutdown monitor for autonomous loop mode
            shutdown_monitor_task = asyncio.create_task(
                monitor_shutdown_event())
            await agi_system.run_autonomous_loop()
            # Cancel the shutdown monitor task (though it should have completed by now)
            shutdown_monitor_task.cancel()

    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ Unexpected error in main: {e}", exc_info=True)
        return 1
    finally:
        # Ensure graceful shutdown
        # Stop bots if they were started
        if 'bot_tasks' in globals() and bot_tasks:
            logger.info("Stopping bots...")
            # Cancel bot tasks
            for task in bot_tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete cancellation
            if bot_tasks:
                try:
                    await asyncio.wait_for(asyncio.gather(*bot_tasks, return_exceptions=True), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for bot tasks to cancel")

            # Stop bot instances if they exist
            if 'discord_bot_instance' in globals() and discord_bot_instance:
                try:
                    await discord_bot_instance.stop()
                    logger.info("Discord bot stopped")
                except Exception as e:
                    logger.error(f"Error stopping Discord bot: {e}")

            if 'telegram_bot_instance' in globals() and telegram_bot_instance:
                try:
                    await telegram_bot_instance.stop()
                    logger.info("Telegram bot stopped")
                except Exception as e:
                    logger.error(f"Error stopping Telegram bot: {e}")

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
