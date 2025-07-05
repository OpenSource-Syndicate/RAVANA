import asyncio
import logging
import sys
import os
import signal
from pythonjsonlogger import jsonlogger

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.system import AGISystem
from database.engine import create_db_and_tables, engine
from core.config import Config

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

async def main():
    """Main function to run the AGI system."""
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