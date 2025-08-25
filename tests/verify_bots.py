#!/usr/bin/env python3
"""
Script to run bots indefinitely
"""
import asyncio
import sys
import os
import logging
import signal

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global shutdown event
shutdown_event = asyncio.Event()
discord_bot_instance = None
telegram_bot_instance = None

def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        shutdown_event.set()
    
    # Only set up signal handlers once
    if not hasattr(setup_signal_handlers, '_configured'):
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        setup_signal_handlers._configured = True

async def run_discord_bot():
    """Run Discord bot indefinitely."""
    global discord_bot_instance
    logger.info("Starting Discord bot...")
    
    try:
        # Load config
        from modules.conversational_ai.main import ConversationalAI
        ai = ConversationalAI()
        
        if not ai.config.get("platforms", {}).get("discord", {}).get("enabled", False):
            logger.info("Discord bot is not enabled")
            return False
            
        token = ai.config.get("discord_token")
        if not token:
            logger.error("Discord token not found")
            return False
            
        # Import and create bot
        from modules.conversational_ai.bots.discord_bot import DiscordBot
        discord_bot_instance = DiscordBot(
            token=token,
            command_prefix=ai.config["platforms"]["discord"]["command_prefix"],
            conversational_ai=ai
        )
        
        logger.info("Discord bot instance created")
        
        # Start bot in a separate task since it blocks
        async def discord_runner():
            await discord_bot_instance.start()
            
        discord_task = asyncio.create_task(discord_runner())
        # Store the task reference
        global discord_bot_task
        discord_bot_task = discord_task
        
        # Wait a bit to see if it connects
        for i in range(10):  # Wait up to 10 seconds
            await asyncio.sleep(1)
            if hasattr(discord_bot_instance, 'connected') and discord_bot_instance.connected:
                logger.info("Discord bot is connected and running")
                return True
                
        logger.warning("Discord bot may not be connected")
        return True  # Still return True as the task is running
        
    except Exception as e:
        logger.error(f"Error starting Discord bot: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_telegram_bot():
    """Run Telegram bot indefinitely."""
    global telegram_bot_instance
    logger.info("Starting Telegram bot...")
    
    try:
        # Load config
        from modules.conversational_ai.main import ConversationalAI
        ai = ConversationalAI()
        
        if not ai.config.get("platforms", {}).get("telegram", {}).get("enabled", False):
            logger.info("Telegram bot is not enabled")
            return False
            
        token = ai.config.get("telegram_token")
        if not token:
            logger.error("Telegram token not found")
            return False
            
        # Import and create bot
        from modules.conversational_ai.bots.telegram_bot import TelegramBot
        telegram_bot_instance = TelegramBot(
            token=token,
            command_prefix=ai.config["platforms"]["telegram"]["command_prefix"],
            conversational_ai=ai
        )
        
        logger.info("Telegram bot instance created")
        
        # Start bot in a separate task since it has its own loop
        async def telegram_runner():
            await telegram_bot_instance.start()
            
        telegram_task = asyncio.create_task(telegram_runner())
        # Store the task reference
        global telegram_bot_task
        telegram_bot_task = telegram_task
        
        # Wait a bit to see if it connects
        for i in range(10):  # Wait up to 10 seconds
            await asyncio.sleep(1)
            if hasattr(telegram_bot_instance, 'connected') and telegram_bot_instance.connected:
                logger.info("Telegram bot is connected and running")
                return True
                
        logger.warning("Telegram bot may not be connected")
        return True  # Still return True as the task is running
        
    except Exception as e:
        logger.error(f"Error starting Telegram bot: {e}")
        import traceback
        traceback.print_exc()
        return False

async def stop_bots():
    """Stop all running bots."""
    logger.info("Stopping bots...")
    
    # Cancel bot tasks if they exist
    tasks_to_cancel = []
    
    if 'discord_bot_task' in globals():
        tasks_to_cancel.append(discord_bot_task)
        
    if 'telegram_bot_task' in globals():
        tasks_to_cancel.append(telegram_bot_task)
    
    # Cancel any running tasks
    for task in tasks_to_cancel:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    # Stop Discord bot if running
    if discord_bot_instance:
        try:
            await discord_bot_instance.stop()
            logger.info("Discord bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Discord bot: {e}")
    
    # Stop Telegram bot if running
    if telegram_bot_instance:
        try:
            await telegram_bot_instance.stop()
            logger.info("Telegram bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {e}")

async def main():
    """Main function to run bots indefinitely."""
    logger.info("Starting bots indefinitely...")
    
    # Set up signal handlers
    setup_signal_handlers()
    
    # Start Discord bot
    discord_success = await run_discord_bot()
    
    # Start Telegram bot
    telegram_success = await run_telegram_bot()
    
    if not discord_success and not telegram_success:
        logger.error("No bots were started successfully")
        return False
    
    # Wait indefinitely until shutdown event
    logger.info("Bots are running indefinitely. Press Ctrl+C to stop.")
    try:
        await shutdown_event.wait()
    except asyncio.CancelledError:
        logger.info("Main task cancelled")
    except Exception as e:
        logger.error(f"Error while waiting: {e}")
        logger.exception("Full traceback:")
    
    # Stop bots
    await stop_bots()
    logger.info("All bots stopped")
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nBots stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error running bots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)