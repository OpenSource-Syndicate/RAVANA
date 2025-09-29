#!/usr/bin/env python3
"""
Test script to verify Telegram bot connection and functionality
"""
import asyncio
import sys
import os
import logging

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_telegram_bot():
    """Test Telegram bot connection and functionality."""
    logger.info("Testing Telegram bot connection and functionality...")

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
        bot = TelegramBot(
            token=token,
            command_prefix=ai.config["platforms"]["telegram"]["command_prefix"],
            conversational_ai=ai
        )

        logger.info("Telegram bot instance created")

        # Start bot
        logger.info("Starting Telegram bot...")
        await bot.start()
        logger.info("Telegram bot started successfully")

        # Wait for a bit to see if it connects
        logger.info("Waiting for Telegram bot to connect...")
        for i in range(10):
            await asyncio.sleep(1)
            if hasattr(bot, 'connected') and bot.connected:
                logger.info("Telegram bot is connected!")
                break
            logger.debug(f"Waiting for Telegram bot to connect... ({i+1}/10)")

        # Stop the bot
        logger.info("Stopping Telegram bot...")
        await bot.stop()
        logger.info("Telegram bot stopped successfully")
        return True

    except Exception as e:
        logger.error(f"Error testing Telegram bot: {e}")
        logger.exception("Full traceback:")
        return False


async def main():
    """Main test function."""
    logger.info("Starting Telegram bot test...")

    # Test Telegram bot
    success = await test_telegram_bot()
    logger.info(
        f"Telegram bot test result: {'SUCCESS' if success else 'FAILED'}")

    logger.info("Telegram bot test completed.")
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error running Telegram bot test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
