#!/usr/bin/env python3
"""
Launcher script for the Conversational AI Module
"""
from modules.conversational_ai.main import ConversationalAI
import asyncio
import sys
import os
import logging
import argparse

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def verify_discord_bot(ai):
    """Verify Discord bot connectivity."""
    logger.info("Testing Discord bot connection...")

    try:
        if not ai.config.get("platforms", {}).get("discord", {}).get("enabled", False):
            logger.info("Discord bot is not enabled")
            return False, "Discord bot is not enabled"

        token = ai.config.get("discord_token")
        if not token:
            logger.error("Discord token not found")
            return False, "Discord token not found"

        # Import and create bot
        from modules.conversational_ai.bots.discord_bot import DiscordBot
        bot = DiscordBot(
            token=token,
            command_prefix=ai.config["platforms"]["discord"]["command_prefix"],
            conversational_ai=ai
        )

        logger.info("Discord bot instance created")

        # Start bot in a task
        bot_task = asyncio.create_task(bot.start())
        logger.info("Discord bot task started")

        # Wait a bit to see if it connects
        for i in range(10):
            await asyncio.sleep(1)
            if hasattr(bot, 'connected') and bot.connected:
                logger.info("Discord bot is connected!")
                # Stop the bot
                await bot.stop()
                return True, "Connected successfully"
            logger.debug(f"Waiting for Discord bot to connect... ({i+1}/10)")

        logger.warning("Discord bot did not connect in time")
        await bot.stop()
        return False, "Discord bot did not connect in time"

    except Exception as e:
        logger.error(f"Error testing Discord bot: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


async def verify_telegram_bot(ai):
    """Verify Telegram bot connectivity."""
    logger.info("Testing Telegram bot connection...")

    try:
        if not ai.config.get("platforms", {}).get("telegram", {}).get("enabled", False):
            logger.info("Telegram bot is not enabled")
            return False, "Telegram bot is not enabled"

        token = ai.config.get("telegram_token")
        if not token:
            logger.error("Telegram token not found")
            return False, "Telegram token not found"

        # Import and create bot
        from modules.conversational_ai.bots.telegram_bot import TelegramBot
        bot = await TelegramBot.get_instance(
            token=token,
            command_prefix=ai.config["platforms"]["telegram"]["command_prefix"],
            conversational_ai=ai
        )

        logger.info("Telegram bot instance created")

        # For Telegram, we need to handle the start method differently since it blocks
        # We'll run it in a task and check the connected status
        bot_task = asyncio.create_task(bot.start())
        logger.info("Telegram bot task started")

        # Wait a bit to see if it connects
        for i in range(10):
            await asyncio.sleep(1)
            if hasattr(bot, 'connected') and bot.connected:
                logger.info("Telegram bot is connected!")
                # Stop the bot
                await bot.stop()
                # Cancel the task since it has its own loop
                bot_task.cancel()
                try:
                    await bot_task
                except asyncio.CancelledError:
                    pass
                return True, "Connected successfully"
            logger.debug(f"Waiting for Telegram bot to connect... ({i+1}/10)")

        logger.warning("Telegram bot did not connect in time")
        await bot.stop()
        # Cancel the task since it has its own loop
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            pass
        return False, "Telegram bot did not connect in time"

    except Exception as e:
        logger.error(f"Error testing Telegram bot: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)


async def verify_bots():
    """Verify bot connectivity for both Discord and Telegram."""
    logger.info("Starting bot verification...")

    results = {
        'discord': {'enabled': False, 'connected': False, 'message': ''},
        'telegram': {'enabled': False, 'connected': False, 'message': ''}
    }

    try:
        # Load config
        ai = ConversationalAI()

        # Test Discord
        discord_enabled = ai.config.get("platforms", {}).get(
            "discord", {}).get("enabled", False)
        results['discord']['enabled'] = discord_enabled

        if discord_enabled:
            connected, message = await verify_discord_bot(ai)
            results['discord']['connected'] = connected
            results['discord']['message'] = message

        # Test Telegram
        telegram_enabled = ai.config.get("platforms", {}).get(
            "telegram", {}).get("enabled", False)
        results['telegram']['enabled'] = telegram_enabled

        if telegram_enabled:
            connected, message = await verify_telegram_bot(ai)
            results['telegram']['connected'] = connected
            results['telegram']['message'] = message

        # Print results
        print("\n=== Bot Verification Results ===")
        for platform in ['discord', 'telegram']:
            info = results[platform]
            if info['enabled']:
                status = "CONNECTED" if info['connected'] else "FAILED"
                print(f"{platform.capitalize()}: {status}")
                if info['message']:
                    print(f"  Message: {info['message']}")
            else:
                print(f"{platform.capitalize()}: DISABLED")

        # Return success if all enabled bots are connected
        success = True
        for platform in ['discord', 'telegram']:
            info = results[platform]
            if info['enabled'] and not info['connected']:
                success = False

        return success

    except Exception as e:
        logger.error(f"Error during bot verification: {e}")
        logger.exception("Full traceback:")
        return False


async def main(verify_bots_flag=False):
    """Main entry point for the Conversational AI Module."""
    if verify_bots_flag:
        # Run bot verification and exit
        success = await verify_bots()
        return 0 if success else 1

    logger.info("Starting Conversational AI Module...")

    try:
        # Initialize the conversational AI
        conversational_ai = ConversationalAI()
        logger.info("Conversational AI instance created")

        # Check configuration
        logger.info("Configuration:")
        logger.info(
            f"  Discord enabled: {conversational_ai.config.get('platforms', {}).get('discord', {}).get('enabled', False)}")
        logger.info(
            f"  Telegram enabled: {conversational_ai.config.get('platforms', {}).get('telegram', {}).get('enabled', False)}")

        # Start the module in standalone mode
        logger.info("Starting Conversational AI in standalone mode...")
        await conversational_ai.start(standalone=True)

    except Exception as e:
        logger.error(f"Error running Conversational AI Module: {e}")
        logger.exception("Full traceback:")
        return 1

    logger.info("Conversational AI Module stopped.")
    return 0

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Launch the Conversational AI Module")
    parser.add_argument("--verify-bots", action="store_true",
                        help="Verify bot connectivity and exit")

    args = parser.parse_args()

    try:
        result = asyncio.run(main(verify_bots_flag=args.verify_bots))
        sys.exit(result)
    except KeyboardInterrupt:
        print("\nConversational AI Module stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error running Conversational AI Module: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
