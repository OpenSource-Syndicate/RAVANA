#!/usr/bin/env python3
"""
Simple test script to verify that the bots can connect with the provided tokens
"""
import asyncio
import sys
import os
import json

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_discord_bot():
    """Test Discord bot connection."""
    print("Testing Discord bot connection...")

    try:
        # Load config
        config_path = os.path.join(os.path.dirname(
            __file__), 'modules', 'conversational_ai', 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        token = config.get("discord_token")
        if not token:
            print("Discord token not found in config")
            return

        # Import discord
        import discord
        from discord.ext import commands

        # Create a simple bot
        intents = discord.Intents.default()
        intents.message_content = True
        bot = commands.Bot(command_prefix='!', intents=intents)

        @bot.event
        async def on_ready():
            print(f'Discord bot logged in as {bot.user}')
            await bot.close()

        # Try to connect
        print("Attempting to connect to Discord...")
        await bot.start(token)
        print("Discord bot connection test completed")

    except Exception as e:
        print(f"Error testing Discord bot: {e}")
        import traceback
        traceback.print_exc()


async def test_telegram_bot():
    """Test Telegram bot connection."""
    print("Testing Telegram bot connection...")

    try:
        # Load config
        config_path = os.path.join(os.path.dirname(
            __file__), 'modules', 'conversational_ai', 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        token = config.get("telegram_token")
        if not token:
            print("Telegram token not found in config")
            return

        # Import telegram
        from telegram.ext import Application

        # Create a simple application
        application = Application.builder().token(token).build()

        # Try to initialize
        print("Attempting to initialize Telegram bot...")
        await application.initialize()
        print("Telegram bot initialized successfully")

        # Try to start
        print("Attempting to start Telegram bot...")
        await application.start()
        print("Telegram bot started successfully")

        # Try to stop
        await application.stop()
        await application.shutdown()
        print("Telegram bot connection test completed")

    except Exception as e:
        print(f"Error testing Telegram bot: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    print("Starting bot connection tests...")

    # Test Discord
    await test_discord_bot()

    # Test Telegram
    await test_telegram_bot()

    print("Bot connection tests completed.")

if __name__ == "__main__":
    asyncio.run(main())
