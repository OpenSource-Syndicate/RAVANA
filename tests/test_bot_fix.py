#!/usr/bin/env python3
"""
Test script to verify the bot initialization fix.
This script tests that the Telegram and Discord bots can be instantiated without errors.
"""

import sys
import os
import asyncio

# Add the project root to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class MockConversationalAI:
    """Mock ConversationalAI class for testing."""

    def __init__(self):
        self.user_profile_manager = MockUserProfileManager()

    def process_user_message_async(self, platform, user_id, message):
        """Mock async message processing."""
        async def mock_process():
            return f"Mock response to: {message}"
        return mock_process()

    def handle_task_from_user(self, user_id, task_description):
        """Mock task handling."""


class MockUserProfileManager:
    """Mock user profile manager for testing."""

    def update_username(self, user_id, username):
        """Mock username update."""


async def test_telegram_bot():
    """Test Telegram bot initialization."""
    print("Testing Telegram bot initialization...")

    try:
        from modules.conversational_ai.bots.telegram_bot import TelegramBot

        # Create first instance using factory method
        bot1 = await TelegramBot.get_instance(
            token="test_token_1",
            command_prefix="/",
            conversational_ai=MockConversationalAI()
        )
        print("✓ First TelegramBot instance created successfully")

        # Create second instance with same token (should return existing)
        bot2 = await TelegramBot.get_instance(
            token="test_token_1",  # Same token
            command_prefix="!",
            conversational_ai=MockConversationalAI()
        )
        print("✓ Second TelegramBot instance created successfully")

        # Check if they're the same instance (same token should return same instance)
        if bot1 is bot2:
            print("✓ Both instances are the same object (singleton pattern working)")
        else:
            print("⚠ Instances are different objects")

        # Create third instance with different token (should be different instance)
        bot3 = await TelegramBot.get_instance(
            token="test_token_2",  # Different token
            command_prefix="/",
            conversational_ai=MockConversationalAI()
        )
        print("✓ Third TelegramBot instance created successfully")

        # Check that different tokens create different instances
        if bot1 is not bot3:
            print("✓ Different tokens create different instances")
        else:
            print("⚠ Different tokens should create different instances")

        return True
    except Exception as e:
        print(f"✗ Error testing Telegram bot: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_discord_bot():
    """Test Discord bot initialization."""
    print("\nTesting Discord bot initialization...")

    try:
        from modules.conversational_ai.bots.discord_bot import DiscordBot

        # Create first instance using factory method
        bot1 = DiscordBot.get_instance(
            token="test_token_1",
            command_prefix="!",
            conversational_ai=MockConversationalAI()
        )
        print("✓ First DiscordBot instance created successfully")

        # Create second instance (should return existing)
        bot2 = DiscordBot.get_instance(
            token="test_token_2",  # Different token
            command_prefix="/",
            conversational_ai=MockConversationalAI()
        )
        print("✓ Second DiscordBot instance created successfully")

        # Check if they're the same instance
        if bot1 is bot2:
            print("✓ Both instances are the same object (singleton pattern working)")
        else:
            print("⚠ Instances are different objects")

        return True
    except Exception as e:
        print(f"✗ Error testing Discord bot: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("Running bot initialization tests...\n")

    telegram_success = await test_telegram_bot()
    discord_success = await test_discord_bot()

    print("\n" + "="*50)
    if telegram_success and discord_success:
        print("✓ All tests passed! Bot initialization fix is working.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
