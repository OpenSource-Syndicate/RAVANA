#!/usr/bin/env python3
"""
Test script to verify the original error is fixed.
This script simulates the exact scenario that was causing the TypeError.
"""

import sys
import os
import asyncio

# Add the project root to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class MockConversationalAI:
    """Mock ConversationalAI class for testing."""
    def __init__(self):
        pass

def test_telegram_bot_original_error():
    """Test that the original Telegram bot error is fixed."""
    print("Testing original Telegram bot error scenario...")
    
    try:
        from modules.conversational_ai.bots.telegram_bot import TelegramBot
        
        # This is the exact pattern that was causing the error:
        # self.telegram_bot = TelegramBot(...)
        bot_instance = TelegramBot(
            token="test_token",
            command_prefix="/",
            conversational_ai=MockConversationalAI()
        )
        
        # If we get here without a TypeError, the fix is working
        print("✓ TelegramBot instantiation successful - original error is fixed!")
        return True
        
    except TypeError as e:
        if "__init__() should return None, not 'TelegramBot'" in str(e):
            print(f"✗ Original error still exists: {e}")
            return False
        else:
            print(f"✗ Different TypeError occurred: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_discord_bot_original_error():
    """Test that the original Discord bot error is fixed."""
    print("Testing original Discord bot error scenario...")
    
    try:
        from modules.conversational_ai.bots.discord_bot import DiscordBot
        
        # This is the exact pattern that was causing the error:
        # self.discord_bot = DiscordBot(...)
        bot_instance = DiscordBot(
            token="test_token",
            command_prefix="!",
            conversational_ai=MockConversationalAI()
        )
        
        # If we get here without a TypeError, the fix is working
        print("✓ DiscordBot instantiation successful - original error is fixed!")
        return True
        
    except TypeError as e:
        if "__init__() should return None, not 'DiscordBot'" in str(e):
            print(f"✗ Original error still exists: {e}")
            return False
        else:
            print(f"✗ Different TypeError occurred: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def main():
    """Main test function."""
    print("Running original error fix verification tests...\n")
    
    telegram_success = test_telegram_bot_original_error()
    discord_success = test_discord_bot_original_error()
    
    print("\n" + "="*50)
    if telegram_success and discord_success:
        print("✓ Original error fix verified! The TypeError should no longer occur.")
        return 0
    else:
        print("✗ Original error fix verification failed.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)