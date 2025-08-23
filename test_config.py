import json
import os

def test_config_loading():
    """Test if the configuration file can be loaded successfully."""
    config_path = os.path.join('modules', 'conversational_ai', 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print('Config loaded successfully')
        print(f'Discord enabled: {config["platforms"]["discord"]["enabled"]}')
        print(f'Telegram enabled: {config["platforms"]["telegram"]["enabled"]}')
        print(f'Discord token present: {bool(config.get("discord_token"))}')
        print(f'Telegram token present: {bool(config.get("telegram_token"))}')
        return True
    except Exception as e:
        print(f'Error loading config: {e}')
        return False

if __name__ == "__main__":
    test_config_loading()