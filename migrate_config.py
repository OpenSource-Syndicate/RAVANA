#!/usr/bin/env python3
"""
Configuration Migration Script for RAVANA AGI

This script helps migrate from the complex old configuration system
to the simplified configuration system.
"""

import json
import os
from pathlib import Path


def migrate_from_old_to_new():
    """
    Migrate configuration from old complex system to simplified system.
    """
    print("Starting RAVANA AGI configuration migration...")
    
    # Try to load the old configuration files
    old_core_config_path = Path("core/config.json")
    old_python_config_path = Path("core/config.py")
    
    migrated_config = {
        "system": {
            "name": "RAVANA",
            "database_url": "sqlite:///ravana_agi.db",
            "log_level": "INFO"
        },
        "operational": {
            "curiosity_chance": 0.4,
            "reflection_chance": 0.15,
            "loop_sleep_duration": 7,
            "max_iterations": 15
        },
        "models": {
            "primary": {
                "provider": "ollama",
                "model_name": "llama3.1:8b",
                "base_url": "http://localhost:11434",
                "temperature": 0.7,
                "max_tokens": 2048,
                "timeout": 300
            }
        },
        "api_keys": {
            "electronhub": "ek-sVvxMYfdFQ0Kl6Aj2tmV7b8n5v0Y0sDHVsOUZWyx2vbs0AbuAc",
            "zuki": "985160dfa1fd499fd12af708d16552e37a8c6f77cbfb50ae400e3ff33fbd791bc7b3b82625379a1f5ca7568f1ee04eb81a0f8c06f0ba6c276d3dddfe13e9c18d",
            "gemini": "AIzaSyBW-aVU-x7JCjBJVVKjPGUacups0-GBHvQ"
        },
        "memory": {
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_use_cuda": True
        },
        "agents": {
            "snake_agent": {
                "enabled": True,
                "interval": 180
            }
        },
        "services": {
            "blog": {
                "enabled": True,
                "api_url": "https://ravana-blog.netlify.app/api/publish"
            },
            "conversational_ai": {
                "enabled": True
            }
        },
        "scheduling": {
            "data_collection_interval": 1800,
            "event_detection_interval": 300,
            "knowledge_compression_interval": 7200
        },
        "persona": {
            "name": "Ravana",
            "creativity": 0.7
        },
        "shutdown": {
            "timeout": 30,
            "graceful_shutdown_enabled": True
        }
    }
    
    # If old config.json exists, try to extract key values
    if old_core_config_path.exists():
        print(f"Found old configuration file: {old_core_config_path}")
        try:
            with open(old_core_config_path, 'r', encoding='utf-8') as f:
                old_config = json.load(f)
            
            # Extract API keys from old config
            if 'electronhub' in old_config and 'api_key' in old_config['electronhub']:
                migrated_config['api_keys']['electronhub'] = old_config['electronhub']['api_key']
            if 'zuki' in old_config and 'api_key' in old_config['zuki']:
                migrated_config['api_keys']['zuki'] = old_config['zuki']['api_key']
            if 'gemini' in old_config and 'api_keys' in old_config['gemini']:
                if isinstance(old_config['gemini']['api_keys'], list) and len(old_config['gemini']['api_keys']) > 0:
                    migrated_config['api_keys']['gemini'] = old_config['gemini']['api_keys'][0]['key']
            
            print("Successfully extracted API keys from old configuration")
        except Exception as e:
            print(f"Warning: Could not read old config.json: {e}")
    
    # Write the new simplified configuration
    new_config_path = Path("simple_config.json")
    with open(new_config_path, 'w', encoding='utf-8') as f:
        json.dump(migrated_config, f, indent=2)
    
    print(f"Migration completed! New configuration saved to {new_config_path}")
    print("\nTo use the new simplified configuration:")
    print("1. Update imports to use core.simple_config instead of core.config")
    print("2. Use the new configuration structure as described in SIMPLIFIED_CONFIG_GUIDE.md")
    print("3. The new config supports all the same environment variables as before")


if __name__ == "__main__":
    migrate_from_old_to_new()