#!/usr/bin/env python3
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.conversational_ai.main import ConversationalAI
    print("Import successful")

    # Try to initialize
    conversational_ai = ConversationalAI()
    print("Initialization successful")

    # Try to access config
    config = conversational_ai.config
    print(f"Config loaded: {config}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
