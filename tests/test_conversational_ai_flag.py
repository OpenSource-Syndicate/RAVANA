#!/usr/bin/env python3
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Try to import the ConversationalAI class
    from modules.conversational_ai.main import ConversationalAI
    print("ConversationalAI import successful")
    
    # Try the same import that's used in system.py
    try:
        from modules.conversational_ai.main import ConversationalAI
        CONVERSATIONAL_AI_AVAILABLE = True
        print("CONVERSATIONAL_AI_AVAILABLE = True")
    except ImportError as e:
        CONVERSATIONAL_AI_AVAILABLE = False
        print(f"CONVERSATIONAL_AI_AVAILABLE = False, error: {e}")
        
    print(f"Final value: CONVERSATIONAL_AI_AVAILABLE = {CONVERSATIONAL_AI_AVAILABLE}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()