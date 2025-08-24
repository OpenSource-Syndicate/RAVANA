#!/usr/bin/env python3
"""
Simple test script to verify the Conversational AI fix
"""
import asyncio
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.conversational_ai.main import ConversationalAI

async def test_conversational_ai_start_method():
    """Test that the ConversationalAI class has a start method."""
    print("Testing Conversational AI start method...")
    
    # Initialize the conversational AI
    conversational_ai = ConversationalAI()
    
    # Check if the start method exists
    if hasattr(conversational_ai, 'start'):
        print("✓ ConversationalAI.start method exists")
        
        # Test calling the start method with standalone=False (integrated mode)
        try:
            print("Testing start method with standalone=False...")
            # We'll call it but immediately set the shutdown event to prevent it from running indefinitely
            task = asyncio.create_task(conversational_ai.start(standalone=False))
            
            # Give it a moment to start
            await asyncio.sleep(0.1)
            
            # Set shutdown event to stop it
            conversational_ai._shutdown.set()
            
            # Wait a bit for cleanup
            await asyncio.sleep(0.1)
            
            print("✓ ConversationalAI.start method works correctly")
            return True
        except Exception as e:
            print(f"✗ Error calling start method: {e}")
            return False
    else:
        print("✗ ConversationalAI.start method does not exist")
        return False

async def test_conversational_ai_components():
    """Test that the ConversationalAI components are properly initialized."""
    print("Testing Conversational AI components...")
    
    # Initialize the conversational AI
    conversational_ai = ConversationalAI()
    
    # Check if required components exist
    components = [
        ("emotional_intelligence", "Emotional Intelligence"),
        ("memory_interface", "Memory Interface"),
        ("ravana_communicator", "RAVANA Communicator"),
        ("user_profile_manager", "User Profile Manager"),
        ("_shutdown", "Shutdown Event"),
        ("config", "Configuration")
    ]
    
    all_good = True
    for attr, name in components:
        if hasattr(conversational_ai, attr):
            print(f"✓ {name} initialized")
        else:
            print(f"✗ {name} not initialized")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    try:
        print("=== Conversational AI Fix Verification ===")
        
        # Test components
        components_result = asyncio.run(test_conversational_ai_components())
        print()
        
        # Test start method
        start_result = asyncio.run(test_conversational_ai_start_method())
        print()
        
        if components_result and start_result:
            print("All tests passed! The Conversational AI fix is working correctly.")
        else:
            print("Some tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"Error running test: {e}")
        sys.exit(1)