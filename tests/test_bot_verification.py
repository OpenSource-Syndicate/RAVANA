#!/usr/bin/env python3
"""
Test script to verify the bot verification functionality
"""
import subprocess
import sys
import os

def test_bot_verification():
    """Test the bot verification functionality."""
    print("Testing bot verification functionality...")
    
    # Change to the project directory
    project_dir = r"c:\Users\ASUS\Documents\GitHub\RAVANA"
    os.chdir(project_dir)
    
    # Run the launcher with --verify-bots flag
    try:
        result = subprocess.run([
            sys.executable, 
            "launch_conversational_ai.py", 
            "--verify-bots"
        ], 
        capture_output=True, 
        text=True, 
        timeout=30  # 30 second timeout
        )
        
        print("Return code:", result.returncode)
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        
        if result.returncode == 0:
            print("SUCCESS: Bot verification completed successfully")
        else:
            print("FAILURE: Bot verification failed")
            
    except subprocess.TimeoutExpired:
        print("ERROR: Bot verification timed out")
    except Exception as e:
        print(f"ERROR: Failed to run bot verification: {e}")

if __name__ == "__main__":
    test_bot_verification()