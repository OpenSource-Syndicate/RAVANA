#!/usr/bin/env python3
"""
Test script to verify command-line argument parsing
"""
import subprocess
import sys
import os

def test_cli_args():
    """Test command-line argument parsing."""
    print("Testing command-line argument parsing...")
    
    # Change to the project directory
    project_dir = r"c:\Users\ASUS\Documents\GitHub\RAVANA"
    os.chdir(project_dir)
    
    # Test without arguments
    try:
        result = subprocess.run([
            sys.executable, 
            "launch_conversational_ai.py", 
            "--help"
        ], 
        capture_output=True, 
        text=True
        )
        
        print("Help output:")
        print(result.stdout)
        
        if "--verify-bots" in result.stdout:
            print("SUCCESS: --verify-bots argument is documented in help")
        else:
            print("FAILURE: --verify-bots argument is not documented in help")
            
    except Exception as e:
        print(f"ERROR: Failed to test CLI arguments: {e}")

if __name__ == "__main__":
    test_cli_args()