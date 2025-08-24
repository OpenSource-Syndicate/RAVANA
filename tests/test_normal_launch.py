#!/usr/bin/env python3
"""
Test script to verify the normal launcher functionality
"""
import subprocess
import sys
import os
import time

def test_normal_launch():
    """Test the normal launcher functionality."""
    print("Testing normal launcher functionality...")
    
    # Change to the project directory
    project_dir = r"c:\Users\ASUS\Documents\GitHub\RAVANA"
    os.chdir(project_dir)
    
    # Run the launcher without --verify-bots flag for a short time
    try:
        process = subprocess.Popen([
            sys.executable, 
            "launch_conversational_ai.py"
        ], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
        )
        
        # Give it a few seconds to start
        time.sleep(5)
        
        # Check if the process is still running
        if process.poll() is None:
            # Process is still running, terminate it
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            print("SUCCESS: Normal launcher started successfully and was terminated")
        else:
            # Process has already exited
            stdout, stderr = process.communicate()
            print("Process exited with code:", process.returncode)
            print("STDOUT:")
            print(stdout)
            print("STDERR:")
            print(stderr)
            
    except Exception as e:
        print(f"ERROR: Failed to run normal launcher: {e}")

if __name__ == "__main__":
    test_normal_launch()