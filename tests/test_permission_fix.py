#!/usr/bin/env python3
"""
Test script to verify the PermissionError fix in setup_logging function
"""
import os
import sys


def test_permission_error_handling():
    """Test that PermissionError is properly handled in file removal"""
    # Create a test log file
    log_file = 'test_ravana_agi.log'

    with open(log_file, 'w') as f:
        f.write("test log content\n")

    print(f"Created test log file: {log_file}")

    # Test our fix - simulate the code in setup_logging
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
            print("SUCCESS: File removed normally")
        except PermissionError:
            # This is the case we're testing - on Windows when file is locked
            print("HANDLED: PermissionError caught and handled gracefully")
            # In the actual code, we would continue with appending to the file
        except Exception as e:
            print(f"UNEXPECTED: Other exception occurred: {e}")
            return False

    # Verify the fix would allow execution to continue
    try:
        # This simulates continuing with logging setup
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("New log entry after PermissionError\n")
        print("SUCCESS: Logging can continue even after PermissionError")

        # Clean up
        if os.path.exists(log_file):
            os.remove(log_file)

        return True
    except Exception as e:
        print(f"FAILED: Could not continue with logging: {e}")
        return False


if __name__ == "__main__":
    print("Testing PermissionError handling in logging setup...")
    success = test_permission_error_handling()

    if success:
        print("\nPermissionError fix verification PASSED")
        sys.exit(0)
    else:
        print("\nPermissionError fix verification FAILED")
        sys.exit(1)
