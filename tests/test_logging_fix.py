#!/usr/bin/env python3
"""
Test script to verify the logging fix for PermissionError issue
"""
import os
import sys
import logging

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def simulate_locked_file():
    """Simulate a locked log file by opening it in another thread"""
    log_file = 'ravana_agi.log'

    # Create the file if it doesn't exist
    with open(log_file, 'w') as f:
        f.write("Test log content\n")

    # Open and hold the file to simulate it being locked by another process
    locked_file = open(log_file, 'a')

    # Write something to ensure it's locked
    locked_file.write("Locked by another process\n")
    locked_file.flush()

    # Return the file handle so the caller can close it later
    return locked_file, log_file


def test_logging_with_locked_file():
    """Test logging setup with a locked file"""
    print("Testing logging setup with locked file...")

    # Simulate a locked file
    locked_file, log_file = simulate_locked_file()

    try:
        # Now try to run our modified setup_logging function
        # We'll import and run just the logging setup part
        from main import setup_logging

        # This should not raise a PermissionError anymore
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Test log message - this should work even with locked file")

        print("SUCCESS: Logging setup completed without PermissionError")
        return True

    except PermissionError as e:
        print(f"FAILED: PermissionError still occurs: {e}")
        return False
    except Exception as e:
        print(f"FAILED: Unexpected error: {e}")
        return False
    finally:
        # Clean up - close the locked file
        locked_file.close()
        # Clean up the test log file
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
            except:
                pass


if __name__ == "__main__":
    success = test_logging_with_locked_file()
    if success:
        print("Logging fix verification PASSED")
        sys.exit(0)
    else:
        print("Logging fix verification FAILED")
        sys.exit(1)
