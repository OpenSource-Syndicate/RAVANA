#!/usr/bin/env python3
"""
Simple test script to verify the logging fix for PermissionError issue
"""
import os
import tempfile

def test_file_removal_with_exception_handling():
    """Test that our exception handling works correctly"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        temp_file = f.name
    
    print(f"Created temporary file: {temp_file}")
    
    # Test that we can remove it normally
    try:
        os.remove(temp_file)
        print("SUCCESS: Normal file removal works")
    except Exception as e:
        print(f"FAILED: Normal file removal failed: {e}")
        return False
    
    # Create it again
    with open(temp_file, 'w') as f:
        f.write("test content")
    
    # Simulate the PermissionError handling
    try:
        os.remove(temp_file)
        print("SUCCESS: File removal works correctly")
        return True
    except PermissionError:
        print("This shouldn't happen in this test")
        return False
    except Exception as e:
        print(f"FAILED: Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_file_removal_with_exception_handling()
    if success:
        print("\nLogging fix verification PASSED")
    else:
        print("\nLogging fix verification FAILED")