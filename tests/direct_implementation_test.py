"""
Direct test of the implementation system to make an actual code change
"""

import asyncio
import os
import tempfile
from pathlib import Path

# Add the project root to the path so imports work
project_root = Path(__file__).parent
import sys
sys.path.insert(0, str(project_root))

from core.snake_agent_implementer import SnakeAgentImplementer
from core.testing_environment import TestRunner
from core.validation_framework import TestValidator


async def direct_implementation_test():
    print("Testing direct implementation with Snake Agent Implementer...")
    
    # Create a temporary directory for our test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file with clear optimization opportunities
        test_file_path = Path(temp_dir) / "optimization_test.py"
        test_content = '''
"""
Test file for optimization
"""

def inefficient_function():
    """An inefficient function that builds a list with append()."""
    result = []
    for i in range(100):
        result.append(i * 2)
    return result

def inefficient_string_concat():
    """Inefficient string concatenation in a loop."""
    text = ""
    for i in range(50):
        text = text + "Item " + str(i) + "\\n"
    return text

if __name__ == "__main__":
    print("Testing...")
    result1 = inefficient_function()
    result2 = inefficient_string_concat()
    print(f"Results: {len(result1)}, {len(result2)}")
'''
        
        with open(test_file_path, 'w') as f:
            f.write(test_content)
        
        print(f"Created test file: {test_file_path}")
        print("Original content:")
        with open(test_file_path, 'r') as f:
            print(f.read())
        
        # Initialize the implementer
        implementer = SnakeAgentImplementer(project_root=temp_dir)
        
        if implementer.autonomous_implementer:
            print("\\nImplementer initialized successfully")
            
            # Create a specific change to implement
            change_spec = {
                'type': 'performance_optimization',
                'subtype': 'performance',
                'description': 'Replace inefficient list building with list comprehension',
                'recommendation': 'Use list comprehension instead of appending in a loop',
                'severity': 'low',
                'scope': 'function',
                'line_start': 7,  # Line where the inefficient function starts
                'line_end': 11    # Line where the inefficient function ends
            }
            
            print(f"\\nAttempting to implement change: {change_spec['description']}")
            
            # Try to implement the change
            success = await implementer.implement_targeted_change(
                file_path=str(test_file_path),
                change_description=change_spec['description'],
                change_spec=change_spec
            )
            
            print(f"Implementation success: {success}")
            
            # Show the content after implementation attempt
            print("\\nContent after implementation attempt:")
            with open(test_file_path, 'r') as f:
                new_content = f.read()
                print(new_content)
                
            # Check if the content actually changed
            if new_content != test_content:
                print("\\nSUCCESS: Content was modified!")
                print("Changes detected:")
                original_lines = test_content.split('\\n')
                new_lines = new_content.split('\\n')
                
                for i, (orig, new) in enumerate(zip(original_lines, new_lines)):
                    if orig != new:
                        print(f"  Line {i+1}:")
                        print(f"    Original: {repr(orig)}")
                        print(f"    Modified:  {repr(new)}")
            else:
                print("\\nNo changes were made to the file.")
        else:
            print("\\nFailed to initialize implementer")


if __name__ == "__main__":
    asyncio.run(direct_implementation_test())