"""
Comprehensive test of the complete Snake Agent implementation system
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


async def comprehensive_implementation_test():
    print("Testing complete Snake Agent implementation system...")
    
    # Create a temporary directory for our test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file with clear optimization opportunities
        test_file_path = Path(temp_dir) / "complete_test_file.py"
        test_content = '''
"""
Complete test file with multiple optimization opportunities
"""

def inefficient_list_building():
    """Build a list inefficiently with append()."""
    result = []
    for i in range(100):
        result.append(i * 2)
    return result

def another_inefficient_function():
    """Another function with inefficient string concatenation."""
    text = ""
    for i in range(50):
        text = text + "Item " + str(i) + "\\n"
    return text

def calculate_squares(numbers):
    """Calculate squares inefficiently."""
    squares = []
    for num in numbers:
        squares.append(num * num)
    return squares

if __name__ == "__main__":
    print("Testing...")
    result1 = inefficient_list_building()
    result2 = another_inefficient_function()
    result3 = calculate_squares([1, 2, 3, 4, 5])
    print(f"Results: {len(result1)}, {len(result2)}, {len(result3)}")
'''
        
        with open(test_file_path, 'w') as f:
            f.write(test_content)
        
        print(f"Created test file: {test_file_path}")
        print("Original content:")
        with open(test_file_path, 'r') as f:
            original_content = f.read()
            print(original_content)
        
        # Initialize the implementer
        implementer = SnakeAgentImplementer(project_root=temp_dir)
        
        print("\\nImplementer initialized successfully")
        
        # Test 1: Optimize the inefficient list building function
        print("\\n=== Test 1: List Comprehension Optimization ===")
        list_building_change = {
            'type': 'performance_optimization',
            'subtype': 'list_building',
            'description': 'Replace inefficient list building with list comprehension',
            'recommendation': 'Use list comprehension instead of appending in a loop',
            'severity': 'low',
            'scope': 'function',
            'line_start': 7,  # Line where the inefficient function starts
            'line_end': 11    # Line where the inefficient function ends
        }
        
        success = await implementer.implement_targeted_change(
            file_path=str(test_file_path),
            change_description='Optimize list building function',
            change_spec=list_building_change
        )
        
        print(f"Implementation success: {success}")
        
        # Show the content after implementation attempt
        print("\\nContent after list comprehension optimization:")
        with open(test_file_path, 'r') as f:
            new_content = f.read()
            print(new_content)
            
        # Check if the content actually changed
        if new_content != original_content:
            print("\\nSUCCESS: Content was modified!")
            print("Changes detected:")
        else:
            print("\\nNo changes were made to the file.")
        
        # Test 2: Optimize the squares calculation function
        print("\\n=== Test 2: Squares Calculation Optimization ===")
        squares_change = {
            'type': 'performance_optimization',
            'subtype': 'list_building',
            'description': 'Replace inefficient squares calculation with list comprehension',
            'recommendation': 'Use list comprehension for squares calculation',
            'severity': 'low',
            'scope': 'function',
            'line_start': 17,  # Line where the squares function starts
            'line_end': 21     # Line where the squares function ends
        }
        
        success = await implementer.implement_targeted_change(
            file_path=str(test_file_path),
            change_description='Optimize squares calculation function',
            change_spec=squares_change
        )
        
        print(f"Implementation success: {success}")
        
        # Show the content after second optimization
        print("\\nContent after squares calculation optimization:")
        with open(test_file_path, 'r') as f:
            final_content = f.read()
            print(final_content)
        
        # Final comparison
        print("\\n=== Final Comparison ===")
        if final_content != original_content:
            print("SUCCESS: Multiple optimizations were applied!")
            print("\\nOriginal:")
            print(original_content)
            print("\\nOptimized:")
            print(final_content)
        else:
            print("No optimizations were successfully applied.")


if __name__ == "__main__":
    asyncio.run(comprehensive_implementation_test())