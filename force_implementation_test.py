"""
Force Snake Agent to attempt an implementation by directly calling the implementation method
"""

import asyncio
import os
from pathlib import Path

# Add the project root to the path so imports work
project_root = Path(__file__).parent
import sys
sys.path.insert(0, str(project_root))

from core.snake_agent_enhanced import EnhancedSnakeAgent
from core.system import AGISystem
from database.engine import get_engine


# Create a minimal AGI system for the Snake Agent to work with
class MinimalAGISystem:
    def __init__(self):
        self.performance_tracker = None
        self.engine = get_engine()
        
    @property
    def performance_tracker(self):
        # Create a mock performance tracker
        class MockTracker:
            def record_improvement(self, **kwargs): 
                print(f"Recorded improvement: {kwargs}")
            def record_metric(self, **kwargs): 
                print(f"Recorded metric: {kwargs}")
            def increment_improvement_count(self): 
                print("Incremented improvement count")
        return MockTracker()
    
    @performance_tracker.setter
    def performance_tracker(self, value):
        self._performance_tracker = value


async def force_implementation_test():
    print("Testing forced implementation with Snake Agent...")
    
    # Enable autonomous implementation
    os.environ['SNAKE_AUTONOMOUS_IMPLEMENTATION_ENABLED'] = 'true'
    
    # Create a simple test file with clear optimization opportunities
    test_file_path = "simple_test_file.py"
    with open(test_file_path, 'w') as f:
        f.write("""
\"\"\"
Simple test file with clear optimization opportunities
\"\"\"

def inefficient_list_building():
    \"\"\"Inefficient way to build a list - could use list comprehension.\"\"\"
    result = []
    for i in range(100):
        result.append(i * 2)
    return result

def inefficient_string_concat():
    \"\"\"Inefficient string concatenation in a loop.\"\"\"
    text = ""
    for i in range(50):
        text = text + "Item " + str(i) + "\\n"
    return text

if __name__ == "__main__":
    print("Testing...")
    result1 = inefficient_list_building()
    result2 = inefficient_string_concat()
    print(f"Results: {len(result1)}, {len(result2)}")
""")
    
    print(f"Created test file: {test_file_path}")
    
    agi_system = MinimalAGISystem()
    snake_agent = EnhancedSnakeAgent(agi_system)
    
    try:
        # Initialize the agent
        await snake_agent.initialize()
        print("Snake Agent initialized successfully")
        
        # Check if the implementer is available
        if snake_agent.implementer:
            print("Implementer is available, attempting direct implementation...")
            
            # Create a simple optimization opportunity
            opportunity = {
                'type': 'optimization',
                'subtype': 'performance',
                'file_path': test_file_path,
                'description': 'Replace inefficient list building with list comprehension',
                'suggestion': 'Use list comprehension instead of appending in a loop',
                'severity': 'low',
                'scope': 'function',
                'line_num': 7  # Approximate line number
            }
            
            print(f"Attempting to implement opportunity: {opportunity['description']}")
            
            # Try to implement the change directly
            # This bypasses the normal filtering to force an implementation attempt
            await snake_agent._attempt_implementation(opportunity)
            
            # Wait a moment for the implementation to complete
            await asyncio.sleep(3)
            
            # Check the status
            status = await snake_agent.get_status()
            improvements = status.get('metrics', {}).get('improvements_applied', 0)
            print(f"Improvements applied: {improvements}")
            
            # Show the content of the test file after processing
            print(f"\nContent of {test_file_path} after processing:")
            with open(test_file_path, 'r') as f:
                content = f.read()
                print(content)
        else:
            print("Implementer is not available")
            
    except Exception as e:
        print(f"Error testing implementation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop the agent
        await snake_agent.stop()
        print("\nSnake Agent stopped.")
        
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            print(f"Cleaned up test file: {test_file_path}")


if __name__ == "__main__":
    asyncio.run(force_implementation_test())