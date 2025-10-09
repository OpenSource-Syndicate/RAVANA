"""
Run Snake Agent with autonomous implementation enabled to observe actual code changes
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


async def run_snake_agent_with_implementation():
    print("Running Snake Agent with autonomous implementation enabled...")
    
    # Enable autonomous implementation
    os.environ['SNAKE_AUTONOMOUS_IMPLEMENTATION_ENABLED'] = 'true'
    
    # Create a simple test file for the Snake Agent to analyze and potentially improve
    test_file_path = "test_optimization_target.py"
    with open(test_file_path, 'w') as f:
        f.write("""
\"\"\"
Test file for Snake Agent optimization
\"\"\"

def inefficient_function():
    \"\"\"An inefficient function that could be optimized.\"\"\"
    result = []
    for i in range(1000):
        result.append(i * 2)
    return result

def another_inefficient_function():
    \"\"\"Another function with inefficient string concatenation.\"\"\"
    text = ""
    for i in range(100):
        text = text + "Item " + str(i) + "\\n"
    return text

if __name__ == "__main__":
    print("Running test functions...")
    result1 = inefficient_function()
    result2 = another_inefficient_function()
    print(f"Result 1 length: {len(result1)}")
    print(f"Result 2 length: {len(result2)}")
""")
    
    print(f"Created test file: {test_file_path}")
    
    agi_system = MinimalAGISystem()
    snake_agent = EnhancedSnakeAgent(agi_system)
    
    try:
        # Initialize the agent
        await snake_agent.initialize()
        print("Snake Agent initialized successfully")
        
        # Get initial status
        status = await snake_agent.get_status()
        print('Initial Status:')
        print(f"  Running: {status.get('running', 'N/A')}")
        print(f"  Initialized: {status.get('initialized', 'N/A')}")
        print(f"  Files analyzed: {status.get('metrics', {}).get('files_analyzed', 0)}")
        print(f"  Improvements applied: {status.get('metrics', {}).get('improvements_applied', 0)}")
        
        # Create an analysis task for our test file
        from core.snake_data_models import AnalysisTask, TaskPriority
        import uuid
        from datetime import datetime
        
        analysis_task = AnalysisTask(
            task_id=f"analysis_{uuid.uuid4().hex[:8]}",
            file_path=test_file_path,
            analysis_type="performance_analysis",
            priority=TaskPriority.HIGH,
            created_at=datetime.now(),
            change_context={
                "reason": "Testing autonomous implementation workflow"
            }
        )
        
        print(f"\nProcessing analysis task for {test_file_path}...")
        snake_agent._process_analysis_task(analysis_task)
        
        # Give it some time to process
        await asyncio.sleep(5)
        
        # Check status after processing
        status = await snake_agent.get_status()
        print('\nStatus after analysis:')
        print(f"  Files analyzed: {status.get('metrics', {}).get('files_analyzed', 0)}")
        print(f"  Improvements applied: {status.get('metrics', {}).get('improvements_applied', 0)}")
        print(f"  Task queue size: {status.get('process_queues', {}).get('task_queue', 0)}")
        
        # Wait a bit more to see if any implementations happen
        print("\nWaiting for potential implementations...")
        for i in range(10):
            await asyncio.sleep(2)
            status = await snake_agent.get_status()
            improvements = status.get('metrics', {}).get('improvements_applied', 0)
            if improvements > 0:
                print(f"SUCCESS: {improvements} improvements applied!")
                break
            else:
                print(f"  Still waiting... ({i+1}/10) No improvements yet.")
        
        # Final check
        status = await snake_agent.get_status()
        print(f"\nFinal status:")
        print(f"  Files analyzed: {status.get('metrics', {}).get('files_analyzed', 0)}")
        print(f"  Improvements applied: {status.get('metrics', {}).get('improvements_applied', 0)}")
        
        # Show the content of the test file after processing
        print(f"\nContent of {test_file_path} after processing:")
        with open(test_file_path, 'r') as f:
            content = f.read()
            print(content)
        
    except Exception as e:
        print(f"Error running Snake Agent: {e}")
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
    asyncio.run(run_snake_agent_with_implementation())