"""
Test the complete Snake Agent autonomous implementation workflow
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


async def test_autonomous_implementation():
    print("Testing Snake Agent autonomous implementation...")
    
    # Set environment variable to enable autonomous implementation
    os.environ['SNAKE_AUTONOMOUS_IMPLEMENTATION_ENABLED'] = 'true'
    
    agi_system = MinimalAGISystem()
    snake_agent = EnhancedSnakeAgent(agi_system)
    
    try:
        # Initialize the agent
        await snake_agent.initialize()
        
        # Get current status
        status = await snake_agent.get_status()
        print('Snake Agent Status:', status)
        
        print('\nSnake Agent with autonomous implementation initialized successfully!')
        
        # Let's trigger a simple analysis task to see if the implementation system works
        from core.snake_data_models import AnalysisTask, TaskPriority
        import uuid
        from datetime import datetime
        
        # Find a file to analyze (let's use a simple file for testing)
        test_file = "core/snake_agent_enhanced.py"
        if os.path.exists(test_file):
            print(f"\nAnalyzing {test_file} for potential improvements...")
            
            analysis_task = AnalysisTask(
                task_id=f"analysis_{uuid.uuid4().hex[:8]}",
                file_path=test_file,
                analysis_type="performance_analysis",
                priority=TaskPriority.MEDIUM,
                created_at=datetime.now(),
                change_context={
                    "reason": "Testing autonomous implementation workflow"
                }
            )
            
            # Process the analysis task - this should trigger potential implementations
            snake_agent._process_analysis_task(analysis_task)
            
            print("Analysis task processed.")
        
        # Check status again after analysis
        status = await snake_agent.get_status()
        print('\nSnake Agent Status after analysis:', status)
        
        print(f"\nAutonomous implementation system is available: {snake_agent.implementer is not None}")
        if snake_agent.implementer:
            impl_status = snake_agent.implementer.get_implementation_status()
            print(f"Implementation status: {impl_status}")
        
    finally:
        # Stop the agent after testing
        await snake_agent.stop()
        print("\nSnake Agent stopped.")


if __name__ == "__main__":
    asyncio.run(test_autonomous_implementation())