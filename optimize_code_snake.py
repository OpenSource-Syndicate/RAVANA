import asyncio
from core.snake_agent_enhanced import EnhancedSnakeAgent
from database.engine import get_engine
from core.system import AGISystem
import tempfile
import os
from pathlib import Path


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


async def run_snake_analysis_and_improvement():
    print("Initializing Snake Agent...")
    agi_system = MinimalAGISystem()
    snake_agent = EnhancedSnakeAgent(agi_system)
    
    try:
        # Initialize the agent
        await snake_agent.initialize()
        
        # Get current status
        status = await snake_agent.get_status()
        print('Snake Agent Status:', status)
        
        print('\nStarting analysis of the codebase...')
        
        # Manually trigger analysis of the snake_agent_enhanced.py file
        file_path = "core/snake_agent_enhanced.py"
        if os.path.exists(file_path):
            print(f"Analyzing {file_path}...")
            
            # Create a file change event to trigger analysis
            from core.snake_data_models import FileChangeEvent
            import uuid
            from datetime import datetime
            file_event = FileChangeEvent(
                event_id=f"event_{uuid.uuid4().hex[:8]}",
                event_type="modified",
                file_path=file_path,
                absolute_path=os.path.abspath(file_path),
                timestamp=datetime.now(),
                old_hash="",
                file_hash="temp"  # This would be calculated in real usage
            )
            
            # Trigger file change processing
            snake_agent._handle_file_change(file_event)
            
            # Let's also manually create an analysis task for a specific function
            from core.snake_data_models import AnalysisTask, TaskPriority
            import uuid
            from datetime import datetime
            
            analysis_task = AnalysisTask(
                task_id=f"analysis_{uuid.uuid4().hex[:8]}",
                file_path=file_path,
                analysis_type="performance_analysis",
                priority=TaskPriority.HIGH,
                created_at=datetime.now(),
                change_context={
                    "reason": "Looking for performance optimizations"
                }
            )
            
            # Process the analysis task
            snake_agent._process_analysis_task(analysis_task)
            
            print("Analysis completed.")
            
            # Manually trigger improvement goal setting
            await snake_agent._set_improvement_goals()
            print("Improvement goals set.")
            
        else:
            print(f"File {file_path} not found")
        
        # Check status again after analysis
        status = await snake_agent.get_status()
        print('\nSnake Agent Status after analysis:', status)
        
    finally:
        # Stop the agent after analysis
        await snake_agent.stop()
        print("\nSnake Agent stopped.")


if __name__ == "__main__":
    asyncio.run(run_snake_analysis_and_improvement())