import asyncio
from core.snake_agent_enhanced import EnhancedSnakeAgent
from database.engine import get_engine


# Create a minimal AGI system for the Snake Agent to work with
class MinimalAGISystem:
    def __init__(self):
        self._performance_tracker = None
        self.engine = get_engine()
        
    @property
    def performance_tracker(self):
        # Create a mock performance tracker
        class MockTracker:
            def record_improvement(self, **kwargs): pass
            def record_metric(self, **kwargs): pass
            def increment_improvement_count(self): pass
        return MockTracker()
    
    @performance_tracker.setter
    def performance_tracker(self, value):
        self._performance_tracker = value

# Initialize the Snake Agent
async def run_snake():
    print("Initializing Snake Agent...")
    agi_system = MinimalAGISystem()
    snake_agent = EnhancedSnakeAgent(agi_system)
    
    # Initialize the agent
    await snake_agent.initialize()
    
    # Get current status
    status = await snake_agent.get_status()
    print('Snake Agent Status:', status)
    
    # Perform analysis on the codebase
    print('\nStarting analysis of the codebase...')
    
    # Stop the agent after analysis
    await snake_agent.stop()
    print("\nSnake Agent stopped.")

# Run the Snake Agent
if __name__ == "__main__":
    asyncio.run(run_snake())