import asyncio
import pytest
from core.snake_agent_enhanced import EnhancedSnakeAgent
from core.system import AGISystem
from database.engine import get_engine


@pytest.mark.asyncio
async def test_snake_agent_initialization():
    """Test that Snake Agent initializes correctly"""
    # Create a real AGI system for the Snake Agent to work with
    engine = get_engine()
    agi_system = AGISystem(engine=engine)
    
    snake_agent = EnhancedSnakeAgent(agi_system)
    
    # Verify the agent was created
    assert snake_agent is not None
    assert hasattr(snake_agent, 'agi_system')
    

@pytest.mark.asyncio
async def test_snake_agent_status():
    """Test getting Snake Agent status"""
    # Create a real AGI system for the Snake Agent to work with
    engine = get_engine()
    agi_system = AGISystem(engine=engine)
    snake_agent = EnhancedSnakeAgent(agi_system)
    
    # Initialize the agent
    await snake_agent.initialize()
    
    # Get current status
    status = await snake_agent.get_status()
    
    # Verify status structure
    assert status is not None
    assert isinstance(status, dict)
    

@pytest.mark.asyncio
async def test_snake_agent_lifecycle():
    """Test complete Snake Agent lifecycle"""
    # Create a real AGI system for the Snake Agent to work with
    engine = get_engine()
    agi_system = AGISystem(engine=engine)
    snake_agent = EnhancedSnakeAgent(agi_system)
    
    # Initialize the agent
    await snake_agent.initialize()
    
    # Get current status
    status = await snake_agent.get_status()
    assert status is not None
    
    # Stop the agent
    await snake_agent.stop()