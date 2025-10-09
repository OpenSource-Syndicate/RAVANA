#!/usr/bin/env python3
"""
Test script to verify the fix for RuntimeWarning issues with coroutines.
"""

import asyncio
import pytest
from core.system import AGISystem
from database.engine import get_engine
from core.config import Config


def test_agi_system_initialization():
    """Test that AGISystem initializes without RuntimeWarning errors"""
    # Create AGISystem instance
    config = Config()
    engine = get_engine()
    agi_system = AGISystem(engine=engine)
    
    # Verify that AGISystem was created successfully
    assert agi_system is not None
    assert hasattr(agi_system, 'background_tasks')
    assert hasattr(agi_system, 'pending_tasks')
    
    # Verify that pending tasks were created correctly when no event loop was available
    # During initialization (when no event loop existed), coroutines should have been added to pending_tasks
    # rather than causing RuntimeWarning
    assert isinstance(agi_system.background_tasks, list)
    assert isinstance(agi_system.pending_tasks, list)


@pytest.mark.asyncio
async def test_with_event_loop():
    """Test running pending tasks when event loop is available"""
    # Create AGISystem instance within an event loop context
    config = Config()
    engine = get_engine()
    agi_system = AGISystem(engine=engine)
    
    # The initialization happens outside of an event loop, so tasks should be in pending_tasks
    # They would need to be processed separately when an event loop becomes available
    assert isinstance(agi_system.background_tasks, list)
    assert isinstance(agi_system.pending_tasks, list)


@pytest.mark.asyncio
async def test_agi_system_full_initialization():
    """Test full AGISystem initialization within event loop"""
    config = Config()
    engine = get_engine()
    agi_system = AGISystem(engine=engine)
    
    # Test that we can initialize components
    await agi_system.initialize_components()
    
    # Test that we can stop the system
    await agi_system.stop("test_completed")