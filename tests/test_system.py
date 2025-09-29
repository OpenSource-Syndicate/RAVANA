import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from core.system import AGISystem
from core.config import Config
from core.state import SharedState


@pytest.fixture
def mock_engine():
    return MagicMock()


@pytest.fixture
def agi_system(mock_engine):
    return AGISystem(mock_engine)


@pytest.mark.asyncio
async def test_agi_system_initialization(agi_system):
    assert agi_system is not None
    assert agi_system.engine is not None
    assert isinstance(agi_system.shared_state, SharedState)
    assert agi_system.actions is not None


@pytest.mark.asyncio
async def test_run_autonomous_loop_stops_on_shutdown(agi_system):
    async def stop_after_a_bit():
        await asyncio.sleep(0.1)
        await agi_system.stop()

    # Mock the background tasks so they don't run forever
    agi_system.data_collection_task = AsyncMock()
    agi_system.event_detection_task = AsyncMock()
    agi_system.knowledge_compression_task = AsyncMock()
    agi_system.memory_consolidation_task = AsyncMock()

    # Mock the main loop components to avoid external dependencies
    agi_system.situation_generator.generate_situation = AsyncMock(
        return_value={"prompt": "test situation"})
    agi_system.memory_service.get_relevant_memories = AsyncMock(
        return_value=MagicMock(relevant_memories=[]))
    agi_system.execute_action = AsyncMock(return_value="test action output")
    agi_system._memorize_interaction = AsyncMock()

    # Run the loop and the stopper concurrently
    loop_task = asyncio.create_task(agi_system.run_autonomous_loop())
    stopper_task = asyncio.create_task(stop_after_a_bit())

    await asyncio.gather(loop_task, stopper_task)

    assert agi_system._shutdown.is_set()


@pytest.mark.asyncio
async def test_execute_action_unknown_action(agi_system):
    decision = {"action": "unknown_action", "params": {}}
    output = await agi_system.execute_action(decision)
    assert output == "No action taken."
