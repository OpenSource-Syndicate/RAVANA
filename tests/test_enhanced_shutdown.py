"""
Test suite for the enhanced graceful shutdown mechanism.
"""

from services.memory_service import MemoryService
from core.config import Config
from core.state_manager import StateManager
from core.shutdown_coordinator import ShutdownCoordinator, ShutdownPhase, ShutdownPriority, Shutdownable
import asyncio
import sys
import os
import tempfile
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockComponent(Shutdownable):
    """Mock component for testing shutdown functionality."""

    def __init__(self, name: str):
        self.name = name
        self.prepare_called = False
        self.shutdown_called = False
        self.metrics_called = False

    async def prepare_shutdown(self) -> bool:
        self.prepare_called = True
        return True

    async def shutdown(self, timeout: float = 30.0) -> bool:
        self.shutdown_called = True
        return True

    def get_shutdown_metrics(self) -> dict:
        self.metrics_called = True
        return {"component": self.name, "status": "shutdown"}


async def test_shutdown_coordinator_enhanced():
    """Test enhanced ShutdownCoordinator functionality."""
    print("🧪 Testing enhanced ShutdownCoordinator functionality...")

    # Create a mock AGI system
    class MockAGISystem:
        def __init__(self):
            self._shutdown = asyncio.Event()
            self.background_tasks = []
            self.session = None

    mock_agi = MockAGISystem()
    coordinator = ShutdownCoordinator(mock_agi)

    # Test initialization
    assert coordinator is not None
    assert coordinator.agi_system == mock_agi
    assert not coordinator.shutdown_in_progress

    # Test component registration
    mock_component = MockComponent("test_component")
    coordinator.register_component(
        mock_component, ShutdownPriority.HIGH, is_async=True)

    assert len(coordinator.registered_components) == 1
    assert coordinator.registered_components[0].component == mock_component
    assert coordinator.registered_components[0].priority == ShutdownPriority.HIGH
    assert coordinator.registered_components[0].is_async == True

    # Test new shutdown phases
    expected_phases = [
        ShutdownPhase.PRE_SHUTDOWN_VALIDATION,
        ShutdownPhase.SIGNAL_RECEIVED,
        ShutdownPhase.COMPONENT_NOTIFICATION,
        ShutdownPhase.TASKS_STOPPING,
        ShutdownPhase.RESOURCE_CLEANUP,
        ShutdownPhase.SERVICE_SHUTDOWN,
        ShutdownPhase.STATE_PERSISTENCE,
        ShutdownPhase.FINAL_VALIDATION,
        ShutdownPhase.SHUTDOWN_COMPLETE
    ]

    # Verify all expected phases exist
    for phase in expected_phases:
        assert isinstance(phase, ShutdownPhase)

    print("✅ Enhanced ShutdownCoordinator functionality test passed")


async def test_state_manager_enhanced():
    """Test enhanced StateManager functionality."""
    print("🧪 Testing enhanced StateManager functionality...")

    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    state_manager = StateManager(temp_dir)

    try:
        # Create an initial state file to trigger backup creation
        initial_state = {
            "test_data": "initial_value",
            "shutdown_info": {"reason": "initial", "phase": "setup"}
        }
        with open(state_manager.state_file, 'w') as f:
            json.dump(initial_state, f)

        # Test state save and load with compression
        test_state = {
            "test_data": "test_value",
            "nested": {"key": "value"},
            "list_data": [1, 2, 3],
            "shutdown_info": {"reason": "test", "phase": "complete"}
        }

        # Save state (this should create a backup of the initial state)
        success = await state_manager.save_state(test_state)
        assert success, "State save should succeed"

        # Load state
        loaded_state = await state_manager.load_state()
        assert loaded_state is not None, "State load should succeed"
        assert loaded_state["test_data"] == "test_value"
        assert loaded_state["nested"]["key"] == "value"
        assert loaded_state["list_data"] == [1, 2, 3]

        # Test backup functionality
        backup_files = list(
            state_manager.backup_dir.glob("state_backup_*.json*"))
        assert len(backup_files) > 0, "Backup should be created"

        print("✅ Enhanced StateManager functionality test passed")

    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


async def test_component_shutdown_interface():
    """Test component shutdown interface implementation."""
    print("🧪 Testing component shutdown interface...")

    # Create mock component
    mock_component = MockComponent("test_shutdown_component")

    # Test prepare_shutdown
    result = await mock_component.prepare_shutdown()
    assert result is True
    assert mock_component.prepare_called is True

    # Test shutdown
    result = await mock_component.shutdown()
    assert result is True
    assert mock_component.shutdown_called is True

    # Test get_shutdown_metrics
    metrics = mock_component.get_shutdown_metrics()
    assert isinstance(metrics, dict)
    assert metrics["component"] == "test_shutdown_component"
    assert metrics["status"] == "shutdown"
    assert mock_component.metrics_called is True

    print("✅ Component shutdown interface test passed")


async def test_memory_service_shutdown():
    """Test MemoryService shutdown interface implementation."""
    print("🧪 Testing MemoryService shutdown interface...")

    # Create MemoryService instance
    memory_service = MemoryService()

    # Test prepare_shutdown
    result = await memory_service.prepare_shutdown()
    assert result is True

    # Test shutdown
    result = await memory_service.shutdown()
    assert result is True

    # Test get_shutdown_metrics
    metrics = memory_service.get_shutdown_metrics()
    assert isinstance(metrics, dict)
    assert "status" in metrics

    print("✅ MemoryService shutdown interface test passed")


async def test_config_enhancements():
    """Test enhanced configuration options."""
    print("🧪 Testing enhanced configuration options...")

    # Check that new shutdown configuration options exist
    assert hasattr(Config, 'SHUTDOWN_HEALTH_CHECK_ENABLED')
    assert hasattr(Config, 'SHUTDOWN_BACKUP_ENABLED')
    assert hasattr(Config, 'SHUTDOWN_BACKUP_COUNT')
    assert hasattr(Config, 'SHUTDOWN_STATE_VALIDATION_ENABLED')
    assert hasattr(Config, 'SHUTDOWN_VALIDATION_ENABLED')
    assert hasattr(Config, 'SHUTDOWN_COMPRESSION_ENABLED')
    assert hasattr(Config, 'COMPONENT_PREPARE_TIMEOUT')
    assert hasattr(Config, 'COMPONENT_SHUTDOWN_TIMEOUT')

    print("✅ Enhanced configuration options test passed")


async def test_import_functionality():
    """Test that all enhanced modules can be imported."""
    print("🧪 Testing import functionality...")

    try:
        from core.shutdown_coordinator import ShutdownCoordinator, ShutdownPhase, ShutdownPriority, Shutdownable
        from core.state_manager import StateManager
        from core.config import Config
        from services.memory_service import MemoryService

        print("✅ Import functionality test passed")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise


async def main():
    """Run all enhanced shutdown validation tests."""
    print("🚀 Starting RAVANA Enhanced Graceful Shutdown Validation Tests")
    print("=" * 70)

    try:
        await test_import_functionality()
        await test_config_enhancements()
        await test_shutdown_coordinator_enhanced()
        await test_state_manager_enhanced()
        await test_component_shutdown_interface()
        await test_memory_service_shutdown()

        print("=" * 70)
        print("🎉 ALL ENHANCED VALIDATION TESTS PASSED!")
        print("✅ Enhanced graceful shutdown implementation is ready for use")

        # Print enhanced configuration summary
        print("\n📋 Enhanced Configuration Summary:")
        print(
            f"   • Health checks: {'enabled' if Config.SHUTDOWN_HEALTH_CHECK_ENABLED else 'disabled'}")
        print(
            f"   • State backup: {'enabled' if Config.SHUTDOWN_BACKUP_ENABLED else 'disabled'}")
        print(
            f"   • State compression: {'enabled' if getattr(Config, 'SHUTDOWN_COMPRESSION_ENABLED', True) else 'disabled'}")
        print(
            f"   • State validation: {'enabled' if Config.SHUTDOWN_STATE_VALIDATION_ENABLED else 'disabled'}")
        print(f"   • Backup count: {Config.SHUTDOWN_BACKUP_COUNT}")
        print(
            f"   • Component prepare timeout: {Config.COMPONENT_PREPARE_TIMEOUT}s")
        print(
            f"   • Component shutdown timeout: {Config.COMPONENT_SHUTDOWN_TIMEOUT}s")
        print(
            f"   • ChromaDB persistence: {'enabled' if Config.CHROMADB_PERSIST_ON_SHUTDOWN else 'disabled'}")

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
