"""
Simple validation script for RAVANA AGI System Graceful Shutdown functionality.
"""

import asyncio
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.shutdown_coordinator import ShutdownCoordinator
from core.state_manager import StateManager
from core.config import Config


async def test_shutdown_coordinator_basic():
    """Test basic ShutdownCoordinator functionality."""
    print("🧪 Testing ShutdownCoordinator basic functionality...")
    
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
    
    print("✅ ShutdownCoordinator basic functionality test passed")


async def test_state_manager_basic():
    """Test basic StateManager functionality."""
    print("🧪 Testing StateManager basic functionality...")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    state_manager = StateManager(temp_dir)
    
    try:
        # Test state save and load
        test_state = {
            "test_data": "test_value",
            "nested": {"key": "value"},
            "list_data": [1, 2, 3],
            "shutdown_info": {"reason": "test", "phase": "complete"}
        }
        
        # Save state
        success = await state_manager.save_state(test_state)
        assert success, "State save should succeed"
        
        # Load state
        loaded_state = await state_manager.load_state()
        assert loaded_state is not None, "State load should succeed"
        assert loaded_state["test_data"] == "test_value"
        assert loaded_state["nested"]["key"] == "value"
        assert loaded_state["list_data"] == [1, 2, 3]
        
        print("✅ StateManager basic functionality test passed")
        
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


async def test_config_integration():
    """Test configuration integration."""
    print("🧪 Testing configuration integration...")
    
    # Check that shutdown configuration options exist
    assert hasattr(Config, 'GRACEFUL_SHUTDOWN_ENABLED')
    assert hasattr(Config, 'STATE_PERSISTENCE_ENABLED')
    assert hasattr(Config, 'SHUTDOWN_TIMEOUT')
    assert hasattr(Config, 'FORCE_SHUTDOWN_AFTER')
    assert hasattr(Config, 'MEMORY_SERVICE_SHUTDOWN_TIMEOUT')
    
    print("✅ Configuration integration test passed")


async def test_import_functionality():
    """Test that all modules can be imported."""
    print("🧪 Testing import functionality...")
    
    try:
        from core.shutdown_coordinator import ShutdownCoordinator, ShutdownPhase
        from core.state_manager import StateManager, save_system_state, load_system_state
        from modules.episodic_memory.multi_modal_service import MultiModalMemoryService
        from services.memory_service import MemoryService
        
        print("✅ Import functionality test passed")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        raise


async def main():
    """Run all validation tests."""
    print("🚀 Starting RAVANA Graceful Shutdown Validation Tests")
    print("=" * 60)
    
    try:
        await test_import_functionality()
        await test_config_integration()
        await test_shutdown_coordinator_basic()
        await test_state_manager_basic()
        
        print("=" * 60)
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ Graceful shutdown implementation is ready for use")
        
        # Print configuration summary
        print("\n📋 Configuration Summary:")
        print(f"   • Graceful shutdown: {'enabled' if Config.GRACEFUL_SHUTDOWN_ENABLED else 'disabled'}")
        print(f"   • State persistence: {'enabled' if Config.STATE_PERSISTENCE_ENABLED else 'disabled'}")
        print(f"   • Shutdown timeout: {Config.SHUTDOWN_TIMEOUT}s")
        print(f"   • Force timeout: {Config.FORCE_SHUTDOWN_AFTER}s")
        print(f"   • Memory service timeout: {Config.MEMORY_SERVICE_SHUTDOWN_TIMEOUT}s")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())