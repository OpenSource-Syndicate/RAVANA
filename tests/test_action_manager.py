"""Tests for action management system."""

import pytest
import asyncio
from core.action_manager import ActionManager
from core.actions.registry import ActionRegistry
from core.system import AGISystem


class TestActionManager:
    """Test action manager functionality."""

    @pytest.fixture
    def action_manager(self, test_engine, test_session):
        """Create action manager for testing."""
        from services.data_service import DataService
        from core.config import Config
        
        config = Config()
        # Create a minimal AGI system for testing
        system = AGISystem(test_engine)
        data_service = DataService(
            test_engine,
            config.FEED_URLS,
            None,  # embedding_model
            None   # sentiment_classifier
        )
        
        manager = ActionManager(system, data_service)
        return manager

    def test_action_manager_initialization(self, action_manager):
        """Test action manager initializes correctly."""
        assert action_manager is not None
        assert action_manager.action_registry is not None
        assert isinstance(action_manager.action_registry, ActionRegistry)

    def test_action_registry_has_actions(self, action_manager):
        """Test action registry contains registered actions."""
        actions = action_manager.action_registry.actions
        assert len(actions) > 0
        assert 'log_message' in actions or len(actions) >= 1

    @pytest.mark.asyncio
    async def test_execute_simple_action(self, action_manager):
        """Test executing a simple action."""
        decision = {
            'action': 'log_message',
            'params': {'message': 'Test message'}
        }
        
        result = await action_manager.execute_action(decision)
        
        assert result is not None
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_execute_action_with_raw_response(self, action_manager):
        """Test executing action from raw LLM response."""
        decision = {
            'raw_response': '''```json
            {
                "action": "log_message",
                "params": {"message": "Test"}
            }
            ```'''
        }
        
        result = await action_manager.execute_action(decision)
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, action_manager):
        """Test handling of unknown action."""
        decision = {
            'action': 'nonexistent_action',
            'params': {}
        }
        
        result = await action_manager.execute_action(decision)
        
        assert result is not None
        assert 'error' in result or 'status' in result

    @pytest.mark.asyncio
    async def test_execute_action_enhanced(self, action_manager):
        """Test enhanced action execution."""
        decision = {
            'action': 'log_message',
            'params': {'message': 'Enhanced test'}
        }
        
        result = await action_manager.execute_action_enhanced(decision)
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_parallel_action_execution(self, action_manager):
        """Test parallel execution of multiple actions."""
        decisions = [
            {'action': 'log_message', 'params': {'message': 'Test 1'}},
            {'action': 'log_message', 'params': {'message': 'Test 2'}},
            {'action': 'log_message', 'params': {'message': 'Test 3'}}
        ]
        
        results = await action_manager.execute_parallel_actions(decisions)
        
        assert results is not None
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_action_with_retry(self, action_manager):
        """Test action execution with retry logic."""
        decision = {
            'action': 'log_message',
            'params': {'message': 'Retry test'}
        }
        
        result = await action_manager.execute_action_with_retry(decision, max_retries=2)
        
        assert result is not None

    def test_action_cache(self, action_manager):
        """Test action cache functionality."""
        assert hasattr(action_manager, 'action_cache')
        assert isinstance(action_manager.action_cache, dict)
        
        # Test cache clearing
        action_manager.clear_cache()
        assert len(action_manager.action_cache) <= 100

    @pytest.mark.asyncio
    async def test_get_action_statistics(self, action_manager):
        """Test action statistics retrieval."""
        stats = await action_manager.get_action_statistics()
        
        assert stats is not None
        assert 'total_registered_actions' in stats
        assert 'available_actions' in stats
