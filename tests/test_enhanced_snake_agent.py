"""
Test Suite for Enhanced Snake Agent

Tests the threading and multiprocessing functionality of the enhanced Snake Agent.
"""

import asyncio
import pytest
import tempfile
import time
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

# Import the enhanced components
from core.snake_data_models import (
    SnakeAgentConfiguration, FileChangeEvent, AnalysisTask, TaskPriority
)
from core.snake_log_manager import SnakeLogManager
from core.snake_threading_manager import SnakeThreadingManager
from core.snake_process_manager import SnakeProcessManager
from core.snake_file_monitor import ContinuousFileMonitor
from core.snake_agent_enhanced import EnhancedSnakeAgent


class TestSnakeLogManager:
    """Test the Snake Log Manager"""
    
    @pytest.fixture
    def temp_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_log_manager_initialization(self, temp_log_dir):
        """Test log manager initialization"""
        log_manager = SnakeLogManager(temp_log_dir)
        assert log_manager.log_dir.exists()
        assert log_manager.improvement_logger is not None
        assert log_manager.experiment_logger is not None
        assert log_manager.analysis_logger is not None
        assert log_manager.communication_logger is not None
        assert log_manager.system_logger is not None
    
    def test_log_processor_start_stop(self, temp_log_dir):
        """Test log processor start and stop"""
        log_manager = SnakeLogManager(temp_log_dir)
        
        # Start processor
        log_manager.start_log_processor()
        assert log_manager.worker_running
        assert log_manager.log_worker_thread is not None
        
        # Stop processor
        log_manager.stop_log_processor()
        assert not log_manager.worker_running
    
    @pytest.mark.asyncio
    async def test_system_event_logging(self, temp_log_dir):
        """Test system event logging"""
        log_manager = SnakeLogManager(temp_log_dir)
        log_manager.start_log_processor()
        
        try:
            await log_manager.log_system_event(
                "test_event",
                {"test_data": "value"},
                worker_id="test_worker"
            )
            
            # Give time for log processing
            await asyncio.sleep(0.5)
            
            # Check that log was processed
            assert log_manager.logs_processed > 0
            
        finally:
            log_manager.stop_log_processor()


class TestSnakeDataModels:
    """Test the Snake data models"""
    
    def test_snake_configuration_validation(self):
        """Test configuration validation"""
        # Valid configuration
        config = SnakeAgentConfiguration()
        issues = config.validate()
        assert len(issues) == 0
        
        # Invalid configuration
        invalid_config = SnakeAgentConfiguration(
            max_threads=0,
            max_processes=0,
            file_monitor_interval=0.05
        )
        issues = invalid_config.validate()
        assert len(issues) > 0
    
    def test_file_change_event(self):
        """Test FileChangeEvent data model"""
        event = FileChangeEvent(
            event_id="test_id",
            event_type="modified",
            file_path="test.py",
            absolute_path="/path/to/test.py",
            timestamp=datetime.now()
        )
        
        event_dict = event.to_dict()
        assert event_dict["event_id"] == "test_id"
        assert event_dict["event_type"] == "modified"
        assert "timestamp" in event_dict
    
    def test_analysis_task(self):
        """Test AnalysisTask data model"""
        task = AnalysisTask(
            task_id="analysis_123",
            file_path="test.py",
            analysis_type="file_change",
            priority=TaskPriority.HIGH,
            created_at=datetime.now()
        )
        
        task_dict = task.to_dict()
        assert task_dict["task_id"] == "analysis_123"
        assert task_dict["priority"] == TaskPriority.HIGH.value


class TestSnakeThreadingManager:
    """Test the Snake Threading Manager"""
    
    @pytest.fixture
    def config(self):
        return SnakeAgentConfiguration(
            max_threads=4,
            analysis_threads=2,
            enable_performance_monitoring=False
        )
    
    @pytest.fixture
    def temp_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.mark.asyncio
    async def test_threading_manager_initialization(self, config, temp_log_dir):
        """Test threading manager initialization"""
        log_manager = SnakeLogManager(temp_log_dir)
        log_manager.start_log_processor()
        
        try:
            threading_manager = SnakeThreadingManager(config, log_manager)
            result = await threading_manager.initialize()
            assert result is True
            
        finally:
            log_manager.stop_log_processor()
    
    @pytest.mark.asyncio
    async def test_file_monitor_thread_start(self, config, temp_log_dir):
        """Test starting file monitor thread"""
        log_manager = SnakeLogManager(temp_log_dir)
        log_manager.start_log_processor()
        
        try:
            threading_manager = SnakeThreadingManager(config, log_manager)
            await threading_manager.initialize()
            
            result = await threading_manager.start_file_monitor_thread()
            assert result is True
            assert len(threading_manager.active_threads) > 0
            
            # Cleanup
            await threading_manager.shutdown(timeout=5.0)
            
        finally:
            log_manager.stop_log_processor()
    
    @pytest.mark.asyncio
    async def test_queue_operations(self, config, temp_log_dir):
        """Test queue operations"""
        log_manager = SnakeLogManager(temp_log_dir)
        log_manager.start_log_processor()
        
        try:
            threading_manager = SnakeThreadingManager(config, log_manager)
            await threading_manager.initialize()
            
            # Test file change queuing
            file_event = FileChangeEvent(
                event_id="test_event",
                event_type="modified",
                file_path="test.py",
                absolute_path="/path/to/test.py",
                timestamp=datetime.now()
            )
            
            result = threading_manager.queue_file_change(file_event)
            assert result is True
            
            # Check queue status
            queue_status = threading_manager.get_queue_status()
            assert queue_status["file_changes"] > 0
            
        finally:
            log_manager.stop_log_processor()


class TestSnakeProcessManager:
    """Test the Snake Process Manager"""
    
    @pytest.fixture
    def config(self):
        return SnakeAgentConfiguration(
            max_processes=2,
            process_heartbeat_interval=5.0
        )
    
    @pytest.fixture
    def temp_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.mark.asyncio
    async def test_process_manager_initialization(self, config, temp_log_dir):
        """Test process manager initialization"""
        log_manager = SnakeLogManager(temp_log_dir)
        log_manager.start_log_processor()
        
        try:
            process_manager = SnakeProcessManager(config, log_manager)
            result = await process_manager.initialize()
            assert result is True
            
        finally:
            log_manager.stop_log_processor()
    
    @pytest.mark.asyncio
    async def test_task_distribution(self, config, temp_log_dir):
        """Test task distribution to processes"""
        log_manager = SnakeLogManager(temp_log_dir)
        log_manager.start_log_processor()
        
        try:
            process_manager = SnakeProcessManager(config, log_manager)
            await process_manager.initialize()
            
            # Distribute a test task
            test_task = {
                "type": "experiment",
                "task_id": "test_123",
                "data": {"test": "data"}
            }
            
            result = process_manager.distribute_task(test_task)
            assert result is True
            
            # Check queue status
            queue_status = process_manager.get_queue_status()
            assert queue_status["task_queue"] > 0
            
            # Cleanup
            await process_manager.shutdown(timeout=5.0)
            
        finally:
            log_manager.stop_log_processor()


class TestContinuousFileMonitor:
    """Test the Continuous File Monitor"""
    
    @pytest.fixture
    def config(self):
        return SnakeAgentConfiguration(
            file_monitor_interval=1.0
        )
    
    @pytest.fixture
    def temp_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_dir = Path(tmpdir)
            (test_dir / "test.py").write_text("print('hello world')")
            (test_dir / "test.json").write_text('{"test": true}')
            yield tmpdir
    
    @pytest.fixture
    def temp_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.mark.asyncio
    async def test_file_monitor_initialization(self, config, temp_log_dir, temp_workspace):
        """Test file monitor initialization"""
        log_manager = SnakeLogManager(temp_log_dir)
        log_manager.start_log_processor()
        
        try:
            # Mock snake agent
            mock_snake = Mock()
            
            # Change to temp workspace
            original_cwd = os.getcwd()
            os.chdir(temp_workspace)
            
            try:
                file_monitor = ContinuousFileMonitor(mock_snake, config, log_manager)
                result = await file_monitor.initialize()
                assert result is True
                assert len(file_monitor.tracked_files) > 0
                
            finally:
                os.chdir(original_cwd)
                
        finally:
            log_manager.stop_log_processor()
    
    @pytest.mark.asyncio
    async def test_file_change_detection(self, config, temp_log_dir, temp_workspace):
        """Test file change detection"""
        log_manager = SnakeLogManager(temp_log_dir)
        log_manager.start_log_processor()
        
        try:
            mock_snake = Mock()
            
            original_cwd = os.getcwd()
            os.chdir(temp_workspace)
            
            try:
                file_monitor = ContinuousFileMonitor(mock_snake, config, log_manager)
                await file_monitor.initialize()
                
                # Start monitoring
                await file_monitor.start_monitoring()
                
                # Give it time to start
                await asyncio.sleep(0.5)
                
                # Modify a file
                test_file = Path(temp_workspace) / "test.py"
                test_file.write_text("print('modified')")
                
                # Give time for detection
                await asyncio.sleep(2.0)
                
                # Check monitoring status
                status = file_monitor.get_monitoring_status()
                assert status["monitoring_active"] is True
                
                # Stop monitoring
                await file_monitor.stop_monitoring()
                
            finally:
                os.chdir(original_cwd)
                
        finally:
            log_manager.stop_log_processor()


class TestEnhancedSnakeAgent:
    """Test the Enhanced Snake Agent integration"""
    
    @pytest.fixture
    def mock_agi_system(self):
        mock_system = Mock()
        mock_system.workspace_path = os.getcwd()
        return mock_system
    
    @pytest.mark.asyncio
    async def test_enhanced_agent_initialization(self, mock_agi_system):
        """Test enhanced agent initialization"""
        with patch('core.snake_llm.create_snake_coding_llm') as mock_coding_llm, \
             patch('core.snake_llm.create_snake_reasoning_llm') as mock_reasoning_llm:
            
            mock_coding_llm.return_value = Mock()
            mock_reasoning_llm.return_value = Mock()
            
            agent = EnhancedSnakeAgent(mock_agi_system)
            
            # Test configuration validation
            config_issues = agent.snake_config.validate()
            assert len(config_issues) == 0
            
            # Note: Full initialization test would require actual LLM setup
            # This tests the basic structure
            assert agent.agi_system is mock_agi_system
            assert agent.snake_config is not None
    
    def test_enhanced_agent_status(self, mock_agi_system):
        """Test enhanced agent status reporting"""
        agent = EnhancedSnakeAgent(mock_agi_system)
        
        status = agent.get_status()
        assert "running" in status
        assert "initialized" in status
        assert "metrics" in status
        assert "components" in status


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_log_manager_integration(self):
        """Test log manager integration across components"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_manager = SnakeLogManager(temp_dir)
            log_manager.start_log_processor()
            
            try:
                config = SnakeAgentConfiguration(max_threads=2, max_processes=1)
                
                # Test threading manager with log manager
                threading_manager = SnakeThreadingManager(config, log_manager)
                await threading_manager.initialize()
                
                # Test process manager with log manager
                process_manager = SnakeProcessManager(config, log_manager)
                await process_manager.initialize()
                
                # Give time for logging
                await asyncio.sleep(1.0)
                
                # Check that logs were created
                log_files = list(Path(temp_dir).glob("*.log"))
                assert len(log_files) > 0
                
                # Cleanup
                await threading_manager.shutdown(timeout=3.0)
                await process_manager.shutdown(timeout=3.0)
                
            finally:
                log_manager.stop_log_processor()


def run_tests():
    """Run the test suite"""
    print("Running Enhanced Snake Agent Test Suite...")
    
    # Run pytest
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ]
    
    return pytest.main(pytest_args)


if __name__ == "__main__":
    import sys
    sys.exit(run_tests())