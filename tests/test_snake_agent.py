"""
Snake Agent Testing Framework

This module provides comprehensive testing and validation for the Snake Agent system,
including unit tests, integration tests, and validation scripts.
"""

import asyncio
import os
import sys
import unittest
import tempfile
import shutil
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import logging

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.config import Config
from core.snake_llm import SnakeConfigValidator, SnakeCodingLLM, SnakeReasoningLLM
from core.snake_agent import SnakeAgent, SnakeAgentState, FileSystemMonitor
from core.snake_code_analyzer import SnakeCodeAnalyzer, ASTAnalyzer, PatternAnalyzer
from core.snake_safe_experimenter import SnakeSafeExperimenter, SandboxEnvironment, CodeSafetyAnalyzer
from core.snake_ravana_communicator import SnakeRavanaCommunicator, CommunicationMessage, Priority

logger = logging.getLogger(__name__)


class MockAGISystem:
    """Mock AGI system for testing"""
    
    def __init__(self):
        self.memory_service = Mock()
        self.shared_state = Mock()
        self.workspace_path = tempfile.mkdtemp()
        self.background_tasks = []
        self._shutdown = asyncio.Event()
        
        # Mock memory service methods
        self.memory_service.add_episodic_memory = AsyncMock()
        self.memory_service.search_memories = AsyncMock(return_value=[])
        
        # Mock shared state
        self.shared_state.snake_communications = []


class TestSnakeConfigValidator(unittest.TestCase):
    """Test Snake Agent configuration validation"""
    
    def test_startup_report_structure(self):
        """Test that startup report has required structure"""
        report = SnakeConfigValidator.get_startup_report()
        
        required_keys = [
            "ollama_connected", "coding_model_available", 
            "reasoning_model_available", "available_models", "config_valid"
        ]
        
        for key in required_keys:
            self.assertIn(key, report)
        
        self.assertIsInstance(report["ollama_connected"], bool)
        self.assertIsInstance(report["coding_model_available"], bool)
        self.assertIsInstance(report["reasoning_model_available"], bool)
        self.assertIsInstance(report["available_models"], list)
        self.assertIsInstance(report["config_valid"], bool)


class TestSnakeAgentState(unittest.TestCase):
    """Test Snake Agent state management"""
    
    def test_state_initialization(self):
        """Test state initialization with defaults"""
        state = SnakeAgentState()
        
        self.assertIsInstance(state.analyzed_files, set)
        self.assertIsInstance(state.pending_experiments, list)
        self.assertIsInstance(state.communication_queue, list)
        self.assertIsInstance(state.learning_history, list)
        self.assertEqual(state.mood, "curious")
        self.assertEqual(state.experiment_success_rate, 0.0)
    
    def test_state_serialization(self):
        """Test state to/from dict conversion"""
        state = SnakeAgentState()
        state.analyzed_files.add("test_file.py")
        state.pending_experiments.append({"id": "test_exp"})
        state.mood = "confident"
        
        # Convert to dict
        state_dict = state.to_dict()
        self.assertIn("analyzed_files", state_dict)
        self.assertIn("pending_experiments", state_dict)
        self.assertIn("mood", state_dict)
        
        # Convert back from dict
        restored_state = SnakeAgentState.from_dict(state_dict)
        self.assertEqual(restored_state.mood, "confident")
        self.assertIn("test_file.py", restored_state.analyzed_files)
        self.assertEqual(len(restored_state.pending_experiments), 1)


class TestFileSystemMonitor(unittest.TestCase):
    """Test file system monitoring functionality"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.monitor = FileSystemMonitor(str(self.temp_dir))
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_monitor_initialization(self):
        """Test monitor initialization"""
        self.assertEqual(self.monitor.root_path, self.temp_dir)
        self.assertIsInstance(self.monitor.file_hashes, dict)
    
    def test_file_change_detection(self):
        """Test detection of file changes"""
        # Create a test file
        test_file = self.temp_dir / "test.py"
        test_file.write_text("print('hello')")
        
        # First scan - should detect new file
        changes = self.monitor.scan_for_changes()
        new_files = [c for c in changes if c["type"] == "new"]
        self.assertEqual(len(new_files), 1)
        self.assertEqual(new_files[0]["path"], "test.py")
        
        # Modify the file
        test_file.write_text("print('hello world')")
        
        # Second scan - should detect modification
        changes = self.monitor.scan_for_changes()
        modified_files = [c for c in changes if c["type"] == "modified"]
        self.assertEqual(len(modified_files), 1)
        self.assertEqual(modified_files[0]["path"], "test.py")


class TestASTAnalyzer(unittest.TestCase):
    """Test AST-based code analysis"""
    
    def setUp(self):
        self.analyzer = ASTAnalyzer()
    
    def test_basic_analysis(self):
        """Test basic code analysis"""
        code = '''
def test_function(a, b, c, d, e, f, g, h):
    """This function has too many parameters"""
    if a > b:
        if c > d:
            if e > f:
                return g + h
    return 0

class TestClass:
    pass
'''
        lines = code.strip().split('\n')
        metrics, issues = self.analyzer.analyze_ast(compile(code, '<test>', 'exec'), lines)
        
        # Check metrics
        self.assertEqual(metrics.function_count, 1)
        self.assertEqual(metrics.class_count, 1)
        self.assertTrue(metrics.lines_of_code > 5)
        
        # Check issues - should detect too many parameters
        param_issues = [issue for issue in issues if "parameters" in issue.description]
        self.assertTrue(len(param_issues) > 0)
    
    def test_complexity_calculation(self):
        """Test cyclomatic complexity calculation"""
        complex_code = '''
def complex_function(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                while i > 0:
                    i -= 1
                    if i < 5:
                        break
            else:
                try:
                    return i / x
                except ZeroDivisionError:
                    return 0
    return -1
'''
        lines = complex_code.strip().split('\n')
        metrics, issues = self.analyzer.analyze_ast(compile(complex_code, '<test>', 'exec'), lines)
        
        # Should detect high complexity
        self.assertTrue(metrics.complexity > 5)
        complexity_issues = [issue for issue in issues if "complexity" in issue.description]
        self.assertTrue(len(complexity_issues) > 0)


class TestPatternAnalyzer(unittest.TestCase):
    """Test pattern-based code analysis"""
    
    def setUp(self):
        self.analyzer = PatternAnalyzer()
    
    def test_security_pattern_detection(self):
        """Test detection of security anti-patterns"""
        insecure_code = '''
password = "hardcoded_secret"
query = "SELECT * FROM users WHERE id = %s" % user_id
'''
        issues = self.analyzer.analyze_patterns(insecure_code, "test.py")
        
        # Should detect hardcoded secret
        secret_issues = [issue for issue in issues if "secret" in issue.description.lower()]
        self.assertTrue(len(secret_issues) > 0)
    
    def test_performance_pattern_detection(self):
        """Test detection of performance anti-patterns"""
        slow_code = '''
import requests

async def fetch_data():
    result = requests.get("http://example.com")
    return result.json()
'''
        issues = self.analyzer.analyze_patterns(slow_code, "test.py")
        
        # Should detect sync request in async function
        sync_issues = [issue for issue in issues if "synchronous" in issue.description.lower()]
        self.assertTrue(len(sync_issues) > 0)


class TestCodeSafetyAnalyzer(unittest.TestCase):
    """Test code safety analysis"""
    
    def setUp(self):
        self.analyzer = CodeSafetyAnalyzer()
    
    def test_safe_code_analysis(self):
        """Test analysis of safe code"""
        safe_code = '''
import math

def calculate_area(radius):
    return math.pi * radius ** 2

result = calculate_area(5)
print(f"Area: {result}")
'''
        is_safe, warnings, score = self.analyzer.analyze_safety(safe_code)
        
        self.assertTrue(is_safe)
        self.assertTrue(score > 0.7)
        self.assertEqual(len(warnings), 0)
    
    def test_dangerous_code_analysis(self):
        """Test analysis of dangerous code"""
        dangerous_code = '''
import os
import subprocess

user_input = input("Enter command: ")
os.system(user_input)
subprocess.call(user_input, shell=True)
exec("print('dangerous')")
'''
        is_safe, warnings, score = self.analyzer.analyze_safety(dangerous_code)
        
        self.assertFalse(is_safe)
        self.assertTrue(score < 0.7)
        self.assertTrue(len(warnings) > 0)
        
        # Check specific warnings
        warning_text = " ".join(warnings)
        self.assertIn("Dangerous", warning_text)


class TestSandboxEnvironment(unittest.IsolatedAsyncioTestCase):
    """Test sandbox environment functionality"""
    
    async def test_sandbox_lifecycle(self):
        """Test sandbox creation and cleanup"""
        async with SandboxEnvironment("test_exp") as sandbox:
            self.assertTrue(sandbox.is_active)
            self.assertTrue(sandbox.sandbox_dir.exists())
        
        # After context exit, sandbox should be cleaned up
        self.assertFalse(sandbox.is_active)
    
    async def test_code_execution(self):
        """Test code execution in sandbox"""
        test_code = '''
print("Hello from sandbox")
result = 2 + 2
print(f"Result: {result}")
'''
        
        async with SandboxEnvironment("test_exec") as sandbox:
            success, output, exec_time = await sandbox.execute_code(
                test_code, "test_script.py"
            )
            
            self.assertTrue(success)
            self.assertIn("Hello from sandbox", output)
            self.assertIn("Result: 4", output)
            self.assertTrue(exec_time > 0)
    
    async def test_timeout_handling(self):
        """Test timeout handling in sandbox"""
        timeout_code = '''
import time
time.sleep(10)  # This should timeout
'''
        
        async with SandboxEnvironment("test_timeout") as sandbox:
            success, output, exec_time = await sandbox.execute_code(
                timeout_code, "timeout_script.py", timeout=2
            )
            
            self.assertFalse(success)
            self.assertIn("timed out", output.lower())


class TestCommunicationMessage(unittest.TestCase):
    """Test communication message functionality"""
    
    def test_message_creation(self):
        """Test communication message creation"""
        from core.snake_ravana_communicator import CommunicationType
        
        message = CommunicationMessage(
            id="test_msg_001",
            type=CommunicationType.PROPOSAL,
            priority=Priority.HIGH,
            timestamp=time.time(),
            subject="Test Proposal",
            content={"test": "data"}
        )
        
        self.assertEqual(message.id, "test_msg_001")
        self.assertEqual(message.type, CommunicationType.PROPOSAL)
        self.assertEqual(message.priority, Priority.HIGH)
        self.assertEqual(message.subject, "Test Proposal")
    
    def test_message_serialization(self):
        """Test message to/from dict conversion"""
        from core.snake_ravana_communicator import CommunicationType
        from datetime import datetime
        
        original_message = CommunicationMessage(
            id="test_msg_002",
            type=CommunicationType.STATUS_UPDATE,
            priority=Priority.MEDIUM,
            timestamp=datetime.now(),
            subject="Status Update",
            content={"status": "running"}
        )
        
        # Convert to dict
        message_dict = original_message.to_dict()
        
        # Convert back
        restored_message = CommunicationMessage.from_dict(message_dict)
        
        self.assertEqual(restored_message.id, original_message.id)
        self.assertEqual(restored_message.type, original_message.type)
        self.assertEqual(restored_message.priority, original_message.priority)
        self.assertEqual(restored_message.subject, original_message.subject)


class SnakeAgentIntegrationTest(unittest.IsolatedAsyncioTestCase):
    """Integration test for Snake Agent components"""
    
    async def asyncSetUp(self):
        self.mock_agi = MockAGISystem()
        
        # Mock LLM interfaces
        self.mock_coding_llm = Mock()
        self.mock_reasoning_llm = Mock()
        
        # Setup async mock methods
        self.mock_coding_llm.analyze_code = AsyncMock(return_value="Code analysis result")
        self.mock_coding_llm.generate_improvement = AsyncMock(return_value="Improved code")
        self.mock_reasoning_llm.evaluate_safety = AsyncMock(return_value={"risk_level": "LOW"})
        self.mock_reasoning_llm.plan_communication = AsyncMock(return_value={"channels": ["logging"]})
    
    async def asyncTearDown(self):
        if hasattr(self.mock_agi, 'workspace_path'):
            shutil.rmtree(self.mock_agi.workspace_path, ignore_errors=True)
    
    async def test_snake_agent_initialization(self):
        """Test Snake Agent initialization"""
        # Mock the configuration validation
        with patch('core.snake_agent.SnakeConfigValidator.get_startup_report') as mock_report:
            mock_report.return_value = {"config_valid": True, "ollama_connected": True, "available_models": []}
            
            with patch('core.snake_agent.create_snake_coding_llm') as mock_coding:
                with patch('core.snake_agent.create_snake_reasoning_llm') as mock_reasoning:
                    mock_coding.return_value = self.mock_coding_llm
                    mock_reasoning.return_value = self.mock_reasoning_llm
                    
                    snake_agent = SnakeAgent(self.mock_agi)
                    initialized = await snake_agent.initialize()
                    
                    self.assertTrue(initialized)
                    self.assertIsNotNone(snake_agent.coding_llm)
                    self.assertIsNotNone(snake_agent.reasoning_llm)
                    self.assertIsNotNone(snake_agent.file_monitor)
    
    async def test_code_analysis_flow(self):
        """Test the complete code analysis flow"""
        # Create a test Python file
        test_code = '''
def example_function():
    print("Hello World")
    return True
'''
        
        test_file = Path(self.mock_agi.workspace_path) / "test_module.py"
        test_file.write_text(test_code)
        
        # Mock the analyzer components
        with patch('core.snake_agent.SnakeCodeAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_code = AsyncMock(return_value={
                "improvements_suggested": True,
                "priority": "medium",
                "confidence": 0.8,
                "static_issues": [],
                "llm_analysis": {}
            })
            mock_analyzer_class.return_value = mock_analyzer
            
            # Initialize Snake Agent
            with patch('core.snake_agent.SnakeConfigValidator.get_startup_report') as mock_report:
                mock_report.return_value = {"config_valid": True, "ollama_connected": True, "available_models": []}
                
                with patch('core.snake_agent.create_snake_coding_llm') as mock_coding:
                    with patch('core.snake_agent.create_snake_reasoning_llm') as mock_reasoning:
                        mock_coding.return_value = self.mock_coding_llm
                        mock_reasoning.return_value = self.mock_reasoning_llm
                        
                        snake_agent = SnakeAgent(self.mock_agi)
                        await snake_agent.initialize()
                        
                        # Test file analysis
                        await snake_agent._analyze_file(str(test_file), "new")
                        
                        # Verify analysis was called
                        mock_analyzer.analyze_code.assert_called_once()
                        
                        # Verify experiment was created
                        self.assertTrue(len(snake_agent.state.pending_experiments) > 0)


class SnakeAgentValidationSuite:
    """Comprehensive validation suite for Snake Agent system"""
    
    def __init__(self):
        self.results = {}
        self.logger = logging.getLogger(__name__)
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests"""
        self.logger.info("Starting Snake Agent validation suite...")
        
        validations = [
            ("configuration", self.validate_configuration),
            ("llm_interfaces", self.validate_llm_interfaces),
            ("code_analysis", self.validate_code_analysis),
            ("sandbox_safety", self.validate_sandbox_safety),
            ("communication", self.validate_communication),
            ("integration", self.validate_integration)
        ]
        
        for name, validation_func in validations:
            try:
                self.logger.info(f"Running validation: {name}")
                result = await validation_func()
                self.results[name] = {"status": "PASS", "details": result}
                self.logger.info(f"✅ {name} validation passed")
            except Exception as e:
                self.results[name] = {"status": "FAIL", "error": str(e)}
                self.logger.error(f"❌ {name} validation failed: {e}")
        
        return self.results
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """Validate Snake Agent configuration"""
        # Check if required configuration is available
        required_configs = [
            "SNAKE_AGENT_ENABLED", "SNAKE_OLLAMA_BASE_URL",
            "SNAKE_CODING_MODEL", "SNAKE_REASONING_MODEL"
        ]
        
        missing_configs = []
        for config_name in required_configs:
            if not hasattr(Config, config_name):
                missing_configs.append(config_name)
        
        if missing_configs:
            raise ValueError(f"Missing configurations: {missing_configs}")
        
        # Validate model configurations
        coding_model = Config.SNAKE_CODING_MODEL
        reasoning_model = Config.SNAKE_REASONING_MODEL
        
        required_model_keys = ["provider", "model_name", "base_url", "temperature", "max_tokens"]
        for model_name, model_config in [("coding", coding_model), ("reasoning", reasoning_model)]:
            missing_keys = [key for key in required_model_keys if key not in model_config]
            if missing_keys:
                raise ValueError(f"Missing keys in {model_name} model config: {missing_keys}")
        
        return {"configs_validated": len(required_configs), "models_validated": 2}
    
    async def validate_llm_interfaces(self) -> Dict[str, Any]:
        """Validate LLM interface functionality"""
        # Mock Ollama connection for testing
        with patch('core.snake_llm.SnakeConfigValidator.validate_ollama_connection') as mock_conn:
            with patch('core.snake_llm.OllamaClient.pull_model_if_needed') as mock_pull:
                mock_conn.return_value = True
                mock_pull.return_value = True
                
                # Test coding LLM interface
                coding_llm = SnakeCodingLLM()
                await coding_llm.initialize()
                
                # Test reasoning LLM interface  
                reasoning_llm = SnakeReasoningLLM()
                await reasoning_llm.initialize()
                
                return {"coding_llm_initialized": True, "reasoning_llm_initialized": True}
    
    async def validate_code_analysis(self) -> Dict[str, Any]:
        """Validate code analysis capabilities"""
        # Test AST analyzer
        ast_analyzer = ASTAnalyzer()
        test_code = "def test(): pass"
        lines = test_code.split('\n')
        
        import ast
        tree = ast.parse(test_code)
        metrics, issues = ast_analyzer.analyze_ast(tree, lines)
        
        # Test pattern analyzer
        pattern_analyzer = PatternAnalyzer()
        pattern_issues = pattern_analyzer.analyze_patterns(test_code, "test.py")
        
        return {
            "ast_analysis_working": True,
            "pattern_analysis_working": True,
            "metrics_generated": len(metrics.to_dict()) > 0,
            "issues_detected": len(issues) >= 0
        }
    
    async def validate_sandbox_safety(self) -> Dict[str, Any]:
        """Validate sandbox environment safety"""
        safety_analyzer = CodeSafetyAnalyzer()
        
        # Test safe code
        safe_code = "print('hello')"
        is_safe, warnings, score = safety_analyzer.analyze_safety(safe_code)
        
        if not is_safe or score < 0.5:
            raise ValueError("Safe code was marked as unsafe")
        
        # Test dangerous code
        dangerous_code = "import os; os.system('rm -rf /')"
        is_unsafe, warnings, score = safety_analyzer.analyze_safety(dangerous_code)
        
        if is_unsafe or score > 0.7:
            raise ValueError("Dangerous code was marked as safe")
        
        return {"safety_detection_working": True, "safe_code_passed": True, "dangerous_code_blocked": True}
    
    async def validate_communication(self) -> Dict[str, Any]:
        """Validate communication system"""
        # Test message creation and serialization
        from core.snake_ravana_communicator import CommunicationType
        from datetime import datetime
        
        message = CommunicationMessage(
            id="test_validation",
            type=CommunicationType.STATUS_UPDATE,
            priority=Priority.MEDIUM,
            timestamp=datetime.now(),
            subject="Validation Test"
        )
        
        # Test serialization
        message_dict = message.to_dict()
        restored_message = CommunicationMessage.from_dict(message_dict)
        
        if message.id != restored_message.id:
            raise ValueError("Message serialization failed")
        
        return {"message_serialization": True, "communication_channels": 3}
    
    async def validate_integration(self) -> Dict[str, Any]:
        """Validate overall system integration"""
        # Create mock AGI system
        mock_agi = MockAGISystem()
        
        # Test Snake Agent creation
        snake_agent = SnakeAgent(mock_agi)
        
        # Test state management
        initial_state = snake_agent.state
        state_dict = initial_state.to_dict()
        restored_state = SnakeAgentState.from_dict(state_dict)
        
        if initial_state.mood != restored_state.mood:
            raise ValueError("State persistence failed")
        
        # Cleanup
        shutil.rmtree(mock_agi.workspace_path, ignore_errors=True)
        
        return {"agent_creation": True, "state_persistence": True, "mock_integration": True}


async def run_snake_agent_tests():
    """Run all Snake Agent tests and validations"""
    print("🧪 Starting Snake Agent Test Suite")
    print("=" * 50)
    
    # Run unit tests
    print("\n📋 Running Unit Tests...")
    unittest.main(module=None, exit=False, verbosity=2)
    
    # Run validation suite
    print("\n🔍 Running Validation Suite...")
    validator = SnakeAgentValidationSuite()
    results = await validator.run_all_validations()
    
    # Print results
    print("\n📊 Validation Results:")
    print("-" * 30)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = result["status"]
        if status == "PASS":
            print(f"✅ {test_name}: PASSED")
            passed += 1
        else:
            print(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
            failed += 1
    
    print(f"\n📈 Summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Snake Agent tests passed!")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    # Setup logging for tests
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run the complete test suite
    result = asyncio.run(run_snake_agent_tests())
    sys.exit(0 if result else 1)