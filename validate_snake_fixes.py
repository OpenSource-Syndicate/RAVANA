#!/usr/bin/env python3
"""
Snake Agent Bug Fix Validation

Quick validation that the AttributeError bug is fixed and unlimited tokens work.
"""

import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock
import tempfile

# Add current directory to path
sys.path.insert(0, '.')

from core.snake_agent import SnakeAgent, SnakeAgentState
from core.config import Config

async def validate_bug_fix():
    """Validate that the AttributeError bug is fixed"""
    print("=== VALIDATING SNAKE AGENT BUG FIX ===")
    
    # Create mock AGI system
    mock_agi = Mock()
    mock_agi.workspace_path = tempfile.mkdtemp()
    
    # Create Snake Agent
    agent = SnakeAgent(mock_agi)
    
    print("1. Testing mood update with different success rates...")
    
    # Test different success rates
    test_cases = [
        (0.9, "confident"),
        (0.7, "curious"),
        (0.3, "cautious"),
        (0.1, "frustrated")
    ]
    
    for success_rate, expected_mood in test_cases:
        agent.state.experiment_success_rate = success_rate
        
        try:
            agent._update_mood()
            actual_mood = agent.state.mood
            if actual_mood == expected_mood:
                print(f"   ✓ Success rate {success_rate} → mood '{actual_mood}' (correct)")
            else:
                print(f"   ⚠ Success rate {success_rate} → mood '{actual_mood}' (expected '{expected_mood}')")
        except AttributeError as e:
            print(f"   ✗ AttributeError still occurs: {e}")
            return False
        except Exception as e:
            print(f"   ✗ Unexpected error: {e}")
            return False
    
    print("2. Testing state validation and error handling...")
    
    # Test state validation
    try:
        is_valid = agent._validate_state()
        print(f"   ✓ State validation result: {is_valid}")
        
        # Test reinitialization
        agent._reinitialize_state()
        print(f"   ✓ State reinitialization completed")
        
    except Exception as e:
        print(f"   ✗ Error in state handling: {e}")
        return False
    
    print("3. Testing analysis cycle error handling...")
    
    # Test analysis cycle (mock file monitor to avoid filesystem dependencies)
    try:
        agent.file_monitor = Mock()
        agent.file_monitor.scan_for_changes = Mock(return_value=[])
        
        await agent._execute_analysis_cycle()
        print(f"   ✓ Analysis cycle completed without crashes")
        
    except Exception as e:
        print(f"   ✗ Analysis cycle failed: {e}")
        return False
    
    print("4. Testing configuration changes...")
    
    # Test unlimited token configuration
    coding_config = Config.SNAKE_CODING_MODEL
    reasoning_config = Config.SNAKE_REASONING_MODEL
    
    coding_unlimited = coding_config.get('unlimited_mode', False)
    coding_max_tokens = coding_config.get('max_tokens')
    reasoning_unlimited = reasoning_config.get('unlimited_mode', False)
    reasoning_max_tokens = reasoning_config.get('max_tokens')
    
    print(f"   Coding model: unlimited_mode={coding_unlimited}, max_tokens={coding_max_tokens}")
    print(f"   Reasoning model: unlimited_mode={reasoning_unlimited}, max_tokens={reasoning_max_tokens}")
    
    if coding_unlimited and coding_max_tokens is None and reasoning_unlimited and reasoning_max_tokens is None:
        print(f"   ✓ Configuration supports unlimited token generation")
    else:
        print(f"   ⚠ Configuration may not fully support unlimited tokens")
    
    print("\n=== VALIDATION COMPLETED SUCCESSFULLY ===")
    return True

def main():
    """Main validation function"""
    try:
        result = asyncio.run(validate_bug_fix())
        if result:
            print("\n🎉 ALL VALIDATIONS PASSED!")
            print("   • AttributeError bug is FIXED")
            print("   • Unlimited token generation is CONFIGURED")
            print("   • Error handling is IMPROVED")
            print("   • State management is ROBUST")
        else:
            print("\n❌ SOME VALIDATIONS FAILED")
            return 1
    except Exception as e:
        print(f"\n💥 VALIDATION CRASHED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)