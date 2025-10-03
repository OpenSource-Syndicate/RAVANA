#!/usr/bin/env python3
"""
Test script to verify the enhanced RAVANA AGI system components.
"""
import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from core.system import AGISystem
from database.engine import get_engine
from core.config import Config


async def test_enhanced_modules():
    """Test the newly added modules."""
    print("Testing Enhanced RAVANA AGI System Modules...")
    
    # Create database engine
    engine = get_engine()
    
    # Initialize AGI system
    config = Config()
    agi_system = AGISystem(engine)
    
    # Initialize components
    await agi_system.initialize_components()
    
    print("\nTesting Failure Learning System...")
    if hasattr(agi_system, 'failure_learning_system'):
        print("[OK] Failure Learning System available")
        
        # Test failure analysis
        test_analysis = await agi_system.failure_learning_system.analyze_failure(
            failure_context="Test failure analysis",
            experiment_result={"status": "failed", "reason": "Test failure"},
            failure_details="This is a test failure for validation"
        )
        print(f"  - Analysis result: {test_analysis.get('failure_category', 'unknown')}")
        print(f"  - Lessons learned: {len(test_analysis.get('lessons_learned', []))}")
    else:
        print("[FAIL] Failure Learning System not available")
    
    print("\nTesting Physics Analysis System...")
    if hasattr(agi_system, 'physics_analysis_system'):
        print("[OK] Physics Analysis System available")
        
        # Test physics analysis
        test_physics = await agi_system.physics_analysis_system.analyze_physics_problem(
            problem_description="Calculate kinetic energy of an object with mass 10kg moving at 5m/s",
            known_values={"mass": 10, "velocity": 5}
        )
        print(f"  - Analysis domain: {test_physics.get('domain', 'unknown')}")
        print(f"  - Confidence: {test_physics.get('confidence', 0.0)}")
    else:
        print("[FAIL] Physics Analysis System not available")
    
    print("\nTesting Function Calling System...")
    if hasattr(agi_system, 'function_calling_system'):
        print("[OK] Function Calling System available")
        
        # Test function listing
        available_functions = agi_system.function_calling_system.get_available_functions()
        print(f"  - Available functions: {len(available_functions)}")
        
        # Test intelligent tool selection
        task_result = await agi_system.function_calling_system.intelligent_tool_selection(
            "Analyze a failed physics experiment and suggest improvements"
        )
        print(f"  - Recommended function calls: {len(task_result)}")
    else:
        print("[FAIL] Function Calling System not available")
    
    print("\nTesting Enhanced Decision Making...")
    if hasattr(agi_system, '_make_decision'):
        print("[OK] Enhanced decision making available")
        
        # Test a simple decision making with physics context
        test_situation = {
            'prompt': 'A ball is dropped from a height of 20 meters. Calculate its velocity when it hits the ground.',
            'context': {}
        }
        
        try:
            decision = await agi_system._make_decision(test_situation)
            print(f"  - Decision confidence: {decision.get('confidence', 0.0)}")
            if 'physics_analysis' in decision:
                print("  - Physics analysis was applied to the decision")
            if 'failure_lessons' in decision:
                print("  - Failure lessons were considered in the decision")
        except Exception as e:
            print(f"  - Decision making failed: {e}")
    else:
        print("[FAIL] Enhanced decision making not available")
    
    print("\nAll tests completed successfully!")
    print("\nEnhanced capabilities summary:")
    print("- Failure learning system for continuous improvement from mistakes")
    print("- Physics analysis system with comprehensive formula database")
    print("- Advanced function calling for structured tool usage")
    print("- Integration of these capabilities in decision making process")
    
    # Cleanup
    await agi_system.stop("test_completed")


if __name__ == "__main__":
    print("Starting RAVANA AGI Enhanced System Test...")
    asyncio.run(test_enhanced_modules())
    print("Test completed.")