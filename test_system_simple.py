#!/usr/bin/env python3
"""
Simple System Test - ASCII only version
Tests AGI physics experimentation components.
"""

import sys
import os
import time
import traceback

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing module imports...")
    
    try:
        # Test physics experiment prompts
        from physics_experiment_prompts import ADVANCED_PHYSICS_EXPERIMENTS, DISCOVERY_PROMPTS
        print(f"[OK] Physics prompts loaded: {len(ADVANCED_PHYSICS_EXPERIMENTS)} experiments, {len(DISCOVERY_PROMPTS)} discovery prompts")
        
        # Test LLM module
        from modules.decision_engine.llm import agi_experimentation_engine, call_llm
        print("[OK] LLM experimentation engine imported successfully")
        
        # Test core components (without initializing)
        from core.system import AGISystem
        from database.engine import create_db_and_tables
        print("[OK] Core AGI system components imported successfully")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        traceback.print_exc()
        return False

def test_experiment_prompts():
    """Test the physics experiment prompt system."""
    print("\nTesting experiment prompt system...")
    
    try:
        from physics_experiment_prompts import (
            get_random_experiment, 
            get_discovery_prompt, 
            get_experiments_by_difficulty
        )
        
        # Test random experiment
        random_exp = get_random_experiment()
        print(f"[OK] Random experiment: {random_exp['name']} ({random_exp['difficulty']})")
        
        # Test discovery prompt
        discovery = get_discovery_prompt()
        print(f"[OK] Discovery prompt: {discovery[:80]}...")
        
        # Test difficulty filtering
        advanced_exps = get_experiments_by_difficulty('advanced')
        print(f"[OK] Advanced experiments: {len(advanced_exps)} found")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Experiment prompt test failed: {e}")
        traceback.print_exc()
        return False

def test_llm_config():
    """Test LLM configuration and providers."""
    print("\nTesting LLM configuration...")
    
    try:
        from modules.decision_engine.llm import PROVIDERS, config
        
        print(f"[OK] LLM config loaded with {len(PROVIDERS)} providers")
        
        # Check provider configuration
        for provider in PROVIDERS:
            print(f"  - {provider['name']}: {len(provider['models'])} models")
        
        # Test config loading
        print(f"[OK] Config keys: {list(config.keys())}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] LLM config test failed: {e}")
        traceback.print_exc()
        return False

def show_available_experiments():
    """Show available physics experiments."""
    print("\nAVAILABLE PHYSICS EXPERIMENTS:")
    print("="*60)
    
    try:
        from physics_experiment_prompts import ADVANCED_PHYSICS_EXPERIMENTS
        
        for i, exp in enumerate(ADVANCED_PHYSICS_EXPERIMENTS, 1):
            print(f"{i:2d}. {exp['name']} ({exp['difficulty']})")
            print(f"    {exp['prompt'][:80]}...")
            print()
        
    except Exception as e:
        print(f"Could not load experiments: {e}")

def run_basic_experiment_test():
    """Run a basic experiment test with mock data."""
    print("\nTesting basic experiment functionality...")
    
    try:
        # Test with a simple quantum mechanics experiment
        experiment_idea = """
        Create a Python simulation to calculate the probability of quantum tunneling 
        for an electron passing through a potential barrier. Use the following parameters:
        - Electron energy: 5 eV
        - Barrier height: 10 eV  
        - Barrier width: 1 nanometer
        Calculate and plot the transmission coefficient.
        """
        
        print("[INFO] Testing experiment idea processing...")
        print(f"[INFO] Experiment prompt: {experiment_idea[:100]}...")
        
        # Import the experimentation engine
        from modules.decision_engine.llm import agi_experimentation_engine
        
        print("[INFO] Starting experimentation engine...")
        print("[WARNING] This will make API calls and may take time...")
        
        # Run with shorter timeout for testing
        start_time = time.time()
        result = agi_experimentation_engine(
            experiment_idea=experiment_idea + "\n\nIMPORTANT: Save the plot as 'quantum_tunneling_plot.png' instead of using plt.show() to avoid blocking.",
            llm_model=None,
            use_chain_of_thought=True,
            online_validation=False,  # Disable to speed up testing
            sandbox_timeout=15,  # Increased timeout
            verbose=True
        )
        
        execution_time = time.time() - start_time
        
        print(f"\n[SUCCESS] Experiment completed in {execution_time:.2f} seconds")
        print(f"[INFO] Simulation type: {result.get('simulation_type', 'Unknown')}")
        print(f"[INFO] Final verdict: {result.get('final_verdict', 'No verdict')[:100]}...")
        
        if result.get('generated_code'):
            print(f"[INFO] Code generated: {len(result['generated_code'])} characters")
        
        if result.get('execution_result'):
            print(f"[INFO] Execution result available: {len(str(result['execution_result']))} characters")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Basic experiment test failed: {e}")
        traceback.print_exc()
        return False

def run_system_tests():
    """Run all system tests."""
    print("STARTING AGI PHYSICS EXPERIMENTATION SYSTEM TESTS")
    print("="*60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Experiment Prompts", test_experiment_prompts),
        ("LLM Configuration", test_llm_config),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"[PASS] {test_name}")
            else:
                print(f"[FAIL] {test_name}")
        except Exception as e:
            print(f"[FAIL] {test_name} - Exception: {e}")
    
    print("\n" + "="*60)
    print("SYSTEM TEST SUMMARY")
    print("="*60)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("[SUCCESS] All system tests passed!")
        print("The AGI physics experimentation system is ready for testing.")
        print("\nTo run a full experiment test:")
        print("python test_system_simple.py experiment")
    else:
        print("[WARNING] Some system tests failed.")
        print("Fix these issues before running experiments.")
    
    return passed == total

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "experiments":
            show_available_experiments()
        elif sys.argv[1] == "experiment":
            run_basic_experiment_test()
        elif sys.argv[1] == "system":
            run_system_tests()
        else:
            print("Usage: python test_system_simple.py [system|experiments|experiment]")
    else:
        # Default: run system tests
        run_system_tests()
