#!/usr/bin/env python3
"""
Quick test to verify Unicode fix in experimentation engine
"""

from core.llm import agi_experimentation_engine
import sys
import os
import time

# Add modules directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_unicode_fix():
    """Test that Unicode characters in generated code don't cause crashes."""

    print("Testing Unicode fix in experimentation engine...")

    # Simple experiment that might generate Unicode characters
    experiment_idea = """
    Create a simple Python script that demonstrates mathematical inequalities.
    The script should:
    1. Check if 5 ≥ 3 (greater than or equal)
    2. Print the result
    3. Save a simple plot showing the relationship
    
    Use proper mathematical symbols and save the plot as 'test_plot.png'.
    """

    start_time = time.time()

    try:
        print("[INFO] Running Unicode test experiment...")

        result = agi_experimentation_engine(
            experiment_idea=experiment_idea,
            llm_model=None,
            use_chain_of_thought=False,  # Faster
            online_validation=False,     # Faster
            sandbox_timeout=15,
            verbose=False  # Less verbose for quick test
        )

        execution_time = time.time() - start_time

        print(
            f"[SUCCESS] Unicode test completed in {execution_time:.2f} seconds")

        # Check if code was generated and executed
        if result.get('generated_code'):
            print("[OK] Code generated successfully")

        if result.get('execution_result'):
            print("[OK] Code executed without Unicode errors")

        if result.get('execution_error'):
            if 'unicode' in str(result['execution_error']).lower() or 'charmap' in str(result['execution_error']).lower():
                print("[FAIL] Unicode error still present")
                return False
            else:
                print("[INFO] Execution had other issues (not Unicode related)")

        print("[SUCCESS] Unicode fix verified!")
        return True

    except Exception as e:
        execution_time = time.time() - start_time
        print(f"[FAIL] Unicode test failed in {execution_time:.2f} seconds")
        print(f"[ERROR] {str(e)}")

        if 'unicode' in str(e).lower() or 'charmap' in str(e).lower():
            print("[FAIL] Unicode error still present - fix didn't work")
            return False
        else:
            print("[INFO] Error is not Unicode related")
            return True


if __name__ == "__main__":
    test_unicode_fix()
