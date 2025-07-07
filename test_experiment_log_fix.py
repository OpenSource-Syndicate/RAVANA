#!/usr/bin/env python3
"""
Test script to validate the save_experiment_log fix.
This tests both calling conventions to ensure backward compatibility.
"""

import sys
import os
import tempfile
import json
from sqlmodel import create_engine, Session
from database.engine import create_db_and_tables
from services.data_service import DataService
from database.models import ExperimentLog

def test_experiment_log_fix():
    """Test both calling conventions for save_experiment_log"""
    
    # Create a temporary in-memory database for testing
    engine = create_engine("sqlite:///:memory:")
    create_db_and_tables(engine)
    
    # Create DataService instance
    data_service = DataService(engine, feed_urls=[])
    
    print("Testing save_experiment_log fix...")
    
    # Test 1: Original 2-parameter style (hypothesis + results dict)
    print("\n1. Testing original 2-parameter style...")
    try:
        results_dict = {
            "test_plan": "Original test plan",
            "final_verdict": "SUCCESS",
            "execution_result": "Test completed successfully"
        }
        data_service.save_experiment_log("Original hypothesis test", results_dict)
        print("✅ Original 2-parameter style works!")
    except Exception as e:
        print(f"❌ Original 2-parameter style failed: {e}")
        return False
    
    # Test 2: New 4-parameter style (hypothesis + test_plan + final_verdict + execution_result)
    print("\n2. Testing new 4-parameter style...")
    try:
        data_service.save_experiment_log(
            "New hypothesis test",
            "New test plan",
            "SUCCESS",
            "New test completed successfully"
        )
        print("✅ New 4-parameter style works!")
    except Exception as e:
        print(f"❌ New 4-parameter style failed: {e}")
        return False
    
    # Test 3: Verify data was saved correctly
    print("\n3. Verifying data was saved correctly...")
    try:
        with Session(engine) as session:
            experiments = session.query(ExperimentLog).all()
            
            if len(experiments) != 2:
                print(f"❌ Expected 2 experiments, found {len(experiments)}")
                return False
            
            # Check first experiment (original style)
            exp1 = experiments[0]
            exp1_results = json.loads(exp1.results)
            if exp1.hypothesis != "Original hypothesis test":
                print(f"❌ First experiment hypothesis mismatch")
                return False
            if exp1_results["test_plan"] != "Original test plan":
                print(f"❌ First experiment results mismatch")
                return False
            
            # Check second experiment (new style)
            exp2 = experiments[1]
            exp2_results = json.loads(exp2.results)
            if exp2.hypothesis != "New hypothesis test":
                print(f"❌ Second experiment hypothesis mismatch")
                return False
            if exp2_results["test_plan"] != "New test plan":
                print(f"❌ Second experiment results mismatch")
                return False
            
            print("✅ Data verification successful!")
            print(f"   - Experiment 1: {exp1.hypothesis}")
            print(f"   - Experiment 2: {exp2.hypothesis}")
            
    except Exception as e:
        print(f"❌ Data verification failed: {e}")
        return False
    
    # Test 4: Invalid parameter count should raise TypeError
    print("\n4. Testing invalid parameter count...")
    try:
        data_service.save_experiment_log("Invalid test")  # Only 1 arg (should fail)
        print("❌ Invalid parameter count should have failed!")
        return False
    except TypeError as e:
        print(f"✅ Invalid parameter count correctly rejected: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    print("\n🎉 All tests passed! The save_experiment_log fix is working correctly.")
    return True

if __name__ == "__main__":
    success = test_experiment_log_fix()
    sys.exit(0 if success else 1)
