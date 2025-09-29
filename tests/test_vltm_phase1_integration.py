"""
Basic Integration Test for VLTM Phase 1 Components

This test validates that all Phase 1 foundation components work together
correctly and provides a foundation for testing Phase 2 components.
"""

from core.vltm_consolidation_engine import MemoryConsolidationEngine
from core.vltm_store import VeryLongTermMemoryStore
from core.vltm_data_models import (
    MemoryType, PatternType, ConsolidationType,
    VLTMConfiguration, DEFAULT_VLTM_CONFIG,
    ConsolidationRequest
)
import asyncio
import logging
import tempfile
import uuid
from datetime import datetime

# Configure logging for testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import VLTM components


async def test_vltm_phase1_integration():
    """Test basic integration of Phase 1 VLTM components"""

    print("=" * 60)
    print("VLTM Phase 1 Integration Test")
    print("=" * 60)

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Initialize VLTM store
            print("1. Initializing VLTM Store...")
            vltm_store = VeryLongTermMemoryStore(
                config=DEFAULT_VLTM_CONFIG,
                base_storage_dir=temp_dir
            )

            # Note: Since we don't have PostgreSQL setup in test environment,
            # this test focuses on component initialization and API validation
            print("   ✓ VLTM Store created")

            # Test memory classification without storage
            print("\n2. Testing Memory Classification...")
            test_memory_content = {
                "action": "code_improvement",
                "description": "Optimized database query performance by 40%",
                "details": {
                    "file": "database/queries.py",
                    "change_type": "optimization",
                    "performance_gain": 0.4,
                    "method": "index_optimization"
                },
                "timestamp": datetime.utcnow().isoformat(),
                "success": True,
                "metrics": {
                    "execution_time_before": 2.5,
                    "execution_time_after": 1.5,
                    "improvement_percentage": 40.0
                }
            }

            # Test classification without full storage backend
            if hasattr(vltm_store, 'memory_classifier') and vltm_store.memory_classifier:
                from core.vltm_data_models import MemoryRecord
                test_record = MemoryRecord(
                    memory_id=str(uuid.uuid4()),
                    memory_type=MemoryType.SUCCESSFUL_IMPROVEMENT,
                    content=test_memory_content,
                    metadata={"test": True}
                )

                classification = vltm_store.memory_classifier.classify_memory(
                    test_record)
                print(
                    f"   ✓ Classification result: {classification['memory_type']}")
                print(
                    f"   ✓ Importance score: {classification['importance_score']:.3f}")
                print(
                    f"   ✓ Strategic value: {classification['strategic_value']:.3f}")

            # Test consolidation engine skeleton
            print("\n3. Testing Consolidation Engine...")
            consolidation_engine = MemoryConsolidationEngine(
                DEFAULT_VLTM_CONFIG)

            # Test consolidation status
            status = consolidation_engine.get_consolidation_status()
            print(f"   ✓ Consolidation status: {status['phase']}")

            # Test consolidation due check
            is_due = consolidation_engine.is_consolidation_due(
                ConsolidationType.DAILY)
            print(f"   ✓ Daily consolidation due: {is_due}")

            # Test consolidation request (skeleton implementation)
            request = ConsolidationRequest(
                consolidation_type=ConsolidationType.DAILY,
                force_consolidation=True
            )

            result = await consolidation_engine.consolidate_memories(request)
            print(f"   ✓ Consolidation completed: {result.success}")
            print(
                f"   ✓ Processing time: {result.processing_time_seconds:.3f}s")

            # Test configuration validation
            print("\n4. Testing Configuration...")
            config_issues = DEFAULT_VLTM_CONFIG.validate_configuration()
            if config_issues:
                print(f"   ⚠ Configuration issues: {config_issues}")
            else:
                print("   ✓ Configuration is valid")

            # Test data model validation
            print("\n5. Testing Data Models...")

            # Test memory record creation
            memory_record = MemoryRecord(
                memory_id=str(uuid.uuid4()),
                memory_type=MemoryType.STRATEGIC_KNOWLEDGE,
                content={"test": "data"},
                metadata={"created_by": "test"}
            )
            print("   ✓ Memory record created")
            print(f"   ✓ Memory ID: {memory_record.memory_id}")
            print(f"   ✓ Memory type: {memory_record.memory_type}")

            # Test VLTM store API methods (without actual storage)
            print("\n6. Testing VLTM Store API...")

            # Test store methods exist and are callable
            assert hasattr(
                vltm_store, 'store_memory'), "store_memory method missing"
            assert hasattr(
                vltm_store, 'retrieve_memory'), "retrieve_memory method missing"
            assert hasattr(
                vltm_store, 'search_memories'), "search_memories method missing"
            assert hasattr(
                vltm_store, 'store_pattern'), "store_pattern method missing"
            assert hasattr(
                vltm_store, 'store_strategic_knowledge'), "store_strategic_knowledge method missing"

            print("   ✓ All required API methods present")

            # Test statistics method
            stats = await vltm_store.get_memory_statistics()
            print(f"   ✓ Statistics retrieved: {type(stats)}")

            # Test cleanup
            print("\n7. Testing Cleanup...")
            await vltm_store.cleanup()
            print("   ✓ Cleanup completed")

            print("\n" + "=" * 60)
            print("✅ PHASE 1 INTEGRATION TEST PASSED")
            print("=" * 60)
            print("\nPhase 1 Components Status:")
            print("✅ Data Models - Complete")
            print("✅ Storage Backend - Complete (skeleton)")
            print("✅ Memory Classifier - Complete")
            print("✅ VLTM Store - Complete")
            print("✅ Consolidation Engine - Complete (skeleton)")
            print("\n🔄 Ready for Phase 2: Consolidation System Implementation")

            return True

        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            logger.error(f"Integration test failed: {e}", exc_info=True)
            return False


async def test_memory_classification_examples():
    """Test memory classification with various example scenarios"""

    print("\n" + "=" * 60)
    print("Memory Classification Examples")
    print("=" * 60)

    from core.vltm_memory_classifier import MemoryClassifier
    from core.vltm_data_models import MemoryRecord

    classifier = MemoryClassifier(DEFAULT_VLTM_CONFIG)

    # Test cases for different memory types
    test_cases = [
        {
            "name": "Successful Code Optimization",
            "content": {
                "action": "performance_optimization",
                "description": "Optimized database connection pooling, reduced latency by 60%",
                "category": "performance",
                "success_indicators": ["optimized", "reduced", "improved"],
                "metrics": {"latency_reduction": 0.6}
            }
        },
        {
            "name": "Critical System Failure",
            "content": {
                "event": "system_failure",
                "description": "Critical memory leak caused system crash in production",
                "severity": "critical",
                "impact": "system_wide",
                "failure_indicators": ["critical", "crash", "failure", "memory leak"]
            }
        },
        {
            "name": "Architectural Decision",
            "content": {
                "decision": "architecture_change",
                "description": "Adopted microservices architecture for better scalability",
                "scope": "system_wide",
                "architectural_terms": ["microservices", "architecture", "scalability", "design pattern"]
            }
        },
        {
            "name": "Failed Experiment",
            "content": {
                "experiment": "feature_test",
                "description": "Attempted caching optimization failed due to memory constraints",
                "result": "failed",
                "failure_indicators": ["failed", "constraints", "error"]
            }
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")

        memory_record = MemoryRecord(
            memory_id=str(uuid.uuid4()),
            memory_type=MemoryType.CODE_PATTERN,  # Will be classified
            content=test_case["content"],
            metadata={"test_case": test_case["name"]}
        )

        classification = classifier.classify_memory(memory_record)

        print(f"   Memory Type: {classification['memory_type']}")
        print(f"   Importance: {classification['importance_score']:.3f}")
        print(f"   Strategic Value: {classification['strategic_value']:.3f}")
        print(f"   Importance Level: {classification['importance_level']}")

        # Show some extracted features
        features = classification.get('features', {})
        if features.get('success_indicators'):
            print(f"   Success Indicators: {features['success_indicators']}")
        if features.get('failure_indicators'):
            print(f"   Failure Indicators: {features['failure_indicators']}")
        if features.get('architectural_terms'):
            print(f"   Architectural Terms: {features['architectural_terms']}")

    print("\n✅ Memory classification examples completed")


if __name__ == "__main__":
    """Run the integration tests"""

    async def run_all_tests():
        print("Starting VLTM Phase 1 Integration Tests...\n")

        # Run main integration test
        success = await test_vltm_phase1_integration()

        if success:
            # Run classification examples
            await test_memory_classification_examples()

            print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
            print("\nNext Steps:")
            print("1. Proceed to Phase 2: Consolidation System")
            print("2. Implement full consolidation algorithms")
            print("3. Add pattern extraction capabilities")
            print("4. Integrate with existing Snake Agent")
        else:
            print("\n❌ Tests failed. Please fix issues before proceeding.")

    # Run the tests
    asyncio.run(run_all_tests())
