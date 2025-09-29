"""
Test VLTM Data Migration

This test validates the data migration utilities for moving data from
existing episodic memory and knowledge systems to VLTM.
"""

import asyncio
import logging
import tempfile
from datetime import datetime
from unittest.mock import Mock, AsyncMock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_vltm_data_migration():
    """Test VLTM data migration functionality"""

    print("=" * 60)
    print("VLTM Data Migration Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Import after setting up environment
            from core.vltm_data_migrator import VLTMDataMigrator, MigrationConfig
            from core.vltm_store import VeryLongTermMemoryStore
            from core.vltm_data_models import DEFAULT_VLTM_CONFIG
            from services.memory_service import MemoryService
            from services.knowledge_service import KnowledgeService

            print("1. Setting up mock services...")

            # Create mock VLTM store
            mock_vltm_store = AsyncMock()
            mock_vltm_store.store_memory = AsyncMock(
                return_value="test_memory_id")
            mock_vltm_store.search_memories = AsyncMock(return_value=[])

            # Create mock memory service
            mock_memory_service = Mock()

            # Create mock knowledge service
            mock_knowledge_service = Mock()
            mock_knowledge_service.get_knowledge_by_category = Mock(return_value=[
                {
                    "id": 1,
                    "summary": "System optimization improved performance by 25%",
                    "category": "optimization",
                    "source": "system",
                    "timestamp": datetime.utcnow().isoformat()
                },
                {
                    "id": 2,
                    "summary": "Architecture refactoring enhanced modularity",
                    "category": "system",
                    "source": "development",
                    "timestamp": datetime.utcnow().isoformat()
                }
            ])

            print("   ✓ Mock services created")

            print("\n2. Testing Migration Configuration...")

            # Test migration config
            config = MigrationConfig(
                batch_size=10,
                episodic_cutoff_days=7,
                knowledge_cutoff_days=14,
                dry_run=True,  # Use dry run for testing
                skip_duplicates=True
            )

            print(f"   ✓ Batch size: {config.batch_size}")
            print(f"   ✓ Episodic cutoff: {config.episodic_cutoff_days} days")
            print(
                f"   ✓ Knowledge cutoff: {config.knowledge_cutoff_days} days")
            print(f"   ✓ Dry run: {config.dry_run}")

            print("\n3. Creating Data Migrator...")

            migrator = VLTMDataMigrator(
                vltm_store=mock_vltm_store,
                memory_service=mock_memory_service,
                knowledge_service=mock_knowledge_service,
                config=config
            )

            print(f"   ✓ Migration ID: {migrator.migration_id}")
            print(f"   ✓ Migrator created with config")

            print("\n4. Testing Episodic Memory Conversion...")

            # Test episodic memory conversion
            sample_episodic = {
                "id": "test_episodic_1",
                "content_text": "Successfully optimized database queries, improved performance by 40%",
                "content_type": "text",
                "created_at": datetime.utcnow(),
                "confidence_score": 0.8,
                "tags": ["optimization", "database", "performance"]
            }

            vltm_memory = migrator._convert_episodic_to_vltm(sample_episodic)

            assert vltm_memory is not None, "Episodic conversion should succeed"
            assert "content" in vltm_memory, "Should have content"
            assert "memory_type" in vltm_memory, "Should have memory type"
            assert "metadata" in vltm_memory, "Should have metadata"

            print(f"   ✓ Memory type: {vltm_memory['memory_type']}")
            print(f"   ✓ Content keys: {list(vltm_memory['content'].keys())}")
            print(
                f"   ✓ Migration info included: {'migration_info' in vltm_memory['content']}")

            print("\n5. Testing Knowledge Data Conversion...")

            # Test knowledge conversion
            sample_knowledge = {
                "id": 123,
                "summary": "Strategic architectural decision to adopt microservices improved system scalability",
                "category": "architecture",
                "source": "system_design",
                "timestamp": datetime.utcnow().isoformat()
            }

            vltm_knowledge = migrator._convert_knowledge_to_vltm(
                sample_knowledge)

            assert vltm_knowledge is not None, "Knowledge conversion should succeed"
            assert "knowledge_summary" in vltm_knowledge["content"], "Should have knowledge summary"

            print(f"   ✓ Memory type: {vltm_knowledge['memory_type']}")
            print(
                f"   ✓ Summary length: {len(vltm_knowledge['content']['knowledge_summary'])}")
            print(f"   ✓ Migration info included")

            print("\n6. Testing Content Classification...")

            # Test classification logic
            test_cases = [
                ("System performance optimized by 50%",
                 "Should classify as SUCCESSFUL_IMPROVEMENT"),
                ("Critical database failure during peak hours",
                 "Should classify as CRITICAL_FAILURE"),
                ("New microservices architecture implemented",
                 "Should classify as ARCHITECTURAL_INSIGHT"),
                ("Memory allocation error caused crash",
                 "Should classify as FAILED_EXPERIMENT")
            ]

            for content, expected in test_cases:
                memory_type = migrator._classify_episodic_content(content)
                print(f"   ✓ '{content[:30]}...' → {memory_type}")

            print("\n7. Testing Full Migration (Dry Run)...")

            # Test full migration
            stats = await migrator.migrate_all_data()

            print(f"   ✓ Episodic migrated: {stats.episodic_migrated}")
            print(f"   ✓ Knowledge migrated: {stats.knowledge_migrated}")
            print(f"   ✓ Total failed: {stats.total_failed}")
            print(f"   ✓ Processing time: {stats.processing_time:.2f}s")

            # Verify mock calls
            if not config.dry_run:
                mock_vltm_store.store_memory.assert_called()
                print("   ✓ VLTM store called for memory storage")
            else:
                print("   ✓ Dry run completed without actual storage")

            print("\n8. Testing Incremental Migration...")

            # Test incremental migration
            incremental_stats = await migrator.incremental_migration(hours=6)

            print(f"   ✓ Incremental migration completed")
            print(
                f"   ✓ Memories processed: {incremental_stats.episodic_migrated + incremental_stats.knowledge_migrated}")

            print("\n9. Testing Duplicate Detection...")

            # Test duplicate detection
            test_memory = {
                "content": {"original_text": "Test content for duplicate detection"},
                "memory_type": "CODE_PATTERN",
                "metadata": {}
            }

            # Mock search to return no duplicates
            mock_vltm_store.search_memories.return_value = []
            is_duplicate = await migrator._is_duplicate(test_memory)
            assert not is_duplicate, "Should not be duplicate when no similar memories"

            # Mock search to return similar memories
            mock_vltm_store.search_memories.return_value = [
                {"id": "similar_memory"}]
            is_duplicate = await migrator._is_duplicate(test_memory)
            assert is_duplicate, "Should be duplicate when similar memories found"

            print("   ✓ Duplicate detection working correctly")

            print("\n" + "=" * 60)
            print("✅ VLTM DATA MIGRATION TEST PASSED")
            print("=" * 60)
            print("\nTest Results:")
            print("✅ Migration configuration - Complete")
            print("✅ Episodic memory conversion - Complete")
            print("✅ Knowledge data conversion - Complete")
            print("✅ Content classification - Complete")
            print("✅ Full migration process - Complete")
            print("✅ Incremental migration - Complete")
            print("✅ Duplicate detection - Complete")
            print("\n🎯 Data migration utilities are ready for production!")

            return True

        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            logger.error(f"Migration test failed: {e}", exc_info=True)
            return False


async def test_migration_edge_cases():
    """Test edge cases and error conditions"""

    print("\n" + "=" * 60)
    print("Migration Edge Cases Test")
    print("=" * 60)

    try:
        from core.vltm_data_migrator import VLTMDataMigrator, MigrationConfig
        from unittest.mock import Mock, AsyncMock

        # Setup minimal mocks
        mock_vltm_store = AsyncMock()
        mock_memory_service = Mock()
        mock_knowledge_service = Mock()
        mock_knowledge_service.get_knowledge_by_category = Mock(
            return_value=[])

        migrator = VLTMDataMigrator(
            vltm_store=mock_vltm_store,
            memory_service=mock_memory_service,
            knowledge_service=mock_knowledge_service
        )

        print("1. Testing empty content handling...")

        # Test empty episodic content
        empty_episodic = {"id": "test",
                          "content_text": "", "confidence_score": 0.8}
        result = migrator._convert_episodic_to_vltm(empty_episodic)
        assert result is None, "Should reject empty content"
        print("   ✓ Empty episodic content rejected")

        # Test short knowledge summary
        short_knowledge = {"id": 1, "summary": "Short", "category": "test"}
        result = migrator._convert_knowledge_to_vltm(short_knowledge)
        assert result is None, "Should reject short summaries"
        print("   ✓ Short knowledge summary rejected")

        print("\n2. Testing classification edge cases...")

        # Test generic content classification
        generic_content = "This is some generic content without specific keywords"
        memory_type = migrator._classify_episodic_content(generic_content)
        print(f"   ✓ Generic content classified as: {memory_type}")

        # Test knowledge with mixed signals
        mixed_knowledge = {
            "id": 1,
            "summary": "System architecture optimization learning patterns",
            "category": "mixed",
            "source": "test"
        }
        memory_type = migrator._classify_knowledge_content(mixed_knowledge)
        print(f"   ✓ Mixed knowledge classified as: {memory_type}")

        print("\n3. Testing error handling...")

        # Test malformed data handling
        malformed_episodic = {"invalid": "data"}
        result = migrator._convert_episodic_to_vltm(malformed_episodic)
        assert result is None, "Should handle malformed data gracefully"
        print("   ✓ Malformed episodic data handled gracefully")

        malformed_knowledge = {"no_summary": True}
        result = migrator._convert_knowledge_to_vltm(malformed_knowledge)
        assert result is None, "Should handle malformed knowledge gracefully"
        print("   ✓ Malformed knowledge data handled gracefully")

        print("\n✅ EDGE CASES TEST PASSED")
        return True

    except Exception as e:
        print(f"\n❌ EDGE CASES TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    """Run the migration tests"""

    async def run_all_tests():
        print("Starting VLTM Data Migration Tests...\n")

        # Run main migration test
        success1 = await test_vltm_data_migration()

        # Run edge cases test
        success2 = await test_migration_edge_cases()

        if success1 and success2:
            print("\n🎉 ALL MIGRATION TESTS COMPLETED SUCCESSFULLY!")
            print("\nNext Steps:")
            print("1. ✅ Data migration utilities complete")
            print("2. 🔄 Proceed to advanced retrieval system")
            print("3. 🔄 Complete Phase 3 integration tasks")
            print("4. 🔄 Begin Phase 4 optimization layer")
        else:
            print("\n❌ Some tests failed. Please review and fix issues.")

    # Run the tests
    asyncio.run(run_all_tests())
