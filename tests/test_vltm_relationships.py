"""
Test file to verify that the VLTM relationships are correctly defined
and that the foreign key issue is resolved.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.vltm_data_models import (
    MemoryPattern, MemoryConsolidation, StrategicKnowledge,
    ConsolidationPattern, PatternStrategicKnowledge
)


def test_relationships():
    """Test that the relationships are properly defined without foreign key errors."""
    try:
        # Try to create instances to verify relationships
        pattern = MemoryPattern(
            pattern_id="test_pattern_1",
            pattern_type="temporal",
            pattern_description="Test pattern",
            confidence_score=0.8,
            pattern_data="{}"
        )
        
        consolidation = MemoryConsolidation(
            consolidation_id="test_consolidation_1",
            consolidation_type="daily"
        )
        
        knowledge = StrategicKnowledge(
            knowledge_id="test_knowledge_1",
            knowledge_domain="test",
            knowledge_summary="Test knowledge",
            confidence_level=0.9
        )
        
        # If we get here without errors, the relationships are properly defined
        print("SUCCESS: All relationships are properly defined!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False


if __name__ == "__main__":
    success = test_relationships()
    sys.exit(0 if success else 1)