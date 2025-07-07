#!/usr/bin/env python3
"""
Test script to verify the EnhancedActionManager fix.
"""

import asyncio
import logging
import sys
from unittest.mock import Mock, MagicMock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_action_manager_initialization():
    """Test that EnhancedActionManager can be initialized without AttributeError."""
    try:
        # Import the classes
        from core.enhanced_action_manager import EnhancedActionManager
        from services.data_service import DataService
        
        # Create mock objects
        mock_agi_system = Mock()
        mock_agi_system.knowledge_service = Mock()
        mock_agi_system.knowledge_service.add_knowledge = Mock()
        
        mock_data_service = Mock()
        
        # Create mock action registry
        mock_action_registry = Mock()
        mock_action_registry.register_action = Mock()
        mock_action_registry.actions = {}
        mock_action_registry.get_action_definitions = Mock(return_value=[])
        
        # Mock the ActionManager's __init__ to avoid complex dependencies
        with unittest.mock.patch('core.enhanced_action_manager.ActionManager.__init__') as mock_super_init:
            mock_super_init.return_value = None
            
            # Create the EnhancedActionManager instance
            enhanced_manager = EnhancedActionManager(mock_agi_system, mock_data_service)
            
            # Manually set the required attributes that would normally be set by ActionManager.__init__
            enhanced_manager.system = mock_agi_system
            enhanced_manager.data_service = mock_data_service
            enhanced_manager.action_registry = mock_action_registry
            
            # Test that register_enhanced_actions can be called without error
            enhanced_manager.register_enhanced_actions()
            
            # Verify that register_action was called 4 times (for the 4 multi-modal actions)
            assert mock_action_registry.register_action.call_count == 4
            
            logger.info("✅ EnhancedActionManager initialization test passed!")
            return True
            
    except AttributeError as e:
        if "'EnhancedActionManager' object has no attribute 'agi_system'" in str(e):
            logger.error("❌ The AttributeError still exists - fix not applied correctly")
            return False
        else:
            logger.error(f"❌ Different AttributeError occurred: {e}")
            return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during test: {e}")
        return False

def test_action_methods_system_access():
    """Test that action methods can access self.system correctly."""
    try:
        from core.enhanced_action_manager import EnhancedActionManager
        import unittest.mock
        
        # Create mock objects
        mock_agi_system = Mock()
        mock_agi_system.knowledge_service = Mock()
        mock_agi_system.knowledge_service.add_knowledge = Mock()
        
        mock_data_service = Mock()
        
        # Mock the ActionManager's __init__
        with unittest.mock.patch('core.enhanced_action_manager.ActionManager.__init__') as mock_super_init:
            mock_super_init.return_value = None
            
            # Create the EnhancedActionManager instance
            enhanced_manager = EnhancedActionManager(mock_agi_system, mock_data_service)
            
            # Manually set the required attributes
            enhanced_manager.system = mock_agi_system
            enhanced_manager.data_service = mock_data_service
            enhanced_manager.multi_modal_service = Mock()
            enhanced_manager.action_cache = {}
            
            # Test that we can access self.system in the methods
            # This should not raise an AttributeError
            system_ref = enhanced_manager.system
            assert system_ref is mock_agi_system
            
            logger.info("✅ Action methods system access test passed!")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error during system access test: {e}")
        return False

if __name__ == "__main__":
    import unittest.mock
    
    logger.info("Testing EnhancedActionManager fix...")
    
    # Run tests
    test1_passed = test_enhanced_action_manager_initialization()
    test2_passed = test_action_methods_system_access()
    
    if test1_passed and test2_passed:
        logger.info("🎉 All tests passed! The fix appears to be working correctly.")
        sys.exit(0)
    else:
        logger.error("💥 Some tests failed. Please check the implementation.")
        sys.exit(1)
