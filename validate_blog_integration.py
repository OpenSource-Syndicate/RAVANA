#!/usr/bin/env python3
"""
Simple validation script for RAVANA Blog Integration
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Validate blog integration components."""
    print("🔍 Validating RAVANA Blog Integration...")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from core.actions.blog import BlogPublishAction
        from core.actions.blog_api import BlogAPIInterface  
        from core.actions.blog_content_generator import BlogContentGenerator
        from core.actions.registry import ActionRegistry
        from core.config import Config
        print("✓ All imports successful")
        
        # Test mock instantiation
        print("\n🏗️  Testing component creation...")
        
        class MockSystem:
            def __init__(self):
                self.memory_service = None
        
        class MockDataService:
            def save_action_log(self, *args):
                pass
        
        system = MockSystem()
        data_service = MockDataService()
        
        # Test API interface
        api = BlogAPIInterface()
        print(f"✓ BlogAPIInterface created")
        
        # Test content generator
        generator = BlogContentGenerator()
        print(f"✓ BlogContentGenerator created")
        
        # Test action creation
        action = BlogPublishAction(system, data_service)
        print(f"✓ BlogPublishAction created: '{action.name}'")
        print(f"  - Description: {action.description}")
        print(f"  - Parameters: {len(action.parameters)} defined")
        
        # Test registry integration
        print("\n🔗 Testing registry integration...")
        registry = ActionRegistry(system, data_service)
        
        try:
            blog_action = registry.get_action('publish_blog_post')
            print(f"✓ Action registered in registry: '{blog_action.name}'")
        except ValueError as e:
            print(f"❌ Action not found in registry: {e}")
            return False
        
        # Test configuration
        print(f"\n⚙️  Testing configuration...")
        print(f"✓ Blog enabled: {Config.BLOG_ENABLED}")
        print(f"✓ API URL: {Config.BLOG_API_URL}")
        print(f"✓ Auth token configured: {'Yes' if Config.BLOG_AUTH_TOKEN else 'No'}")
        print(f"✓ Default style: {Config.BLOG_DEFAULT_STYLE}")
        print(f"✓ Available styles: {', '.join(Config.BLOG_CONTENT_STYLES)}")
        
        # Test action parameters validation
        print(f"\n📋 Testing parameter schema...")
        for param in action.parameters:
            print(f"  - {param['name']} ({param['type']}): {param['description']}")
        
        print("\n" + "=" * 50)
        print("🎉 Blog integration validation PASSED!")
        print("The blog publishing functionality is ready for use.")
        print("\nTo use the blog action, call:")
        print("  publish_blog_post(topic='Your Topic', style='technical')")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)