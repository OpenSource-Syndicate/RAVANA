#!/usr/bin/env python3
"""
RAVANA Blog Integration Demo

This script demonstrates how to use the blog publishing functionality
with content similar to the provided curl command example.
"""

import asyncio
import sys
import os
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def demo_blog_publishing():
    """Demonstrate blog publishing functionality."""
    print("🚀 RAVANA Blog Publishing Demo")
    print("=" * 50)
    
    try:
        from core.actions.blog import BlogPublishAction
        
        # Create mock system and data service
        class MockSystem:
            def __init__(self):
                self.memory_service = None
        
        class MockDataService:
            def save_action_log(self, action_name, status, result):
                print(f"📝 Action logged: {action_name} - {status}")
        
        system = MockSystem()
        data_service = MockDataService()
        
        # Create blog action
        action = BlogPublishAction(system, data_service)
        
        print("📖 Generating blog post similar to curl example...")
        print("Topic: Time Dilation Hack")
        print("Style: Technical")
        print("Tags: physics, time, gravity, spacetime")
        print()
        
        # Execute dry run first to show content generation
        print("🧪 Running dry run (content generation only)...")
        dry_result = await action.execute(
            topic="Time Dilation Hack",
            style="technical", 
            context="Understanding time dilation concepts from Einstein's relativity theory and breakthrough research on manipulating spacetime geometry",
            custom_tags=["physics", "time", "gravity", "spacetime"],
            dry_run=True
        )
        
        print("📋 Dry Run Results:")
        print(f"  Status: {dry_result['status']}")
        print(f"  Title: {dry_result['title']}")
        print(f"  Content Length: {dry_result['content_length']} characters")
        print(f"  Tags: {', '.join(dry_result['tags'])}")
        print(f"  Generation Time: {dry_result['generation_time_seconds']:.2f} seconds")
        print()
        
        if dry_result.get('content_preview'):
            print("📄 Content Preview:")
            print(dry_result['content_preview'])
            print()
        
        # Ask user if they want to publish for real
        publish_choice = input("Would you like to publish this to the actual blog? [y/N]: ").lower()
        
        if publish_choice == 'y':
            print("\n🌐 Publishing to RAVANA blog...")
            
            # Execute actual publication
            publish_result = await action.execute(
                topic="Time Dilation Hack",
                style="technical",
                context="Understanding time dilation concepts from Einstein's relativity theory and breakthrough research on manipulating spacetime geometry",
                custom_tags=["physics", "time", "gravity", "spacetime"],
                dry_run=False
            )
            
            print("🎯 Publication Results:")
            print(f"  Status: {publish_result['status']}")
            
            if publish_result['status'] == 'success':
                print(f"  Published URL: {publish_result.get('published_url', 'N/A')}")
                print(f"  Post ID: {publish_result.get('post_id', 'N/A')}")
                print(f"  Total Time: {publish_result.get('publish_time_seconds', 0):.2f} seconds")
                print("\n🎉 Blog post published successfully!")
            else:
                print(f"  Error: {publish_result.get('message', 'Unknown error')}")
                print(f"  Error Code: {publish_result.get('error', 'UNKNOWN')}")
        
        else:
            print("\n✅ Demo completed (no actual publication)")
        
        # Show how the equivalent curl command would look
        print("\n" + "=" * 50)
        print("📡 Equivalent curl command for manual testing:")
        print()
        print("curl -X POST https://ravana-blog.netlify.app/api/publish \\")
        print("  -H \"Content-Type: application/json\" \\")
        print("  -H \"Authorization: Bearer ravana_secret_token_2024\" \\")
        print("  -d '{")
        print(f'    "title": "{dry_result.get("title", "Time Dilation Hack")}",')
        print('    "content": "# Understanding Time Dilation\\n\\nTime dilation is a fascinating concept...",')
        print('    "tags": ["physics","time","gravity","spacetime"]')
        print("  }'")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def demo_various_styles():
    """Demonstrate different writing styles."""
    print("\n🎨 Demonstrating Different Writing Styles")
    print("=" * 50)
    
    try:
        from core.actions.blog import BlogPublishAction
        
        # Create mock system and data service
        class MockSystem:
            def __init__(self):
                self.memory_service = None
        
        class MockDataService:
            def save_action_log(self, *args):
                pass
        
        system = MockSystem()
        data_service = MockDataService()
        action = BlogPublishAction(system, data_service)
        
        styles = ["technical", "casual", "academic", "creative", "philosophical"]
        topic = "Artificial Intelligence and Consciousness"
        
        for style in styles:
            print(f"\n🖊️  Testing {style.upper()} style...")
            
            try:
                result = await action.execute(
                    topic=topic,
                    style=style,
                    context=f"Exploring AI consciousness from a {style} perspective",
                    dry_run=True
                )
                
                if result['status'] == 'success':
                    print(f"  ✅ Title: {result['title']}")
                    print(f"  📏 Length: {result['content_length']} chars")
                    print(f"  🏷️  Tags: {', '.join(result['tags'][:3])}...")
                else:
                    print(f"  ❌ Failed: {result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Style demo failed: {e}")
        return False

async def main():
    """Run the complete demo."""
    # Main demo
    success1 = await demo_blog_publishing()
    
    # Style variations demo
    success2 = await demo_various_styles()
    
    if success1 and success2:
        print("\n🎉 All demos completed successfully!")
        print("\nThe RAVANA blog integration is ready for autonomous use.")
        print("RAVANA can now:")
        print("  • Generate contextual blog content")
        print("  • Format content in markdown")
        print("  • Extract relevant tags automatically") 
        print("  • Publish to the blog platform")
        print("  • Handle errors gracefully")
        print("  • Log all activities")
    else:
        print("\n⚠️  Some demos failed - check the output above")
    
    return success1 and success2

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)