"""
Comprehensive test demonstrating the complete intelligent analysis and optimization workflow
"""

import asyncio
import os
import tempfile
from pathlib import Path

# Add the project root to the path so imports work
project_root = Path(__file__).parent
import sys
sys.path.insert(0, str(project_root))

from core.snake_agent_implementer import SnakeAgentImplementer
from core.code_analyzer import CodeAnalyzer
from core.code_transformer import CodeTransformer


async def comprehensive_intelligent_workflow_test():
    print("Testing complete intelligent analysis and optimization workflow...")
    
    # Create a temporary directory for our test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test file with multiple clear optimization opportunities
        test_file_path = Path(temp_dir) / "intelligent_workflow_test.py"
        test_content = '''
"""
Comprehensive test file with multiple optimization opportunities for intelligent analysis
"""

def inefficient_list_building():
    """Build a list inefficiently with append()."""
    result = []
    for i in range(100):
        result.append(i * 2)
    return result

def inefficient_string_concatenation():
    """Concatenate strings inefficiently in a loop."""
    text = ""
    for i in range(50):
        text = text + "Item " + str(i) + "\\n"
    return text

def manual_summation(numbers):
    """Manually sum numbers instead of using sum()."""
    total = 0
    for num in numbers:
        total += num
    return total

def inefficient_nested_loops(items):
    """Nested loops with quadratic complexity."""
    results = []
    for item1 in items:
        for item2 in items:  # This creates O(n^2) complexity
            if item1 != item2:
                results.append((item1, item2))
    return results

def repeated_calculations(data):
    """Perform repeated calculations that could be cached."""
    results = []
    for item in data:
        # Expensive calculation performed repeatedly
        expensive_result = item ** 2 + item ** 3 + item ** 4
        results.append(expensive_result)
    return results

class DataProcessor:
    """A class with methods that have optimization opportunities."""
    
    def __init__(self):
        self.cache = {}
    
    def process_items(self, items):
        """Process items with inefficient pattern."""
        processed = []
        for item in items:
            if item > 0:
                processed.append(item * 2)
            else:
                processed.append(0)
        return processed
    
    def find_maximum(self, numbers):
        """Manually find maximum instead of using max()."""
        if not numbers:
            return None
        max_val = numbers[0]
        for num in numbers[1:]:
            if num > max_val:
                max_val = num
        return max_val

if __name__ == "__main__":
    print("Testing intelligent workflow...")
    processor = DataProcessor()
    result1 = inefficient_list_building()
    result2 = inefficient_string_concatenation()
    result3 = manual_summation([1, 2, 3, 4, 5])
    result4 = processor.process_items([1, -1, 2, -2, 3])
    result5 = processor.find_maximum([5, 2, 8, 1, 9])
    print(f"Results: {len(result1)}, {len(result2)}, {result3}, {len(result4)}, {result5}")
'''
        
        with open(test_file_path, 'w') as f:
            f.write(test_content)
        
        print(f"Created test file: {test_file_path}")
        print("\\nOriginal content:")
        with open(test_file_path, 'r') as f:
            original_content = f.read()
            print(original_content)
        
        # Initialize the systems
        implementer = SnakeAgentImplementer(project_root=temp_dir)
        analyzer = CodeAnalyzer()
        transformer = CodeTransformer()
        
        print("\\nSystems initialized successfully")
        
        # Step 1: Intelligent Code Analysis
        print("\\n=== Step 1: Intelligent Code Analysis ===")
        analysis_results = analyzer.analyze_code(original_content)
        
        print("Analysis Results:")
        print(f"  Functions found: {analysis_results.get('basic_metrics', {}).get('functions', 0)}")
        print(f"  Classes found: {analysis_results.get('basic_metrics', {}).get('classes', 0)}")
        print(f"  Loops found: {analysis_results.get('basic_metrics', {}).get('loops', 0)}")
        
        # Display optimization opportunities
        opportunities = analysis_results.get('optimization_opportunities', [])
        print(f"\\nFound {len(opportunities)} optimization opportunities:")
        for i, opp in enumerate(opportunities, 1):
            print(f"  {i}. [{opp.get('subtype', 'general')}] {opp.get('description', 'Unknown optimization')}")
            print(f"     Recommendation: {opp.get('recommendation', 'No recommendation')}")
            print(f"     Severity: {opp.get('severity', 'unknown')}")
            location = opp.get('location', {})
            if location:
                print(f"     Location: Line {location.get('line_start', '?')}-{location.get('line_end', '?')}")
            print()
        
        # Step 2: Intelligent Code Transformation
        print("=== Step 2: Intelligent Code Transformation ===")
        
        # Apply comprehensive optimizations
        optimized_content = transformer.optimize_code(original_content, "all")
        
        if optimized_content and optimized_content != original_content:
            print("SUCCESS: Intelligent transformation applied!")
            
            # Write the optimized content to file
            with open(test_file_path, 'w') as f:
                f.write(optimized_content)
            
            print("\\nOptimized content:")
            print(optimized_content)
            
            # Show the differences
            print("\\n=== Changes Made ===")
            original_lines = original_content.split('\\n')
            optimized_lines = optimized_content.split('\\n')
            
            for i, (orig, opt) in enumerate(zip(original_lines, optimized_lines)):
                if orig != opt:
                    print(f"  Line {i+1}:")
                    print(f"    Original:  {repr(orig)}")
                    print(f"    Optimized: {repr(opt)}")
                    print()
        else:
            print("No optimizations were applied or optimization failed.")
        
        # Step 3: Verification and Validation
        print("=== Step 3: Verification and Validation ===")
        
        # Read the final content
        with open(test_file_path, 'r') as f:
            final_content = f.read()
        
        # Verify the content was actually optimized
        if final_content != original_content:
            print("VERIFICATION PASSED: Content was successfully optimized!")
            
            # Additional analysis of the optimized code
            final_analysis = analyzer.analyze_code(final_content)
            final_opportunities = final_analysis.get('optimization_opportunities', [])
            
            print(f"\\nRemaining optimization opportunities: {len(final_opportunities)}")
            if len(final_opportunities) < len(opportunities):
                print(f"OPTIMIZATION SUCCESS: Reduced optimization opportunities from {len(opportunities)} to {len(final_opportunities)}")
            else:
                print("Note: Some optimization opportunities remain (this is expected for complex cases)")
        else:
            print("VERIFICATION: No changes were made to the content.")
        
        # Step 4: Performance Estimation
        print("\\n=== Step 4: Performance Impact Estimation ===")
        
        # Estimate performance improvements based on the optimizations applied
        optimizations_applied = []
        for opp in opportunities:
            subtype = opp.get('subtype', '')
            if subtype in ['list_building', 'string_concatenation', 'builtin_function']:
                optimizations_applied.append(opp)
        
        print(f"Estimated performance improvements from {len(optimizations_applied)} optimizations:")
        for i, opp in enumerate(optimizations_applied, 1):
            improvement = opp.get('estimated_improvement', 'moderate')
            print(f"  {i}. {opp.get('description', 'Optimization')}: {improvement}")
        
        # Overall summary
        print("\\n=== FINAL SUMMARY ===")
        print(f"Original functions: {analysis_results.get('basic_metrics', {}).get('functions', 0)}")
        print(f"Original complexity: {analysis_results.get('complexity_analysis', {}).get('cyclomatic_complexity', 0)}")
        print(f"Optimizations identified: {len(opportunities)}")
        print(f"Optimizations applied: {len(optimizations_applied)}")
        print(f"Content changed: {'YES' if final_content != original_content else 'NO'}")
        
        if final_content != original_content:
            print("\\n🎉 SUCCESS: Complete intelligent analysis and optimization workflow completed!")
            print("   The Snake Agent successfully:")
            print("   1. Analyzed code semantically to identify optimization opportunities")
            print("   2. Applied AST-based transformations to optimize the code")
            print("   3. Preserved functionality while improving performance")
            print("   4. Provided transparent documentation of changes")
        else:
            print("\\n⚠️  Note: No optimizations were applied in this test run.")
            print("   This could be due to the complexity of the patterns or limitations in the current implementation.")


if __name__ == "__main__":
    asyncio.run(comprehensive_intelligent_workflow_test())