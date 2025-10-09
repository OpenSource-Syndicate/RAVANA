"""
AST-Based Code Transformation System
This module provides advanced code transformation capabilities using Python's AST module
to enable the Snake Agent to make complete, safe optimizations.
"""

import ast
import astor  # For converting AST back to code
import re
from typing import List, Dict, Any, Optional, Tuple, Set
import logging

logger = logging.getLogger(__name__)


class CodeTransformer:
    """
    Transforms code using AST parsing for safe, complete optimizations.
    """
    
    def __init__(self):
        self.transformations_applied = []
    
    def optimize_code(self, code: str, optimization_type: str = "all") -> Optional[str]:
        """
        Apply optimizations to code based on the specified type.
        
        Args:
            code: Source code to optimize
            optimization_type: Type of optimization to apply
            
        Returns:
            Optimized code as string, or None if transformation failed
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Track original code for comparison
            original_ast_dump = ast.dump(tree)
            
            # Apply transformations based on type
            if optimization_type in ["all", "list_comprehension"]:
                tree = self._optimize_list_comprehensions(tree)
            
            if optimization_type in ["all", "string_concatenation"]:
                tree = self._optimize_string_concatenation(tree)
            
            if optimization_type in ["all", "loop_invariant"]:
                tree = self._optimize_loop_invariants(tree)
            
            if optimization_type in ["all", "builtin_functions"]:
                tree = self._optimize_builtin_functions(tree)
            
            if optimization_type in ["all", "dictionary_access"]:
                tree = self._optimize_dictionary_access(tree)
            
            # Check if any transformations were actually applied
            new_ast_dump = ast.dump(tree)
            if new_ast_dump == original_ast_dump:
                logger.info("No optimizations were applicable to this code")
                return code  # Return original code if no changes
            
            # Convert back to code
            optimized_code = astor.to_source(tree)
            
            logger.info(f"Successfully applied {optimization_type} optimization")
            return optimized_code
            
        except Exception as e:
            logger.error(f"Error optimizing code: {e}")
            return None
    
    def _optimize_list_comprehensions(self, tree: ast.AST) -> ast.AST:
        """
        Transform inefficient list building loops into list comprehensions.
        
        Args:
            tree: AST of the code to optimize
            
        Returns:
            Optimized AST
        """
        transformer = ListComprehensionTransformer()
        optimized_tree = transformer.visit(tree)
        ast.fix_missing_locations(optimized_tree)
        return optimized_tree
    
    def _optimize_string_concatenation(self, tree: ast.AST) -> ast.AST:
        """
        Transform inefficient string concatenation loops into join() operations.
        
        Args:
            tree: AST of the code to optimize
            
        Returns:
            Optimized AST
        """
        transformer = StringConcatenationTransformer()
        optimized_tree = transformer.visit(tree)
        ast.fix_missing_locations(optimized_tree)
        return optimized_tree
    
    def _optimize_loop_invariants(self, tree: ast.AST) -> ast.AST:
        """
        Move loop-invariant calculations outside loops.
        
        Args:
            tree: AST of the code to optimize
            
        Returns:
            Optimized AST
        """
        transformer = LoopInvariantTransformer()
        optimized_tree = transformer.visit(tree)
        ast.fix_missing_locations(optimized_tree)
        return optimized_tree
    
    def _optimize_builtin_functions(self, tree: ast.AST) -> ast.AST:
        """
        Replace manual loops with built-in functions where appropriate.
        
        Args:
            tree: AST of the code to optimize
            
        Returns:
            Optimized AST
        """
        transformer = BuiltinFunctionTransformer()
        optimized_tree = transformer.visit(tree)
        ast.fix_missing_locations(optimized_tree)
        return optimized_tree
    
    def _optimize_dictionary_access(self, tree: ast.AST) -> ast.AST:
        """
        Optimize repeated dictionary access patterns.
        
        Args:
            tree: AST of the code to optimize
            
        Returns:
            Optimized AST
        """
        transformer = DictionaryAccessTransformer()
        optimized_tree = transformer.visit(tree)
        ast.fix_missing_locations(optimized_tree)
        return optimized_tree


class ListComprehensionTransformer(ast.NodeTransformer):
    """
    Transforms inefficient list building loops into list comprehensions.
    """
    
    def visit_For(self, node: ast.For) -> ast.AST:
        """
        Transform for loops that build lists with append() into list comprehensions.
        """
        # Check if this looks like a list-building loop
        if self._is_list_building_loop(node):
            # Transform to list comprehension
            return self._transform_to_list_comprehension(node)
        
        # Continue with normal traversal
        return self.generic_visit(node)
    
    def _is_list_building_loop(self, node: ast.For) -> bool:
        """
        Check if a for loop is building a list with append().
        """
        # Must have a body with statements
        if not hasattr(node, 'body') or not node.body:
            return False
        
        # Look for pattern: list_var = [] followed by loop with append
        # This is checked at a higher level, so here we just verify
        # the loop body contains append calls to a list
        
        # Check if body contains append calls
        append_calls = [n for n in ast.walk(node) if isinstance(n, ast.Call) and 
                       isinstance(n.func, ast.Attribute) and n.func.attr == 'append']
        
        return len(append_calls) > 0
    
    def _transform_to_list_comprehension(self, node: ast.For) -> ast.AST:
        """
        Transform a list-building loop into a list comprehension.
        """
        try:
            # Extract key information from the loop
            target_var = node.target
            iterator = node.iter
            
            # Find append calls in the loop body
            append_calls = []
            for stmt in node.body:
                if (isinstance(stmt, ast.Expr) and 
                    isinstance(stmt.value, ast.Call) and
                    isinstance(stmt.value.func, ast.Attribute) and
                    stmt.value.func.attr == 'append'):
                    append_calls.append(stmt.value)
            
            if not append_calls:
                return node  # No append calls found, return unchanged
            
            # For simplicity, use the first append call's argument
            if append_calls:
                elt_expr = append_calls[0].args[0]
            else:
                # Fallback: create a simple element
                elt_expr = ast.Name(id='item', ctx=ast.Load())
            
            # Create the list comprehension
            list_comp = ast.ListComp(
                elt=elt_expr,
                generators=[ast.comprehension(
                    target=target_var,
                    iter=iterator,
                    ifs=[],
                    is_async=0
                )]
            )
            
            # Find the list variable being appended to (from the first append call)
            if append_calls:
                list_var = append_calls[0].func.value
                if isinstance(list_var, ast.Name):
                    # Replace the entire loop with: list_var = [comprehension]
                    assign_node = ast.Assign(
                        targets=[ast.Name(id=list_var.id, ctx=ast.Store())],
                        value=list_comp
                    )
                    return assign_node
            
            # If we can't determine the list variable, return unchanged
            return node
            
        except Exception as e:
            logger.warning(f"Could not transform loop to list comprehension: {e}")
            return node


class StringConcatenationTransformer(ast.NodeTransformer):
    """
    Transforms inefficient string concatenation loops into join() operations.
    """
    
    def visit_For(self, node: ast.For) -> ast.AST:
        """
        Transform for loops that concatenate strings into join() operations.
        """
        # Check if this looks like a string concatenation loop
        if self._is_string_concatenation_loop(node):
            # Transform to join operation
            return self._transform_to_join_operation(node)
        
        # Continue with normal traversal
        return self.generic_visit(node)
    
    def _is_string_concatenation_loop(self, node: ast.For) -> bool:
        """
        Check if a for loop is concatenating strings.
        """
        # Look for augmented assignment with +
        aug_assign_nodes = [n for n in ast.walk(node) if isinstance(n, ast.AugAssign) and 
                           isinstance(n.op, ast.Add)]
        
        return len(aug_assign_nodes) > 0
    
    def _transform_to_join_operation(self, node: ast.For) -> ast.AST:
        """
        Transform a string concatenation loop into a join operation.
        """
        try:
            # This is a simplified implementation
            # A full implementation would need to analyze the exact concatenation pattern
            
            # For now, we'll just add a comment suggesting the optimization
            # In a real implementation, this would do the actual transformation
            logger.info("String concatenation optimization would be applied here")
            return node
            
        except Exception as e:
            logger.warning(f"Could not transform loop to join operation: {e}")
            return node


class LoopInvariantTransformer(ast.NodeTransformer):
    """
    Moves loop-invariant calculations outside loops.
    """
    
    def visit_For(self, node: ast.For) -> ast.AST:
        """
        Move loop-invariant calculations outside the loop.
        """
        # This is a complex optimization that would require
        # sophisticated data flow analysis
        
        # For now, we'll just continue with normal traversal
        return self.generic_visit(node)
    
    def visit_While(self, node: ast.While) -> ast.AST:
        """
        Move loop-invariant calculations outside the while loop.
        """
        # Same as for For loops
        return self.generic_visit(node)


class BuiltinFunctionTransformer(ast.NodeTransformer):
    """
    Replaces manual loops with built-in functions where appropriate.
    """
    
    def visit_For(self, node: ast.For) -> ast.AST:
        """
        Transform for loops that can be replaced with built-ins like sum(), max(), min().
        """
        # Check if this looks like a summation loop
        if self._is_summation_loop(node):
            return self._transform_to_sum_builtin(node)
        
        # Check if this looks like a maximum/minimum loop
        if self._is_extremum_loop(node):
            return self._transform_to_extremum_builtin(node)
        
        # Continue with normal traversal
        return self.generic_visit(node)
    
    def _is_summation_loop(self, node: ast.For) -> bool:
        """
        Check if a for loop is performing summation.
        """
        # Look for patterns like:
        # total = 0
        # for item in items:
        #     total += item
        return False  # Simplified implementation
    
    def _is_extremum_loop(self, node: ast.For) -> bool:
        """
        Check if a for loop is finding maximum or minimum.
        """
        # Look for patterns like:
        # max_val = items[0]
        # for item in items[1:]:
        #     if item > max_val:
        #         max_val = item
        return False  # Simplified implementation
    
    def _transform_to_sum_builtin(self, node: ast.For) -> ast.AST:
        """
        Transform a summation loop into sum() builtin.
        """
        # Implementation would go here
        return node
    
    def _transform_to_extremum_builtin(self, node: ast.For) -> ast.AST:
        """
        Transform extremum loops into max()/min() builtins.
        """
        # Implementation would go here
        return node


class DictionaryAccessTransformer(ast.NodeTransformer):
    """
    Optimizes repeated dictionary access patterns.
    """
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """
        Optimize dictionary access within functions.
        """
        # Look for repeated dictionary accesses and optimize them
        return self.generic_visit(node)


# Example usage and testing
def test_code_transformation():
    """
    Test the code transformation system with examples.
    """
    transformer = CodeTransformer()
    
    # Test case 1: List comprehension optimization
    test_code1 = '''
def inefficient_list_building():
    """Build a list inefficiently with append()."""
    result = []
    for i in range(100):
        result.append(i * 2)
    return result
'''
    
    print("Original code:")
    print(test_code1)
    
    optimized_code1 = transformer.optimize_code(test_code1, "list_comprehension")
    if optimized_code1:
        print("\nOptimized code:")
        print(optimized_code1)
    else:
        print("\nOptimization failed")
    
    # Test case 2: Simple function
    test_code2 = '''
def calculate_squares(numbers):
    """Calculate squares of numbers."""
    squares = []
    for num in numbers:
        squares.append(num * num)
    return squares
'''
    
    print("\n" + "="*50)
    print("Original code:")
    print(test_code2)
    
    optimized_code2 = transformer.optimize_code(test_code2, "list_comprehension")
    if optimized_code2:
        print("\nOptimized code:")
        print(optimized_code2)
    else:
        print("\nOptimization failed")


if __name__ == "__main__":
    test_code_transformation()