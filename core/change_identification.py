"""
Change Identification System for Snake Agent
This module provides functionality for the Snake Agent to identify potential
small, targeted improvements in the codebase.
"""

import ast
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ChangeIdentifier:
    """
    Identifies potential changes/improvements in code that can be safely applied.
    Focuses on small, targeted optimizations that improve performance or maintainability.
    """
    
    def __init__(self):
        self.change_types = {
            'performance_optimization': self._identify_performance_optimizations,
            'code_clarity': self._identify_clarity_improvements,
            'redundancy_reduction': self._identify_redundancy,
            'function_efficiency': self._identify_function_efficiency
        }
    
    def identify_changes_in_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Identify potential changes in a specific file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            List of potential changes that can be made
        """
        changes = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the file as an AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                logger.warning(f"Could not parse {file_path} as valid Python")
                return []
            
            # Analyze the AST for different types of improvements
            for change_type, analyzer_func in self.change_types.items():
                try:
                    potential_changes = analyzer_func(tree, content, file_path)
                    changes.extend(potential_changes)
                except Exception as e:
                    logger.error(f"Error analyzing {change_type} in {file_path}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error reading or analyzing {file_path}: {e}")
        
        return changes
    
    def identify_changes_in_project(self, project_root: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify potential changes across the entire project.
        
        Args:
            project_root: Root directory of the project
            
        Returns:
            Dictionary mapping file paths to lists of potential changes
        """
        changes_by_file = {}
        
        # Walk through all Python files in the project
        for root, dirs, files in os.walk(project_root):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    abs_path = str(file_path.resolve())
                    
                    # Only analyze files that are part of the main codebase
                    if self._should_analyze_file(abs_path):
                        changes = self.identify_changes_in_file(abs_path)
                        if changes:
                            changes_by_file[abs_path] = changes
        
        return changes_by_file
    
    def _should_analyze_file(self, file_path: str) -> bool:
        """
        Determines if a file should be analyzed for changes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file should be analyzed
        """
        # Skip test files, configuration files, and generated files
        path_parts = file_path.lower().split('/')
        file_name = path_parts[-1]
        
        # Skip if it's a test file
        if file_name.startswith('test_') or file_name.endswith('_test.py'):
            return False
        
        # Skip if it's a config file or setup file
        if file_name in ['setup.py', 'pyproject.toml', 'requirements.txt']:
            return False
        
        return True
    
    def _identify_performance_optimizations(self, tree: ast.AST, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Identify potential performance optimizations in the code.
        
        Args:
            tree: AST of the code
            content: Original content of the file
            file_path: Path to the file
            
        Returns:
            List of performance optimization opportunities
        """
        changes = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            # Look for inefficient operations like repeated list calculations
            if isinstance(node, ast.For) and hasattr(node, 'iter'):
                # Check for inefficient iteration patterns
                inefficiencies = self._find_iteration_inefficiencies(node, lines)
                changes.extend(inefficiencies)
            
            # Look for string concatenation in loops
            if isinstance(node, ast.While) or isinstance(node, ast.For):
                string_concat_issues = self._find_string_concat_in_loops(node, lines)
                changes.extend(string_concat_issues)
        
        return changes
    
    def _find_iteration_inefficiencies(self, node: ast.For, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Find inefficient iteration patterns in for loops.
        
        Args:
            node: AST node representing a for loop
            lines: List of lines in the file
            
        Returns:
            List of inefficiency findings
        """
        changes = []
        
        try:
            # Check if the loop is iterating over range(len(...))
            if (isinstance(node.iter, ast.Call) and 
                isinstance(node.iter.func, ast.Name) and 
                node.iter.func.id == 'range'):
                
                # Check if it's range(len(...))
                if (len(node.iter.args) == 1 and 
                    isinstance(node.iter.args[0], ast.Call) and
                    isinstance(node.iter.args[0].func, ast.Name) and
                    node.iter.args[0].func.id == 'len'):
                    
                    # This is likely a range(len()) pattern which might be inefficient
                    target_var = ast.unparse(node.target) if hasattr(ast, 'unparse') else str(node.target)
                    container = ast.unparse(node.iter.args[0].args[0]) if hasattr(ast, 'unparse') else str(node.iter.args[0].args[0])
                    
                    change = {
                        'type': 'performance_optimization',
                        'subtype': 'iteration_pattern',
                        'line_start': node.lineno,
                        'line_end': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                        'description': f'Consider using direct iteration over {container} instead of range(len({container}))',
                        'recommendation': f'Iterate directly over the container: for {target_var} in {container}:',
                        'severity': 'medium',
                        'scope': 'function'
                    }
                    changes.append(change)
        
        except Exception as e:
            logger.debug(f"Error analyzing iteration ineficiencies: {e}")
        
        return changes
    
    def _find_string_concat_in_loops(self, node: ast.AST, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Find inefficient string concatenation in loops.
        
        Args:
            node: AST node representing a loop
            lines: List of lines in the file
            
        Returns:
            List of string concatenation findings
        """
        changes = []
        
        # Look for assignments that concatenate strings in the loop body
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        # Check if it's assigning with string concatenation
                        if (isinstance(stmt.value, ast.BinOp) and 
                            isinstance(stmt.value.op, ast.Add)):
                            
                            # Check if it's string concatenation (one of the operands is a string)
                            left_str = self._is_string_expr(stmt.value.left)
                            right_str = self._is_string_expr(stmt.value.right)
                            
                            if left_str or right_str:
                                change = {
                                    'type': 'performance_optimization',
                                    'subtype': 'string_concatenation',
                                    'line_start': stmt.lineno,
                                    'line_end': stmt.end_lineno if hasattr(stmt, 'end_lineno') else stmt.lineno,
                                    'description': 'Inefficient string concatenation in loop',
                                    'recommendation': 'Use str.join() or f-strings for better performance',
                                    'severity': 'medium',
                                    'scope': 'function'
                                }
                                changes.append(change)
        
        return changes
    
    def _is_string_expr(self, expr) -> bool:
        """
        Check if an expression is likely to be a string.
        
        Args:
            expr: AST expression to check
            
        Returns:
            True if expression is likely a string
        """
        if isinstance(expr, ast.Str):
            return True
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return True
        if isinstance(expr, ast.Name):  # Could be a string variable
            return True  # Conservative assumption
        return False
    
    def _identify_clarity_improvements(self, tree: ast.AST, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Identify opportunities to improve code clarity.
        
        Args:
            tree: AST of the code
            content: Original content of the file
            file_path: Path to the file
            
        Returns:
            List of clarity improvement opportunities
        """
        changes = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            # Look for magic numbers that could be constants
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if self._is_likely_magic_number(node, lines):
                    change = {
                        'type': 'code_clarity',
                        'subtype': 'magic_number',
                        'line_start': node.lineno,
                        'line_end': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                        'description': f'Replace magic number {node.value} with named constant',
                        'recommendation': 'Define this number as a named constant with descriptive name',
                        'severity': 'low',
                        'scope': 'function'
                    }
                    changes.append(change)
        
        return changes
    
    def _is_likely_magic_number(self, node: ast.Constant, lines: List[str]) -> bool:
        """
        Determine if a number is likely a "magic number" that should be named.
        
        Args:
            node: AST node with the number
            lines: List of lines in the file
            
        Returns:
            True if the number is likely a magic number
        """
        # Common "safe" numbers that are usually OK to have in code
        safe_numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000}
        
        if node.value in safe_numbers:
            return False
        
        # Check if it's in a context that's clearly a constant (like assignment to uppercase var)
        # This is complex to do with AST, so we'll keep it simple for now
        return abs(node.value) > 10 or node.value < 0  # Simplified heuristic
    
    def _identify_redundancy(self, tree: ast.AST, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Identify redundant code that could be simplified.
        
        Args:
            tree: AST of the code
            content: Original content of the file
            file_path: Path to the file
            
        Returns:
            List of redundancy reduction opportunities
        """
        changes = []
        lines = content.split('\n')
        
        # Look for duplicated code blocks
        all_functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        for i, func1 in enumerate(all_functions):
            for func2 in all_functions[i+1:]:
                similarity = self._compare_functions(func1, func2, lines)
                if similarity > 0.8:  # 80% similarity threshold
                    change = {
                        'type': 'redundancy_reduction',
                        'subtype': 'duplicated_code',
                        'line_start': func1.lineno,
                        'line_end': func1.end_lineno if hasattr(func1, 'end_lineno') else func1.lineno,
                        'description': f'Function {func1.name} is highly similar to {func2.name}',
                        'recommendation': f'Consider refactoring to avoid code duplication',
                        'severity': 'medium',
                        'scope': 'function'
                    }
                    changes.append(change)
        
        return changes
    
    def _compare_functions(self, func1: ast.FunctionDef, func2: ast.FunctionDef, lines: List[str]) -> float:
        """
        Compare two functions for similarity.
        
        Args:
            func1: First function AST node
            func2: Second function AST node
            lines: List of lines in the file
            
        Returns:
            Similarity score between 0 and 1
        """
        # This is a basic implementation - in a real system, this would be more sophisticated
        try:
            code1 = ast.unparse(func1) if hasattr(ast, 'unparse') else str(func1)
            code2 = ast.unparse(func2) if hasattr(ast, 'unparse') else str(func2)
            
            # Simple similarity based on common tokens
            tokens1 = set(re.findall(r'\w+', code1.lower()))
            tokens2 = set(re.findall(r'\w+', code2.lower()))
            
            if not tokens1 and not tokens2:
                return 1.0 if code1 == code2 else 0.0
            
            common_tokens = tokens1.intersection(tokens2)
            all_tokens = tokens1.union(tokens2)
            
            return len(common_tokens) / len(all_tokens) if all_tokens else 0.0
        except:
            return 0.0  # If we can't parse, assume no similarity
    
    def _identify_function_efficiency(self, tree: ast.AST, content: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Identify functions that could be made more efficient.
        
        Args:
            tree: AST of the code
            content: Original content of the file
            file_path: Path to the file
            
        Returns:
            List of function efficiency opportunities
        """
        changes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for functions that could benefit from optimization
                efficiency_issues = self._analyze_function_efficiency(node, content)
                changes.extend(efficiency_issues)
        
        return changes
    
    def _analyze_function_efficiency(self, func_node: ast.FunctionDef, content: str) -> List[Dict[str, Any]]:
        """
        Analyze a single function for efficiency issues.
        
        Args:
            func_node: AST node for the function
            content: Content of the file
            
        Returns:
            List of efficiency issues in the function
        """
        changes = []
        lines = content.split('\n')
        
        # Count nested loops - potential performance issue
        nested_loop_count = self._count_nested_loops(func_node)
        if nested_loop_count >= 3:  # Triple nested loops or more
            change = {
                'type': 'function_efficiency',
                'subtype': 'nested_loops',
                'line_start': func_node.lineno,
                'line_end': func_node.end_lineno if hasattr(func_node, 'end_lineno') else func_node.lineno,
                'description': f'Function {func_node.name} has {nested_loop_count} levels of nested loops',
                'recommendation': 'Consider algorithm optimization or data structure changes',
                'severity': 'high',
                'scope': 'function'
            }
            changes.append(change)
        
        # Check for inefficient list operations in loops
        inefficient_ops = self._find_inefficient_operations(func_node)
        for op in inefficient_ops:
            change = {
                'type': 'function_efficiency',
                'subtype': 'inefficient_operation',
                'line_start': op['line'],
                'line_end': op['line'],
                'description': op['description'],
                'recommendation': op['recommendation'],
                'severity': op['severity'],
                'scope': 'function'
            }
            changes.append(change)
        
        return changes
    
    def _count_nested_loops(self, node: ast.AST) -> int:
        """
        Count the maximum level of nested loops in a function.
        
        Args:
            node: AST node to analyze
            
        Returns:
            Maximum level of nested loops
        """
        max_depth = 0
        loops = [n for n in ast.walk(node) if isinstance(n, (ast.For, ast.While))]
        
        for loop in loops:
            # Count how many loops contain this one
            depth = 0
            current = loop
            # Navigate up the tree
            for potential_container in ast.walk(node):
                if (isinstance(potential_container, (ast.For, ast.While)) and 
                    self._is_parent_of(potential_container, loop)):
                    depth += 1
            
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _is_parent_of(self, parent, child) -> bool:
        """
        Check if parent node contains child node in its subtree.
        
        Args:
            parent: Potential parent node
            child: Potential child node
            
        Returns:
            True if parent contains child
        """
        for node in ast.walk(parent):
            if node == child:
                return True
        return False
    
    def _find_inefficient_operations(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """
        Find inefficient operations in a function.
        
        Args:
            func_node: AST node for the function
            
        Returns:
            List of inefficient operations
        """
        inefficient_ops = []
        
        for node in ast.walk(func_node):
            # Look for list indexing in loops that could be optimized
            if (isinstance(node, ast.Subscript) and 
                isinstance(node.ctx, ast.Load) and
                self._is_in_loop(node)):
                
                # Check for inefficient list operations
                if (isinstance(node.value, ast.Name) and 
                    isinstance(node.slice, ast.Constant)):
                    
                    # Might be repeated indexing in a loop
                    inefficient_ops.append({
                        'line': node.lineno,
                        'description': 'Repeated indexing in loop context may be inefficient',
                        'recommendation': 'Consider using enumerate or direct iteration',
                        'severity': 'medium'
                    })
        
        return inefficient_ops
    
    def _is_in_loop(self, node: ast.AST) -> bool:
        """
        Check if a node is inside a loop.
        
        Args:
            node: Node to check
            
        Returns:
            True if node is inside a loop
        """
        # For simplicity, we'll use a basic approach
        # In a real implementation, this would traverse up the AST
        return False  # Simplified for now


class ChangeValidator:
    """
    Validates potential changes to ensure they are safe to apply.
    """
    
    def __init__(self):
        self.safe_change_types = {
            'performance_optimization',
            'code_clarity',
            'redundancy_reduction',
            'function_efficiency'
        }
    
    def is_change_safe(self, change: Dict[str, Any]) -> bool:
        """
        Determine if a change is safe to apply.
        
        Args:
            change: Change dictionary to validate
            
        Returns:
            True if change is safe
        """
        # Check that the change type is in our approved list
        if change.get('type') not in self.safe_change_types:
            return False
        
        # Check severity - high severity changes may need more validation
        severity = change.get('severity', 'medium')
        if severity == 'high':
            # For now, we'll allow high severity changes but might want more validation later
            pass
        
        # Check scope - only allow changes to functions, not entire files or modules
        scope = change.get('scope', 'function')
        if scope not in ['function', 'method', 'block', 'line']:
            return False
        
        # Basic validation passed
        return True
    
    def filter_safe_changes(self, changes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter a list of changes to only include safe ones.
        
        Args:
            changes: List of changes to filter
            
        Returns:
            List of safe changes
        """
        return [change for change in changes if self.is_change_safe(change)]