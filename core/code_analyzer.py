"""
Intelligent Code Analysis System
This module provides semantic analysis capabilities for the Snake Agent to better
understand code intent and identify optimization opportunities.
"""

import ast
import astor
import re
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """
    Analyzes code semantically to identify optimization opportunities and understand intent.
    """
    
    def __init__(self):
        self.analysis_results = []
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """
        Perform comprehensive semantic analysis of code.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            
            # Perform various analyses
            analysis_results = {
                'basic_metrics': self._calculate_basic_metrics(tree),
                'optimization_opportunities': self._identify_optimization_opportunities(tree),
                'code_patterns': self._identify_code_patterns(tree),
                'complexity_analysis': self._analyze_complexity(tree),
                'data_flow': self._analyze_data_flow(tree),
                'potential_bugs': self._identify_potential_bugs(tree),
                'performance_antipatterns': self._identify_performance_antipatterns(tree)
            }
            
            logger.info("Successfully completed code analysis")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return {'error': str(e)}
    
    def _calculate_basic_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Calculate basic code metrics.
        
        Args:
            tree: AST of the code
            
        Returns:
            Dictionary of basic metrics
        """
        # Count various AST nodes
        node_counts = defaultdict(int)
        for node in ast.walk(tree):
            node_counts[type(node).__name__] += 1
        
        # Calculate specific metrics
        function_count = node_counts['FunctionDef']
        class_count = node_counts['ClassDef']
        loop_count = node_counts['For'] + node_counts['While']
        conditional_count = node_counts['If']
        call_count = node_counts['Call']
        
        return {
            'functions': function_count,
            'classes': class_count,
            'loops': loop_count,
            'conditionals': conditional_count,
            'function_calls': call_count,
            'total_nodes': sum(node_counts.values()),
            'node_distribution': dict(node_counts)
        }
    
    def _identify_optimization_opportunities(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Identify specific optimization opportunities in the code.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of optimization opportunities
        """
        opportunities = []
        
        # Look for list building loops
        list_building_loops = self._find_list_building_loops(tree)
        opportunities.extend(list_building_loops)
        
        # Look for string concatenation loops
        string_concat_loops = self._find_string_concatenation_loops(tree)
        opportunities.extend(string_concat_loops)
        
        # Look for inefficient built-in function usage
        builtin_inefficiencies = self._find_builtin_inefficiencies(tree)
        opportunities.extend(builtin_inefficiencies)
        
        # Look for redundant computations
        redundant_computations = self._find_redundant_computations(tree)
        opportunities.extend(redundant_computations)
        
        return opportunities
    
    def _find_list_building_loops(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Find loops that build lists inefficiently with append().
        
        Args:
            tree: AST of the code
            
        Returns:
            List of list building optimization opportunities
        """
        opportunities = []
        
        # Find for loops
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check if this looks like a list-building loop
                append_calls = [n for n in ast.walk(node) if isinstance(n, ast.Call) and
                               isinstance(n.func, ast.Attribute) and n.func.attr == 'append']
                
                if len(append_calls) > 0:
                    # Get function context if available
                    function_context = self._get_enclosing_function(node, tree)
                    
                    opportunity = {
                        'type': 'performance_optimization',
                        'subtype': 'list_building',
                        'description': 'Inefficient list building with append() in loop',
                        'recommendation': 'Use list comprehension instead',
                        'severity': 'medium',
                        'scope': 'function',
                        'location': {
                            'line_start': node.lineno,
                            'line_end': getattr(node, 'end_lineno', node.lineno),
                            'function': function_context
                        },
                        'confidence': 0.8,
                        'estimated_improvement': '20-50% performance gain'
                    }
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _find_string_concatenation_loops(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Find loops that concatenate strings inefficiently.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of string concatenation optimization opportunities
        """
        opportunities = []
        
        # Find for loops with augmented assignment containing +
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Look for augmented assignment with +
                aug_assign_nodes = [n for n in ast.walk(node) if isinstance(n, ast.AugAssign) and
                                   isinstance(n.op, ast.Add)]
                
                if len(aug_assign_nodes) > 0:
                    # Check if any of the operands are strings
                    function_context = self._get_enclosing_function(node, tree)
                    
                    opportunity = {
                        'type': 'performance_optimization',
                        'subtype': 'string_concatenation',
                        'description': 'Inefficient string concatenation in loop',
                        'recommendation': 'Use \'\'.join() instead',
                        'severity': 'medium',
                        'scope': 'function',
                        'location': {
                            'line_start': node.lineno,
                            'line_end': getattr(node, 'end_lineno', node.lineno),
                            'function': function_context
                        },
                        'confidence': 0.7,
                        'estimated_improvement': 'Significant performance gain for large strings'
                    }
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _find_builtin_inefficiencies(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Find inefficient usage of built-in functions.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of builtin function optimization opportunities
        """
        opportunities = []
        
        # Look for manual implementations that could use builtins
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Look for summation patterns
                if self._is_manual_summation(node):
                    function_context = self._get_enclosing_function(node, tree)
                    
                    opportunity = {
                        'type': 'performance_optimization',
                        'subtype': 'builtin_function',
                        'description': 'Manual summation loop can use sum() builtin',
                        'recommendation': 'Replace with sum() builtin function',
                        'severity': 'low',
                        'scope': 'function',
                        'location': {
                            'line_start': node.lineno,
                            'line_end': getattr(node, 'end_lineno', node.lineno),
                            'function': function_context
                        },
                        'confidence': 0.9,
                        'estimated_improvement': '10-30% performance gain'
                    }
                    opportunities.append(opportunity)
        
        return opportunities
    
    def _find_redundant_computations(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Find redundant computations that could be cached.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of redundant computation opportunities
        """
        opportunities = []
        
        # This would require more sophisticated analysis
        # For now, we'll just identify potential candidates
        
        return opportunities
    
    def _is_manual_summation(self, node: ast.For) -> bool:
        """
        Check if a for loop is manually summing values.
        
        Args:
            node: For loop AST node
            
        Returns:
            True if this looks like a manual summation loop
        """
        # Look for pattern:
        # total = 0
        # for item in items:
        #     total += item
        
        try:
            # Check if body contains augmented assignment with +=
            aug_assign_nodes = [n for n in ast.walk(node) if isinstance(n, ast.AugAssign) and
                               isinstance(n.op, ast.Add)]
            
            if len(aug_assign_nodes) > 0:
                # Check if one operand is a simple name (the accumulator)
                for aug_assign in aug_assign_nodes:
                    if isinstance(aug_assign.target, ast.Name):
                        return True
                        
        except Exception:
            pass
            
        return False
    
    def _get_enclosing_function(self, node: ast.AST, tree: ast.AST) -> Optional[str]:
        """
        Get the name of the function containing a node.
        
        Args:
            node: AST node
            tree: Root AST of the code
            
        Returns:
            Name of enclosing function, or None if not in a function
        """
        try:
            # Walk up the tree to find the enclosing function
            for parent_node in ast.walk(tree):
                if (isinstance(parent_node, ast.FunctionDef) and 
                    node in ast.walk(parent_node)):
                    return parent_node.name
        except Exception:
            pass
            
        return None
    
    def _identify_code_patterns(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Identify common code patterns and antipatterns.
        
        Args:
            tree: AST of the code
            
        Returns:
            Dictionary of identified patterns
        """
        patterns = {
            'design_patterns': [],
            'antipatterns': [],
            'code_smells': []
        }
        
        # Look for common patterns
        patterns['design_patterns'].extend(self._find_design_patterns(tree))
        patterns['antipatterns'].extend(self._find_antipatterns(tree))
        patterns['code_smells'].extend(self._find_code_smells(tree))
        
        return patterns
    
    def _find_design_patterns(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Identify design patterns in the code.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of identified design patterns
        """
        patterns = []
        
        # Look for factory patterns, decorator patterns, etc.
        # This is a simplified implementation
        
        return patterns
    
    def _find_antipatterns(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Identify code antipatterns.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of identified antipatterns
        """
        antipatterns = []
        
        # Look for god objects, spaghetti code, etc.
        # This is a simplified implementation
        
        return antipatterns
    
    def _find_code_smells(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Identify code smells.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of identified code smells
        """
        smells = []
        
        # Look for long methods, large classes, etc.
        # This is a simplified implementation
        
        return smells
    
    def _analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Analyze code complexity metrics.
        
        Args:
            tree: AST of the code
            
        Returns:
            Dictionary of complexity metrics
        """
        complexity_metrics = {
            'cyclomatic_complexity': 0,
            'nesting_depth': 0,
            'function_lengths': [],
            'class_sizes': []
        }
        
        # Calculate cyclomatic complexity
        complexity_metrics['cyclomatic_complexity'] = self._calculate_cyclomatic_complexity(tree)
        
        # Calculate nesting depth
        complexity_metrics['nesting_depth'] = self._calculate_nesting_depth(tree)
        
        # Collect function lengths
        complexity_metrics['function_lengths'] = self._collect_function_lengths(tree)
        
        # Collect class sizes
        complexity_metrics['class_sizes'] = self._collect_class_sizes(tree)
        
        return complexity_metrics
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """
        Calculate cyclomatic complexity of the code.
        
        Args:
            tree: AST of the code
            
        Returns:
            Cyclomatic complexity
        """
        # Cyclomatic complexity = number of decision points + 1
        decision_points = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                decision_points += 1
            elif isinstance(node, ast.BoolOp):
                # Each boolean operator adds complexity
                decision_points += len(node.values) - 1
        
        return decision_points + 1
    
    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
        """
        Calculate maximum nesting depth.
        
        Args:
            tree: AST of the code
            
        Returns:
            Maximum nesting depth
        """
        max_depth = 0
        
        def calculate_depth(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            # Recursively calculate depth for child nodes
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.FunctionDef)):
                    calculate_depth(child, current_depth + 1)
                else:
                    calculate_depth(child, current_depth)
        
        calculate_depth(tree)
        return max_depth
    
    def _collect_function_lengths(self, tree: ast.AST) -> List[int]:
        """
        Collect lengths of all functions.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of function lengths (in lines)
        """
        function_lengths = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Calculate function length
                if hasattr(node, 'end_lineno') and node.end_lineno:
                    length = node.end_lineno - node.lineno + 1
                else:
                    # Estimate based on body
                    length = len(node.body)
                function_lengths.append(length)
        
        return function_lengths
    
    def _collect_class_sizes(self, tree: ast.AST) -> List[int]:
        """
        Collect sizes of all classes.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of class sizes (number of methods/attributes)
        """
        class_sizes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Count methods and attributes
                method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                class_sizes.append(method_count)
        
        return class_sizes
    
    def _analyze_data_flow(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Analyze data flow patterns in the code.
        
        Args:
            tree: AST of the code
            
        Returns:
            Dictionary of data flow analysis results
        """
        data_flow_analysis = {
            'variable_usage': {},
            'data_dependencies': [],
            'unused_variables': [],
            'reassigned_variables': []
        }
        
        # This would require sophisticated data flow analysis
        # For now, we'll provide a basic implementation
        
        return data_flow_analysis
    
    def _identify_potential_bugs(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Identify potential bugs in the code.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of potential bugs
        """
        potential_bugs = []
        
        # Look for common bug patterns
        potential_bugs.extend(self._find_off_by_one_errors(tree))
        potential_bugs.extend(self._find_unhandled_exceptions(tree))
        potential_bugs.extend(self._find_resource_leaks(tree))
        
        return potential_bugs
    
    def _find_off_by_one_errors(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Find potential off-by-one errors.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of potential off-by-one errors
        """
        off_by_one_errors = []
        
        # Look for common patterns that might indicate off-by-one errors
        # This is a simplified implementation
        
        return off_by_one_errors
    
    def _find_unhandled_exceptions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Find unhandled exceptions.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of unhandled exceptions
        """
        unhandled_exceptions = []
        
        # Look for function calls that might raise exceptions without handling
        # This is a simplified implementation
        
        return unhandled_exceptions
    
    def _find_resource_leaks(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Find potential resource leaks.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of potential resource leaks
        """
        resource_leaks = []
        
        # Look for opened resources without proper closing
        # This is a simplified implementation
        
        return resource_leaks
    
    def _identify_performance_antipatterns(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Identify performance-related antipatterns.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of performance antipatterns
        """
        antipatterns = []
        
        # Look for inefficient patterns
        antipatterns.extend(self._find_quadratic_algorithms(tree))
        antipatterns.extend(self._find_inefficient_data_structures(tree))
        
        return antipatterns
    
    def _find_quadratic_algorithms(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Find algorithms with quadratic time complexity.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of quadratic algorithm antipatterns
        """
        quadratic_algorithms = []
        
        # Look for nested loops that might indicate quadratic complexity
        # This is a simplified implementation
        
        return quadratic_algorithms
    
    def _find_inefficient_data_structures(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """
        Find inefficient data structure usage.
        
        Args:
            tree: AST of the code
            
        Returns:
            List of inefficient data structure antipatterns
        """
        inefficient_ds = []
        
        # Look for patterns like using list for membership testing
        # This is a simplified implementation
        
        return inefficient_ds


# Example usage
def test_code_analysis():
    """
    Test the code analysis system with examples.
    """
    analyzer = CodeAnalyzer()
    
    # Test case 1: Code with optimization opportunities
    test_code1 = '''
def inefficient_list_building():
    """Build a list inefficiently with append()."""
    result = []
    for i in range(100):
        result.append(i * 2)
    return result

def another_inefficient_function():
    """Another function with inefficient string concatenation."""
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
'''
    
    print("Analyzing code for optimization opportunities...")
    analysis_results = analyzer.analyze_code(test_code1)
    
    print("\\nAnalysis Results:")
    print("=" * 50)
    
    # Display basic metrics
    if 'basic_metrics' in analysis_results:
        metrics = analysis_results['basic_metrics']
        print(f"Basic Metrics:")
        print(f"  Functions: {metrics['functions']}")
        print(f"  Classes: {metrics['classes']}")
        print(f"  Loops: {metrics['loops']}")
        print(f"  Conditionals: {metrics['conditionals']}")
        print(f"  Function calls: {metrics['function_calls']}")
        print()
    
    # Display optimization opportunities
    if 'optimization_opportunities' in analysis_results:
        opportunities = analysis_results['optimization_opportunities']
        print(f"Found {len(opportunities)} optimization opportunities:")
        for i, opp in enumerate(opportunities, 1):
            print(f"  {i}. {opp['description']}")
            print(f"     Recommendation: {opp['recommendation']}")
            print(f"     Severity: {opp['severity']}")
            print(f"     Confidence: {opp['confidence']}")
            print()


if __name__ == "__main__":
    test_code_analysis()