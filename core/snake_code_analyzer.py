"""
Snake Code Analyzer

This module provides intelligent code analysis capabilities for the Snake Agent,
identifying improvement opportunities, performance issues, and architectural enhancements.
"""

import ast
import asyncio
import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CodeMetrics:
    """Code metrics and statistics"""
    lines_of_code: int = 0
    complexity: int = 0
    class_count: int = 0
    function_count: int = 0
    import_count: int = 0
    todo_count: int = 0
    comment_ratio: float = 0.0
    max_line_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lines_of_code": self.lines_of_code,
            "complexity": self.complexity,
            "class_count": self.class_count,
            "function_count": self.function_count,
            "import_count": self.import_count,
            "todo_count": self.todo_count,
            "comment_ratio": self.comment_ratio,
            "max_line_length": self.max_line_length
        }


@dataclass
class CodeIssue:
    """Represents a code issue or improvement opportunity"""
    type: str  # 'performance', 'quality', 'security', 'architecture'
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "description": self.description,
            "line_number": self.line_number,
            "suggestion": self.suggestion,
            "confidence": self.confidence
        }


class ASTAnalyzer:
    """Analyzes Python AST for various code patterns and issues"""

    def __init__(self):
        self.metrics = CodeMetrics()
        self.issues: List[CodeIssue] = []

    def analyze_ast(self, tree: ast.AST, source_lines: List[str]) -> Tuple[CodeMetrics, List[CodeIssue]]:
        """Analyze AST and return metrics and issues"""
        self.metrics = CodeMetrics()
        self.issues = []

        # Accept either an AST node or a compiled code object; if a code object is passed
        # reconstruct the AST from the provided source_lines.
        if not isinstance(tree, ast.AST):
            try:
                # If a code object was passed (from compile()), parse the source_lines
                tree = ast.parse('\n'.join(source_lines))
            except Exception:
                # Fallback: treat as empty tree
                tree = ast.parse('')

        # Calculate basic metrics
        self.metrics.lines_of_code = len(
            [line for line in source_lines if line.strip()])
        self.metrics.max_line_length = max(
            (len(line) for line in source_lines), default=0)

        # Count comments and TODOs
        comment_lines = 0
        for line in source_lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                comment_lines += 1
            if 'TODO' in line.upper() or 'FIXME' in line.upper():
                self.metrics.todo_count += 1

        self.metrics.comment_ratio = comment_lines / max(len(source_lines), 1)

        # Analyze AST nodes
        self._analyze_node(tree, source_lines)

        return self.metrics, self.issues

    def _analyze_node(self, node: ast.AST, source_lines: List[str]):
        """Recursively analyze AST nodes"""
        if isinstance(node, ast.ClassDef):
            self.metrics.class_count += 1
            self._analyze_class(node, source_lines)

        elif isinstance(node, ast.FunctionDef):
            self.metrics.function_count += 1
            self._analyze_function(node, source_lines)

        elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            self.metrics.import_count += 1
            self._analyze_import(node)

        elif isinstance(node, ast.For) or isinstance(node, ast.While):
            self._analyze_loop(node, source_lines)

        elif isinstance(node, ast.Try):
            self._analyze_exception_handling(node, source_lines)

        # Recurse into child nodes
        for child in ast.iter_child_nodes(node):
            self._analyze_node(child, source_lines)

    def _analyze_class(self, node: ast.ClassDef, source_lines: List[str]):
        """Analyze class definition"""
        # Check for very large classes
        if len(node.body) > 50:
            self.issues.append(CodeIssue(
                type="architecture",
                severity="medium",
                description=f"Class '{node.name}' is very large ({len(node.body)} methods/attributes)",
                line_number=node.lineno,
                suggestion="Consider breaking this class into smaller, more focused classes",
                confidence=0.8
            ))

        # Check for missing docstring
        if not ast.get_docstring(node):
            self.issues.append(CodeIssue(
                type="quality",
                severity="low",
                description=f"Class '{node.name}' missing docstring",
                line_number=node.lineno,
                suggestion="Add a docstring describing the class purpose",
                confidence=0.9
            ))

    def _analyze_function(self, node: ast.FunctionDef, source_lines: List[str]):
        """Analyze function definition"""
        # Calculate cyclomatic complexity
        complexity = self._calculate_complexity(node)
        self.metrics.complexity += complexity

        # Check for high complexity (lower threshold to surface issues earlier)
        if complexity > 5:
            self.issues.append(CodeIssue(
                type="quality",
                severity="high",
                description=f"Function '{node.name}' has high complexity ({complexity})",
                line_number=node.lineno,
                suggestion="Consider breaking this function into smaller functions",
                confidence=0.9
            ))

        # Check for very long functions
        if hasattr(node, 'end_lineno') and node.end_lineno:
            func_length = node.end_lineno - node.lineno
            if func_length > 100:
                self.issues.append(CodeIssue(
                    type="quality",
                    severity="medium",
                    description=f"Function '{node.name}' is very long ({func_length} lines)",
                    line_number=node.lineno,
                    suggestion="Consider breaking this function into smaller functions",
                    confidence=0.8
                ))

        # Check for missing docstring
        if not ast.get_docstring(node) and not node.name.startswith('_'):
            self.issues.append(CodeIssue(
                type="quality",
                severity="low",
                description=f"Public function '{node.name}' missing docstring",
                line_number=node.lineno,
                suggestion="Add a docstring describing the function purpose and parameters",
                confidence=0.7
            ))

        # Check for too many parameters
        if len(node.args.args) > 7:
            self.issues.append(CodeIssue(
                type="quality",
                severity="medium",
                description=f"Function '{node.name}' has too many parameters ({len(node.args.args)})",
                line_number=node.lineno,
                suggestion="Consider using a configuration object or breaking the function",
                confidence=0.8
            ))

    def _analyze_import(self, node):
        """Analyze import statements"""
        # Check for unused imports (basic heuristic)
        if isinstance(node, ast.Import):
            for alias in node.names:
                # This is a basic check - would need more sophisticated analysis
                pass

    def _analyze_loop(self, node, source_lines: List[str]):
        """Analyze loop constructs"""
        # Check for potential infinite loops (very basic)
        if isinstance(node, ast.While):
            # Look for common infinite loop patterns
            if isinstance(node.test, ast.Constant) and node.test.value is True:
                self.issues.append(CodeIssue(
                    type="quality",
                    severity="medium",
                    description="Potential infinite loop detected (while True:)",
                    line_number=node.lineno,
                    suggestion="Ensure there's a proper break condition",
                    confidence=0.6
                ))

    def _analyze_exception_handling(self, node: ast.Try, source_lines: List[str]):
        """Analyze exception handling"""
        # Check for bare except clauses
        for handler in node.handlers:
            if handler.type is None:
                self.issues.append(CodeIssue(
                    type="quality",
                    severity="medium",
                    description="Bare except clause detected",
                    line_number=handler.lineno,
                    suggestion="Catch specific exceptions instead of using bare except",
                    confidence=0.9
                ))

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity


class PatternAnalyzer:
    """Analyzes code for specific patterns and anti-patterns"""

    def __init__(self):
        self.issues: List[CodeIssue] = []

    def analyze_patterns(self, code: str, file_path: str) -> List[CodeIssue]:
        """Analyze code for patterns and anti-patterns"""
        self.issues = []
        lines = code.split('\n')

        self._check_security_patterns(lines)
        self._check_performance_patterns(lines)
        self._check_async_patterns(lines)
        self._check_ravana_specific_patterns(lines, file_path)

        return self.issues

    def _check_security_patterns(self, lines: List[str]):
        """Check for security-related patterns"""
        for i, line in enumerate(lines, 1):
            # Check for SQL injection patterns
            if re.search(r'execute\s*\(\s*["\'][^"\']*%', line):
                self.issues.append(CodeIssue(
                    type="security",
                    severity="high",
                    description="Potential SQL injection vulnerability",
                    line_number=i,
                    suggestion="Use parameterized queries instead of string formatting",
                    confidence=0.8
                ))

            # Check for hardcoded secrets
            if re.search(r'(password|secret|key)\s*=\s*["\'][^"\']+["\']', line, re.IGNORECASE):
                self.issues.append(CodeIssue(
                    type="security",
                    severity="critical",
                    description="Potential hardcoded secret detected",
                    line_number=i,
                    suggestion="Use environment variables or secure configuration",
                    confidence=0.7
                ))

    def _check_performance_patterns(self, lines: List[str]):
        """Check for performance-related patterns"""
        for i, line in enumerate(lines, 1):
            # Check for inefficient string concatenation in loops
            if 'for ' in line and i < len(lines) - 1 and '+=' in lines[i]:
                self.issues.append(CodeIssue(
                    type="performance",
                    severity="medium",
                    description="Inefficient string concatenation in loop",
                    line_number=i + 1,
                    suggestion="Use join() or list comprehension for better performance",
                    confidence=0.7
                ))

            # Check for synchronous operations in async functions
            if 'requests.get(' in line or 'requests.post(' in line:
                # Look back for async def
                for j in range(max(0, i - 10), i):
                    if 'async def' in lines[j]:
                        self.issues.append(CodeIssue(
                            type="performance",
                            severity="medium",
                            description="Synchronous HTTP request in async function",
                            line_number=i,
                            suggestion="Use aiohttp or httpx for async HTTP requests",
                            confidence=0.8
                        ))
                        break

    def _check_async_patterns(self, lines: List[str]):
        """Check for async/await patterns"""
        for i, line in enumerate(lines, 1):
            # Check for missing await
            if 'async def' in line:
                # Look for potential missing awaits in the function
                func_start = i
                func_end = min(len(lines), i + 50)  # Check next 50 lines

                for j in range(func_start, func_end):
                    if lines[j].strip().startswith('return ') and '(' in lines[j]:
                        # Potential missing await on return
                        if not 'await' in lines[j]:
                            self.issues.append(CodeIssue(
                                type="quality",
                                severity="medium",
                                description="Potential missing 'await' in async function",
                                line_number=j + 1,
                                suggestion="Check if this function call should be awaited",
                                confidence=0.5
                            ))

    def _check_ravana_specific_patterns(self, lines: List[str], file_path: str):
        """Check for RAVANA-specific patterns and best practices"""
        for i, line in enumerate(lines, 1):
            # Check for proper logging usage
            if 'print(' in line and not line.strip().startswith('#'):
                self.issues.append(CodeIssue(
                    type="quality",
                    severity="low",
                    description="Using print() instead of logging",
                    line_number=i,
                    suggestion="Use logger.info(), logger.debug(), etc. instead of print()",
                    confidence=0.8
                ))

            # Check for proper error handling in AGI modules
            if 'modules/' in file_path and 'try:' in line:
                # Look for specific exception handling
                has_specific_except = False
                for j in range(i, min(len(lines), i + 10)):
                    if 'except Exception' in lines[j] or 'except:' in lines[j]:
                        has_specific_except = True
                        break

                if has_specific_except:
                    self.issues.append(CodeIssue(
                        type="quality",
                        severity="low",
                        description="Generic exception handling in AGI module",
                        line_number=i,
                        suggestion="Consider catching specific exceptions for better error handling",
                        confidence=0.6
                    ))


class SnakeCodeAnalyzer:
    """Main code analyzer for Snake Agent"""

    def __init__(self, coding_llm):
        self.coding_llm = coding_llm
        self.ast_analyzer = ASTAnalyzer()
        self.pattern_analyzer = PatternAnalyzer()

    async def analyze_code(self, code_content: str, file_path: str, change_type: str) -> Dict[str, Any]:
        """Perform comprehensive code analysis"""
        try:
            analysis_result = {
                "file_path": file_path,
                "change_type": change_type,
                "timestamp": asyncio.get_event_loop().time(),
                "metrics": {},
                "static_issues": [],
                "llm_analysis": {},
                "improvements_suggested": False,
                "priority": "medium",
                "confidence": 0.0
            }

            # Static analysis
            metrics, static_issues = await self._perform_static_analysis(code_content, file_path)
            analysis_result["metrics"] = metrics.to_dict()
            analysis_result["static_issues"] = [issue.to_dict()
                                                for issue in static_issues]

            # LLM-based analysis
            llm_analysis = await self._perform_llm_analysis(code_content, file_path, change_type)
            analysis_result["llm_analysis"] = llm_analysis

            # Combine results and determine overall assessment
            overall_assessment = self._combine_analysis_results(
                metrics, static_issues, llm_analysis)
            analysis_result.update(overall_assessment)

            logger.info(
                f"Code analysis completed for {file_path}: {len(static_issues)} static issues found")

            return analysis_result

        except Exception as e:
            logger.error(f"Error analyzing code {file_path}: {e}")
            return {
                "file_path": file_path,
                "error": str(e),
                "improvements_suggested": False,
                "priority": "low",
                "confidence": 0.0
            }

    async def _perform_static_analysis(self, code_content: str, file_path: str) -> Tuple[CodeMetrics, List[CodeIssue]]:
        """Perform static analysis using AST and pattern matching"""
        all_issues = []

        try:
            # AST analysis
            tree = ast.parse(code_content)
            source_lines = code_content.split('\n')
            metrics, ast_issues = self.ast_analyzer.analyze_ast(
                tree, source_lines)
            all_issues.extend(ast_issues)

        except SyntaxError as e:
            # Handle syntax errors
            all_issues.append(CodeIssue(
                type="quality",
                severity="critical",
                description=f"Syntax error: {e.msg}",
                line_number=e.lineno,
                suggestion="Fix syntax error before proceeding",
                confidence=1.0
            ))
            metrics = CodeMetrics()

        # Pattern analysis
        pattern_issues = self.pattern_analyzer.analyze_patterns(
            code_content, file_path)
        all_issues.extend(pattern_issues)

        return metrics, all_issues

    async def _perform_llm_analysis(self, code_content: str, file_path: str, change_type: str) -> Dict[str, Any]:
        """Perform LLM-based analysis"""
        try:
            # Prepare analysis prompt based on change type
            if change_type == "new":
                analysis_type = "new_file_review"
            elif change_type == "modified":
                analysis_type = "modification_review"
            else:
                analysis_type = "general_review"

            # Get LLM analysis
            llm_response = await self.coding_llm.analyze_code(code_content, analysis_type)

            # Parse LLM response
            return self._parse_llm_analysis(llm_response)

        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return {"error": str(e), "suggestions": [], "quality_score": 0.5}

    def _parse_llm_analysis(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM analysis response"""
        try:
            # Try to extract JSON from response
            if '```json' in llm_response:
                json_start = llm_response.find('```json') + 7
                json_end = llm_response.find('```', json_start)
                json_content = llm_response[json_start:json_end].strip()
                return json.loads(json_content)

            # Fallback: parse structured text response
            analysis = {
                "suggestions": [],
                "quality_score": 0.5,
                "performance_notes": [],
                "architecture_suggestions": [],
                "raw_response": llm_response
            }

            # Extract suggestions from text
            lines = llm_response.split('\n')
            for line in lines:
                if 'suggestion:' in line.lower():
                    analysis["suggestions"].append(line.strip())
                elif 'performance:' in line.lower():
                    analysis["performance_notes"].append(line.strip())
                elif 'architecture:' in line.lower():
                    analysis["architecture_suggestions"].append(line.strip())

            return analysis

        except Exception as e:
            logger.error(f"Error parsing LLM analysis: {e}")
            return {
                "error": str(e),
                "raw_response": llm_response,
                "suggestions": [],
                "quality_score": 0.5
            }

    def _combine_analysis_results(self, metrics: CodeMetrics, static_issues: List[CodeIssue],
                                  llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Combine static and LLM analysis results"""
        # Calculate overall priority
        high_severity_count = len(
            [issue for issue in static_issues if issue.severity in ['high', 'critical']])
        medium_severity_count = len(
            [issue for issue in static_issues if issue.severity == 'medium'])

        # Determine if improvements are suggested
        improvements_suggested = (
            high_severity_count > 0 or
            medium_severity_count > 2 or
            len(llm_analysis.get("suggestions", [])) > 0 or
            metrics.complexity > 20 or
            metrics.lines_of_code > 500
        )

        # Calculate priority
        if high_severity_count > 0 or metrics.complexity > 30:
            priority = "high"
        elif medium_severity_count > 1 or metrics.complexity > 15:
            priority = "medium"
        else:
            priority = "low"

        # Calculate confidence
        confidence = min(1.0, (
            len(static_issues) * 0.1 +
            len(llm_analysis.get("suggestions", [])) * 0.2 +
            (llm_analysis.get("quality_score", 0.5) * 0.5)
        ))

        return {
            "improvements_suggested": improvements_suggested,
            "priority": priority,
            "confidence": confidence,
            "total_issues": len(static_issues),
            "high_severity_issues": high_severity_count,
            "medium_severity_issues": medium_severity_count
        }
