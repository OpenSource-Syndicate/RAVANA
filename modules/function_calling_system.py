"""
Function Calling System for RAVANA AGI

This module implements advanced function calling capabilities for Ollama
and other LLM providers, allowing the system to use structured tools.
"""
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
import json
import inspect
from functools import wraps

from core.llm import async_safe_call_llm
from core.config import Config
from modules.failure_learning_system import FailureLearningSystem
from modules.physics_analysis_system import PhysicsAnalysisSystem

logger = logging.getLogger(__name__)


class FunctionRegistry:
    """Registry for managing available functions/tools that LLMs can call."""
    
    def __init__(self):
        self.functions = {}
        self.function_descriptions = {}
        
    def register(self, name: str, description: str, parameters: Dict[str, Any], 
                 function: Callable, return_value_description: str = ""):
        """
        Register a function that can be called by LLMs.
        
        Args:
            name: Name of the function
            description: Description of what the function does
            parameters: JSON schema describing function parameters
            function: The actual function to call
            return_value_description: Description of what the function returns
        """
        self.functions[name] = function
        self.function_descriptions[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "return_value_description": return_value_description
        }
        
    def get_function(self, name: str) -> Optional[Callable]:
        """Get a registered function by name."""
        return self.functions.get(name)
        
    def get_function_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the schema for a function by name."""
        return self.function_descriptions.get(name)
        
    def get_all_function_schemas(self) -> List[Dict[str, Any]]:
        """Get all registered function schemas."""
        return list(self.function_descriptions.values())


class FunctionCallingSystem:
    """System for managing advanced function calling capabilities."""
    
    def __init__(self, agi_system, blog_scheduler=None):
        self.agi_system = agi_system
        self.blog_scheduler = blog_scheduler
        self.config = Config()
        self.registry = FunctionRegistry()
        
        # Initialize the failure learning and physics analysis systems
        self.failure_learning_system = FailureLearningSystem(agi_system, blog_scheduler)
        self.physics_analysis_system = PhysicsAnalysisSystem(agi_system, blog_scheduler)
        
        # Register system functions
        self._register_system_functions()
        
    def _register_system_functions(self):
        """Register all core functions available to the system."""
        # Failure learning functions
        self.registry.register(
            name="analyze_failure",
            description="Analyze a failure to understand root causes and lessons learned",
            parameters={
                "type": "object",
                "properties": {
                    "failure_context": {"type": "string", "description": "Context of what was being attempted"},
                    "experiment_result": {"type": "object", "description": "Result from the failed experiment"},
                    "failure_details": {"type": "string", "description": "Specific details about what went wrong"}
                },
                "required": ["failure_context", "experiment_result"]
            },
            return_value_description="Analysis of the failure with root causes, lessons learned, and alternative approaches",
            function=self._analyze_failure_wrapper
        )
        
        self.registry.register(
            name="create_prototype_from_failure",
            description="Create a prototype system based on lessons learned from a failure",
            parameters={
                "type": "object",
                "properties": {
                    "failure_analysis": {"type": "object", "description": "Analysis of the failure to learn from"},
                    "context": {"type": "string", "description": "Context for the new prototype"}
                },
                "required": ["failure_analysis", "context"]
            },
            return_value_description="A prototype that addresses identified issues from the failure analysis",
            function=self._create_prototype_from_failure_wrapper
        )
        
        # Physics analysis functions
        self.registry.register(
            name="analyze_physics_problem",
            description="Analyze a physics problem and suggest appropriate formulas and approach",
            parameters={
                "type": "object",
                "properties": {
                    "problem_description": {"type": "string", "description": "Description of the physics problem"},
                    "known_values": {"type": "object", "description": "Dictionary of known variable values"},
                    "domain": {"type": "string", "description": "Optional physics domain to focus on"}
                },
                "required": ["problem_description"]
            },
            return_value_description="Analysis of the physics problem with suggested formulas and solution approach",
            function=self._analyze_physics_problem_wrapper
        )
        
        self.registry.register(
            name="search_physics_formulas",
            description="Search for physics formulas based on query and domain",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (formula name, variable, or keyword)"},
                    "domain": {"type": "string", "description": "Optional domain to filter results"}
                },
                "required": ["query"]
            },
            return_value_description="List of matching physics formulas with names, equations, and descriptions",
            function=self._search_physics_formulas_wrapper
        )
        
        # Memory and learning functions
        self.registry.register(
            name="apply_lessons_to_task",
            description="Apply lessons learned from past failures to a new task",
            parameters={
                "type": "object",
                "properties": {
                    "task_description": {"type": "string", "description": "Description of the new task"}
                },
                "required": ["task_description"]
            },
            return_value_description="Recommendations for avoiding past failures and improving the approach",
            function=self._apply_lessons_to_task_wrapper
        )
        
        # General utility functions
        self.registry.register(
            name="get_current_datetime",
            description="Get the current date and time",
            parameters={"type": "object", "properties": {}},
            return_value_description="Current date and time in ISO format",
            function=self._get_current_datetime
        )
    
    async def _analyze_failure_wrapper(self, failure_context: str, experiment_result: Dict[str, Any], 
                                      failure_details: str = None) -> Dict[str, Any]:
        """Wrapper for the failure analysis function."""
        return await self.failure_learning_system.analyze_failure(failure_context, experiment_result, failure_details)
    
    async def _create_prototype_from_failure_wrapper(self, failure_analysis: Dict[str, Any], 
                                                    context: str) -> Dict[str, Any]:
        """Wrapper for the prototype creation function."""
        return await self.failure_learning_system.create_prototype_from_failure(failure_analysis, context)
    
    async def _analyze_physics_problem_wrapper(self, problem_description: str, 
                                              known_values: Dict[str, Any] = None,
                                              domain: str = None) -> Dict[str, Any]:
        """Wrapper for physics problem analysis."""
        domain_enum = None
        if domain:
            try:
                from modules.physics_analysis_system import PhysicsDomain
                domain_enum = PhysicsDomain(domain.lower())
            except:
                logger.warning(f"Invalid physics domain: {domain}")
        
        return await self.physics_analysis_system.analyze_physics_problem(
            problem_description, known_values, domain_enum)
    
    async def _search_physics_formulas_wrapper(self, query: str, domain: str = None) -> List[Dict[str, Any]]:
        """Wrapper for physics formula search."""
        domain_enum = None
        if domain:
            try:
                from modules.physics_analysis_system import PhysicsDomain
                domain_enum = PhysicsDomain(domain.lower())
            except:
                logger.warning(f"Invalid physics domain: {domain}")
        
        formulas = self.physics_analysis_system.search_formulas(query, domain_enum)
        return [
            {
                "name": f.name,
                "formula": f.formula,
                "description": f.description,
                "domain": f.domain.value,
                "variables": f.variables,
                "applications": f.applications
            } for f in formulas
        ]
    
    async def _apply_lessons_to_task_wrapper(self, task_description: str) -> Dict[str, Any]:
        """Wrapper for applying lessons to a new task."""
        return await self.failure_learning_system.apply_lessons_to_task(task_description)
    
    def _get_current_datetime(self) -> str:
        """Get the current date and time."""
        return datetime.now().isoformat()
    
    def get_available_functions(self) -> List[Dict[str, Any]]:
        """Get list of all available functions with their schemas."""
        return self.registry.get_all_function_schemas()
    
    async def execute_function_call(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a function call with the given arguments.
        
        Args:
            function_name: Name of the function to call
            arguments: Arguments to pass to the function
            
        Returns:
            Result of the function call
        """
        logger.info(f"Executing function call: {function_name} with args {arguments}")
        
        function = self.registry.get_function(function_name)
        if not function:
            logger.error(f"Function {function_name} not found in registry")
            return {"error": f"Function {function_name} not found"}
        
        try:
            # Check if function is async
            if asyncio.iscoroutinefunction(function):
                result = await function(**arguments)
            else:
                result = function(**arguments)
            
            logger.info(f"Function {function_name} executed successfully")
            return {"result": result}
        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")
            return {"error": f"Error in function execution: {str(e)}"}
    
    async def call_with_functions(self, prompt: str, available_functions: List[str] = None) -> Dict[str, Any]:
        """
        Call the LLM with function calling capabilities.
        
        Args:
            prompt: The prompt to send to the LLM
            available_functions: List of function names that can be called (if None, all are available)
            
        Returns:
            Dictionary with the response and any function calls made
        """
        # Get all function schemas or filter by available functions
        all_schemas = self.get_available_functions()
        if available_functions:
            schemas = [s for s in all_schemas if s["name"] in available_functions]
        else:
            schemas = all_schemas
        
        # Create a system message that describes the available functions
        system_message = f"""
        You are an advanced AI assistant with access to various tools/functions.
        
        Available tools/functions:
        {json.dumps(schemas, indent=2)}
        
        When you need to use a tool, respond with a JSON object containing:
        {{
            "function_call": {{
                "name": "function_name",
                "arguments": {{"arg1": "value1", "arg2": "value2"}}
            }}
        }}
        
        You may make multiple function calls, and you should only call functions that are
        appropriate for the user's request. After executing function(s), provide your
        final response based on the results.
        
        The current date and time is: {datetime.now().isoformat()}
        """
        
        # Combine system message and user prompt
        full_prompt = f"{system_message}\n\nUser Query: {prompt}"
        
        try:
            response = await async_safe_call_llm(full_prompt)
            logger.info("LLM response with potential function calls received")
            
            # Try to parse function calls from response
            try:
                parsed_response = json.loads(response)
                function_calls = []
                
                # Check if response contains function calls
                if isinstance(parsed_response, dict) and "function_call" in parsed_response:
                    function_calls.append(parsed_response["function_call"])
                elif isinstance(parsed_response, list):
                    # Multiple function calls
                    for item in parsed_response:
                        if isinstance(item, dict) and "function_call" in item:
                            function_calls.append(item["function_call"])
                
                # Execute function calls if any
                function_results = []
                for call in function_calls:
                    result = await self.execute_function_call(call["name"], call["arguments"])
                    function_results.append({
                        "name": call["name"],
                        "arguments": call["arguments"],
                        "result": result
                    })
                
                # Return both the original response and function results
                return {
                    "original_response": response,
                    "parsed_response": parsed_response,
                    "function_calls": function_calls,
                    "function_results": function_results
                }
                
            except json.JSONDecodeError:
                # The response wasn't JSON, so no function calls were made
                logger.debug("No function calls detected in response")
                return {
                    "original_response": response,
                    "function_calls": [],
                    "function_results": []
                }
                
        except Exception as e:
            logger.error(f"Error in function calling LLM: {e}")
            return {
                "error": str(e),
                "function_calls": [],
                "function_results": []
            }
    
    async def intelligent_tool_selection(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Intelligently select which tools/functions would be most appropriate for a given task.
        
        Args:
            task_description: Description of the task to be performed
            
        Returns:
            List of function calls that would be appropriate for the task
        """
        # Analyze the task to determine which functions would be most useful
        analysis_prompt = f"""
        Analyze this task and determine which functions/tools would be most appropriate:

        Task: {task_description}
        
        Available functions:
        {json.dumps(self.get_available_functions(), indent=2)}
        
        Identify which functions would be most useful for this task and why.
        Return your analysis as JSON with these keys:
        - relevant_functions: List of function names that would be useful
        - reasoning: Brief explanation for why each function was selected
        - execution_order: Suggested order to execute the functions
        - parameters: Suggested parameters for each function call
        """
        
        try:
            response = await async_safe_call_llm(analysis_prompt)
            
            try:
                analysis = json.loads(response)
            except json.JSONDecodeError:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    analysis = json.loads(json_str)
                else:
                    # Default fallback if parsing fails
                    analysis = {
                        "relevant_functions": [],
                        "reasoning": [],
                        "execution_order": [],
                        "parameters": {}
                    }
            
            # Create suggested function calls based on the analysis
            suggested_calls = []
            for func_name in analysis.get("relevant_functions", []):
                params = analysis.get("parameters", {}).get(func_name, {})
                suggested_calls.append({
                    "name": func_name,
                    "arguments": params,
                    "reasoning": analysis.get("reasoning", {}).get(func_name, "Selected by AI analysis")
                })
            
            return suggested_calls
            
        except Exception as e:
            logger.error(f"Error in intelligent tool selection: {e}")
            return []
    
    async def execute_task_with_functions(self, task_description: str) -> Dict[str, Any]:
        """
        Execute a task using appropriate functions/tools selected by the system.
        
        Args:
            task_description: Description of the task to execute
            
        Returns:
            Dictionary with task results and function execution information
        """
        logger.info(f"Executing task with function selection: {task_description[:100]}...")
        
        # First, intelligently select appropriate functions
        suggested_calls = await self.intelligent_tool_selection(task_description)
        
        task_results = {
            "task": task_description,
            "function_calls": [],
            "execution_results": [],
            "final_analysis": None
        }
        
        # Execute each suggested function call
        for call in suggested_calls:
            logger.info(f"Executing function: {call['name']}")
            
            result = await self.execute_function_call(call['name'], call.get('arguments', {}))
            
            task_results["function_calls"].append({
                "name": call['name'],
                "arguments": call.get('arguments', {}),
                "reasoning": call.get('reasoning', 'Selected by system')
            })
            
            task_results["execution_results"].append({
                "function_name": call['name'],
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
        
        # If we executed any functions, provide a final analysis
        if task_results["function_calls"]:
            analysis_prompt = f"""
            Task: {task_description}
            
            Function Calls Made:
            {json.dumps(task_results["function_calls"], indent=2)}
            
            Results:
            {json.dumps(task_results["execution_results"], indent=2)}
            
            Provide a final analysis of the task based on the function execution results.
            """
            
            try:
                final_analysis = await async_safe_call_llm(analysis_prompt)
                task_results["final_analysis"] = final_analysis
            except Exception as e:
                logger.error(f"Error in final analysis: {e}")
                task_results["final_analysis"] = f"Could not generate final analysis: {e}"
        
        return task_results