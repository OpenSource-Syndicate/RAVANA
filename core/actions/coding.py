import asyncio
import logging
import re
import subprocess
from typing import Dict, Any, List
from core.actions.action import Action
from core.actions.exceptions import ActionException
from core.llm import call_llm

logger = logging.getLogger(__name__)

CODE_GENERATION_PROMPT = """
[ROLE DEFINITION]
You are an expert AI programmer with deep knowledge of software engineering principles and best practices.

[CONTEXT]
Hypothesis to test: {hypothesis}
Test plan: {test_plan}

[TASK INSTRUCTIONS]
Generate high-quality Python code by following these steps:
1. Analyze the hypothesis and test plan thoroughly
2. Design a robust solution architecture
3. Implement with clean, maintainable code
4. Include comprehensive error handling
5. Add clear documentation and comments
6. Validate against all requirements

[REASONING FRAMEWORK]
Apply software engineering best practices:
1. Decompose complex problems into manageable modules
2. Choose appropriate algorithms and data structures
3. Prioritize code readability and maintainability
4. Implement defensive programming techniques
5. Consider performance and scalability requirements
6. Plan for future extensibility

[OUTPUT REQUIREMENTS]
Provide complete, executable Python code with:
- Clear, descriptive variable and function names
- Comprehensive inline documentation
- Proper error handling and edge case management
- Efficient algorithms and data structures
- Adherence to Python conventions and best practices
- Confidence score for solution correctness (0.0-1.0)

[SAFETY CONSTRAINTS]
- Avoid security vulnerabilities (injection, buffer overflows, etc.)
- Prevent resource leaks and memory issues
- Ensure code does not perform unintended actions
- Validate all inputs and outputs
- Follow secure coding practices

The script should be self-contained and not require any external files unless they are standard Python libraries.
The script should print any results or outputs to standard output.
"""


class WritePythonCodeAction(Action):
    @property
    def name(self) -> str:
        return "write_python_code"

    @property
    def description(self) -> str:
        return "Writes a Python script to test a hypothesis."

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {"name": "file_path", "type": "string",
                "description": "The path to the file to write the code to.", "required": True},
            {"name": "hypothesis", "type": "string",
                "description": "The hypothesis to test.", "required": True},
            {"name": "test_plan", "type": "string",
                "description": "The plan to test the hypothesis.", "required": True},
        ]

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        file_path = kwargs.get("file_path")
        hypothesis = kwargs.get("hypothesis")
        test_plan = kwargs.get("test_plan")

        if not all([file_path, hypothesis, test_plan]):
            raise ActionException(
                "Missing required parameters for write_python_code.")

        logger.info(f"Generating Python code for hypothesis: {hypothesis}")
        prompt = CODE_GENERATION_PROMPT.format(
            hypothesis=hypothesis, test_plan=test_plan)

        try:
            # Using the existing llm_call for code generation
            response = await asyncio.to_thread(call_llm, prompt)

            # Extract code from the response using regex
            code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                # If no markdown block is found, assume the whole response is code
                code = response.strip()

            with open(file_path, "w") as f:
                f.write(code)

            logger.info(f"Successfully wrote Python code to {file_path}")
            return {"status": "success", "file_path": file_path, "code": code}
        except Exception as e:
            logger.error(
                f"Failed to write Python code to {file_path}: {e}", exc_info=True)
            raise ActionException(
                f"Failed to generate or write Python code: {e}")


class ExecutePythonFileAction(Action):
    @property
    def name(self) -> str:
        return "execute_python_file"

    @property
    def description(self) -> str:
        return "Executes a Python script and captures its output."

    @property
    def parameters(self) -> List[Dict[str, Any]]:
        return [
            {"name": "file_path", "type": "string",
                "description": "The path to the Python script to execute.", "required": True}
        ]

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        file_path = kwargs.get("file_path")
        if not file_path:
            raise ActionException(
                "Missing file_path parameter for execute_python_file.")

        logger.info(f"Executing Python script: {file_path}")
        try:
            process = await asyncio.create_subprocess_shell(
                f"python {file_path}",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            output = stdout.decode().strip()
            error = stderr.decode().strip()

            if process.returncode != 0:
                logger.error(
                    f"Script {file_path} executed with error: {error}")
                return {"status": "error", "output": output, "error": error, "return_code": process.returncode}

            logger.info(
                f"Script {file_path} executed successfully. Output: {output}")
            return {"status": "success", "output": output, "error": error, "return_code": process.returncode}

        except Exception as e:
            logger.error(
                f"Failed to execute Python script {file_path}: {e}", exc_info=True)
            raise ActionException(
                f"An unexpected error occurred while executing the Python script: {e}")
