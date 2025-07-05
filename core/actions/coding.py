import asyncio
import logging
import subprocess
from typing import Dict, Any, List
from core.actions.action import Action
from core.actions.exceptions import ActionException
from modules.decision_engine.llm import call_llm

logger = logging.getLogger(__name__)

CODE_GENERATION_PROMPT = """
You are an expert AI programmer. Your task is to write a Python script to test the following hypothesis.
The script should be self-contained and not require any external files unless they are standard Python libraries.
The script should print any results or outputs to standard output.

Hypothesis: {hypothesis}

Test Plan: {test_plan}

Write the Python code.
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
            {"name": "file_path", "type": "string", "description": "The path to the file to write the code to.", "required": True},
            {"name": "hypothesis", "type": "string", "description": "The hypothesis to test.", "required": True},
            {"name": "test_plan", "type": "string", "description": "The plan to test the hypothesis.", "required": True},
        ]

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        file_path = kwargs.get("file_path")
        hypothesis = kwargs.get("hypothesis")
        test_plan = kwargs.get("test_plan")

        if not all([file_path, hypothesis, test_plan]):
            raise ActionException("Missing required parameters for write_python_code.")

        logger.info(f"Generating Python code for hypothesis: {hypothesis}")
        prompt = CODE_GENERATION_PROMPT.format(hypothesis=hypothesis, test_plan=test_plan)
        
        try:
            # Using the existing llm_call for code generation
            response = await asyncio.to_thread(call_llm, prompt)
            code = response
            
            # Clean up the code if it's wrapped in markdown
            if code.startswith("```python"):
                code = code[len("```python"):].strip()
            if code.endswith("```"):
                code = code[:-len("```")].strip()

            with open(file_path, "w") as f:
                f.write(code)
            
            logger.info(f"Successfully wrote Python code to {file_path}")
            return {"status": "success", "file_path": file_path, "code": code}
        except Exception as e:
            logger.error(f"Failed to write Python code to {file_path}: {e}", exc_info=True)
            raise ActionException(f"Failed to generate or write Python code: {e}")


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
            {"name": "file_path", "type": "string", "description": "The path to the Python script to execute.", "required": True}
        ]

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        file_path = kwargs.get("file_path")
        if not file_path:
            raise ActionException("Missing file_path parameter for execute_python_file.")

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
                logger.error(f"Script {file_path} executed with error: {error}")
                return {"status": "error", "output": output, "error": error, "return_code": process.returncode}
            
            logger.info(f"Script {file_path} executed successfully. Output: {output}")
            return {"status": "success", "output": output, "error": error, "return_code": process.returncode}

        except Exception as e:
            logger.error(f"Failed to execute Python script {file_path}: {e}", exc_info=True)
            raise ActionException(f"An unexpected error occurred while executing the Python script: {e}") 