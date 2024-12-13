from typing import Dict, Any

from src.application.langchain.models.script_execution_tool_model import ScriptExecutionTool

def execute_bash_script(script_path: str, timeout: int = 120) -> Dict[str, Any]:
    """Execute a bash script and return its output."""
    executor = ScriptExecutionTool()
    try:
        # Pass the script path directly to the _run method
        result = executor._run(script_path=script_path, timeout=timeout)
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "script_path": script_path
        }