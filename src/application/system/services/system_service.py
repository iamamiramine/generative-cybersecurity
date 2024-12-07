import subprocess
import os

def execute_script(script_number: int = 1) -> dict:
    """Execute a generated bash script and return its output"""
    script_path = f"generated_scripts/command_{script_number}.sh"
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script {script_path} not found. Generate a script first.")
    
    try:
        # Run the script and capture output
        result = subprocess.run(
            [script_path],
            capture_output=True,
            text=True,
            check=True  # This will raise CalledProcessError if script fails
        )
        
        return {
            "success": True,
            "output": result.stdout,
            "script_path": script_path
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": str(e),
            "output": e.stderr,
            "script_path": script_path
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "script_path": script_path
        }