import subprocess
import os
import time

def build_project(project_path: str) -> dict:
    '''Build the project and return result dict with: success (bool), stdout (str), stderr (str), duration (float), errors (list)'''
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "duration": 0.0,
        "errors": []
    }
    
    start_time = time.time()
    
    try:
        # Check if project path exists
        if not os.path.exists(project_path):
            result["errors"].append(f"Project path does not exist: {project_path}")
            return result

        # Define the build command
        # Using bash explicitly to ensure the script executes correctly
        cmd = ["bash", "rebuild.sh"]
        
        # Execute the build process
        process = subprocess.Popen(
            cmd,
            cwd=project_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        result["stdout"] = stdout
        result["stderr"] = stderr
        result["success"] = (process.returncode == 0)
        
        # Parse common C++ compilation error patterns
        # Standard formats: "file:line:column: error: ..." or "file:line: error: ..."
        for line in stderr.splitlines():
            if ": error:" in line:
                result["errors"].append(line.strip())
        
        # If return code is non-zero and no specific errors found, add generic failure
        if process.returncode != 0 and not result["errors"]:
            result["errors"].append(f"Build failed with exit code {process.returncode}")

    except Exception as e:
        result["errors"].append(str(e))
    
    result["duration"] = time.time() - start_time
    return result