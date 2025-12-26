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
    original_cwd = os.getcwd()
    
    try:
        # Change directory to the project path
        os.chdir(project_path)
        
        # Define the build script command
        cmd = ["/bin/bash", "rebuild.sh"]
        
        # Use subprocess to execute the build command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        result["stdout"] = stdout
        result["stderr"] = stderr
        result["success"] = (process.returncode == 0)
        
        # Parse common compilation/make error patterns
        # Look for typical C++ error indicators
        for line in stderr.splitlines():
            if ": error:" in line or "fatal error:" in line:
                result["errors"].append(line.strip())
        
        # If return code is non-zero but no specific error found, add general error
        if not result["success"] and not result["errors"]:
            result["errors"].append(f"Build failed with exit code {process.returncode}")

    except Exception as e:
        result["success"] = False
        result["errors"].append(str(e))
    finally:
        result["duration"] = time.time() - start_time
        # Revert to the original working directory
        os.chdir(original_cwd)
        
    return result