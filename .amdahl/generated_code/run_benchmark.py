import subprocess
import time
import os

def run_benchmark(project_path: str) -> dict:
    '''Run benchmark and return result dict with: success (bool), stdout (str), stderr (str), duration (float)'''
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "duration": 0.0
    }
    
    # Define the command to execute
    cmd = ["python3", "./benchmark/compare_hnsw.py", "--skip", "usearch,faiss"]
    
    start_time = time.perf_counter()
    try:
        # Execute the process from the specified project directory
        process = subprocess.run(
            cmd,
            cwd=project_path,
            capture_output=True,
            text=True,
            check=False
        )
        
        result["stdout"] = process.stdout
        result["stderr"] = process.stderr
        result["success"] = (process.returncode == 0)
        
    except Exception as e:
        result["stderr"] = str(e)
        result["success"] = False
    finally:
        end_time = time.perf_counter()
        result["duration"] = end_time - start_time
        
    return result