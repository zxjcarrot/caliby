import subprocess
import time
import os

def run_benchmark(project_path: str) -> dict:
    '''Run benchmark and return result dict with: success (bool), stdout (str), stderr (str), duration (float)'''
    cmd = ["python3", "./benchmark/compare_hnsw.py", "--skip", "usearch,faiss"]
    
    start_time = time.perf_counter()
    try:
        # Execute the command from the project directory
        process = subprocess.run(
            cmd,
            cwd=project_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        duration = time.perf_counter() - start_time
        
        return {
            "success": process.returncode == 0,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "duration": duration
        }
    except Exception as e:
        duration = time.perf_counter() - start_time
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "duration": duration
        }