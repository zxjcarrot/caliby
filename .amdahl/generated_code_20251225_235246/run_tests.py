import subprocess
import os
import re

def run_tests(project_path: str) -> dict:
    '''Run tests and return result dict with: success (bool), stdout (str), stderr (str), tests_passed (int), tests_failed (int)'''
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "tests_passed": 0,
        "tests_failed": 0
    }
    
    try:
        # Execute the test command from the specified project directory
        process = subprocess.Popen(
            ["bash", "./run_tests.sh"],
            cwd=project_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        result["stdout"] = stdout
        result["stderr"] = stderr

        # Regex to parse the summary line: e.g., "2 failed, 143 passed in 23.67s"
        # We look for the patterns in the short test summary info
        summary_pattern = r"(\d+)\s+failed.*(\d+)\s+passed|(\d+)\s+passed"
        
        # Search from the bottom of the output for the summary
        lines = stdout.splitlines()
        for line in reversed(lines):
            if "passed" in line and ("failed" in line or "====" in line):
                fail_match = re.search(r"(\d+)\s+failed", line)
                pass_match = re.search(r"(\d+)\s+passed", line)
                
                if fail_match:
                    result["tests_failed"] = int(fail_match.group(1))
                if pass_match:
                    result["tests_passed"] = int(pass_match.group(1))
                break

        # Check success criteria: success=True if all tests pass (0 failures)
        # Based on requirements, we are allowed up to 5 failing tests for the process to be "acceptable" 
        # but the standard definition of success in a test runner is 0 failures. 
        # Given "Return success=True if all tests pass", we check for 0 failures:
        if result["tests_failed"] == 0 and result["tests_passed"] > 0:
            result["success"] = True
        else:
            result["success"] = False

    except Exception as e:
        result["stderr"] = str(e)
        result["success"] = False

    return result