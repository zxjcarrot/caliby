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
        # Execute the test command from the project directory
        process = subprocess.run(
            ["bash", "./run_tests.sh"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=300 # 5 minute safety timeout
        )
        
        result["stdout"] = process.stdout
        result["stderr"] = process.stderr
        
        # Parse output for summary: e.g., "2 failed, 143 passed in 23.67s"
        # We look for the short test summary info or the final summary line
        summary_pattern = re.compile(r"(\d+)\s+failed.*(\d+)\s+passed|(\d+)\s+passed.*(\d+)\s+failed")
        
        passed = 0
        failed = 0
        
        # Search from bottom of output as summary is usually at the end
        for line in reversed(process.stdout.splitlines()):
            match = summary_pattern.search(line)
            if match:
                groups = match.groups()
                # Determine which groups matched based on order in regex
                if groups[0] is not None:
                    failed = int(groups[0])
                    passed = int(groups[1])
                else:
                    passed = int(groups[2])
                    failed = int(groups[3])
                break
        
        # If no failures found in pattern but passed tests exist, 
        # check for the "143 passed in 23.67s" case (0 failures)
        if failed == 0 and passed == 0:
            pass_only_match = re.search(r"(\d+)\s+passed", process.stdout)
            if pass_only_match:
                passed = int(pass_only_match.group(1))

        result["tests_passed"] = passed
        result["tests_failed"] = failed
        
        # Requirement: Map success based on specific threshold (up to 5 failing tests allowed)
        # Note: Usually success means 0 failures, but prompt specifies a tolerance.
        if failed <= 5 and (passed > 0 or failed >= 0):
            result["success"] = True
            
    except Exception as e:
        result["stderr"] += f"\nException during test execution: {str(e)}"
        result["success"] = False
        
    return result