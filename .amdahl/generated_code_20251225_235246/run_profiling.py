import os
import sys
import subprocess
import time
import re
import signal
import shlex

def run_profiling(project_path: str, benchmark_cmd: str) -> dict:
    """
    Run perf profiling on the Caliby project and return performance insights.
    """
    initial_dir = os.getcwd()
    result = {
        "success": False,
        "perf_report": "",
        "perf_stat": "",
        "perf_annotate": "",
        "flamegraph_path": None,
        "stderr": ""
    }

    try:
        # Change to project directory
        os.chdir(project_path)
        print(f"Working directory: {os.getcwd()}", file=sys.stderr)

        # 1. Start the benchmark process
        # Split cmd safely; assuming benchmark_cmd is provided as a string like "python3 ..."
        args = shlex.split(benchmark_cmd)
        print(f"Executing benchmark: {' '.join(args)}", file=sys.stderr)
        
        proc = subprocess.Popen(
            args, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid # Use process group to ensure we can clean up
        )
        pid = proc.pid
        print(f"Benchmark started with PID: {pid}", file=sys.stderr)

        # Wait for initialization (10 seconds as requested)
        print("Waiting 10s for benchmark initialization...", file=sys.stderr)
        time.sleep(10)

        if proc.poll() is not None:
            raise ChildProcessError(f"Benchmark process exited prematurely with code {proc.returncode}")

        # 2. Run perf record
        perf_data_path = "./perf.data"
        record_cmd = ["sudo", "perf", "record", "-F", "997", "-g", "-p", str(pid), "-o", perf_data_path, "sleep", "3"]
        print(f"Executing: {' '.join(record_cmd)}", file=sys.stderr)
        subprocess.run(record_cmd, check=True)

        # 3. Run perf stat for microarch metrics
        stat_cmd = ["sudo", "perf", "stat", "-p", str(pid), "sleep", "3"]
        print(f"Executing: {' '.join(stat_cmd)}", file=sys.stderr)
        stat_proc = subprocess.run(stat_cmd, capture_output=True, text=True, errors='replace')
        result["perf_stat"] = stat_proc.stderr # perf stat outputs metrics to stderr

        # 4. Generate perf report
        # --no-children focuses on self-time; -n includes sample counts; folded for LLM analysis
        report_cmd = ["sudo", "perf", "report", "--stdio", "-i", perf_data_path, "--no-children", "-n"]
        print(f"Executing: {' '.join(report_cmd)}", file=sys.stderr)
        report_proc = subprocess.run(report_cmd, capture_output=True, text=True, errors='replace')
        
        # Keep top 100 lines of report as requested
        report_lines = report_proc.stdout.splitlines()
        result["perf_report"] = "\n".join(report_lines[:100])

        # 5. Extract hottest function and annotate
        try:
            hottest_func = None
            # Regex to find the first line starting with a percentage (typical perf stdio format)
            for line in report_lines:
                # Format usually: 15.23%     1234  caliby  caliby_lib.so  [.] HottestFunc
                match = re.search(r'^\s*([0-9.]+)%\s+\d+\s+.*?\s+.*?\s+\[\.\]\s+(\S+)', line)
                if match:
                    hottest_func = match.group(2)
                    print(f"Hottest function identified: {hottest_func}", file=sys.stderr)
                    break
            
            if hottest_func:
                annotate_cmd = ["sudo", "perf", "annotate", "--stdio", "-s", hottest_func, "-i", perf_data_path]
                print(f"Executing: {' '.join(annotate_cmd)}", file=sys.stderr)
                anno_proc = subprocess.run(annotate_cmd, capture_output=True, text=True, errors='replace', timeout=30)
                result["perf_annotate"] = "\n".join(anno_proc.stdout.splitlines()[:150])
        except Exception as e:
            result["perf_annotate"] = f"Assembly annotation failed: {str(e)}"

        # 6. Optional FlameGraph
        fg_tool_path = "/home/zxjcarrot/Workspace/FlameGraph"
        if os.path.exists(fg_tool_path):
            try:
                print("Generating FlameGraph...", file=sys.stderr)
                script_collapse = os.path.join(fg_tool_path, "stackcollapse-perf.pl")
                script_fg = os.path.join(fg_tool_path, "flamegraph.pl")
                
                p1 = subprocess.Popen(["sudo", "perf", "script", "-i", perf_data_path], stdout=subprocess.PIPE)
                p2 = subprocess.Popen([script_collapse], stdin=p1.stdout, stdout=subprocess.PIPE)
                p1.stdout.close()
                
                fg_out_path = os.path.join(project_path, "flamegraph.svg")
                with open(fg_out_path, "w") as f:
                    subprocess.run([script_fg], stdin=p2.stdout, stdout=f)
                result["flamegraph_path"] = fg_out_path
            except Exception as e:
                print(f"Flamegraph generation failed: {e}", file=sys.stderr)

        result["success"] = True

    except Exception as e:
        result["stderr"] = f"Error during profiling: {str(e)}"
    finally:
        # Cleanup: Stop the benchmark process if still running
        if 'proc' in locals() and proc.poll() is None:
            print("Terminating benchmark process...", file=sys.stderr)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except:
                proc.terminate()
        
        # Capture remaining benchmark output
        if 'proc' in locals():
            stdout, stderr = proc.communicate()
            result["stderr"] += f"\n--- Benchmark Stderr ---\n{stderr}"
        
        os.chdir(initial_dir)

    return result