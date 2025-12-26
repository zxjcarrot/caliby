import os
import sys
import subprocess
import time
import re
import signal
import shlex

def run_profiling(project_path: str, benchmark_cmd: str) -> dict:
    """
    Run perf profiling on the Caliby project and return performance metrics and reports.
    """
    original_cwd = os.getcwd()
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

        # Split benchmark_cmd into list for Popen to avoid shell=True PID issues
        cmd_args = shlex.split(benchmark_cmd)
        
        print(f"Starting benchmark command: {benchmark_cmd}", file=sys.stderr)
        proc = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for initialization (requirement: > 10 seconds)
        print("Waiting 12 seconds for benchmark to initialize and start building/searching...", file=sys.stderr)
        time.sleep(12)

        if proc.poll() is not None:
            raise ChildProcessError(f"Benchmark process died prematurely with return code {proc.returncode}")

        pid = proc.pid
        perf_data_path = "./perf.data"
        print(f"Profiling PID {pid}, writing to {perf_data_path}", file=sys.stderr)

        # 1. Perf Record (Profile for 3 seconds)
        record_cmd = ["sudo", "perf", "record", "-F", "997", "-g", "-p", str(pid), "-o", perf_data_path, "--", "sleep", "3"]
        print(f"Executing: {' '.join(record_cmd)}", file=sys.stderr)
        subprocess.run(record_cmd, check=True, capture_output=True)

        # 2. Perf Stat (Microarch metrics for 3 seconds)
        stat_cmd = ["sudo", "perf", "stat", "-p", str(pid), "sleep", "3"]
        print(f"Executing: {' '.join(stat_cmd)}", file=sys.stderr)
        stat_res = subprocess.run(stat_cmd, capture_output=True, text=True, errors='replace')
        result["perf_stat"] = stat_res.stderr # perf stat outputs to stderr

        # 3. Generate Perf Report
        report_cmd = ["sudo", "perf", "report", "--stdio", "-i", perf_data_path, "--no-children", "-n"]
        print(f"Executing: {' '.join(report_cmd)}", file=sys.stderr)
        report_res = subprocess.run(report_cmd, capture_output=True)
        perf_report = report_res.stdout.decode('utf-8', errors='replace')
        
        # Take top 100 lines for the LLM report
        result["perf_report"] = "\n".join(perf_report.splitlines()[:100])

        # 4. Extract Hottest Function and Annotate
        try:
            hottest_func = None
            # Look for line with highest %: "[ ] 12.34%  caliby  libcalico.so  [.] HNSW::search"
            for line in perf_report.splitlines():
                match = re.search(r'^\s*(\d+\.\d+)%.*?\s+(\S+)\s*$', line)
                if match:
                    hottest_func = match.group(2)
                    print(f"Found hottest function for annotation: {hottest_func}", file=sys.stderr)
                    break
            
            if hottest_func:
                annotate_cmd = ["sudo", "perf", "annotate", "--stdio", "-s", hottest_func, "-i", perf_data_path]
                print(f"Executing: {' '.join(annotate_cmd)}", file=sys.stderr)
                ann_res = subprocess.run(annotate_cmd, capture_output=True, timeout=30)
                ann_out = ann_res.stdout.decode('utf-8', errors='replace')
                result["perf_annotate"] = "\n".join(ann_out.splitlines()[:150])
        except Exception as e:
            result["perf_annotate"] = f"Assembly annotation failed: {str(e)}"

        # 5. Optional FlameGraph
        fg_dir = "/home/zxjcarrot/Workspace/FlameGraph"
        if os.path.isdir(fg_dir):
            try:
                print("Generating FlameGraph...", file=sys.stderr)
                script_path = os.path.join(fg_dir, "stackcollapse-perf.pl")
                fg_script = os.path.join(fg_dir, "flamegraph.pl")
                
                # sudo perf script | stackcollapse-perf.pl | flamegraph.pl > out.svg
                p1 = subprocess.Popen(["sudo", "perf", "script", "-i", perf_data_path], stdout=subprocess.PIPE)
                p2 = subprocess.Popen([script_path], stdin=p1.stdout, stdout=subprocess.PIPE)
                with open("flamegraph.svg", "w") as f:
                    subprocess.run([fg_script], stdin=p2.stdout, stdout=f)
                result["flamegraph_path"] = os.path.abspath("flamegraph.svg")
            except Exception as e:
                print(f"FlameGraph generation failed: {e}", file=sys.stderr)

        # Cleanup process if still running
        if proc.poll() is None:
            print("Terminating benchmark process...", file=sys.stderr)
            proc.terminate()
            time.sleep(1)
            proc.kill()

        result["success"] = True

    except Exception as e:
        result["stderr"] = f"Error during profiling: {str(e)}"
        print(result["stderr"], file=sys.stderr)
    finally:
        os.chdir(original_cwd)

    return result