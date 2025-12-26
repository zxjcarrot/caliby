import re

def extract_metrics(output: str) -> dict:
    """
    Extract performance metrics from Caliby benchmark output.
    Returns a dictionary of metric names and their float values.
    """
    metrics = {}
    if not output:
        return metrics

    # 1. Try to parse the Summary Table format (Primary Method)
    # Library         Build(s)     Size(MB)     QPS          P50(ms)    P95(ms)    P99(ms)    Recall@10
    # ----------------------------------------------------------------------------------------------------
    # Caliby          21.55        488.28       9933.4       0.102      0.135      0.155      0.8977
    table_pattern = r"Caliby\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)"
    table_match = re.search(table_pattern, output)
    if table_match:
        try:
            metrics["build_time_s"] = float(table_match.group(1))
            metrics["size_mb"] = float(table_match.group(2))
            metrics["qps"] = float(table_match.group(3))
            metrics["p50_ms"] = float(table_match.group(4))
            metrics["p95_ms"] = float(table_match.group(5))
            metrics["p99_ms"] = float(table_match.group(6))
            metrics["recall_at_10"] = float(table_match.group(7))
            return metrics
        except (ValueError, IndexError):
            pass

    # 2. Fallback: Parse individual lines using regex patterns if table parsing fails
    patterns = {
        "build_time_s": r"Build time:\s*([\d\.]+)s",
        "size_mb": r"Estimated index size:\s*([\d\.]+)\s*MB",
        "qps": r"QPS:\s*Caliby\s*\(([\d\.]+)\s*queries/sec\)",
        "recall_at_10": r"Best Recall@10:\s*Caliby\s*\(([\d\.]+)\)",
        "p50_ms": r"Lowest P50 Latency:\s*Caliby\s*\(([\d\.]+)\s*ms\)",
        "throughput": r"Processed\s+\d+/\d+\s+queries\s+in\s+([\d\.]+)s",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except ValueError:
                continue

    # 3. Clean up generic names or handle alternative naming
    if "recall_at_10" not in metrics:
        recall_match = re.search(r"Recall@\d+\s+([\d\.]+)", output)
        if recall_match:
            metrics["recall"] = float(recall_match.group(1))

    return metrics