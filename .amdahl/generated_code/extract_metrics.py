import re

def extract_metrics(output: str) -> dict:
    """
    Extract performance metrics from Caliby benchmark output.
    Returns a dictionary of metric names and numeric values.
    """
    metrics = {}
    if not output:
        return metrics

    # 1. Try strategy: Parse the summary table row for Caliby
    # Format: Caliby  21.55  488.28  9933.4  0.102  0.135  0.155  0.8977
    table_pattern = re.compile(
        r"Caliby\s+(?P<build_s>[\d.]+)\s+(?P<size_mb>[\d.]+)\s+(?P<qps>[\d.]+)\s+"
        r"(?P<p50_ms>[\d.]+)\s+(?P<p95_ms>[\d.]+)\s+(?P<p99_ms>[\d.]+)\s+(?P<recall>[\d.]+)"
    )
    
    table_match = table_pattern.search(output)
    if table_match:
        for key, val in table_match.groupdict().items():
            try:
                metrics[key] = float(val)
            except ValueError:
                continue

    # 2. Try strategy: Regex fallback for specific labels if table parsing fails or is incomplete
    patterns = {
        "qps": r"QPS[:\s]+([\d.]+)",
        "build_time_s": r"Build time[:\s]+([\d.]+)s",
        "index_size_mb": r"index size[:\s]+([\d.]+)\s+MB",
        "p50_ms": r"P50\(ms\)[:\s]+([\d.]+)",
        "p95_ms": r"P95\(ms\)[:\s]+([\d.]+)",
        "p99_ms": r"P99\(ms\)[:\s]+([\d.]+)",
        "recall": r"Recall@\d+[:\s]+([\d.]+)",
        "throughput": r"([\d.]+)\s+queries/sec"
    }

    for key, pattern in patterns.items():
        if key not in metrics:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    metrics[key] = float(match.group(1))
                except (ValueError, IndexError):
                    continue

    # 3. Handle WINNERS section aliases
    winner_patterns = {
        "highest_qps": r"Highest QPS:\s+Caliby\s+\(([\d.]+)\)",
        "best_recall": r"Best Recall@\d+:\s+Caliby\s+\(([\d.]+)\)",
        "lowest_p50": r"Lowest P50 Latency:\s+Caliby\s+\(([\d.]+)\)",
        "smallest_index": r"Smallest Index:\s+Caliby\s+\(([\d.]+)\)",
        "fastest_build": r"Fastest Build:\s+Caliby\s+\(([\d.]+)\)"
    }

    for key, pattern in winner_patterns.items():
        match = re.search(pattern, output)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except (ValueError, IndexError):
                continue

    return metrics