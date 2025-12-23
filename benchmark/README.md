# Caliby Benchmarks

Comprehensive benchmarks comparing Caliby against other popular vector search libraries.

## HNSW Benchmark

Compare HNSW implementations across Caliby, Usearch, and Faiss using the SIFT1M dataset.

### Requirements

```bash
# Install benchmark dependencies
pip install usearch faiss-cpu numpy

# Or install all at once
pip install usearch faiss-cpu numpy
```

### Usage

Run the full benchmark:
```bash
python compare_hnsw.py
```

Customize parameters:
```bash
python compare_hnsw.py --M 32 --ef-construction 400 --ef-search 100
```

Skip specific libraries:
```bash
python compare_hnsw.py --skip usearch faiss  # Only benchmark caliby
```

Specify data directory:
```bash
python compare_hnsw.py --data-dir /path/to/sift1m
```

### Command-line Options

- `--data-dir`: Directory for SIFT1M dataset (default: `./sift1m`)
- `--M`: HNSW M parameter - controls connectivity (default: 16)
- `--ef-construction`: Construction time search depth (default: 200)
- `--ef-search`: Query time search depth (default: 50)
- `--skip`: Libraries to skip (choices: `caliby`, `usearch`, `faiss`)

### Metrics

The benchmark measures:

1. **Build Time**: Time to construct the index
2. **Index Size**: Disk/memory footprint in MB
3. **QPS**: Queries per second (throughput)
4. **Latency**: P50, P95, P99 percentiles in milliseconds
5. **Recall@10**: Accuracy compared to ground truth

### Dataset

The benchmark uses [SIFT1M](http://corpus-texmex.irisa.fr/):
- 1,000,000 base vectors (128 dimensions)
- 10,000 query vectors
- Ground truth for Recall@k computation

The dataset is automatically downloaded on first run (~200MB compressed).

### Example Output

```
==================================================================================================
BENCHMARK RESULTS SUMMARY
==================================================================================================

Library         Build(s)     Size(MB)     QPS          P50(ms)    P95(ms)    P99(ms)    Recall@10   
----------------------------------------------------------------------------------------------------
Usearch         45.23        345.12       8234.5       0.121      0.234      0.456      0.9823      
Faiss           52.18        389.45       7891.2       0.127      0.245      0.501      0.9845      
Caliby          48.76        356.78       7654.3       0.131      0.252      0.523      0.9801      

==================================================================================================

WINNERS:
  🏆 Highest QPS:        Usearch (8234.5 queries/sec)
  🏆 Best Recall@10:     Faiss (0.9845)
  🏆 Lowest P50 Latency: Usearch (0.121 ms)
  🏆 Smallest Index:     Usearch (345.12 MB)
  🏆 Fastest Build:      Usearch (45.23 s)

==================================================================================================
```

### Interpreting Results

- **QPS**: Higher is better - measures search throughput
- **Latency**: Lower is better - measures per-query response time
- **Recall@10**: Higher is better (max 1.0) - measures accuracy
- **Build Time**: Lower is better - time to construct index
- **Index Size**: Lower is better - storage efficiency

### Notes

- First run downloads SIFT1M dataset (~200MB)
- Benchmark includes warm-up phase before measurements
- All libraries use L2 distance metric
- Results may vary based on hardware and configuration
