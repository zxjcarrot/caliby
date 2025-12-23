# Caliby Initialization and Directory Management

## Overview

Caliby now provides robust initialization with data directory management, file locking, and automatic recovery. This ensures that:

1. Only one process can access a data directory at a time (prevents data corruption)
2. Indexes persist across process restarts
3. Proper recovery happens automatically on startup

## Basic Usage

```python
import caliby

# 1. Configure buffer pool (must be done before initialization)
caliby.set_buffer_config(size_gb=2.0)

# 2. Initialize caliby with a data directory
caliby.open("/path/to/data/directory")

# 3. Use the catalog API to create persistent indexes
catalog = caliby.IndexCatalog.instance()

config = caliby.IndexConfig()
config.dimensions = 128
config.max_elements = 10000

hnsw_config = caliby.HNSWConfig()
hnsw_config.M = 16
hnsw_config.ef_construction = 200
config.hnsw = hnsw_config

# Create a persistent index
handle = catalog.create_index("my_index", caliby.IndexType.HNSW, config)

# Flush to disk


# Shutdown cleanly
caliby.close()
```

## Recovery After Restart

When you initialize caliby with an existing data directory, it automatically recovers all previously created indexes:

```python
import caliby

# Configure and initialize
caliby.set_buffer_config(size_gb=2.0)
caliby.open("/path/to/data/directory")

catalog = caliby.IndexCatalog.instance()

# List recovered indexes
indexes = catalog.list_indexes()
for info in indexes:
    print(f"Found index: {info.name}, dimensions: {info.dimensions}")

# Open an existing index
handle = catalog.open_index("my_index")
print(f"Recovered index: {handle.name()}")
```

## File Locking

The data directory is locked using POSIX fcntl file locks. This prevents multiple processes from accessing the same data directory concurrently:

```python
# Process 1
caliby.set_buffer_config(size_gb=2.0)
caliby.open("/data/dir")  # Acquires lock

# Process 2 (will fail)
caliby.set_buffer_config(size_gb=2.0)
caliby.open("/data/dir")  # Raises: "Data directory is already locked"
```

The lock is automatically released when:
- `caliby.close()` is called
- The process exits

## Directory Structure

After initialization, the data directory contains:

```
/path/to/data/directory/
├── .caliby.lock          # Lock file (fcntl lock)
├── caliby_catalog        # Catalog metadata
├── heapfile             # Buffer pool backing file
└── index_*.dat          # Individual index files
```

## Configuration Guidelines

### Buffer Pool Size

Set `physgb` based on your available RAM:

```python
# For systems with 16GB RAM, use ~2-4GB for indexes
caliby.set_buffer_config(size_gb=2.0)

# For systems with 64GB RAM, you can allocate more
caliby.set_buffer_config(size_gb=16.0)
```

Note: `virtgb` is automatically computed based on index sizes and should not be manually configured.

### Initialization Order

**IMPORTANT**: Always call functions in this order:

1. `caliby.set_buffer_config()` - **MUST be called first**
2. `caliby.open()` - **MUST be called before creating indexes**
3. Create indexes or open existing ones
4. Use indexes
5. `` - Flush changes to disk
6. `caliby.close()` - Clean shutdown

### Error Handling

```python
import caliby

try:
    caliby.set_buffer_config(size_gb=2.0)
    caliby.open("/data/dir")
except RuntimeError as e:
    if "already locked" in str(e):
        print("Another process is using this directory")
    else:
        print(f"Initialization failed: {e}")
```

## Direct API vs Catalog API

Caliby provides two ways to create indexes:

### 1. Direct API (Non-persistent)

For temporary indexes that don't need to survive process restarts:

```python
import caliby

# No explicit initialization needed (lazy init)
index = caliby.HnswIndex(dims=128, max_elements=10000)

# Use the index
index.addPoint(vector, node_id=-1)

# Cleanup happens automatically via destructor
del index
```

### 2. Catalog API (Persistent)

For persistent indexes that survive restarts:

```python
import caliby

# Must initialize with data directory
caliby.set_buffer_config(size_gb=2.0)
caliby.open("/data/dir")

catalog = caliby.IndexCatalog.instance()

# Create persistent index
config = caliby.IndexConfig()
config.dimensions = 128
config.max_elements = 10000
# ... configure hnsw or diskann settings ...

handle = catalog.create_index("persistent_index", caliby.IndexType.HNSW, config)

# Cleanup via catalog API
catalog.drop_index("persistent_index")
caliby.close()
```

## Best Practices

1. **Always use a dedicated data directory** - Don't mix caliby data with other application data
2. **Call shutdown() before exit** - Ensures clean release of file locks
3. **Use absolute paths** - Relative paths can cause confusion across processes
4. **Handle lock errors gracefully** - Check for concurrent access
5. **Flush regularly** - Call `` after bulk operations
6. **Monitor disk space** - Indexes can grow large, especially with large `max_elements`

## Example: Production Application

```python
import caliby
import signal
import sys
import os

DATA_DIR = "/var/lib/myapp/caliby"
PHYSGB = 4.0

def cleanup(signum, frame):
    """Handle shutdown signals gracefully."""
    print("Shutting down...")
    try:
        catalog = caliby.IndexCatalog.instance()
        
        caliby.close()
    except:
        pass
    sys.exit(0)

def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Initialize caliby
    try:
        caliby.set_buffer_config(size_gb=PHYSGB)
        caliby.open(DATA_DIR)
        print(f"Initialized caliby with data directory: {DATA_DIR}")
    except RuntimeError as e:
        if "already locked" in str(e):
            print("ERROR: Another instance is already running")
            sys.exit(1)
        else:
            raise
    
    catalog = caliby.IndexCatalog.instance()
    
    # List existing indexes
    indexes = catalog.list_indexes()
    print(f"Found {len(indexes)} existing indexes")
    
    # Create or open index
    if "embeddings" not in [idx.name for idx in indexes]:
        print("Creating new embeddings index...")
        config = caliby.IndexConfig()
        config.dimensions = 384
        config.max_elements = 1000000
        
        hnsw_config = caliby.HNSWConfig()
        hnsw_config.M = 32
        hnsw_config.ef_construction = 400
        config.hnsw = hnsw_config
        
        handle = catalog.create_index("embeddings", caliby.IndexType.HNSW, config)
    else:
        print("Opening existing embeddings index...")
        handle = catalog.open_index("embeddings")
    
    print(f"Index ready: {handle.name()}, {handle.dimensions()} dimensions")
    
    # Your application logic here
    # ...
    
    # Clean shutdown
    print("Flushing changes...")
    
    caliby.close()
    print("Shutdown complete")

if __name__ == "__main__":
    main()
```

## Testing

Comprehensive tests are available in `tests/test_initialization.py`:

```bash
# Run initialization tests
pytest tests/test_initialization.py -v

# Run all tests
pytest tests/ -v
```

Test coverage includes:
- Basic initialization
- Directory locking (concurrent access prevention)
- Recovery after process restart
- Multiple index recovery
- Lock release on shutdown

## Troubleshooting

### "Data directory is already locked"

**Cause**: Another process is using the same data directory.

**Solutions**:
1. Check for running processes: `ps aux | grep python | grep caliby`
2. Check lock file: `lsof /data/dir/.caliby.lock`
3. If process crashed without releasing lock, manually remove `.caliby.lock` (only if sure no process is running)

### "Cannot change buffer config after system is initialized"

**Cause**: Trying to call `set_buffer_config()` after an index was already created.

**Solution**: Call `set_buffer_config()` before any index creation or caliby operations.

### Slow recovery on startup

**Cause**: Large number of indexes or large indexes.

**Solution**: This is normal. Recovery time scales with the number and size of indexes. Consider:
- Reducing the number of indexes
- Using smaller `max_elements` values
- Dropping unused indexes with `catalog.drop_index()`

## Migration from Old Code

If you're upgrading from code that used environment variables:

**Old way:**
```python
import os
os.environ["PHYSGB"] = "2.0"
os.environ["VIRTGB"] = "24.0"

import caliby
index = caliby.HnswIndex(dims=128, max_elements=10000)
```

**New way:**
```python
import caliby

# Option 1: Direct API (non-persistent)
caliby.set_buffer_config(size_gb=2.0)  # virtgb auto-computed
index = caliby.HnswIndex(dims=128, max_elements=10000)

# Option 2: Catalog API (persistent)
caliby.set_buffer_config(size_gb=2.0)
caliby.open("/data/dir")
catalog = caliby.IndexCatalog.instance()
# ... use catalog API ...
```

## Performance Considerations

- **Flushing**: Call `` periodically, not after every operation
- **Recovery time**: O(number of indexes), happens at initialization
- **Lock overhead**: Minimal (POSIX fcntl locks are very fast)
- **Disk I/O**: Catalog operations may trigger fsync for durability

## Future Enhancements

Potential future improvements:
- Read-only mode (multiple readers)
- Catalog compaction
- Index versioning
- Backup/restore utilities
- Monitoring/metrics APIs
