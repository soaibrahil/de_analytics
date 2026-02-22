Bronze layer
============

This package contains simple helpers to move raw files from a landing
zone into a bronze storage area.

Quick usage
-----------

From Python:

```python
from bronze_layer import ingest_directory, DEFAULT_LANDING_DIR, DEFAULT_BRONZE_DIR

ingest_directory(DEFAULT_LANDING_DIR, DEFAULT_BRONZE_DIR)
```

Or via CLI:

```bash
python -m src.bronze_layer.ingest data/landing data/bronze
```

Notes
-----
- These helpers are intentionally small and dependency-free.
- They copy files and partition by ingestion date by default.
