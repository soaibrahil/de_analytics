"""Configuration defaults for the bronze layer."""
from pathlib import Path
import os


DEFAULT_BRONZE_DIR = Path(os.getenv("BRONZE_DIR", "Volumes/workspace/default/landing_zone/data/bronze"))

TABLES = [
    "patient",
    "encounter",
    "facility",
    "payer",
    "provider",
    "transaction"
]

TABLE_FILES = {
    "patient": f"{DEFAULT_BRONZE_DIR}/bronze_dim_patient.parquet",
    "encounter": f"{DEFAULT_BRONZE_DIR}/bronze_fact_encounter.parquet",
    "facility": f"{DEFAULT_BRONZE_DIR}/bronze_dim_facility.parquet",
    "payer": f"{DEFAULT_BRONZE_DIR}/bronze_dim_payer.parquet",
    "provider": f"{DEFAULT_BRONZE_DIR}/bronze_dim_provider.parquet",
    "transaction": f"{DEFAULT_BRONZE_DIR}/bronze_fact_transaction.parquet"
}