"""Orchestration helpers for bronze ingestion."""
from databricks.connect import DatabricksSession
import typer
from .utils import create_bronze_table
from pyspark.sql.utils import AnalysisException
from .config import TABLES, TABLE_FILES

class LoadBronze:
    """Class to handle loading data into the bronze layer."""
    def __init__(self, table: str, database: str, file_location = TABLE_FILES,  all_tables=False, mode: str = "append"):
        self.table = table
        self.database = database
        self.file_location = file_location
        self.all_tables = all_tables
        self.mode = mode


    def read(self, table: str, spark: DatabricksSession):
        """Ingest files from landing to bronze."""
        # Here you would implement the logic to copy files from landing to bronze
        try: 
            file_path = self.file_location[table]
            raw_df = spark.read.parquet(file_path)
            typer.echo(f"Successfully read {table} from {file_path}")
            return raw_df
        except Exception as e:
            raise Exception(f"Error reading {table}: {e}")


    def load(self, table: str, raw_df, spark: DatabricksSession):
        """Load a specific table into the bronze layer."""
        # Here you would implement the logic to load a specific table
        try:
            raw_df.write.saveAsTable(table, mode=self.mode)
            typer.echo(f"Successfully loaded {table}")
        except AnalysisException as e:
            if "Table or view not found" in str(e):
                typer.echo(f"Table {table} not found. Attempting to create it...")
                create_bronze_table(table=table)
                raw_df.write.saveAsTable(table, mode=self.mode)
                typer.echo(f"Successfully loaded {table}")
            else:
                raise AnalysisException(f"Error loading {table}: {e}")
        except Exception as e:
            raise Exception(f"Error loading {table}: {e}")


    def execute(self, all: bool = False, table: str = None):
        """Execute the loading process based on the provided options."""
        spark = DatabricksSession.builder.getOrCreate()
        if self.all_tables:
            # Load all tables
            for t in TABLES:
                raw_df = self.read(t, spark)
                self.load(t, raw_df, spark)
        elif self.table:
            raw_df = self.read(self.table, spark)
            self.load(self.table, raw_df, spark)
        else:
            raise ValueError("Either --all or --table must be specified.")