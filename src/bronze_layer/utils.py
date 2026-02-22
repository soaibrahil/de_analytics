"""Utility helpers for bronze-layer operations."""
from pathlib import Path
from typing import List
import typer 
from databricks.connect import DatabricksSession

app = typer.Typer()

def create_bronze_table(all: bool = False, table: str = None):  
    """Create tables in the bronze layer based on provided options."""
    spark = DatabricksSession.builder.getOrCreate()
    if all:
        typer.echo("Creating all bronze tables...")
        for ddl in Path(__file__).parent.joinpath("schemas").glob("*.sql"):
            sql = ddl.read_text()
            # Here you would execute the SQL against your database
            typer.echo(f"Creating table from schema file: {ddl.name}")
            spark.sql(sql)
    elif table:
        ddl = Path(__file__).parent.parent.joinpath("schemas", f"{table}.sql")
        if ddl.exists():
            sql = ddl.read_text()
            typer.echo(f"Creating table from schema file: {table}")
            spark.sql(sql)
        else:
            typer.echo(f"Table schema {table}.sql not found.")