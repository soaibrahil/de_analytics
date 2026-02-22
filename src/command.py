import typer
from .bronze_layer.load import LoadBronze


app = typer.Typer()


@app.command(name="load_bronze")
def load_bronze_table(
    all: bool = typer.Option(False, "-a", "--all", help="Load all bronze tables"),
    database: str = typer.Option("bronze", "-d", "--database", help="Database to load into"),
    table: str = typer.Option(None, "-t", "--table", help="Name of a specific table to load"),
):
    
    loader = LoadBronze(table=table, database=database, all_tables=all)
    loader.execute(all=all, table=table)
