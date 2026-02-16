(
echo from __future__ import annotations
echo.
echo import typer
echo.
echo app = typer.Typer(add_completion=False, help="GPU Workload Benchmarking Suite (CLI)")
echo.
echo @app.command()
echo def hello() -^> None:
echo     """Sanity check command."""
echo     typer.echo("benchkit is installed and working ✅")
echo.
echo if __name__ == "__main__":
echo     app()
) > benchkit\cli.py
