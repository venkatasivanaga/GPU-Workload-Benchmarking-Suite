from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, help="GPU Workload Benchmarking Suite (CLI)")

@app.callback()
def main() -> None:
    """Benchkit CLI."""
    return

@app.command()
def hello() -> None:
    """Sanity check command."""
    typer.echo("benchkit is installed and working OK")

@app.command()
def run(config: str = typer.Option(..., "--config", "-c", help="Path to YAML config")) -> None:
    """Load config and print it (runner next)."""
    from benchkit.config import load_config
    cfg = load_config(config)
    typer.echo(f"Loaded config: {cfg}")

if __name__ == "__main__":
    app()
