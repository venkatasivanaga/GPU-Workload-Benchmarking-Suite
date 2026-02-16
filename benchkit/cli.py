from __future__ import annotations

import json
from pathlib import Path

import typer

from benchkit.config import load_config
from benchkit.runners.torch_infer import run_inference
from benchkit.utils.seed import seed_everything

app = typer.Typer(add_completion=False, help="GPU Workload Benchmarking Suite (CLI)")


@app.callback()
def main() -> None:
    """Benchkit CLI root."""
    return


@app.command()
def hello() -> None:
    """Sanity check command."""
    typer.echo("benchkit is installed and working OK")


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config"),
    out: str = typer.Option("results/last.json", "--out", "-o", help="Output JSON path"),
) -> None:
    """
    Run a benchmark from a YAML config.

    Currently supports:
      - task: infer
    """
    cfg = load_config(config)
    seed_everything(cfg.seed)

    task = cfg.task.lower().strip()
    if task != "infer":
        raise typer.BadParameter("Only task: infer is implemented right now")

    result = run_inference(cfg)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    typer.echo(f"Wrote: {out_path.as_posix()}")
    typer.echo(f"Throughput: {result['throughput_samples_per_s']:.2f} samples/s")
    typer.echo(f"Latency: {result['latency_ms_per_step']:.2f} ms/step")


if __name__ == "__main__":
    app()
