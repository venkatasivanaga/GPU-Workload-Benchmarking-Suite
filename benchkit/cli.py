from __future__ import annotations

import json
from pathlib import Path

import typer

from benchkit.config import load_config
from benchkit.runners.torch_infer import run_inference
from benchkit.utils.seed import seed_everything
from benchkit.metrics.system import system_metrics
from benchkit.utils.env import env_snapshot
from benchkit.utils.jsonl import append_json
from benchkit.reporting.export import jsonl_to_dataframe, write_summary_csv
from benchkit.reporting.plots import plot_latency_vs_batch, plot_throughput_vs_batch
from benchkit.reporting.report_md import write_report_md


app = typer.Typer(add_completion=False, help="GPU Workload Benchmarking Suite (CLI)")


@app.callback()
def main() -> None:
    """Benchkit CLI root."""
    return


@app.command()
def report(
    jsonl: str = typer.Option("results/runs.jsonl", help="Input JSONL path"),
    out_dir: str = typer.Option("results", help="Output directory"),
) -> None:
    """Generate summary CSV + report markdown + plots from runs.jsonl."""
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    df = jsonl_to_dataframe(jsonl)

    csv_path = write_summary_csv(jsonl, outp / "summary.csv")
    plot_dir = outp / "plots"
    plot1 = plot_throughput_vs_batch(df, plot_dir / "throughput_vs_batch.png")
    plot2 = plot_latency_vs_batch(df, plot_dir / "latency_vs_batch.png")
    md_path = write_report_md(df, outp / "report.md")

    typer.echo(f"Wrote: {csv_path}")
    typer.echo(f"Wrote: {md_path}")
    typer.echo(f"Wrote: {plot1}")
    typer.echo(f"Wrote: {plot2}")


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

    result["system"] = system_metrics()
    result["env"] = env_snapshot()

    append_jsonl("results/runs.jsonl", result)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    typer.echo(f"Wrote: {out_path.as_posix()}")
    typer.echo(f"Throughput: {result['throughput_samples_per_s']:.2f} samples/s")
    typer.echo(f"Latency: {result['latency_ms_per_step']:.2f} ms/step")
    typer.echo(f"CPU%: {result['system']['cpu_percent']:.1f} | RAM%: {result['system']['ram_percent']:.1f}")
    if "cuda_max_memory_allocated_bytes" in result:
        mb = result["cuda_max_memory_allocated_bytes"] / (1024**2)
        typer.echo(f"Peak GPU allocated: {mb:.1f} MB")


if __name__ == "__main__":
    app()
