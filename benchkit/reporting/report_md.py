from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_report_md(df: pd.DataFrame, out_path: str | Path) -> Path:
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    n = len(df)
    best_tp = None
    if "throughput_samples_per_s" in df.columns:
        best_tp = df["throughput_samples_per_s"].max()

    lines = []
    lines.append("# Benchkit Report\n")
    lines.append(f"- Runs: **{n}**\n")
    if best_tp is not None:
        lines.append(f"- Best throughput: **{best_tp:.2f} samples/s**\n")

    # show a compact table
    show_cols = [
        c
        for c in [
            "config.name",
            "config.task",
            "config.model",
            "config.device",
            "config.batch_size",
            "throughput_samples_per_s",
            "latency_ms_per_step",
            "cuda_max_memory_allocated_bytes",
            "env.cuda_device_name",
        ]
        if c in df.columns
    ]
    if show_cols:
        preview = df[show_cols].copy()
        lines.append("\n## Runs (summary)\n")
        lines.append(preview.to_markdown(index=False))
        lines.append("\n")

    # embed plots if they exist
    lines.append("\n## Plots\n")
    lines.append("- Throughput vs Batch Size: `plots/throughput_vs_batch.png`\n")
    lines.append("- Latency vs Batch Size: `plots/latency_vs_batch.png`\n")

    outp.write_text("\n".join(lines), encoding="utf-8")
    return outp
