from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def plot_throughput_vs_batch(df: pd.DataFrame, out_path: str | Path) -> Path:
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if "config.batch_size" not in df.columns or "throughput_samples_per_s" not in df.columns:
        raise ValueError("Missing columns for throughput plot")

    x = _safe_num(df, "config.batch_size")
    y = _safe_num(df, "throughput_samples_per_s")

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("batch_size")
    plt.ylabel("throughput (samples/s)")
    plt.title("Throughput vs Batch Size")
    plt.savefig(outp, bbox_inches="tight", dpi=150)
    plt.close()
    return outp


def plot_latency_vs_batch(df: pd.DataFrame, out_path: str | Path) -> Path:
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if "config.batch_size" not in df.columns or "latency_ms_per_step" not in df.columns:
        raise ValueError("Missing columns for latency plot")

    x = _safe_num(df, "config.batch_size")
    y = _safe_num(df, "latency_ms_per_step")

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("batch_size")
    plt.ylabel("latency (ms/step)")
    plt.title("Latency vs Batch Size")
    plt.savefig(outp, bbox_inches="tight", dpi=150)
    plt.close()
    return outp
