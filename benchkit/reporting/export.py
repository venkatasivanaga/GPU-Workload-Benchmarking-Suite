from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    # Keep useful top-level fields + flatten common nested blocks
    flat: Dict[str, Any] = {}

    # top-level
    keep = [
        "device_used",
        "total_time_s",
        "throughput_samples_per_s",
        "latency_ms_per_step",
        "cuda_max_memory_allocated_bytes",
        "cuda_max_memory_reserved_bytes",
    ]
    for k in keep:
        if k in rec:
            flat[k] = rec[k]

    # config.*
    cfg = rec.get("config", {}) or {}
    for k, v in cfg.items():
        flat[f"config.{k}"] = v

    # env.*
    env = rec.get("env", {}) or {}
    for k, v in env.items():
        flat[f"env.{k}"] = v

    # system.*
    sysm = rec.get("system", {}) or {}
    for k, v in sysm.items():
        flat[f"system.{k}"] = v

    return flat


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")
    rows: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def jsonl_to_dataframe(path: str | Path) -> pd.DataFrame:
    rows = read_jsonl(path)
    flat_rows = [_flatten_record(r) for r in rows]
    df = pd.DataFrame(flat_rows)
    # nicer ordering if present
    preferred = [
        "config.name",
        "config.task",
        "config.model",
        "config.device",
        "config.batch_size",
        "config.steps",
        "config.warmup",
        "throughput_samples_per_s",
        "latency_ms_per_step",
        "cuda_max_memory_allocated_bytes",
        "system.cpu_percent",
        "system.ram_percent",
        "env.torch_version",
        "env.cuda_device_name",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def write_summary_csv(jsonl_path: str | Path, out_csv: str | Path) -> Path:
    df = jsonl_to_dataframe(jsonl_path)
    outp = Path(out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    return outp
