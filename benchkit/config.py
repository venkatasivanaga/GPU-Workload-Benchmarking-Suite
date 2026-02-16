from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import yaml

PathLike = Union[str, Path]


@dataclass
class RunConfig:
    name: str
    task: str  # infer | train (train later)
    model: str

    device: str = "cuda"  # cpu | cuda
    batch_size: int = 32
    steps: int = 50
    warmup: int = 10

    precision: str = "fp32"  # fp32 | fp16 | bf16 (used later)
    input_shape: Tuple[int, int, int] = (3, 224, 224)

    seed: int = 42


def load_config(path: PathLike) -> RunConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    data: Dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    required = ["name", "task", "model"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing required keys in config: {missing}")

    inp = data.get("input_shape", [3, 224, 224])
    if isinstance(inp, list):
        inp = tuple(int(x) for x in inp)

    return RunConfig(
        name=str(data["name"]),
        task=str(data["task"]),
        model=str(data["model"]),
        device=str(data.get("device", "cuda")),
        batch_size=int(data.get("batch_size", 32)),
        steps=int(data.get("steps", 50)),
        warmup=int(data.get("warmup", 10)),
        precision=str(data.get("precision", "fp32")),
        input_shape=inp,
        seed=int(data.get("seed", 42)),
    )
