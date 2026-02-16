from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn


def build_mlp(in_shape: tuple[int, int, int], num_classes: int = 1000) -> nn.Module:
    c, h, w = in_shape
    in_features = c * h * w
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Linear(1024, num_classes),
    )


REGISTRY: Dict[str, Callable[[tuple[int, int, int]], nn.Module]] = {
    "mlp": build_mlp,
}


def get_model(name: str, in_shape: tuple[int, int, int]) -> nn.Module:
    key = name.lower().strip()
    if key not in REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(REGISTRY.keys())}")
    return REGISTRY[key](in_shape)
