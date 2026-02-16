from __future__ import annotations

import os
from typing import Dict

import psutil


def system_metrics() -> Dict[str, float]:
    proc = psutil.Process(os.getpid())
    cpu_pct = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    rss_bytes = proc.memory_info().rss

    return {
        "cpu_percent": float(cpu_pct),
        "ram_percent": float(vm.percent),
        "process_rss_bytes": float(rss_bytes),
    }
