from __future__ import annotations

import platform
import sys
from typing import Any, Dict


def env_snapshot() -> Dict[str, Any]:
    snap: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    try:
        import torch

        snap["torch_version"] = torch.__version__
        snap["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            snap["cuda_device_name"] = torch.cuda.get_device_name(0)
            snap["cuda_device_count"] = int(torch.cuda.device_count())
    except Exception as e:
        snap["torch_error"] = str(e)

    return snap
