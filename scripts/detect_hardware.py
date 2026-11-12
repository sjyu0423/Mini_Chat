"""Report confirmed hardware and PyTorch backend information (macOS-focused).

Usage: python scripts/detect_hardware.py [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from mini_chatgpt.device import describe_device, select_device


def sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def gpu_cores() -> str | None:
    try:
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.stdout.splitlines():
        if "Total Number of Cores" in line:
            return line.split(":", 1)[1].strip()
    return None


def collect() -> dict[str, object]:
    info: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    if sys.platform == "darwin":
        mem = sysctl("hw.memsize")
        info.update(
            chip=sysctl("machdep.cpu.brand_string"),
            memory_gib=round(int(mem) / 2**30, 2) if mem else None,
            cpu_cores_total=sysctl("hw.physicalcpu"),
            macos=platform.mac_ver()[0],
            gpu_cores=gpu_cores(),
        )
        levels = sysctl("hw.nperflevels")
        for i in range(int(levels) if levels else 0):
            name = sysctl(f"hw.perflevel{i}.name") or f"level{i}"
            info[f"cpu_cores_{name.lower()}"] = sysctl(f"hw.perflevel{i}.physicalcpu")
    dev = select_device("auto")
    info["device_info"] = vars(describe_device(dev))
    if dev.type == "mps":
        # A tiny real op proves the backend works, not just that it reports available.
        x = torch.ones(4, device=dev)
        info["mps_smoke_test"] = float((x * 2).sum().item()) == 8.0
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="also write results to this JSON file")
    args = parser.parse_args()
    info = collect()
    text = json.dumps(info, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
