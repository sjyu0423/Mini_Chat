"""Device selection for Apple Silicon (MPS), CUDA, and CPU.

CUDA is never assumed. ``"auto"`` picks the best available backend in the order
CUDA -> MPS -> CPU, so on a Mac it resolves to MPS when available and otherwise
falls back to CPU. An explicitly requested device that is unavailable raises
:class:`DeviceUnavailableError` instead of silently falling back, so experiments
never run on a different backend than the config says.

Availability probes are injectable to make the selection logic unit-testable on
any machine.
"""

from __future__ import annotations

import platform
from collections.abc import Callable
from dataclasses import dataclass

import torch

from mini_chatgpt.config import DeviceName, DTypeName


class DeviceUnavailableError(RuntimeError):
    """Raised when an explicitly requested device is not available."""


def mps_available() -> bool:
    return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def select_device(
    preference: DeviceName | str = "auto",
    *,
    is_mps_available: Callable[[], bool] = mps_available,
    is_cuda_available: Callable[[], bool] = cuda_available,
) -> torch.device:
    """Resolve a device preference to a concrete ``torch.device``."""
    pref = preference.lower()
    if pref == "auto":
        if is_cuda_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "mps":
        if not is_mps_available():
            raise DeviceUnavailableError(
                "MPS requested but not available (requires Apple Silicon, macOS 12.3+, "
                "and a PyTorch build with MPS). Use device='cpu' or 'auto'."
            )
        return torch.device("mps")
    if pref == "cuda":
        if not is_cuda_available():
            raise DeviceUnavailableError("CUDA requested but not available. Use 'auto' or 'cpu'.")
        return torch.device("cuda")
    raise ValueError(f"unknown device preference {preference!r}")


_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def resolve_dtype(name: DTypeName | str, device: torch.device) -> torch.dtype:
    """Map a dtype name to ``torch.dtype``, rejecting poor device/dtype combos."""
    if name not in _DTYPES:
        raise ValueError(f"unknown dtype {name!r}; choose from {sorted(_DTYPES)}")
    if device.type == "cpu" and name == "float16":
        raise ValueError("float16 is not supported for CPU training; use float32 or bfloat16")
    return _DTYPES[name]


@dataclass(frozen=True)
class DeviceInfo:
    selected: str
    torch_version: str
    mps_built: bool
    mps_available: bool
    cuda_available: bool
    machine: str
    python_version: str


def describe_device(device: torch.device) -> DeviceInfo:
    return DeviceInfo(
        selected=str(device),
        torch_version=torch.__version__,
        mps_built=bool(torch.backends.mps.is_built()),
        mps_available=mps_available(),
        cuda_available=cuda_available(),
        machine=platform.machine(),
        python_version=platform.python_version(),
    )
