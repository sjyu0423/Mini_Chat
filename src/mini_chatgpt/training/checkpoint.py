"""Checkpoint paths, metadata, and crash-safe file writing.

All writes go to a temporary file in the destination directory, are fsync'd,
and then atomically renamed with ``os.replace``. A crash mid-write therefore
leaves either the old file or the new one — never a truncated checkpoint.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

import torch

CHECKPOINT_DIRNAME = "checkpoints"
CHECKPOINT_SUFFIX = ".pt"


def _atomic_write(path: str | Path, writer: Callable[[IO[bytes]], None]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        # mkstemp creates 0600 files; use the normal umask-derived mode instead.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_name, 0o666 & ~umask)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    def writer(f: IO[bytes]) -> None:
        f.write(data)

    return _atomic_write(path, writer)


def atomic_write_text(path: str | Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj: Any) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def atomic_torch_save(path: str | Path, obj: Any) -> Path:
    return _atomic_write(path, lambda f: torch.save(obj, f))


def git_commit_hash(cwd: str | Path | None = None) -> str | None:
    """Current commit hash, or None outside a repo / before the first commit."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


@dataclass
class CheckpointMetadata:
    step: int
    config: dict[str, Any]
    val_loss: float | None = None
    train_loss: float | None = None
    tokens_seen: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    git_commit: str | None = field(default_factory=git_commit_hash)
    torch_version: str = str(torch.__version__)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CheckpointMetadata:
        return cls(**raw)


def checkpoint_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / CHECKPOINT_DIRNAME


def checkpoint_path(run_dir: str | Path, step: int) -> Path:
    if step < 0:
        raise ValueError("step must be non-negative")
    return checkpoint_dir(run_dir) / f"step_{step:08d}{CHECKPOINT_SUFFIX}"


def metadata_path(ckpt_path: str | Path) -> Path:
    """Sidecar JSON next to a checkpoint, readable without loading tensors."""
    return Path(ckpt_path).with_suffix(".json")


def list_checkpoints(run_dir: str | Path) -> list[Path]:
    d = checkpoint_dir(run_dir)
    return sorted(d.glob(f"step_*{CHECKPOINT_SUFFIX}")) if d.is_dir() else []


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    ckpts = list_checkpoints(run_dir)
    return ckpts[-1] if ckpts else None


def save_checkpoint(
    run_dir: str | Path, state: dict[str, Any], metadata: CheckpointMetadata
) -> Path:
    """Atomically write ``{"state": ..., "metadata": ...}`` plus a JSON sidecar.

    ``state`` will typically hold model/optimizer state_dicts and RNG states
    once training exists (M1+).
    """
    path = checkpoint_path(run_dir, metadata.step)
    atomic_torch_save(path, {"state": state, "metadata": metadata.to_dict()})
    atomic_write_json(metadata_path(path), metadata.to_dict())
    return path


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[dict[str, Any], CheckpointMetadata]:
    # weights_only=True: never unpickle arbitrary objects from checkpoint files.
    blob = torch.load(path, map_location=map_location, weights_only=True)
    return blob["state"], CheckpointMetadata.from_dict(blob["metadata"])
