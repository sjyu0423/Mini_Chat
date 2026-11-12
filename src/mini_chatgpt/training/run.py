"""Run directories and structured training logs.

Layout of a run::

    runs/<name>-<timestamp>/
        config.json        # full ExperimentConfig
        environment.json   # device / library versions / git commit
        metrics.jsonl      # one JSON object per logged event
        checkpoints/       # step_00001000.pt + .json sidecars
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from mini_chatgpt.config import ExperimentConfig, format_config
from mini_chatgpt.device import DeviceInfo
from mini_chatgpt.training.checkpoint import atomic_write_json, git_commit_hash

LOGGER_NAME = "mini_chatgpt"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def make_run_dir(base: str | Path, name: str, *, timestamp: str | None = None) -> Path:
    ts = timestamp or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = Path(base) / f"{name}-{ts}"
    path.mkdir(parents=True, exist_ok=False)  # never reuse/overwrite a run
    return path


def _json_safe(value: Any) -> Any:
    # JSON has no NaN/Inf; store them as strings so lines stay valid JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class MetricsLogger:
    """Append-only JSON Lines logger, flushed after every record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.path.open("a", encoding="utf-8")

    def log(self, event: str, step: int, **values: Any) -> dict[str, Any]:
        record = {"event": event, "step": step, "time": time.time()}
        record.update({k: _json_safe(v) for k, v in values.items()})
        self._fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._fh.flush()
        return record

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_metrics(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_run_metadata(
    run_dir: str | Path, config: ExperimentConfig, device_info: DeviceInfo
) -> None:
    run_dir = Path(run_dir)
    atomic_write_json(run_dir / "config.json", config.to_dict())
    atomic_write_json(
        run_dir / "environment.json",
        {**asdict(device_info), "git_commit": git_commit_hash()},
    )
    logging.getLogger(LOGGER_NAME).info("configuration:\n%s", format_config(config))
