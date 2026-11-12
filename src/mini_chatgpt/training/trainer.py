"""Training engine: accumulation, clipping, eval, logging, checkpoint/resume.

One *step* is one optimizer update over ``gradient_accumulation_steps``
micro-batches of ``batch_size`` sequences, i.e.
``tokens_per_step = batch_size * gradient_accumulation_steps * context_length``.

Checkpoints contain everything needed to continue bit-for-bit where the
backend is deterministic: model and optimizer state, step, token count, the
data-sampler generator state, and torch CPU/MPS/CUDA RNG states (dropout).
"""

from __future__ import annotations

import logging
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from mini_chatgpt.config import ExperimentConfig
from mini_chatgpt.data.dataset import get_batch, iter_sequential_batches
from mini_chatgpt.model.gpt import GPT
from mini_chatgpt.training.checkpoint import (
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
)
from mini_chatgpt.training.metrics import RunningMean
from mini_chatgpt.training.optim import build_optimizer, learning_rate
from mini_chatgpt.training.run import LOGGER_NAME, MetricsLogger
from mini_chatgpt.training.seed import make_generator, set_seed

logger = logging.getLogger(LOGGER_NAME)

EVAL_SEED_OFFSET = 1_000_003  # eval batches use a separate, fixed stream


class NonFiniteError(RuntimeError):
    """Raised when the loss or gradient norm becomes NaN/Inf."""


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def memory_stats(device: torch.device) -> dict[str, int]:
    if device.type == "mps":
        return {
            "mps_current_allocated": int(torch.mps.current_allocated_memory()),
            "mps_driver_allocated": int(torch.mps.driver_allocated_memory()),
        }
    if device.type == "cuda":
        return {"cuda_max_allocated": int(torch.cuda.max_memory_allocated(device))}
    return {}


def _rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "python": list(random.getstate()[1]),
    }
    if device.type == "mps":
        state["torch_mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def _set_rng_state(state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["torch_cpu"])
    py = random.getstate()
    random.setstate((py[0], tuple(state["python"]), py[2]))
    if device.type == "mps" and "torch_mps" in state:
        torch.mps.set_rng_state(state["torch_mps"])
    elif device.type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"], device)


@dataclass
class TrainStats:
    step_times: list[float] = field(default_factory=list)
    peak_memory: dict[str, int] = field(default_factory=dict)

    def record_memory(self, device: torch.device, phase: str) -> None:
        """Track the max of each memory counter, keyed by ``<counter>@<phase>``."""
        for k, v in memory_stats(device).items():
            key = f"{k}@{phase}"
            self.peak_memory[key] = max(self.peak_memory.get(key, 0), v)

    def timing_summary(self, skip: int) -> dict[str, float]:
        times = self.step_times[skip:]
        if not times:
            return {}
        return {
            "measured_steps": len(times),
            "mean_step_s": statistics.fmean(times),
            "median_step_s": statistics.median(times),
            "min_step_s": min(times),
            "max_step_s": max(times),
        }


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        device: torch.device,
        run_dir: str | Path | None = None,
    ) -> None:
        if config.training.dtype != "float32":
            raise ValueError(
                f"dtype {config.training.dtype!r} is not verified for training in M1; "
                "use float32 (bf16/mixed precision are future, explicitly-tested experiments)"
            )
        self.config = config
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.run_dir = Path(run_dir) if run_dir is not None else None

        tcfg = config.training
        set_seed(tcfg.seed)
        self.model = GPT(config.model).to(device)  # init happens on CPU RNG -> device-independent
        self.optimizer = build_optimizer(self.model, tcfg)
        self.data_generator = make_generator(tcfg.seed)
        self.step = 0
        self.tokens_seen = 0
        self.best_val_loss = math.inf
        self.stats = TrainStats()
        self.loss_history: list[float] = []  # per-step train loss (this process only)
        self.metrics: MetricsLogger | None = (
            MetricsLogger(self.run_dir / "metrics.jsonl") if self.run_dir else None
        )

    # ------------------------------------------------------------------ #
    @property
    def tokens_per_step(self) -> int:
        t = self.config.training
        return t.batch_size * t.gradient_accumulation_steps * self.config.model.context_length

    def _batch(self, data: torch.Tensor, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        return get_batch(
            data,
            self.config.training.batch_size,
            self.config.model.context_length,
            generator=gen,
            device=self.device,
        )

    def train_step(self) -> dict[str, float]:
        """One optimizer step with gradient accumulation and clipping."""
        tcfg = self.config.training
        self.model.train()
        lr = learning_rate(self.step, tcfg)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        synchronize(self.device)
        t0 = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=self.device)
        for micro in range(tcfg.gradient_accumulation_steps):
            x, y = self._batch(self.train_data, self.data_generator)
            _, loss = self.model(x, y)
            assert loss is not None
            if micro == tcfg.gradient_accumulation_steps - 1:
                # Activations are live here: the closest cheap probe to the true peak.
                self.stats.record_memory(self.device, phase="after_forward")
            # Average over micro-batches so the gradient matches one big batch.
            (loss / tcfg.gradient_accumulation_steps).backward()
            loss_sum += loss.detach()

        if tcfg.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tcfg.grad_clip)
        else:
            grad_norm = torch.linalg.vector_norm(
                torch.stack([p.grad.norm() for p in self.model.parameters() if p.grad is not None])
            )
        # A single host sync per step: check finiteness before touching weights.
        mean_loss = float(loss_sum.item()) / tcfg.gradient_accumulation_steps
        gnorm = float(grad_norm.item())
        if not math.isfinite(mean_loss) or not math.isfinite(gnorm):
            raise NonFiniteError(
                f"non-finite value at step {self.step}: loss={mean_loss}, grad_norm={gnorm}; "
                "optimizer step skipped, weights unchanged"
            )
        self.optimizer.step()
        synchronize(self.device)
        dt = time.perf_counter() - t0

        self.step += 1
        self.tokens_seen += self.tokens_per_step
        self.stats.step_times.append(dt)
        self.loss_history.append(mean_loss)
        self.stats.record_memory(self.device, phase="after_step")
        return {
            "loss": mean_loss,
            "lr": lr,
            "grad_norm": gnorm,
            "step_time_s": dt,
            "tokens_per_s": self.tokens_per_step / dt,
        }

    @torch.no_grad()
    def estimate_loss(self, num_batches: int | None = None) -> dict[str, float]:
        """Mean loss on ``num_batches`` random batches per split.

        Uses a freshly seeded generator every call, so successive evaluations
        see the same batches and are directly comparable.
        """
        n = num_batches or self.config.training.eval_steps
        was_training = self.model.training
        self.model.eval()
        out = {}
        for split, data in (("train", self.train_data), ("val", self.val_data)):
            gen = make_generator(self.config.training.seed + EVAL_SEED_OFFSET)
            mean = RunningMean()
            for _ in range(n):
                x, y = self._batch(data, gen)
                _, loss = self.model(x, y)
                assert loss is not None
                mean.update(float(loss.item()))
            out[split] = mean.mean
        self.model.train(was_training)
        return out

    @torch.no_grad()
    def full_loss(self, data: torch.Tensor, batch_size: int = 32) -> float:
        """Exact token-weighted loss over non-overlapping windows of a split."""
        was_training = self.model.training
        self.model.eval()
        mean = RunningMean()
        ctx = self.config.model.context_length
        for x, y in iter_sequential_batches(data, batch_size, ctx, device=self.device):
            _, loss = self.model(x, y)
            assert loss is not None
            mean.update(float(loss.item()), weight=y.numel())
        self.model.train(was_training)
        return mean.mean

    # ------------------------------------------------------------------ #
    def fit(self, max_steps: int | None = None) -> dict[str, float]:
        """Train until ``max_steps`` (defaults to config). Returns last eval."""
        tcfg = self.config.training
        end = max_steps if max_steps is not None else tcfg.max_steps
        last_eval: dict[str, float] = {}
        while self.step < end:
            m = self.train_step()
            if self.step % tcfg.log_interval == 0 or self.step == 1:
                logger.info(
                    "step %d | loss %.4f | lr %.2e | gnorm %.3f | %.0f tok/s",
                    self.step, m["loss"], m["lr"], m["grad_norm"], m["tokens_per_s"],
                )  # fmt: skip
                self._log("train", **m)
            if self.step % tcfg.eval_interval == 0 or self.step == end:
                last_eval = self.estimate_loss()
                logger.info(
                    "step %d | eval train %.4f | val %.4f",
                    self.step, last_eval["train"], last_eval["val"],
                )  # fmt: skip
                self._log("eval", train_loss=last_eval["train"], val_loss=last_eval["val"])
                self.best_val_loss = min(self.best_val_loss, last_eval["val"])
            if self.run_dir and (self.step % tcfg.checkpoint_interval == 0 or self.step == end):
                self.save(val_loss=last_eval.get("val"), train_loss=m["loss"])
        return last_eval

    def _log(self, event: str, **values: float) -> None:
        if self.metrics:
            self.metrics.log(event, self.step, tokens_seen=self.tokens_seen, **values)

    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "best_val_loss": self.best_val_loss,
            "data_generator": self.data_generator.get_state(),
            "rng": _rng_state(self.device),
        }

    def save(
        self,
        run_dir: str | Path | None = None,
        *,
        val_loss: float | None = None,
        train_loss: float | None = None,
    ) -> Path:
        target = Path(run_dir) if run_dir is not None else self.run_dir
        if target is None:
            raise ValueError("no run_dir to save into")
        meta = CheckpointMetadata(
            step=self.step,
            config=self.config.to_dict(),
            val_loss=val_loss,
            train_loss=train_loss,
            tokens_seen=self.tokens_seen,
        )
        path = save_checkpoint(target, self.state_dict(), meta)
        logger.info("saved checkpoint %s", path)
        return path

    def load_state(self, path: str | Path) -> CheckpointMetadata:
        """Restore a checkpoint into this trainer (config must match exactly)."""
        state, meta = load_checkpoint(path, map_location="cpu")
        saved = ExperimentConfig.from_dict(meta.config)
        if saved.model != self.config.model:
            raise ValueError("checkpoint model config differs from the current config")
        if saved.training != self.config.training:
            logger.warning("checkpoint training config differs; resume may not be exact")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])  # moves state to param device
        self.step = int(state["step"])
        self.tokens_seen = int(state["tokens_seen"])
        self.best_val_loss = float(state["best_val_loss"])
        self.data_generator.set_state(state["data_generator"])
        _set_rng_state(state["rng"], self.device)
        logger.info("resumed from %s at step %d", path, self.step)
        return meta


def load_model(path: str | Path, device: torch.device) -> tuple[GPT, ExperimentConfig]:
    """Load model weights only (for inference) from a trainer checkpoint."""
    state, meta = load_checkpoint(path, map_location="cpu")
    config = ExperimentConfig.from_dict(meta.config)
    model = GPT(config.model)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), config
