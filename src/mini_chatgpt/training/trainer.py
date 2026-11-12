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
import resource
import statistics
import sys
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
    apply_retention,
    best_checkpoint_path,
    load_checkpoint,
    read_metadata,
    save_checkpoint,
)
from mini_chatgpt.training.metrics import RunningMean, bits_per_token, perplexity
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


def process_peak_rss() -> int:
    """True high-water mark of this process's resident set size, in bytes.

    ``ru_maxrss`` is bytes on macOS and KiB on Linux. On Apple Silicon, MPS
    buffers live in unified memory but are not necessarily counted in RSS, so
    this complements (does not replace) the MPS counters.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def memory_stats(device: torch.device) -> dict[str, int]:
    """Memory counters. Keys ending in ``_peak`` are true high-water marks.

    * CUDA: ``torch.cuda.max_memory_allocated`` is an exact allocator peak.
    * MPS: PyTorch exposes no peak counter. ``mps_current_allocated`` is a
      point sample; ``mps_driver_allocated`` is the allocator's pool, which only
      grows unless ``torch.mps.empty_cache()`` is called, so its max over
      samples is (close to) the peak reserved memory.
    * all devices: ``process_rss_peak`` from ``getrusage``.
    """
    out = {"process_rss_peak": process_peak_rss()}
    if device.type == "mps":
        out["mps_current_allocated"] = int(torch.mps.current_allocated_memory())
        out["mps_driver_allocated"] = int(torch.mps.driver_allocated_memory())
    elif device.type == "cuda":
        out["cuda_allocated_peak"] = int(torch.cuda.max_memory_allocated(device))
        out["cuda_reserved_peak"] = int(torch.cuda.max_memory_reserved(device))
    return out


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
        self.best_val_loss = math.inf  # on the selection metric (see best_metric)
        self.best_step: int | None = None
        self.stats = TrainStats()
        # Persisted in checkpoints, so after a resume these cover the whole run.
        self.loss_history: list[float] = []  # per-step mean train loss
        self.eval_history: list[dict[str, float]] = []  # one record per evaluation
        self._stop_requested = False
        self.metrics: MetricsLogger | None = (
            MetricsLogger(self.run_dir / "metrics.jsonl") if self.run_dir else None
        )

    # ------------------------------------------------------------------ #
    @property
    def best_metric(self) -> str:
        """Eval key used for best-checkpoint selection."""
        return "val_exact" if self.config.training.exact_val_eval else "val"

    def request_stop(self) -> None:
        """Ask ``fit`` to checkpoint and return after the current step (signal-safe)."""
        self._stop_requested = True

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
    def evaluate(self) -> dict[str, float]:
        """Evaluate, log, update the best checkpoint. Returns the eval record."""
        tcfg = self.config.training
        est = self.estimate_loss()
        rec: dict[str, float] = {"train": est["train"], "val": est["val"]}
        if tcfg.exact_val_eval:
            rec["val_exact"] = self.full_loss(self.val_data)
        self.stats.record_memory(self.device, phase="eval")

        metric = rec[self.best_metric]
        is_best = metric < self.best_val_loss
        if is_best:
            self.best_val_loss = metric
            self.best_step = self.step
        record = {
            "step": float(self.step),
            "tokens_seen": float(self.tokens_seen),
            "lr": learning_rate(self.step, tcfg),
            **rec,
        }
        self.eval_history.append(record)

        derived: dict[str, float] = {
            "train_loss": rec["train"],
            "val_loss": rec["val"],
            "train_ppl": perplexity(rec["train"]),
            "val_ppl": perplexity(rec["val"]),
            "train_bpc": bits_per_token(rec["train"]),
            "val_bpc": bits_per_token(rec["val"]),
            "gap": rec["val"] - rec["train"],
            "best_val_loss": self.best_val_loss,
            "is_best": float(is_best),
            "lr": record["lr"],
        }
        if "val_exact" in rec:
            derived["val_loss_exact"] = rec["val_exact"]
            derived["val_ppl_exact"] = perplexity(rec["val_exact"])
            derived["val_bpc_exact"] = bits_per_token(rec["val_exact"])
        self._log("eval", **derived)
        logger.info(
            "step %d | eval train %.4f | val %.4f%s | ppl %.2f | %.3f bpc | gap %+.4f%s",
            self.step, rec["train"], rec["val"],
            f" | val(exact) {rec['val_exact']:.4f}" if "val_exact" in rec else "",
            perplexity(metric), bits_per_token(metric), derived["gap"],
            " | new best" if is_best else "",
        )  # fmt: skip
        if is_best and tcfg.save_best and self.run_dir is not None and self.step > 0:
            self.save(
                val_loss=rec["val"],
                train_loss=rec["train"],
                val_loss_exact=rec.get("val_exact"),
                kind="best",
                path=best_checkpoint_path(self.run_dir),
            )
        return rec

    def fit(self, max_steps: int | None = None) -> dict[str, float]:
        """Train until ``max_steps`` (defaults to config). Returns the last eval.

        A fresh run (step 0) is evaluated before the first update, so curves
        start at the initialization. Periodic checkpoints follow
        ``checkpoint_interval`` and ``keep_last_checkpoints``; the last step is
        always saved (``kind="final"`` when it is ``max_steps``).
        ``request_stop()`` makes it save and return early (``kind="interrupt"``).
        """
        tcfg = self.config.training
        end = max_steps if max_steps is not None else tcfg.max_steps
        last_eval: dict[str, float] = {}
        if self.step == 0 and not self.eval_history:
            last_eval = self.evaluate()
        t_start = time.perf_counter()
        start_step = self.step
        while self.step < end:
            m = self.train_step()
            self._log("train", **m)  # every step: cheap, and makes curves exact
            if self.step % tcfg.log_interval == 0 or self.step == 1:
                elapsed = time.perf_counter() - t_start
                done = self.step - start_step
                eta = elapsed / done * (end - self.step) if done else math.nan
                mem = memory_stats(self.device)
                self._log("memory", **{k: float(v) for k, v in mem.items()})
                logger.info(
                    "step %d/%d (%.1f%%) | loss %.4f | lr %.2e | gnorm %.3f | %.0f ms | "
                    "%.0f tok/s | eta %.1f min",
                    self.step, end, 100 * self.step / end, m["loss"], m["lr"], m["grad_norm"],
                    1000 * m["step_time_s"], m["tokens_per_s"], eta / 60,
                )  # fmt: skip
            if self.step % tcfg.eval_interval == 0 or self.step == end:
                last_eval = self.evaluate()
            if self.run_dir is None:
                if self._stop_requested:
                    break
                continue
            if self.step == end:
                self.save(
                    val_loss=last_eval.get("val"),
                    val_loss_exact=last_eval.get("val_exact"),
                    train_loss=m["loss"],
                    # Only the scheduled end of training is "final"; a partial
                    # fit(max_steps=...) ends with an ordinary checkpoint.
                    kind="final" if end == tcfg.max_steps else "periodic",
                )
            elif self._stop_requested or self.step % tcfg.checkpoint_interval == 0:
                self.save(
                    val_loss=last_eval.get("val") if self.step % tcfg.eval_interval == 0 else None,
                    train_loss=m["loss"],
                    kind="interrupt" if self._stop_requested else "periodic",
                )
            deleted = apply_retention(self.run_dir, tcfg.keep_last_checkpoints)
            for path in deleted:
                logger.info("retention: deleted %s", path.name)
            if self._stop_requested:
                logger.warning("stop requested: checkpointed at step %d and exiting", self.step)
                break
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
            "best_step": -1 if self.best_step is None else self.best_step,
            "loss_history": list(self.loss_history),
            "eval_history": [dict(r) for r in self.eval_history],
            "data_generator": self.data_generator.get_state(),
            "rng": _rng_state(self.device),
        }

    def save(
        self,
        run_dir: str | Path | None = None,
        *,
        val_loss: float | None = None,
        train_loss: float | None = None,
        val_loss_exact: float | None = None,
        kind: str = "periodic",
        path: str | Path | None = None,
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
            kind=kind,
            val_loss_exact=val_loss_exact,
        )
        path = save_checkpoint(target, self.state_dict(), meta, path=path)
        logger.info("saved %s checkpoint %s", kind, path)
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
        # Fields added in M2; M1 checkpoints lack them.
        best_step = int(state.get("best_step", -1))
        self.best_step = None if best_step < 0 else best_step
        self.loss_history = [float(x) for x in state.get("loss_history", [])]
        self.eval_history = [
            {k: float(v) for k, v in r.items()} for r in state.get("eval_history", [])
        ]
        self.data_generator.set_state(state["data_generator"])
        _set_rng_state(state["rng"], self.device)
        if self.run_dir is not None:
            best = read_metadata(best_checkpoint_path(self.run_dir))
            if best is not None and best.step > self.step:
                logger.warning(
                    "best.pt is from step %d, after the resume point %d; it will be replaced "
                    "only if the resumed run beats best_val_loss %.4f",
                    best.step, self.step, self.best_val_loss,
                )  # fmt: skip
        logger.info("resumed from %s at step %d", path, self.step)
        return meta


def load_model(path: str | Path, device: torch.device) -> tuple[GPT, ExperimentConfig]:
    """Load model weights only (for inference) from a trainer checkpoint."""
    state, meta = load_checkpoint(path, map_location="cpu")
    config = ExperimentConfig.from_dict(meta.config)
    model = GPT(config.model)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), config
