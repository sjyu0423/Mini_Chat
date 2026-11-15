"""Supervised fine-tuning (SFT) on conversations with an assistant-only loss (M5).

Differences from pretraining (``training/trainer.py``):

* Data unit is a whole conversation, not a fixed 512-token window. Each batch holds
  ``batch_size`` conversations right-padded to the longest one; padded positions and
  every non-target position carry the label ``IGNORE`` (-100).
* Loss = sum of next-token cross-entropy over assistant target tokens (assistant text +
  its ``<|end|>``) / number of target tokens in the batch. User text conditions the
  model but contributes nothing to the loss.
* The model starts from the M4.1 pretrained weights; the optimizer (AdamW) is created
  fresh: pretraining moments are deliberately not restored.
* Epochs over a small dataset: every conversation is presented ``epochs`` times in a
  seeded order (``batch_plan``); the report states this repetition explicitly.
* Extra validation metric: end-of-turn accuracy — at positions whose target is
  ``<|end|>``, how often argmax predicts ``<|end|>`` (recall), and among target
  positions where argmax is ``<|end|>``, how often that is right (precision).
"""

from __future__ import annotations

import json
import logging
import math
import time
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from mini_chatgpt.chat.format import IGNORE
from mini_chatgpt.config import ExperimentConfig, TrainingConfig
from mini_chatgpt.data.prepare import file_sha256
from mini_chatgpt.model.gpt import GPT
from mini_chatgpt.training.checkpoint import (
    CheckpointMetadata,
    atomic_write_json,
    load_checkpoint,
    save_checkpoint,
)
from mini_chatgpt.training.optim import build_optimizer, learning_rate

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    name: str
    base_checkpoint: str
    data_dir: str = "data/processed/chat_m5"
    run_dir: str = "runs"
    description: str = ""
    seed: int = 1337
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 250
    log_interval: int = 25
    bucket_factor: int = 50  # length-bucketing pool, in batches
    # Pad every batch to a multiple of this many positions (capped at the context).
    # MPS keeps a separate cached buffer set per tensor shape; with free-length batches
    # the driver pool grew from 2 GB to 13.6 GB in 250 steps (attempt 1). Multiples of
    # 64 bound the run to 8 shapes.
    pad_multiple: int = 64
    # Even with 8 shapes the cached pool kept stepping up (7.8 -> 9.9 GB in 400 steps);
    # releasing the cache every 25 steps holds it at ~4.3 GB at no measurable speed cost.
    empty_cache_interval: int = 25
    max_driver_gb: float = 10.0  # abort instead of swapping if the MPS pool exceeds this
    device: str = "mps"
    expected_fingerprint: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: str | Path) -> SFTConfig:
        raw = tomllib.loads(Path(path).read_text())
        flat: dict[str, Any] = {}
        for value in raw.values():
            if isinstance(value, dict):
                flat.update(value)
        flat.update({k: v for k, v in raw.items() if not isinstance(v, dict)})
        cfg = cls(**flat)
        if cfg.batch_size <= 0 or cfg.epochs <= 0 or not 0 < cfg.learning_rate < 1e-2:
            raise ValueError("invalid batch_size, epochs or learning_rate")
        return cfg

    def training_config(self, max_steps: int) -> TrainingConfig:
        """Optimizer/schedule settings in the shape the shared helpers expect."""
        return TrainingConfig(
            seed=self.seed,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            min_learning_rate=self.min_learning_rate,
            warmup_steps=self.warmup_steps,
            max_steps=max_steps,
            weight_decay=self.weight_decay,
            beta1=self.beta1,
            beta2=self.beta2,
            grad_clip=self.grad_clip,
            device=self.device,  # type: ignore[arg-type]
            dtype="float32",
        )


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class ChatSplit:
    tokens: np.ndarray  # uint16
    mask: np.ndarray  # uint8
    offsets: np.ndarray  # int64, len = n + 1

    def __len__(self) -> int:
        return int(self.offsets.size - 1)

    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def conversation(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.tokens[a:b], self.mask[a:b]


def load_split(data_dir: str | Path, split: str, *, verify: bool = True) -> ChatSplit:
    d = Path(data_dir)
    if verify:
        manifest = json.loads((d / "manifest.json").read_text())["splits"][split]
        for kind, fname in (
            ("tokens", f"{split}_tokens.bin"),
            ("mask", f"{split}_mask.bin"),
            ("offsets", f"{split}_offsets.npy"),
        ):
            if file_sha256(d / fname) != manifest[f"{kind}_sha256"]:
                raise RuntimeError(f"{d / fname}: sha256 differs from manifest.json")
    return ChatSplit(
        tokens=np.fromfile(d / f"{split}_tokens.bin", dtype=np.uint16),
        mask=np.fromfile(d / f"{split}_mask.bin", dtype=np.uint8),
        offsets=np.load(d / f"{split}_offsets.npy"),
    )


def make_batch(
    split: ChatSplit,
    indices: list[int],
    device: torch.device | str = "cpu",
    *,
    pad_multiple: int = 1,
    max_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded ``x`` (B, T) and labels ``y`` (B, T); T = longest conversation - 1,
    rounded up to a multiple of ``pad_multiple`` (but at most ``max_len``).

    ``x[b, t] = ids[t]``, ``y[b, t] = ids[t + 1]`` if that token is a target else
    ``IGNORE``. Padding positions use ``<|endoftext|>`` as input (any id would do: the
    model is causal, so padding after a sequence never affects earlier positions) and
    ``IGNORE`` as label.
    """
    convs = [split.conversation(i) for i in indices]
    t = max(len(ids) for ids, _ in convs) - 1
    t = -(-t // pad_multiple) * pad_multiple
    if max_len is not None:
        t = min(t, max(max_len, max(len(ids) for ids, _ in convs) - 1))
    pad_id = int(convs[0][0][0])  # conversations start with <|endoftext|>
    x = torch.full((len(convs), t), pad_id, dtype=torch.long)
    y = torch.full((len(convs), t), IGNORE, dtype=torch.long)
    for b, (ids, mask) in enumerate(convs):
        n = len(ids) - 1
        ids_t = torch.from_numpy(ids.astype(np.int64))
        x[b, :n] = ids_t[:-1]
        target = torch.from_numpy(mask[1:].astype(bool))
        y[b, :n] = torch.where(target, ids_t[1:], torch.full_like(ids_t[1:], IGNORE))
    return x.to(device), y.to(device)


def batch_plan(
    lengths: np.ndarray, batch_size: int, seed: int, epoch: int, bucket_factor: int = 50
) -> list[list[int]]:
    """Seeded order for one epoch with light length bucketing: shuffle, cut into pools
    of ``bucket_factor`` batches, sort each pool by length, cut into batches, shuffle the
    batches. Every conversation appears exactly once per epoch."""
    g = torch.Generator().manual_seed(seed * 1000 + epoch)
    order = torch.randperm(len(lengths), generator=g).tolist()
    pool = batch_size * bucket_factor
    batches: list[list[int]] = []
    for start in range(0, len(order), pool):
        chunk = sorted(order[start : start + pool], key=lambda i: (int(lengths[i]), i))
        batches += [chunk[k : k + batch_size] for k in range(0, len(chunk), batch_size)]
    perm = torch.randperm(len(batches), generator=g).tolist()
    return [batches[i] for i in perm]


# --------------------------------------------------------------------------- #
# loss and evaluation
# --------------------------------------------------------------------------- #
def masked_loss(logits: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Sum of cross-entropy over target positions and the number of targets."""
    loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=IGNORE, reduction="sum"
    )
    return loss_sum, int((y != IGNORE).sum().item())


@torch.no_grad()
def evaluate(
    model: GPT,
    split: ChatSplit,
    end_id: int,
    device: torch.device,
    batch_size: int = 16,
    pad_multiple: int = 64,
) -> dict[str, float]:
    """Exact assistant-token loss over the whole split, plus end-of-turn statistics."""
    was = model.training
    model.eval()
    order = np.argsort(split.lengths(), kind="stable").tolist()
    total, n_tok = 0.0, 0
    end_targets = end_hits = end_pred = end_pred_ok = 0
    for k in range(0, len(order), batch_size):
        x, y = make_batch(
            split,
            order[k : k + batch_size],
            device,
            pad_multiple=pad_multiple,
            max_len=model.config.context_length,
        )
        logits, _ = model(x)
        s, n = masked_loss(logits, y)
        total += float(s.item())
        n_tok += n
        pred = logits.argmax(-1)
        tgt = y != IGNORE
        is_end = y == end_id
        end_targets += int(is_end.sum().item())
        end_hits += int((is_end & (pred == end_id)).sum().item())
        p_end = tgt & (pred == end_id)
        end_pred += int(p_end.sum().item())
        end_pred_ok += int((p_end & is_end).sum().item())
    model.train(was)
    loss = total / max(n_tok, 1)
    return {
        "loss": loss,
        "ppl": math.exp(loss),
        "target_tokens": float(n_tok),
        "end_recall": end_hits / max(end_targets, 1),
        "end_precision": end_pred_ok / max(end_pred, 1),
        "end_targets": float(end_targets),
    }


def mps_memory() -> dict[str, int]:
    if not torch.backends.mps.is_available():
        return {}
    return {
        "mps_current_allocated": int(torch.mps.current_allocated_memory()),
        "mps_driver_allocated": int(torch.mps.driver_allocated_memory()),
    }


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #
class SFTTrainer:
    def __init__(
        self,
        cfg: SFTConfig,
        *,
        run_dir: Path | None,
        device: torch.device,
        train: ChatSplit,
        val: ChatSplit,
        end_id: int,
    ) -> None:
        self.cfg, self.run_dir, self.device = cfg, run_dir, device
        self.train, self.val, self.end_id = train, val, end_id
        state, meta = load_checkpoint(cfg.base_checkpoint, map_location="cpu")
        self.base_meta = meta
        self.exp_config = ExperimentConfig.from_dict(meta.config)
        torch.manual_seed(cfg.seed)
        self.model = GPT(self.exp_config.model)
        self.model.load_state_dict(state["model"], strict=True)
        self.model.to(device).train()
        self.plans = [
            batch_plan(train.lengths(), cfg.batch_size, cfg.seed, e, cfg.bucket_factor)
            for e in range(cfg.epochs)
        ]
        self.steps_per_epoch = len(self.plans[0])
        self.max_steps = self.steps_per_epoch * cfg.epochs
        self.tcfg = cfg.training_config(self.max_steps)
        self.optimizer = build_optimizer(self.model, self.tcfg)  # fresh moments
        self.step = 0
        self.target_tokens_seen = 0
        self.best_val = float("inf")
        self.best_step = -1
        self.history: list[dict[str, Any]] = []

    # -- state ------------------------------------------------------------- #
    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "sft_step": self.step,
            "target_tokens_seen": self.target_tokens_seen,
            "best_val": self.best_val,
            "best_step": self.best_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = int(state["sft_step"])
        self.target_tokens_seen = int(state["target_tokens_seen"])
        self.best_val = float(state["best_val"])
        self.best_step = int(state["best_step"])

    def save(self, kind: str, val: dict[str, float] | None) -> Path:
        assert self.run_dir is not None
        config = dict(self.base_meta.config)
        config["name"] = self.cfg.name
        config["description"] = self.cfg.description
        meta = CheckpointMetadata(
            step=self.step,
            config=config,
            val_loss=val["loss"] if val else None,
            val_loss_exact=val["loss"] if val else None,
            tokens_seen=self.target_tokens_seen,
            kind=kind if kind in ("best", "final") else "periodic",
        )
        name = "best.pt" if kind == "best" else f"{kind}.pt"
        path = save_checkpoint(
            self.run_dir, self.state_dict(), meta, path=self.run_dir / "checkpoints" / name
        )
        atomic_write_json(
            path.with_suffix(".json"),
            {
                **meta.to_dict(),
                "sft": asdict(self.cfg),
                "base_checkpoint": self.cfg.base_checkpoint,
                "val": val,
            },
        )
        return path

    # -- training ---------------------------------------------------------- #
    def batch_indices(self, step: int) -> list[int]:
        return self.plans[step // self.steps_per_epoch][step % self.steps_per_epoch]

    def train_step(self) -> dict[str, float]:
        lr = learning_rate(self.step, self.tcfg)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        idx = self.batch_indices(self.step)
        x, y = make_batch(
            self.train,
            idx,
            self.device,
            pad_multiple=self.cfg.pad_multiple,
            max_len=self.model.config.context_length,
        )
        real = int(sum(self.train.lengths()[i] - 1 for i in idx))
        logits, _ = self.model(x)
        loss_sum, n = masked_loss(logits, y)
        loss = loss_sum / max(n, 1)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.step += 1
        self.target_tokens_seen += n
        interval = self.cfg.empty_cache_interval
        if self.device.type == "mps" and interval and self.step % interval == 0:
            torch.mps.empty_cache()
        return {
            "loss": float(loss.item()),
            "lr": lr,
            "grad_norm": float(gnorm.item()),
            "targets": float(n),
            "tokens": float(x.numel()),
            "real_tokens": float(real),
        }

    def log(self, row: dict[str, Any]) -> None:
        self.history.append(row)
        if self.run_dir is not None:
            with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    def run_eval(self) -> dict[str, float]:
        t0 = time.time()
        val = evaluate(
            self.model, self.val, self.end_id, self.device, pad_multiple=self.cfg.pad_multiple
        )
        if self.device.type == "mps":
            torch.mps.empty_cache()
        row = {
            "event": "eval",
            "step": self.step,
            "epoch": self.step / self.steps_per_epoch,
            "target_tokens_seen": self.target_tokens_seen,
            **val,
            "seconds": time.time() - t0,
            **mps_memory(),
        }
        if val["loss"] < self.best_val:
            self.best_val, self.best_step = val["loss"], self.step
            row["is_best"] = True
            if self.run_dir is not None:
                self.save("best", val)
        self.log(row)
        logger.info(
            "eval step %d: val %.4f (ppl %.2f) end recall %.3f precision %.3f%s",
            self.step,
            val["loss"],
            val["ppl"],
            val["end_recall"],
            val["end_precision"],
            " *best*" if row.get("is_best") else "",
        )
        return val

    def fit(self) -> dict[str, Any]:
        t_start = time.time()
        self.run_eval()  # step 0 = the untouched base model
        while self.step < self.max_steps:
            t0 = time.time()
            m = self.train_step()
            if self.device.type == "mps":
                torch.mps.synchronize()
            dt = time.time() - t0
            if self.step % self.cfg.log_interval == 0 or self.step == 1:
                self.log(
                    {
                        "event": "train",
                        "step": self.step,
                        **m,
                        "step_s": dt,
                        "tokens_per_s": m["tokens"] / dt,
                        "real_tokens_per_s": m["real_tokens"] / dt,
                        "epoch": self.step / self.steps_per_epoch,
                        **mps_memory(),
                    }
                )
                logger.info(
                    "step %d/%d loss %.4f lr %.2e gnorm %.2f %.0f tok/s",
                    self.step,
                    self.max_steps,
                    m["loss"],
                    m["lr"],
                    m["grad_norm"],
                    m["tokens"] / dt,
                )
            if not math.isfinite(m["loss"]):
                raise RuntimeError(f"non-finite loss at step {self.step}")
            pool = mps_memory().get("mps_driver_allocated", 0) / 1e9
            if pool > self.cfg.max_driver_gb:
                raise RuntimeError(
                    f"MPS driver pool {pool:.1f} GB > {self.cfg.max_driver_gb} GB "
                    f"at step {self.step}; stopping instead of swapping"
                )
            if self.step % self.cfg.eval_interval == 0 or self.step == self.max_steps:
                val = self.run_eval()
        final = self.save("final", val)
        return {
            "steps": self.step,
            "steps_per_epoch": self.steps_per_epoch,
            "epochs": self.cfg.epochs,
            "best_val": self.best_val,
            "best_step": self.best_step,
            "final_val": val,
            "final_checkpoint": str(final),
            "target_tokens_seen": self.target_tokens_seen,
            "wall_s": time.time() - t_start,
        }
