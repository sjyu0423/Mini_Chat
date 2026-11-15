"""M5 SFT preflight: correctness checks on the real base model and data before training.

1. Base checkpoint: SHA-256, architecture, parameter count, tokenizer fingerprint.
2. Data integrity: hashes (``load_split``), id range, leakage check, and an exact
   re-serialization of 300 random conversations from their stored text (tokens + mask).
3. Masked loss: equals a manual computation over assistant targets only; user tokens are
   never targets; label alignment ``y[t] = ids[t+1]``.
4. Chat-token gradients: after one backward pass the embedding rows of ``<|user|>``,
   ``<|assistant|>`` and ``<|end|>`` receive non-zero gradient.
5. Overfit: 4 real conversations, 80 steps at lr 1e-3 -> assistant loss near zero and
   every ``<|end|>`` predicted (then discarded).
6. Stability, speed, memory: 30 real steps with the run's settings, all losses and
   gradient norms finite; projected wall-clock for the full run. 6b: 400 more steps,
   the MPS driver pool must plateau (attempt 1 grew to 13.6 GB without padding buckets).
7. Checkpoint restoration: save, reload into a fresh trainer, the next step matches;
   ``load_model`` (inference path) loads the SFT checkpoint.

Usage:
    python scripts/preflight_sft.py configs/sft/m5_sft_chat.toml \
        --out docs/reports/m5/preflight.json
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from mini_chatgpt.chat.format import IGNORE, ChatTokens, encode_conversation
from mini_chatgpt.chat.sft import (
    SFTConfig,
    SFTTrainer,
    evaluate,
    load_split,
    make_batch,
    masked_loss,
    mps_memory,
)
from mini_chatgpt.data.prepare import file_sha256
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import atomic_write_json
from mini_chatgpt.training.trainer import load_model

BASE_SHA = "fb21b80a03828206f3cd337b3579c1335ee3e47327a49b66e6a560bc66148082"  # M4.1 (M5)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-sha", default=BASE_SHA, help="expected SHA-256 of the base checkpoint")
    p.add_argument("--expected-params", type=int, default=13_989_888)
    args = p.parse_args()
    cfg = SFTConfig.from_toml(args.config)
    dev = torch.device(cfg.device)
    c: dict[str, Any] = {}
    tok = ByteBPETokenizer.load(Path(cfg.data_dir) / "tokenizer.json")
    ct = ChatTokens.from_tokenizer(tok)
    train, val = load_split(cfg.data_dir, "train"), load_split(cfg.data_dir, "val")
    manifest = json.loads((Path(cfg.data_dir) / "manifest.json").read_text())

    # 1. base checkpoint
    tr = SFTTrainer(cfg, run_dir=None, device=dev, train=train, val=val, end_id=ct.end)
    n_params = sum(q.numel() for q in tr.model.parameters())
    c["1_base"] = {
        "sha256_ok": file_sha256(cfg.base_checkpoint) == args.base_sha,
        "step": tr.base_meta.step,
        "model": tr.exp_config.model.__dict__,
        "parameters": n_params,
        "parameters_ok": n_params == args.expected_params,
        "context_ok": tr.exp_config.model.context_length == 512,
        "fingerprint_ok": tok.fingerprint() == cfg.expected_fingerprint,
        "chat_token_ids": {
            "eot": ct.eot,
            "user": ct.user,
            "assistant": ct.assistant,
            "system": ct.system,
            "end": ct.end,
        },
        "steps": tr.max_steps,
        "steps_per_epoch": tr.steps_per_epoch,
    }

    # 2. data integrity
    rows = [json.loads(line) for line in (Path(cfg.data_dir) / "train_conversations.jsonl").open()]
    rng = random.Random(0)
    sample = rng.sample(range(len(rows)), 300)
    mismatches = 0
    for i in sample:
        ids, mask = encode_conversation(tok, rows[i]["messages"])
        sids, smask = train.conversation(i)
        mismatches += ids != sids.tolist() or mask != [bool(m) for m in smask]
    starts_ok = all(int(train.tokens[o]) == ct.eot for o in train.offsets[:-1])
    c["2_data"] = {
        "hashes_verified": True,
        "train_conversations": len(train),
        "val_conversations": len(val),
        "ids_in_vocab": int(train.tokens.max()) < tok.vocab_size,
        "all_start_with_endoftext": starts_ok,
        "max_len_ok": int(train.lengths().max()) <= 513 and int(val.lengths().max()) <= 513,
        "reserialization_mismatches_of_300": int(mismatches),
        "leakage": manifest["leakage_check"],
    }

    # 3. masked loss on a real batch
    idx = list(range(8))
    x, y = make_batch(train, idx, dev, pad_multiple=cfg.pad_multiple, max_len=512)
    model = tr.model
    logits, _ = model(x)
    s, n = masked_loss(logits, y)
    keep = y != IGNORE
    manual = -F.log_softmax(logits.float(), -1).gather(-1, y.clamp(min=0)[..., None])[..., 0]
    user_targets = 0
    for b, i in enumerate(idx):
        cids, cmask = train.conversation(i)
        in_user = False
        for t, tid in enumerate(cids.tolist()):
            if tid in (ct.user, ct.system):
                in_user = True
            elif tid == ct.assistant:
                in_user = False
            if in_user and cmask[t]:
                user_targets += 1
        assert x[b, : len(cids) - 1].tolist() == cids[:-1].tolist()
    c["3_masked_loss"] = {
        "masked_mean_loss": float(s.item() / n),
        "manual_equal": bool(torch.allclose(s, manual[keep].sum(), rtol=1e-4)),
        "target_tokens": n,
        "input_tokens": int(x.numel()),
        "user_or_system_tokens_marked_as_targets": user_targets,
    }

    # 4. chat-token gradients
    model.zero_grad(set_to_none=True)
    (s / n).backward()  # type: ignore[no-untyped-call]
    g = model.wte.weight.grad
    assert g is not None
    rows_norm = g.norm(dim=1)
    c["4_chat_token_gradients"] = {
        name: float(rows_norm[i].item())
        for name, i in (
            ("user", ct.user),
            ("assistant", ct.assistant),
            ("system", ct.system),
            ("end", ct.end),
            ("endoftext", ct.eot),
        )
    } | {
        "median_row": float(rows_norm.median().item()),
        "finite": bool(torch.isfinite(g).all().item()),
    }
    model.zero_grad(set_to_none=True)

    # 5. overfit 4 real conversations (separate trainer, discarded)
    ocfg = SFTConfig(
        **{
            **cfg.__dict__,
            "learning_rate": 1e-3,
            "min_learning_rate": 1e-3,
            "warmup_steps": 1,
            "epochs": 1,
        }
    )
    ov = SFTTrainer(ocfg, run_dir=None, device=dev, train=train, val=val, end_id=ct.end)
    tiny = list(range(4))
    ov.batch_indices = lambda step: tiny  # type: ignore[method-assign]
    from mini_chatgpt.chat.sft import ChatSplit

    sub = ChatSplit(train.tokens, train.mask, train.offsets[: len(tiny) + 1])
    start = evaluate(ov.model, sub, ct.end, dev)["loss"]
    for _ in range(80):
        ov.train_step()
    end = evaluate(ov.model, sub, ct.end, dev)
    c["5_overfit"] = {
        "start_loss": start,
        "end_loss": end["loss"],
        "end_recall": end["end_recall"],
        "passed": end["loss"] < 0.1 and end["end_recall"] == 1.0,
    }
    del ov

    # 6. stability / speed / memory with the real settings
    base_val = evaluate(tr.model, val, ct.end, dev)
    times, losses, gn, toks = [], [], [], []
    for _ in range(30):
        t0 = time.time()
        m = tr.train_step()
        torch.mps.synchronize() if dev.type == "mps" else None
        times.append(time.time() - t0)
        losses.append(m["loss"])
        gn.append(m["grad_norm"])
        toks.append(m["tokens"])
    sec = sorted(times[5:])[len(times[5:]) // 2]
    eval_t0 = time.time()
    evaluate(tr.model, val, ct.end, dev)
    eval_s = time.time() - eval_t0
    evals = tr.max_steps // cfg.eval_interval + 2
    c["6_stability_speed"] = {
        "base_val_loss": base_val["loss"],
        "base_val_end_recall": base_val["end_recall"],
        "losses_finite": all(map(lambda v: v == v and abs(v) < 1e9, losses)),
        "grad_norms_finite": all(map(lambda v: v == v and abs(v) < 1e9, gn)),
        "first_losses": losses[:5],
        "last_losses": losses[-5:],
        "max_grad_norm": max(gn),
        "median_step_s": sec,
        "mean_padded_tokens_per_step": sum(toks) / len(toks),
        "eval_s": eval_s,
        "projected_minutes": (sec * tr.max_steps + eval_s * evals) / 60,
        **mps_memory(),
    }

    # 6b. memory soak: 400 more steps; the MPS pool must plateau below the guard
    pools = []
    for k in range(400):
        tr.train_step()
        if k % 50 == 49:
            torch.mps.synchronize()
            pools.append(mps_memory().get("mps_driver_allocated", 0) / 1e9)
    c["6b_memory_soak"] = {
        "driver_pool_gb_every_50_steps": [round(v, 2) for v in pools],
        "passed": max(pools) < cfg.max_driver_gb and pools[-1] - pools[3] < 0.5,
    }

    # 7. checkpoint restoration
    with tempfile.TemporaryDirectory(dir="runs") as tmp:
        tr.run_dir = Path(tmp)
        (tr.run_dir / "checkpoints").mkdir()
        path = tr.save("final", None)
        nxt = tr.train_step()
        blob = torch.load(path, map_location="cpu", weights_only=True)
        tr2 = SFTTrainer(cfg, run_dir=None, device=dev, train=train, val=val, end_id=ct.end)
        tr2.load_state_dict(blob["state"])
        nxt2 = tr2.train_step()
        diff = max(
            (a - b).abs().max().item()
            for a, b in zip(
                tr.model.state_dict().values(), tr2.model.state_dict().values(), strict=True
            )
        )
        loaded, _ = load_model(path, torch.device("cpu"))
        c["7_checkpoint"] = {
            "resumed_step": int(blob["state"]["sft_step"]),
            "next_loss": nxt["loss"],
            "next_loss_after_restore": nxt2["loss"],
            "max_param_diff_after_next_step": diff,
            "restore_ok": abs(nxt["loss"] - nxt2["loss"]) < 1e-4 and diff < 1e-5,
            "load_model_ok": sum(q.numel() for q in loaded.parameters()) == args.expected_params,
            "size_mb": path.stat().st_size / 1e6,
        }

    c["passed"] = bool(
        all(c["1_base"][k] for k in ("sha256_ok", "parameters_ok", "context_ok", "fingerprint_ok"))
        and c["2_data"]["ids_in_vocab"]
        and c["2_data"]["all_start_with_endoftext"]
        and c["2_data"]["max_len_ok"]
        and c["2_data"]["reserialization_mismatches_of_300"] == 0
        and not any(c["2_data"]["leakage"].values())
        and c["3_masked_loss"]["manual_equal"]
        and c["3_masked_loss"]["user_or_system_tokens_marked_as_targets"] == 0
        and all(c["4_chat_token_gradients"][k] > 0 for k in ("user", "assistant", "end"))
        and c["4_chat_token_gradients"]["finite"]
        and c["5_overfit"]["passed"]
        and c["6_stability_speed"]["losses_finite"]
        and c["6_stability_speed"]["grad_norms_finite"]
        and c["6b_memory_soak"]["passed"]
        and c["7_checkpoint"]["restore_ok"]
        and c["7_checkpoint"]["load_model_ok"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, c)
    print(json.dumps(c, indent=1, default=str))
    print("SFT PREFLIGHT", "PASSED" if c["passed"] else "FAILED")
    if not c["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
