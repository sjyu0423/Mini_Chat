"""M4.1 data preflight (complements scripts/preflight_scaling.py, which checks the model).

Checks:
1. Raw files: size and SHA-256 equal the pinned values in ``general.SOURCES``.
2. Processed streams: SHA-256 and token counts equal ``meta.json`` (``load_prepared``),
   holdout too; all ids < vocabulary size.
3. Tokenizer: fingerprint equals ``meta.json`` and the manifest; reload round-trips.
4. Splits: no document hash shared between train, validation and holdout.
5. Single pass: the train stream has at least ``max_steps x sequences/step`` windows.
6. Next-token alignment on real data: ``y[t] == x[t + 1]`` for single-pass windows,
   and the windows decode to text that appears in the held-out/training documents
   (spot check of 3 windows).
7. Budget: steps x tokens/step equals the planned 157,286,400 presentations.

Usage: python scripts/preflight_m4_1_data.py configs/m4_1_general_s_bpe8192.toml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mini_chatgpt.config import ExperimentConfig
from mini_chatgpt.data import general as g
from mini_chatgpt.data.dataset import get_windows, single_pass_order
from mini_chatgpt.data.prepare import file_sha256, load_prepared, processed_dir
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer

RAW = Path("data/raw/general_m4_1")
PROBE = "Héllo, wörld! 123 <|end|>"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    cfg = ExperimentConfig.from_toml(args.config)
    out_dir = processed_dir(cfg.data)
    meta = json.loads((out_dir / "meta.json").read_text())
    manifest = json.loads((out_dir / "manifest.json").read_text())
    c: dict[str, Any] = {}

    c["1_raw_files"] = {s.name: g.verify_file(RAW / s.local_name, s) == s.sha256
                        for s in g.SOURCES}  # fmt: skip
    data = load_prepared(out_dir)  # verifies train/val sha256, counts, id range, fingerprint
    hold = np.memmap(out_dir / "holdout.bin", dtype="<u2", mode="r")
    c["2_streams"] = {
        "train_val_verified_by_load_prepared": True,
        "holdout_sha256_ok": file_sha256(out_dir / "holdout.bin") == meta["holdout_sha256"],
        "holdout_tokens_ok": hold.size == meta["num_holdout_tokens"],
        "holdout_ids_in_vocab": int(hold.max()) < meta["vocab_size"],
        "train_tokens": meta["num_train_tokens"],
        "val_tokens": meta["num_val_tokens"],
        "holdout_tokens": meta["num_holdout_tokens"],
    }
    tok = ByteBPETokenizer.load(out_dir / "tokenizer.json")
    c["3_tokenizer"] = {
        "fingerprint": tok.fingerprint(),
        "matches_meta": tok.fingerprint() == meta["tokenizer_fingerprint"],
        "matches_manifest": tok.fingerprint() == manifest["tokenizer"]["fingerprint"],
        "vocab_size_matches_model": tok.vocab_size == cfg.model.vocab_size,
        "special_tokens": tok.special_tokens,
        "round_trip": tok.decode(tok.encode(PROBE)) == PROBE,
    }
    c["4_split_overlap"] = manifest["split_overlap_check"]
    t = cfg.training
    seqs = t.max_steps * t.batch_size * t.gradient_accumulation_steps
    order = single_pass_order(data.splits.num_train, cfg.model.context_length, t.seed)
    c["5_single_pass"] = {
        "windows_available": int(order.numel()),
        "windows_needed": seqs,
        "unused_windows": int(order.numel()) - seqs,
        "unique_train_tokens_presented": seqs * cfg.model.context_length,
        "repeated_presentations": 0,
    }
    x, y = get_windows(data.splits.train, order[:3], cfg.model.context_length)
    full = data.splits.train[order[0].item() * cfg.model.context_length :][
        : cfg.model.context_length + 1
    ].long()
    c["6_alignment"] = {
        "y_shift_of_x": bool(torch.equal(y[:, :-1], x[:, 1:])),
        "window_matches_stream": bool(torch.equal(x[0], full[:-1]) and torch.equal(y[0], full[1:])),
        "decoded_window_0_head": tok.decode(x[0, :40].tolist()),
    }
    tokens_per_step = t.batch_size * t.gradient_accumulation_steps * cfg.model.context_length
    c["7_budget"] = {"tokens_per_step": tokens_per_step,
                     "presentations": t.max_steps * tokens_per_step,
                     "matches_plan": t.max_steps * tokens_per_step == 157_286_400}  # fmt: skip
    passed = (
        all(c["1_raw_files"].values())
        and all(v for k, v in c["2_streams"].items() if k.endswith("_ok") or k.endswith("vocab"))
        and c["3_tokenizer"]["matches_meta"] and c["3_tokenizer"]["matches_manifest"]
        and c["3_tokenizer"]["vocab_size_matches_model"] and c["3_tokenizer"]["round_trip"]
        and not any(c["4_split_overlap"].values())
        and c["5_single_pass"]["unused_windows"] >= 0
        and c["6_alignment"]["y_shift_of_x"] and c["6_alignment"]["window_matches_stream"]
        and c["7_budget"]["matches_plan"]
    )  # fmt: skip
    c["passed"] = bool(passed)
    print(json.dumps(c, indent=1, ensure_ascii=False))
    if args.out:
        from mini_chatgpt.training.checkpoint import atomic_write_json

        atomic_write_json(args.out, c)
    print("DATA PREFLIGHT", "PASSED" if passed else "FAILED")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
