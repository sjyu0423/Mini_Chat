"""Run the fixed general-language suite on a checkpoint (evaluation only, no training).

Usage:
    python scripts/evaluate_general.py runs/<run>/checkpoints/step_00004800.pt \
        --suite evals/general_v1.json --out docs/reports/m4/general_eval_<name>.json \
        --heldout fineweb_edu=data/raw/samples/fineweb_edu_10bt ...

``--heldout NAME=DIR`` measures bits per UTF-8 byte on up to ``--heldout-docs``
documents of ``DIR/sample.jsonl`` that fall in the 10% hash validation split
(``int(sha256(text)[:8], 16) % 100 < 10``, the split used for tokenizer research),
so they never took part in tokenizer fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from mini_chatgpt.analysis.general_eval import load_suite, run_suite
from mini_chatgpt.device import select_device
from mini_chatgpt.model.gpt import GPT
from mini_chatgpt.tokenizer import load_tokenizer
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import atomic_write_json
from mini_chatgpt.training.trainer import load_model


def is_val(text: str, val_percent: int = 10) -> bool:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) % 100 < val_percent


def val_docs(directory: Path, limit: int) -> list[str]:
    docs = []
    with (directory / "sample.jsonl").open(encoding="utf-8") as f:
        for line in f:
            text = json.loads(line).get("text")
            if text and is_val(text):
                docs.append(text)
    return docs[:limit]


@torch.no_grad()
def heldout_bpb(
    model: GPT, tok: ByteBPETokenizer, docs: list[str], device: torch.device
) -> dict[str, Any]:
    """Bits per byte: total next-token NLL (nats) / ln 2 / UTF-8 bytes. Each document
    starts after <|endoftext|> and is cut into non-overlapping context windows."""
    ctx = model.config.context_length
    nll, n_bytes, n_tokens = 0.0, 0, 0
    for doc in docs:
        ids = [tok.eot_id, *tok.encode(doc)] if tok.eot_id is not None else tok.encode(doc)
        for start in range(0, len(ids) - 1, ctx):
            chunk = ids[start : start + ctx + 1]
            x = torch.tensor([chunk[:-1]], device=device)
            y = torch.tensor(chunk[1:], device=device)
            logits, _ = model(x)
            nll += float(F.cross_entropy(logits[0].float(), y, reduction="sum"))
            n_tokens += len(chunk) - 1
        n_bytes += len(doc.encode("utf-8"))
    return {"docs": len(docs), "bytes": n_bytes, "tokens": n_tokens,
            "bits_per_byte": nll / math.log(2) / n_bytes if n_bytes else None,
            "nats_per_token": nll / n_tokens if n_tokens else None}  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--suite", type=Path, default=Path("evals/general_v1.json"))
    p.add_argument("--tokenizer", type=Path, default=None,
                   help="default: tokenizer.json in the checkpoint's run directory")  # fmt: skip
    p.add_argument("--heldout", action="append", default=[], metavar="NAME=DIR")
    p.add_argument("--heldout-docs", type=int, default=150)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = select_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    tok_path = args.tokenizer or args.checkpoint.parent.parent / "tokenizer.json"
    tok = load_tokenizer(tok_path)
    if not isinstance(tok, ByteBPETokenizer):
        raise SystemExit("the general suite needs a byte-level BPE tokenizer")
    suite, suite_sha = load_suite(args.suite)
    t0 = time.perf_counter()
    torch.manual_seed(0)
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "model_parameters": model.num_parameters(),
        "tokenizer_fingerprint": tok.fingerprint(),
        "suite": str(args.suite),
        "suite_sha256": suite_sha,
        "device": str(device),
        "heldout": {},
    }
    for spec in args.heldout:
        name, directory = spec.split("=", 1)
        docs = val_docs(Path(directory), args.heldout_docs)
        result["heldout"][name] = heldout_bpb(model, tok, docs, device)
    result.update(run_suite(model, tok, suite, device))
    result["seconds"] = time.perf_counter() - t0
    atomic_write_json(args.out, result)
    nt = result["next_token"]
    summary = {
        "heldout_bpb": {k: round(v["bits_per_byte"], 4) for k, v in result["heldout"].items()},
        "facts_acc_norm": nt["facts_mc"]["acc_norm"]["rate"],
        "qa_acc_norm": nt["qa_mc"]["acc_norm"]["rate"],
        "code_mc_acc_norm": nt["code_mc"]["acc_norm"]["rate"],
        "grammar_acc": nt["grammar_pairs"]["acc"]["rate"],
        "code_gen_acc": nt["code_generation"]["acc"]["rate"],
        "continuation_rep4": nt["continuation"]["mean_repeated_4gram_fraction"],
        "stress_rep4": nt["repetition_stress"]["mean_repeated_4gram_fraction"],
        "chat": result["instruction"]["chat"].get("applicable"),
        "seconds": round(result["seconds"], 1),
        "context_length": cfg.model.context_length,
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
