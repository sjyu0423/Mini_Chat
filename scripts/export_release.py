"""Export the v1.0 inference-only model release and verify it reproduces the originals.

For each model (``m5`` = default chatbot, ``m6_3`` = research comparison):

* ``runs/release/mini_chatgpt-v1.0/<name>/checkpoints/model.pt``: model weights + config
  metadata only (no optimizer state), in the same ``{"state": {"model": ...}, "metadata": ...}``
  layout, so ``load_model`` and ``scripts/serve_chat.py --checkpoint`` accept it unchanged;
* ``tokenizer.json`` (fingerprint-checked) and ``sft_config.json`` (expected fingerprint), so
  ``ChatEngine.from_checkpoint`` verifies the tokenizer exactly as for the training run.

Verification (CPU, deterministic): every tensor equal to the source checkpoint; logits on fixed
inputs identical (max |diff| = 0); greedy and seeded-sampling replies to fixed prompts identical
text and token ids. Results and SHA-256 of every file go to ``docs/release/v1.0_manifest.json``.
The release directory is git-ignored (weights are not committed).

Usage: python scripts/export_release.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from mini_chatgpt.chat.generate import chat_reply
from mini_chatgpt.data.prepare import file_sha256
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.checkpoint import atomic_torch_save, load_checkpoint
from mini_chatgpt.training.trainer import load_model

OUT = Path("runs/release/mini_chatgpt-v1.0")
MANIFEST = Path("docs/release/v1.0_manifest.json")
FINGERPRINT = "e6be77ccff4929187ed84ee1b71d15b7654824abd9e23f3ad889c57232ce91c8"
MODELS = {
    "m5": {
        "role": "default chatbot",
        "source": "runs/m5_sft_chat-20261114-175334/checkpoints/best.pt",
        "report": "docs/reports/m5_conversational_training.md",
    },
    "m6_3": {
        "role": "research comparison (not served by default)",
        "source": "runs/m6_3_sft_30m-20261115-001347/checkpoints/best.pt",
        "report": "docs/reports/m6_3_conversational_scaling.md",
    },
}
PROMPTS = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "List three fruits."}],
    [
        {"role": "user", "content": "My name is Priya."},
        {"role": "assistant", "content": "Nice to meet you, Priya!"},
        {"role": "user", "content": "What is my name?"},
    ],
]
DEFAULTS = {"temperature": 0.7, "top_k": 40, "repetition_penalty": 1.2, "max_new_tokens": 200}


def export(name: str, spec: dict[str, str]) -> dict[str, Any]:
    src = Path(spec["source"])
    run_dir = src.parent.parent
    dst = OUT / name
    (dst / "checkpoints").mkdir(parents=True, exist_ok=True)
    state, meta = load_checkpoint(src)
    atomic_torch_save(
        dst / "checkpoints" / "model.pt",
        {"state": {"model": state["model"]}, "metadata": meta.to_dict()},
    )
    tok = ByteBPETokenizer.load(run_dir / "tokenizer.json")
    if tok.fingerprint() != FINGERPRINT:
        raise SystemExit(f"{name}: tokenizer fingerprint mismatch")
    shutil.copyfile(run_dir / "tokenizer.json", dst / "tokenizer.json")
    (dst / "sft_config.json").write_text(json.dumps({"expected_fingerprint": FINGERPRINT}) + "\n")

    cpu = torch.device("cpu")
    a, cfg = load_model(src, cpu)
    b, _ = load_model(dst / "checkpoints" / "model.pt", cpu)
    sa, sb = a.state_dict(), b.state_dict()
    tensors_equal = sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    torch.manual_seed(0)
    x = torch.randint(0, cfg.model.vocab_size, (2, 64))
    with torch.no_grad():
        logit_diff = float((a(x)[0] - b(x)[0]).abs().max())
    replies = []
    for turns in PROMPTS:
        for greedy, seed in ((True, None), (False, 11)):
            kw = {k: v for k, v in DEFAULTS.items()}
            ra = chat_reply(a, tok, turns, greedy=greedy, seed=seed, **kw)  # type: ignore[arg-type]
            rb = chat_reply(b, tok, turns, greedy=greedy, seed=seed, **kw)  # type: ignore[arg-type]
            replies.append(
                {
                    "prompt": turns[-1]["content"],
                    "greedy": greedy,
                    "seed": seed,
                    "identical": ra.ids == rb.ids and ra.text == rb.text,
                    "stop_reason": rb.stop_reason,
                    "reply": rb.text,
                }
            )
    files = {
        str(p.relative_to(OUT)): {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
        for p in sorted(dst.rglob("*"))
        if p.is_file()
    }
    return {
        "role": spec["role"],
        "report": spec["report"],
        "source_checkpoint": str(src),
        "source_sha256": file_sha256(src),
        "source_bytes": src.stat().st_size,
        "step": meta.step,
        "val_loss": meta.val_loss,
        "git_commit_at_training": meta.git_commit,
        "torch_version_at_training": meta.torch_version,
        "parameters": sum(p.numel() for p in b.parameters()),
        "architecture": cfg.model.__dict__,
        "tokenizer_fingerprint": tok.fingerprint(),
        "special_tokens": tok.special_tokens,
        "files": files,
        "verification": {
            "tensors_equal": tensors_equal,
            "max_abs_logit_diff": logit_diff,
            "replies_identical": all(r["identical"] for r in replies),
            "replies": replies,
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    models = {name: export(name, spec) for name, spec in MODELS.items()}
    manifest = {
        "release": "mini_chatgpt v1.0 (inference-only)",
        "default_model": "m5",
        "decoding_defaults": DEFAULTS,
        "local_path": str(OUT),
        "torch_version_at_export": str(torch.__version__),
        "models": models,
        "all_verified": all(
            m["verification"]["tensors_equal"]
            and m["verification"]["max_abs_logit_diff"] == 0.0
            and m["verification"]["replies_identical"]
            for m in models.values()
        ),
        "how_to_run": ".venv/bin/python scripts/serve_chat.py --checkpoint "
        f"{OUT}/m5/checkpoints/model.pt",
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "models"}, indent=1))
    for name, m in models.items():
        print(
            name,
            m["parameters"],
            {k: v for k, v in m["verification"].items() if k != "replies"},
            {f: d["bytes"] for f, d in m["files"].items()},
        )


if __name__ == "__main__":
    main()
