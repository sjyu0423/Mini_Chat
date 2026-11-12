"""Generate text from a checkpoint using the tokenizer it was trained with.

Example:
    python scripts/generate.py runs/<run>/checkpoints/step_00000100.pt --prompt "ROMEO:" \
        --max-new-tokens 200 --temperature 0.8 --top-k 40 --seed 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mini_chatgpt.data.prepare import processed_dir
from mini_chatgpt.device import select_device
from mini_chatgpt.inference.generate import generate_text
from mini_chatgpt.tokenizer.char import CharTokenizer
from mini_chatgpt.training.trainer import load_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    model, cfg = load_model(args.checkpoint, select_device(args.device))
    run_tokenizer = args.checkpoint.resolve().parent.parent / "tokenizer.json"
    tok_path = (
        run_tokenizer if run_tokenizer.exists() else processed_dir(cfg.data) / "tokenizer.json"
    )
    tokenizer = CharTokenizer.load(tok_path)
    if tokenizer.vocab_size != cfg.model.vocab_size:
        raise SystemExit("tokenizer does not match the checkpoint's vocab_size")
    print(
        generate_text(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            greedy=args.greedy,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
