"""Chat with a mini_chatgpt checkpoint in the terminal (M5).

Interactive (multi-turn history kept until /reset):
    python scripts/chat.py runs/<m5 run>/checkpoints/best.pt
Single prompt:
    python scripts/chat.py runs/<m5 run>/checkpoints/best.pt --prompt "What is Python?"

Options: --temperature, --top-k, --top-p, --greedy, --repetition-penalty (1.0 = off),
--max-new-tokens, --seed. A reply that hit the token limit instead of ending with
<|end|> is marked "[truncated]"; a format error is marked "[stopped: ...]".
Commands in interactive mode: /reset clears the history, /quit exits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mini_chatgpt.chat.format import Message
from mini_chatgpt.chat.generate import ChatReply, chat_reply
from mini_chatgpt.device import select_device
from mini_chatgpt.tokenizer.bpe import ByteBPETokenizer
from mini_chatgpt.training.trainer import load_model


def note(r: ChatReply) -> str:
    if r.stop_reason == "end_of_turn":
        return ""
    if r.truncated:
        return f"  [truncated: {r.stop_reason}, {len(r.ids)} tokens]"
    return f"  [stopped: {r.stop_reason}]"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--prompt", default=None)
    p.add_argument("--tokenizer", type=Path, default=None)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40, help="0 disables top-k")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=None, help="nucleus sampling; off by default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    model, _ = load_model(args.checkpoint, select_device(args.device))
    tok = ByteBPETokenizer.load(args.tokenizer or args.checkpoint.parent.parent / "tokenizer.json")
    kw = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k or None,
        "greedy": args.greedy,
        "repetition_penalty": args.repetition_penalty,
        "top_p": args.top_p,
    }
    if args.prompt is not None:
        r = chat_reply(model, tok, [{"role": "user", "content": args.prompt}], seed=args.seed, **kw)
        print(r.text + note(r))
        return
    history: list[Message] = []
    turn = 0
    print("mini_chatgpt — /reset to clear history, /quit to exit")
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if text == "/quit":
            break
        if text == "/reset":
            history, turn = [], 0
            continue
        if not text:
            continue
        history.append({"role": "user", "content": text})
        r = chat_reply(model, tok, history, seed=args.seed + turn, **kw)
        turn += 1
        print("bot> " + r.text + note(r))
        if r.dropped_history_messages:
            history = history[r.dropped_history_messages :]
        history.append({"role": "assistant", "content": r.text or "…"})


if __name__ == "__main__":
    main()
