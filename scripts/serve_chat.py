"""Run the local chat demo: inference API + (if built) the web frontend on one port.

    python scripts/serve_chat.py
    # then open http://127.0.0.1:8000

Options: --checkpoint (default: the M5 best checkpoint), --device (auto: MPS if
available, else CPU), --port, --host (default 127.0.0.1; the model is meant to stay
local — binding elsewhere prints a warning).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from mini_chatgpt.serve.app import create_app
from mini_chatgpt.serve.engine import ChatEngine

DEFAULT_CHECKPOINT = "runs/m5_sft_chat-20261114-175334/checkpoints/best.pt"
STATIC = Path("apps/chat-web/dist")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    p.add_argument("--device", default="auto")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        logging.warning("binding to %s exposes the model beyond this machine", args.host)
    if not args.checkpoint.exists():
        raise SystemExit(f"{args.checkpoint} not found; train M5 or pass --checkpoint")
    if not (STATIC / "index.html").exists():
        logging.warning(
            "%s not built; API only (run: npm --prefix apps/chat-web run build)", STATIC
        )
    app = create_app(
        lambda: ChatEngine.from_checkpoint(args.checkpoint, device=args.device), static_dir=STATIC
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
