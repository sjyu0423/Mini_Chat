"""Print configs, selected device, and parameter-count ESTIMATES.

Usage: python scripts/show_config.py configs/gpt_10m.toml [configs/...]
"""

from __future__ import annotations

import argparse

from mini_chatgpt.config import ExperimentConfig, format_config
from mini_chatgpt.device import describe_device, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()
    for path in args.configs:
        cfg = ExperimentConfig.from_toml(path)
        device = select_device(cfg.training.device)
        print(f"# {path}")
        print(format_config(cfg))
        print(f"  {'device':<28} = {describe_device(device)}")
        est50k = cfg.with_vocab_size(50_257).model.estimate_parameters()["total"]
        print(f"  {'total if vocab=50,257 (BPE)':<28} = {est50k:,}")
        print()


if __name__ == "__main__":
    main()
