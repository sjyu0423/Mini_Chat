"""M6.3 matched evaluation: M5 vs M6.3, implementing docs/reports/m6_3/preregistration.md.

Step 1 (``run``): generate both models' chat_v1 + chat_diag_v1 replies with the production
decoding (``scripts/m6_1_decoding.py --settings t0.7_k40_rp1.2``, seeds 11/22/33, same
session), run ``scripts/evaluate_general.py`` on both, compute the pre-registered per-item
paired statistics, and write an anonymized blind-review packet. Nothing printed by this step
contains reply text, so the reviewer stays blind.

Step 2 (``unblind``): after ``blind_grades.json`` has been filled in, read the key, score the
blind review and apply the promotion rule.

Usage:
    python scripts/evaluate_m6_3.py run runs/<m6_3 run> --out docs/reports/m6_3
    python scripts/evaluate_m6_3.py unblind --out docs/reports/m6_3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics as st
import subprocess
import sys
from pathlib import Path
from typing import Any

M5 = "runs/m5_sft_chat-20261114-175334/checkpoints/best.pt"
SETTING = "t0.7_k40_rp1.2"
PROC = Path("data/processed/general_m4_1_bpe/heldout_text")
HELDOUT = {
    "fineweb_edu_val": PROC / "fineweb_edu_val",
    "cosmopedia_v2_val": PROC / "cosmopedia_v2_val",
    "wikipedia_en_ood": Path("data/raw/samples/wikipedia_en"),
    "tinystories_v2_val": PROC / "tinystories_v2_val",
}
GROUPS = {
    "primary": None,
    "factual": {("chat_v1", "qa"), ("diag_v1", "fact")},
    "instruction": {("chat_v1", "instruction"), ("diag_v1", "instruction")},
    "entity": {("diag_v1", "entity")},
    "multiturn": {("chat_v1", "multiturn"), ("diag_v1", "multiturn")},
}
BLIND_SECTIONS = [
    ("chat_v1", "qa"),
    ("diag_v1", "fact"),
    ("diag_v1", "instruction"),
    ("diag_v1", "entity"),
    ("diag_v1", "multiturn"),
]


def sh(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def per_item(rows: list[dict[str, Any]], key: str) -> dict[tuple[str, str], float]:
    """Mean over seeds of a boolean per (suite, id); rows with a None value are skipped."""
    acc: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if r[key] is None:
            continue
        acc.setdefault((r["suite"], r["id"]), []).append(float(bool(r[key])))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def paired(a: dict[Any, float], b: dict[Any, float]) -> dict[str, Any]:
    keys = sorted(a)
    diffs = [b[k] - a[k] for k in keys]
    rng = random.Random(0)
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(10_000))
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    m, k = pos + neg, min(pos, neg)
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2**m) if m else 1.0
    return {
        "items": n,
        "m5_mean": sum(a[k] for k in keys) / n,
        "m6_3_mean": sum(b[k] for k in keys) / n,
        "mean_diff": sum(diffs) / n,
        "bootstrap95": [boots[249], boots[9749]],
        "items_better": pos,
        "items_worse": neg,
        "sign_test_p": p,
    }


def run(args: argparse.Namespace) -> None:
    out: Path = args.out
    cand = str(args.run / "checkpoints" / "best.pt")
    for label, ckpt in (("m5", M5), ("m6_3", cand)):
        subprocess.run(
            [
                sys.executable,
                "scripts/m6_1_decoding.py",
                ckpt,
                "--label",
                label,
                "--out",
                str(out),
                "--settings",
                SETTING,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        cmd = [
            sys.executable,
            "scripts/evaluate_general.py",
            ckpt,
            "--out",
            str(out / f"general_eval_{label}.json"),
            "--heldout-docs",
            "400",
            "--tokenizer",
            "data/processed/general_m4_1_bpe/tokenizer.json",
        ]
        for name, d in HELDOUT.items():
            cmd += ["--heldout", f"{name}={d}"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    rows = {
        m: json.loads((out / f"decoding_{m}.json").read_text())["settings"][SETTING]["rows"]
        for m in ("m5", "m6_3")
    }
    for rs in rows.values():
        for r in rs:
            r["eot"] = r["stop_reason"] == "end_of_turn"
    a_ok, b_ok = per_item(rows["m5"], "both_ok"), per_item(rows["m6_3"], "both_ok")
    stats: dict[str, Any] = {}
    for g, members in GROUPS.items():
        ka = {
            k: v
            for k, v in a_ok.items()
            if members is None or (k[0], cat(rows["m5"], k)) in members
        }
        stats[g] = paired(ka, {k: b_ok[k] for k in ka})
    eot = paired(per_item(rows["m5"], "eot"), per_item(rows["m6_3"], "eot"))
    behaviour = {}
    for m, rs in rows.items():
        lat = [r["latency_ms"] for r in rs]
        behaviour[m] = {
            "replies": len(rs),
            "natural_end_of_turn": sum(r["eot"] for r in rs) / len(rs),
            "repetitive": sum(r["repetitive"] for r in rs) / len(rs),
            "median_words": st.median(r["words"] for r in rs),
            "mean_latency_ms": st.mean(lat),
            "median_latency_ms": st.median(lat),
            "empty_replies": sum(not r["reply"].strip() for r in rs),
            "structural_stops": sum(r["stop_reason"] == "structural_token" for r in rs),
        }
    general = {}
    for m in ("m5", "m6_3"):
        g = json.loads((out / f"general_eval_{m}.json").read_text())
        nt = g["next_token"]
        general[m] = {
            "heldout_bpb": {k: v["bits_per_byte"] for k, v in g["heldout"].items()},
            **{k: nt[k]["acc_norm"]["correct"] for k in ("facts_mc", "qa_mc", "code_mc")},
            "grammar_pairs": nt["grammar_pairs"]["acc"]["correct"],
            "code_generation": nt["code_generation"]["acc"]["correct"],
            "chat_section": {
                k: g["instruction"]["chat"][k]["correct"] for k in ("format", "content", "both")
            },
        }
    write_packet(rows, out)
    summary = {
        "preregistration_sha256": sh(out / "preregistration.md"),
        "checkpoints": {"m5": M5, "m6_3": cand},
        "checkpoint_sha256": {"m5": sh(M5), "m6_3": sh(cand)},
        "setting": SETTING,
        "paired_both_ok": stats,
        "paired_end_of_turn": eot,
        "behaviour": behaviour,
        "general": general,
    }
    (out / "comparison.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "general"}, indent=1))


def cat(rows: list[dict[str, Any]], key: tuple[str, str]) -> str:
    return str(next(r["category"] for r in rows if (r["suite"], r["id"]) == key))


def write_packet(rows: dict[str, list[dict[str, Any]]], out: Path) -> None:
    suites = {
        "chat_v1": {
            i["id"]: i for i in json.loads(Path("evals/chat_v1.json").read_text())["items"]
        },
        "diag_v1": {
            i["id"]: i for i in json.loads(Path("evals/chat_diag_v1.json").read_text())["items"]
        },
    }
    chosen = []
    for suite, category in BLIND_SECTIONS:
        ids = [i for i, it in suites[suite].items() if it["category"] == category][:4]
        chosen += [(suite, i) for i in ids]
    rng = random.Random(2026)
    key, md = (
        {},
        [
            "# M6.3 blind review packet\n",
            "Grade each reply: `correct` (right and usable), `fluent_wrong` (readable but "
            "wrong/off-task), `malformed` (broken, looping, truncated or nonsense). Then a "
            "preference: `A`, `B` or `tie`.\n",
        ],
    )
    for n, (suite, iid) in enumerate(chosen, 1):
        pair = ["m5", "m6_3"]
        rng.shuffle(pair)
        key[f"{n:02d}"] = {"suite": suite, "id": iid, "A": pair[0], "B": pair[1]}
        reply = {
            m: next(
                r for r in rows[m] if r["suite"] == suite and r["id"] == iid and r["seed"] == 11
            )
            for m in pair
        }
        md.append(f"\n## Item {n:02d}\n")
        for t in suites[suite][iid]["turns"]:
            md.append(f"*{t['role']}:* {t['content']}\n")
        for label, m in zip(("A", "B"), pair, strict=True):
            r = reply[m]
            md.append(
                f"\n**Reply {label}** (stop: {r['stop_reason']})\n\n````text\n"
                f"{r['reply'] or '(empty)'}\n````\n"
            )
    (out / "blind_packet.md").write_text("\n".join(md), encoding="utf-8")
    (out / "blind_key.json").write_text(json.dumps(key, indent=1) + "\n")
    template = {f"{n:02d}": {"A": "", "B": "", "preference": ""} for n in range(1, len(chosen) + 1)}
    grades = out / "blind_grades.json"
    if not grades.exists():
        grades.write_text(json.dumps(template, indent=1) + "\n")


def unblind(args: argparse.Namespace) -> None:
    out: Path = args.out
    key = json.loads((out / "blind_key.json").read_text())
    grades = json.loads((out / "blind_grades.json").read_text())
    tally = {
        m: {"correct": 0, "fluent_wrong": 0, "malformed": 0, "preferred": 0} for m in ("m5", "m6_3")
    }
    ties = 0
    for n, k in key.items():
        g = grades[n]
        for label in ("A", "B"):
            if g[label] not in ("correct", "fluent_wrong", "malformed"):
                raise SystemExit(f"item {n}: grade {label} missing or invalid")
            tally[k[label]][g[label]] += 1
        if g["preference"] == "tie":
            ties += 1
        elif g["preference"] in ("A", "B"):
            tally[k[g["preference"]]]["preferred"] += 1
        else:
            raise SystemExit(f"item {n}: preference missing")
    comp = json.loads((out / "comparison.json").read_text())
    p, b = comp["paired_both_ok"]["primary"], comp["behaviour"]
    bpb = {m: comp["general"][m]["heldout_bpb"]["fineweb_edu_val"] for m in ("m5", "m6_3")}
    rules = {
        "1_credible": p["mean_diff"] > 0 and p["bootstrap95"][0] > 0 and p["sign_test_p"] < 0.05,
        "2_meaningful_5pp": p["m6_3_mean"] - p["m5_mean"] >= 0.05,
        "3_blind_review": tally["m6_3"]["correct"] > tally["m5"]["correct"]
        and tally["m6_3"]["preferred"] > tally["m5"]["preferred"],
        "4a_end_of_turn": b["m6_3"]["natural_end_of_turn"] >= 0.95
        and b["m6_3"]["natural_end_of_turn"] >= b["m5"]["natural_end_of_turn"] - 0.02,
        "4b_repetition": b["m6_3"]["repetitive"] <= 0.05,
        "4c_latency": b["m6_3"]["mean_latency_ms"] <= 2 * b["m5"]["mean_latency_ms"]
        and b["m6_3"]["median_latency_ms"] < 1000,
        "4d_retention": bpb["m6_3"] <= bpb["m5"] + 0.02,
    }
    decision = {
        "blind_review": {"tally": tally, "ties": ties, "items": len(key)},
        "rules": rules,
        "stability_note": "4 (no crashes/NaN/guard aborts) is checked from the training log",
        "promote": all(rules.values()),
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=1) + "\n")
    print(json.dumps(decision, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("run", type=Path)
    r.add_argument("--out", type=Path, default=Path("docs/reports/m6_3"))
    u = sub.add_parser("unblind")
    u.add_argument("--out", type=Path, default=Path("docs/reports/m6_3"))
    args = p.parse_args()
    run(args) if args.cmd == "run" else unblind(args)


if __name__ == "__main__":
    main()
