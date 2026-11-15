"""M6 performance and stability benchmark of the real local chat server (no mocks).

Starts ``scripts/serve_chat.py`` as a subprocess on a spare port, then over HTTP:

1. startup: wall-clock from process start to ``/api/health`` = ready (Python + torch
   import, checkpoint load, tokenizer check), and the server's own model-load time;
2. single-turn latency: every chat_v1 user prompt once (40 requests, fixed seeds);
3. repeated conversations: 10 conversations x 6 turns, each turn sending the full
   history back (exercises history serialization and context trimming);
4. a soak of 60 more requests while sampling MPS memory and the process RSS, to show
   that memory stays flat across many generations.

Writes ``docs/reports/m6/performance.json``.

Usage: python scripts/benchmark_chat_server.py [--out docs/reports/m6/performance.json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

PORT = 8791
URL = f"http://127.0.0.1:{PORT}"
FOLLOW_UPS = [
    "Tell me more.",
    "Why is that?",
    "Can you give an example?",
    "What else should I know?",
    "Thanks! Can you summarize?",
]


def rss_mb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
    return int(out.stdout.strip() or 0) / 1024


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lat = [r["client_ms"] for r in rows]
    tps = [r["tokens_per_s"] for r in rows if r["new_tokens"] >= 10]
    stops: dict[str, int] = {}
    for r in rows:
        stops[r["stop_reason"]] = stops.get(r["stop_reason"], 0) + 1
    return {
        "requests": len(rows),
        "latency_ms_mean": round(st.mean(lat), 1),
        "latency_ms_median": round(st.median(lat), 1),
        "latency_ms_p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1),
        "latency_ms_max": round(max(lat), 1),
        "new_tokens_mean": round(st.mean(r["new_tokens"] for r in rows), 1),
        "tokens_per_s_median": round(st.median(tps), 1) if tps else None,
        "server_ms_mean": round(st.mean(r["duration_ms"] for r in rows), 1),
        "stop_reasons": stops,
        "history_messages_dropped_total": sum(r["dropped_history_messages"] for r in rows),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("docs/reports/m6/performance.json"))
    args = p.parse_args()
    items = json.loads(Path("evals/chat_v1.json").read_text())["items"]
    prompts = [i["turns"][-1]["content"] for i in items]

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "scripts/serve_chat.py", "--port", str(PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    result: dict[str, Any] = {}
    try:
        client = httpx.Client(base_url=URL, timeout=120)
        health: dict[str, Any] = {}
        while time.perf_counter() - t0 < 120:
            try:
                health = client.get("/api/health").json()
                if health["status"] in ("ready", "error"):
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.1)
        if health.get("status") != "ready":
            raise SystemExit(f"server not ready: {health}")
        result["startup"] = {
            "process_start_to_ready_s": round(time.perf_counter() - t0, 2),
            "model_load_s_reported_by_server": health["model"]["load_seconds"],
            "device": health["model"]["device"],
            "memory_after_load": health["memory"],
            "rss_mb_after_load": round(rss_mb(proc.pid), 1),
        }

        def ask(message: str, history: list[dict[str, str]], seed: int) -> dict[str, Any]:
            c0 = time.perf_counter()
            r = client.post(
                "/api/chat",
                json={"message": message, "history": history, "settings": {"seed": seed}},
            )
            r.raise_for_status()
            row: dict[str, Any] = r.json()
            row["client_ms"] = (time.perf_counter() - c0) * 1000
            return row

        ask("Hi", [], 0)  # warm-up (first MPS kernels), not counted
        single = [ask(q, [], 1000 + k) for k, q in enumerate(prompts)]
        result["single_turn"] = summarize(single)

        conv_rows: list[dict[str, Any]] = []
        for c in range(10):
            history: list[dict[str, str]] = []
            turns = [prompts[(c * 4) % len(prompts)], *FOLLOW_UPS]
            for t, msg in enumerate(turns):
                row = ask(msg, history, 2000 + 10 * c + t)
                row["turn"] = t + 1
                conv_rows.append(row)
                if row["reply"].strip():
                    history += [
                        {"role": "user", "content": msg},
                        {"role": "assistant", "content": row["reply"]},
                    ]
        result["conversations"] = summarize(conv_rows)
        result["conversations"]["by_turn_latency_ms_mean"] = {
            t: round(st.mean(r["client_ms"] for r in conv_rows if r["turn"] == t), 1)
            for t in range(1, 7)
        }
        result["conversations"]["by_turn_prompt_tokens_mean"] = {
            t: round(st.mean(r["prompt_tokens"] for r in conv_rows if r["turn"] == t), 1)
            for t in range(1, 7)
        }
        result["example_conversation"] = [
            {"turn": r["turn"], "user": turns_msg, "reply": r["reply"], "stop": r["stop_reason"]}
            for r, turns_msg in zip(conv_rows[:6], [prompts[0], *FOLLOW_UPS], strict=True)
        ]

        soak: list[dict[str, Any]] = []
        for k in range(60):
            soak.append(ask(prompts[k % len(prompts)], [], 3000 + k))
            if k % 10 == 9:
                h = client.get("/api/health").json()
                soak[-1]["memory"] = h["memory"]
                soak[-1]["rss_mb"] = rss_mb(proc.pid)
        result["soak"] = summarize(soak)
        result["soak"]["memory_every_10_requests"] = [
            {
                "after_requests": i + 1,
                "mps_driver_gb": round(r["memory"].get("mps_driver_allocated", 0) / 1e9, 3),
                "mps_allocated_gb": round(r["memory"].get("mps_current_allocated", 0) / 1e9, 3),
                "rss_mb": round(r["rss_mb"], 1),
            }
            for i, r in enumerate(soak)
            if "memory" in r
        ]
        final = client.get("/api/health").json()
        result["requests_served_total"] = final["requests_served"]
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "example_conversation"}, indent=1))


if __name__ == "__main__":
    main()
