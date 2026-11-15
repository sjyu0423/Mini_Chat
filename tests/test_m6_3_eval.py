"""The M6.3 pre-registered statistics: per-item seed aggregation, sign test, bootstrap."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "evaluate_m6_3", ROOT / "scripts" / "evaluate_m6_3.py"
)
assert spec and spec.loader
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_per_item_averages_seeds_and_skips_unchecked():
    rows = [
        {"suite": "s", "id": "a", "ok": True},
        {"suite": "s", "id": "a", "ok": False},
        {"suite": "s", "id": "a", "ok": True},
        {"suite": "s", "id": "b", "ok": None},
    ]
    assert ev.per_item(rows, "ok") == {("s", "a"): pytest.approx(2 / 3)}


def test_paired_sign_test_and_bootstrap():
    a = {i: 0.0 for i in range(10)}
    b = {i: 1.0 for i in range(10)}
    r = ev.paired(a, b)
    assert r["items_better"] == 10 and r["items_worse"] == 0
    assert r["sign_test_p"] == pytest.approx(2 / 2**10)
    assert r["mean_diff"] == 1.0 and r["bootstrap95"] == [1.0, 1.0]
    same = ev.paired(a, dict(a))
    assert same["sign_test_p"] == 1.0 and same["mean_diff"] == 0.0
    mixed = ev.paired({0: 0.0, 1: 1.0}, {0: 1.0, 1: 0.0})
    assert mixed["items_better"] == mixed["items_worse"] == 1 and mixed["sign_test_p"] == 1.0
