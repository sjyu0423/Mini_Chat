"""Statistics helpers and unblinding in scripts/evaluate_m3_3.py."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "evaluate_m3_3", ROOT / "scripts" / "evaluate_m3_3.py"
)
assert spec and spec.loader
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_sign_test_and_wilson():
    assert ev.sign_test(0, 0) == 1.0
    assert ev.sign_test(5, 5) == 1.0
    assert ev.sign_test(10, 0) == pytest.approx(2 / 1024)
    assert ev.sign_test(7, 5) == pytest.approx(0.774, abs=1e-3)  # matches the M3.2 value
    lo, hi = ev.wilson(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2
    lo, hi = ev.wilson(10, 20)
    assert lo < 0.5 < hi


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0] * 4
    lo, hi = ev.bootstrap_ci(xs)
    assert lo < 3.0 < hi
    assert (lo, hi) == ev.bootstrap_ci(xs)
    assert ev.bootstrap_ci([2.0] * 10) == (2.0, 2.0)


def test_flops_per_token_formula():
    from mini_chatgpt.config import ExperimentConfig

    s = ExperimentConfig.from_toml(ROOT / "configs" / "tinystories_s_bpe4096.toml")
    m = ExperimentConfig.from_toml(ROOT / "configs" / "tinystories_m_bpe4096.toml")
    assert ev.flops_per_token(s) == 6 * 12_417_024 + 6 * 6 * 512 * 384
    assert ev.flops_per_token(m) == 6 * 27_579_392 + 6 * 8 * 512 * 512


def test_unblind_paired_and_rates(tmp_path):
    key, scores = {}, {}
    n = 0
    for prompt in ("p1", "p2"):
        for seed in (1, 2):
            for model, total in (("A", 4), ("B", 5), ("C", 8)):
                n += 1
                sid = f"S{n:03d}"
                key[sid] = {"model": model, "prompt_id": prompt, "category": "cat", "seed": seed}
                g = [2, 2, 2, 2, 0] if total == 8 else ([1, 1, 1, 1, 0] if total == 4 else
                                                        [1, 1, 1, 1, 1])  # fmt: skip
                scores[sid] = dict(zip(ev.GRADED, g, strict=True)) | {"X": int(model == "A")}
    (tmp_path / "blind_key.json").write_text(json.dumps(key))
    (tmp_path / "blind_scores.json").write_text(json.dumps(scores))
    ev.unblind(tmp_path)
    res = json.loads((tmp_path / "blind_results.json").read_text())
    assert res["per_model"]["C"]["total_0_10"]["mean"] == 8
    assert res["paired"]["C-A"]["mean_total_diff"] == 4 and res["paired"]["C-A"]["wins"] == 4
    assert res["paired"]["C-A"]["contradiction_discordant"]["only_A"] == 4
    assert res["per_model"]["A"]["contradiction_rate"]["rate"] == 1.0
    del scores["S001"]
    (tmp_path / "blind_scores.json").write_text(json.dumps(scores))
    with pytest.raises(SystemExit):
        ev.unblind(tmp_path)
