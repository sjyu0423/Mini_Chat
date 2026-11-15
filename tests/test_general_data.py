"""M4.1 general-data preparation helpers and single-pass training traversal."""

import hashlib

import pytest
import torch

from mini_chatgpt.config import DataOrder, ExperimentConfig, ModelConfig, TrainingConfig
from mini_chatgpt.data import general as g
from mini_chatgpt.data.dataset import get_windows, single_pass_order
from mini_chatgpt.data.tinystories import MinHashConfig
from mini_chatgpt.training.trainer import Trainer

LONG = " ".join(f"word{i % 37} alpha beta gamma." for i in range(40))


def test_clean_text_is_conservative_and_idempotent():
    raw = "  Title\r\n\r\n\r\n\r\nBody text  \t\nmore\x00 text\x07 é  "
    out = g.clean_text(raw)
    assert out == "Title\n\nBody text\nmore text é"
    assert g.clean_text(out) == out


def test_quality_reject_rules():
    assert g.quality_reject("too short") == "fewer_than_50_words"
    assert g.quality_reject(LONG) is None
    assert g.quality_reject(LONG + "�") == "replacement_character"
    dup = "\n".join(["this exact sentence is repeated over and over again."] * 30)
    assert g.quality_reject(dup) == "duplicate_lines_over_30pct"
    fenced = "\n".join(["```python", "    x = f(i)", "```"] * 10)
    assert g.quality_reject(LONG + "\n" + fenced) is None  # short structural lines ignored
    assert g.quality_reject(" ".join(["12345 67890"] * 60)) == "letters_under_50pct"


def test_split_rule_matches_buckets():
    for i in range(2000):
        sha = hashlib.sha256(str(i).encode()).hexdigest()
        b = int(sha[:8], 16) % 1000
        expected = "val" if b < 5 else "holdout" if b < 10 else "train"
        assert g.split_of(sha) == expected


def test_dedup_exact_near_and_cross_source():
    base = " ".join(f"token{i}" for i in range(300))
    near = base.replace("token150", "changed150")
    docs = [g.Doc("a", base), g.Doc("b", base), g.Doc("b", near), g.Doc("a", LONG)]
    kept, st = g.dedup_documents(docs, MinHashConfig())
    assert len(kept) == 2
    assert sum(st.exact_removed.values()) == 1 and sum(st.near_removed.values()) == 1
    assert st.cross_source_near_pairs == 1
    assert [d.sha for d in kept] == sorted(d.sha for d in kept)


def test_decontamination_ngrams():
    index = g.eval_ngrams({"long": "one two three four five six seven eight nine", "short": "hi"})
    assert all(len(k) == 8 for k in index)
    assert g.contaminating_items("x One two three four five six seven EIGHT y", index) == {"long"}
    assert g.contaminating_items("one two three four five six seven", index) == set()


def test_verify_file_and_read_texts(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"text": ["a", None, "b"], "other": [1, 2, 3]}), path)
    assert list(g.read_texts(path, "text")) == ["a", "b"]
    data = path.read_bytes()
    src = g.Source("t", "r", "rev", "p", len(data), hashlib.sha256(data).hexdigest(), "", "text",
                   1.0)  # fmt: skip
    assert g.verify_file(path, src) == src.sha256
    bad = g.Source("t", "r", "rev", "p", len(data), "0" * 64, "", "text", 1.0)
    with pytest.raises(RuntimeError, match="sha256"):
        g.verify_file(path, bad)


def test_pinned_sources_match_manifest_plan():
    import json
    from pathlib import Path

    plan = json.loads(Path("docs/reports/m4/m4_1_data_manifest_plan.json").read_text())
    for src, planned in zip(g.SOURCES, plan["sources"], strict=True):
        assert (src.revision, src.path, src.size, src.sha256) == (
            planned["revision"], planned["file"], planned["bytes"], planned["sha256"])  # fmt: skip
        assert src.token_share == planned["share_of_training_tokens"]


# --------------------------------------------------------------------------- #
# single-pass traversal
# --------------------------------------------------------------------------- #
def test_single_pass_order_is_a_seeded_permutation():
    order = single_pass_order(1001, 10, seed=3)
    assert sorted(order.tolist()) == list(range(100))
    assert torch.equal(order, single_pass_order(1001, 10, seed=3))
    x, y = get_windows(torch.arange(1001), order[:2], 10)
    assert torch.equal(y[:, :-1], x[:, 1:]) and x[0, 0] == order[0] * 10


def _cfg(max_steps: int, order: DataOrder = "single_pass") -> ExperimentConfig:
    return ExperimentConfig(
        name="sp",
        model=ModelConfig(vocab_size=50, context_length=8, n_layer=1, n_head=1, n_embd=8),
        training=TrainingConfig(
            batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=max_steps,
            warmup_steps=0,
            data_order=order,
            device="cpu",
        ),
    )


def test_trainer_single_pass_visits_each_window_once(monkeypatch):
    data = torch.arange(161) % 50  # 20 windows of 8 tokens
    t = Trainer(_cfg(5), data, data, torch.device("cpu"))
    seen: list[int] = []
    import mini_chatgpt.training.trainer as tr

    real = tr.get_windows

    def spy(tokens, ids, ctx, *, device="cpu"):
        seen.extend(ids.tolist())
        return real(tokens, ids, ctx, device=device)

    monkeypatch.setattr(tr, "get_windows", spy)
    for _ in range(5):
        t.train_step()
    assert sorted(seen) == list(range(20))  # 5 steps x 2 micro x 2 = all 20, no repeats
    with pytest.raises(RuntimeError, match="exhausted"):
        t.train_step()


def test_trainer_single_pass_refuses_plans_that_would_repeat():
    data = torch.arange(161) % 50
    with pytest.raises(ValueError, match="refusing to repeat"):
        Trainer(_cfg(6), data, data, torch.device("cpu"))
    Trainer(_cfg(6, order="random"), data, data, torch.device("cpu"))  # old behaviour intact


def test_single_pass_resume_continues_the_same_order(tmp_path):
    data = torch.arange(161) % 50
    a = Trainer(_cfg(5), data, data, torch.device("cpu"))
    a.train_step()
    a.train_step()
    path = a.save(tmp_path)
    b = Trainer(_cfg(5), data, data, torch.device("cpu"))
    b.load_state(path)
    a.train_step()
    b.train_step()
    for pa_, pb in zip(a.model.parameters(), b.model.parameters(), strict=True):
        assert torch.equal(pa_, pb)
