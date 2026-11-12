"""Pipeline tests that use a synthetic corpus (no network access)."""

import torch

from mini_chatgpt.config import DataConfig
from mini_chatgpt.data import prepare, sources
from mini_chatgpt.data.prepare import build, load_prepared, save


def fake_corpus(monkeypatch, tmp_path, text):
    data = text.encode()
    src = sources.TextSource(
        name="fake",
        url="http://invalid",
        filename="input.txt",
        sha256=sources.sha256_bytes(data),
        num_bytes=len(data),
        license="test",
        description="",
    )
    monkeypatch.setitem(sources.SOURCES, "fake", src)
    path = sources.raw_path(src, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return DataConfig(dataset="fake", data_dir=str(tmp_path), val_fraction=0.25)


def test_build_save_load(monkeypatch, tmp_path):
    text = "To be, or not to be, that is the question.\n" * 10
    cfg = fake_corpus(monkeypatch, tmp_path, text)
    p = build(cfg, allow_download=False)
    assert p.meta["num_tokens"] == len(text)
    assert p.meta["num_train_tokens"] + p.meta["num_val_tokens"] == len(text)
    assert p.tokenizer.decode(torch.cat([p.splits.train, p.splits.val]).tolist()) == text
    out = save(p, prepare.processed_dir(cfg))
    loaded = load_prepared(out)
    assert torch.equal(loaded.splits.train, p.splits.train)
    assert loaded.tokenizer.chars == p.tokenizer.chars


def test_hash_mismatch_detected(monkeypatch, tmp_path):
    cfg = fake_corpus(monkeypatch, tmp_path, "hello world\n" * 5)
    sources.raw_path(sources.SOURCES["fake"], tmp_path).write_text("tampered")
    try:
        build(cfg, allow_download=False)
    except sources.DataIntegrityError:
        pass
    else:
        raise AssertionError("expected DataIntegrityError")
