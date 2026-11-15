"""TinyStories acquisition/preprocessing and the BPE memmap data path (no network)."""

import itertools
import json

import numpy as np
import pytest
import torch

from mini_chatgpt.data import tinystories as ts
from mini_chatgpt.data.dataset import get_batch, iter_sequential_batches
from mini_chatgpt.data.prepare import file_sha256, load_prepared
from mini_chatgpt.tokenizer import ByteBPETokenizer, CharTokenizer, load_tokenizer

# --------------------------------------------------------------------------- #
# range planning and verification
# --------------------------------------------------------------------------- #


def test_plan_ranges_non_overlapping_inside_strata_and_seeded():
    plan = ts.plan_ranges(10_000, 8, 500, seed=1)
    assert plan == ts.plan_ranges(10_000, 8, 500, seed=1)
    assert plan != ts.plan_ranges(10_000, 8, 500, seed=2)
    stratum = 10_000 // 8
    for i, (s, e) in enumerate(plan):
        assert e - s + 1 == 500
        assert i * stratum <= s and e < (i + 1) * stratum
    assert all(b[0] > a[1] for a, b in itertools.pairwise(plan))
    with pytest.raises(ValueError):
        ts.plan_ranges(1000, 8, 500, seed=0)


class FakeResp:
    def __init__(self, status, content_range, body):
        self.status, self.body = status, body
        self.headers = {"Content-Range": content_range, "ETag": '"x"'}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.body


def opener_for(status, content_range, body):
    return lambda req: FakeResp(status, content_range, body)


def test_fetch_range_accepts_exact_206():
    body, info = ts.fetch_range(
        "https://example.test/f", 10, 19, 100, opener=opener_for(206, "bytes 10-19/100", b"x" * 10)
    )
    assert body == b"x" * 10 and info["status"] == 206 and len(info["sha256"]) == 64


@pytest.mark.parametrize(
    ("status", "content_range", "body", "match"),
    [
        (200, "", b"x" * 100, "206"),  # server ignored the Range header
        (206, "bytes 0-9/100", b"x" * 10, "does not match"),  # wrong range
        (206, "bytes 10-19/999", b"x" * 10, "file size"),  # upstream file changed
        (206, "bytes 10-19/100", b"x" * 9, "got 9 bytes"),  # truncated body
        (206, "garbage", b"x" * 10, "does not match"),
    ],
)
def test_fetch_range_rejects_bad_responses(status, content_range, body, match):
    with pytest.raises(RuntimeError, match=match):
        ts.fetch_range(
            "https://example.test/f", 10, 19, 100, opener=opener_for(status, content_range, body)
        )


# --------------------------------------------------------------------------- #
# story boundary recovery
# --------------------------------------------------------------------------- #


def test_split_stories_drops_partial_edges():
    blob = (
        b"tail of a story.\n<|endoftext|>\nOne.\n<|endoftext|>\n  Two  \n<|endoftext|>\nhalf a st"
    )
    stats = ts.ParseStats()
    assert ts.split_stories(blob, partial_edges=True, stats=stats) == ["One.", "Two"]
    assert stats.head_bytes_dropped == len(b"tail of a story.\n")
    assert stats.tail_bytes_dropped == len(b"\nhalf a st")


def test_split_stories_never_cuts_utf8_and_drops_invalid():
    good = "Café “quotes” 😀".encode()
    blob = b"x<|endoftext|>" + good + b"<|endoftext|>\xff\xfe bad<|endoftext|><|endoftext|>y"
    stats = ts.ParseStats()
    out = ts.split_stories(blob, partial_edges=True, stats=stats)
    assert out == ["Café “quotes” 😀"]
    assert stats.invalid_utf8_dropped == 1 and stats.empty_dropped == 1


def test_split_stories_full_file_keeps_edges():
    stats = ts.ParseStats()
    assert ts.split_stories(b"A\n<|endoftext|>\nB\n", partial_edges=False, stats=stats) == [
        "A",
        "B",
    ]


def test_range_without_two_delimiters_yields_nothing():
    stats = ts.ParseStats()
    assert (
        ts.split_stories(b"no delimiter <|endoftext|> here", partial_edges=True, stats=stats) == []
    )


# --------------------------------------------------------------------------- #
# deduplication and splits
# --------------------------------------------------------------------------- #

WORDS = [
    "cat",
    "dog",
    "sun",
    "tree",
    "ball",
    "red",
    "blue",
    "park",
    "happy",
    "sad",
    "jump",
    "run",
    "play",
    "friend",
    "mom",
    "dad",
    "house",
    "bird",
    "fish",
    "cake",
    "rain",
    "wind",
    "star",
    "moon",
    "boat",
    "car",
    "toy",
    "book",
    "hat",
    "shoe",
    "door",
    "cup",
]


def story(seed: int, n: int = 80) -> str:
    rng = np.random.default_rng(seed)
    return " ".join(WORDS[i] for i in rng.integers(0, len(WORDS), n))


BASE = story(1)


def test_deduplicate_exact_near_and_cross_split():
    near = BASE + " extra"  # one word appended -> Jaccard ~ 0.99
    other = story(2)
    val_story = story(3)
    val_near = "and " + val_story
    train = [BASE, BASE, near, other, val_story, val_near]
    kept, rep = ts.deduplicate(train, [val_story], ts.MinHashConfig())
    assert rep.exact_duplicates_removed == 1
    assert rep.removed_for_matching_validation_exact == 1
    assert rep.removed_for_matching_validation_near == 1
    assert rep.near_duplicates_removed == 1
    assert kept == [BASE, other]  # earliest member of each cluster survives
    assert rep.kept == 2


def test_minhash_estimates_jaccard():
    cfg = ts.MinHashConfig()
    sigs = ts.minhash_signatures(
        [BASE, BASE.replace("sunny", "warm"), "totally different text"], cfg
    )
    assert np.mean(sigs[0] == sigs[1]) > 0.7
    assert np.mean(sigs[0] == sigs[2]) < 0.1


def test_holdout_split_is_deterministic_and_disjoint():
    stories = [f"story number {i}" for i in range(2000)]
    keep, hold = ts.holdout_split(stories, 1.0)
    assert (keep, hold) == ts.holdout_split(list(stories), 1.0)
    assert set(keep).isdisjoint(hold) and len(keep) + len(hold) == 2000
    assert 5 <= len(hold) <= 40  # ~1%


def test_list_digest_is_order_sensitive():
    assert ts.list_digest(["a", "b"]) != ts.list_digest(["b", "a"])


# --------------------------------------------------------------------------- #
# BPE memmap data path
# --------------------------------------------------------------------------- #


def write_bpe_dataset(tmp_path, n_train=3000):
    stories = [f"Tom had {i} red balls. He was happy." for i in range(200)]
    tok = ByteBPETokenizer.train(stories, vocab_size=280)
    eot = tok.eot_id
    out = tmp_path / "processed"
    out.mkdir()
    streams = {}
    for name, ss in (("train", stories[:150]), ("val", stories[150:])):
        ids = [t for s in ss for t in [*tok.encode(s), eot]]
        arr = np.asarray(ids, dtype="<u2")
        (out / f"{name}.bin").write_bytes(arr.tobytes())
        streams[name] = arr
    tok.save(out / "tokenizer.json")
    meta = {
        "tokenizer": "bpe",
        "vocab_size": tok.vocab_size,
        "tokenizer_fingerprint": tok.fingerprint(),
        "num_train_tokens": int(streams["train"].size),
        "num_val_tokens": int(streams["val"].size),
        "train_sha256": file_sha256(out / "train.bin"),
        "val_sha256": file_sha256(out / "val.bin"),
    }
    (out / "meta.json").write_text(json.dumps(meta))
    return out, tok, streams


def test_bpe_memmap_load_and_batches(tmp_path):
    out, tok, streams = write_bpe_dataset(tmp_path)
    data = load_prepared(out)
    assert isinstance(data.tokenizer, ByteBPETokenizer)
    assert data.splits.train.dtype == torch.uint16  # memory-mapped, not copied to int64
    assert torch.equal(
        data.splits.train.long(), torch.from_numpy(streams["train"].astype(np.int64))
    )
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    x1, y1 = get_batch(data.splits.train, 4, 16, generator=g1)
    x2, y2 = get_batch(data.splits.train, 4, 16, generator=g2)
    assert x1.dtype == torch.int64 and torch.equal(x1, x2) and torch.equal(y1, y2)
    assert torch.equal(x1[:, 1:], y1[:, :-1])
    xs = list(iter_sequential_batches(data.splits.val, 2, 8))
    assert all(x.dtype == torch.int64 for x, _ in xs)
    # the stream decodes back to the stories joined by the end-of-text token
    text = tok.decode(streams["val"].tolist())
    assert text.count("<|endoftext|>") == 50 and text.startswith("Tom had 150 red balls.")


@pytest.mark.parametrize("corruption", ["hash", "fingerprint", "count", "range"])
def test_bpe_memmap_integrity_checks(tmp_path, corruption):
    out, tok, streams = write_bpe_dataset(tmp_path)
    meta = json.loads((out / "meta.json").read_text())
    if corruption == "hash":
        arr = streams["train"].copy()
        arr[0] = (int(arr[0]) + 1) % 256
        (out / "train.bin").write_bytes(arr.tobytes())
    elif corruption == "fingerprint":
        meta["tokenizer_fingerprint"] = "0" * 64
    elif corruption == "count":
        meta["num_train_tokens"] += 1
    else:  # an id outside the vocabulary, with a matching hash
        arr = streams["train"].copy()
        arr[0] = tok.vocab_size + 5
        (out / "train.bin").write_bytes(arr.tobytes())
        meta["train_sha256"] = file_sha256(out / "train.bin")
    (out / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError):
        load_prepared(out)


def test_memmap_load_never_writes_the_file(tmp_path):
    out, _, _ = write_bpe_dataset(tmp_path)
    before = file_sha256(out / "train.bin")
    data = load_prepared(out)
    data.splits.train[0] = 1  # copy-on-write mapping: only the in-memory page changes
    assert file_sha256(out / "train.bin") == before


def test_load_tokenizer_dispatches(tmp_path):
    c = CharTokenizer.from_text("abc")
    c.save(tmp_path / "c.json")
    b = ByteBPETokenizer.train(["abc abc abc"], vocab_size=260)
    b.save(tmp_path / "b.json")
    assert isinstance(load_tokenizer(tmp_path / "c.json"), CharTokenizer)
    assert isinstance(load_tokenizer(tmp_path / "b.json"), ByteBPETokenizer)
    (tmp_path / "x.json").write_text('{"type": "nope"}')
    with pytest.raises(ValueError):
        load_tokenizer(tmp_path / "x.json")
