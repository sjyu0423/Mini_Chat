import json
import random

import pytest

from mini_chatgpt.tokenizer import ByteBPETokenizer, Tokenizer

CORPUS = [
    "The quick brown fox jumps over the lazy dog. The dog sleeps.",
    "Once upon a time, there was a little girl named Lily. She liked to play.",
    "low lower lowest newer wider wildest, 1234567 + 89 = 1234656!",
    "Unicode: héllo wörld, naïve café, Ελληνικά, русский, 日本語, 😀🚀 emoji.",
    "def foo_bar(x):\n\treturn x  # tabs\tand  double  spaces\n\n",
] * 20

EDGE_CASES = [
    "",
    " ",
    "   ",
    "\n",
    "\n\n\t \r\n",
    "a",
    "hello world",
    "  leading and trailing  ",
    "punctuation!?.,;:'\"()[]{}<>/\\|@#$%^&*-_=+~`",
    "don't won't they're I'll we've she'd",
    "snake_case_name __dunder__ _x_",
    "numbers 0 12 345 6789 1234567890 3.14159 -42",
    "héllo wörld naïve café",
    "日本語のテキスト、中文文本。한국어",
    "😀🚀👍🏽 family: 👨‍👩‍👧‍👦 flags: 🇯🇵",
    "combining: é ä (decomposed accents)",
    "RTL: שלום عربى",
    "zero​width‍joiner and ﻿BOM",
    "<|endoftext|>",
    "text <|endoftext|> more text",
    "\x00\x01 control \x7f chars",
]


@pytest.fixture(scope="module")
def tok() -> ByteBPETokenizer:
    return ByteBPETokenizer.train(CORPUS, vocab_size=400)


def test_vocab_size_is_exact(tok):
    assert tok.vocab_size == 400
    assert len(tok.merges) == 400 - 256 - 1
    assert tok.special_tokens == {"<|endoftext|>": 399}


@pytest.mark.parametrize("text", EDGE_CASES)
def test_round_trip_edge_cases(tok, text):
    ids = tok.encode(text)
    assert all(isinstance(i, int) and 0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == text


def test_empty_string(tok):
    assert tok.encode("") == []
    assert tok.decode([]) == ""
    assert tok.encode("", allowed_special="all") == []


def test_pretokenizer_covers_input():
    t = ByteBPETokenizer([])
    rng = random.Random(0)
    alphabet = "ab _1\n\t.!'é日😀́‍<|>"
    for _ in range(500):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert "".join(t.pretokenize(s)) == s


def test_random_unicode_round_trip(tok):
    rng = random.Random(1)
    for _ in range(300):
        # random code points, excluding surrogates (not encodable as UTF-8)
        ranges = [(32, 126), (0x80, 0xD7FF), (0xE000, 0x10FFFF)]
        cps = [rng.randint(*rng.choice(ranges)) for _ in range(rng.randint(0, 30))]
        s = "".join(map(chr, cps))
        assert tok.decode(tok.encode(s)) == s


def test_byte_fallback_needs_no_unknown_token():
    t = ByteBPETokenizer([])  # no merges: pure bytes
    s = "Ω≈ç√∫"
    assert t.encode(s) == list(s.encode("utf-8"))
    assert t.decode(t.encode(s)) == s


def test_compresses_training_text(tok):
    text = "".join(CORPUS)
    assert len(tok.encode(text)) < 0.6 * len(text.encode("utf-8"))


def test_training_is_deterministic():
    a = ByteBPETokenizer.train(CORPUS, vocab_size=350)
    b = ByteBPETokenizer.train(list(CORPUS), vocab_size=350)
    assert a.merges == b.merges and a.fingerprint() == b.fingerprint()


def test_first_merge_is_most_frequent_pair_with_deterministic_tiebreak():
    # "ab" occurs 3x, "bc" 3x -> tie broken by smallest (left, right) ids: (a, b)
    t = ByteBPETokenizer.train(["abc abc abc"], vocab_size=258, special_tokens=())
    assert t.merges[0] == (ord("a"), ord("b"))
    assert t.token_bytes(256) == b"ab"


def test_merges_never_cross_pretoken_boundaries(tok):
    for i in range(256, 256 + len(tok.merges)):
        piece = tok.token_bytes(i).decode("utf-8", errors="ignore")
        if piece:
            assert len(tok.pretokenize(piece)) == 1, piece


def test_special_tokens_are_text_unless_allowed(tok):
    s = "a<|endoftext|>b"
    plain = tok.encode(s)
    assert tok.eot_id not in plain and tok.decode(plain) == s
    special = tok.encode(s, allowed_special="all")
    assert special.count(tok.eot_id) == 1
    assert tok.decode(special) == s
    assert tok.encode(s, allowed_special={"<|endoftext|>"}) == special
    with pytest.raises(ValueError, match="unknown special"):
        tok.encode(s, allowed_special={"<|nope|>"})


def test_multiple_special_tokens():
    t = ByteBPETokenizer.train(CORPUS, vocab_size=300, special_tokens=("<|endoftext|>", "<|pad|>"))
    assert t.special_tokens == {"<|endoftext|>": 298, "<|pad|>": 299}
    ids = t.encode("x<|pad|>y<|endoftext|>", allowed_special="all")
    assert 299 in ids and 298 in ids and t.decode(ids) == "x<|pad|>y<|endoftext|>"


def test_decode_rejects_out_of_range_ids(tok):
    with pytest.raises(ValueError, match="out of range"):
        tok.decode([tok.vocab_size])
    with pytest.raises(ValueError, match="out of range"):
        tok.decode([-1])


def test_decode_partial_utf8_is_replaced_not_raised(tok):
    ids = tok.encode("😀")
    first_byte = list("😀".encode())[:1]
    assert tok.decode(first_byte) == "�"
    assert tok.decode(ids) == "😀"


def test_save_load_round_trip(tmp_path, tok):
    path = tmp_path / "bpe.json"
    tok.save(path)
    loaded = ByteBPETokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens
    assert loaded.fingerprint() == tok.fingerprint()
    for text in EDGE_CASES:
        assert loaded.encode(text) == tok.encode(text)
    raw = json.loads(path.read_text())
    assert raw["type"] == "byte_bpe" and raw["vocab_size"] == 400
    assert raw["metadata"]["trained_on_documents"] == len(CORPUS)


def test_load_rejects_tampered_file(tmp_path, tok):
    raw = tok.to_dict()
    raw["merges"][0] = [raw["merges"][0][1], raw["merges"][0][0]]
    with pytest.raises(ValueError):
        ByteBPETokenizer.from_dict(raw)
    raw = tok.to_dict()
    raw["type"] = "char"
    with pytest.raises(ValueError):
        ByteBPETokenizer.from_dict(raw)


def test_invalid_construction():
    with pytest.raises(ValueError, match="smaller than"):
        ByteBPETokenizer.train(CORPUS, vocab_size=200)
    with pytest.raises(ValueError, match="undefined token"):
        ByteBPETokenizer([(300, 1)])
    with pytest.raises(ValueError, match="duplicate"):
        ByteBPETokenizer([(1, 2), (1, 2)])
    with pytest.raises(ValueError, match="unique"):
        ByteBPETokenizer([], special_tokens=("<a>", "<a>"))
    with pytest.raises(ValueError, match="supports only"):
        ByteBPETokenizer.train(["ab"], vocab_size=300, special_tokens=())


def test_satisfies_tokenizer_protocol(tok):
    t: Tokenizer = tok
    assert t.decode(t.encode("protocol")) == "protocol"
