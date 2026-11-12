import pytest

from mini_chatgpt.tokenizer import CharTokenizer

TEXT = "First Citizen:\nBefore we proceed any further, hear me speak.\n"


def test_round_trip():
    tok = CharTokenizer.from_text(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_vocab_is_sorted_unique_and_deterministic():
    a, b = CharTokenizer.from_text(TEXT), CharTokenizer.from_text(TEXT[::-1])
    assert a.chars == sorted(set(TEXT)) == b.chars
    assert a.vocab_size == len(set(TEXT))


def test_ids_are_in_range():
    tok = CharTokenizer.from_text(TEXT)
    assert all(0 <= i < tok.vocab_size for i in tok.encode(TEXT))


def test_empty_string():
    tok = CharTokenizer.from_text(TEXT)
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_unknown_character_raises():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="not in the vocabulary"):
        tok.encode("abz")


def test_out_of_range_id_raises():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="out of range"):
        tok.decode([0, 3])
    with pytest.raises(ValueError):
        tok.decode([-1])


def test_invalid_vocab_rejected():
    with pytest.raises(ValueError):
        CharTokenizer([])
    with pytest.raises(ValueError):
        CharTokenizer(["a", "a"])
    with pytest.raises(ValueError):
        CharTokenizer(["ab"])


def test_save_load(tmp_path):
    tok = CharTokenizer.from_text(TEXT + "é漢")
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.chars == tok.chars
    assert loaded.encode(TEXT) == tok.encode(TEXT)
