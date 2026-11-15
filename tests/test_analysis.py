from mini_chatgpt.analysis.svgplot import Chart, Series, ema, nice_ticks, render
from mini_chatgpt.analysis.text_stats import (
    longest_match,
    ngram_overlap,
    speaker_stats,
    speakers,
    vocabulary,
    word_stats,
)

CORPUS = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nROMEO:\nWhat light?\n"


def test_longest_match_brute_force_agreement():
    for query in ["xx hear me speak!", "zzz", "", "ROMEO:\nWhat lig", "any furthe proceed"]:
        m = longest_match(query, CORPUS)
        brute = max(
            (j - i for i in range(len(query)) for j in range(i, len(query) + 1)
             if query[i:j] in CORPUS),
            default=0,
        )  # fmt: skip
        assert m.length == brute
        assert m.text in CORPUS and query[m.start : m.start + m.length] == m.text


def test_word_and_speaker_stats():
    vocab = vocabulary(CORPUS)
    assert {"before", "citizen", "light"} <= vocab
    ws = word_stats("Before we proceed, hear thee xqzt", vocab)
    assert ws.n_words == 6 and abs(ws.valid_fraction - 4 / 6) < 1e-12
    assert ws.n_len4 == 5 and abs(ws.valid_fraction_len4 - 3 / 5) < 1e-12
    known = speakers(CORPUS)
    assert known == {"First Citizen", "ROMEO"}
    s = speaker_stats("ROMEO:\nhi\nJULIET:\nho\nROMEO:\n", known)
    assert s.n_labels == 3 and s.distinct == 2 and abs(s.known_fraction - 2 / 3) < 1e-12


def test_ngram_overlap():
    assert ngram_overlap("hear me", CORPUS, 4) == 1.0
    assert ngram_overlap("qqqqqq", CORPUS, 4) == 0.0
    assert ngram_overlap("ab", CORPUS, 4) == 0.0


def test_svg_render_and_helpers():
    assert nice_ticks(0, 1) == [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert all(abs(v - 2.0) < 1e-12 for v in ema([2.0, 2.0, 2.0]))
    svg = render(
        Chart(
            "Loss",
            "step",
            "loss",
            [
                Series("train", [0, 1, 2], [3, 2, 1]),
                Series("val", [0, 2], [3, 1.5], slot=1, markers=True),
            ],
        )
    )
    assert svg.startswith("<svg") and svg.count("<path") == 2 and "val — step 2" in svg


def test_entity_check_supports_and_conflicts():
    from mini_chatgpt.analysis.story_eval import entity_check

    facts = [("Tom", "basket"), ("Lily", "lantern")]
    text = ("Tom held his basket tight. Lily lifted the lantern high. Then Lily dropped the "
            "basket. Tom and Lily laughed. Max the fox came.")  # fmt: skip
    r = entity_check(text, ["Tom", "Lily"], facts)
    assert r["fact_supports"] == 2 and r["fact_conflicts"] == 1
    assert r["conflict_examples"] == ["Then Lily dropped the basket."]
    assert r["names_kept_fraction"] == 1.0 and r["new_capitalized_names"] == ["Max"]
    assert entity_check("They went home.", ["Tom"], [("Tom", "hat")])["names_kept_fraction"] == 0.0
