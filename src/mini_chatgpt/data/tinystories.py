"""TinyStories V2 (GPT-4-only): bounded, verifiable acquisition and preprocessing.

Source: Hugging Face dataset ``roneneldan/TinyStories`` (CDLA-Sharing-1.0), pinned
to a commit. The training file is 2.23 GB, so only ``n_ranges`` byte ranges are
fetched with HTTP ``Range`` requests. The validation file (22.5 MB) is fetched
whole and checked against its published LFS SHA-256.

Range selection is reproducible and non-overlapping by construction: the file
is cut into ``n_ranges`` equal strata and one range of ``range_bytes`` is placed
at a seeded random offset inside each stratum. Every response must be HTTP 206
with ``Content-Range: bytes start-end/total`` exactly as requested and the
expected total file size.

Stories in the file are separated by ``<|endoftext|>``. A range almost always
starts and ends mid-story, so the bytes before the first delimiter and after the
last one are discarded; only complete stories are kept. Splitting happens on
raw bytes (the delimiter is ASCII), so multi-byte UTF-8 characters can never be
cut, and stories that fail strict UTF-8 decoding are dropped and counted.
"""

from __future__ import annotations

import hashlib
import itertools
import random
import re
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

REPO = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"  # dataset repo commit (2024-08-12)
TRAIN_FILE = "TinyStoriesV2-GPT4-train.txt"
VALID_FILE = "TinyStoriesV2-GPT4-valid.txt"
TRAIN_SIZE = 2_227_753_162
VALID_SIZE = 22_502_601
TRAIN_SHA256 = "6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443"  # LFS oid
VALID_SHA256 = "6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d"  # LFS oid
LICENSE = "CDLA-Sharing-1.0"
DELIMITER = b"<|endoftext|>"
USER_AGENT = "mini-chatgpt-research/0.1 (bounded range fetch)"
CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def file_url(name: str, revision: str = REVISION) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{name}"


# --------------------------------------------------------------------------- #
# Range planning and verified fetching
# --------------------------------------------------------------------------- #
def plan_ranges(total: int, n_ranges: int, range_bytes: int, seed: int) -> list[tuple[int, int]]:
    """``n_ranges`` inclusive ``(start, end)`` byte ranges, one per equal stratum."""
    stratum = total // n_ranges
    if range_bytes > stratum:
        raise ValueError(f"range_bytes {range_bytes} exceeds stratum size {stratum}")
    rng = random.Random(seed)
    out = []
    for i in range(n_ranges):
        lo = i * stratum
        start = lo + rng.randrange(0, stratum - range_bytes + 1)
        out.append((start, start + range_bytes - 1))
    for (_, e0), (s1, _) in itertools.pairwise(out):
        if s1 <= e0:
            raise AssertionError("planned ranges overlap")  # impossible by construction
    return out


Opener = Callable[[urllib.request.Request], Any]


def fetch_range(
    url: str,
    start: int,
    end: int,
    expected_total: int,
    *,
    opener: Opener = lambda req: urllib.request.urlopen(req, timeout=120),
    retries: int = 4,
) -> tuple[bytes, dict[str, Any]]:
    """GET ``bytes=start-end`` and verify status, Content-Range, total, and length."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Range": f"bytes={start}-{end}"})  # fmt: skip
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with opener(req) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                content_range = resp.headers.get("Content-Range", "")
                etag = resp.headers.get("ETag")
                body = resp.read()
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_exc = exc
            time.sleep(5.0 * 2**attempt)
    else:
        raise RuntimeError(f"range {start}-{end} failed after {retries} attempts") from last_exc
    if status != 206:
        raise RuntimeError(f"expected HTTP 206 for range {start}-{end}, got {status}")
    m = CONTENT_RANGE.match(content_range.strip())
    if not m or (int(m[1]), int(m[2])) != (start, end):
        raise RuntimeError(f"Content-Range {content_range!r} does not match {start}-{end}")
    if int(m[3]) != expected_total:
        raise RuntimeError(f"file size {m[3]} != expected {expected_total} (upstream changed?)")
    if len(body) != end - start + 1:
        raise RuntimeError(f"got {len(body)} bytes for range {start}-{end}")
    info = {
        "start": start,
        "end": end,
        "status": status,
        "content_range": content_range,
        "etag": etag,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    return body, info


# --------------------------------------------------------------------------- #
# Story recovery
# --------------------------------------------------------------------------- #
@dataclass
class ParseStats:
    stories: int = 0
    head_bytes_dropped: int = 0
    tail_bytes_dropped: int = 0
    empty_dropped: int = 0
    invalid_utf8_dropped: int = 0


def split_stories(blob: bytes, *, partial_edges: bool, stats: ParseStats) -> list[str]:
    """Split on the delimiter; if ``partial_edges``, drop the first and last pieces."""
    pieces = blob.split(DELIMITER)
    if partial_edges:
        if len(pieces) < 3:
            stats.head_bytes_dropped += len(blob)
            return []
        stats.head_bytes_dropped += len(pieces[0])
        stats.tail_bytes_dropped += len(pieces[-1])
        pieces = pieces[1:-1]
    out = []
    for p in pieces:
        try:
            text = p.decode("utf-8", errors="strict").strip()
        except UnicodeDecodeError:
            stats.invalid_utf8_dropped += 1
            continue
        if not text:
            stats.empty_dropped += 1
            continue
        out.append(text)
    stats.stories += len(out)
    return out


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def list_digest(texts: Sequence[str]) -> str:
    """Order-sensitive SHA-256 over the SHA-256 of every text."""
    h = hashlib.sha256()
    for t in texts:
        h.update(bytes.fromhex(text_sha256(t)))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Near-duplicate detection (MinHash + LSH)
# --------------------------------------------------------------------------- #
P31 = np.uint64((1 << 31) - 1)


@dataclass(frozen=True)
class MinHashConfig:
    shingle: int = 5  # word n-grams
    num_perm: int = 128
    bands: int = 16  # 16 x 8 rows: candidate probability 0.5 at Jaccard ~0.71
    threshold: float = 0.8  # estimated Jaccard needed to call a pair a near-duplicate
    seed: int = 0


def minhash_signatures(texts: Sequence[str], cfg: MinHashConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    a = rng.integers(1, int(P31) - 1, cfg.num_perm).astype(np.uint64)[:, None]
    b = rng.integers(0, int(P31) - 1, cfg.num_perm).astype(np.uint64)[:, None]
    sigs = np.empty((len(texts), cfg.num_perm), dtype=np.uint32)
    n = cfg.shingle
    for i, t in enumerate(texts):
        w = t.lower().split()
        grams = {" ".join(w[j : j + n]) for j in range(max(1, len(w) - n + 1))}
        x = np.fromiter((zlib.crc32(g.encode()) for g in grams), np.uint64, len(grams)) % P31
        sigs[i] = ((a * x[None, :] + b) % P31).min(axis=1)
    return sigs


def near_duplicate_pairs(sigs: np.ndarray, cfg: MinHashConfig) -> list[tuple[int, int, float]]:
    """Pairs ``(i, j, est_jaccard)`` with i < j and estimated Jaccard >= threshold."""
    rows = cfg.num_perm // cfg.bands
    rng = np.random.default_rng(cfg.seed + 1)
    mult = rng.integers(1, 2**63 - 1, rows, dtype=np.int64).astype(np.uint64) | np.uint64(1)
    found: dict[tuple[int, int], float] = {}
    for band in range(cfg.bands):
        block = sigs[:, band * rows : (band + 1) * rows].astype(np.uint64)
        keys = (block * mult).sum(axis=1)  # uint64 wraparound: a 64-bit band hash
        order = np.argsort(keys, kind="stable")
        sk = keys[order]
        cuts = np.flatnonzero(np.diff(sk)) + 1
        for group in np.split(order, cuts):
            if len(group) < 2:
                continue
            members = np.sort(group)
            # Large buckets: compare every member with the first few only (bounded cost).
            anchors = members if len(members) <= 32 else members[:8]
            for x in anchors:
                for y in members:
                    if y <= x or (int(x), int(y)) in found:
                        continue
                    est = float(np.mean(sigs[x] == sigs[y]))
                    if est >= cfg.threshold:
                        found[(int(x), int(y))] = est
    return sorted((i, j, e) for (i, j), e in found.items())


@dataclass
class DedupReport:
    input_stories: int = 0
    exact_duplicates_removed: int = 0
    near_duplicates_removed: int = 0
    near_duplicate_pairs: int = 0
    removed_for_matching_validation_exact: int = 0
    removed_for_matching_validation_near: int = 0
    validation_internal_exact_duplicates: int = 0
    validation_internal_near_pairs: int = 0
    kept: int = 0
    examples: list[list[str]] = field(default_factory=list)


def deduplicate(
    train: Sequence[str], val: Sequence[str], cfg: MinHashConfig
) -> tuple[list[str], DedupReport]:
    """Remove exact and near duplicates from ``train``, including any that match a
    validation story. Validation is left untouched (it is the official held-out set);
    its internal duplicates are only counted."""
    rep = DedupReport(input_stories=len(train))
    val_hashes = [text_sha256(v) for v in val]
    rep.validation_internal_exact_duplicates = len(val_hashes) - len(set(val_hashes))
    val_set = set(val_hashes)
    seen: set[str] = set()
    exact_kept: list[str] = []
    for t in train:
        h = text_sha256(t)
        if h in val_set:
            rep.removed_for_matching_validation_exact += 1
        elif h in seen:
            rep.exact_duplicates_removed += 1
        else:
            seen.add(h)
            exact_kept.append(t)
    # Near duplicates over validation + train; index < len(val) means validation.
    allt = list(val) + exact_kept
    sigs = minhash_signatures(allt, cfg)
    pairs = near_duplicate_pairs(sigs, cfg)
    nv = len(val)
    parent = list(range(len(allt)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, _ in pairs:
        if i < nv and j < nv:
            rep.validation_internal_near_pairs += 1
            continue
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)  # root = earliest index (validation first)
    rep.near_duplicate_pairs = sum(1 for i, j, _ in pairs if not (i < nv and j < nv))
    kept: list[str] = []
    for k in range(nv, len(allt)):
        root = find(k)
        if root == k:
            kept.append(allt[k])
        elif root < nv:
            rep.removed_for_matching_validation_near += 1
        else:
            rep.near_duplicates_removed += 1
    for i, j, _ in pairs[:3]:
        rep.examples.append([allt[i][:160], allt[j][:160]])
    rep.kept = len(kept)
    return kept, rep


def holdout_split(texts: Sequence[str], percent: float) -> tuple[list[str], list[str]]:
    """Deterministic hash split: a story is held out iff sha256 % 10_000 < percent * 100."""
    cut = round(percent * 100)
    keep: list[str] = []
    hold: list[str] = []
    for t in texts:
        (hold if int(text_sha256(t)[:8], 16) % 10_000 < cut else keep).append(t)
    return keep, hold


def report_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)
