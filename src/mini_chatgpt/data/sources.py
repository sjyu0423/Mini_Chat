"""Dataset sources with documented provenance and usage terms.

Tiny Shakespeare
----------------
* Content: ~1.1 MB of plays by William Shakespeare, concatenated, as assembled
  by Andrej Karpathy for the char-rnn project (2015).
* Source URL: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
* Usage terms: Shakespeare's works are in the public domain. The char-rnn
  repository that hosts this file is MIT licensed. Suitable for research and
  education; no personal data.
* Integrity: the file is pinned by SHA-256 below. A mismatch aborts loading
  rather than silently training on different data.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from mini_chatgpt.training.checkpoint import atomic_write_bytes

logger = logging.getLogger("mini_chatgpt")


@dataclass(frozen=True)
class TextSource:
    name: str
    url: str
    filename: str
    sha256: str
    num_bytes: int
    license: str
    description: str


TINY_SHAKESPEARE = TextSource(
    name="tiny_shakespeare",
    url="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    filename="input.txt",
    sha256="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
    num_bytes=1_115_394,
    license="Public domain text (Shakespeare); hosted in the MIT-licensed karpathy/char-rnn repo",
    description="~1.1 MB of concatenated Shakespeare plays (char-rnn tinyshakespeare)",
)

SOURCES: dict[str, TextSource] = {TINY_SHAKESPEARE.name: TINY_SHAKESPEARE}

# Refuse anything much bigger than expected: M0 never starts large downloads.
_MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024


class DataIntegrityError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_source(name: str) -> TextSource:
    try:
        return SOURCES[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(SOURCES)}") from None


def raw_path(source: TextSource, data_dir: str | Path) -> Path:
    return Path(data_dir) / "raw" / source.name / source.filename


def download(source: TextSource, dest: Path, timeout: float = 60.0) -> None:
    logger.info("downloading %s from %s", source.name, source.url)
    with urllib.request.urlopen(source.url, timeout=timeout) as resp:
        data: bytes = resp.read(_MAX_DOWNLOAD_BYTES + 1)
    if len(data) > _MAX_DOWNLOAD_BYTES:
        raise DataIntegrityError(f"download exceeds {_MAX_DOWNLOAD_BYTES} bytes; aborting")
    digest = sha256_bytes(data)
    if digest != source.sha256:
        raise DataIntegrityError(
            f"SHA-256 mismatch for {source.name}: expected {source.sha256}, got {digest}"
        )
    atomic_write_bytes(dest, data)


def load_text(name: str, data_dir: str | Path, *, allow_download: bool = True) -> str:
    """Load a dataset's raw text, downloading it once if allowed and missing."""
    source = get_source(name)
    path = raw_path(source, data_dir)
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"{path} not found and downloads are disabled")
        download(source, path)
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if digest != source.sha256:
        raise DataIntegrityError(
            f"{path} has SHA-256 {digest}, expected {source.sha256}. "
            "Delete the file to re-download, or investigate the change."
        )
    return data.decode("utf-8")
