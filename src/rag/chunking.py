"""
Text chunking strategies for the RAG pipeline.

Three strategies are available:

  "recursive"  (default) — splits on progressively finer natural-language
                boundaries (paragraphs → lines → sentences → words).
                Best general-purpose choice.

  "fixed"      — fixed-size character windows with overlap.
                Predictable, good for structured/code content.

  "sentence"   — groups complete sentences into chunks.
                Preserves semantic units; good for prose.

Chunk sizes throughout are expressed in *approximate tokens*
(1 token ≈ 4 characters) to match the CHUNK_SIZE / CHUNK_OVERLAP
values in config.py.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from src.rag.ingestion import Document
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


# ─── Data model ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """A piece of a document, ready for embedding."""

    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    @property
    def chunk_index(self) -> int:
        return self.metadata.get("chunk_index", 0)

    def __len__(self) -> int:
        return len(self.content)


# ─── Public API ──────────────────────────────────────────────────────────────

def chunk_document(
    document: Document,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    strategy: str = "recursive",
) -> List[Chunk]:
    """
    Split a Document into overlapping Chunks.

    Args:
        document:      Document to split.
        chunk_size:    Target chunk size in approximate tokens.
        chunk_overlap: Token overlap between adjacent chunks (must be < chunk_size).
        strategy:      "recursive" | "fixed" | "sentence"

    Returns:
        List of Chunk objects that inherit the document's metadata plus
        chunk_index and chunk_total fields.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )

    _strategies = {
        "recursive": _recursive_split,
        "fixed":     _fixed_split,
        "sentence":  _sentence_split,
    }
    if strategy not in _strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {list(_strategies)}"
        )

    texts = _strategies[strategy](document.content, chunk_size, chunk_overlap)
    texts = [t for t in texts if t.strip()]

    chunks: List[Chunk] = []
    for i, text in enumerate(texts):
        meta = {**document.metadata, "chunk_index": i, "chunk_total": len(texts)}
        chunks.append(Chunk(content=text, metadata=meta))

    return chunks


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    strategy: str = "recursive",
) -> List[Chunk]:
    """Chunk a list of documents into a flat list of Chunks."""
    result: List[Chunk] = []
    for doc in documents:
        result.extend(chunk_document(doc, chunk_size, chunk_overlap, strategy))
    return result


# ─── Token ↔ character conversion ────────────────────────────────────────────

_CHARS_PER_TOKEN = 4  # rough average for English text


def _to_chars(tokens: int) -> int:
    return tokens * _CHARS_PER_TOKEN


# ─── Strategy: fixed-size windows ────────────────────────────────────────────

def _fixed_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Slide a fixed-width character window over the text."""
    size = _to_chars(chunk_size)
    overlap = _to_chars(chunk_overlap)
    step = size - overlap

    chunks: List[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


# ─── Strategy: recursive character splitting ─────────────────────────────────

# Tried in order from coarsest to finest boundary.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Split on the coarsest natural boundary that exists in *text*, then recurse
    on any pieces that are still too large using the next finer separator.
    Falls back to fixed-size windows when no separator remains.
    """
    seps = separators if separators is not None else _SEPARATORS
    size = _to_chars(chunk_size)
    overlap = _to_chars(chunk_overlap)

    if len(text) <= size:
        return [text.strip()] if text.strip() else []

    # Find the coarsest separator that is present in this text.
    chosen_sep: Optional[str] = None
    remaining_seps: List[str] = []
    for i, sep in enumerate(seps):
        if sep in text:
            chosen_sep = sep
            remaining_seps = seps[i + 1 :]
            break

    if chosen_sep is None:
        # No natural boundary found — fall back to fixed windows.
        return _fixed_split(text, chunk_size, chunk_overlap)

    sep_len = len(chosen_sep)
    splits = text.split(chosen_sep)
    result: List[str] = []
    window: List[str] = []
    window_len = 0

    for split in splits:
        split_len = len(split)

        if split_len > size:
            # This single piece is already too large — flush and recurse deeper.
            if window:
                result.append(_join(window, chosen_sep))
                window, window_len = [], 0
            result.extend(
                _recursive_split(split, chunk_size, chunk_overlap, remaining_seps)
            )
            continue

        # How many chars would be added to the current window?
        added = split_len + (sep_len if window else 0)

        if window_len + added > size:
            # Flush current window as a chunk.
            result.append(_join(window, chosen_sep))
            # Rebuild overlap: keep trailing splits whose total ≤ overlap chars.
            keep: List[str] = []
            keep_len = 0
            for s in reversed(window):
                s_cost = len(s) + sep_len
                if keep_len + s_cost > overlap:
                    break
                keep.insert(0, s)
                keep_len += s_cost
            window = keep
            window_len = keep_len

        window.append(split)
        window_len += split_len + (sep_len if len(window) > 1 else 0)

    if window:
        result.append(_join(window, chosen_sep))

    return [c for c in result if c.strip()]


def _join(parts: List[str], sep: str) -> str:
    return sep.join(p for p in parts if p).strip()


# ─── Strategy: sentence grouping ─────────────────────────────────────────────

# Matches sentence-ending punctuation followed by whitespace.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _sentence_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Group complete sentences into chunks of approximately chunk_size tokens.
    Sentences that exceed chunk_size on their own are kept as-is.
    """
    size = _to_chars(chunk_size)
    overlap = _to_chars(chunk_overlap)

    sentences = _SENTENCE_RE.split(text.strip())
    result: List[str] = []
    window: List[str] = []
    window_len = 0

    for sentence in sentences:
        s_len = len(sentence)

        if window_len + s_len > size and window:
            result.append(" ".join(window).strip())
            # Overlap: keep trailing sentences whose total ≤ overlap chars.
            keep: List[str] = []
            keep_len = 0
            for s in reversed(window):
                if keep_len + len(s) > overlap:
                    break
                keep.insert(0, s)
                keep_len += len(s)
            window = keep
            window_len = keep_len

        window.append(sentence)
        window_len += s_len

    if window:
        result.append(" ".join(window).strip())

    return [c for c in result if c.strip()]
