"""
Document ingestion: load files from disk into Document objects.

Supported formats: .txt, .md, .py, .json, .csv, .pdf (requires pypdf)
Any other extension is attempted as UTF-8 text.
"""

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Document:
    """A loaded document with its text content and source metadata."""

    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.metadata.get("source", "unknown")

    def __len__(self) -> int:
        return len(self.content)


# ─── Public API ──────────────────────────────────────────────────────────────

def load_document(path: str) -> Document:
    """
    Load a single document from disk, dispatching on file extension.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        A Document containing the file's text and source metadata.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file cannot be read as text.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {path}")

    ext = p.suffix.lower()
    loaders = {
        ".txt":  _load_text,
        ".md":   _load_text,
        ".py":   _load_text,
        ".js":   _load_text,
        ".ts":   _load_text,
        ".yaml": _load_text,
        ".yml":  _load_text,
        ".toml": _load_text,
        ".json": _load_json,
        ".csv":  _load_csv,
        ".pdf":  _load_pdf,
    }
    loader = loaders.get(ext, _load_text)
    return loader(p)


def load_directory(
    directory: str,
    extensions: Optional[List[str]] = None,
    recursive: bool = False,
) -> List[Document]:
    """
    Load all documents from a directory.

    Args:
        directory: Path to the directory.
        extensions: Whitelist of extensions to include (e.g. [".txt", ".md"]).
                    Defaults to common text formats.
        recursive: If True, walk subdirectories as well.

    Returns:
        List of successfully loaded Documents. Files that fail to load are
        skipped with a warning rather than raising.
    """
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    _default_extensions = {".txt", ".md", ".py", ".json", ".csv"}
    allowed = {e.lower() for e in extensions} if extensions else _default_extensions

    pattern = "**/*" if recursive else "*"
    docs: List[Document] = []
    for p in sorted(d.glob(pattern)):
        if not p.is_file():
            continue
        if p.suffix.lower() not in allowed:
            continue
        try:
            docs.append(load_document(str(p)))
        except Exception as exc:
            print(f"[ingestion] Warning: skipping {p} — {exc}")

    return docs


# ─── Format-specific loaders ─────────────────────────────────────────────────

def _load_text(path: Path) -> Document:
    content = path.read_text(encoding="utf-8")
    return Document(
        content=content,
        metadata={"source": str(path), "type": "text", "extension": path.suffix},
    )


def _load_json(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        content = json.dumps(data, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        # Fallback: treat as plain text if JSON is malformed
        content = raw
    return Document(
        content=content,
        metadata={"source": str(path), "type": "json"},
    )


def _load_csv(path: Path) -> Document:
    rows: List[str] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            rows.append(", ".join(reader.fieldnames))
        for row in reader:
            rows.append(", ".join(f"{k}: {v}" for k, v in row.items()))
    return Document(
        content="\n".join(rows),
        metadata={"source": str(path), "type": "csv"},
    )


def _load_pdf(path: Path) -> Document:
    try:
        import pypdf  # type: ignore
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF support. Install with: pip install pypdf"
        )

    reader = pypdf.PdfReader(str(path))
    pages: List[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text}")

    return Document(
        content="\n\n".join(pages),
        metadata={"source": str(path), "type": "pdf", "pages": len(reader.pages)},
    )
