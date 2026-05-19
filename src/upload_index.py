"""
src/upload_index.py
===================
In-memory FAISS index builder for user-uploaded PDFs.

Mirrors the ingestion configuration of src/ingest.py (same chunk size,
overlap, embedding model, and metadata schema) but operates on raw bytes
rather than files on disk. The resulting FAISS object lives entirely in
memory; nothing is written to db_faiss/.

Importable only — no CLI, no __main__ block.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingest import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    chunk_pages,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
LOG_DIR: Path = PROJECT_ROOT / "logs"

MAX_UPLOAD_BYTES: int = 20 * 1024 * 1024  # 20 MB per file
MAX_PAGES_PER_FILE: int = 200
MAX_FILES_PER_UPLOAD: int = 10


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "upload.log"

    logger = logging.getLogger("upload")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger


log = _configure_logging()


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class UploadTooLargeError(Exception):
    def __init__(self, filename: str, size_bytes: int) -> None:
        mb = size_bytes / (1024 * 1024)
        limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        self._msg = (
            f"File '{filename}' is {mb:.1f} MB; maximum is {limit_mb:.0f} MB. "
            f"Split it into smaller files."
        )

    def __str__(self) -> str:
        return self._msg


class UploadTooManyPagesError(Exception):
    def __init__(self, filename: str, n_pages: int) -> None:
        self._msg = (
            f"File '{filename}' has {n_pages} pages; maximum is {MAX_PAGES_PER_FILE}. "
            f"Split it into smaller files."
        )

    def __str__(self) -> str:
        return self._msg


class UploadTooManyFilesError(Exception):
    def __init__(self, n_files: int) -> None:
        self._msg = (
            f"Upload contains {n_files} files; maximum is {MAX_FILES_PER_UPLOAD}. "
            f"Upload fewer files at a time."
        )

    def __str__(self) -> str:
        return self._msg


class UploadEmptyError(Exception):
    def __init__(self, filename: str) -> None:
        self._msg = (
            f"File '{filename}' produced no text after parsing. "
            f"Image-only PDFs (scans without OCR) are not supported."
        )

    def __str__(self) -> str:
        return self._msg


# ---------------------------------------------------------------------------
# UploadedDoc summary dataclass
# ---------------------------------------------------------------------------
@dataclass
class UploadedDoc:
    filename: str
    size_bytes: int
    n_pages: int
    n_chunks: int


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def build_index_from_uploads(
    files: List[Tuple[str, bytes]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Tuple[FAISS, List[UploadedDoc]]:
    """
    Build an in-memory FAISS index from uploaded PDF bytes.

    Parameters
    ----------
    files:
        List of (filename, raw_bytes) tuples. Streamlit UploadedFile objects
        can be converted with [(uf.name, uf.getvalue()) for uf in uploaded].
    chunk_size:
        Character-based chunk size (matches src/ingest.py default).
    chunk_overlap:
        Overlap between chunks (matches src/ingest.py default).
    embedding_model_name:
        HuggingFace sentence-transformers model (matches src/ingest.py default).

    Returns
    -------
    (FAISS, List[UploadedDoc])

    Raises
    ------
    UploadTooManyFilesError, UploadTooLargeError, UploadTooManyPagesError,
    UploadEmptyError
    """
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise UploadTooManyFilesError(len(files))

    log.info(
        "Building in-memory index from %d uploaded file(s) "
        "(chunk_size=%d, overlap=%d, model=%s)",
        len(files), chunk_size, chunk_overlap, embedding_model_name,
    )

    all_chunks = []
    docs_summary: List[UploadedDoc] = []
    global_chunk_index = 0

    for filename, raw_bytes in files:
        size_bytes = len(raw_bytes)

        if size_bytes > MAX_UPLOAD_BYTES:
            raise UploadTooLargeError(filename, size_bytes)

        # Write to tempfile so PyPDFLoader can open it, then clean up.
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(raw_bytes)
                tmp_path = Path(tmp.name)
            loader = PyPDFLoader(str(tmp_path))
            pages = loader.load()
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()

        n_pages = len(pages)
        if n_pages > MAX_PAGES_PER_FILE:
            raise UploadTooManyPagesError(filename, n_pages)

        # Normalize metadata — use the uploaded filename, not the temp path.
        for page in pages:
            zero_based = int(page.metadata.get("page", 0))
            page.metadata["page"] = zero_based + 1
            page.metadata["source"] = filename

        # Chunk using the same function as src/ingest.py.
        file_chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if not file_chunks:
            raise UploadEmptyError(filename)

        # Reassign chunk_index so it is monotonically increasing across
        # all files in the upload batch (chunk_pages restarts from 0 per call).
        for chunk in file_chunks:
            chunk.metadata["chunk_index"] = global_chunk_index
            global_chunk_index += 1

        all_chunks.extend(file_chunks)
        docs_summary.append(
            UploadedDoc(
                filename=filename,
                size_bytes=size_bytes,
                n_pages=n_pages,
                n_chunks=len(file_chunks),
            )
        )
        log.info(
            "Processed '%s': %d page(s), %d chunk(s)",
            filename, n_pages, len(file_chunks),
        )

    log.info(
        "Embedding %d chunk(s) from %d file(s)…", len(all_chunks), len(files)
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(documents=all_chunks, embedding=embeddings)
    log.info("In-memory index built: %d total chunk(s).", len(all_chunks))

    return vectorstore, docs_summary
