"""
src/retrieve.py
===============
Transparent retrieval layer for the Document Navigator RAG system.

Loads the FAISS index produced by src/ingest.py and exposes similarity
search with cosine-similarity scoring, per-query JSONL tracing, and a
fixed-width CLI for interactive exploration.

Usage
-----
    python -m src.retrieve "What is precision at k?"
    python -m src.retrieve "escalation policy" --k 3 --no-trace

Importable entry point (used by generation and evaluation):
    from src.retrieve import run_query, Retriever
    trace = run_query("my question", k=5)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
INDEX_DIR: Path = PROJECT_ROOT / "db_faiss"
LOG_DIR: Path = PROJECT_ROOT / "logs"

DEFAULT_INDEX_NAME: str = "index"
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_K: int = 5
TRACE_FILE: Path = LOG_DIR / "retrieval_traces.jsonl"
TRACE_TEXT_LIMIT: int = 240  # chars kept per hit in the JSONL trace


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _configure_logging() -> logging.Logger:
    """Configure a console + file logger. Safe to call multiple times."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "retrieve.log"

    logger = logging.getLogger("retrieve")
    if logger.handlers:
        return logger  # already configured

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
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class RetrievalHit:
    rank: int
    distance: float       # L2² distance returned by FAISS
    similarity: float     # cosine similarity derived from L2² on unit vectors
    source: str
    page: int
    chunk_id: int
    chunk_index: int
    text: str
    citation: str         # e.g. "[report.pdf:3]"


@dataclass
class RetrievalTrace:
    timestamp: str        # UTC ISO-8601, seconds precision
    query: str
    k: int
    elapsed_ms: float
    embedding_model: str
    index_dir: str
    hits: List[RetrievalHit] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------
def _l2sq_to_cosine(distance: float) -> float:
    """
    Convert FAISS inner-product L2² distance to cosine similarity.

    For unit-normalised vectors: ||a-b||² = 2 - 2·cos(a,b)
    → cos = 1 - distance/2, clamped to [0, 1].
    """
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


# ---------------------------------------------------------------------------
# Retriever (lazy-loading)
# ---------------------------------------------------------------------------
class Retriever:
    """
    Loads the FAISS index once on first use and reuses the same instance
    for all subsequent queries.
    """

    def __init__(
        self,
        index_dir: Path = INDEX_DIR,
        index_name: str = DEFAULT_INDEX_NAME,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._index_dir = index_dir
        self._index_name = index_name
        self._embedding_model_name = embedding_model_name
        self._vectorstore: Optional[FAISS] = None

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_vectorstore(
        cls,
        vectorstore: FAISS,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> "Retriever":
        """Build a Retriever directly from an in-memory FAISS object (e.g. from uploads)."""
        instance = cls(embedding_model_name=embedding_model)
        instance._vectorstore = vectorstore
        instance._index_dir = Path("<in-memory>")
        return instance

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._vectorstore is not None:
            return

        index_faiss = self._index_dir / f"{self._index_name}.faiss"
        if not self._index_dir.exists() or not index_faiss.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self._index_dir!s}. "
                f"Run  python -m src.ingest  first to build it."
            )

        log.info("Loading embedding model: %s", self._embedding_model_name)
        embeddings = HuggingFaceEmbeddings(
            model_name=self._embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        log.info("Loading FAISS index from: %s", self._index_dir)
        self._vectorstore = FAISS.load_local(
            folder_path=str(self._index_dir),
            embeddings=embeddings,
            index_name=self._index_name,
            allow_dangerous_deserialization=True,
        )
        log.info("Index loaded successfully.")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = DEFAULT_K) -> List[RetrievalHit]:
        """Run similarity search and return scored, ranked hits."""
        self._ensure_loaded()
        assert self._vectorstore is not None  # mypy / type checkers

        raw = self._vectorstore.similarity_search_with_score(query, k=k)

        hits: List[RetrievalHit] = []
        for rank, (doc, distance) in enumerate(raw, start=1):
            meta = doc.metadata
            source = str(meta.get("source", "unknown"))
            page = int(meta.get("page", -1))
            hits.append(
                RetrievalHit(
                    rank=rank,
                    distance=float(distance),
                    similarity=_l2sq_to_cosine(float(distance)),
                    source=source,
                    page=page,
                    chunk_id=int(meta.get("chunk_id", -1)),
                    chunk_index=int(meta.get("chunk_index", -1)),
                    text=doc.page_content,
                    citation=f"[{source}:{page}]",
                )
            )
        return hits

    @property
    def embedding_model_name(self) -> str:
        return self._embedding_model_name

    @property
    def index_dir(self) -> Path:
        return self._index_dir


# ---------------------------------------------------------------------------
# JSONL tracing
# ---------------------------------------------------------------------------
def _trace_to_dict(trace: RetrievalTrace) -> dict:
    """Serialise a RetrievalTrace to a JSON-safe dict, truncating hit text."""
    d = dataclasses.asdict(trace)
    for hit in d["hits"]:
        text = hit["text"]
        if len(text) > TRACE_TEXT_LIMIT:
            hit["text"] = text[:TRACE_TEXT_LIMIT] + "…"
    return d


def _write_trace(trace: RetrievalTrace) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with TRACE_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_trace_to_dict(trace), ensure_ascii=False) + "\n")
    log.info("Trace appended to: %s", TRACE_FILE)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------
_PREVIEW_WIDTH: int = 72  # chars for text preview lines


def _pretty_print(trace: RetrievalTrace) -> None:
    print(
        f"\nQuery : {trace.query!r}\n"
        f"k     : {trace.k}   elapsed: {trace.elapsed_ms:.1f} ms\n"
        f"{'─' * 78}"
    )
    for hit in trace.hits:
        header = (
            f"#{hit.rank:<2}  sim={hit.similarity:.4f}  dist={hit.distance:.4f}  "
            f"{hit.citation}  chunk_id={hit.chunk_id}"
        )
        print(header)
        # Wrap text preview at _PREVIEW_WIDTH chars, indented by 4 spaces
        preview = hit.text.replace("\n", " ")
        while len(preview) > _PREVIEW_WIDTH:
            print(f"    {preview[:_PREVIEW_WIDTH]}")
            preview = preview[_PREVIEW_WIDTH:]
        if preview:
            print(f"    {preview}")
        print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_default_retriever: Optional[Retriever] = None


def run_query(
    query: str,
    k: int = DEFAULT_K,
    retriever: Optional[Retriever] = None,
    write_jsonl_trace: bool = True,
) -> RetrievalTrace:
    """
    Run a similarity search and return a fully-populated RetrievalTrace.

    Parameters
    ----------
    query:
        The natural-language query string.
    k:
        Number of results to return.
    retriever:
        An existing Retriever instance. When *None*, a module-level singleton
        is created on the first call and reused thereafter.
    write_jsonl_trace:
        When *True*, appends one JSON line to logs/retrieval_traces.jsonl.
    """
    global _default_retriever

    if retriever is None:
        if _default_retriever is None:
            _default_retriever = Retriever()
        retriever = _default_retriever

    t0 = time.perf_counter()
    hits = retriever.search(query, k=k)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    trace = RetrievalTrace(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        query=query,
        k=k,
        elapsed_ms=round(elapsed_ms, 2),
        embedding_model=retriever.embedding_model_name,
        index_dir=str(retriever.index_dir),
        hits=hits,
    )

    log.info(
        "Query %r → %d hit(s) in %.1f ms", query, len(hits), elapsed_ms
    )

    if write_jsonl_trace:
        _write_trace(trace)

    return trace


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Document Navigator — retrieval layer."
    )
    parser.add_argument("query", help="Natural-language query string.")
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of results to return (default: {DEFAULT_K}).",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=INDEX_DIR,
        help=f"FAISS index folder (default: {INDEX_DIR}).",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=DEFAULT_INDEX_NAME,
        help=f"Index filename stem (default: {DEFAULT_INDEX_NAME}).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Sentence-transformers model id (default: {DEFAULT_EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Do not append a JSONL entry to logs/retrieval_traces.jsonl.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        retriever = Retriever(
            index_dir=args.index_dir,
            index_name=args.index_name,
            embedding_model_name=args.embedding_model,
        )
        trace = run_query(
            query=args.query,
            k=args.k,
            retriever=retriever,
            write_jsonl_trace=not args.no_trace,
        )
        _pretty_print(trace)
        return 0
    except FileNotFoundError as exc:
        log.exception("%s", exc)
        return 1
    except Exception:  # noqa: BLE001
        log.exception("Retrieval failed with an unhandled exception.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
