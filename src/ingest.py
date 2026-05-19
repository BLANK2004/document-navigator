"""
src/ingest.py
=============
PDF ingestion pipeline for the Document Navigator RAG system.

Pipeline
--------
1. Discover PDFs under ./documents/ (non-recursive, deterministic order).
2. Load each PDF page-by-page with PyPDFLoader.
3. Chunk pages with RecursiveCharacterTextSplitter (chunk_size / overlap).
4. Embed chunks locally with sentence-transformers/all-MiniLM-L6-v2.
5. Persist a FAISS index + docstore to ./db_faiss/.

Each stored chunk carries metadata:
    {
        "source":      "<filename.pdf>",
        "page":        <1-based page number>,
        "chunk_id":    <0-based index within (source, page)>,
        "chunk_index": <0-based global chunk index>,
    }

Run
---
    python -m src.ingest                          # use defaults
    python -m src.ingest --rebuild                # delete existing index first
    python -m src.ingest --chunk-size 600 \
                         --chunk-overlap 80       # override defaults
    python -m src.ingest --smoke-test             # reload index and assert it works
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR: Path = PROJECT_ROOT / "documents"
INDEX_DIR: Path = PROJECT_ROOT / "db_faiss"
LOG_DIR: Path = PROJECT_ROOT / "logs"

DEFAULT_CHUNK_SIZE: int = 800            # characters (not tokens)
DEFAULT_CHUNK_OVERLAP: int = 120         # characters
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_NAME: str = "index"

# Recursive splitter separators, ordered from coarsest to finest.
SPLIT_SEPARATORS: Tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _configure_logging() -> logging.Logger:
    """Configure a console + file logger. Safe to call multiple times."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "ingest.log"

    logger = logging.getLogger("ingest")
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
# Stage 1: discover PDFs
# ---------------------------------------------------------------------------
def discover_pdfs(documents_dir: Path) -> List[Path]:
    """Return a sorted list of PDF files in ``documents_dir``."""
    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {documents_dir}. "
            f"Create it and place your PDFs inside before running ingestion."
        )
    pdfs = sorted(p for p in documents_dir.glob("*.pdf") if p.is_file())
    return pdfs


# ---------------------------------------------------------------------------
# Stage 2: load PDFs
# ---------------------------------------------------------------------------
def load_pdf(pdf_path: Path) -> List[Document]:
    """
    Load a single PDF as one LangChain ``Document`` per page.

    Normalizes metadata so every downstream consumer sees:
        - ``source``: the bare filename (not the absolute path)
        - ``page``:   1-based page number (PyPDFLoader uses 0-based)
    """
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    for page in pages:
        zero_based_page = int(page.metadata.get("page", 0))
        page.metadata["page"] = zero_based_page + 1
        page.metadata["source"] = pdf_path.name
    return pages


# ---------------------------------------------------------------------------
# Stage 3: chunk pages
# ---------------------------------------------------------------------------
def chunk_pages(
    pages: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Recursively split pages into overlapping character-based chunks
    and stamp each chunk with rich metadata.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size})."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(SPLIT_SEPARATORS),
        length_function=len,
        add_start_index=True,  # records char offset inside the page
    )

    raw_chunks = splitter.split_documents(pages)

    # Per-(source, page) counters give a stable, human-readable chunk_id.
    per_page_counters: Dict[Tuple[str, int], int] = {}
    chunks: List[Document] = []
    global_index = 0

    for chunk in raw_chunks:
        text = chunk.page_content.strip()
        if not text:
            continue  # drop whitespace-only chunks

        source = chunk.metadata.get("source", "unknown")
        page = int(chunk.metadata.get("page", -1))
        key = (source, page)
        per_page_counters[key] = per_page_counters.get(key, -1) + 1

        chunk.page_content = text
        chunk.metadata.update(
            {
                "source": source,
                "page": page,
                "chunk_id": per_page_counters[key],
                "chunk_index": global_index,
            }
        )
        chunks.append(chunk)
        global_index += 1

    return chunks


# ---------------------------------------------------------------------------
# Stage 4: embed + build FAISS index
# ---------------------------------------------------------------------------
def build_index(chunks: List[Document], embedding_model_name: str) -> FAISS:
    """Embed ``chunks`` locally and return an in-memory FAISS index."""
    if not chunks:
        raise ValueError("Cannot build an index from zero chunks.")

    log.info("Loading embedding model: %s", embedding_model_name)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # enables cosine via inner product
    )

    log.info("Embedding %d chunks (first run downloads the model)…", len(chunks))
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore


# ---------------------------------------------------------------------------
# Stage 5: persist index
# ---------------------------------------------------------------------------
def persist_index(vectorstore: FAISS, index_dir: Path, index_name: str) -> None:
    """Save the FAISS index and its docstore to ``index_dir``."""
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(folder_path=str(index_dir), index_name=index_name)
    log.info("FAISS index persisted to: %s (name: %s)", index_dir, index_name)


# ---------------------------------------------------------------------------
# Optional smoke test
# ---------------------------------------------------------------------------
def smoke_test_index(
    index_dir: Path,
    index_name: str,
    embedding_model_name: str,
    query: str = "What is precision at k?",
    k: int = 3,
) -> None:
    """
    Reload the persisted index and run a single similarity search.
    Useful as a one-shot sanity check after ingestion.
    """
    log.info("Smoke test: reloading FAISS index from %s", index_dir)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        folder_path=str(index_dir),
        embeddings=embeddings,
        index_name=index_name,
        allow_dangerous_deserialization=True,  # required by langchain-community FAISS
    )
    results = vectorstore.similarity_search_with_score(query, k=k)
    log.info("Smoke test query: %r (top-%d)", query, k)
    for rank, (doc, score) in enumerate(results, start=1):
        log.info(
            "  #%d  score=%.4f  [%s:p%s#chunk%s]",
            rank,
            score,
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("chunk_id"),
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_ingestion(
    documents_dir: Path = DOCUMENTS_DIR,
    index_dir: Path = INDEX_DIR,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    index_name: str = DEFAULT_INDEX_NAME,
    rebuild: bool = False,
) -> Dict[str, object]:
    """Run the full ingestion pipeline and return a summary dict."""
    started = time.perf_counter()

    if rebuild and index_dir.exists():
        log.info("--rebuild set: deleting existing index at %s", index_dir)
        shutil.rmtree(index_dir)

    pdfs = discover_pdfs(documents_dir)
    if not pdfs:
        log.warning("No PDFs found in %s. Nothing to ingest.", documents_dir)
        return {"pdfs": 0, "pages": 0, "chunks": 0}
    log.info("Discovered %d PDF(s) in %s", len(pdfs), documents_dir)

    all_pages: List[Document] = []
    for pdf in tqdm(pdfs, desc="Loading PDFs", unit="pdf"):
        try:
            pages = load_pdf(pdf)
            all_pages.extend(pages)
            log.info("Loaded %-40s -> %d page(s)", pdf.name, len(pages))
        except Exception:  # noqa: BLE001 - we want to keep going on bad PDFs
            log.exception("Failed to load %s — skipping.", pdf.name)

    if not all_pages:
        log.error("Every PDF failed to load. Aborting.")
        return {"pdfs": len(pdfs), "pages": 0, "chunks": 0}

    log.info(
        "Chunking %d page(s) with chunk_size=%d, chunk_overlap=%d",
        len(all_pages),
        chunk_size,
        chunk_overlap,
    )
    chunks = chunk_pages(
        all_pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    log.info("Produced %d chunk(s) from %d page(s)", len(chunks), len(all_pages))

    vectorstore = build_index(chunks, embedding_model_name=embedding_model)
    persist_index(vectorstore, index_dir=index_dir, index_name=index_name)

    elapsed = time.perf_counter() - started
    summary: Dict[str, object] = {
        "pdfs": len(pdfs),
        "pages": len(all_pages),
        "chunks": len(chunks),
        "elapsed_seconds": round(elapsed, 2),
        "index_dir": str(index_dir),
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    log.info("Ingestion complete in %.2fs", elapsed)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Document Navigator — PDF ingestion pipeline."
    )
    parser.add_argument("--documents-dir", type=Path, default=DOCUMENTS_DIR,
                        help=f"Folder containing input PDFs (default: {DOCUMENTS_DIR})")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR,
                        help=f"Folder where the FAISS index is saved (default: {INDEX_DIR})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Recursive splitter chunk size in chars (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help=f"Chunk overlap in chars (default: {DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                        help=f"HuggingFace sentence-transformers model id "
                             f"(default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--index-name", type=str, default=DEFAULT_INDEX_NAME,
                        help=f"FAISS index filename stem (default: {DEFAULT_INDEX_NAME})")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete the existing index folder before ingestion.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="After ingestion, reload the index and run a sample query.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        summary = run_ingestion(
            documents_dir=args.documents_dir,
            index_dir=args.index_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model,
            index_name=args.index_name,
            rebuild=args.rebuild,
        )
        log.info("Summary: %s", summary)

        if args.smoke_test and summary.get("chunks"):
            smoke_test_index(
                index_dir=args.index_dir,
                index_name=args.index_name,
                embedding_model_name=args.embedding_model,
            )
        return 0
    except Exception:  # noqa: BLE001
        log.exception("Ingestion failed with an unhandled exception.")
        return 1


if __name__ == "__main__":
    sys.exit(main())