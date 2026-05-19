"""
app.py — Document Navigator Streamlit demo.

Pure UI layer: imports generate_answer and Retriever from the pipeline;
no pipeline logic lives here.

Run: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.generate import GenerationResult, generate_answer
from src.retrieve import Retriever
_MODE_INDEXED = "Indexed corpus (evaluated)"
_MODE_UPLOAD = "Upload your own PDFs"

from src.upload_index import (
    UploadEmptyError,
    UploadTooLargeError,
    UploadTooManyFilesError,
    UploadTooManyPagesError,
    build_index_from_uploads,
)

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Document Navigator",
    page_icon="📄",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached retriever — loads the FAISS index exactly once across all reruns
# ---------------------------------------------------------------------------
@st.cache_resource
def get_retriever() -> Retriever:
    r = Retriever()
    r._ensure_loaded()
    return r


# Try loading the persistent indexed retriever — may fail if not yet built.
_indexed_retriever: Optional[Retriever] = None
_indexed_error: Optional[str] = None
try:
    _indexed_retriever = get_retriever()
except FileNotFoundError:
    _indexed_error = (
        "FAISS index not found. Run python -m src.ingest first, then refresh this page."
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    mode = st.radio(
        "Corpus mode",
        [_MODE_INDEXED, _MODE_UPLOAD],
        help=(
            "The indexed corpus is the 10-PDF evaluated configuration. "
            "Upload mode lets you query your own documents in this session only — "
            "uploads are not saved."
        ),
    )

    st.divider()

    k = st.slider("Top-k chunks", 1, 10, value=5)
    model = st.selectbox("Ollama model", ["qwen2.5:7b", "llama3.2:3b"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, value=0.0, step=0.1)
    strict = st.toggle(
        "Strict evidence mode",
        value=False,
        help="Raises evidence thresholds to 0.40/0.55",
    )

    st.divider()

    if mode == _MODE_INDEXED:
        if _indexed_retriever is not None:
            n = _indexed_retriever._vectorstore.index.ntotal
            st.caption(f"Index: {n} chunks (persistent, evaluated corpus)")
        else:
            st.caption("Index: not loaded")
    else:
        if "uploaded_index" in st.session_state:
            n = st.session_state["uploaded_index"].index.ntotal
            n_docs = len(st.session_state.get("uploaded_docs", []))
            st.caption(f"Index: {n} chunks (session-only, {n_docs} uploaded files)")
        else:
            st.caption("Index: none built yet")


# ---------------------------------------------------------------------------
# Clear stale results when the user switches modes
# ---------------------------------------------------------------------------
if st.session_state.get("_last_mode") != mode:
    st.session_state["_last_mode"] = mode
    st.session_state.pop("result", None)
    st.session_state.pop("last_query", None)


# ---------------------------------------------------------------------------
# Resolve active retriever and corpus tag for this render
# ---------------------------------------------------------------------------
active_retriever: Optional[Retriever] = None
corpus_mode_tag = "indexed"

if mode == _MODE_INDEXED:
    if _indexed_retriever is None:
        st.error(_indexed_error)
        st.stop()
    active_retriever = _indexed_retriever
else:
    corpus_mode_tag = "uploaded"
    if "uploaded_index" in st.session_state:
        active_retriever = Retriever.from_vectorstore(
            st.session_state["uploaded_index"]
        )


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("Document Navigator")
st.caption("Grounded answers from your PDF corpus, with full transparency.")

# Upload UI — only rendered in upload mode
if mode == _MODE_UPLOAD:
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    st.caption(
        "Max 10 files, 20 MB and 200 pages per file. "
        "Uploads are session-only; nothing is saved."
    )

    if uploaded_files:
        col_build, col_clear = st.columns(2)
        build_clicked = col_build.button("Build index from uploads")
        clear_clicked = (
            col_clear.button("Clear uploaded index")
            if "uploaded_index" in st.session_state
            else False
        )

        if clear_clicked:
            st.session_state.pop("uploaded_index", None)
            st.session_state.pop("uploaded_docs", None)
            st.session_state.pop("result", None)
            st.session_state.pop("last_query", None)
            st.rerun()

        if build_clicked:
            file_tuples = [(uf.name, uf.getvalue()) for uf in uploaded_files]
            with st.spinner("Building index from uploaded PDFs…"):
                try:
                    vectorstore, docs = build_index_from_uploads(file_tuples)
                except (
                    UploadTooLargeError,
                    UploadTooManyPagesError,
                    UploadTooManyFilesError,
                    UploadEmptyError,
                ) as exc:
                    st.error(str(exc))
                    st.stop()

            st.session_state["uploaded_index"] = vectorstore
            st.session_state["uploaded_docs"] = docs
            st.session_state.pop("result", None)
            st.session_state.pop("last_query", None)
            active_retriever = Retriever.from_vectorstore(vectorstore)

            total_chunks = sum(d.n_chunks for d in docs)
            st.success(f"Indexed {len(docs)} file(s), {total_chunks} chunks total.")
            with st.expander("Indexed files"):
                for doc in docs:
                    st.write(
                        f"**{doc.filename}** — {doc.n_pages} page(s), "
                        f"{doc.n_chunks} chunk(s), "
                        f"{doc.size_bytes / 1024:.0f} KB"
                    )

    elif "uploaded_index" in st.session_state:
        # Uploader was cleared but session index still exists; offer to clear it.
        if st.button("Clear uploaded index"):
            st.session_state.pop("uploaded_index", None)
            st.session_state.pop("uploaded_docs", None)
            st.session_state.pop("result", None)
            st.session_state.pop("last_query", None)
            st.rerun()

    if "uploaded_docs" in st.session_state and not uploaded_files:
        with st.expander("Current uploaded index"):
            for doc in st.session_state["uploaded_docs"]:
                st.write(
                    f"**{doc.filename}** — {doc.n_pages} page(s), "
                    f"{doc.n_chunks} chunk(s), "
                    f"{doc.size_bytes / 1024:.0f} KB"
                )

# Guard: no retriever available yet — nothing more to show
if active_retriever is None:
    st.info("Upload PDFs and click 'Build index from uploads' to start querying.")
    st.stop()

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
query = st.text_input(
    "Ask a question about your documents:",
    placeholder="e.g., How long does standard shipping take?",
)
button_clicked = st.button("Ask")

# Trigger on button click OR Enter (text_input reruns on Enter by default).
# The last_query guard prevents duplicate submissions on unrelated reruns.
if query and (button_clicked or st.session_state.get("last_query") != query):
    st.session_state["last_query"] = query
    with st.spinner("Retrieving and generating…"):
        try:
            result: GenerationResult = generate_answer(
                query=query,
                k=k,
                model=model,
                temperature=temperature,
                retriever=active_retriever,
                write_jsonl_trace=True,
                strict=strict,
                corpus_mode=corpus_mode_tag,
            )
            st.session_state["result"] = result
        except Exception as exc:
            exc_repr = repr(exc).lower()
            if any(kw in exc_repr for kw in ("connect", "refused", "unreachable", "timeout")):
                st.error(
                    f"Cannot reach Ollama at http://localhost:11434. "
                    f"Start it with `ollama serve` and ensure the model is pulled "
                    f"with `ollama pull {model}`."
                )
            else:
                st.error(f"Generation failed: {exc}")
            st.stop()

# ---------------------------------------------------------------------------
# Results display (persists across sidebar-triggered reruns)
# ---------------------------------------------------------------------------
if "result" in st.session_state:
    result: GenerationResult = st.session_state["result"]  # type: ignore[no-redef]

    # Evidence-strength badge
    sim_label = f"{result.top_similarity:.3f}"
    if result.evidence_strength == "strong":
        st.success(f"Strong evidence (top similarity: {sim_label})")
    elif result.evidence_strength == "weak":
        st.warning(f"Weak evidence (top similarity: {sim_label})")
    else:
        st.error(f"No evidence — refused (top similarity: {sim_label})")

    # Answer
    st.subheader("Answer")
    st.markdown(result.answer)

    # Citations
    st.subheader("Citations used")
    if result.citations_used:
        for cit in result.citations_used:
            st.markdown(f"- {cit}")
    else:
        st.markdown("*No citations — the system refused due to insufficient evidence.*")

    # Retrieved chunks — one expander per hit
    st.subheader("Retrieved chunks")
    for hit in result.retrieval_trace.hits:
        with st.expander(
            f"#{hit.rank} — {hit.citation} (similarity: {hit.similarity:.3f})"
        ):
            st.caption(f"chunk_id={hit.chunk_id}  chunk_index={hit.chunk_index}")
            st.text(hit.text)

    # Metrics footer
    col1, col2, col3 = st.columns(3)
    col1.metric("Top similarity", sim_label)
    col2.metric("Retrieval time", f"{result.retrieval_trace.elapsed_ms:.0f} ms")
    col3.metric("Generation time", f"{result.elapsed_ms_generation:.0f} ms")
