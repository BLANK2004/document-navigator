# Document Navigator

A transparent, locally-run RAG assistant over a PDF corpus. Every answer
cites its sources in [filename.pdf:page] format, refuses politely when
evidence is weak, and ships with a 15-question evaluation harness.

**Stack:** Python · LangChain · FAISS · sentence-transformers (MiniLM-L6-v2)
· Ollama (qwen2.5:7b) · Streamlit — no API keys, runs fully offline.

## What it does

`src/ingest.py` loads PDFs page-by-page, splits them into overlapping
character-level chunks with metadata (source, page, chunk_id), embeds
them with MiniLM-L6-v2, and persists a FAISS index to `db_faiss/`.
`src/retrieve.py` runs top-k similarity search against that index,
converts FAISS L2² distances to cosine scores, and appends a JSONL trace
per query to `logs/retrieval_traces.jsonl`. `src/generate.py` gates on
evidence strength — queries whose top-1 similarity falls below 0.25 are
refused before the LLM is called — then assembles retrieved chunks into a
structured CONTEXT block, invokes ChatOllama with separated system and
human messages, and extracts inline [filename.pdf:page] citations from the
response. `eval/evaluate.py` replays every row in `eval/eval_set.csv`
through the full pipeline and scores two cohorts: answer quality (15
questions, retrieval precision + citation + key-phrase accuracy) and safety
behavior (5 adversarial questions expected to trigger the refusal path,
scored on refusal correctness and false-refusal rate). `app.py` exposes the
full pipeline as a Streamlit UI with per-query evidence badges, retrieved
chunk expanders, and sidebar controls for top-k, model, temperature, and
strict-evidence mode. `notebooks/demo.ipynb` narrates the complete pipeline
from ingestion config through retrieval traces, generation, and evaluation
results. Includes an optional upload mode in the Streamlit app: users can
supply their own PDFs and query them in a session-scoped in-memory index,
without modifying or contaminating the evaluated persistent corpus.

## Evaluation results

On the 20-question eval set (15 answer-cohort, 5 safety-cohort):

| Metric | Score |
|---|---|
| precision@1 (answer cohort) | 0.867 |
| precision@3 (answer cohort) | 1.000 |
| precision@5 (answer cohort) | 1.000 |
| citation_accuracy | 0.800 |
| key_phrase_accuracy | 0.733 |
| answer_pass_rate | 0.733 |
| refusal_correctness (safety cohort) | 1.000 |
| false_refusal_rate (answer cohort) | 0.000 |
| mean latency | 4.7 s per answered query |

Manual review of the 4 failing answer-cohort rows shows only 1 is a genuine
answer-quality failure (distractor confusion); the other 3 are correct
answers penalised by the single-gold-phrase eval rubric. The 5 safety rows
(3 out-of-scope, 2 prompt-injection attempts) all triggered the pre-LLM
refusal path with top similarity below 0.25 — the LLM was never invoked.
See [`reports/retrieval_report.md`](reports/retrieval_report.md) for the
full findings, configuration choices, limitations, and next steps.

![Precision@k bar chart](reports/precision_at_k.png)

## Architecture

```
documents/*.pdf
│
▼  src/ingest.py          PyPDFLoader → RecursiveCharacterTextSplitter
│                          → HuggingFaceEmbeddings (MiniLM-L6-v2, normalized)
▼
db_faiss/                  FAISS index + docstore, persisted locally
│
▼  src/retrieve.py        Top-k similarity search; cosine = 1 - dist/2
│                          → logs/retrieval_traces.jsonl
▼
src/generate.py            Evidence-strength gating (refuse < 0.25);
│                          ChatOllama (qwen2.5:7b) with separated
│                          system/human messages; citation regex
▼                          → logs/generation_traces.jsonl
Grounded answer + [filename.pdf:page] citations
│
▼  eval/evaluate.py        Replays eval_set.csv end-to-end
                           → reports/{eval_results.csv,
                                      eval_summary.json,
                                      retrieval_report.md,
                                      precision_at_k.png}

app.py                     Streamlit UI: query → answer + citations
                           + similarity-scored retrieved chunks

src/upload_index.py        In-memory FAISS builder for uploaded PDFs;
                           reuses ingest.py's chunking and embedding
                           (identical config); session-scoped only

notebooks/demo.ipynb       Narrative walkthrough of the full pipeline
```

## Quickstart

Prerequisites: Python 3.10+, ~6 GB free RAM, Ollama installed
([ollama.com/download](https://ollama.com/download)).

```bash
# 1. Setup
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Pull the local LLM (one-time, ~4.7 GB)
ollama pull qwen2.5:7b

# 3. Place PDFs in documents/ , then build the index
python -m src.ingest --smoke-test

# 4. Ask a question from the CLI
python -m src.generate "How long does standard shipping take?"

# 5. Run the full evaluation (~2-5 min on CPU)
python -m eval.evaluate

# 6. Launch the interactive demo UI
streamlit run app.py               # opens http://localhost:8501
                                   # toggle the sidebar to upload your own PDFs

# 7. Or open the narrative walkthrough notebook
jupyter notebook notebooks/demo.ipynb
```

## Example output

Strong-evidence answer:

```
$ python -m src.generate "How long does standard shipping take?"

Standard delivery takes 3–6 business days depending on location.
[policy_shipping_returns.pdf:1]

Sources used: [policy_shipping_returns.pdf:1]
Evidence strength: strong (top sim: 0.733)
```

Out-of-scope query — refused before the LLM is called:

```
$ python -m src.generate "What is the capital of France?"

I don't have enough information in the indexed documents to answer this question.

Sources used: (none)
Evidence strength: none (top sim: 0.041)
```

## Demo UI

The Streamlit app at `app.py` is the visual proof of the system. After
`streamlit run app.py`, the browser opens to a query input with a sidebar
for top-k, model, temperature, and a strict-evidence toggle. Each submitted
query returns three vertically stacked sections: the grounded answer with
inline citations, an evidence-strength badge (green for strong, yellow for
weak, red for refused), and per-hit expanders showing source, page,
similarity score, chunk ID, and full chunk text.

A reviewer can validate the system in three queries: "How long does standard
shipping take?" (strong evidence, ~0.73 similarity, cited answer), "What is
precision at k?" (weak evidence, ~0.34 similarity, cited answer), and "What
is the capital of France?" (no evidence, ~0.04 similarity, pre-LLM refusal
with the LLM never invoked). Toggling strict mode pushes the precision@k
query into the refusal band, making the threshold behavior visible
end-to-end.

The app supports two corpus modes via a sidebar toggle. **Indexed corpus
(evaluated)** uses the persistent 10-PDF FAISS index that the eval ran
against — this is the configuration the metrics above apply to. **Upload
your own PDFs** opens a file uploader (max 10 files, 20 MB and 200 pages
per file) and builds a session-scoped in-memory index after the user clicks
Build. Uploaded indexes never touch `db_faiss/` and disappear when the
browser session ends. The retrieval, generation, citation, and refusal logic
are identical across both modes — the upload feature is a corpus source, not
an alternate pipeline. Generation traces include a `corpus_mode` field
(`"indexed"` or `"uploaded"`) so any future evaluation can filter to the
persistent corpus.

## Project layout

```
document-navigator/
├── src/            Core pipeline: ingest.py, retrieve.py, generate.py, config.py,
│                   upload_index.py (in-memory builder for upload mode)
├── eval/           Evaluation harness (evaluate.py) and eval_set.csv
├── notebooks/      demo.ipynb — narrative walkthrough of the full pipeline
├── documents/      Input PDFs — place your files here before ingesting
├── db_faiss/       Persisted FAISS index and docstore (git-ignored)
├── logs/           Per-query JSONL traces for retrieval and generation
├── reports/        eval_results.csv, eval_summary.json, retrieval_report.md,
│                   precision_at_k.png
├── tests/          Unit tests
├── app.py          Streamlit demo: toggle between indexed corpus
│                   and session-scoped uploaded PDFs
├── requirements.txt
└── README.md
```

## Design decisions

- **Cosine via normalized embeddings.** `normalize_embeddings=True` lets FAISS
  inner-product mathematically equal cosine similarity, giving interpretable
  [0, 1] scores in retrieval traces without a separate normalization pass.
- **Pre-LLM refusal at sim < 0.25.** Saves an LLM call and removes any chance
  of the model answering from training data when the index holds no relevant
  evidence.
- **Separate system and human messages.** Treating retrieved chunks as untrusted
  data — not instructions — defends against prompt injection from inside the
  corpus without any extra filtering logic.
- **JSONL traces for both retrieval and generation.** Every query produces a
  structured record with scores, citations, and elapsed times, giving the
  evaluator a paper trail for any answer the system produced without
  re-embedding.
- **Two-axis evaluation.** The eval set scores both answer quality (15
  questions with gold citations and key phrases) and safety behavior (5
  adversarial questions expected to refuse). Refusal correctness and
  false-refusal rate are first-class metrics, not afterthoughts.
- **Session-scoped uploads.** The upload mode builds an in-memory FAISS
  index that never persists to `db_faiss/` — this preserves the integrity
  of the evaluated corpus and keeps the reported metrics honest, while still
  giving users a way to try the system on their own documents.

## Limitations

- The 20-question eval set is small; results should be treated as directional
  rather than statistically robust.
- Each answer question has a single gold key phrase, which penalises valid
  paraphrases — 3 of the 4 answer-cohort failures are eval-rubric artifacts
  rather than real quality issues.
- Adversarial robustness is measured only at the similarity-gating layer; all
  5 safety rows fell below the 0.25 threshold and never reached the LLM, so
  the second and third defense layers (separated system/human messages,
  in-prompt CONTEXT-as-data instruction) remain untested under realistic
  injection pressure.
- Adversarial PDFs in upload mode are untested. The same prompt-injection
  defenses apply (separated system/human messages, CONTEXT-as-data
  instruction, similarity-floor refusal), but the eval set does not include
  uploaded-PDF attack cases. This is in Next Steps in the full report.

See [`reports/retrieval_report.md`](reports/retrieval_report.md) for the full
limitations list and next steps.

## License

MIT — see LICENSE
