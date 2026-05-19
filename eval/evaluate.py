"""
eval/evaluate.py
================
End-to-end evaluator for Document Navigator.

Reads a CSV of (question, gold_citation, gold_key_phrase, expected_behavior)
rows, runs generate_answer() on each, and scores two cohorts:

  Answer rows (expected_behavior="answer"):
    - Retrieval:  does the gold document appear in top-1 / top-3 / top-5?
    - Generation: does the answer contain the gold citation and key phrase?

  Refuse rows (expected_behavior="refuse"):
    - Safety:     did the system refuse (refused=True)?

Writes three output files:
  reports/eval_results.csv      — per-row scores, flattened
  reports/eval_summary.json     — aggregate metric dict
  reports/retrieval_report.md   — auto-generated sections updated in-place;
                                   human-written sections (Findings /
                                   Limitations / Next Steps) are preserved.

Run
---
    python -m eval.evaluate
    python -m eval.evaluate --limit 5          # quick smoke test
    python eval/evaluate.py                    # script invocation also works
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for both -m and direct-script invocations.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import csv
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.generate import CITATION_RE, generate_answer
from src.ingest import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from src.retrieve import DEFAULT_EMBEDDING_MODEL, Retriever


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = _PROJECT_ROOT
EVAL_DIR: Path = PROJECT_ROOT / "eval"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
LOG_DIR: Path = PROJECT_ROOT / "logs"

DEFAULT_EVAL_CSV: Path = EVAL_DIR / "eval_set.csv"
DEFAULT_K: int = 5
DEFAULT_MODEL: str = "qwen2.5:7b"

RESULTS_CSV: Path = REPORTS_DIR / "eval_results.csv"
SUMMARY_JSON: Path = REPORTS_DIR / "eval_summary.json"
REPORT_MD: Path = REPORTS_DIR / "retrieval_report.md"
PRECISION_AT_K_PNG: Path = REPORTS_DIR / "precision_at_k.png"

_CSV_FIELDNAMES = [
    "id", "question", "gold_citation", "gold_key_phrase", "expected_behavior",
    "gold_filename", "retrieved_top_k", "generated_answer", "citations_used",
    "evidence_strength", "top_similarity", "refused",
    "gold_in_top_1", "gold_in_top_3", "gold_in_top_5",
    "gold_citation_in_answer", "key_phrase_in_answer", "passed",
    "elapsed_ms_total",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _configure_logging() -> logging.Logger:
    """Configure a console + file logger. Safe to call multiple times."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "evaluate.log"

    logger = logging.getLogger("evaluate")
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
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class EvalRow:
    # Input
    id: str
    question: str
    gold_citation: str             # normalized "[filename.pdf:page]" or "" for refuse rows
    gold_key_phrase: str
    expected_behavior: str         # "answer" | "refuse"
    gold_filename: str             # parsed from gold_citation; "" for refuse rows

    # Retrieval snapshot
    retrieved_top_k: List[Dict[str, Any]]  # {rank, source, page, similarity, citation}

    # Generation output
    generated_answer: str
    citations_used: List[str]
    evidence_strength: str
    top_similarity: float
    refused: bool

    # Metric booleans
    gold_in_top_1: bool
    gold_in_top_3: bool
    gold_in_top_5: bool
    gold_citation_in_answer: bool
    key_phrase_in_answer: bool
    passed: bool                   # behavior-aware: see _evaluate_row

    # Timing
    elapsed_ms_total: float


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _normalize_text(text: str) -> str:
    """Collapse whitespace and lower-case — used for key-phrase matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _normalize_citation(raw: str) -> str:
    """Add brackets to a bare 'file.pdf:N' value if they are missing."""
    raw = raw.strip()
    if not raw.startswith("["):
        raw = f"[{raw}]"
    return raw


# ---------------------------------------------------------------------------
# Row evaluation
# ---------------------------------------------------------------------------
def _evaluate_row(
    csv_row: Dict[str, str],
    retriever: Retriever,
    k: int,
    model: str,
) -> Optional[EvalRow]:
    row_id = csv_row["id"]
    question = csv_row["question"].strip()
    raw_citation = csv_row.get("gold_citation", "").strip()
    gold_key_phrase = csv_row.get("gold_key_phrase", "").strip()
    expected_behavior = (csv_row.get("expected_behavior", "") or "").strip() or "answer"

    # Parse gold citation for "answer" rows; skip gracefully for "refuse" rows.
    if expected_behavior == "answer":
        normalized = _normalize_citation(raw_citation)
        m = CITATION_RE.search(normalized)
        if not m:
            log.warning(
                "Row %s: cannot parse gold_citation %r — skipping", row_id, raw_citation
            )
            return None
        gold_filename = m.group(1)
        gold_citation = f"[{m.group(1)}:{m.group(2)}]"
    else:
        gold_filename = ""
        gold_citation = ""

    # Generate answer (also performs retrieval internally).
    t0 = time.perf_counter()
    result = generate_answer(
        query=question,
        k=k,
        model=model,
        retriever=retriever,
        write_jsonl_trace=False,
    )
    elapsed_ms_total = round((time.perf_counter() - t0) * 1000.0, 2)

    hits = result.retrieval_trace.hits
    retrieved_top_k = [
        {
            "rank": h.rank,
            "source": h.source,
            "page": h.page,
            "similarity": round(h.similarity, 4),
            "citation": h.citation,
        }
        for h in hits
    ]

    def _sources(n: int) -> List[str]:
        return [h.source for h in hits[:n]]

    if expected_behavior == "answer":
        gold_in_top_1 = gold_filename in _sources(1)
        gold_in_top_3 = gold_filename in _sources(3)
        gold_in_top_5 = gold_filename in _sources(5)
        gold_citation_in_answer = gold_citation in result.citations_used
        key_phrase_in_answer = _normalize_text(gold_key_phrase) in _normalize_text(result.answer)
        passed = gold_in_top_5 and key_phrase_in_answer
    else:  # refuse
        gold_in_top_1 = gold_in_top_3 = gold_in_top_5 = False
        gold_citation_in_answer = False
        key_phrase_in_answer = False
        passed = result.refused

    log.info(
        "%s | expect=%s | refused=%s | passed=%s | sim=%.3f",
        row_id, expected_behavior, result.refused, passed, result.top_similarity,
    )

    return EvalRow(
        id=row_id,
        question=question,
        gold_citation=gold_citation,
        gold_key_phrase=gold_key_phrase,
        expected_behavior=expected_behavior,
        gold_filename=gold_filename,
        retrieved_top_k=retrieved_top_k,
        generated_answer=result.answer,
        citations_used=result.citations_used,
        evidence_strength=result.evidence_strength,
        top_similarity=result.top_similarity,
        refused=result.refused,
        gold_in_top_1=gold_in_top_1,
        gold_in_top_3=gold_in_top_3,
        gold_in_top_5=gold_in_top_5,
        gold_citation_in_answer=gold_citation_in_answer,
        key_phrase_in_answer=key_phrase_in_answer,
        passed=passed,
        elapsed_ms_total=elapsed_ms_total,
    )


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
def _compute_metrics(rows: List[EvalRow]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n_rows": 0}

    answer_rows = [r for r in rows if r.expected_behavior == "answer"]
    refuse_rows = [r for r in rows if r.expected_behavior == "refuse"]
    n_answer = len(answer_rows)
    n_refuse = len(refuse_rows)

    def _frac(values: List[bool], total: int) -> Optional[float]:
        return round(sum(values) / total, 4) if total > 0 else None

    # Per-strength breakdown for all rows and split by behavior.
    strength_counts: Dict[str, int] = {"strong": 0, "weak": 0, "none": 0}
    strength_by_behavior: Dict[str, Dict[str, int]] = {
        "answer": {"strong": 0, "weak": 0, "none": 0},
        "refuse": {"strong": 0, "weak": 0, "none": 0},
    }
    for r in rows:
        strength_counts[r.evidence_strength] = (
            strength_counts.get(r.evidence_strength, 0) + 1
        )
        beh_bucket = strength_by_behavior.setdefault(
            r.expected_behavior, {"strong": 0, "weak": 0, "none": 0}
        )
        beh_bucket[r.evidence_strength] = beh_bucket.get(r.evidence_strength, 0) + 1

    return {
        "n_rows": n,
        "n_answer_rows": n_answer,
        "n_refuse_rows": n_refuse,
        # Retrieval metrics — meaningful only for answer rows.
        "precision_at_1": _frac([r.gold_in_top_1 for r in answer_rows], n_answer),
        "precision_at_3": _frac([r.gold_in_top_3 for r in answer_rows], n_answer),
        "precision_at_5": _frac([r.gold_in_top_5 for r in answer_rows], n_answer),
        # Generation quality — answer rows only.
        "citation_accuracy": _frac([r.gold_citation_in_answer for r in answer_rows], n_answer),
        "key_phrase_accuracy": _frac([r.key_phrase_in_answer for r in answer_rows], n_answer),
        # Pass rates — cohort-aware.
        "answer_pass_rate": _frac([r.passed for r in answer_rows], n_answer),
        "refusal_correctness": _frac([r.passed for r in refuse_rows], n_refuse),
        "false_refusal_rate": _frac([r.refused for r in answer_rows], n_answer),
        # Combined pass rate across all rows.
        "overall_pass_rate": round(sum(r.passed for r in rows) / n, 4),
        # Misc — all rows.
        "refusal_rate": round(sum(r.refused for r in rows) / n, 4),
        "mean_top_similarity": round(sum(r.top_similarity for r in rows) / n, 4),
        "mean_elapsed_ms_total": round(sum(r.elapsed_ms_total for r in rows) / n, 2),
        "evidence_strength_counts": strength_counts,
        "evidence_strength_by_behavior": strength_by_behavior,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _write_csv(rows: List[EvalRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            retrieved_str = " | ".join(
                f"rank={h['rank']} source={h['source']} page={h['page']} sim={h['similarity']:.3f}"
                for h in row.retrieved_top_k
            )
            writer.writerow({
                "id": row.id,
                "question": row.question,
                "gold_citation": row.gold_citation,
                "gold_key_phrase": row.gold_key_phrase,
                "expected_behavior": row.expected_behavior,
                "gold_filename": row.gold_filename,
                "retrieved_top_k": retrieved_str,
                "generated_answer": row.generated_answer,
                "citations_used": ", ".join(row.citations_used),
                "evidence_strength": row.evidence_strength,
                "top_similarity": round(row.top_similarity, 4),
                "refused": row.refused,
                "gold_in_top_1": row.gold_in_top_1,
                "gold_in_top_3": row.gold_in_top_3,
                "gold_in_top_5": row.gold_in_top_5,
                "gold_citation_in_answer": row.gold_citation_in_answer,
                "key_phrase_in_answer": row.key_phrase_in_answer,
                "passed": row.passed,
                "elapsed_ms_total": row.elapsed_ms_total,
            })
    log.info("Results CSV written: %s", path)


def _write_json(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    log.info("Summary JSON written: %s", path)


def _write_report(
    rows: List[EvalRow],
    metrics: Dict[str, Any],
    path: Path,
) -> None:
    """
    Regenerate the auto-produced sections of the Markdown report and preserve
    any human-written sections (## Findings and beyond) that follow them.
    """
    # Preserve human-written sections from the existing file.
    preserved_tail = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        marker = "\n## Configuration choices"
        idx = existing.find(marker)
        if idx != -1:
            preserved_tail = existing[idx:]

    by_beh = metrics["evidence_strength_by_behavior"]
    ans_s = by_beh.get("answer", {})
    ref_s = by_beh.get("refuse", {})

    # --- Examples (Good): answer rows that passed, highest similarity first ---
    good = sorted(
        [r for r in rows if r.passed and r.expected_behavior == "answer"],
        key=lambda r: r.top_similarity,
        reverse=True,
    )[:3]

    # --- Examples (Failures): answer rows that failed, lowest similarity first ---
    answer_failures = sorted(
        [r for r in rows if not r.passed and r.expected_behavior == "answer"],
        key=lambda r: r.top_similarity,
    )[:3]

    # --- Refuse rows that failed (did not refuse) ---
    refuse_failures = [r for r in rows if not r.passed and r.expected_behavior == "refuse"]

    def _top1_label(row: EvalRow) -> str:
        if row.retrieved_top_k:
            h = row.retrieved_top_k[0]
            return f"{h['source']} (sim={h['similarity']:.3f})"
        return "—"

    def _fail_reason(row: EvalRow) -> str:
        if row.expected_behavior == "refuse":
            return "system produced an answer instead of refusing"
        parts = []
        if not row.gold_in_top_5:
            parts.append("gold document not in top-5 retrieval")
        if not row.key_phrase_in_answer:
            parts.append("key phrase absent from generated answer")
        return "; ".join(parts) or "unknown"

    n_answer = metrics["n_answer_rows"]
    n_refuse = metrics["n_refuse_rows"]

    lines: List[str] = [
        "# retrieval_report.md — Document Navigator",
        "",
        "## Setup",
        "- PDFs: 10",
        f"- Chunk size: {DEFAULT_CHUNK_SIZE} chars",
        f"- Overlap: {DEFAULT_CHUNK_OVERLAP} chars",
        f"- Embeddings model: {DEFAULT_EMBEDDING_MODEL}",
        "- Vector index: FAISS (local, cosine similarity via normalized L2)",
        "",
        "## Metrics",
        f"- n_rows: {metrics['n_rows']} ({n_answer} answer, {n_refuse} refuse)",
        f"- precision@1: {metrics['precision_at_1']:.4f}  (answer rows)",
        f"- precision@3: {metrics['precision_at_3']:.4f}  (answer rows)",
        f"- precision@5: {metrics['precision_at_5']:.4f}  (= recall@5 for single-gold queries)",
        f"- citation_accuracy: {metrics['citation_accuracy']:.4f}",
        f"- key_phrase_accuracy: {metrics['key_phrase_accuracy']:.4f}",
        f"- answer_pass_rate: {metrics['answer_pass_rate']:.4f}",
        f"- refusal_correctness: {metrics['refusal_correctness']:.4f}  (refuse rows: fraction correctly refused)",
        f"- false_refusal_rate: {metrics['false_refusal_rate']:.4f}  (answer rows incorrectly refused)",
        f"- overall_pass_rate: {metrics['overall_pass_rate']:.4f}  (all {metrics['n_rows']} rows)",
        f"- refusal_rate: {metrics['refusal_rate']:.4f}  (all rows)",
        f"- mean_top_similarity: {metrics['mean_top_similarity']:.4f}",
        f"- mean_elapsed_ms: {metrics['mean_elapsed_ms_total']:.1f} ms",
        f"- evidence_strength (answer rows): strong={ans_s.get('strong', 0)}"
        f"  weak={ans_s.get('weak', 0)}"
        f"  none={ans_s.get('none', 0)}",
        f"- evidence_strength (refuse rows): strong={ref_s.get('strong', 0)}"
        f"  weak={ref_s.get('weak', 0)}"
        f"  none={ref_s.get('none', 0)}",
        "",
        "![Precision@k bar chart](precision_at_k.png)",
        "",
        "## Examples (Good)",
    ]

    if good:
        for row in good:
            lines += [
                f"- Q: {row.question}",
                f"  - Retrieved: {_top1_label(row)}",
                f"  - Answer + citation: {row.generated_answer}",
                f"  - Gold: {row.gold_citation}",
                "",
            ]
    else:
        lines += ["_(no passing answer rows)_", ""]

    lines.append("## Examples (Failures)")

    if answer_failures:
        for row in answer_failures:
            lines += [
                f"- Q: {row.question}",
                f"  - Retrieved: {_top1_label(row)}",
                f"  - Answer + citation: {row.generated_answer}",
                f"  - Gold: {row.gold_citation}",
                f"  - Why it failed: {_fail_reason(row)}",
                "  - Fix attempted: —",
                "",
            ]
    else:
        lines += [f"All {n_answer} answer rows passed — no failure examples to show.", ""]

    if refuse_failures:
        lines.append("## Refusal Failures (system answered when it should have refused)")
        for row in refuse_failures:
            lines += [
                f"- Q: {row.question}",
                f"  - Top-1 sim: {row.retrieved_top_k[0]['similarity']:.4f}" if row.retrieved_top_k else "  - Top-1 sim: —",
                f"  - Generated answer: {row.generated_answer}",
                "",
            ]

    content = "\n".join(lines)
    if preserved_tail:
        content = content + preserved_tail

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    log.info("Retrieval report written: %s", path)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def _print_summary(metrics: Dict[str, Any]) -> None:
    n = metrics["n_rows"]
    n_answer = metrics["n_answer_rows"]
    n_refuse = metrics["n_refuse_rows"]
    passed = int(round(metrics["overall_pass_rate"] * n))
    ans_s = metrics["evidence_strength_by_behavior"].get("answer", {})
    ref_s = metrics["evidence_strength_by_behavior"].get("refuse", {})
    sep = "─" * 54

    def _pct(v: Optional[float]) -> str:
        return f"{v:.4f}" if v is not None else "N/A"

    print(f"\n{sep}")
    print(f"EVALUATION SUMMARY  (n={n}  answer={n_answer}  refuse={n_refuse})")
    print(sep)
    print(f"  Answer cohort (n={n_answer})")
    print(f"    precision@1:          {_pct(metrics['precision_at_1'])}")
    print(f"    precision@3:          {_pct(metrics['precision_at_3'])}")
    print(f"    precision@5:          {_pct(metrics['precision_at_5'])}")
    print(f"    citation_accuracy:    {_pct(metrics['citation_accuracy'])}")
    print(f"    key_phrase_accuracy:  {_pct(metrics['key_phrase_accuracy'])}")
    print(f"    answer_pass_rate:     {_pct(metrics['answer_pass_rate'])}")
    print(f"    false_refusal_rate:   {_pct(metrics['false_refusal_rate'])}")
    print(f"    evidence_strength:    strong={ans_s.get('strong', 0)}"
          f"  weak={ans_s.get('weak', 0)}"
          f"  none={ans_s.get('none', 0)}")
    print(f"  Refuse cohort (n={n_refuse})")
    print(f"    refusal_correctness:  {_pct(metrics['refusal_correctness'])}")
    print(f"    evidence_strength:    strong={ref_s.get('strong', 0)}"
          f"  weak={ref_s.get('weak', 0)}"
          f"  none={ref_s.get('none', 0)}")
    print("  Combined")
    print(f"    overall_pass_rate:    {metrics['overall_pass_rate']:.4f}")
    print(f"    refusal_rate:         {metrics['refusal_rate']:.4f}")
    print(f"    mean_top_similarity:  {metrics['mean_top_similarity']:.4f}")
    print(f"    mean_elapsed_ms:      {metrics['mean_elapsed_ms_total']:.1f} ms")
    print(sep)
    print(f"Eval complete: {passed}/{n} passed ({metrics['overall_pass_rate'] * 100:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Precision@k chart
# ---------------------------------------------------------------------------
def plot_precision_at_k(summary: Dict[str, Any], output_path: Path) -> None:
    """Save a precision@k bar chart to output_path. Skips if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        log.warning("matplotlib not available — skipping precision@k chart.")
        return

    labels = ["@1", "@3", "@5"]
    values = [
        summary["precision_at_1"],
        summary["precision_at_3"],
        summary["precision_at_5"],
    ]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    bars = ax.bar(labels, values, color="#4C72B0")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Precision")
    ax.set_xlabel("k")
    ax.set_title(f"Retrieval precision@k (answer cohort, n={summary.get('n_answer_rows', '?')})")
    ax.axhline(y=1.0, linestyle="--", color="lightgray", linewidth=1)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    log.info("Precision@k chart saved: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Document Navigator — end-to-end evaluator."
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=DEFAULT_EVAL_CSV,
        help=f"Path to the evaluation CSV (default: {DEFAULT_EVAL_CSV}).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of chunks to retrieve per query (default: {DEFAULT_K}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama model tag (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows (useful for smoke tests).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.eval_csv.exists():
        log.error("Eval CSV not found: %s", args.eval_csv)
        return 1

    # Verify the CSV parses cleanly before doing any heavy work.
    with args.eval_csv.open(encoding="utf-8", newline="") as fh:
        all_rows = list(csv.DictReader(fh))

    if args.limit is not None:
        all_rows = all_rows[: args.limit]

    log.info(
        "Evaluating %d row(s) from %s (k=%d, model=%s)",
        len(all_rows), args.eval_csv, args.k, args.model,
    )

    # One Retriever instance reused across all rows — avoids re-loading the index.
    retriever = Retriever()

    eval_rows: List[EvalRow] = []
    for csv_row in tqdm(all_rows, desc="Evaluating", unit="row"):
        try:
            row = _evaluate_row(csv_row, retriever=retriever, k=args.k, model=args.model)
        except Exception:  # noqa: BLE001
            log.exception("Row %s raised an unexpected error — skipping.", csv_row.get("id"))
            row = None
        if row is not None:
            eval_rows.append(row)

    if not eval_rows:
        log.error("No rows evaluated successfully — aborting output.")
        return 1

    metrics = _compute_metrics(eval_rows)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(eval_rows, RESULTS_CSV)
    _write_json(metrics, SUMMARY_JSON)
    _write_report(eval_rows, metrics, REPORT_MD)
    plot_precision_at_k(metrics, PRECISION_AT_K_PNG)
    _print_summary(metrics)

    return 0


if __name__ == "__main__":
    sys.exit(main())
