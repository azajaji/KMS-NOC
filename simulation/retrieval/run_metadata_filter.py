"""Condition C2: metadata-filtered retrieval.

Cascade filter on (technology_domain, incident_type), then narrow further by severity
and region when enough candidates remain. Rank the surviving candidates by TF-IDF
cosine similarity on the scenario text.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import RESULTS_DIR, load_records, load_tasks, record_text, write_rankings

CONDITION = "C2_metadata_filtered"
TOP_K = 10
MIN_CANDIDATES_AFTER_NARROWING = 6


def _filter(records: list[dict], task: dict) -> list[int]:
    """Return indices of records that survive the metadata cascade."""
    dom = task["technology_domain"]
    inc = task["incident_type"]
    sev = task["severity"]
    reg = task["region"]

    base = [i for i, r in enumerate(records)
            if r["technology_domain"] == dom and r["incident_type"] == inc]
    if not base:
        # Fall back to domain-only so we never produce an empty candidate set.
        base = [i for i, r in enumerate(records) if r["technology_domain"] == dom]

    # Optional narrowing: keep severity match, then region, only if enough remain.
    narrowed = [i for i in base if records[i]["severity"] in {sev, "S2"}]
    if len(narrowed) >= MIN_CANDIDATES_AFTER_NARROWING:
        base = narrowed
    narrowed = [i for i in base if records[i]["region"] in {reg, "All"}]
    if len(narrowed) >= MIN_CANDIDATES_AFTER_NARROWING:
        base = narrowed
    return base


def main() -> None:
    records = load_records()
    tasks = load_tasks()
    corpus_texts = [record_text(r) for r in records]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
    doc_matrix = vectorizer.fit_transform(corpus_texts)
    query_texts = [t["scenario"] for t in tasks]
    query_matrix = vectorizer.transform(query_texts)
    rows = []
    cand_counts = []
    for i, task in enumerate(tasks):
        cand_idx = _filter(records, task)
        cand_counts.append(len(cand_idx))
        # Score only the candidate subset
        sub_sims = cosine_similarity(query_matrix[i], doc_matrix[cand_idx]).ravel()
        order = np.argsort(-sub_sims)[:TOP_K]
        for rank, k in enumerate(order, start=1):
            j = cand_idx[k]
            rows.append({
                "task_id": task["task_id"],
                "condition": CONDITION,
                "rank": rank,
                "record_id": records[j]["record_id"],
                "score": float(sub_sims[k]),
            })
    out = RESULTS_DIR / "rankings_c2_metadata_filtered.csv"
    write_rankings(rows, out)
    print(f"wrote -> {out}")
    print(f"mean candidate-set size after filtering: {np.mean(cand_counts):.1f} "
          f"(min={min(cand_counts)}, max={max(cand_counts)})")
    # Also store candidate-set sizes for the paper
    import csv as _csv
    with (RESULTS_DIR / "c2_candidate_set_sizes.csv").open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["task_id", "candidate_set_size"])
        for task, n in zip(tasks, cand_counts):
            w.writerow([task["task_id"], n])


if __name__ == "__main__":
    main()
