"""Condition C3: hybrid metadata filtering + text-similarity ranking.

Step 1: metadata filter on technology_domain (looser than C2).
Step 2: rank surviving candidates using TF-IDF cosine.
Step 3: apply modest diagnostic-feature boosts for matches on incident_type,
        severity, and region.

"Hybrid" here means combining metadata filtering with text-similarity ranking
and diagnostic re-ranking; it does not denote dense-embedding or
transformer-based semantic retrieval. Re-ranking weights are intentionally
modest so the hybrid retains some of the false-positive risk of free-text
similarity.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import RESULTS_DIR, load_records, load_tasks, record_text, write_rankings

CONDITION = "C3_hybrid"
TOP_K = 10
BOOSTS = {
    "incident_type": 0.12,
    "severity": 0.04,
    "region": 0.04,
}


def _boost(rec: dict, task: dict) -> float:
    """Diagnostic-feature re-rank weights. We deliberately exclude
    ``affected_system`` and ``root_cause`` because in real NOC searches
    operators rarely know the exact system identifier or root cause in
    advance; using them as boosts would also amount to ground-truth
    leakage in this synthetic setup. Boosts on incident_type, severity,
    and region re-rank along diagnostic axes without uniquely identifying
    the target record."""
    score = 0.0
    if rec["incident_type"] == task["incident_type"]:
        score += BOOSTS["incident_type"]
    if rec["severity"] == task["severity"]:
        score += BOOSTS["severity"]
    if rec["region"] in {task["region"], "All"}:
        score += BOOSTS["region"]
    return score


def main() -> None:
    records = load_records()
    tasks = load_tasks()
    corpus_texts = [record_text(r) for r in records]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
    doc_matrix = vectorizer.fit_transform(corpus_texts)
    query_texts = [t["scenario"] for t in tasks]
    query_matrix = vectorizer.transform(query_texts)
    rows = []
    for i, task in enumerate(tasks):
        dom = task["technology_domain"]
        cand_idx = [j for j, r in enumerate(records) if r["technology_domain"] == dom]
        sub_sims = cosine_similarity(query_matrix[i], doc_matrix[cand_idx]).ravel()
        boosts = np.array([_boost(records[j], task) for j in cand_idx])
        combined = sub_sims + boosts
        order = np.argsort(-combined)[:TOP_K]
        for rank, k in enumerate(order, start=1):
            j = cand_idx[k]
            rows.append({
                "task_id": task["task_id"],
                "condition": CONDITION,
                "rank": rank,
                "record_id": records[j]["record_id"],
                "score": float(combined[k]),
            })
    out = RESULTS_DIR / "rankings_c3_hybrid.csv"
    write_rankings(rows, out)
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
