"""Optional Condition C4: generic taxonomy filter.

Uses only document_type as the filter (Incident Closure / SOP / etc.). All tasks
are looking for prior incidents, so we filter to Incident Closure + Lessons
Learned. No diagnostic metadata is used. Surviving candidates are ranked by
TF-IDF cosine.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import RESULTS_DIR, load_records, load_tasks, record_text, write_rankings

CONDITION = "C4_generic_taxonomy"
TOP_K = 10
ALLOWED_TYPES = {"Incident Closure", "Lessons Learned"}


def main() -> None:
    records = load_records()
    tasks = load_tasks()
    cand_idx = [i for i, r in enumerate(records) if r["document_type"] in ALLOWED_TYPES]
    corpus_texts = [record_text(records[i]) for i in cand_idx]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
    doc_matrix = vectorizer.fit_transform(corpus_texts)
    query_texts = [t["scenario"] for t in tasks]
    query_matrix = vectorizer.transform(query_texts)
    sims = cosine_similarity(query_matrix, doc_matrix)
    rows = []
    for i, task in enumerate(tasks):
        order = np.argsort(-sims[i])[:TOP_K]
        for rank, k in enumerate(order, start=1):
            j = cand_idx[k]
            rows.append({
                "task_id": task["task_id"],
                "condition": CONDITION,
                "rank": rank,
                "record_id": records[j]["record_id"],
                "score": float(sims[i, k]),
            })
    out = RESULTS_DIR / "rankings_c4_generic_taxonomy.csv"
    write_rankings(rows, out)
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
