"""Condition C1: free-text baseline using TF-IDF cosine similarity."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import RESULTS_DIR, load_records, load_tasks, record_text, write_rankings

CONDITION = "C1_free_text"
TOP_K = 10


def main() -> None:
    records = load_records()
    tasks = load_tasks()
    corpus_texts = [record_text(r) for r in records]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
    doc_matrix = vectorizer.fit_transform(corpus_texts)
    query_texts = [t["scenario"] for t in tasks]
    query_matrix = vectorizer.transform(query_texts)
    sims = cosine_similarity(query_matrix, doc_matrix)
    rows = []
    for i, task in enumerate(tasks):
        order = np.argsort(-sims[i])[:TOP_K]
        for rank, j in enumerate(order, start=1):
            rows.append({
                "task_id": task["task_id"],
                "condition": CONDITION,
                "rank": rank,
                "record_id": records[j]["record_id"],
                "score": float(sims[i, j]),
            })
    out = RESULTS_DIR / "rankings_c1_free_text.csv"
    write_rankings(rows, out)
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
