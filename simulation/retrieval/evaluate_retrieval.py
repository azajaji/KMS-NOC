"""Compute retrieval metrics for all conditions.

Inputs (from results/):
  rankings_c1_free_text.csv
  rankings_c2_metadata_filtered.csv
  rankings_c3_hybrid.csv
  rankings_c4_generic_taxonomy.csv  (optional)

Outputs:
  results/retrieval_metrics.csv         per-condition summary
  results/retrieval_per_task.csv        per-task per-condition reciprocal rank, hit@1/3/5
  results/retrieval_pairwise.csv        McNemar/Wilcoxon vs free-text baseline
  results/ablation_metrics.csv          metadata-field ablations (re-runs C2 with one field removed)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import RESULTS_DIR, load_records, load_tasks, record_text

TOP_K = 10
P_AT = 5
R_AT = 5

CONDITION_FILES = [
    ("C1_free_text", "rankings_c1_free_text.csv"),
    ("C2_metadata_filtered", "rankings_c2_metadata_filtered.csv"),
    ("C3_hybrid", "rankings_c3_hybrid.csv"),
    ("C4_generic_taxonomy", "rankings_c4_generic_taxonomy.csv"),
]


def load_rankings(path: Path) -> dict[str, list[str]]:
    """Return task_id -> [record_id ordered by rank]."""
    by_task: dict[str, list[tuple[int, str]]] = {}
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_task.setdefault(row["task_id"], []).append((int(row["rank"]), row["record_id"]))
    return {tid: [rid for _, rid in sorted(rows)] for tid, rows in by_task.items()}


def per_task_metrics(
    rankings: dict[str, list[str]],
    tasks: list[dict],
) -> list[dict]:
    rows = []
    for task in tasks:
        ranking = rankings.get(task["task_id"], [])
        gt = task["ground_truth_record_id"]
        relevant = set([gt, *task["secondary_relevant_record_ids"]])
        hit1 = int(bool(ranking) and ranking[0] == gt)
        hit3 = int(gt in ranking[:3])
        hit5 = int(gt in ranking[:5])
        # MRR on ground-truth only
        rr = 0.0
        for i, rid in enumerate(ranking, start=1):
            if rid == gt:
                rr = 1.0 / i
                break
        top5 = ranking[:5]
        p5 = sum(1 for rid in top5 if rid in relevant) / 5.0
        r5 = sum(1 for rid in top5 if rid in relevant) / max(1, len(relevant))
        rows.append({
            "task_id": task["task_id"],
            "hit@1": hit1, "hit@3": hit3, "hit@5": hit5,
            "rr": rr, "p@5": p5, "r@5": r5,
        })
    return rows


def aggregate(per_task: list[dict]) -> dict:
    n = len(per_task)
    return {
        "n_tasks": n,
        "Accuracy@1": np.mean([r["hit@1"] for r in per_task]),
        "Accuracy@3": np.mean([r["hit@3"] for r in per_task]),
        "Accuracy@5": np.mean([r["hit@5"] for r in per_task]),
        "MRR": np.mean([r["rr"] for r in per_task]),
        "Precision@5": np.mean([r["p@5"] for r in per_task]),
        "Recall@5": np.mean([r["r@5"] for r in per_task]),
    }


def mcnemar(a: list[int], b: list[int]) -> dict:
    """Paired binary outcomes — McNemar test on discordant pairs."""
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    disc = b10 + b01
    if disc == 0:
        return {"b10": b10, "b01": b01, "stat": 0.0, "p": 1.0}
    # Exact two-sided binomial
    k = min(b10, b01)
    p = 2.0 * stats.binom.cdf(k, disc, 0.5)
    p = min(1.0, p)
    return {"b10": b10, "b01": b01, "stat": float((b10 - b01) ** 2 / disc), "p": float(p)}


def bootstrap_ci_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs = b - a
    n = len(diffs)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = diffs[idx].mean()
    mean = float(diffs.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _filter_for_ablation(records: list[dict], task: dict, removed_field: str) -> list[int]:
    """C2-style cascade, but the named diagnostic field is removed from filtering."""
    use = lambda f: f != removed_field  # noqa: E731
    base = [i for i, r in enumerate(records)
            if (not use("technology_domain") or r["technology_domain"] == task["technology_domain"])
            and (not use("incident_type") or r["incident_type"] == task["incident_type"])]
    if not base:
        base = list(range(len(records)))
    narrowed = [i for i in base
                if (not use("severity") or records[i]["severity"] in {task["severity"], "S2"})]
    if len(narrowed) >= 6:
        base = narrowed
    narrowed = [i for i in base
                if (not use("region") or records[i]["region"] in {task["region"], "All"})]
    if len(narrowed) >= 6:
        base = narrowed
    return base


def run_ablation(records: list[dict], tasks: list[dict]) -> list[dict]:
    corpus_texts = [record_text(r) for r in records]
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, ngram_range=(1, 2))
    doc_matrix = vectorizer.fit_transform(corpus_texts)
    query_texts = [t["scenario"] for t in tasks]
    query_matrix = vectorizer.transform(query_texts)

    abl_fields = ["none", "technology_domain", "incident_type", "severity", "region"]
    rows = []
    for removed in abl_fields:
        per_task = []
        for i, task in enumerate(tasks):
            cand_idx = _filter_for_ablation(records, task, removed)
            sub_sims = cosine_similarity(query_matrix[i], doc_matrix[cand_idx]).ravel()
            order = np.argsort(-sub_sims)[:TOP_K]
            ranking = [records[cand_idx[k]]["record_id"] for k in order]
            gt = task["ground_truth_record_id"]
            relevant = set([gt, *task["secondary_relevant_record_ids"]])
            hit1 = int(bool(ranking) and ranking[0] == gt)
            hit3 = int(gt in ranking[:3])
            rr = 0.0
            for j2, rid in enumerate(ranking, start=1):
                if rid == gt:
                    rr = 1.0 / j2
                    break
            per_task.append({"hit@1": hit1, "hit@3": hit3, "rr": rr})
        rows.append({
            "removed_field": removed,
            "Accuracy@1": np.mean([r["hit@1"] for r in per_task]),
            "Accuracy@3": np.mean([r["hit@3"] for r in per_task]),
            "MRR": np.mean([r["rr"] for r in per_task]),
        })
    return rows


def main() -> None:
    tasks = load_tasks()
    records = load_records()

    summary_rows = []
    per_task_long = []
    rr_by_condition: dict[str, np.ndarray] = {}
    hit1_by_condition: dict[str, list[int]] = {}

    for cond, fname in CONDITION_FILES:
        rankings = load_rankings(RESULTS_DIR / fname)
        if not rankings:
            print(f"skip {cond}: {fname} not found")
            continue
        per_task = per_task_metrics(rankings, tasks)
        agg = aggregate(per_task)
        agg["condition"] = cond
        summary_rows.append(agg)
        for row in per_task:
            row2 = dict(row)
            row2["condition"] = cond
            per_task_long.append(row2)
        rr_by_condition[cond] = np.array([r["rr"] for r in per_task])
        hit1_by_condition[cond] = [r["hit@1"] for r in per_task]

    fields = ["condition", "n_tasks", "Accuracy@1", "Accuracy@3", "Accuracy@5", "MRR", "Precision@5", "Recall@5"]
    write_csv([{k: r[k] for k in fields} for r in summary_rows], RESULTS_DIR / "retrieval_metrics.csv", fields)

    write_csv(per_task_long, RESULTS_DIR / "retrieval_per_task.csv",
              ["condition", "task_id", "hit@1", "hit@3", "hit@5", "rr", "p@5", "r@5"])

    # Pairwise vs C1
    baseline = "C1_free_text"
    pairwise = []
    if baseline in hit1_by_condition:
        b_hits = hit1_by_condition[baseline]
        b_rr = rr_by_condition[baseline]
        for cond in hit1_by_condition:
            if cond == baseline:
                continue
            mc = mcnemar(b_hits, hit1_by_condition[cond])
            try:
                w = stats.wilcoxon(b_rr, rr_by_condition[cond], zero_method="wilcox", alternative="two-sided")
                w_stat, w_p = float(w.statistic), float(w.pvalue)
            except ValueError:
                w_stat, w_p = math.nan, math.nan
            mean_d, lo, hi = bootstrap_ci_diff(b_rr, rr_by_condition[cond])
            pairwise.append({
                "comparison": f"{cond} vs {baseline}",
                "mcnemar_b10": mc["b10"], "mcnemar_b01": mc["b01"],
                "mcnemar_p": mc["p"],
                "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
                "mean_MRR_diff": mean_d,
                "MRR_diff_ci_low": lo,
                "MRR_diff_ci_high": hi,
            })
    write_csv(pairwise, RESULTS_DIR / "retrieval_pairwise.csv",
              ["comparison", "mcnemar_b10", "mcnemar_b01", "mcnemar_p",
               "wilcoxon_stat", "wilcoxon_p", "mean_MRR_diff",
               "MRR_diff_ci_low", "MRR_diff_ci_high"])

    # Ablation
    abl = run_ablation(records, tasks)
    write_csv(abl, RESULTS_DIR / "ablation_metrics.csv",
              ["removed_field", "Accuracy@1", "Accuracy@3", "MRR"])

    # Pretty print
    print("\nRetrieval metrics:")
    for row in summary_rows:
        print(f"  {row['condition']}: A@1={row['Accuracy@1']:.3f} "
              f"A@3={row['Accuracy@3']:.3f} A@5={row['Accuracy@5']:.3f} "
              f"MRR={row['MRR']:.3f} P@5={row['Precision@5']:.3f} R@5={row['Recall@5']:.3f}")
    if pairwise:
        print("\nPairwise vs free-text baseline:")
        for row in pairwise:
            print(f"  {row['comparison']}: McNemar p={row['mcnemar_p']:.4f}, "
                  f"Wilcoxon p={row['wilcoxon_p']:.4f}, mean MRR diff={row['mean_MRR_diff']:+.3f} "
                  f"[{row['MRR_diff_ci_low']:+.3f}, {row['MRR_diff_ci_high']:+.3f}]")
    print("\nAblation:")
    for row in abl:
        print(f"  removed={row['removed_field']}: A@1={row['Accuracy@1']:.3f} "
              f"A@3={row['Accuracy@3']:.3f} MRR={row['MRR']:.3f}")


if __name__ == "__main__":
    main()
