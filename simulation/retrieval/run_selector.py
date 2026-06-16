"""Deterministic candidate selector for the synthetic retrieval simulation.

For each (task, retrieval-condition) pair, the selector takes the top-ranked
candidate from that condition's ranking and reports whether it is the exact
ground-truth record (strict) or any accepted record in the task's relevance set
(lenient). Because the selector always picks the rank-1 candidate, strict
selection accuracy tracks the underlying retrieval Accuracy@1; the lenient
metric captures how often the top candidate is at least a related record in the
same diagnostic cell. The result is a candidate-set utility measure and is fully
deterministic (no model, no network).

Outputs:
  results/selector_metrics.csv   - per-condition strict/lenient selection accuracy
  results/selector_per_task.csv  - per-task, per-condition selection outcomes
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TASKS_CSV = ROOT / "tasks" / "incident_tasks.csv"

CONDITION_FILES = [
    ("C1_free_text", "rankings_c1_free_text.csv"),
    ("C2_metadata_filtered", "rankings_c2_metadata_filtered.csv"),
    ("C3_hybrid", "rankings_c3_hybrid.csv"),
    ("C4_generic_taxonomy", "rankings_c4_generic_taxonomy.csv"),
]


def load_tasks() -> dict[str, dict]:
    with TASKS_CSV.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        secondary = (
            r["secondary_relevant_record_ids"].split("|")
            if r["secondary_relevant_record_ids"] else []
        )
        r["accepted"] = {r["ground_truth_record_id"], *secondary}
    return {r["task_id"]: r for r in rows}


def load_top_candidate(path: Path) -> dict[str, str]:
    """Return the rank-1 record_id per task for a ranking file."""
    best: dict[str, tuple[int, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rank = int(row["rank"])
            tid = row["task_id"]
            if tid not in best or rank < best[tid][0]:
                best[tid] = (rank, row["record_id"])
    return {tid: rid for tid, (_, rid) in best.items()}


def main() -> None:
    tasks = load_tasks()
    per_task_rows = []
    agg = []
    for cond, fname in CONDITION_FILES:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"skip {cond}: {fname} missing")
            continue
        top = load_top_candidate(path)
        strict_hits, lenient_hits, n = 0, 0, 0
        for tid, task in tasks.items():
            sel = top.get(tid, "")
            gt = task["ground_truth_record_id"]
            strict = int(sel == gt)
            lenient = int(sel in task["accepted"]) if task["accepted"] else strict
            strict_hits += strict
            lenient_hits += lenient
            n += 1
            per_task_rows.append({
                "task_id": tid,
                "condition": cond,
                "selected_record_id": sel,
                "ground_truth_record_id": gt,
                "selection_correct_strict": strict,
                "selection_correct_lenient": lenient,
            })
        agg.append({
            "condition": cond,
            "n": n,
            "selection_accuracy_strict": round(strict_hits / n, 4) if n else 0.0,
            "selection_accuracy_lenient": round(lenient_hits / n, 4) if n else 0.0,
        })

    out_metrics = RESULTS_DIR / "selector_metrics.csv"
    with out_metrics.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "n",
                                          "selection_accuracy_strict",
                                          "selection_accuracy_lenient"])
        w.writeheader()
        w.writerows(agg)

    out_per_task = RESULTS_DIR / "selector_per_task.csv"
    with out_per_task.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_task_rows[0].keys()))
        w.writeheader()
        w.writerows(per_task_rows)

    print("Deterministic selector candidate-selection accuracy:")
    for row in agg:
        print(f"  {row['condition']}: strict={row['selection_accuracy_strict']:.3f} "
              f"lenient={row['selection_accuracy_lenient']:.3f}")


if __name__ == "__main__":
    main()
