"""Shared loaders and helpers for retrieval scripts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
CORPUS_CSV = ROOT / "corpus" / "synthetic_noc_corpus.csv"
TASKS_CSV = ROOT / "tasks" / "incident_tasks.csv"
RESULTS_DIR = ROOT / "results"


def load_records(path: Path = CORPUS_CSV) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_tasks(path: Path = TASKS_CSV) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        tasks = list(csv.DictReader(f))
    for t in tasks:
        t["secondary_relevant_record_ids"] = (
            t["secondary_relevant_record_ids"].split("|") if t["secondary_relevant_record_ids"] else []
        )
    return tasks


def record_text(rec: dict) -> str:
    """Concatenated free-text representation used for TF-IDF scoring."""
    parts = [
        rec.get("title", ""),
        rec.get("description", ""),
        rec.get("resolution_steps", ""),
        rec.get("lessons_learned", ""),
        rec.get("tags", "").replace("|", " "),
    ]
    return " ".join(parts)


def write_rankings(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields = ["task_id", "condition", "rank", "record_id", "score"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
