"""Generate publication figures for the synthetic retrieval simulation."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
# Where the LaTeX build will look for them (matches \graphicspath in main.tex)
PAPER_FIG_DIR = ROOT.parent / "outputs" / "figures"

CONDITION_LABELS = {
    "C1_free_text": "Free-text\n(C1)",
    "C2_metadata_filtered": "Metadata-filtered\n(C2)",
    "C3_hybrid": "Hybrid\n(C3)",
    "C4_generic_taxonomy": "Generic taxonomy\n(C4)",
}

ORDER = ["C1_free_text", "C4_generic_taxonomy", "C2_metadata_filtered", "C3_hybrid"]


def _load_metrics() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with (RESULTS_DIR / "retrieval_metrics.csv").open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["condition"]] = {k: float(v) if k != "condition" else v for k, v in r.items()}
    return rows


def fig_retrieval_metrics() -> None:
    metrics = _load_metrics()
    labels = [CONDITION_LABELS[c] for c in ORDER]
    a1 = [metrics[c]["Accuracy@1"] for c in ORDER]
    a3 = [metrics[c]["Accuracy@3"] for c in ORDER]
    mrr = [metrics[c]["MRR"] for c in ORDER]

    x = np.arange(len(ORDER))
    width = 0.27
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.bar(x - width, a1, width, label="Accuracy@1", color="#3b6fb0")
    ax.bar(x, a3, width, label="Accuracy@3", color="#7aa6d6")
    ax.bar(x + width, mrr, width, label="MRR", color="#b6c8df")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title("Synthetic retrieval simulation: ranking metrics by condition")
    ax.legend(loc="lower right", fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "fig_retrieval_metrics.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    # Also drop into the paper figure folder so \graphicspath finds it
    paper_out = PAPER_FIG_DIR / "fig_retrieval_metrics.pdf"
    paper_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_out)
    plt.close(fig)
    print(f"wrote -> {out}")
    print(f"wrote -> {paper_out}")


def main() -> None:
    fig_retrieval_metrics()


if __name__ == "__main__":
    main()
