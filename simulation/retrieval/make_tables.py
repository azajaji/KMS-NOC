"""Emit LaTeX tables for the synthetic retrieval simulation results.

Writes to the paper's outputs/tables directory so main.tex can \\input them.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
PAPER_TABLES_DIR = ROOT.parent / "outputs" / "tables"

ORDER = [
    ("C1_free_text", "Free-text baseline (C1)"),
    ("C4_generic_taxonomy", "Generic taxonomy (C4)"),
    ("C2_metadata_filtered", "Metadata-filtered (C2)"),
    ("C3_hybrid", "Hybrid metadata + text ranking (C3)"),
]


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt(x: str | float, digits: int = 3) -> str:
    if isinstance(x, str):
        try:
            x = float(x)
        except ValueError:
            return x
    return f"{x:.{digits}f}"


def table_main_results() -> None:
    ret = {r["condition"]: r for r in _load_csv(RESULTS_DIR / "retrieval_metrics.csv")}
    sel_path = RESULTS_DIR / "selector_metrics.csv"
    cla = {r["condition"]: r for r in _load_csv(sel_path)} if sel_path.exists() else {}

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(r"  \footnotesize")
    lines.append(r"  \caption{Controlled synthetic retrieval simulation: retrieval and deterministic-selector metrics by condition (n=40 tasks). The selector column reflects a rule-based selection over the top-5 candidates per condition and should be read as a candidate-set utility measure.}")
    lines.append(r"  \label{tab:sim_main}")
    lines.append(r"  \resizebox{\textwidth}{!}{%")
    lines.append(r"  \begin{tabular}{lccccccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Retrieval condition} & \textbf{Acc@1} & \textbf{Acc@3} & \textbf{Acc@5} & \textbf{MRR} & \textbf{P@5} & \textbf{R@5} & \textbf{Selector Acc.} \\")
    lines.append(r"    \midrule")
    for key, label in ORDER:
        r = ret.get(key, {})
        c = cla.get(key, {})
        lines.append(
            "    " + label + " & "
            + _fmt(r.get("Accuracy@1", "")) + " & "
            + _fmt(r.get("Accuracy@3", "")) + " & "
            + _fmt(r.get("Accuracy@5", "")) + " & "
            + _fmt(r.get("MRR", "")) + " & "
            + _fmt(r.get("Precision@5", "")) + " & "
            + _fmt(r.get("Recall@5", "")) + " & "
            + (_fmt(c.get("selection_accuracy_lenient", ""), 3) if c else "---") + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"\end{table}")
    out = PAPER_TABLES_DIR / "table_sim_main.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote -> {out}")


def table_pairwise() -> None:
    rows = _load_csv(RESULTS_DIR / "retrieval_pairwise.csv")
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(r"  \footnotesize")
    lines.append(r"  \caption{Pairwise comparisons against the free-text baseline (McNemar on Accuracy@1; Wilcoxon and bootstrap CI on reciprocal rank).}")
    lines.append(r"  \label{tab:sim_pairwise}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Comparison} & \textbf{McNemar $p$} & \textbf{Wilcoxon $p$} & \textbf{Mean MRR diff.} & \textbf{95\% CI} \\")
    lines.append(r"    \midrule")
    for r in rows:
        comp = r["comparison"].replace("C1_free_text", "C1").replace("C2_metadata_filtered", "C2").replace("C3_hybrid", "C3").replace("C4_generic_taxonomy", "C4")
        lines.append(
            "    " + comp + " & "
            + _fmt(r["mcnemar_p"], 4) + " & "
            + _fmt(r["wilcoxon_p"], 4) + " & "
            + ("+" if float(r["mean_MRR_diff"]) >= 0 else "") + _fmt(r["mean_MRR_diff"], 3) + " & "
            + "[" + _fmt(r["MRR_diff_ci_low"], 3) + ", " + _fmt(r["MRR_diff_ci_high"], 3) + "]"
            + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    out = PAPER_TABLES_DIR / "table_sim_pairwise.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote -> {out}")


def table_ablation() -> None:
    rows = _load_csv(RESULTS_DIR / "ablation_metrics.csv")
    interp = {
        "none": "Full diagnostic schema (C2)",
        "technology_domain": "Removes domain narrowing",
        "incident_type": "Removes incident-type narrowing",
        "severity": "Removes criticality filter",
        "region": "Removes localization filter",
    }
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(r"  \footnotesize")
    lines.append(r"  \caption{Metadata-field ablation on Condition C2. Each row drops one diagnostic field from the cascade and re-ranks the surviving candidates with TF--IDF.}")
    lines.append(r"  \label{tab:sim_ablation}")
    lines.append(r"  \begin{tabular}{lcccp{4.5cm}}")
    lines.append(r"    \toprule")
    lines.append(r"    \textbf{Removed field} & \textbf{Acc@1} & \textbf{Acc@3} & \textbf{MRR} & \textbf{Interpretation} \\")
    lines.append(r"    \midrule")
    for r in rows:
        label = "None" if r["removed_field"] == "none" else r["removed_field"].replace("_", "\\_")
        lines.append(
            "    " + label + " & "
            + _fmt(r["Accuracy@1"]) + " & "
            + _fmt(r["Accuracy@3"]) + " & "
            + _fmt(r["MRR"]) + " & "
            + interp.get(r["removed_field"], "") + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    out = PAPER_TABLES_DIR / "table_sim_ablation.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote -> {out}")


def main() -> None:
    table_main_results()
    table_pairwise()
    table_ablation()


if __name__ == "__main__":
    main()
