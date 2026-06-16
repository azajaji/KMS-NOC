# Synthetic NOC KMS — Reproducibility Repository

Reproducibility materials for the manuscript *"A Workflow-Integrated Knowledge
Management System for Telecom Network Operations Centers: A Socio-Technical
Design Science Study"* (submitted to MDPI *Systems*).

This repository reproduces the **controlled synthetic candidate-set retrieval
simulation** reported in the Materials and Methods and Results sections. The
simulation is a **mechanism test of candidate-set construction**: it tests
whether the proposed nine-field diagnostic metadata schema improves candidate
retrieval when ground-truth relevance is known, holding the ranking algorithm
(TF–IDF cosine similarity) constant and varying only the retrieval condition
(C1 free-text, C2 metadata-filtered, C3 hybrid, C4 generic-taxonomy control).
Secondary relevance labels are defined by diagnostic-cell membership (same
technology-domain / incident-type cell), so the results concern diagnostic-cell
candidate-set construction, not open-domain semantic retrieval. Everything here
is deterministic (fixed seeds, no model, no network).

## What is and is not included

**Included** (fully synthetic / non-identifying):
- the 300-record synthetic NOC knowledge corpus, 40 retrieval tasks, and
  relevance labels;
- the deterministic generators (fixed seeds), the four retrieval conditions,
  the metric/ablation evaluation, and the deterministic candidate selector;
- the generated result CSVs and the ranking-metrics figure;
- the pilot KPI survey instrument items (`survey/survey_instrument.csv`).

**Not included** (confidentiality):
- host operational documents, audit logs, incident records, platform naming,
  and any organization-identifying details;
- survey **respondent-level data** (only the instrument items are provided).

## Layout

```
.
├── README.md
├── LICENSE                         # MIT
├── requirements.txt
├── simulation/
│   ├── corpus/                     # build_corpus.py + synthetic_noc_corpus.csv
│   ├── tasks/                      # build_tasks.py + tasks + ground-truth labels
│   ├── retrieval/                  # C1–C4 runners, evaluate_retrieval.py,
│   │                               #   run_selector.py, make_tables.py, make_figures.py
│   └── results/                    # generated CSVs and figure
└── survey/
    └── survey_instrument.csv       # pilot KPI instrument items (no respondent data)
```

## Reproducing the results

```bash
pip install -r requirements.txt
cd simulation

# 1. Regenerate corpus, tasks, and labels (bit-identical under fixed seeds)
python corpus/build_corpus.py
python tasks/build_tasks.py

# 2. Run the four retrieval conditions
python retrieval/run_tfidf_baseline.py
python retrieval/run_metadata_filter.py
python retrieval/run_hybrid_retrieval.py
python retrieval/run_generic_taxonomy.py

# 3. Compute retrieval metrics, pairwise tests, and the metadata ablation
python retrieval/evaluate_retrieval.py

# 4. Compute the deterministic candidate-selector accuracy
python retrieval/run_selector.py

# 5. Regenerate tables and figure
python retrieval/make_tables.py
python retrieval/make_figures.py
```

All random seeds are fixed, so the corpus, task set, and label set regenerate
deterministically and the reported metrics reproduce exactly. The candidate
selector picks the rank-1 candidate per condition, so its strict accuracy tracks
the underlying Accuracy@1; it is reported as a candidate-set utility measure.

## Citation

Please cite the manuscript once published.
