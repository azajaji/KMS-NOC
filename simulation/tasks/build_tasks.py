"""Generate 40 incident retrieval tasks against the synthetic NOC corpus.

Each task has:
  - one ground-truth incident closure record
  - 2-5 secondary-relevant records (same domain+incident_type, different specifics)
  - a free-text scenario paraphrasing the description (so BM25 is possible but imperfect)
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

SEED = 20260527
HERE = Path(__file__).resolve().parent
CORPUS_CSV = HERE.parent / "corpus" / "synthetic_noc_corpus.csv"

PARAPHRASE_OPENINGS = [
    "Operations is reporting",
    "On shift, engineers are seeing",
    "An incident ticket has been raised for",
    "We are tracking",
    "The NMS is currently flagging",
    "A field team has escalated",
]

INCIDENT_PHRASE = {
    "Outage": "a hard service outage",
    "Alarm": "a sustained critical alarm",
    "Degradation": "service degradation",
    "Configuration Issue": "post-change instability",
    "Performance Issue": "performance issues",
    "Connectivity Loss": "loss of connectivity",
    "Polling Failure": "intermittent polling failure",
    "Synchronization Failure": "a synchronization problem",
}

DIAGNOSTIC_HINTS = {
    "Configuration Error": "shortly after a configuration change",
    "Hardware Failure": "after a card replacement that appears unstable",
    "Link Failure": "with a confirmed loss of one transport leg",
    "Vendor Bug": "matching a known vendor advisory profile",
    "Power Issue": "while a power-room alarm was active",
    "Firmware Issue": "shortly after a firmware upgrade window",
    "Capacity Saturation": "under sustained peak load",
    "Human Error": "after an operator action on the element",
    "Unknown": "with no obvious recent change",
}


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_ground_truths(records: list[dict], rnd: random.Random, n: int = 40) -> list[dict]:
    """Pick 40 Incident Closure records that span domains and incident types."""
    incidents = [r for r in records if r["document_type"] == "Incident Closure"]
    rnd.shuffle(incidents)
    chosen: list[dict] = []
    seen_combos: set[tuple[str, str]] = set()
    # Pass 1: spread across (domain, incident_type)
    for r in incidents:
        key = (r["technology_domain"], r["incident_type"])
        if key in seen_combos:
            continue
        chosen.append(r)
        seen_combos.add(key)
        if len(chosen) >= n:
            break
    # Pass 2: top up if needed
    for r in incidents:
        if len(chosen) >= n:
            break
        if r in chosen:
            continue
        chosen.append(r)
    return chosen[:n]


def find_secondary(records: list[dict], gt: dict, k_max: int = 5) -> list[str]:
    """Records partially relevant to gt: same domain+incident_type but different region/system."""
    candidates = [
        r for r in records
        if r["record_id"] != gt["record_id"]
        and r["technology_domain"] == gt["technology_domain"]
        and r["incident_type"] == gt["incident_type"]
    ]
    # Prefer non-incident-closure helpers first (SOPs / lessons / vendor notes)
    helpers = [r for r in candidates if r["document_type"] != "Incident Closure"]
    closures = [r for r in candidates if r["document_type"] == "Incident Closure"]
    ordered = helpers + closures
    return [r["record_id"] for r in ordered[:k_max]]


DOMAIN_PHRASES = {
    "IP/MPLS": ["an IP backbone node", "an MPLS PE", "a core router", "an LSP-terminating element"],
    "DWDM": ["an optical transport shelf", "a ROADM site", "an OTN line card", "a DWDM amplifier"],
    "SCADA": ["a remote SCADA controller", "an RTU under our supervisory control", "a SCADA gateway", "a control-loop element"],
    "Microwave": ["a microwave hop", "an MW radio at a relay site", "a microwave link"],
    "Core Network": ["a core mobility element", "an MSC site", "a serving gateway", "a packet-core node"],
    "Access Network": ["a radio access site", "an eNodeB", "an OLT shelf", "an access aggregation point"],
    "Power Systems": ["a power-room rectifier", "a UPS plant", "a DC power bank"],
    "NMS/OSS": ["the network management platform", "an OSS collector", "the fault-management feed"],
}


def build_scenario(gt: dict, rnd: random.Random) -> str:
    """Scenario describes symptoms in everyday NOC language WITHOUT naming the
    exact affected_system identifier. The exact system identifier is supplied
    only as task metadata, so free-text retrieval cannot exploit it as a
    unique token match. Severity and region are also held back from the
    scenario in roughly half of tasks so metadata filtering provides
    discriminative value beyond surface lexical cues."""

    opening = rnd.choice(PARAPHRASE_OPENINGS)
    inc_phrase = INCIDENT_PHRASE.get(gt["incident_type"], gt["incident_type"].lower())
    hint = DIAGNOSTIC_HINTS.get(gt["root_cause"], "")
    domain_phrase = rnd.choice(DOMAIN_PHRASES[gt["technology_domain"]])

    include_region = rnd.random() < 0.5
    include_severity = rnd.random() < 0.5
    region_part = f" in the {gt['region']} region" if include_region else ""
    sev_part = f" (severity {gt['severity']})" if include_severity else ""

    text = (
        f"{opening} {inc_phrase} affecting {domain_phrase}{region_part}{sev_part}. "
        f"The behaviour began {hint}; first-line engineers want the most relevant "
        f"prior record before they escalate."
    )
    return text


def build_tasks(records: list[dict], rnd: random.Random, n: int = 40) -> list[dict]:
    gts = pick_ground_truths(records, rnd, n=n)
    tasks: list[dict] = []
    for i, gt in enumerate(gts, start=1):
        secondaries = find_secondary(records, gt, k_max=rnd.randint(2, 5))
        scenario = build_scenario(gt, rnd)
        tasks.append({
            "task_id": f"TASK-{i:03d}",
            "scenario": scenario,
            "technology_domain": gt["technology_domain"],
            "incident_type": gt["incident_type"],
            "severity": gt["severity"],
            "region": gt["region"],
            "affected_system": gt["affected_system"],
            "root_cause": gt["root_cause"],
            "ground_truth_record_id": gt["record_id"],
            "secondary_relevant_record_ids": "|".join(secondaries),
        })
    return tasks


TASK_FIELDS = [
    "task_id", "scenario", "technology_domain", "incident_type",
    "severity", "region", "affected_system", "root_cause",
    "ground_truth_record_id", "secondary_relevant_record_ids",
]

GT_FIELDS = ["task_id", "ground_truth_record_id", "secondary_relevant_record_ids"]


def write_tasks(tasks: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TASK_FIELDS)
        writer.writeheader()
        writer.writerows(tasks)


def write_ground_truth(tasks: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GT_FIELDS)
        writer.writeheader()
        for t in tasks:
            writer.writerow({k: t[k] for k in GT_FIELDS})


def main() -> None:
    rnd = random.Random(SEED)
    records = load_records(CORPUS_CSV)
    tasks = build_tasks(records, rnd, n=40)
    out_tasks = HERE / "incident_tasks.csv"
    out_gt = HERE / "ground_truth_labels.csv"
    write_tasks(tasks, out_tasks)
    write_ground_truth(tasks, out_gt)
    print(f"wrote {len(tasks)} tasks -> {out_tasks}")
    print(f"wrote ground-truth labels -> {out_gt}")


if __name__ == "__main__":
    main()
