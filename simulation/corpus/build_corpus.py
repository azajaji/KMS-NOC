"""Generate Synthetic NOC Knowledge Corpus v1.0.

Deterministic given the seed. Produces:
  synthetic_noc_corpus.csv  - 300 records across four document types
  example_records.json      - 5 illustrative records pulled from the CSV
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

SEED = 20260526
HERE = Path(__file__).resolve().parent

TECHNOLOGY_DOMAINS = [
    "IP/MPLS", "DWDM", "SCADA", "Microwave",
    "Core Network", "Access Network", "Power Systems", "NMS/OSS",
]
INCIDENT_TYPES = [
    "Outage", "Alarm", "Degradation", "Configuration Issue",
    "Performance Issue", "Connectivity Loss", "Polling Failure", "Synchronization Failure",
]
SEVERITIES = ["S1", "S2", "S3"]
REGIONS = ["Central", "Western", "Eastern", "Northern", "Southern"]
ROOT_CAUSES = [
    "Configuration Error", "Hardware Failure", "Link Failure", "Vendor Bug",
    "Power Issue", "Firmware Issue", "Capacity Saturation", "Human Error", "Unknown",
]
RESOLUTION_STATUSES = ["Resolved", "Workaround", "Under Review", "Archived"]

# domain -> realistic system identifiers
SYSTEMS_BY_DOMAIN = {
    "IP/MPLS": ["Router R-{n}", "PE-Router PE-{n}", "Core Router CR-{n}"],
    "DWDM": ["DWDM Node D-{n}", "OTN Shelf OTN-{n}", "ROADM R-{n}"],
    "SCADA": ["Controller C-{n}", "RTU R-{n}", "SCADA Gateway G-{n}"],
    "Microwave": ["MW Link MW-{n}", "Microwave Hop H-{n}", "MW Radio MR-{n}"],
    "Core Network": ["MSC M-{n}", "MME E-{n}", "S-GW S-{n}"],
    "Access Network": ["BTS B-{n}", "eNodeB E-{n}", "OLT O-{n}"],
    "Power Systems": ["Rectifier P-{n}", "UPS U-{n}", "Battery Bank BB-{n}"],
    "NMS/OSS": ["NMS Node N-{n}", "OSS Collector OC-{n}", "Fault Manager FM-{n}"],
}

# domain -> typical owner team
OWNERS_BY_DOMAIN = {
    "IP/MPLS": "IP Backbone Team",
    "DWDM": "Optical Transport Team",
    "SCADA": "SCADA Operations Team",
    "Microwave": "Microwave Team",
    "Core Network": "Core Mobility Team",
    "Access Network": "Access Engineering Team",
    "Power Systems": "Power Infrastructure Team",
    "NMS/OSS": "NMS Engineering Team",
}

# domain -> base technical vocabulary used to colour descriptions
VOCAB_BY_DOMAIN = {
    "IP/MPLS": ["BGP session", "LSP", "MPLS label", "OSPF adjacency", "VRF", "routing convergence", "BFD"],
    "DWDM": ["optical channel", "OSNR", "amplifier gain", "wavelength", "ROADM port", "dispersion"],
    "SCADA": ["polling interval", "telemetry", "RTU heartbeat", "Modbus register", "DNP3 frame", "control loop"],
    "Microwave": ["radio link", "RSL", "modulation", "ATPC", "ACM", "antenna alignment"],
    "Core Network": ["GTP tunnel", "diameter session", "PDP context", "attach procedure", "bearer setup"],
    "Access Network": ["RF carrier", "PRACH", "UE attach", "PON splitter", "uplink throughput"],
    "Power Systems": ["DC bus", "rectifier output", "battery autonomy", "alarm contact", "AC mains"],
    "NMS/OSS": ["alarm correlation", "trap forwarding", "northbound feed", "polling cycle", "inventory sync"],
}

# Symptom phrases per incident type (used in description)
SYMPTOMS_BY_INCIDENT = {
    "Outage": "complete service loss with all monitored interfaces unreachable",
    "Alarm": "persistent critical alarm with no traffic impact yet reported",
    "Degradation": "throughput degradation with rising error counters",
    "Configuration Issue": "post-change misbehaviour with operators reporting unexpected state",
    "Performance Issue": "latency spikes and increasing queue depth under steady load",
    "Connectivity Loss": "loss of reachability between segments while interfaces remain up",
    "Polling Failure": "intermittent telemetry loss and repeated timeout alarms in the NMS",
    "Synchronization Failure": "clock drift across nodes with sync source losing lock",
}

# Resolution snippets per (incident_type, root_cause)
RESOLUTION_TEMPLATES = {
    "Configuration Error": "Roll back the most recent change, re-validate the parameter set in the lab profile, then re-apply with peer review.",
    "Hardware Failure": "Swap the faulty card, verify inventory, run vendor diagnostic, and confirm clean state for 15 minutes.",
    "Link Failure": "Bring the protection path into service, dispatch field team to inspect fibre/RF, and validate end-to-end after splice/realignment.",
    "Vendor Bug": "Apply the vendor-supplied workaround, raise a TAC case, and schedule the patched release in the next change window.",
    "Power Issue": "Confirm rectifier health, restore AC mains via genset if required, and verify battery autonomy before clearing.",
    "Firmware Issue": "Downgrade to the last known-good firmware, freeze further upgrades, and validate counters before reattempting.",
    "Capacity Saturation": "Shed non-critical load, re-balance traffic, and submit a capacity expansion request through change.",
    "Human Error": "Reverse the operator action, retrain on the standard operating procedure, and add a procedural guard rail.",
    "Unknown": "Capture diagnostic bundle, escalate to L3 engineering, and monitor under controlled conditions.",
}

LESSONS_TEMPLATES = [
    "Validate configuration changes in a test profile before production deployment.",
    "Add a pre-change checklist item for this parameter family.",
    "Tighten the stale-record review threshold for this technology domain.",
    "Record vendor case number in the closure form so future searches resolve faster.",
    "Cross-link this record to the affected SOP to shorten retrieval during similar incidents.",
    "Update the diagnostic playbook with the first-action step used here.",
]


def _system_for(domain: str, rnd: random.Random) -> str:
    return rnd.choice(SYSTEMS_BY_DOMAIN[domain]).format(n=rnd.randint(1, 99))


def _description(domain: str, incident_type: str, system: str, region: str, rnd: random.Random) -> str:
    sym = SYMPTOMS_BY_INCIDENT[incident_type]
    vocab = rnd.sample(VOCAB_BY_DOMAIN[domain], k=2)
    return (
        f"A {domain} {incident_type.lower()} was observed on {system} in the {region} region. "
        f"Operations reported {sym}. Key indicators included {vocab[0]} behaviour and {vocab[1]} anomalies. "
        f"The condition persisted across two polling cycles before triage began."
    )


def _resolution(root_cause: str) -> str:
    return RESOLUTION_TEMPLATES[root_cause]


def _lessons(rnd: random.Random) -> str:
    return rnd.choice(LESSONS_TEMPLATES)


def _tags(domain: str, incident_type: str, root_cause: str) -> list[str]:
    base = [domain.lower().replace("/", "-"), incident_type.lower().replace(" ", "-"), root_cause.lower().split()[0]]
    return list(dict.fromkeys(base))  # dedupe, preserve order


def _date(rnd: random.Random) -> str:
    year = rnd.choice([2024, 2025, 2026])
    month = rnd.randint(1, 12)
    day = rnd.randint(1, 28)
    return f"{year:04d}-{month:02d}-{day:02d}"


def build_incident_closures(rnd: random.Random, n: int = 180) -> list[dict]:
    """One Incident Closure per row. Cover the (domain x incident_type) matrix
    multiple times so the metadata schema has discriminative power."""
    records = []
    idx = 0
    # Cover the full matrix at least once
    combos = [(d, i) for d in TECHNOLOGY_DOMAINS for i in INCIDENT_TYPES]
    rnd.shuffle(combos)
    # n records, cycling through combos
    while len(records) < n:
        for (domain, incident_type) in combos:
            if len(records) >= n:
                break
            severity = rnd.choices(SEVERITIES, weights=[1, 2, 3])[0]
            region = rnd.choice(REGIONS)
            system = _system_for(domain, rnd)
            root_cause = rnd.choice(ROOT_CAUSES)
            status = rnd.choices(RESOLUTION_STATUSES, weights=[7, 2, 1, 1])[0]
            idx += 1
            rid = f"INC-{idx:04d}"
            title = f"{incident_type} on {system} ({domain}, {region})"
            description = _description(domain, incident_type, system, region, rnd)
            resolution = _resolution(root_cause)
            lessons = _lessons(rnd)
            records.append({
                "record_id": rid,
                "title": title,
                "document_type": "Incident Closure",
                "technology_domain": domain,
                "incident_type": incident_type,
                "severity": severity,
                "region": region,
                "affected_system": system,
                "root_cause": root_cause,
                "resolution_status": status,
                "document_owner": OWNERS_BY_DOMAIN[domain],
                "last_updated": _date(rnd),
                "description": description,
                "resolution_steps": resolution,
                "lessons_learned": lessons,
                "tags": "|".join(_tags(domain, incident_type, root_cause)),
            })
    return records


def build_sops(rnd: random.Random, n: int = 50) -> list[dict]:
    records = []
    combos = [(d, i) for d in TECHNOLOGY_DOMAINS for i in INCIDENT_TYPES]
    rnd.shuffle(combos)
    for k in range(n):
        domain, incident_type = combos[k % len(combos)]
        rid = f"SOP-{k+1:04d}"
        title = f"SOP: handling {incident_type.lower()} in {domain}"
        description = (
            f"Standard operating procedure for diagnosing and resolving {incident_type.lower()} "
            f"events in the {domain} domain. Covers triage, escalation, and closure steps."
        )
        resolution = (
            f"Step 1: confirm alarm scope. Step 2: classify by technology and severity. "
            f"Step 3: apply the {VOCAB_BY_DOMAIN[domain][0]} check. Step 4: escalate per matrix if unresolved in 10 minutes."
        )
        records.append({
            "record_id": rid,
            "title": title,
            "document_type": "SOP",
            "technology_domain": domain,
            "incident_type": incident_type,
            "severity": "S2",
            "region": "All",
            "affected_system": "All",
            "root_cause": "Procedural",
            "resolution_status": "Resolved",
            "document_owner": OWNERS_BY_DOMAIN[domain],
            "last_updated": _date(rnd),
            "description": description,
            "resolution_steps": resolution,
            "lessons_learned": "Refer to the SOP first before opening a new incident closure record.",
            "tags": "|".join([domain.lower().replace("/", "-"), incident_type.lower().replace(" ", "-"), "sop"]),
        })
    return records


def build_vendor_notes(rnd: random.Random, n: int = 40) -> list[dict]:
    records = []
    combos = [(d, rnd.choice(ROOT_CAUSES)) for d in TECHNOLOGY_DOMAINS for _ in range(6)]
    rnd.shuffle(combos)
    for k in range(n):
        domain, cause = combos[k % len(combos)]
        rid = f"VND-{k+1:04d}"
        title = f"Vendor note: {cause.lower()} workaround for {domain}"
        description = (
            f"Vendor-published workaround for {cause.lower()} affecting {domain} elements. "
            f"Applicable to firmware ranges shipped between 2024 and 2026."
        )
        resolution = (
            f"Apply the vendor-supplied {VOCAB_BY_DOMAIN[domain][0]} workaround and schedule the fixed release."
        )
        records.append({
            "record_id": rid,
            "title": title,
            "document_type": "Vendor Note",
            "technology_domain": domain,
            "incident_type": "Configuration Issue",
            "severity": "S3",
            "region": "All",
            "affected_system": "Vendor-wide",
            "root_cause": cause,
            "resolution_status": "Workaround",
            "document_owner": OWNERS_BY_DOMAIN[domain],
            "last_updated": _date(rnd),
            "description": description,
            "resolution_steps": resolution,
            "lessons_learned": "Cross-link this note to incident closures that reference the same firmware family.",
            "tags": "|".join([domain.lower().replace("/", "-"), "vendor", cause.lower().split()[0]]),
        })
    return records


def build_lessons_learned(rnd: random.Random, n: int = 30) -> list[dict]:
    records = []
    combos = [(d, i) for d in TECHNOLOGY_DOMAINS for i in INCIDENT_TYPES]
    rnd.shuffle(combos)
    for k in range(n):
        domain, incident_type = combos[k % len(combos)]
        rid = f"LL-{k+1:04d}"
        title = f"Lessons learned: {incident_type.lower()} in {domain} ({rnd.choice(REGIONS)})"
        description = (
            f"Retrospective summary of recent {incident_type.lower()} events in the {domain} domain. "
            f"Captures recurring failure patterns and the procedural changes adopted after closure review."
        )
        resolution = "See linked incident closures for resolution steps; this record captures the retrospective."
        records.append({
            "record_id": rid,
            "title": title,
            "document_type": "Lessons Learned",
            "technology_domain": domain,
            "incident_type": incident_type,
            "severity": "S2",
            "region": "All",
            "affected_system": "Multiple",
            "root_cause": rnd.choice(ROOT_CAUSES),
            "resolution_status": "Archived",
            "document_owner": OWNERS_BY_DOMAIN[domain],
            "last_updated": _date(rnd),
            "description": description,
            "resolution_steps": resolution,
            "lessons_learned": _lessons(rnd),
            "tags": "|".join([domain.lower().replace("/", "-"), incident_type.lower().replace(" ", "-"), "retrospective"]),
        })
    return records


FIELDS = [
    "record_id", "title", "document_type", "technology_domain", "incident_type",
    "severity", "region", "affected_system", "root_cause", "resolution_status",
    "document_owner", "last_updated", "description", "resolution_steps",
    "lessons_learned", "tags",
]


def write_corpus(records: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def write_example_records(records: list[dict], path: Path, k: int = 5) -> None:
    sample = [records[0], records[60], records[180], records[230], records[270]]
    sample = [r for r in sample if r is not None][:k]
    with path.open("w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)


def main() -> None:
    rnd = random.Random(SEED)
    incidents = build_incident_closures(rnd, n=180)
    sops = build_sops(rnd, n=50)
    vendor = build_vendor_notes(rnd, n=40)
    lessons = build_lessons_learned(rnd, n=30)
    records = incidents + sops + vendor + lessons
    out_csv = HERE / "synthetic_noc_corpus.csv"
    out_json = HERE / "example_records.json"
    write_corpus(records, out_csv)
    write_example_records(records, out_json, k=5)
    print(f"wrote {len(records)} records -> {out_csv}")
    print(f"wrote example records -> {out_json}")


if __name__ == "__main__":
    main()
