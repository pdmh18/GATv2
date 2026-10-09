"""Read and validate JSONL without importing training/runtime dependencies."""
from __future__ import annotations

import hashlib
import json
from sys import intern
from datetime import datetime
from pathlib import Path


class Audit:
    def __init__(self):
        self.issues: list[dict[str, str]] = []

    def check(self, condition, code, location, message, severity="error"):
        if not condition:
            self.issues.append(dict(severity=severity, code=code, location=location, message=message))
        return bool(condition)

    def warning(self, code, location, message):
        self.check(False, code, location, message, "warning")


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be an ISO string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("timestamp requires timezone")
    return parsed


def integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**63 - 1


def fields(audit, row, spec, location, *, exact=False):
    if not audit.check(isinstance(row, dict), "schema", location, "Expected object"):
        return False
    ok = True
    for key, predicate in spec.items():
        try:
            valid = key in row and predicate(row[key])
        except (ValueError, TypeError, KeyError):
            valid = False
        ok = audit.check(valid, "schema", f"{location}.{key}", "Missing or invalid field") and ok
    if exact:
        ok = audit.check(set(row) <= set(spec), "schema", location, "Unexpected fields") and ok
    return ok


def text(value):
    return isinstance(value, str) and bool(value)


def time_value(value):
    timestamp(value)
    return True


def no_leakage(value):
    if isinstance(value, dict):
        forbidden = {"risk", "risk_score", "riskScore", "is_anomaly", "isAnomaly", "scenario_id", "scenarioId", "label", "labels"}
        return not (set(value) & forbidden) and all(no_leakage(v) for v in value.values())
    if isinstance(value, list):
        return all(no_leakage(v) for v in value)
    return True


def read_dataset(root: Path, audit: Audit):
    hashes, tables = {}, {}
    manifest = {}
    try:
        raw = (root / "manifest.json").read_bytes()
        hashes["manifest.json"] = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
    except (OSError, ValueError) as exc:
        audit.check(False, "manifest", "manifest.json", str(exc))
        manifest = {}
    counts = manifest.get("counts", {})
    audit.check(isinstance(counts, dict), "manifest", "manifest.counts", "Expected object")
    names = {"customers", "accounts", "transactions", "labels", "counterparties", "events", "transfer_events", "loan_packages", "loan_applications", "loans"}
    if isinstance(counts, dict):
        names.update(counts)
    for name in sorted(names):
        if not isinstance(name, str) or not name.replace("_", "").isalnum():
            audit.check(False, "manifest", "manifest.counts", "Invalid table name")
            continue
        rows = []
        try:
            digest = hashlib.sha256()
            # Hash and parse line-by-line to avoid whole-file byte/text copies.
            # Reuse field names across rows in large event streams.
            with (root / f"{name}.jsonl").open("rb") as stream:
                for n, line in enumerate(stream, 1):
                    digest.update(line)
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line.decode("utf-8"), object_hook=lambda obj: {intern(k): v for k, v in obj.items()})
                        if not isinstance(row, dict):
                            raise ValueError("row must be an object")
                        rows.append(row)
                    except ValueError as exc:
                        audit.check(False, "jsonl", f"{name}:{n}", str(exc))
            hashes[f"{name}.jsonl"] = digest.hexdigest()
        except (OSError, UnicodeError) as exc:
            audit.check(False, "input_file", name, str(exc))
        tables[name] = rows
        if isinstance(counts, dict):
            expected = counts.get(name)
            audit.check(integer(expected) and expected == len(rows), "manifest_count", name,
                        f"Manifest count {expected!r}, parsed rows {len(rows)}")
    audit.check(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 2, "manifest", "manifest.schema_version", "Expected dataset schema 2")
    audit.check(type(manifest.get("seed")) is int and text(manifest.get("source_profile")), "manifest", "manifest", "Seed must be integer and source_profile nonempty string")
    return tables, manifest, hashes


def index_rows(rows, key, audit, name):
    result = {}
    for n, row in enumerate(rows, 1):
        identity = row.get(key)
        if not audit.check(text(identity), "identity", f"{name}:{n}", f"Missing string {key}"):
            continue
        if not audit.check(identity not in result, "duplicate_id", f"{name}:{n}", f"Duplicate {key}: {identity}"):
            continue
        result[identity] = row
    return result
