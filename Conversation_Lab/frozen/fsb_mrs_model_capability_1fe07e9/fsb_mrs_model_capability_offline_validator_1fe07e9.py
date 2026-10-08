"""Offline validator for the matched FSB-M/R/S model-capability isolation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
MANIFEST = PACKAGE / "fsb_mrs_model_capability_manifest_1fe07e9.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != "1fe07e9" or manifest["head"] != head:
        raise RuntimeError(f"HEAD mismatch: actual={head!r}")

    source = ROOT / manifest["source_freeze"]
    for name, expected in manifest["source_artifact_sha256"].items():
        actual = sha256(source / name)
        if actual != expected:
            raise RuntimeError(f"Source SHA mismatch for {name}: {actual}")

    source_manifest_path = source / "fsb_mrs_manifest_1fe07e9.json"
    if sha256(source_manifest_path) != manifest["source_manifest_sha256"]:
        raise RuntimeError("Source manifest binding mismatch")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    invariant_fields = (
        "adaptive_confirmation",
        "api_budget",
        "case_order",
        "case_sha256",
        "challenger_sha256",
        "contract_boundary",
        "evaluator_model",
        "evaluator_order",
        "evaluator_orientations_selected_on_left",
        "head",
        "production_boundary",
        "production_order",
        "retry",
        "smoke_calls",
        "state_sha256",
        "stop_rules",
    )
    for field in invariant_fields:
        if manifest[field] != source_manifest[field]:
            raise RuntimeError(f"Frozen protocol drift in {field}")
    if source_manifest["production_model"] != manifest["baseline_production_model"]:
        raise RuntimeError("Baseline model binding mismatch")
    if manifest["production_model"] != "gpt-6-astra":
        raise RuntimeError("Isolation production model must be gpt-6-astra")

    source_validator = subprocess.run(
        [sys.executable, str(source / "fsb_mrs_offline_validator_1fe07e9.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if source_validator.returncode != 0:
        raise RuntimeError(
            "Source offline validator failed: " + source_validator.stderr.strip()
        )
    source_result = json.loads(source_validator.stdout)
    if source_result.get("result") != "READY":
        raise RuntimeError(f"Source freeze is not READY: {source_result!r}")

    print(
        json.dumps(
            {
                "result": "READY",
                "head": head,
                "cases": manifest["case_order"],
                "states_unchanged": True,
                "challengers_unchanged": True,
                "production_prompt_unchanged": True,
                "wire_schema_unchanged": True,
                "evaluator_policy_unchanged": True,
                "only_experimental_variable": "production_model",
                "baseline_production_model": manifest["baseline_production_model"],
                "isolation_production_model": manifest["production_model"],
                "future_api_calls": 12,
                "api_calls": 0,
                "source_validation": source_result.get("result"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
