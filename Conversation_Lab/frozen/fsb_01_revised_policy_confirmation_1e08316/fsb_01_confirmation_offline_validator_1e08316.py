"""Offline validator for the FSB-01 revised-policy confirmation freeze."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


HEAD = "1e08316"
ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
MANIFEST = PACKAGE / "fsb_01_confirmation_manifest_1e08316.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Conversation_Lab import fsb_01_revised_policy_runner_1e08316 as source  # noqa: E402


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _verdict(value: Any) -> str:
    return value["verdict"] if isinstance(value, dict) else value


def validate() -> dict[str, Any]:
    network_events: list[str] = []

    def audit(event: str, _args: tuple[Any, ...]) -> None:
        if event in {"socket.connect", "socket.connect_ex"}:
            network_events.append(event)
            raise RuntimeError("Network access is forbidden in offline validation")

    sys.addaudithook(audit)
    actual_head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if actual_head != HEAD:
        raise AssertionError(f"HEAD={actual_head}; expected {HEAD}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["head"] != HEAD:
        raise AssertionError("Manifest HEAD mismatch")
    source_root = ROOT / manifest["source_freeze"]["path"]
    for name, expected in manifest["source_freeze"]["sha256"].items():
        if sha_file(source_root / name) != expected:
            raise AssertionError(f"Source freeze changed: {name}")
    source_runner = ROOT / manifest["source_freeze"]["runner_path"]
    if sha_file(source_runner) != manifest["source_freeze"]["runner_sha256"]:
        raise AssertionError("Source runner changed")
    for name, expected in manifest["new_artifact_sha256"].items():
        if sha_file(PACKAGE / name) != expected:
            raise AssertionError(f"Confirmation artifact changed: {name}")

    frozen = source.preflight()
    run_root = ROOT / manifest["source_run"]["path"]
    run_json = run_root / "run.json"
    checkpoint_path = run_root / "checkpoint.jsonl"
    if sha_file(run_json) != manifest["source_run"]["run_json_sha256"]:
        raise AssertionError("Source run metadata changed")
    if sha_file(checkpoint_path) != manifest["source_run"]["checkpoint_sha256"]:
        raise AssertionError("Source checkpoint changed")
    metadata = json.loads(run_json.read_text(encoding="utf-8"))
    checkpoint = source.base.Checkpoint(checkpoint_path)
    events = checkpoint.read()
    completed = source.validate_checkpoint(metadata, events, frozen, for_resume=False)
    if len(completed) != 34 or events[-1]["event_type"] != "RUN_COMPLETED":
        raise AssertionError("Source run is not the exact completed 34-call run")
    aggregates = {
        event["case_id"]: _verdict(event["aggregate"])
        for event in events if event["event_type"] == "AGGREGATE"
    }
    if {
        case: aggregates.get(case)
        for case in manifest["source_run"]["required_baseline_aggregates"]
    } != manifest["source_run"]["required_baseline_aggregates"]:
        raise AssertionError("Source candidate-miss aggregates changed")

    cases = manifest["cases"]
    if cases != ["FSB-M", "FSB-R", "FSB-S", "FSB-T", "FSB-D"]:
        raise AssertionError("Confirmation case set changed")
    if manifest["new_production_repetitions_per_case"] != 2:
        raise AssertionError("Production repetition count changed")
    expected_order = [
        "FSB-M-C1", "FSB-R-C1", "FSB-S-C1", "FSB-T-C1", "FSB-D-C1",
        "FSB-M-C2", "FSB-R-C2", "FSB-S-C2", "FSB-T-C2", "FSB-D-C2",
    ]
    if manifest["production_order"] != expected_order or manifest["evaluator_order"] != expected_order:
        raise AssertionError("Frozen round-major order changed")
    for sample_id in expected_order:
        base_case = sample_id.rsplit("-C", 1)[0]
        if manifest["sample_base_case"].get(sample_id) != base_case:
            raise AssertionError(f"Sample mapping changed: {sample_id}")
        if base_case not in frozen["cases"]:
            raise AssertionError(f"Source state missing: {base_case}")
    if manifest["production_model"] != "gpt-5-mini" or manifest["evaluator_model"] != "gpt-6-astra":
        raise AssertionError("Model assignment changed")
    if manifest["api_budget"] != {"production": 10, "evaluator": 30, "maximum_total": 40}:
        raise AssertionError("API budget changed")
    protocol = manifest["protocol"]
    if (
        protocol["retry"] != 0 or protocol["smoke_calls"] != 0
        or protocol["adaptive_sampling"] or protocol["result_conditioned_calls"]
        or protocol["extra_confirmation_calls"]
        or not protocol["all_production_precedes_evaluation"]
        or not protocol["explicit_live_authorization_required"]
    ):
        raise AssertionError("Non-adaptive protocol changed")
    if network_events:
        raise AssertionError(f"Network events occurred: {network_events}")
    return {
        "result": "READY", "api_calls": 0, "network_events": 0,
        "head": actual_head, "source_run_valid": True,
        "source_candidate_misses": aggregates,
        "states_reused": len(expected_order),
        "independent_new_production_selections": 10,
        "evaluator_votes": 30, "maximum_live_calls": 40,
        "pseudoreplication_guard": "production selection is the repeatability unit",
    }


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())], cwd=ROOT,
            env=environment, capture_output=True, text=True,
        )
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        return completed.returncode
    print(json.dumps(validate(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
