"""Resume-safe runner for the frozen FSB-H/T/A Astra coverage regression."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


HEAD = "1550ec2"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Conversation_Lab import fsb_mrs_live_runner_1fe07e9 as base  # noqa: E402


FREEZE = ROOT / "Conversation_Lab/frozen/fsb_hta_astra_coverage_1550ec2"
SOURCE = ROOT / "Conversation_Lab/frozen/fsb_01_672455a"
RUNS_ROOT = ROOT / "Conversation_Lab/traces/fsb_hta_astra_coverage_1550ec2_runs"
MANIFEST_NAME = "fsb_hta_astra_coverage_manifest_1550ec2.json"
VALIDATOR_NAME = "fsb_hta_astra_coverage_offline_validator_1550ec2.py"
SELF = Path(__file__).resolve()
FREEZE_SHA256 = {
    "fsb_hta_astra_coverage_freeze_report_1550ec2.md": "6F5D98802D1DE9F9F3006BE7079C2F3A4E67DE98C743CFC1A2CAFB0B079B8BFE",
    "fsb_hta_astra_coverage_manifest_1550ec2.json": "CF4EA3D8189F45C4D619D269C0213AC043C70C4FC1DD86C332F1E429D6AE4906",
    "fsb_hta_astra_coverage_offline_validator_1550ec2.py": "285B83C26E1A0837C54FDD462034A4425614E780A12D5D1C8CE353F24C398BD0",
}


def _git_bytes(spec: str) -> bytes:
    return subprocess.run(
        ["git", "show", spec], cwd=ROOT, check=True, capture_output=True
    ).stdout


def preflight() -> dict[str, Any]:
    actual_head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if actual_head != HEAD:
        raise base.RunnerError("head_failure", f"HEAD={actual_head}; expected {HEAD}")

    names = {path.name for path in FREEZE.iterdir() if path.is_file()}
    if names != set(FREEZE_SHA256):
        raise base.RunnerError("hash_failure", f"Freeze file set mismatch: {sorted(names)}")
    actual_hashes: dict[str, str] = {}
    for name, expected in FREEZE_SHA256.items():
        actual = base._sha_file(FREEZE / name)
        if actual != expected:
            raise base.RunnerError("hash_failure", f"Freeze SHA mismatch: {name}")
        actual_hashes[name] = actual

    manifest = base._read_json(FREEZE / MANIFEST_NAME)
    for name, expected in manifest["source_artifact_sha256"].items():
        if base._sha_file(SOURCE / name) != expected:
            raise base.RunnerError("hash_failure", f"Source artifact changed: {name}")
    prompt = _git_bytes(manifest["prompt"]["git_path"])
    if hashlib.sha256(prompt).hexdigest().upper() != manifest["prompt"]["sha256"]:
        raise base.RunnerError("hash_failure", "Production prompt SHA mismatch")

    validator = subprocess.run(
        [sys.executable, str(FREEZE / VALIDATOR_NAME)], cwd=ROOT,
        capture_output=True, text=True,
    )
    if validator.returncode != 0:
        raise base.RunnerError(
            "offline_validation_failure",
            f"Offline validator exit={validator.returncode}: {validator.stderr}",
        )
    validation = json.loads(validator.stdout)
    if validation.get("result") != "READY" or validation.get("api_calls") != 0:
        raise base.RunnerError("offline_validation_failure", repr(validation))

    order = ["FSB-H", "FSB-T", "FSB-A"]
    for field in ("case_order", "production_order", "evaluator_order"):
        if manifest.get(field) != order:
            raise base.RunnerError("protocol_failure", f"Frozen {field} mismatch")
    if manifest["production_model"] != "gpt-6-astra" or manifest["evaluator_model"] != "gpt-6-astra":
        raise base.RunnerError("protocol_failure", "Frozen model assignment mismatch")
    if manifest["api_budget"] != {"production": 3, "evaluator": 9, "maximum_total": 12}:
        raise base.RunnerError("protocol_failure", "Frozen API budget mismatch")
    schedule = base._specs(manifest)
    if len(schedule) != 12 or any(spec.role != "production_decision" for spec in schedule[:3]) or any(spec.role != "blind_evaluator" for spec in schedule[3:]):
        raise base.RunnerError("protocol_failure", "Frozen 3+9 schedule mismatch")

    all_cases = base._read_json(SOURCE / "fsb_01_cases_672455a.json")["cases"]
    all_challengers = base._read_json(SOURCE / "fsb_01_challengers_672455a.json")["challengers"]
    cases = {item["case_id"]: item for item in all_cases}
    challengers = {item["case_id"]: item for item in all_challengers}
    return {
        "head": actual_head,
        "manifest_sha256": actual_hashes[MANIFEST_NAME],
        "freeze_sha256": {"coverage": actual_hashes, "source": manifest["source_artifact_sha256"]},
        "manifest": manifest,
        "cases": {"cases": [cases[case_id] for case_id in order]},
        "challengers": {"challengers": [challengers[case_id] for case_id in order]},
        "offline_validation": validation,
        "runner_dependency_sha256": base._sha_file(SELF),
        "evaluator_implementation_sha256": base._sha_file(
            ROOT / "Conversation_Lab/selected_action_evaluator.py"
        ),
    }


def _run_production(
    client: Any, checkpoint: base.Checkpoint, spec: base.CallSpec,
    call_index: int, case: dict[str, Any],
) -> None:
    from ai.cognitive_core import CognitiveCore
    from Conversation_Lab.selected_action_evaluator import selected_action_from_cognitive_turn

    manifest = base._read_json(FREEZE / MANIFEST_NAME)
    prompt_bytes = _git_bytes(manifest["prompt"]["git_path"])
    if hashlib.sha256(prompt_bytes).hexdigest().upper() != manifest["prompt"]["sha256"]:
        raise base.RunnerError("hash_failure", "Production prompt changed after preflight")
    FrozenCore = type(
        "FrozenPromptCognitiveCore", (CognitiveCore,),
        {"_system_prompt": staticmethod(lambda: prompt_bytes.decode("utf-8"))},
    )
    current, human_model, previous_acs, history = base._load_production_input(case)

    def captured(response: Any, request: Any) -> None:
        base._append_response(checkpoint, spec, call_index, response, request)

    capturing = base._CapturingClient(client, captured)
    result = FrozenCore(capturing, model=spec.model).propose(
        current, human_model, previous_acs, history
    )
    if capturing.completions.calls != 1:
        raise base.RunnerError("protocol_failure", "Production slot made != 1 API call")
    checkpoint.append(
        "CALL_COMPLETED", call_index=call_index, call_spec=asdict(spec),
        normalized_result={
            "cognitive_turn_result": result,
            "observable_action": selected_action_from_cognitive_turn(result),
            "production_prompt_sha256": manifest["prompt"]["sha256"],
        },
    )


def _new_run(frozen: dict[str, Any], run_root: Optional[Path] = None):
    root = RUNS_ROOT if run_root is None else run_root
    run_id = f"fsbhta-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = root / run_id
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "head": HEAD,
        "manifest_sha256": frozen["manifest_sha256"],
        "freeze_sha256": frozen["freeze_sha256"],
        "runner_dependency_sha256": frozen["runner_dependency_sha256"],
        "evaluator_implementation_sha256": frozen["evaluator_implementation_sha256"],
        "retry": 0, "smoke_calls": 0, "confirmations": 0,
        "adaptive_sampling": False, "maximum_api_calls": 12,
    }
    base._write_metadata(run_dir / "run.json", metadata)
    checkpoint = base.Checkpoint(run_dir / "checkpoint.jsonl")
    checkpoint.append(
        "RUN_STARTED", run_id=run_id, head=HEAD,
        manifest_sha256=frozen["manifest_sha256"],
    )
    return run_id, run_dir, checkpoint


base.HEAD = HEAD
base.PACKAGE = SOURCE
base.RUNS_ROOT = RUNS_ROOT
base.EVALUATOR_PROMPT_NAME = "fsb_01_evaluator_prompt_672455a.txt"
base.EVALUATOR_SCHEMA_NAME = "fsb_01_evaluator_schema_672455a.json"
base.preflight = preflight
base._run_production = _run_production
base._new_run = _new_run


if __name__ == "__main__":
    raise SystemExit(base.main())
