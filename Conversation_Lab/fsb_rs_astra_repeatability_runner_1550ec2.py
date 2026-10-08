"""Resume-safe runner for the frozen FSB-R/S Astra repeatability regression."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


HEAD = "1550ec2"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Conversation_Lab import fsb_01_full_regression_runner_1550ec2 as parent  # noqa: E402


base = parent.base
FREEZE = ROOT / "Conversation_Lab/frozen/fsb_rs_astra_repeatability_1550ec2"
SOURCE = ROOT / "Conversation_Lab/frozen/fsb_01_672455a"
RUNS_ROOT = ROOT / "Conversation_Lab/traces/fsb_rs_astra_repeatability_1550ec2_runs"
MANIFEST_NAME = "fsb_rs_astra_repeatability_manifest_1550ec2.json"
VALIDATOR_NAME = "fsb_rs_astra_repeatability_offline_validator_1550ec2.py"
SELF = Path(__file__).resolve()
PARENT_RUNNER = ROOT / "Conversation_Lab/fsb_01_full_regression_runner_1550ec2.py"
PARENT_RUNNER_SHA256 = "1A189CD02648285C876E1FEE3E5F7C91F6CBF0CE3AC4DDA8FC21A091E9707F6F"
FREEZE_SHA256 = {
    "fsb_rs_astra_repeatability_freeze_report_1550ec2.md": "FC5AFD0BC82A2545ABAD630C0DC19BE6487A063468115139C1EFA1435747D4CB",
    "fsb_rs_astra_repeatability_manifest_1550ec2.json": "5765F3FF17B59F3E998C02998C75C8EECC6AE20DFE1B745E36C430FCE01ACC0A",
    "fsb_rs_astra_repeatability_offline_validator_1550ec2.py": "A5363F475F241502F17E676B0B4AB35DD42FEDFD4EC78775102042AA3E7D51FE",
}


def _git_bytes(spec: str) -> bytes:
    return subprocess.run(
        ["git", "show", spec], cwd=ROOT, check=True, capture_output=True
    ).stdout


def _specs(manifest: dict[str, Any]) -> list[base.CallSpec]:
    specs = [
        base.CallSpec(
            key=f"production.{sample_id}", phase="astra_repeatability",
            case_id=sample_id, role="production_decision",
            model=manifest["production_model"],
        )
        for sample_id in manifest["production_order"]
    ]
    for sample_id in manifest["evaluator_order"]:
        specs.extend(
            base.CallSpec(
                key=f"evaluator.{sample_id}.{repeat}", phase="astra_repeatability",
                case_id=sample_id, role="blind_evaluator",
                model=manifest["evaluator_model"], repeat=repeat,
                selected_on_left=selected_on_left,
            )
            for repeat, selected_on_left in enumerate(
                manifest["evaluator_orientations_selected_on_left"][sample_id], 1
            )
        )
    return specs


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
    if base._sha_file(PARENT_RUNNER) != PARENT_RUNNER_SHA256:
        raise base.RunnerError("hash_failure", "Checkpoint runner dependency changed")

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

    order = ["FSB-R-1", "FSB-R-2", "FSB-R-3", "FSB-S-1", "FSB-S-2", "FSB-S-3"]
    if manifest["case_order"] != order or manifest["production_order"] != order or manifest["evaluator_order"] != order:
        raise base.RunnerError("protocol_failure", "Frozen sample order mismatch")
    if manifest["production_model"] != "gpt-6-astra" or manifest["evaluator_model"] != "gpt-6-astra":
        raise base.RunnerError("protocol_failure", "Frozen model assignment mismatch")
    if manifest["api_budget"] != {"production": 6, "evaluator": 18, "maximum_total": 24}:
        raise base.RunnerError("protocol_failure", "Frozen API budget mismatch")
    schedule = _specs(manifest)
    if len(schedule) != 24 or any(spec.role != "production_decision" for spec in schedule[:6]) or any(spec.role != "blind_evaluator" for spec in schedule[6:]):
        raise base.RunnerError("protocol_failure", "Frozen 6+18 schedule mismatch")

    source_cases = {
        item["case_id"]: item
        for item in base._read_json(SOURCE / "fsb_01_cases_672455a.json")["cases"]
    }
    source_challengers = {
        item["case_id"]: item
        for item in base._read_json(SOURCE / "fsb_01_challengers_672455a.json")["challengers"]
    }
    cases, challengers = [], []
    for sample_id in order:
        base_id = manifest["sample_base_case"][sample_id]
        case = json.loads(json.dumps(source_cases[base_id]))
        case["case_id"] = sample_id
        cases.append(case)
        challenger = json.loads(json.dumps(source_challengers[base_id]))
        challenger["case_id"] = sample_id
        challengers.append(challenger)
    return {
        "head": actual_head,
        "manifest": manifest,
        "manifest_sha256": actual_hashes[MANIFEST_NAME],
        "freeze_sha256": {"repeatability": actual_hashes, "source": manifest["source_artifact_sha256"]},
        "cases": {"cases": cases},
        "challengers": {"challengers": challengers},
        "offline_validation": validation,
        "runner_dependency_sha256": base._sha_file(SELF),
        "evaluator_implementation_sha256": base._sha_file(
            ROOT / "Conversation_Lab/selected_action_evaluator.py"
        ),
    }


def validate_checkpoint(
    metadata: dict[str, Any], events: list[dict[str, Any]], frozen: dict[str, Any],
    *, for_resume: bool,
) -> list[dict[str, Any]]:
    return parent.validate_checkpoint(metadata, events, frozen, for_resume=for_resume)


def _new_run(frozen: dict[str, Any], root: Optional[Path] = None):
    output_root = RUNS_ROOT if root is None else root
    run_id = f"fsbrs-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = output_root / run_id
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "head": HEAD,
        "manifest_sha256": frozen["manifest_sha256"],
        "freeze_sha256": frozen["freeze_sha256"],
        "runner_dependency_sha256": frozen["runner_dependency_sha256"],
        "evaluator_implementation_sha256": frozen["evaluator_implementation_sha256"],
        "retry": 0, "smoke_calls": 0, "confirmations": 0,
        "adaptive_sampling": False, "maximum_api_calls": 24,
    }
    base._write_metadata(run_dir / "run.json", metadata)
    checkpoint = base.Checkpoint(run_dir / "checkpoint.jsonl")
    checkpoint.append(
        "RUN_STARTED", run_id=run_id, head=HEAD,
        manifest_sha256=frozen["manifest_sha256"],
    )
    return run_id, run_dir, checkpoint


def dry_run(frozen: dict[str, Any]) -> dict[str, Any]:
    result = parent._AB_DRY_RUN(frozen)
    return {
        "result": result["result"], "api_calls": 0, "head": HEAD,
        "cases": ["FSB-R", "FSB-S"], "samples_per_case": 3,
        "production_calls_simulated": 6, "evaluator_calls_simulated": 18,
        "maximum_call_budget": 24,
        "production_model": "gpt-6-astra", "evaluator_model": "gpt-6-astra",
        "all_production_precedes_evaluator": result["all_production_precedes_evaluator"],
        "checkpoint_hash_chain": result["checkpoint_hash_chain"],
        "clean_prefix_resume": result["clean_prefix_resume"],
        "dangling_call_resume_rejected": result["dangling_call_resume_rejected"],
        "out_of_order_call_rejected": result["out_of_order_call_rejected"],
        "confirmations": 0, "adaptive_sampling": "DISABLED",
    }


parent.HEAD = HEAD
parent.FREEZE = FREEZE
parent.SOURCE = SOURCE
parent.RUNS_ROOT = RUNS_ROOT
parent._specs = _specs
parent.preflight = preflight
parent._new_run = _new_run
parent.dry_run = dry_run

parent.ab.HEAD = HEAD
parent.ab.FREEZE = FREEZE
parent.ab.SOURCE = SOURCE
parent.ab.RUNS_ROOT = RUNS_ROOT
parent.ab._specs = _specs
parent.ab.preflight = preflight
parent.ab._new_run = _new_run
parent.ab.dry_run = dry_run

base.HEAD = HEAD
base.PACKAGE = SOURCE
base.RUNS_ROOT = RUNS_ROOT
base.EVALUATOR_PROMPT_NAME = "fsb_01_evaluator_prompt_672455a.txt"
base.EVALUATOR_SCHEMA_NAME = "fsb_01_evaluator_schema_672455a.json"
base._specs = _specs
base.validate_checkpoint = validate_checkpoint
base.preflight = preflight
base._new_run = _new_run
base.dry_run = dry_run


if __name__ == "__main__":
    raise SystemExit(base.main())
