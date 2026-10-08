"""Resume-safe live runner for the focused two-arm FSB-M prompt regression."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


HEAD = "d34fdca"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Conversation_Lab import fsb_mrs_live_runner_1fe07e9 as base  # noqa: E402


FREEZE = ROOT / "Conversation_Lab" / "frozen" / "fsb_m_prompt_regression_d34fdca"
SOURCE = ROOT / "Conversation_Lab" / "frozen" / "fsb_01_mrs_policy_patch_1fe07e9"
RUNS_ROOT = ROOT / "Conversation_Lab" / "traces" / "fsb_m_prompt_regression_d34fdca_runs"
MANIFEST_NAME = "fsb_m_prompt_regression_manifest_d34fdca.json"
VALIDATOR_NAME = "fsb_m_prompt_regression_offline_validator_d34fdca.py"
SELF = Path(__file__).resolve()

FREEZE_SHA256 = {
    "fsb_m_prompt_regression_freeze_report_d34fdca.md": "970C89D7601C756BF8293696FFD46DAA3102025E75484928D0AB900D9EA78D50",
    "fsb_m_prompt_regression_manifest_d34fdca.json": "BFFF76AE957473F945628746337083261509E0F1B4193CB1EF5342B6DE9C7BC2",
    "fsb_m_prompt_regression_offline_validator_d34fdca.py": "52D639812B083399CE4098FDE36616695CC1F38F7FB287BB237AEE7C14B809B7",
}


def _specs(manifest: dict[str, Any]) -> list[base.CallSpec]:
    model_by_arm = {arm["arm_id"]: arm["production_model"] for arm in manifest["arms"]}
    specs = [
        base.CallSpec(
            key=f"production.{arm_id}", phase="focused_prompt_regression",
            case_id=arm_id, role="production_decision", model=model_by_arm[arm_id]
        )
        for arm_id in manifest["production_order"]
    ]
    for arm_id in manifest["evaluator_order"]:
        specs.extend(
            base.CallSpec(
                key=f"evaluator.{arm_id}.{repeat}", phase="focused_prompt_regression",
                case_id=arm_id, role="blind_evaluator",
                model=manifest["evaluator_model"], repeat=repeat,
                selected_on_left=selected_on_left,
            )
            for repeat, selected_on_left in enumerate(
                manifest["evaluator_orientations_selected_on_left"][arm_id], 1
            )
        )
    return specs


def preflight() -> dict[str, Any]:
    actual_head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True
    ).stdout.strip()
    if actual_head != HEAD:
        raise base.RunnerError("head_failure", f"HEAD={actual_head}; expected {HEAD}")
    names = {path.name for path in FREEZE.iterdir() if path.is_file()}
    if names != set(FREEZE_SHA256):
        raise base.RunnerError("hash_failure", f"Freeze file set mismatch: {sorted(names)}")
    freeze_hashes = {}
    for name, expected in FREEZE_SHA256.items():
        actual = base._sha_file(FREEZE / name)
        if actual != expected:
            raise base.RunnerError("hash_failure", f"Freeze SHA mismatch: {name}")
        freeze_hashes[name] = actual

    manifest = base._read_json(FREEZE / MANIFEST_NAME)
    for name, expected in manifest["source_artifact_sha256"].items():
        if base._sha_file(SOURCE / name) != expected:
            raise base.RunnerError("hash_failure", f"Source SHA mismatch: {name}")
    validator = subprocess.run(
        [sys.executable, str(FREEZE / VALIDATOR_NAME)], cwd=ROOT, check=False,
        capture_output=True, text=True
    )
    if validator.returncode != 0:
        raise base.RunnerError(
            "offline_validation_failure",
            f"Offline validator exit={validator.returncode}: {validator.stderr}",
        )
    validation = json.loads(validator.stdout)
    if validation.get("result") != "READY" or validation.get("api_calls") != 0:
        raise base.RunnerError("offline_validation_failure", repr(validation))

    source_cases = base._read_json(SOURCE / "fsb_mrs_cases_1fe07e9.json")["cases"]
    source_case = next(item for item in source_cases if item["case_id"] == "FSB-M")
    source_challengers = base._read_json(
        SOURCE / "fsb_mrs_challengers_1fe07e9.json"
    )["challengers"]
    source_challenger = next(
        item for item in source_challengers if item["case_id"] == "FSB-M"
    )
    cases = []
    challengers = []
    for arm in manifest["arms"]:
        case = json.loads(json.dumps(source_case))
        case["case_id"] = arm["arm_id"]
        cases.append(case)
        challenger = json.loads(json.dumps(source_challenger))
        challenger["case_id"] = arm["arm_id"]
        challengers.append(challenger)

    schedule = _specs(manifest)
    if len(schedule) != 8 or [spec.role for spec in schedule[:2]] != [
        "production_decision", "production_decision"
    ] or any(spec.role != "blind_evaluator" for spec in schedule[2:]):
        raise base.RunnerError("protocol_failure", "Frozen 2+6 schedule mismatch")
    return {
        "head": actual_head,
        "manifest_sha256": freeze_hashes[MANIFEST_NAME],
        "freeze_sha256": {"focused": freeze_hashes, "source": manifest["source_artifact_sha256"]},
        "manifest": manifest,
        "cases": {"cases": cases},
        "challengers": {"challengers": challengers},
        "offline_validation": validation,
        "runner_dependency_sha256": base._sha_file(SELF),
        "evaluator_implementation_sha256": base._sha_file(
            ROOT / "Conversation_Lab" / "selected_action_evaluator.py"
        ),
    }


def _new_run(frozen: dict[str, Any], run_root: Optional[Path] = None):
    root = RUNS_ROOT if run_root is None else run_root
    run_id = (
        f"fsbm-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    run_dir = root / run_id
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "head": HEAD,
        "manifest_sha256": frozen["manifest_sha256"],
        "freeze_sha256": frozen["freeze_sha256"],
        "runner_dependency_sha256": frozen["runner_dependency_sha256"],
        "evaluator_implementation_sha256": frozen["evaluator_implementation_sha256"],
        "retry": 0,
        "smoke_calls": 0,
        "maximum_api_calls": 8,
    }
    base._write_metadata(run_dir / "run.json", metadata)
    checkpoint = base.Checkpoint(run_dir / "checkpoint.jsonl")
    checkpoint.append(
        "RUN_STARTED", run_id=run_id, head=HEAD,
        manifest_sha256=frozen["manifest_sha256"]
    )
    return run_id, run_dir, checkpoint


def execute_live(frozen: dict[str, Any], metadata: dict[str, Any], checkpoint: base.Checkpoint) -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(dotenv_path=ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise base.RunnerError("api_failure", "OPENAI_API_KEY is missing")
    client = OpenAI(max_retries=0, timeout=120.0)
    cases, challengers = base._case_maps(frozen)
    schedule = _specs(frozen["manifest"])
    while True:
        events = checkpoint.read()
        done = base.validate_checkpoint(metadata, events, frozen, for_resume=False)
        base._append_missing_aggregates(checkpoint, events, frozen["manifest"])
        events = checkpoint.read()
        done = base.validate_checkpoint(metadata, events, frozen, for_resume=False)
        if len(done) == len(schedule):
            checkpoint.append("RUN_COMPLETED", api_calls=8, production_calls=2, evaluator_calls=6)
            print(f"RUN_COMPLETED={metadata['run_id']}", flush=True)
            print(f"CHECKPOINT={checkpoint.path}", flush=True)
            return
        spec = schedule[len(done)]
        call_index = len(done) + 1
        base._append_prepared(checkpoint, spec, call_index, frozen)
        try:
            if spec.role == "production_decision":
                base._run_production(client, checkpoint, spec, call_index, cases[spec.case_id])
            else:
                base._run_evaluator(
                    client, checkpoint, spec, call_index, frozen,
                    cases[spec.case_id], challengers[spec.case_id], events
                )
        except BaseException as error:
            checkpoint.append(
                "STOP", category=getattr(error, "category", "api_failure"),
                call_index=call_index, call_spec=asdict(spec),
                error_type=type(error).__name__, error_message=str(error)
            )
            raise


def dry_run(frozen: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="fsbm-prompt-dry-") as temporary:
        root = Path(temporary)
        _, run_dir, checkpoint = _new_run(frozen, root)
        metadata = base._read_json(run_dir / "run.json")
        cases, _ = base._case_maps(frozen)
        schedule = _specs(frozen["manifest"])
        for spec in schedule:
            if spec.role == "production_decision":
                base._load_production_input(cases[spec.case_id])
            base._synthetic_complete(checkpoint, metadata, frozen, spec)
        base._append_missing_aggregates(checkpoint, checkpoint.read(), frozen["manifest"])
        done = base.validate_checkpoint(metadata, checkpoint.read(), frozen, for_resume=False)
        if len(done) != 8:
            raise base.RunnerError("protocol_failure", "Dry-run count mismatch")

        _, prefix_dir, prefix = _new_run(frozen, root)
        prefix_meta = base._read_json(prefix_dir / "run.json")
        for spec in schedule[:3]:
            base._synthetic_complete(prefix, prefix_meta, frozen, spec)
        base.validate_checkpoint(prefix_meta, prefix.read(), frozen, for_resume=True)

        _, dangling_dir, dangling = _new_run(frozen, root)
        dangling_meta = base._read_json(dangling_dir / "run.json")
        base._append_prepared(dangling, schedule[0], 1, frozen)
        try:
            base.validate_checkpoint(dangling_meta, dangling.read(), frozen, for_resume=True)
        except base.RunnerError:
            dangling_rejected = True
        else:
            dangling_rejected = False
        if not dangling_rejected:
            raise base.RunnerError("protocol_failure", "Dangling call was resumable")

        _, order_dir, wrong = _new_run(frozen, root)
        wrong_meta = base._read_json(order_dir / "run.json")
        base._synthetic_complete(wrong, wrong_meta, frozen, schedule[1])
        try:
            base.validate_checkpoint(wrong_meta, wrong.read(), frozen, for_resume=True)
        except base.RunnerError:
            order_rejected = True
        else:
            order_rejected = False
        if not order_rejected:
            raise base.RunnerError("protocol_failure", "Out-of-order call was accepted")

        return {
            "result": "PASS", "api_calls": 0, "head": HEAD,
            "state_identity": frozen["offline_validation"]["state_identity"],
            "challenger_identity": frozen["offline_validation"]["challenger_identity"],
            "production_calls_simulated": 2, "evaluator_calls_simulated": 6,
            "maximum_call_budget": 8, "adaptive_confirmation": "DISABLED",
            "both_production_calls_precede_evaluator": "PASS",
            "checkpoint_hash_chain": "PASS", "clean_prefix_resume": "PASS",
            "dangling_call_resume_rejected": "PASS", "out_of_order_call_rejected": "PASS"
        }


base.HEAD = HEAD
base.PACKAGE = SOURCE
base.RUNS_ROOT = RUNS_ROOT
base.EVALUATOR_PROMPT_NAME = "fsb_mrs_evaluator_prompt_1fe07e9.txt"
base.EVALUATOR_SCHEMA_NAME = "fsb_mrs_evaluator_schema_1fe07e9.json"
base._specs = _specs
base.preflight = preflight
base._new_run = _new_run
base.execute_live = execute_live
base.dry_run = dry_run


if __name__ == "__main__":
    raise SystemExit(base.main())
