"""Resume-safe runner for the full FSB-01 regression at HEAD 1550ec2."""

from __future__ import annotations

import hashlib
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


HEAD = "1550ec2"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Conversation_Lab import fsb_m_prompt_ab_runner_1550ec2 as ab  # noqa: E402


base = ab.base
_AB_DRY_RUN = ab.dry_run
_AB_VALIDATE_CHECKPOINT = ab.validate_checkpoint
FREEZE = ROOT / "Conversation_Lab/frozen/fsb_01_full_regression_1550ec2"
SOURCE = ROOT / "Conversation_Lab/frozen/fsb_01_672455a"
RUNS_ROOT = ROOT / "Conversation_Lab/traces/fsb_01_full_regression_1550ec2_runs"
MANIFEST_NAME = "fsb_01_full_manifest_1550ec2.json"
VALIDATOR_NAME = "fsb_01_full_offline_validator_1550ec2.py"
SELF = Path(__file__).resolve()
PARENT_RUNNER = ROOT / "Conversation_Lab/fsb_m_prompt_ab_runner_1550ec2.py"
PARENT_RUNNER_SHA256 = "280FBD22705E7918686F08B26C68A6A766A2F474EE1512D9B4CB4F46FDB8BBB8"
FREEZE_SHA256 = {
    "fsb_01_full_freeze_report_1550ec2.md": "432813FD4B2B3F48B7D552BF1F4AEAC9B65F770CA0F69030C63F7EBBFAB7849F",
    "fsb_01_full_manifest_1550ec2.json": "1A4B95A3C15EF56C8E32459A4F762320EB53CD52690B17BB8DEF735964D23E5A",
    "fsb_01_full_offline_validator_1550ec2.py": "E5D08B3D55EC2D882291029406CF746232D52775F4D79917604FF498CC74663D",
}


def _git_bytes(spec: str) -> bytes:
    return subprocess.run(
        ["git", "show", spec], cwd=ROOT, check=True, capture_output=True
    ).stdout


def _specs(manifest: dict[str, Any]) -> list[base.CallSpec]:
    specs = [
        base.CallSpec(
            key=f"production.{case_id}", phase="full_regression",
            case_id=case_id, role="production_decision",
            model=manifest["production_model"],
        )
        for case_id in manifest["production_order"]
    ]
    for case_id in manifest["evaluator_order"]:
        specs.extend(
            base.CallSpec(
                key=f"evaluator.{case_id}.{repeat}", phase="full_regression",
                case_id=case_id, role="blind_evaluator",
                model=manifest["evaluator_model"], repeat=repeat,
                selected_on_left=selected_on_left,
            )
            for repeat, selected_on_left in enumerate(
                manifest["evaluator_orientations_selected_on_left"][case_id], 1
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
    actual_hashes = {}
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

    order = ["FSB-H", "FSB-M", "FSB-R", "FSB-S", "FSB-T", "FSB-A"]
    if manifest["case_order"] != order or manifest["production_order"] != order or manifest["evaluator_order"] != order:
        raise base.RunnerError("protocol_failure", "Frozen case order mismatch")
    if manifest["production_model"] != "gpt-5-mini" or manifest["evaluator_model"] != "gpt-6-astra":
        raise base.RunnerError("protocol_failure", "Frozen model assignment mismatch")
    if manifest["api_budget"] != {"production": 6, "evaluator": 18, "maximum_total": 24}:
        raise base.RunnerError("protocol_failure", "Frozen API budget mismatch")
    if len(_specs(manifest)) != 24 or any(spec.role != "production_decision" for spec in _specs(manifest)[:6]):
        raise base.RunnerError("protocol_failure", "Frozen 6+18 schedule mismatch")

    cases = base._read_json(SOURCE / "fsb_01_cases_672455a.json")
    challengers = base._read_json(SOURCE / "fsb_01_challengers_672455a.json")
    return {
        "head": actual_head,
        "manifest": manifest,
        "manifest_sha256": actual_hashes[MANIFEST_NAME],
        "freeze_sha256": {"full": actual_hashes, "source": manifest["source_artifact_sha256"]},
        "cases": cases,
        "challengers": challengers,
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
    return _AB_VALIDATE_CHECKPOINT(metadata, events, frozen, for_resume=for_resume)


def _append_prepared(
    checkpoint: base.Checkpoint, spec: base.CallSpec, index: int, frozen: dict[str, Any]
) -> None:
    checkpoint.append(
        "CALL_PREPARED", call_index=index, call_spec=asdict(spec),
        request_role=spec.role,
        relevant_hashes={
            "manifest_sha256": frozen["manifest_sha256"],
            "case_sha256": frozen["manifest"]["case_sha256"][spec.case_id],
            "challenger_sha256": frozen["manifest"]["challenger_sha256"][spec.case_id],
            "production_prompt_sha256": (
                frozen["manifest"]["prompt"]["sha256"]
                if spec.role == "production_decision" else None
            ),
        },
    )


def _run_production(
    client: Any, checkpoint: base.Checkpoint, spec: base.CallSpec, index: int,
    case: dict[str, Any], frozen: dict[str, Any],
) -> None:
    from ai.cognitive_core import CognitiveCore
    from Conversation_Lab.selected_action_evaluator import selected_action_from_cognitive_turn

    prompt_bytes = _git_bytes(frozen["manifest"]["prompt"]["git_path"])
    if hashlib.sha256(prompt_bytes).hexdigest().upper() != frozen["manifest"]["prompt"]["sha256"]:
        raise base.RunnerError("hash_failure", "Production prompt changed after preflight")
    prompt_text = prompt_bytes.decode("utf-8")
    FrozenCore = type(
        "FrozenPromptCognitiveCore", (CognitiveCore,),
        {"_system_prompt": staticmethod(lambda: prompt_text)},
    )
    current, human_model, previous_acs, history = base._load_production_input(case)

    def captured(response: Any, request: Any) -> None:
        base._append_response(checkpoint, spec, index, response, request)

    capturing = base._CapturingClient(client, captured)
    result = FrozenCore(capturing, model=spec.model).propose(
        current, human_model, previous_acs, history
    )
    if capturing.completions.calls != 1:
        raise base.RunnerError("protocol_failure", "Production slot made != 1 API call")
    checkpoint.append(
        "CALL_COMPLETED", call_index=index, call_spec=asdict(spec),
        normalized_result={
            "cognitive_turn_result": result,
            "observable_action": selected_action_from_cognitive_turn(result),
            "production_prompt_sha256": frozen["manifest"]["prompt"]["sha256"],
        },
    )


def _new_run(frozen: dict[str, Any], root: Optional[Path] = None):
    output_root = RUNS_ROOT if root is None else root
    run_id = f"fsbfull-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
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


def execute_live(
    frozen: dict[str, Any], metadata: dict[str, Any], checkpoint: base.Checkpoint
) -> None:
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
        completed = validate_checkpoint(metadata, events, frozen, for_resume=False)
        base._append_missing_aggregates(checkpoint, events, frozen["manifest"])
        events = checkpoint.read()
        completed = validate_checkpoint(metadata, events, frozen, for_resume=False)
        if len(completed) == 24:
            checkpoint.append(
                "RUN_COMPLETED", api_calls=24, production_calls=6, evaluator_calls=18,
            )
            print(f"RUN_COMPLETED={metadata['run_id']}", flush=True)
            print(f"CHECKPOINT={checkpoint.path}", flush=True)
            return
        spec = schedule[len(completed)]
        index = len(completed) + 1
        _append_prepared(checkpoint, spec, index, frozen)
        try:
            if spec.role == "production_decision":
                _run_production(client, checkpoint, spec, index, cases[spec.case_id], frozen)
            else:
                base._run_evaluator(
                    client, checkpoint, spec, index, frozen,
                    cases[spec.case_id], challengers[spec.case_id], events,
                )
        except BaseException as error:
            checkpoint.append(
                "STOP", category=getattr(error, "category", "api_failure"),
                call_index=index, call_spec=asdict(spec),
                error_type=type(error).__name__, error_message=str(error),
            )
            raise


def dry_run(frozen: dict[str, Any]) -> dict[str, Any]:
    result = _AB_DRY_RUN(frozen)
    return {
        "result": result["result"], "api_calls": 0, "head": HEAD,
        "cases": frozen["manifest"]["case_order"],
        "production_calls_simulated": 6, "evaluator_calls_simulated": 18,
        "maximum_call_budget": 24,
        "production_model": "gpt-5-mini", "evaluator_model": "gpt-6-astra",
        "all_production_precedes_evaluator": result["all_production_precedes_evaluator"],
        "checkpoint_hash_chain": result["checkpoint_hash_chain"],
        "clean_prefix_resume": result["clean_prefix_resume"],
        "dangling_call_resume_rejected": result["dangling_call_resume_rejected"],
        "out_of_order_call_rejected": result["out_of_order_call_rejected"],
        "confirmations": 0, "adaptive_sampling": "DISABLED",
    }


ab.HEAD = HEAD
ab.FREEZE = FREEZE
ab.SOURCE = SOURCE
ab.RUNS_ROOT = RUNS_ROOT
ab._specs = _specs
ab.preflight = preflight
ab.validate_checkpoint = validate_checkpoint
ab._append_prepared = _append_prepared
ab._new_run = _new_run
ab.execute_live = execute_live
ab.dry_run = dry_run

base.HEAD = HEAD
base.PACKAGE = SOURCE
base.RUNS_ROOT = RUNS_ROOT
base.EVALUATOR_PROMPT_NAME = "fsb_01_evaluator_prompt_672455a.txt"
base.EVALUATOR_SCHEMA_NAME = "fsb_01_evaluator_schema_672455a.json"
base._specs = _specs
base.validate_checkpoint = validate_checkpoint
base._append_prepared = _append_prepared
base.preflight = preflight
base._new_run = _new_run
base.execute_live = execute_live
base.dry_run = dry_run


if __name__ == "__main__":
    raise SystemExit(base.main())
