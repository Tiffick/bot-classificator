"""Resume-safe runner for the frozen matched FSB-M prompt A/B experiment."""

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

from Conversation_Lab import fsb_mrs_live_runner_1fe07e9 as base  # noqa: E402


FREEZE = ROOT / "Conversation_Lab" / "frozen" / "fsb_m_prompt_ab_1550ec2"
SOURCE = ROOT / "Conversation_Lab" / "frozen" / "fsb_01_mrs_policy_patch_1fe07e9"
RUNS_ROOT = ROOT / "Conversation_Lab" / "traces" / "fsb_m_prompt_ab_1550ec2_runs"
MANIFEST_NAME = "fsb_m_prompt_ab_manifest_1550ec2.json"
VALIDATOR_NAME = "fsb_m_prompt_ab_offline_validator_1550ec2.py"
SELF = Path(__file__).resolve()

FREEZE_SHA256 = {
    "fsb_m_prompt_ab_freeze_report_1550ec2.md": "2BF0D9F4532D74463366053082828E66CC09769BB3AB1340409EFBF118E290F4",
    "fsb_m_prompt_ab_manifest_1550ec2.json": "FE835721A4ABFACACB067E681B8EAEBAFB202A3CC396779CBA2F9D95FC87DCEB",
    "fsb_m_prompt_ab_offline_validator_1550ec2.py": "22371D4E5B893992F56AEC96AB9E9353C88EE098FAA7EBBDA68034A4F72B94BC",
}


def _git_bytes(spec: str) -> bytes:
    return subprocess.run(
        ["git", "show", spec], cwd=ROOT, check=True, capture_output=True
    ).stdout


def _specs(manifest: dict[str, Any]) -> list[base.CallSpec]:
    specs = [
        base.CallSpec(
            key=f"production.{sample_id}", phase="prompt_ab",
            case_id=sample_id, role="production_decision",
            model=manifest["production_model"]
        )
        for sample_id in manifest["production_order"]
    ]
    orientations = manifest["evaluator_orientations_selected_on_left"]
    for sample_id in manifest["evaluator_order"]:
        specs.extend(
            base.CallSpec(
                key=f"evaluator.{sample_id}.{repeat}", phase="prompt_ab",
                case_id=sample_id, role="blind_evaluator",
                model=manifest["evaluator_model"], repeat=repeat,
                selected_on_left=selected_on_left,
            )
            for repeat, selected_on_left in enumerate(orientations, 1)
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
    for arm_key in ("arm_a", "arm_b"):
        prompt = _git_bytes(manifest[arm_key]["prompt_git_path"])
        if hashlib.sha256(prompt).hexdigest().upper() != manifest[arm_key]["prompt_sha256"]:
            raise base.RunnerError("hash_failure", f"Prompt SHA mismatch: {arm_key}")
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

    source_case = next(
        item for item in base._read_json(SOURCE / "fsb_mrs_cases_1fe07e9.json")["cases"]
        if item["case_id"] == "FSB-M"
    )
    source_challenger = next(
        item for item in base._read_json(SOURCE / "fsb_mrs_challengers_1fe07e9.json")["challengers"]
        if item["case_id"] == "FSB-M"
    )
    cases, challengers = [], []
    for sample_id in manifest["production_order"]:
        case = json.loads(json.dumps(source_case))
        case["case_id"] = sample_id
        cases.append(case)
        challenger = json.loads(json.dumps(source_challenger))
        challenger["case_id"] = sample_id
        challengers.append(challenger)
    schedule = _specs(manifest)
    if len(schedule) != 24 or any(
        spec.role != "production_decision" for spec in schedule[:6]
    ) or any(spec.role != "blind_evaluator" for spec in schedule[6:]):
        raise base.RunnerError("protocol_failure", "Frozen 6+18 schedule mismatch")
    return {
        "head": actual_head,
        "manifest_sha256": freeze_hashes[MANIFEST_NAME],
        "freeze_sha256": {"ab": freeze_hashes, "source": manifest["source_artifact_sha256"]},
        "manifest": manifest,
        "cases": {"cases": cases},
        "challengers": {"challengers": challengers},
        "offline_validation": validation,
        "runner_dependency_sha256": base._sha_file(SELF),
        "evaluator_implementation_sha256": base._sha_file(
            ROOT / "Conversation_Lab" / "selected_action_evaluator.py"
        ),
    }


def _case_verdicts(events: list[dict[str, Any]], case_id: str) -> list[str]:
    matches = [
        event for event in events
        if event["event_type"] == "CALL_COMPLETED"
        and event["call_spec"]["role"] == "blind_evaluator"
        and event["call_spec"]["case_id"] == case_id
    ]
    matches.sort(key=lambda event: event["call_spec"]["repeat"])
    return [event["normalized_result"]["observable_verdict"] for event in matches]


def validate_checkpoint(
    metadata: dict[str, Any], events: list[dict[str, Any]], frozen: dict[str, Any],
    *, for_resume: bool
) -> list[dict[str, Any]]:
    if metadata.get("head") != HEAD or metadata.get("head") != frozen["head"]:
        raise base.RunnerError("protocol_failure", "Run metadata HEAD mismatch")
    for field in (
        "manifest_sha256", "freeze_sha256", "runner_dependency_sha256",
        "evaluator_implementation_sha256"
    ):
        if metadata.get(field) != frozen[field]:
            raise base.RunnerError("protocol_failure", f"Run {field} mismatch")
    if metadata.get("maximum_api_calls") != 24:
        raise base.RunnerError("protocol_failure", "Run budget metadata mismatch")
    if not events or events[0]["event_type"] != "RUN_STARTED":
        raise base.RunnerError("protocol_failure", "Missing RUN_STARTED")
    if sum(event["event_type"] == "RUN_STARTED" for event in events) != 1:
        raise base.RunnerError("protocol_failure", "RUN_STARTED must be unique")
    completed_markers = [i for i, event in enumerate(events) if event["event_type"] == "RUN_COMPLETED"]
    if completed_markers and (len(completed_markers) != 1 or completed_markers[0] != len(events) - 1):
        raise base.RunnerError("protocol_failure", "RUN_COMPLETED must be unique and final")
    if completed_markers and for_resume:
        raise base.RunnerError("protocol_failure", "Completed run cannot be resumed")
    if for_resume and any(
        event["event_type"] == "STOP" and event.get("category") in base.TERMINAL_STOP_CATEGORIES
        for event in events
    ):
        raise base.RunnerError("protocol_failure", "Run has a terminal STOP")

    prepared = [e for e in events if e["event_type"] == "CALL_PREPARED"]
    responses = [e for e in events if e["event_type"] == "API_RESPONSE"]
    done = [e for e in events if e["event_type"] == "CALL_COMPLETED"]
    if len(prepared) != len(responses) or len(responses) != len(done):
        raise base.RunnerError(
            "protocol_failure", "Dangling/partial call could cause duplicate request"
        )
    for index, (before, response, after) in enumerate(zip(prepared, responses, done), 1):
        if {before["call_index"], response["call_index"], after["call_index"]} != {index}:
            raise base.RunnerError("protocol_failure", "Non-contiguous call indices")
        if not (before["call_spec"] == response["call_spec"] == after["call_spec"]):
            raise base.RunnerError("protocol_failure", "Call spec changed within slot")
        positions = [events.index(before), events.index(response), events.index(after)]
        if positions != [positions[0], positions[0] + 1, positions[0] + 2]:
            raise base.RunnerError("protocol_failure", "Call events are not adjacent")
    schedule = _specs(frozen["manifest"])
    if [base.CallSpec(**e["call_spec"]) for e in done] != schedule[:len(done)]:
        raise base.RunnerError("protocol_failure", "Calls are not a frozen-order prefix")
    if len(done) > 24:
        raise base.RunnerError("protocol_failure", "Maximum API budget exceeded")
    aggregates = [e for e in events if e["event_type"] == "AGGREGATE"]
    if len({e["case_id"] for e in aggregates}) != len(aggregates):
        raise base.RunnerError("protocol_failure", "Duplicate aggregate")
    for event in aggregates:
        values = _case_verdicts(events, event["case_id"])
        if len(values) != 3 or event.get("raw_normalized_verdicts") != values:
            raise base.RunnerError("protocol_failure", "Aggregate inputs mismatch")
        if event.get("aggregate") != base._aggregate(values):
            raise base.RunnerError("protocol_failure", "Aggregate result mismatch")
    return done


def _append_prepared(
    checkpoint: base.Checkpoint, spec: base.CallSpec, call_index: int,
    frozen: dict[str, Any]
) -> None:
    checkpoint.append(
        "CALL_PREPARED", call_index=call_index, call_spec=asdict(spec),
        request_role=spec.role,
        relevant_hashes={
            "manifest_sha256": frozen["manifest_sha256"],
            "case_sha256": frozen["manifest"]["case_sha256"],
            "challenger_sha256": frozen["manifest"]["challenger_sha256"],
            "production_prompt_sha256": (
                frozen["manifest"]["arm_a"]["prompt_sha256"]
                if "-A" in spec.case_id else frozen["manifest"]["arm_b"]["prompt_sha256"]
            ) if spec.role == "production_decision" else None,
        },
    )


def _run_production(
    client: Any, checkpoint: base.Checkpoint, spec: base.CallSpec,
    call_index: int, case: dict[str, Any], frozen: dict[str, Any]
) -> None:
    from ai.cognitive_core import CognitiveCore
    from Conversation_Lab.selected_action_evaluator import selected_action_from_cognitive_turn

    arm = frozen["manifest"]["arm_a"] if "-A" in spec.case_id else frozen["manifest"]["arm_b"]
    prompt_bytes = _git_bytes(arm["prompt_git_path"])
    if hashlib.sha256(prompt_bytes).hexdigest().upper() != arm["prompt_sha256"]:
        raise base.RunnerError("hash_failure", "Production prompt changed after preflight")
    prompt_text = prompt_bytes.decode("utf-8")
    FrozenPromptCore = type(
        "FrozenPromptCognitiveCore", (CognitiveCore,),
        {"_system_prompt": staticmethod(lambda: prompt_text)}
    )
    current, model, acs, history = base._load_production_input(case)

    def captured(response: Any, request: Any) -> None:
        base._append_response(checkpoint, spec, call_index, response, request)

    capturing = base._CapturingClient(client, captured)
    result = FrozenPromptCore(capturing, model=spec.model).propose(current, model, acs, history)
    if capturing.completions.calls != 1:
        raise base.RunnerError("protocol_failure", "Production slot made != 1 API call")
    checkpoint.append(
        "CALL_COMPLETED", call_index=call_index, call_spec=asdict(spec),
        normalized_result={
            "cognitive_turn_result": result,
            "observable_action": selected_action_from_cognitive_turn(result),
            "production_prompt_sha256": arm["prompt_sha256"],
            "arm": arm["label"],
        },
    )


def _new_run(frozen: dict[str, Any], run_root: Optional[Path] = None):
    root = RUNS_ROOT if run_root is None else run_root
    run_id = f"fsbmab-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = root / run_id
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "head": HEAD, "manifest_sha256": frozen["manifest_sha256"],
        "freeze_sha256": frozen["freeze_sha256"],
        "runner_dependency_sha256": frozen["runner_dependency_sha256"],
        "evaluator_implementation_sha256": frozen["evaluator_implementation_sha256"],
        "retry": 0, "smoke_calls": 0, "maximum_api_calls": 24,
    }
    base._write_metadata(run_dir / "run.json", metadata)
    checkpoint = base.Checkpoint(run_dir / "checkpoint.jsonl")
    checkpoint.append("RUN_STARTED", run_id=run_id, head=HEAD, manifest_sha256=frozen["manifest_sha256"])
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
        done = validate_checkpoint(metadata, events, frozen, for_resume=False)
        base._append_missing_aggregates(checkpoint, events, frozen["manifest"])
        events = checkpoint.read()
        done = validate_checkpoint(metadata, events, frozen, for_resume=False)
        if len(done) == 24:
            checkpoint.append("RUN_COMPLETED", api_calls=24, production_calls=6, evaluator_calls=18)
            print(f"RUN_COMPLETED={metadata['run_id']}", flush=True)
            print(f"CHECKPOINT={checkpoint.path}", flush=True)
            return
        spec = schedule[len(done)]
        index = len(done) + 1
        _append_prepared(checkpoint, spec, index, frozen)
        try:
            if spec.role == "production_decision":
                _run_production(client, checkpoint, spec, index, cases[spec.case_id], frozen)
            else:
                base._run_evaluator(
                    client, checkpoint, spec, index, frozen,
                    cases[spec.case_id], challengers[spec.case_id], events
                )
        except BaseException as error:
            checkpoint.append(
                "STOP", category=getattr(error, "category", "api_failure"),
                call_index=index, call_spec=asdict(spec), error_type=type(error).__name__,
                error_message=str(error)
            )
            raise


def dry_run(frozen: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="fsbm-ab-dry-") as temporary:
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
        if len(validate_checkpoint(metadata, checkpoint.read(), frozen, for_resume=False)) != 24:
            raise base.RunnerError("protocol_failure", "Dry-run count mismatch")

        _, prefix_dir, prefix = _new_run(frozen, root)
        prefix_meta = base._read_json(prefix_dir / "run.json")
        for spec in schedule[:7]:
            base._synthetic_complete(prefix, prefix_meta, frozen, spec)
        validate_checkpoint(prefix_meta, prefix.read(), frozen, for_resume=True)

        _, dangling_dir, dangling = _new_run(frozen, root)
        dangling_meta = base._read_json(dangling_dir / "run.json")
        _append_prepared(dangling, schedule[0], 1, frozen)
        try:
            validate_checkpoint(dangling_meta, dangling.read(), frozen, for_resume=True)
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
            validate_checkpoint(wrong_meta, wrong.read(), frozen, for_resume=True)
        except base.RunnerError:
            order_rejected = True
        else:
            order_rejected = False
        if not order_rejected:
            raise base.RunnerError("protocol_failure", "Out-of-order call was accepted")

        _, hash_dir, hash_checkpoint = _new_run(frozen, root)
        text = hash_checkpoint.path.read_text(encoding="utf-8")
        hash_checkpoint.path.write_text(text.replace("RUN_STARTED", "RUN_TAMPERED", 1), encoding="utf-8")
        try:
            hash_checkpoint.read()
        except base.RunnerError:
            tamper_rejected = True
        else:
            tamper_rejected = False
        if not tamper_rejected:
            raise base.RunnerError("protocol_failure", "Hash-chain tampering was accepted")

        return {
            "result": "PASS", "api_calls": 0, "head": HEAD,
            "samples_per_arm": 3, "production_calls_simulated": 6,
            "evaluator_calls_simulated": 18, "maximum_call_budget": 24,
            "both_arms_model": "gpt-6-astra", "adaptive_sampling": "DISABLED",
            "all_production_precedes_evaluator": "PASS",
            "checkpoint_hash_chain": "PASS", "clean_prefix_resume": "PASS",
            "dangling_call_resume_rejected": "PASS", "out_of_order_call_rejected": "PASS"
        }


base.HEAD = HEAD
base.PACKAGE = SOURCE
base.RUNS_ROOT = RUNS_ROOT
base.EVALUATOR_PROMPT_NAME = "fsb_mrs_evaluator_prompt_1fe07e9.txt"
base.EVALUATOR_SCHEMA_NAME = "fsb_mrs_evaluator_schema_1fe07e9.json"
base._specs = _specs
base.validate_checkpoint = validate_checkpoint
base._append_prepared = _append_prepared
base.preflight = preflight
base._new_run = _new_run
base.execute_live = execute_live
base.dry_run = dry_run


if __name__ == "__main__":
    raise SystemExit(base.main())
