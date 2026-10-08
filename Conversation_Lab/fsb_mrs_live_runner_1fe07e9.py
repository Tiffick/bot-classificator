"""Resume-safe live runner for the frozen FSB-M/R/S policy regression.

Live execution is possible only through an explicit ``--new-run`` or
``--resume RUN_ID``.  ``--dry-run`` verifies the complete protocol and
checkpoint state machine without importing or calling the OpenAI client.
"""

from __future__ import annotations

import argparse
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


HEAD = "1fe07e9"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Conversation_Lab.fsb_01_live_runner_672455a import (  # noqa: E402
    CallSpec,
    Checkpoint,
    RunnerError,
    _CapturingClient,
    _aggregate,
    _append_response,
    _jsonable,
    _load_production_input,
    _normalize_blind,
    _read_json,
    _sha_file,
    _sha_value,
    _usage,
)


PACKAGE = ROOT / "Conversation_Lab" / "frozen" / "fsb_01_mrs_policy_patch_1fe07e9"
RUNS_ROOT = ROOT / "Conversation_Lab" / "traces" / "fsb_mrs_1fe07e9_runs"
MANIFEST_NAME = "fsb_mrs_manifest_1fe07e9.json"
CASES_NAME = "fsb_mrs_cases_1fe07e9.json"
CHALLENGERS_NAME = "fsb_mrs_challengers_1fe07e9.json"
VALIDATOR_NAME = "fsb_mrs_offline_validator_1fe07e9.py"
EVALUATOR_PROMPT_NAME = "fsb_mrs_evaluator_prompt_1fe07e9.txt"
EVALUATOR_SCHEMA_NAME = "fsb_mrs_evaluator_schema_1fe07e9.json"
BASE_RUNNER_PATH = ROOT / "Conversation_Lab" / "fsb_01_live_runner_672455a.py"
BASE_RUNNER_SHA256 = "00038CB5913D51C54A454DD3B3F0D5CF2444CA9B39E6C9BDC14C144637E562BF"
EVALUATOR_PATH = ROOT / "Conversation_Lab" / "selected_action_evaluator.py"
EVALUATOR_SHA256 = "8FB8B98D58911D9AEDAFD9A1B6778CB9ABEE3F81C7F17DAAF9AAF52E708B10D0"

FROZEN_SHA256 = {
    "fsb_mrs_cases_1fe07e9.json": "0607913232C57D389F48996E4305ADA1E0B0172A2D6E07A7E23E5C5F1CD5F0CD",
    "fsb_mrs_challengers_1fe07e9.json": "37EFD89E28A7F6DB089E498AB153F08B70913BD195F8643844011CD400C1C486",
    "fsb_mrs_evaluator_prompt_1fe07e9.txt": "C5ACFB34717B6F548CE7F0F65349E7F0C56DA3203580242E1B40A9BD3DB1FCF6",
    "fsb_mrs_evaluator_schema_1fe07e9.json": "6D9B6D568D3F58C1D622F18B93FB3D428CA9631AB0B3C8029C134E9270FCB54B",
    "fsb_mrs_freeze_report_1fe07e9.md": "9F916BBBECE09A5FFBAE54CDDFEFB8975162BCA4BCBF2316B6CC8726EBFFA293",
    "fsb_mrs_manifest_1fe07e9.json": "6D4942F9DFD66CA9D56602BABBE583A21984DD36AAE3999A38DE6960C6BAC717",
    "fsb_mrs_measurement_plan_1fe07e9.md": "11E4235C7A438D8A59ED6365DA7BDB55BACE1E2BCCA6648E55EA62A33A6CA24C",
    "fsb_mrs_offline_validator_1fe07e9.py": "AA9747D1279C99EB445A38D05AD727983799BA295AAC3E54263CD931BEDD791B",
}

TERMINAL_STOP_CATEGORIES = {
    "hash_failure",
    "head_failure",
    "offline_validation_failure",
    "protocol_failure",
    "structural_failure",
    "structured_output_failure",
    "validation_failure",
    "api_failure",
    "billing_failure",
    "transport_failure",
}


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def preflight() -> dict[str, Any]:
    actual_head = _git_head()
    if actual_head != HEAD:
        raise RunnerError("head_failure", f"HEAD={actual_head}; expected {HEAD}")

    actual_names = {path.name for path in PACKAGE.iterdir() if path.is_file()}
    if actual_names != set(FROZEN_SHA256):
        raise RunnerError(
            "hash_failure", f"Frozen file set mismatch: actual={sorted(actual_names)}"
        )
    actual_hashes: dict[str, str] = {}
    for name, expected in FROZEN_SHA256.items():
        actual = _sha_file(PACKAGE / name)
        if actual != expected:
            raise RunnerError(
                "hash_failure", f"SHA mismatch for {name}: {actual} != {expected}"
            )
        actual_hashes[name] = actual

    if _sha_file(BASE_RUNNER_PATH) != BASE_RUNNER_SHA256:
        raise RunnerError("hash_failure", "Shared checkpoint helper SHA mismatch")
    if _sha_file(EVALUATOR_PATH) != EVALUATOR_SHA256:
        raise RunnerError("hash_failure", "Selected-action evaluator SHA mismatch")

    validator = subprocess.run(
        [sys.executable, str(PACKAGE / VALIDATOR_NAME)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if validator.returncode != 0:
        raise RunnerError(
            "offline_validation_failure",
            f"Offline validator exit={validator.returncode}: {validator.stderr}",
        )
    try:
        validation = json.loads(validator.stdout)
    except json.JSONDecodeError as error:
        raise RunnerError(
            "offline_validation_failure", "Offline validator returned non-JSON output"
        ) from error
    if validation.get("result") != "READY":
        raise RunnerError(
            "offline_validation_failure", f"Offline result={validation.get('result')}"
        )

    manifest = _read_json(PACKAGE / MANIFEST_NAME)
    order = ["FSB-M", "FSB-R", "FSB-S"]
    if manifest.get("head") != HEAD or manifest.get("retry") != 0:
        raise RunnerError("protocol_failure", "Manifest HEAD/retry mismatch")
    if manifest.get("adaptive_confirmation") is not False:
        raise RunnerError("protocol_failure", "Adaptive confirmation must be disabled")
    for field in ("case_order", "production_order", "evaluator_order"):
        if manifest.get(field) != order:
            raise RunnerError("protocol_failure", f"Unexpected {field}")
    if manifest.get("api_budget") != {
        "production": 3,
        "evaluator": 9,
        "maximum_total": 12,
    }:
        raise RunnerError("protocol_failure", "Frozen call budget mismatch")
    if manifest.get("production_model") != "gpt-5-mini":
        raise RunnerError("protocol_failure", "Production model mismatch")
    if manifest.get("evaluator_model") != "gpt-6-astra":
        raise RunnerError("protocol_failure", "Evaluator model mismatch")
    if (
        manifest.get("contract_boundary", {}).get(
            "selected_action_evaluator_file_sha256"
        )
        != EVALUATOR_SHA256
    ):
        raise RunnerError("protocol_failure", "Evaluator implementation binding mismatch")

    cases = _read_json(PACKAGE / CASES_NAME)
    challengers = _read_json(PACKAGE / CHALLENGERS_NAME)
    if [item["case_id"] for item in cases["cases"]] != order:
        raise RunnerError("protocol_failure", "Case order mismatch")
    if [item["case_id"] for item in challengers["challengers"]] != order:
        raise RunnerError("protocol_failure", "Challenger order mismatch")
    return {
        "head": actual_head,
        "manifest_sha256": actual_hashes[MANIFEST_NAME],
        "freeze_sha256": actual_hashes,
        "manifest": manifest,
        "cases": cases,
        "challengers": challengers,
        "offline_validation": validation,
        "runner_dependency_sha256": BASE_RUNNER_SHA256,
        "evaluator_implementation_sha256": EVALUATOR_SHA256,
    }


def _specs(manifest: dict[str, Any]) -> list[CallSpec]:
    specs = [
        CallSpec(
            key=f"production.{case_id}",
            phase="focused",
            case_id=case_id,
            role="production_decision",
            model=manifest["production_model"],
        )
        for case_id in manifest["production_order"]
    ]
    for case_id in manifest["evaluator_order"]:
        orientations = manifest["evaluator_orientations_selected_on_left"][case_id]
        specs.extend(
            CallSpec(
                key=f"evaluator.{case_id}.{repeat}",
                phase="focused",
                case_id=case_id,
                role="blind_evaluator",
                model=manifest["evaluator_model"],
                repeat=repeat,
                selected_on_left=selected_on_left,
            )
            for repeat, selected_on_left in enumerate(orientations, 1)
        )
    return specs


def _completed(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event["event_type"] == "CALL_COMPLETED"]


def _event_spec(event: dict[str, Any]) -> CallSpec:
    return CallSpec(**event["call_spec"])


def _case_verdicts(events: list[dict[str, Any]], case_id: str) -> list[str]:
    matches = [
        event
        for event in _completed(events)
        if event["call_spec"]["role"] == "blind_evaluator"
        and event["call_spec"]["case_id"] == case_id
    ]
    matches.sort(key=lambda event: event["call_spec"]["repeat"])
    return [event["normalized_result"]["observable_verdict"] for event in matches]


def validate_checkpoint(
    metadata: dict[str, Any],
    events: list[dict[str, Any]],
    frozen: dict[str, Any],
    *,
    for_resume: bool,
) -> list[dict[str, Any]]:
    if metadata.get("head") != HEAD or metadata.get("head") != frozen["head"]:
        raise RunnerError("protocol_failure", "Run metadata HEAD mismatch")
    if metadata.get("manifest_sha256") != frozen["manifest_sha256"]:
        raise RunnerError("protocol_failure", "Run manifest SHA mismatch")
    if metadata.get("freeze_sha256") != frozen["freeze_sha256"]:
        raise RunnerError("protocol_failure", "Run freeze SHA mismatch")
    if metadata.get("runner_dependency_sha256") != frozen["runner_dependency_sha256"]:
        raise RunnerError("protocol_failure", "Run helper dependency SHA mismatch")
    if (
        metadata.get("evaluator_implementation_sha256")
        != frozen["evaluator_implementation_sha256"]
    ):
        raise RunnerError("protocol_failure", "Run evaluator implementation SHA mismatch")
    if not events or events[0]["event_type"] != "RUN_STARTED":
        raise RunnerError("protocol_failure", "Missing RUN_STARTED")
    if sum(event["event_type"] == "RUN_STARTED" for event in events) != 1:
        raise RunnerError("protocol_failure", "RUN_STARTED must be unique")

    completed_markers = [
        index for index, event in enumerate(events) if event["event_type"] == "RUN_COMPLETED"
    ]
    if completed_markers and (
        len(completed_markers) != 1 or completed_markers[0] != len(events) - 1
    ):
        raise RunnerError("protocol_failure", "RUN_COMPLETED must be unique and final")
    if completed_markers and for_resume:
        raise RunnerError("protocol_failure", "Completed run cannot be resumed")
    if for_resume and any(
        event["event_type"] == "STOP"
        and event.get("category") in TERMINAL_STOP_CATEGORIES
        for event in events
    ):
        raise RunnerError("protocol_failure", "Run has a terminal STOP")

    prepared = [event for event in events if event["event_type"] == "CALL_PREPARED"]
    responses = [event for event in events if event["event_type"] == "API_RESPONSE"]
    done = _completed(events)
    if len(prepared) != len(done) or len(responses) != len(done):
        raise RunnerError(
            "protocol_failure",
            "Dangling/partial call exists; resume would risk a duplicate request",
        )
    for index, (before, response, after) in enumerate(zip(prepared, responses, done), 1):
        if {before["call_index"], response["call_index"], after["call_index"]} != {index}:
            raise RunnerError("protocol_failure", "Call indices are not contiguous")
        if not (before["call_spec"] == response["call_spec"] == after["call_spec"]):
            raise RunnerError("protocol_failure", "Call spec changed within slot")
        positions = [events.index(before), events.index(response), events.index(after)]
        if positions != [positions[0], positions[0] + 1, positions[0] + 2]:
            raise RunnerError(
                "protocol_failure", "Call events must be adjacent PREPARED/RESPONSE/COMPLETED"
            )

    schedule = _specs(frozen["manifest"])
    completed_specs = [_event_spec(event) for event in done]
    if completed_specs != schedule[: len(completed_specs)]:
        raise RunnerError("protocol_failure", "Calls are not an exact frozen-order prefix")
    if len(done) > 12:
        raise RunnerError("protocol_failure", "Maximum API call budget exceeded")

    aggregates = [event for event in events if event["event_type"] == "AGGREGATE"]
    if len({event["case_id"] for event in aggregates}) != len(aggregates):
        raise RunnerError("protocol_failure", "Duplicate aggregate")
    for event in aggregates:
        values = _case_verdicts(events, event["case_id"])
        if len(values) != 3:
            raise RunnerError("protocol_failure", "Aggregate precedes three judgments")
        if event.get("raw_normalized_verdicts") != values:
            raise RunnerError("protocol_failure", "Aggregate inputs mismatch")
        if event.get("aggregate") != _aggregate(values):
            raise RunnerError("protocol_failure", "Aggregate result mismatch")
    return done


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _new_run(
    frozen: dict[str, Any], run_root: Path = RUNS_ROOT
) -> tuple[str, Path, Checkpoint]:
    run_id = (
        f"fsbmrs-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    run_dir = run_root / run_id
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "head": HEAD,
        "manifest_sha256": frozen["manifest_sha256"],
        "freeze_sha256": frozen["freeze_sha256"],
        "runner_dependency_sha256": frozen["runner_dependency_sha256"],
        "evaluator_implementation_sha256": frozen[
            "evaluator_implementation_sha256"
        ],
        "retry": 0,
        "smoke_calls": 0,
        "maximum_api_calls": 12,
    }
    _write_metadata(run_dir / "run.json", metadata)
    checkpoint = Checkpoint(run_dir / "checkpoint.jsonl")
    checkpoint.append(
        "RUN_STARTED",
        run_id=run_id,
        head=HEAD,
        manifest_sha256=frozen["manifest_sha256"],
    )
    return run_id, run_dir, checkpoint


def _load_run(
    run_id: str, frozen: dict[str, Any]
) -> tuple[Path, dict[str, Any], Checkpoint]:
    if not run_id or Path(run_id).name != run_id:
        raise RunnerError("protocol_failure", "Invalid run id")
    run_dir = RUNS_ROOT / run_id
    metadata_path = run_dir / "run.json"
    checkpoint = Checkpoint(run_dir / "checkpoint.jsonl")
    if not metadata_path.is_file() or not checkpoint.path.is_file():
        raise RunnerError("protocol_failure", f"Unknown/incomplete run: {run_id}")
    metadata = _read_json(metadata_path)
    validate_checkpoint(metadata, checkpoint.read(), frozen, for_resume=True)
    return run_dir, metadata, checkpoint


def _case_maps(frozen: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cases = {item["case_id"]: item for item in frozen["cases"]["cases"]}
    challengers = {
        item["case_id"]: item["challenger"]
        for item in frozen["challengers"]["challengers"]
    }
    return cases, challengers


def _append_prepared(
    checkpoint: Checkpoint,
    spec: CallSpec,
    call_index: int,
    frozen: dict[str, Any],
) -> None:
    checkpoint.append(
        "CALL_PREPARED",
        call_index=call_index,
        call_spec=asdict(spec),
        request_role=spec.role,
        relevant_hashes={
            "manifest_sha256": frozen["manifest_sha256"],
            "case_sha256": frozen["manifest"]["case_sha256"][spec.case_id],
            "challenger_sha256": frozen["manifest"]["challenger_sha256"][spec.case_id],
        },
    )


def _production_selected(events: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    key = f"production.{case_id}"
    matches = [
        event for event in _completed(events) if event["call_spec"]["key"] == key
    ]
    if len(matches) != 1:
        raise RunnerError("protocol_failure", f"Missing production result {key}")
    return matches[0]["normalized_result"]["observable_action"]


def _run_production(
    client: Any,
    checkpoint: Checkpoint,
    spec: CallSpec,
    call_index: int,
    case: dict[str, Any],
) -> None:
    from ai.cognitive_core import CognitiveCore
    from Conversation_Lab.selected_action_evaluator import selected_action_from_cognitive_turn

    current, human_model, previous_acs, history = _load_production_input(case)

    def captured(response: Any, request: Any) -> None:
        _append_response(checkpoint, spec, call_index, response, request)

    capturing = _CapturingClient(client, captured)
    result = CognitiveCore(capturing, model=spec.model).propose(
        current, human_model, previous_acs, history
    )
    if capturing.completions.calls != 1:
        raise RunnerError("protocol_failure", "Production slot made != 1 API call")
    checkpoint.append(
        "CALL_COMPLETED",
        call_index=call_index,
        call_spec=asdict(spec),
        normalized_result={
            "cognitive_turn_result": result,
            "observable_action": selected_action_from_cognitive_turn(result),
        },
    )


def _evaluator_request(
    frozen: dict[str, Any],
    case: dict[str, Any],
    selected: dict[str, Any],
    challenger: dict[str, Any],
    selected_on_left: bool,
) -> dict[str, Any]:
    left = selected if selected_on_left else challenger
    right = challenger if selected_on_left else selected
    payload = {
        "frozen_context": case["frozen_input"],
        "left_action": left,
        "right_action": right,
    }
    return {
        "model": frozen["manifest"]["evaluator_model"],
        "messages": [
            {
                "role": "system",
                "content": (PACKAGE / EVALUATOR_PROMPT_NAME).read_text(encoding="utf-8"),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "selected_action_pairwise_verdict",
                "strict": True,
                "schema": _read_json(PACKAGE / EVALUATOR_SCHEMA_NAME),
            },
        },
    }


def _run_evaluator(
    client: Any,
    checkpoint: Checkpoint,
    spec: CallSpec,
    call_index: int,
    frozen: dict[str, Any],
    case: dict[str, Any],
    challenger: dict[str, Any],
    events: list[dict[str, Any]],
) -> None:
    from Conversation_Lab.selected_action_evaluator import BlindVerdict

    if spec.repeat is None or spec.selected_on_left is None:
        raise RunnerError("protocol_failure", "Evaluator orientation is missing")
    request = _evaluator_request(
        frozen,
        case,
        _production_selected(events, spec.case_id),
        challenger,
        spec.selected_on_left,
    )
    response = client.chat.completions.create(**request)
    _append_response(checkpoint, spec, call_index, response, request)
    try:
        content = response.choices[0].message.content
        payload = json.loads(content)
        if set(payload) != {"verdict"}:
            raise ValueError("Unexpected evaluator fields")
        blind = BlindVerdict(payload["verdict"])
    except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RunnerError("structural_failure", "Invalid evaluator response") from error
    checkpoint.append(
        "CALL_COMPLETED",
        call_index=call_index,
        call_spec=asdict(spec),
        normalized_result={
            "blind_verdict": blind.value,
            "observable_verdict": _normalize_blind(blind.value, spec.selected_on_left),
        },
    )


def _append_missing_aggregates(
    checkpoint: Checkpoint, events: list[dict[str, Any]], manifest: dict[str, Any]
) -> None:
    existing = {
        event["case_id"] for event in events if event["event_type"] == "AGGREGATE"
    }
    for case_id in manifest["case_order"]:
        values = _case_verdicts(events, case_id)
        if len(values) == 3 and case_id not in existing:
            checkpoint.append(
                "AGGREGATE",
                case_id=case_id,
                raw_normalized_verdicts=values,
                aggregate=_aggregate(values),
            )


def execute_live(
    frozen: dict[str, Any], metadata: dict[str, Any], checkpoint: Checkpoint
) -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(dotenv_path=ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RunnerError("api_failure", "OPENAI_API_KEY is missing")
    client = OpenAI(max_retries=0, timeout=120.0)
    cases, challengers = _case_maps(frozen)
    schedule = _specs(frozen["manifest"])

    while True:
        events = checkpoint.read()
        done = validate_checkpoint(metadata, events, frozen, for_resume=False)
        _append_missing_aggregates(checkpoint, events, frozen["manifest"])
        events = checkpoint.read()
        done = validate_checkpoint(metadata, events, frozen, for_resume=False)
        if len(done) == len(schedule):
            checkpoint.append(
                "RUN_COMPLETED",
                api_calls=12,
                production_calls=3,
                evaluator_calls=9,
            )
            print(f"RUN_COMPLETED={metadata['run_id']}", flush=True)
            print(f"CHECKPOINT={checkpoint.path}", flush=True)
            return

        spec = schedule[len(done)]
        call_index = len(done) + 1
        _append_prepared(checkpoint, spec, call_index, frozen)
        try:
            if spec.role == "production_decision":
                _run_production(client, checkpoint, spec, call_index, cases[spec.case_id])
            else:
                _run_evaluator(
                    client,
                    checkpoint,
                    spec,
                    call_index,
                    frozen,
                    cases[spec.case_id],
                    challengers[spec.case_id],
                    events,
                )
        except BaseException as error:
            checkpoint.append(
                "STOP",
                category=getattr(error, "category", "api_failure"),
                call_index=call_index,
                call_spec=asdict(spec),
                error_type=type(error).__name__,
                error_message=str(error),
            )
            raise


def _synthetic_complete(
    checkpoint: Checkpoint,
    metadata: dict[str, Any],
    frozen: dict[str, Any],
    spec: CallSpec,
) -> None:
    done = validate_checkpoint(metadata, checkpoint.read(), frozen, for_resume=False)
    index = len(done) + 1
    _append_prepared(checkpoint, spec, index, frozen)
    checkpoint.append(
        "API_RESPONSE",
        call_index=index,
        call_spec=asdict(spec),
        request_sha256=_sha_value({"dry": spec.key}),
        response_raw={"synthetic": True, "role": spec.role},
        token_usage={"total_tokens": 0},
    )
    normalized = (
        {
            "cognitive_turn_result": {"dry_run": True},
            "observable_action": {
                "decision_intent": "dry_run",
                "semantic_focus": spec.key,
                "basis_evidence_refs": [],
            },
        }
        if spec.role == "production_decision"
        else {
            "blind_verdict": "EQUIVALENT_OR_BOTH_ACCEPTABLE",
            "observable_verdict": "EQUIVALENT_OR_BOTH_ACCEPTABLE",
        }
    )
    checkpoint.append(
        "CALL_COMPLETED",
        call_index=index,
        call_spec=asdict(spec),
        normalized_result=normalized,
    )


def dry_run(frozen: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="fsbmrs-dry-") as temporary:
        root = Path(temporary)
        run_id, run_dir, checkpoint = _new_run(frozen, root)
        metadata = _read_json(run_dir / "run.json")
        cases, _ = _case_maps(frozen)
        schedule = _specs(frozen["manifest"])
        if len(schedule) != 12:
            raise RunnerError("protocol_failure", "Schedule is not exactly 12 calls")
        for spec in schedule:
            if spec.role == "production_decision":
                _load_production_input(cases[spec.case_id])
            _synthetic_complete(checkpoint, metadata, frozen, spec)
        _append_missing_aggregates(checkpoint, checkpoint.read(), frozen["manifest"])
        done = validate_checkpoint(metadata, checkpoint.read(), frozen, for_resume=False)
        if len(done) != 12:
            raise RunnerError("protocol_failure", "Dry-run call count mismatch")

        # Exact completed prefix is safe to resume.
        _, prefix_dir, prefix = _new_run(frozen, root)
        prefix_meta = _read_json(prefix_dir / "run.json")
        for spec in schedule[:4]:
            _synthetic_complete(prefix, prefix_meta, frozen, spec)
        validate_checkpoint(prefix_meta, prefix.read(), frozen, for_resume=True)

        # PREPARED without a completed response is permanently ambiguous.
        _, dangling_dir, dangling = _new_run(frozen, root)
        dangling_meta = _read_json(dangling_dir / "run.json")
        _append_prepared(dangling, schedule[0], 1, frozen)
        try:
            validate_checkpoint(dangling_meta, dangling.read(), frozen, for_resume=True)
        except RunnerError:
            dangling_rejected = True
        else:
            dangling_rejected = False
        if not dangling_rejected:
            raise RunnerError("protocol_failure", "Dangling resume was accepted")

        # Terminal failures can never resume.
        _, stopped_dir, stopped = _new_run(frozen, root)
        stopped_meta = _read_json(stopped_dir / "run.json")
        stopped.append("STOP", category="structural_failure", error_message="dry")
        try:
            validate_checkpoint(stopped_meta, stopped.read(), frozen, for_resume=True)
        except RunnerError:
            stop_rejected = True
        else:
            stop_rejected = False
        if not stop_rejected:
            raise RunnerError("protocol_failure", "Terminal STOP resume was accepted")

        # A non-prefix call detects duplicate/out-of-order execution.
        _, order_dir, wrong = _new_run(frozen, root)
        wrong_meta = _read_json(order_dir / "run.json")
        _append_prepared(wrong, schedule[1], 1, frozen)
        wrong.append(
            "API_RESPONSE",
            call_index=1,
            call_spec=asdict(schedule[1]),
            request_sha256=_sha_value({"dry": schedule[1].key}),
            response_raw={"synthetic": True},
            token_usage={"total_tokens": 0},
        )
        wrong.append(
            "CALL_COMPLETED",
            call_index=1,
            call_spec=asdict(schedule[1]),
            normalized_result={"dry_run": True},
        )
        try:
            validate_checkpoint(wrong_meta, wrong.read(), frozen, for_resume=True)
        except RunnerError:
            order_rejected = True
        else:
            order_rejected = False
        if not order_rejected:
            raise RunnerError("protocol_failure", "Out-of-order call was accepted")

        # Metadata binds resume to exact HEAD/manifest/freeze.
        binding_rejections = 0
        for field in (
            "head",
            "manifest_sha256",
            "freeze_sha256",
            "runner_dependency_sha256",
            "evaluator_implementation_sha256",
        ):
            altered = dict(prefix_meta)
            altered[field] = "DRY_MISMATCH"
            try:
                validate_checkpoint(altered, prefix.read(), frozen, for_resume=True)
            except RunnerError:
                binding_rejections += 1
        if binding_rejections != 5:
            raise RunnerError("protocol_failure", "Resume binding test failed")

        # Any byte edit breaks the event hash chain.
        _, hash_dir, hash_checkpoint = _new_run(frozen, root)
        text = hash_checkpoint.path.read_text(encoding="utf-8")
        with hash_checkpoint.path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text.replace("RUN_STARTED", "RUN_TAMPERED", 1))
        try:
            hash_checkpoint.read()
        except RunnerError:
            tamper_rejected = True
        else:
            tamper_rejected = False
        if not tamper_rejected:
            raise RunnerError("protocol_failure", "Checkpoint tampering was accepted")

        return {
            "result": "PASS",
            "api_calls": 0,
            "head": frozen["head"],
            "frozen_files_verified": len(frozen["freeze_sha256"]),
            "offline_validator": frozen["offline_validation"]["result"],
            "production_calls_simulated": 3,
            "evaluator_calls_simulated": 9,
            "maximum_call_budget": 12,
            "adaptive_confirmation": "DISABLED",
            "clean_prefix_resume": "PASS",
            "dangling_call_resume_rejected": "PASS",
            "terminal_stop_resume_rejected": "PASS",
            "out_of_order_call_rejected": "PASS",
            "resume_head_manifest_freeze_binding": "PASS",
            "checkpoint_hash_chain": "PASS",
            "temporary_run_id": run_id,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--new-run", action="store_true")
    modes.add_argument("--resume", metavar="RUN_ID")
    modes.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        frozen = preflight()
        if args.dry_run:
            print(json.dumps(dry_run(frozen), ensure_ascii=False, indent=2), flush=True)
            return 0
        if args.new_run:
            run_id, run_dir, checkpoint = _new_run(frozen)
            metadata = _read_json(run_dir / "run.json")
            print(f"NEW_RUN_ID={run_id}", flush=True)
        else:
            run_dir, metadata, checkpoint = _load_run(args.resume, frozen)
            print(f"RESUMING_RUN_ID={args.resume}", flush=True)
        execute_live(frozen, metadata, checkpoint)
        return 0
    except RunnerError as error:
        print(f"STOP_CATEGORY={error.category}", file=sys.stderr, flush=True)
        print(f"STOP_MESSAGE={error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
