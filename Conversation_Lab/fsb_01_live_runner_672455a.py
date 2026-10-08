"""Resume-safe live runner for the frozen FSB-01 package at HEAD 672455a.

This is lab-only orchestration.  ``--dry-run`` performs the complete preflight
and exercises the checkpoint state machine without importing or calling the
OpenAI client.  Live execution must be explicitly selected with ``--new-run``
or ``--resume RUN_ID``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional

from pydantic import BaseModel


HEAD = "672455a"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PACKAGE = ROOT / "Conversation_Lab" / "frozen" / "fsb_01_672455a"
RUNS_ROOT = ROOT / "Conversation_Lab" / "traces" / "fsb_01_672455a_runs"
MANIFEST_NAME = "fsb_01_manifest_672455a.json"
CASES_NAME = "fsb_01_cases_672455a.json"
CHALLENGERS_NAME = "fsb_01_challengers_672455a.json"
VALIDATOR_NAME = "fsb_01_offline_validator_672455a.py"

FROZEN_SHA256 = {
    "fsb_01_cases_672455a.json": "E9CB63B9D54F19EBCEF06A592C83DF70C3E75561C934C0D453BD9D40E914CEA1",
    "fsb_01_challengers_672455a.json": "23F38F097ED4B2A30902D22EEE128CB88EAE9B446EDA5E8C9912F1752960AF51",
    "fsb_01_evaluator_prompt_672455a.txt": "C5ACFB34717B6F548CE7F0F65349E7F0C56DA3203580242E1B40A9BD3DB1FCF6",
    "fsb_01_evaluator_schema_672455a.json": "6D9B6D568D3F58C1D622F18B93FB3D428CA9631AB0B3C8029C134E9270FCB54B",
    "fsb_01_freeze_report_672455a.md": "2A8EBEC1D93014A96A21F8B75546D3AD2B180D70EF954243311BEA45DD8EC5D4",
    "fsb_01_manifest_672455a.json": "08A4CB3EC1AD94D2DB2A286F4698DDF3EA5504B7105FBEA72CD39EB8A309506A",
    "fsb_01_measurement_plan_672455a.md": "2A83EAB01B8F5D2E4E2BC08B82063453FD9CF00F452273206913CD24CDD425E0",
    "fsb_01_offline_validator_672455a.py": "B8F71D2DC24631D363E61FF0AABB41BC2806E499EF3979CC066B7FBC4EB837F1",
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


class RunnerError(RuntimeError):
    """A preflight, protocol, or structural boundary failed."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class CallSpec:
    key: str
    phase: str
    case_id: str
    role: str
    model: str
    repeat: Optional[int] = None
    selected_on_left: Optional[bool] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump(mode="json"))
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest().upper()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def preflight() -> dict[str, Any]:
    actual_head = _git_head()
    if actual_head != HEAD:
        raise RunnerError("head_failure", f"HEAD={actual_head}; expected {HEAD}")

    actual_hashes: dict[str, str] = {}
    actual_names = {path.name for path in PACKAGE.iterdir() if path.is_file()}
    if actual_names != set(FROZEN_SHA256):
        raise RunnerError(
            "hash_failure",
            f"Frozen file set mismatch: actual={sorted(actual_names)}",
        )
    for name, expected in FROZEN_SHA256.items():
        path = PACKAGE / name
        actual = _sha_file(path)
        if actual != expected:
            raise RunnerError(
                "hash_failure", f"SHA mismatch for {name}: {actual} != {expected}"
            )
        actual_hashes[name] = actual

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
    if validation.get("result") != "READY FOR LIVE RUN":
        raise RunnerError(
            "offline_validation_failure", f"Offline result={validation.get('result')}"
        )

    manifest = _read_json(PACKAGE / MANIFEST_NAME)
    if manifest.get("head") != HEAD or manifest.get("api_budget", {}).get("retry") != 0:
        raise RunnerError("protocol_failure", "Manifest HEAD/retry contract mismatch")
    expected_order = ["FSB-H", "FSB-M", "FSB-R", "FSB-S", "FSB-T", "FSB-A"]
    for field in (
        "case_order",
        "baseline_production_order",
        "baseline_evaluator_order",
        "conditional_confirmation_order",
    ):
        if manifest.get(field) != expected_order:
            raise RunnerError("protocol_failure", f"Unexpected manifest {field}")
    budget = manifest["api_budget"]
    if (
        budget.get("baseline_production") != 6
        or budget.get("baseline_evaluator") != 18
        or budget.get("maximum_production") != 12
        or budget.get("maximum_evaluator") != 36
        or budget.get("maximum_total") != 48
    ):
        raise RunnerError("protocol_failure", "Frozen call budget mismatch")

    cases = _read_json(PACKAGE / CASES_NAME)
    challengers = _read_json(PACKAGE / CHALLENGERS_NAME)
    if [item["case_id"] for item in cases["cases"]] != expected_order:
        raise RunnerError("protocol_failure", "Case document order mismatch")
    if [item["case_id"] for item in challengers["challengers"]] != expected_order:
        raise RunnerError("protocol_failure", "Challenger document order mismatch")
    return {
        "head": actual_head,
        "manifest_sha256": actual_hashes[MANIFEST_NAME],
        "freeze_sha256": actual_hashes,
        "manifest": manifest,
        "cases": cases,
        "challengers": challengers,
        "offline_validation": validation,
    }


class Checkpoint:
    """Hash-chained, append-only JSONL event store."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        previous: Optional[str] = None
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.endswith("\n"):
                    raise RunnerError(
                        "protocol_failure", f"Checkpoint line {line_number} is incomplete"
                    )
                event = json.loads(line)
                event_hash = event.pop("event_sha256", None)
                if event.get("sequence") != line_number:
                    raise RunnerError("protocol_failure", "Checkpoint sequence mismatch")
                if event.get("previous_event_sha256") != previous:
                    raise RunnerError("protocol_failure", "Checkpoint hash chain mismatch")
                expected = _sha_value(event)
                if event_hash != expected:
                    raise RunnerError("protocol_failure", "Checkpoint event hash mismatch")
                event["event_sha256"] = event_hash
                events.append(event)
                previous = event_hash
        return events

    def append(self, event_type: str, **payload: Any) -> dict[str, Any]:
        events = self.read()
        event = {
            "sequence": len(events) + 1,
            "timestamp": _utc_now(),
            "event_type": event_type,
            "previous_event_sha256": (
                events[-1]["event_sha256"] if events else None
            ),
            **_jsonable(payload),
        }
        event["event_sha256"] = _sha_value(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return event


def _base_specs(manifest: dict[str, Any]) -> list[CallSpec]:
    specs = [
        CallSpec(
            key=f"baseline.production.{case_id}",
            phase="baseline",
            case_id=case_id,
            role="production_decision",
            model=manifest["production_model"],
        )
        for case_id in manifest["baseline_production_order"]
    ]
    for case_id in manifest["baseline_evaluator_order"]:
        orientations = manifest["evaluator_orientations_selected_on_left"][case_id][
            "baseline"
        ]
        specs.extend(
            CallSpec(
                key=f"baseline.evaluator.{case_id}.{repeat}",
                phase="baseline",
                case_id=case_id,
                role="blind_evaluator",
                model=manifest["evaluator_model"],
                repeat=repeat,
                selected_on_left=selected_left,
            )
            for repeat, selected_left in enumerate(orientations, 1)
        )
    return specs


def _event_spec(event: dict[str, Any]) -> CallSpec:
    return CallSpec(**event["call_spec"])


def _completed_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event["event_type"] == "CALL_COMPLETED"]


def _aggregate_values(
    completed: Iterable[dict[str, Any]], phase: str, case_id: str
) -> list[str]:
    prefix = f"{phase}.evaluator.{case_id}."
    matching = [
        event
        for event in completed
        if event["call_spec"]["key"].startswith(prefix)
    ]
    matching.sort(key=lambda event: event["call_spec"]["repeat"])
    return [event["normalized_result"]["observable_verdict"] for event in matching]


def _aggregate(verdicts: list[str]) -> dict[str, Any]:
    from Conversation_Lab.selected_action_evaluator import (
        ObservableVerdict,
        aggregate_judgments,
    )

    if len(verdicts) != 3:
        raise RunnerError("protocol_failure", "Aggregate requires exactly 3 verdicts")
    result = aggregate_judgments(ObservableVerdict(value) for value in verdicts)
    return _jsonable(result)


def _confirmation_specs(
    manifest: dict[str, Any], completed: list[dict[str, Any]]
) -> list[CallSpec]:
    specs: list[CallSpec] = []
    for case_id in manifest["conditional_confirmation_order"]:
        verdicts = _aggregate_values(completed, "baseline", case_id)
        if len(verdicts) != 3:
            continue
        if _aggregate(verdicts)["verdict"] != "CHALLENGER_HIGHER":
            continue
        specs.append(
            CallSpec(
                key=f"confirmation.production.{case_id}",
                phase="confirmation",
                case_id=case_id,
                role="production_decision",
                model=manifest["production_model"],
            )
        )
        orientations = manifest["evaluator_orientations_selected_on_left"][case_id][
            "confirmation"
        ]
        specs.extend(
            CallSpec(
                key=f"confirmation.evaluator.{case_id}.{repeat}",
                phase="confirmation",
                case_id=case_id,
                role="blind_evaluator",
                model=manifest["evaluator_model"],
                repeat=repeat,
                selected_on_left=selected_left,
            )
            for repeat, selected_left in enumerate(orientations, 1)
        )
    return specs


def _allowed_specs(
    manifest: dict[str, Any], completed: list[dict[str, Any]]
) -> list[CallSpec]:
    return _base_specs(manifest) + _confirmation_specs(manifest, completed)


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
        raise RunnerError("protocol_failure", "Run freeze SHA set mismatch")
    if not events or events[0]["event_type"] != "RUN_STARTED":
        raise RunnerError("protocol_failure", "Missing RUN_STARTED event")
    if sum(event["event_type"] == "RUN_STARTED" for event in events) != 1:
        raise RunnerError("protocol_failure", "RUN_STARTED must occur exactly once")
    completed_markers = [
        index for index, event in enumerate(events) if event["event_type"] == "RUN_COMPLETED"
    ]
    if completed_markers and (
        len(completed_markers) != 1 or completed_markers[0] != len(events) - 1
    ):
        raise RunnerError("protocol_failure", "RUN_COMPLETED must be unique and final")

    stops = [event for event in events if event["event_type"] == "STOP"]
    if for_resume and any(
        event.get("category") in TERMINAL_STOP_CATEGORIES for event in stops
    ):
        raise RunnerError("protocol_failure", "Run has a terminal STOP event")
    if any(event["event_type"] == "RUN_COMPLETED" for event in events) and for_resume:
        raise RunnerError("protocol_failure", "Completed run cannot be resumed")

    prepared = [event for event in events if event["event_type"] == "CALL_PREPARED"]
    responses = [event for event in events if event["event_type"] == "API_RESPONSE"]
    completed = _completed_events(events)
    if len(responses) != len(completed):
        raise RunnerError(
            "protocol_failure", "Successful response lacks a completed validation record"
        )
    if len(prepared) != len(completed):
        raise RunnerError(
            "protocol_failure",
            "Ambiguous/in-flight call exists; resume would risk a duplicate request",
        )

    for index, (before, response, after) in enumerate(
        zip(prepared, responses, completed), 1
    ):
        if before.get("call_index") != index:
            raise RunnerError("protocol_failure", "Call indices are not contiguous")
        if response.get("call_index") != index or after.get("call_index") != index:
            raise RunnerError("protocol_failure", "Call event indices do not match")
        if before["call_spec"] != response["call_spec"] or before["call_spec"] != after["call_spec"]:
            raise RunnerError("protocol_failure", "Call spec changed within checkpoint")
        before_position = events.index(before)
        response_position = events.index(response)
        after_position = events.index(after)
        if not (
            response_position == before_position + 1
            and after_position == response_position + 1
        ):
            raise RunnerError(
                "protocol_failure",
                "Each call must be an adjacent PREPARED/RESPONSE/COMPLETED triple",
            )

    allowed = _allowed_specs(frozen["manifest"], completed)
    completed_specs = [_event_spec(event) for event in completed]
    if completed_specs != allowed[: len(completed_specs)]:
        raise RunnerError(
            "protocol_failure", "Completed calls are not an exact frozen-order prefix"
        )
    if len(completed_specs) > frozen["manifest"]["api_budget"]["maximum_total"]:
        raise RunnerError("protocol_failure", "Maximum API call budget exceeded")

    aggregate_events = [
        event for event in events if event["event_type"] == "AGGREGATE"
    ]
    aggregate_keys = [(event["phase"], event["case_id"]) for event in aggregate_events]
    if len(aggregate_keys) != len(set(aggregate_keys)):
        raise RunnerError("protocol_failure", "Duplicate aggregate event")
    for event in aggregate_events:
        verdicts = _aggregate_values(completed, event["phase"], event["case_id"])
        if len(verdicts) != 3:
            raise RunnerError("protocol_failure", "Aggregate precedes three judgments")
        if event.get("raw_normalized_verdicts") != verdicts:
            raise RunnerError("protocol_failure", "Aggregate inputs mismatch checkpoint")
        if event.get("aggregate") != _aggregate(verdicts):
            raise RunnerError("protocol_failure", "Aggregate result mismatch checkpoint")
    return completed


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(metadata, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _new_run(frozen: dict[str, Any], run_root: Path = RUNS_ROOT) -> tuple[str, Path, Checkpoint]:
    run_id = f"fsb01-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = run_root / run_id
    metadata = {
        "run_id": run_id,
        "created_at": _utc_now(),
        "head": HEAD,
        "manifest_sha256": frozen["manifest_sha256"],
        "freeze_sha256": frozen["freeze_sha256"],
        "retry": 0,
        "smoke_calls": 0,
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


def _load_run(run_id: str, frozen: dict[str, Any]) -> tuple[Path, dict[str, Any], Checkpoint, list[dict[str, Any]]]:
    if not run_id or Path(run_id).name != run_id:
        raise RunnerError("protocol_failure", "Invalid run id")
    run_dir = RUNS_ROOT / run_id
    metadata_path = run_dir / "run.json"
    checkpoint = Checkpoint(run_dir / "checkpoint.jsonl")
    if not metadata_path.is_file() or not checkpoint.path.is_file():
        raise RunnerError("protocol_failure", f"Unknown/incomplete run id: {run_id}")
    metadata = _read_json(metadata_path)
    events = checkpoint.read()
    validate_checkpoint(metadata, events, frozen, for_resume=True)
    return run_dir, metadata, checkpoint, events


def _case_maps(frozen: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    cases = {item["case_id"]: item for item in frozen["cases"]["cases"]}
    challengers = {
        item["case_id"]: item["challenger"]
        for item in frozen["challengers"]["challengers"]
    }
    return cases, challengers


def _usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    return {} if usage is None else _jsonable(usage)


class _CapturingCompletions:
    def __init__(self, base: Any, on_response: Any) -> None:
        self._base = base
        self._on_response = on_response
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        if self.calls != 0:
            raise RunnerError("protocol_failure", "More than one request in one call slot")
        self.calls += 1
        response = self._base.create(**kwargs)
        self._on_response(response, kwargs)
        return response


class _CapturingClient:
    def __init__(self, base: Any, on_response: Any) -> None:
        completions = _CapturingCompletions(base.chat.completions, on_response)
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


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


def _append_response(
    checkpoint: Checkpoint,
    spec: CallSpec,
    call_index: int,
    response: Any,
    request: Any,
) -> None:
    checkpoint.append(
        "API_RESPONSE",
        call_index=call_index,
        call_spec=asdict(spec),
        request_sha256=_sha_value(request),
        response_raw=_jsonable(response),
        token_usage=_usage(response),
    )


def _load_production_input(case: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    from ai.discovery_data_model import (
        ActiveConversationState,
        DialogueHistory,
        DialogueMessage,
        HumanModel,
        validate_discovery_memory,
    )

    frozen_input = case["frozen_input"]
    if frozen_input.get("human_experience") != {"enabled": False}:
        raise RunnerError("structural_failure", "Human Experience must remain disabled")
    human_model = HumanModel.model_validate(frozen_input["human_model"])
    previous = frozen_input["previous_acs"]
    previous_acs = (
        None if previous is None else ActiveConversationState.model_validate(previous)
    )
    history = DialogueHistory.model_validate(frozen_input["dialogue_history"])
    current = DialogueMessage.model_validate(frozen_input["current_user_message"])
    validate_discovery_memory(human_model, previous_acs, history)
    return current, human_model, previous_acs, history


def _production_selected(completed: list[dict[str, Any]], phase: str, case_id: str) -> dict[str, Any]:
    key = f"{phase}.production.{case_id}"
    matches = [
        event for event in completed if event["call_spec"]["key"] == key
    ]
    if len(matches) != 1:
        raise RunnerError("protocol_failure", f"Missing unique production result {key}")
    return matches[0]["normalized_result"]["observable_action"]


def _normalize_blind(verdict: str, selected_on_left: bool) -> str:
    if verdict == "EQUIVALENT_OR_BOTH_ACCEPTABLE":
        return "EQUIVALENT_OR_BOTH_ACCEPTABLE"
    if verdict == "INSUFFICIENT_BASIS":
        return "ABSTAIN"
    selected_higher = (
        verdict == "LEFT_HIGHER" if selected_on_left else verdict == "RIGHT_HIGHER"
    )
    return "SELECTED_HIGHER" if selected_higher else "CHALLENGER_HIGHER"


def _run_production(
    client: Any,
    checkpoint: Checkpoint,
    spec: CallSpec,
    call_index: int,
    case: dict[str, Any],
) -> None:
    from ai.cognitive_core import CognitiveCore
    from Conversation_Lab.selected_action_evaluator import (
        selected_action_from_cognitive_turn,
    )

    current, human_model, previous_acs, history = _load_production_input(case)

    def captured(response: Any, request: Any) -> None:
        _append_response(checkpoint, spec, call_index, response, request)

    capturing = _CapturingClient(client, captured)
    result = CognitiveCore(capturing, model=spec.model).propose(
        current, human_model, previous_acs, history
    )
    if capturing.completions.calls != 1:
        raise RunnerError("protocol_failure", "Production slot did not make exactly one call")
    observable = selected_action_from_cognitive_turn(result)
    checkpoint.append(
        "CALL_COMPLETED",
        call_index=call_index,
        call_spec=asdict(spec),
        normalized_result={
            "cognitive_turn_result": result,
            "observable_action": observable,
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
    prompt = (PACKAGE / "fsb_01_evaluator_prompt_672455a.txt").read_text(
        encoding="utf-8"
    )
    schema = _read_json(PACKAGE / "fsb_01_evaluator_schema_672455a.json")
    payload = {
        "frozen_context": case["frozen_input"],
        "left_action": left,
        "right_action": right,
    }
    return {
        "model": frozen["manifest"]["evaluator_model"],
        "messages": [
            {"role": "system", "content": prompt},
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
                "schema": schema,
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
    completed: list[dict[str, Any]],
) -> None:
    from Conversation_Lab.selected_action_evaluator import BlindVerdict

    if spec.repeat is None or spec.selected_on_left is None:
        raise RunnerError("protocol_failure", "Evaluator spec lacks orientation")
    selected = _production_selected(completed, spec.phase, spec.case_id)
    request = _evaluator_request(
        frozen, case, selected, challenger, spec.selected_on_left
    )
    response = client.chat.completions.create(**request)
    _append_response(checkpoint, spec, call_index, response, request)
    try:
        content = response.choices[0].message.content
        payload = json.loads(content)
        if set(payload) != {"verdict"}:
            raise ValueError("Evaluator payload has unexpected fields")
        blind = BlindVerdict(payload["verdict"])
    except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RunnerError("structural_failure", "Invalid evaluator response") from error
    checkpoint.append(
        "CALL_COMPLETED",
        call_index=call_index,
        call_spec=asdict(spec),
        normalized_result={
            "blind_verdict": blind.value,
            "observable_verdict": _normalize_blind(
                blind.value, spec.selected_on_left
            ),
        },
    )


def _append_missing_aggregates(
    checkpoint: Checkpoint,
    events: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    completed = _completed_events(events)
    existing = {
        (event["phase"], event["case_id"])
        for event in events
        if event["event_type"] == "AGGREGATE"
    }
    for phase in ("baseline", "confirmation"):
        for case_id in manifest["case_order"]:
            verdicts = _aggregate_values(completed, phase, case_id)
            if len(verdicts) == 3 and (phase, case_id) not in existing:
                checkpoint.append(
                    "AGGREGATE",
                    phase=phase,
                    case_id=case_id,
                    raw_normalized_verdicts=verdicts,
                    aggregate=_aggregate(verdicts),
                )


def execute_live(
    frozen: dict[str, Any],
    run_dir: Path,
    metadata: dict[str, Any],
    checkpoint: Checkpoint,
) -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(dotenv_path=ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RunnerError("api_failure", "OPENAI_API_KEY is missing")
    client = OpenAI(max_retries=0, timeout=120.0)
    cases, challengers = _case_maps(frozen)

    while True:
        events = checkpoint.read()
        completed = validate_checkpoint(metadata, events, frozen, for_resume=False)
        _append_missing_aggregates(checkpoint, events, frozen["manifest"])
        events = checkpoint.read()
        completed = validate_checkpoint(metadata, events, frozen, for_resume=False)
        allowed = _allowed_specs(frozen["manifest"], completed)
        if len(completed) == len(allowed):
            checkpoint.append(
                "RUN_COMPLETED",
                api_calls=len(completed),
                production_calls=sum(
                    event["call_spec"]["role"] == "production_decision"
                    for event in completed
                ),
                evaluator_calls=sum(
                    event["call_spec"]["role"] == "blind_evaluator"
                    for event in completed
                ),
            )
            print(f"RUN_COMPLETED={metadata['run_id']}", flush=True)
            print(f"CHECKPOINT={checkpoint.path}", flush=True)
            return

        spec = allowed[len(completed)]
        call_index = len(completed) + 1
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
                    completed,
                )
        except BaseException as error:
            category = getattr(error, "category", "api_failure")
            checkpoint.append(
                "STOP",
                category=category,
                call_index=call_index,
                call_spec=asdict(spec),
                error_type=type(error).__name__,
                error_message=str(error),
            )
            raise


def _dry_response_payload(role: str, verdict: Optional[str] = None) -> dict[str, Any]:
    return {
        "synthetic": True,
        "role": role,
        "verdict": verdict,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def dry_run(frozen: dict[str, Any]) -> dict[str, Any]:
    """Exercise new/resume/order/confirmation mechanics with zero API calls."""

    with tempfile.TemporaryDirectory(prefix="fsb01-dry-") as temporary:
        root = Path(temporary)
        run_id, run_dir, checkpoint = _new_run(frozen, root)
        metadata = _read_json(run_dir / "run.json")
        cases, _ = _case_maps(frozen)

        def synthetic_complete(spec: CallSpec, observable_verdict: Optional[str] = None) -> None:
            events = checkpoint.read()
            completed = validate_checkpoint(metadata, events, frozen, for_resume=False)
            index = len(completed) + 1
            _append_prepared(checkpoint, spec, index, frozen)
            checkpoint.append(
                "API_RESPONSE",
                call_index=index,
                call_spec=asdict(spec),
                request_sha256=_sha_value({"dry": spec.key}),
                response_raw=_dry_response_payload(spec.role, observable_verdict),
                token_usage={"total_tokens": 0},
            )
            normalized = (
                {
                    "cognitive_turn_result": {"dry_run": True},
                    "observable_action": {
                        "decision_intent": "dry_run",
                        "semantic_focus": f"dry {spec.case_id} {spec.phase}",
                        "basis_evidence_refs": [],
                    },
                }
                if spec.role == "production_decision"
                else {
                    "blind_verdict": "RIGHT_HIGHER",
                    "observable_verdict": observable_verdict,
                }
            )
            checkpoint.append(
                "CALL_COMPLETED",
                call_index=index,
                call_spec=asdict(spec),
                normalized_result=normalized,
            )

        base = _base_specs(frozen["manifest"])
        for spec in base:
            if spec.role == "production_decision":
                _load_production_input(cases[spec.case_id])
                synthetic_complete(spec)
            else:
                verdict = (
                    "CHALLENGER_HIGHER"
                    if spec.case_id == "FSB-H"
                    else "SELECTED_HIGHER"
                )
                synthetic_complete(spec, verdict)
        _append_missing_aggregates(checkpoint, checkpoint.read(), frozen["manifest"])

        completed = validate_checkpoint(
            metadata, checkpoint.read(), frozen, for_resume=False
        )
        allowed = _allowed_specs(frozen["manifest"], completed)
        confirmation = allowed[len(base) :]
        if len(confirmation) != 4 or any(
            spec.case_id != "FSB-H" for spec in confirmation
        ):
            raise RunnerError("protocol_failure", "Confirmation gating dry test failed")
        for spec in confirmation:
            synthetic_complete(
                spec,
                "CHALLENGER_HIGHER" if spec.role == "blind_evaluator" else None,
            )
        _append_missing_aggregates(checkpoint, checkpoint.read(), frozen["manifest"])
        completed = validate_checkpoint(
            metadata, checkpoint.read(), frozen, for_resume=False
        )
        if len(completed) != 28:
            raise RunnerError("protocol_failure", "Dry call budget simulation failed")

        # A clean exact prefix is resumable.
        partial_id, partial_dir, partial = _new_run(frozen, root)
        partial_meta = _read_json(partial_dir / "run.json")
        for spec in base[:2]:
            events = partial.read()
            done = validate_checkpoint(partial_meta, events, frozen, for_resume=False)
            index = len(done) + 1
            _append_prepared(partial, spec, index, frozen)
            partial.append(
                "API_RESPONSE",
                call_index=index,
                call_spec=asdict(spec),
                request_sha256=_sha_value({"dry": spec.key}),
                response_raw=_dry_response_payload(spec.role),
                token_usage={"total_tokens": 0},
            )
            partial.append(
                "CALL_COMPLETED",
                call_index=index,
                call_spec=asdict(spec),
                normalized_result={
                    "cognitive_turn_result": {"dry_run": True},
                    "observable_action": {
                        "decision_intent": "dry_run",
                        "semantic_focus": spec.key,
                        "basis_evidence_refs": [],
                    },
                },
            )
        validate_checkpoint(partial_meta, partial.read(), frozen, for_resume=True)

        # A prepared but uncompleted request must never be resumed.
        dangling_id, dangling_dir, dangling = _new_run(frozen, root)
        dangling_meta = _read_json(dangling_dir / "run.json")
        _append_prepared(dangling, base[0], 1, frozen)
        dangling_rejected = False
        try:
            validate_checkpoint(
                dangling_meta, dangling.read(), frozen, for_resume=True
            )
        except RunnerError:
            dangling_rejected = True
        if not dangling_rejected:
            raise RunnerError("protocol_failure", "Dangling call resume was accepted")

        # Structural STOP is terminal.
        stopped_id, stopped_dir, stopped = _new_run(frozen, root)
        stopped_meta = _read_json(stopped_dir / "run.json")
        stopped.append("STOP", category="structural_failure", error_message="dry")
        stop_rejected = False
        try:
            validate_checkpoint(stopped_meta, stopped.read(), frozen, for_resume=True)
        except RunnerError:
            stop_rejected = True
        if not stop_rejected:
            raise RunnerError("protocol_failure", "Terminal STOP resume was accepted")

        # An out-of-order/duplicate-like call key is not an admissible prefix.
        order_id, order_dir, order_checkpoint = _new_run(frozen, root)
        order_meta = _read_json(order_dir / "run.json")
        wrong = base[1]
        _append_prepared(order_checkpoint, wrong, 1, frozen)
        order_checkpoint.append(
            "API_RESPONSE",
            call_index=1,
            call_spec=asdict(wrong),
            request_sha256=_sha_value({"dry": wrong.key}),
            response_raw=_dry_response_payload(wrong.role),
            token_usage={"total_tokens": 0},
        )
        order_checkpoint.append(
            "CALL_COMPLETED",
            call_index=1,
            call_spec=asdict(wrong),
            normalized_result={
                "cognitive_turn_result": {"dry_run": True},
                "observable_action": {
                    "decision_intent": "dry_run",
                    "semantic_focus": wrong.key,
                    "basis_evidence_refs": [],
                },
            },
        )
        order_rejected = False
        try:
            validate_checkpoint(
                order_meta, order_checkpoint.read(), frozen, for_resume=True
            )
        except RunnerError:
            order_rejected = True
        if not order_rejected:
            raise RunnerError("protocol_failure", "Out-of-order prefix was accepted")

        # Resume binds to the exact HEAD, manifest, and complete freeze hash set.
        binding_rejections = 0
        for field in ("head", "manifest_sha256", "freeze_sha256"):
            altered = dict(partial_meta)
            altered[field] = "DRY_MISMATCH"
            try:
                validate_checkpoint(altered, partial.read(), frozen, for_resume=True)
            except RunnerError:
                binding_rejections += 1
        if binding_rejections != 3:
            raise RunnerError("protocol_failure", "Resume binding check failed")

        # The event hash chain detects edited checkpoint bytes.
        hash_id, hash_dir, hash_checkpoint = _new_run(frozen, root)
        original_text = hash_checkpoint.path.read_text(encoding="utf-8")
        with hash_checkpoint.path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(original_text.replace("RUN_STARTED", "RUN_TAMPERED", 1))
        tamper_rejected = False
        try:
            hash_checkpoint.read()
        except RunnerError:
            tamper_rejected = True
        if not tamper_rejected:
            raise RunnerError("protocol_failure", "Checkpoint tampering was accepted")

        return {
            "result": "PASS",
            "api_calls": 0,
            "head": frozen["head"],
            "frozen_files_verified": len(frozen["freeze_sha256"]),
            "offline_validator": frozen["offline_validation"]["result"],
            "baseline_calls_simulated": 24,
            "confirmation_calls_simulated": 4,
            "maximum_call_budget": frozen["manifest"]["api_budget"]["maximum_total"],
            "clean_prefix_resume": "PASS",
            "dangling_call_resume_rejected": "PASS",
            "terminal_stop_resume_rejected": "PASS",
            "out_of_order_call_rejected": "PASS",
            "resume_head_manifest_freeze_binding": "PASS",
            "checkpoint_tamper_rejected": "PASS",
            "checkpoint_hash_chain": "PASS",
            "temporary_run_ids": [
                run_id,
                partial_id,
                dangling_id,
                stopped_id,
                order_id,
                hash_id,
            ],
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
            run_dir, metadata, checkpoint, _ = _load_run(args.resume, frozen)
            print(f"RESUMING_RUN_ID={args.resume}", flush=True)
        execute_live(frozen, run_dir, metadata, checkpoint)
        return 0
    except RunnerError as error:
        print(f"STOP_CATEGORY={error.category}", file=sys.stderr, flush=True)
        print(f"STOP_MESSAGE={error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
