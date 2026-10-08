"""Resume-safe runner for the frozen FSB-D D-only matched model swap.

No mode is implicit. ``--dry-run`` is offline; ``--new-run`` and ``--resume``
are the only modes that may construct an OpenAI client and consume live calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


HEAD = "1e08316"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Conversation_Lab import fsb_01_live_runner_672455a as base  # noqa: E402


FREEZE = ROOT / "Conversation_Lab/frozen/fsb_d_model_swap_1e08316"
RUNS_ROOT = ROOT / "Conversation_Lab/traces/fsb_d_model_swap_1e08316_runs"
MANIFEST_NAME = "fsb_d_model_swap_manifest_1e08316.json"
VALIDATOR_NAME = "fsb_d_model_swap_offline_validator_1e08316.py"
SELF = Path(__file__).resolve()
BASE_RUNNER = ROOT / "Conversation_Lab/fsb_01_live_runner_672455a.py"
BASE_RUNNER_SHA256 = "00038CB5913D51C54A454DD3B3F0D5CF2444CA9B39E6C9BDC14C144637E562BF"
FREEZE_SHA256 = {
    "fsb_d_model_swap_freeze_report_1e08316.md": "C66F7688ACBD91DB1B590F5B5718C6E478475D5D4D8C8418FEE612582E2B40A4",
    "fsb_d_model_swap_manifest_1e08316.json": "F1A0A94C687EEB191EBD30C35CCBCC4A5613C4C685F512F805E40104E3B14323",
    "fsb_d_model_swap_measurement_plan_1e08316.md": "BD874BABE8BFBD00816C82D3C729F23499B29E943EBF39CC648521B0E7509530",
    "fsb_d_model_swap_offline_validator_1e08316.py": "51634EA0300D5ECEE70A2A8207C6228053E58FD4E8BCED3923870B2F427CDB4F",
}


def _git_bytes(spec: str) -> bytes:
    return subprocess.run(
        ["git", "show", spec], cwd=ROOT, check=True, capture_output=True
    ).stdout


def _specs(manifest: dict[str, Any]) -> list[base.CallSpec]:
    return [
        base.CallSpec(
            key=f"production.{sample_id}", phase="model_swap",
            case_id=sample_id, role="production_decision",
            model=manifest["production_model"],
        )
        for sample_id in manifest["sample_order"]
    ]


def _request_without_model(request: dict[str, Any]) -> dict[str, Any]:
    value = dict(request)
    value.pop("model", None)
    return value


def _expected_requests(manifest: dict[str, Any], case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    from ai.cognitive_core import COGNITIVE_TURN_JSON_SCHEMA, CognitiveCore

    current, human_model, previous_acs, history = base._load_production_input(case)
    shared = {
        "messages": CognitiveCore._messages(
            current, human_model, previous_acs, history
        ),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "cognitive_turn_result",
                "strict": True,
                "schema": COGNITIVE_TURN_JSON_SCHEMA,
            },
        },
    }
    return (
        {"model": manifest["baseline_model"], **shared},
        {"model": manifest["production_model"], **shared},
    )


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
    if base._sha_file(BASE_RUNNER) != BASE_RUNNER_SHA256:
        raise base.RunnerError("hash_failure", "Checkpoint/input helper dependency changed")

    validator = subprocess.run(
        [sys.executable, "-B", str(FREEZE / VALIDATOR_NAME)], cwd=ROOT,
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

    manifest = base._read_json(FREEZE / MANIFEST_NAME)
    if manifest["head"] != HEAD or manifest["api_budget"] != {
        "production": 3, "evaluator": 0, "maximum_total": 3,
    }:
        raise base.RunnerError("protocol_failure", "Manifest HEAD/budget mismatch")
    if manifest["production_model"] != "gpt-6-astra" or len(_specs(manifest)) != 3:
        raise base.RunnerError("protocol_failure", "Astra schedule mismatch")

    case_path = ROOT / manifest["reused_frozen_artifacts"]["case_path"]
    case = base._read_json(case_path)
    mini_request, astra_request = _expected_requests(manifest, case)
    identity = manifest["request_identity"]
    if base._sha_value(mini_request) != identity["historical_mini_request_sha256"]:
        raise base.RunnerError("protocol_failure", "Historical mini request drift")
    if base._sha_value(astra_request) != identity["expected_astra_request_sha256"]:
        raise base.RunnerError("protocol_failure", "Expected Astra request drift")
    if base._sha_value(_request_without_model(astra_request)) != identity["model_excluded_request_sha256"]:
        raise base.RunnerError("protocol_failure", "Model-excluded request drift")
    differing = sorted(key for key in mini_request if mini_request[key] != astra_request[key])
    if differing != identity["allowed_changed_request_keys"]:
        raise base.RunnerError("protocol_failure", f"Unexpected request difference: {differing}")

    return {
        "head": actual_head,
        "manifest": manifest,
        "manifest_sha256": actual_hashes[MANIFEST_NAME],
        "freeze_sha256": actual_hashes,
        "offline_validation": validation,
        "case": case,
        "runner_sha256": base._sha_file(SELF),
        "base_runner_sha256": BASE_RUNNER_SHA256,
        "matched_request_difference": differing,
    }


def _projection(result: Any) -> dict[str, str]:
    contents = tuple(result.system_action.contents)
    targets = tuple(result.system_action.response_targets)
    return {
        "decision_intent": str(getattr(result.decision_intent, "value", result.decision_intent)),
        "action_content_kind": " + ".join(
            str(getattr(content.kind, "value", content.kind)) for content in contents
        ),
        "semantic_content": "\n".join(content.semantic_content for content in contents),
        "response_target_interaction": " + ".join(
            str(getattr(target.interaction, "value", target.interaction))
            for target in targets
        ) if targets else "none",
        "reply_text": "\n".join(segment.text for segment in result.reply_segments),
    }


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().replace("ё", "е")).strip()


def classify_observable(projection: dict[str, str]) -> str:
    """Closed deterministic semantic classifier for the frozen FSB-D context."""
    intent = projection["decision_intent"]
    kinds = set(projection["action_content_kind"].split(" + "))
    interactions = set(projection["response_target_interaction"].split(" + "))
    semantic = _normalized_text(projection["semantic_content"])
    reply = _normalized_text(projection["reply_text"])
    text = f"{semantic} {reply}"

    appetite = bool(re.search(
        r"желан\w*\s+(?:поест|есть)|хотел\w*\s+(?:поест|есть)|"
        r"тян\w*(?:\s+к)?\s+ед|голод\w*|аппетит\w*", text
    ))
    before = bool(re.search(
        r"(?:до|перед)(?:\s+\w+){0,4}\s+разговор\w*|до\s+него", text
    ))
    after = bool(re.search(
        r"после(?:\s+\w+){0,4}\s+разговор\w*|после\s+него", text
    ))
    question_cue = "?" in reply or bool(re.search(
        r"уточн\w*|вспомн\w*|мыслен\w*|было\s+ли|появил\w*\s+ли|"
        r"возник\w*\s+ли", text
    ))
    if (
        intent == "mechanism_discovery"
        and "system_question" in kinds
        and bool(interactions.intersection({"open_response", "clarification"}))
        and appetite and before and after and question_cue
    ):
        return "DIRECT_DISCRIMINATION"

    working_version = bool(re.search(
        r"физическ\w*\s+голод|эмоциональн\w*|напряж\w*|стресс\w*|"
        r"дв[еи]\s+(?:вещ|верс|механ)|могл\w*\s+(?:одновременно\s+)?действ", text
    ))
    known_elements = sum((
        bool(re.search(r"пропуст\w*\s+обед", text)),
        bool(re.search(r"тяжел\w*\s+разговор", text)),
        bool(re.search(r"переел\w*", text)),
        bool(
            re.search(r"не\s+(?:увер\w*|сопостав\w*|помн\w*)", text)
            and before and after
        ),
    ))
    if (
        intent == "reflection"
        and "system_reflection" in kinds
        and known_elements >= 2
        and not working_version
    ):
        return "LOCAL_REFLECTION"
    return "OTHER"


def _verdict(classes: list[str]) -> str:
    direct = classes.count("DIRECT_DISCRIMINATION")
    local = classes.count("LOCAL_REFLECTION")
    if direct == 3:
        return "STRONG_MODEL_SENSITIVITY_SUPPORT"
    if direct == 2:
        return "MODEL_SENSITIVITY_SUPPORT"
    if local == 3:
        return "MINI_SPECIFICITY_FALSIFIED"
    return "MIXED_MODEL_EXECUTION"


def _completed(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event["event_type"] == "CALL_COMPLETED"]


def validate_checkpoint(
    metadata: dict[str, Any], events: list[dict[str, Any]], frozen: dict[str, Any],
    *, for_resume: bool,
) -> list[dict[str, Any]]:
    for field in (
        "head", "manifest_sha256", "freeze_sha256", "runner_sha256",
        "base_runner_sha256",
    ):
        expected = frozen[field]
        if metadata.get(field) != expected:
            raise base.RunnerError("protocol_failure", f"Run {field} mismatch")
    if metadata.get("maximum_api_calls") != 3 or metadata.get("retry") != 0:
        raise base.RunnerError("protocol_failure", "Run budget/retry metadata mismatch")
    if not events or events[0]["event_type"] != "RUN_STARTED":
        raise base.RunnerError("protocol_failure", "Missing RUN_STARTED")
    if sum(event["event_type"] == "RUN_STARTED" for event in events) != 1:
        raise base.RunnerError("protocol_failure", "RUN_STARTED must be unique")
    markers = [i for i, event in enumerate(events) if event["event_type"] == "RUN_COMPLETED"]
    if markers and (len(markers) != 1 or markers[0] != len(events) - 1):
        raise base.RunnerError("protocol_failure", "RUN_COMPLETED must be unique and final")
    if markers and for_resume:
        raise base.RunnerError("protocol_failure", "Completed run cannot be resumed")
    if any(event["event_type"] == "STOP" for event in events) and for_resume:
        raise base.RunnerError("protocol_failure", "Stopped run cannot be resumed")

    prepared = [event for event in events if event["event_type"] == "CALL_PREPARED"]
    responses = [event for event in events if event["event_type"] == "API_RESPONSE"]
    completed = _completed(events)
    if not (len(prepared) == len(responses) == len(completed)):
        raise base.RunnerError("protocol_failure", "Dangling call could duplicate an API request")
    schedule = _specs(frozen["manifest"])
    for index, (before, response, after) in enumerate(zip(prepared, responses, completed), 1):
        if not (before["call_index"] == response["call_index"] == after["call_index"] == index):
            raise base.RunnerError("protocol_failure", "Non-contiguous call index")
        if not (before["call_spec"] == response["call_spec"] == after["call_spec"]):
            raise base.RunnerError("protocol_failure", "Call spec changed within slot")
        positions = [events.index(before), events.index(response), events.index(after)]
        if positions != [positions[0], positions[0] + 1, positions[0] + 2]:
            raise base.RunnerError("protocol_failure", "Call events are not adjacent")
        if response.get("request_sha256") != frozen["manifest"]["request_identity"]["expected_astra_request_sha256"]:
            raise base.RunnerError("protocol_failure", "Astra request SHA drift")
        observed_class = after.get("normalized_result", {}).get("observable_class")
        if observed_class not in frozen["manifest"]["observable_classes"]:
            raise base.RunnerError("protocol_failure", "Missing/invalid observable class")
    if [base.CallSpec(**event["call_spec"]) for event in completed] != schedule[:len(completed)]:
        raise base.RunnerError("protocol_failure", "Calls are not a frozen-order prefix")
    if len(completed) > 3:
        raise base.RunnerError("protocol_failure", "Maximum API call budget exceeded")
    if any(event["call_spec"]["role"] != "production_decision" for event in completed):
        raise base.RunnerError("protocol_failure", "Non-production call detected")

    aggregates = [event for event in events if event["event_type"] == "EXPERIMENT_AGGREGATE"]
    if aggregates:
        if len(aggregates) != 1 or len(completed) != 3:
            raise base.RunnerError("protocol_failure", "Aggregate placement/count mismatch")
        classes = [event["normalized_result"]["observable_class"] for event in completed]
        expected_counts = {name: classes.count(name) for name in frozen["manifest"]["observable_classes"]}
        if aggregates[0].get("classes") != classes:
            raise base.RunnerError("protocol_failure", "Aggregate classes mismatch")
        if aggregates[0].get("counts") != expected_counts:
            raise base.RunnerError("protocol_failure", "Aggregate counts mismatch")
        if aggregates[0].get("verdict") != _verdict(classes):
            raise base.RunnerError("protocol_failure", "Aggregate verdict mismatch")
        aggregate_pos = events.index(aggregates[0])
        if markers and aggregate_pos != markers[0] - 1:
            raise base.RunnerError("protocol_failure", "Aggregate must precede RUN_COMPLETED")
    if markers and not aggregates:
        raise base.RunnerError("protocol_failure", "Completed run lacks aggregate")
    return completed


def _append_prepared(
    checkpoint: base.Checkpoint, spec: base.CallSpec, index: int,
    frozen: dict[str, Any],
) -> None:
    identity = frozen["manifest"]["request_identity"]
    checkpoint.append(
        "CALL_PREPARED", call_index=index, call_spec=asdict(spec),
        request_role="production_decision",
        relevant_hashes={
            "manifest_sha256": frozen["manifest_sha256"],
            "case_sha256": frozen["manifest"]["reused_frozen_artifacts"]["case_sha256"],
            "state_sha256": frozen["manifest"]["reused_frozen_artifacts"]["state_sha256"],
            "prompt_sha256": frozen["manifest"]["production_prompt"]["sha256"],
            "expected_astra_request_sha256": identity["expected_astra_request_sha256"],
            "model_excluded_request_sha256": identity["model_excluded_request_sha256"],
        },
    )


def _run_production(
    client: Any, checkpoint: base.Checkpoint, spec: base.CallSpec, index: int,
    frozen: dict[str, Any],
) -> None:
    from ai.cognitive_core import CognitiveCore

    prompt_bytes = _git_bytes(frozen["manifest"]["production_prompt"]["git_path"])
    if hashlib.sha256(prompt_bytes).hexdigest().upper() != frozen["manifest"]["production_prompt"]["sha256"]:
        raise base.RunnerError("hash_failure", "Prompt changed after preflight")
    FrozenCore = type(
        "FrozenPromptCognitiveCore", (CognitiveCore,),
        {"_system_prompt": staticmethod(lambda: prompt_bytes.decode("utf-8"))},
    )
    current, human_model, previous_acs, history = base._load_production_input(
        frozen["case"]
    )
    identity = frozen["manifest"]["request_identity"]

    def captured(response: Any, request: Any) -> None:
        if base._sha_value(request) != identity["expected_astra_request_sha256"]:
            raise base.RunnerError("protocol_failure", "Actual Astra request drift")
        without_model = _request_without_model(request)
        if base._sha_value(without_model) != identity["model_excluded_request_sha256"]:
            raise base.RunnerError("protocol_failure", "Actual model-excluded request drift")
        mini_equivalent = {"model": frozen["manifest"]["baseline_model"], **without_model}
        if base._sha_value(mini_equivalent) != identity["historical_mini_request_sha256"]:
            raise base.RunnerError("protocol_failure", "Actual request is not the matched mini request")
        base._append_response(checkpoint, spec, index, response, request)

    capturing = base._CapturingClient(client, captured)
    result = FrozenCore(capturing, model=spec.model).propose(
        current, human_model, previous_acs, history
    )
    if capturing.completions.calls != 1:
        raise base.RunnerError("protocol_failure", "Production slot made != 1 API call")
    observable = _projection(result)
    checkpoint.append(
        "CALL_COMPLETED", call_index=index, call_spec=asdict(spec),
        normalized_result={
            "cognitive_turn_result": result,
            "observable_action": observable,
            "observable_class": classify_observable(observable),
            "classification_contract": frozen["manifest"]["classification_contract"]["version"],
        },
    )


def _new_run(frozen: dict[str, Any], root: Optional[Path] = None):
    output_root = RUNS_ROOT if root is None else root
    run_id = f"fsbdms-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = output_root / run_id
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "head": HEAD,
        "manifest_sha256": frozen["manifest_sha256"],
        "freeze_sha256": frozen["freeze_sha256"],
        "runner_sha256": frozen["runner_sha256"],
        "base_runner_sha256": frozen["base_runner_sha256"],
        "retry": 0,
        "smoke_calls": 0,
        "evaluator_calls": 0,
        "maximum_api_calls": 3,
    }
    base._write_metadata(run_dir / "run.json", metadata)
    checkpoint = base.Checkpoint(run_dir / "checkpoint.jsonl")
    checkpoint.append(
        "RUN_STARTED", run_id=run_id, head=HEAD,
        manifest_sha256=frozen["manifest_sha256"],
    )
    return run_id, run_dir, metadata, checkpoint


def _load_run(run_id: str, frozen: dict[str, Any]):
    if not run_id or Path(run_id).name != run_id:
        raise base.RunnerError("protocol_failure", "Invalid run id")
    run_dir = RUNS_ROOT / run_id
    metadata_path = run_dir / "run.json"
    checkpoint = base.Checkpoint(run_dir / "checkpoint.jsonl")
    if not metadata_path.is_file() or not checkpoint.path.is_file():
        raise base.RunnerError("protocol_failure", f"Unknown/incomplete run: {run_id}")
    metadata = base._read_json(metadata_path)
    validate_checkpoint(metadata, checkpoint.read(), frozen, for_resume=True)
    return run_dir, metadata, checkpoint


def execute_live(
    frozen: dict[str, Any], metadata: dict[str, Any], checkpoint: base.Checkpoint,
) -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(dotenv_path=ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise base.RunnerError("api_failure", "OPENAI_API_KEY is missing")
    client = OpenAI(max_retries=0, timeout=120.0)
    schedule = _specs(frozen["manifest"])
    while True:
        events = checkpoint.read()
        completed = validate_checkpoint(metadata, events, frozen, for_resume=False)
        if len(completed) == 3:
            if not any(event["event_type"] == "EXPERIMENT_AGGREGATE" for event in events):
                classes = [event["normalized_result"]["observable_class"] for event in completed]
                checkpoint.append(
                    "EXPERIMENT_AGGREGATE", classes=classes,
                    counts={name: classes.count(name) for name in frozen["manifest"]["observable_classes"]},
                    verdict=_verdict(classes),
                )
                events = checkpoint.read()
                validate_checkpoint(metadata, events, frozen, for_resume=False)
            checkpoint.append(
                "RUN_COMPLETED", api_calls=3, production_calls=3,
                evaluator_calls=0,
            )
            print(f"RUN_COMPLETED={metadata['run_id']}", flush=True)
            print(f"CHECKPOINT={checkpoint.path}", flush=True)
            return
        spec = schedule[len(completed)]
        index = len(completed) + 1
        _append_prepared(checkpoint, spec, index, frozen)
        try:
            _run_production(client, checkpoint, spec, index, frozen)
        except BaseException as error:
            checkpoint.append(
                "STOP", category=getattr(error, "category", "api_failure"),
                call_index=index, call_spec=asdict(spec),
                error_type=type(error).__name__, error_message=str(error),
            )
            raise


def _synthetic_projection(kind: str) -> dict[str, str]:
    if kind == "direct":
        return {
            "decision_intent": "mechanism_discovery",
            "action_content_kind": "system_question",
            "semantic_content": "Уточнить, было ли сильное желание поесть до тяжёлого разговора или появилось после него.",
            "response_target_interaction": "open_response",
            "reply_text": "Сильное желание поесть уже было до тяжёлого разговора или появилось после него?",
        }
    if kind == "local":
        return {
            "decision_intent": "reflection",
            "action_content_kind": "system_reflection",
            "semantic_content": "Вы пропустили обед, потом был тяжёлый разговор и вечером переели; вы не уверены, было желание до разговора или после него.",
            "response_target_interaction": "clarification",
            "reply_text": "Я правильно понял?",
        }
    if kind == "working":
        return {
            "decision_intent": "recognition",
            "action_content_kind": "recognition_option",
            "semantic_content": "Рабочая версия: могли действовать физический голод и напряжение.",
            "response_target_interaction": "evaluation",
            "reply_text": "Это похоже на ваш опыт или было иначе?",
        }
    return {
        "decision_intent": "transition", "action_content_kind": "system_transition",
        "semantic_content": "Переход", "response_target_interaction": "consent",
        "reply_text": "Перейдем дальше?",
    }


def dry_run(frozen: dict[str, Any]) -> dict[str, Any]:
    classifier_checks = {
        "direct": classify_observable(_synthetic_projection("direct")),
        "local": classify_observable(_synthetic_projection("local")),
        "working": classify_observable(_synthetic_projection("working")),
        "other": classify_observable(_synthetic_projection("other")),
    }
    if classifier_checks != {
        "direct": "DIRECT_DISCRIMINATION",
        "local": "LOCAL_REFLECTION",
        "working": "OTHER",
        "other": "OTHER",
    }:
        raise base.RunnerError("protocol_failure", f"Classifier self-test failed: {classifier_checks}")

    with tempfile.TemporaryDirectory(prefix="fsbd-model-swap-dry-") as temporary:
        root = Path(temporary)
        _, _, metadata, checkpoint = _new_run(frozen, root)
        specs = _specs(frozen["manifest"])
        dry_classes = ["DIRECT_DISCRIMINATION", "LOCAL_REFLECTION", "OTHER"]
        for index, (spec, observed_class) in enumerate(zip(specs, dry_classes), 1):
            _append_prepared(checkpoint, spec, index, frozen)
            checkpoint.append(
                "API_RESPONSE", call_index=index, call_spec=asdict(spec),
                request_sha256=frozen["manifest"]["request_identity"]["expected_astra_request_sha256"],
                response_raw={"dry": True}, token_usage={},
            )
            checkpoint.append(
                "CALL_COMPLETED", call_index=index, call_spec=asdict(spec),
                normalized_result={"observable_class": observed_class, "dry": True},
            )
        completed = validate_checkpoint(metadata, checkpoint.read(), frozen, for_resume=False)
        classes = [event["normalized_result"]["observable_class"] for event in completed]
        checkpoint.append(
            "EXPERIMENT_AGGREGATE", classes=classes,
            counts={name: classes.count(name) for name in frozen["manifest"]["observable_classes"]},
            verdict=_verdict(classes),
        )
        validate_checkpoint(metadata, checkpoint.read(), frozen, for_resume=False)

        _, _, prefix_metadata, prefix = _new_run(frozen, root / "prefix")
        _append_prepared(prefix, specs[0], 1, frozen)
        prefix.append(
            "API_RESPONSE", call_index=1, call_spec=asdict(specs[0]),
            request_sha256=frozen["manifest"]["request_identity"]["expected_astra_request_sha256"],
            response_raw={"dry": True}, token_usage={},
        )
        prefix.append(
            "CALL_COMPLETED", call_index=1, call_spec=asdict(specs[0]),
            normalized_result={"observable_class": "OTHER", "dry": True},
        )
        validate_checkpoint(prefix_metadata, prefix.read(), frozen, for_resume=True)

        _, _, dangling_metadata, dangling = _new_run(frozen, root / "dangling")
        _append_prepared(dangling, specs[0], 1, frozen)
        dangling_rejected = False
        try:
            validate_checkpoint(dangling_metadata, dangling.read(), frozen, for_resume=True)
        except base.RunnerError:
            dangling_rejected = True
        if not dangling_rejected:
            raise base.RunnerError("protocol_failure", "Dangling call was resumable")

        _, _, stopped_metadata, stopped = _new_run(frozen, root / "stopped")
        stopped.append("STOP", category="structural_failure", error_message="dry")
        stop_rejected = False
        try:
            validate_checkpoint(stopped_metadata, stopped.read(), frozen, for_resume=True)
        except base.RunnerError:
            stop_rejected = True
        if not stop_rejected:
            raise base.RunnerError("protocol_failure", "Stopped run was resumable")

        _, _, order_metadata, wrong_order = _new_run(frozen, root / "order")
        wrong = specs[1]
        _append_prepared(wrong_order, wrong, 1, frozen)
        wrong_order.append(
            "API_RESPONSE", call_index=1, call_spec=asdict(wrong),
            request_sha256=frozen["manifest"]["request_identity"]["expected_astra_request_sha256"],
            response_raw={"dry": True}, token_usage={},
        )
        wrong_order.append(
            "CALL_COMPLETED", call_index=1, call_spec=asdict(wrong),
            normalized_result={"observable_class": "OTHER", "dry": True},
        )
        order_rejected = False
        try:
            validate_checkpoint(order_metadata, wrong_order.read(), frozen, for_resume=True)
        except base.RunnerError:
            order_rejected = True
        if not order_rejected:
            raise base.RunnerError("protocol_failure", "Out-of-order prefix was resumable")

        tampered = checkpoint.path.read_text(encoding="utf-8").replace(
            '"response_raw": {"dry": true}', '"response_raw": {"dry": false}', 1
        )
        checkpoint.path.write_text(tampered, encoding="utf-8")
        hash_rejected = False
        try:
            checkpoint.read()
        except base.RunnerError:
            hash_rejected = True
        if not hash_rejected:
            raise base.RunnerError("protocol_failure", "Hash-chain tampering was accepted")

        return {
            "result": "PASS",
            "api_calls": 0,
            "head": HEAD,
            "samples_simulated": 3,
            "evaluator_calls_simulated": 0,
            "maximum_call_budget": 3,
            "classifier_checks": classifier_checks,
            "request_difference": frozen["matched_request_difference"],
            "checkpoint_hash_chain": "PASS",
            "clean_prefix_resume": "PASS",
            "dangling_resume_rejected": "PASS",
            "terminal_stop_resume_rejected": "PASS",
            "out_of_order_resume_rejected": "PASS",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--new-run", action="store_true")
    modes.add_argument("--resume", metavar="RUN_ID")
    modes.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        frozen = preflight()
        if args.dry_run:
            print(json.dumps(dry_run(frozen), ensure_ascii=False, indent=2))
            return 0
        if args.new_run:
            _, _, metadata, checkpoint = _new_run(frozen)
        else:
            _, metadata, checkpoint = _load_run(args.resume, frozen)
        execute_live(frozen, metadata, checkpoint)
        return 0
    except base.RunnerError as error:
        print(f"STOP_CATEGORY={error.category}", file=sys.stderr)
        print(f"STOP_MESSAGE={error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
