"""Resume-safe evaluator-only runner for frozen Astra FSB-D actions vs Direct."""

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


HEAD = "1e08316"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Conversation_Lab import fsb_01_live_runner_672455a as base  # noqa: E402
from Conversation_Lab import fsb_d_model_swap_runner_1e08316 as source  # noqa: E402


FREEZE = ROOT / "Conversation_Lab/frozen/fsb_d_astra_posthoc_1e08316"
RUNS_ROOT = ROOT / "Conversation_Lab/traces/fsb_d_astra_posthoc_1e08316_runs"
MANIFEST_NAME = "fsb_d_astra_posthoc_manifest_1e08316.json"
VALIDATOR_NAME = "fsb_d_astra_posthoc_offline_validator_1e08316.py"
SELF = Path(__file__).resolve()
BASE_RUNNER = ROOT / "Conversation_Lab/fsb_01_live_runner_672455a.py"
BASE_RUNNER_SHA256 = "00038CB5913D51C54A454DD3B3F0D5CF2444CA9B39E6C9BDC14C144637E562BF"
SOURCE_RUNNER = ROOT / "Conversation_Lab/fsb_d_model_swap_runner_1e08316.py"
SOURCE_RUNNER_SHA256 = "A852EBAE646896CE18A425475EB6003D9A574E359829B1B897401E6E2718B236"
FREEZE_SHA256 = {
    "fsb_d_astra_posthoc_freeze_report_1e08316.md": "4615469010874AE0A0D5E8769FD155EF2F5849BBD0A5CE7CBD4B87A662D82A15",
    "fsb_d_astra_posthoc_manifest_1e08316.json": "1B8CFC065FA9B0265F42BD7C7ACD0B6735955A28C6DF27E59AD43EC0BBF6F608",
    "fsb_d_astra_posthoc_measurement_plan_1e08316.md": "72D754ED8271B6359A201EAAF568C9E30E09E7F9C92297782CC28406B29AB02D",
    "fsb_d_astra_posthoc_offline_validator_1e08316.py": "1D386A17C73777DF687104FD0F3A6F706F8C7667C88C8E70064ECBD1981B1951",
}


def _specs(manifest: dict[str, Any]) -> list[base.CallSpec]:
    specs: list[base.CallSpec] = []
    for sample_id in manifest["sample_order"]:
        for repeat, astra_left in enumerate(
            manifest["orientation_astra_on_left"][sample_id], 1
        ):
            specs.append(base.CallSpec(
                key=f"evaluation.{sample_id}.{repeat}", phase="posthoc",
                case_id=sample_id, role="product_evaluator",
                model=manifest["evaluator_model"], repeat=repeat,
                selected_on_left=astra_left,
            ))
    return specs


def _evaluator_request(
    *, model: str, prompt: str, schema: dict[str, Any], context: dict[str, Any],
    left: dict[str, Any], right: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(
                {"frozen_context": context, "left_action": left, "right_action": right},
                ensure_ascii=False, sort_keys=True,
            )},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "fsb_d_astra_posthoc_verdict", "strict": True,
                "schema": schema,
            },
        },
    }


def _normalize(verdict: str, astra_on_left: bool) -> str:
    if verdict == "EQUIVALENT_OR_BOTH_ACCEPTABLE":
        return verdict
    if verdict == "INSUFFICIENT_BASIS":
        return "ABSTAIN"
    astra_higher = (
        verdict == "LEFT_HIGHER" if astra_on_left else verdict == "RIGHT_HIGHER"
    )
    return "ASTRA_HIGHER" if astra_higher else "DIRECT_HIGHER"


def _sample_aggregate(values: list[str]) -> str:
    if len(values) != 3:
        raise base.RunnerError("protocol_failure", "Sample needs exactly three judgments")
    if len(set(values)) == 1:
        return values[0]
    if values.count("ASTRA_HIGHER") >= 2:
        return "ASTRA_HIGHER"
    if values.count("DIRECT_HIGHER") >= 2:
        return "DIRECT_HIGHER"
    return "ABSTAIN"


def _overall(values: list[str]) -> str:
    if len(values) != 3:
        raise base.RunnerError("protocol_failure", "Overall result needs three samples")
    if "ABSTAIN" in values:
        return "INCONCLUSIVE"
    if values.count("DIRECT_HIGHER") == 3:
        return "DIRECT_REPRODUCIBLY_HIGHER"
    if values.count("ASTRA_HIGHER") == 3:
        return "ASTRA_REPRODUCIBLY_HIGHER"
    if values.count("DIRECT_HIGHER") >= 2 and values.count("ASTRA_HIGHER") == 0:
        return "DIRECT_MAJORITY_HIGHER"
    if values.count("ASTRA_HIGHER") >= 2 and values.count("DIRECT_HIGHER") == 0:
        return "ASTRA_MAJORITY_HIGHER"
    if values.count("EQUIVALENT_OR_BOTH_ACCEPTABLE") == 3:
        return "EQUIVALENT_PATTERN"
    return "MIXED_COMPARATIVE_VALUE"


def _load_source(manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    source_frozen = source.preflight()
    run_dir = (ROOT / manifest["source_run"]["checkpoint_path"]).parent
    metadata = base._read_json(run_dir / "run.json")
    events = base.Checkpoint(run_dir / "checkpoint.jsonl").read()
    completed = source.validate_checkpoint(
        metadata, events, source_frozen, for_resume=False
    )
    actions: dict[str, dict[str, Any]] = {}
    for sample_id in manifest["sample_order"]:
        matches = [
            event for event in completed
            if event["call_spec"]["case_id"] == sample_id
        ]
        if len(matches) != 1:
            raise base.RunnerError("protocol_failure", f"Missing source action: {sample_id}")
        event = matches[0]
        binding = manifest["saved_actions"][sample_id]
        if event["normalized_result"].get("observable_class") != "OTHER":
            raise base.RunnerError(
                "protocol_failure", f"Source action is not frozen as OTHER: {sample_id}"
            )
        action = event["normalized_result"]["observable_action"]
        if event["event_sha256"] != binding["completed_event_sha256"]:
            raise base.RunnerError("hash_failure", f"Source event drift: {sample_id}")
        if base._sha_value(action) != binding["observable_action_sha256"]:
            raise base.RunnerError("hash_failure", f"Source action drift: {sample_id}")
        actions[sample_id] = action
    return source_frozen, actions


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
        raise base.RunnerError("hash_failure", "Checkpoint helper changed")
    if base._sha_file(SOURCE_RUNNER) != SOURCE_RUNNER_SHA256:
        raise base.RunnerError("hash_failure", "Source-run validator changed")

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
        "production": 0, "evaluator": 9, "maximum_total": 9,
    }:
        raise base.RunnerError("protocol_failure", "Manifest HEAD/budget mismatch")
    if manifest["evaluator_model"] != "gpt-6-astra" or len(_specs(manifest)) != 9:
        raise base.RunnerError("protocol_failure", "Evaluator schedule mismatch")

    _, actions = _load_source(manifest)
    case = base._read_json(ROOT / manifest["frozen_context"]["case_path"])
    pair = base._read_json(ROOT / manifest["direct_benchmark"]["pair_path"])
    direct = {
        key: value for key, value in pair["direct_discrimination"].items()
        if key != "strategy"
    }
    if base._sha_value(direct) != manifest["direct_benchmark"]["direct_projection_sha256"]:
        raise base.RunnerError("hash_failure", "Direct projection drift")
    prompt = (ROOT / manifest["evaluator_contract"]["prompt_path"]).read_text(encoding="utf-8")
    schema = base._read_json(ROOT / manifest["evaluator_contract"]["schema_path"])

    for spec in _specs(manifest):
        astra = actions[spec.case_id]
        left, right = (astra, direct) if spec.selected_on_left else (direct, astra)
        request = _evaluator_request(
            model=spec.model, prompt=prompt, schema=schema,
            context=case["frozen_input"], left=left, right=right,
        )
        expected = manifest["expected_request_sha256"][spec.case_id][spec.repeat - 1]
        if base._sha_value(request) != expected:
            raise base.RunnerError("protocol_failure", f"Request drift: {spec.key}")

    return {
        "head": actual_head,
        "manifest": manifest,
        "manifest_sha256": actual_hashes[MANIFEST_NAME],
        "freeze_sha256": actual_hashes,
        "offline_validation": validation,
        "actions": actions,
        "direct": direct,
        "context": case["frozen_input"],
        "prompt": prompt,
        "schema": schema,
        "runner_sha256": base._sha_file(SELF),
        "base_runner_sha256": BASE_RUNNER_SHA256,
        "source_runner_sha256": SOURCE_RUNNER_SHA256,
    }


def _completed(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event["event_type"] == "CALL_COMPLETED"]


def validate_checkpoint(
    metadata: dict[str, Any], events: list[dict[str, Any]], frozen: dict[str, Any],
    *, for_resume: bool,
) -> list[dict[str, Any]]:
    for field in (
        "head", "manifest_sha256", "freeze_sha256", "runner_sha256",
        "base_runner_sha256", "source_runner_sha256",
    ):
        if metadata.get(field) != frozen[field]:
            raise base.RunnerError("protocol_failure", f"Run {field} mismatch")
    if metadata.get("maximum_api_calls") != 9 or metadata.get("retry") != 0:
        raise base.RunnerError("protocol_failure", "Run budget/retry mismatch")
    if metadata.get("production_calls") != 0 or metadata.get("evaluator_calls") != 9:
        raise base.RunnerError("protocol_failure", "Run role budget mismatch")
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
    allowed_blind = set(frozen["schema"]["properties"]["verdict"]["enum"])
    allowed_normalized = set(frozen["manifest"]["per_sample_verdicts"])
    for index, (before, response, after) in enumerate(zip(prepared, responses, completed), 1):
        if not (before["call_index"] == response["call_index"] == after["call_index"] == index):
            raise base.RunnerError("protocol_failure", "Non-contiguous call index")
        if not (before["call_spec"] == response["call_spec"] == after["call_spec"]):
            raise base.RunnerError("protocol_failure", "Call spec changed within slot")
        positions = [events.index(before), events.index(response), events.index(after)]
        if positions != [positions[0], positions[0] + 1, positions[0] + 2]:
            raise base.RunnerError("protocol_failure", "Call events are not adjacent")
        spec = base.CallSpec(**after["call_spec"])
        expected_request = frozen["manifest"]["expected_request_sha256"][spec.case_id][spec.repeat - 1]
        if response.get("request_sha256") != expected_request:
            raise base.RunnerError("protocol_failure", "Evaluator request SHA drift")
        result = after.get("normalized_result", {})
        if result.get("blind_verdict") not in allowed_blind:
            raise base.RunnerError("protocol_failure", "Invalid blind verdict")
        if result.get("comparison_verdict") not in allowed_normalized:
            raise base.RunnerError("protocol_failure", "Invalid normalized verdict")
    if [base.CallSpec(**event["call_spec"]) for event in completed] != schedule[:len(completed)]:
        raise base.RunnerError("protocol_failure", "Calls are not a frozen-order prefix")
    if len(completed) > 9:
        raise base.RunnerError("protocol_failure", "API call budget exceeded")
    if any(event["call_spec"]["role"] != "product_evaluator" for event in completed):
        raise base.RunnerError("protocol_failure", "Production/non-evaluator call detected")

    sample_events = [event for event in events if event["event_type"] == "SAMPLE_AGGREGATE"]
    overall_events = [event for event in events if event["event_type"] == "EXPERIMENT_AGGREGATE"]
    if sample_events or overall_events:
        if len(completed) != 9 or len(sample_events) != 3 or len(overall_events) != 1:
            raise base.RunnerError("protocol_failure", "Aggregate count/placement mismatch")
        aggregates: list[str] = []
        for sample_id, aggregate_event in zip(frozen["manifest"]["sample_order"], sample_events):
            values = [
                event["normalized_result"]["comparison_verdict"]
                for event in completed if event["call_spec"]["case_id"] == sample_id
            ]
            aggregate = _sample_aggregate(values)
            if aggregate_event.get("sample_id") != sample_id:
                raise base.RunnerError("protocol_failure", "Sample aggregate order mismatch")
            if aggregate_event.get("raw_normalized_verdicts") != values:
                raise base.RunnerError("protocol_failure", "Sample verdict list mismatch")
            if aggregate_event.get("aggregate") != aggregate:
                raise base.RunnerError("protocol_failure", "Sample aggregate mismatch")
            aggregates.append(aggregate)
        overall = overall_events[0]
        if overall.get("sample_aggregates") != aggregates:
            raise base.RunnerError("protocol_failure", "Overall sample list mismatch")
        if overall.get("result") != _overall(aggregates):
            raise base.RunnerError("protocol_failure", "Overall result mismatch")
        aggregate_positions = [events.index(event) for event in sample_events + overall_events]
        if aggregate_positions != list(range(aggregate_positions[0], aggregate_positions[0] + 4)):
            raise base.RunnerError("protocol_failure", "Aggregates must be contiguous")
        if markers and aggregate_positions[-1] != markers[0] - 1:
            raise base.RunnerError("protocol_failure", "Overall aggregate must precede completion")
    if markers and (len(sample_events) != 3 or len(overall_events) != 1):
        raise base.RunnerError("protocol_failure", "Completed run lacks aggregates")
    return completed


def _append_prepared(
    checkpoint: base.Checkpoint, spec: base.CallSpec, index: int,
    frozen: dict[str, Any],
) -> None:
    expected = frozen["manifest"]["expected_request_sha256"][spec.case_id][spec.repeat - 1]
    checkpoint.append(
        "CALL_PREPARED", call_index=index, call_spec=asdict(spec),
        request_role="product_evaluator",
        relevant_hashes={
            "manifest_sha256": frozen["manifest_sha256"],
            "source_action_sha256": frozen["manifest"]["saved_actions"][spec.case_id]["observable_action_sha256"],
            "direct_projection_sha256": frozen["manifest"]["direct_benchmark"]["direct_projection_sha256"],
            "rubric_sha256": frozen["manifest"]["evaluator_contract"]["prompt_sha256"],
            "schema_sha256": frozen["manifest"]["evaluator_contract"]["schema_sha256"],
            "expected_request_sha256": expected,
        },
    )


def _run_evaluator(
    client: Any, checkpoint: base.Checkpoint, spec: base.CallSpec, index: int,
    frozen: dict[str, Any],
) -> None:
    astra = frozen["actions"][spec.case_id]
    direct = frozen["direct"]
    left, right = (astra, direct) if spec.selected_on_left else (direct, astra)
    request = _evaluator_request(
        model=spec.model, prompt=frozen["prompt"], schema=frozen["schema"],
        context=frozen["context"], left=left, right=right,
    )
    expected = frozen["manifest"]["expected_request_sha256"][spec.case_id][spec.repeat - 1]
    if base._sha_value(request) != expected:
        raise base.RunnerError("protocol_failure", "Evaluator request changed after preflight")
    response = client.chat.completions.create(**request)
    base._append_response(checkpoint, spec, index, response, request)
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise base.RunnerError("structural_failure", "Empty evaluator response")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise base.RunnerError(
            "structural_failure", "Evaluator response is not valid JSON"
        ) from error
    if set(payload) != {"verdict"}:
        raise base.RunnerError("structural_failure", "Evaluator payload fields changed")
    verdict = payload["verdict"]
    allowed = set(frozen["schema"]["properties"]["verdict"]["enum"])
    if verdict not in allowed:
        raise base.RunnerError("structural_failure", "Unknown evaluator verdict")
    checkpoint.append(
        "CALL_COMPLETED", call_index=index, call_spec=asdict(spec),
        normalized_result={
            "blind_verdict": verdict,
            "comparison_verdict": _normalize(verdict, bool(spec.selected_on_left)),
        },
    )


def _append_aggregates(checkpoint: base.Checkpoint, frozen: dict[str, Any]) -> None:
    events = checkpoint.read()
    completed = _completed(events)
    aggregates: list[str] = []
    for sample_id in frozen["manifest"]["sample_order"]:
        values = [
            event["normalized_result"]["comparison_verdict"]
            for event in completed if event["call_spec"]["case_id"] == sample_id
        ]
        aggregate = _sample_aggregate(values)
        checkpoint.append(
            "SAMPLE_AGGREGATE", sample_id=sample_id,
            raw_normalized_verdicts=values, aggregate=aggregate,
        )
        aggregates.append(aggregate)
    checkpoint.append(
        "EXPERIMENT_AGGREGATE", sample_aggregates=aggregates,
        result=_overall(aggregates),
    )


def _new_run(frozen: dict[str, Any], root: Optional[Path] = None):
    output_root = RUNS_ROOT if root is None else root
    run_id = f"fsbdph-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = output_root / run_id
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "head": HEAD,
        "manifest_sha256": frozen["manifest_sha256"],
        "freeze_sha256": frozen["freeze_sha256"],
        "runner_sha256": frozen["runner_sha256"],
        "base_runner_sha256": frozen["base_runner_sha256"],
        "source_runner_sha256": frozen["source_runner_sha256"],
        "retry": 0, "smoke_calls": 0,
        "production_calls": 0, "evaluator_calls": 9,
        "maximum_api_calls": 9,
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
        if len(completed) == 9:
            if not any(event["event_type"] == "EXPERIMENT_AGGREGATE" for event in events):
                _append_aggregates(checkpoint, frozen)
                events = checkpoint.read()
                validate_checkpoint(metadata, events, frozen, for_resume=False)
            checkpoint.append(
                "RUN_COMPLETED", api_calls=9, production_calls=0,
                evaluator_calls=9,
            )
            print(f"RUN_COMPLETED={metadata['run_id']}", flush=True)
            print(f"CHECKPOINT={checkpoint.path}", flush=True)
            return
        spec = schedule[len(completed)]
        index = len(completed) + 1
        _append_prepared(checkpoint, spec, index, frozen)
        try:
            _run_evaluator(client, checkpoint, spec, index, frozen)
        except BaseException as error:
            checkpoint.append(
                "STOP", category=getattr(error, "category", "api_failure"),
                call_index=index, call_spec=asdict(spec),
                error_type=type(error).__name__, error_message=str(error),
            )
            raise


def dry_run(frozen: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="fsbd-astra-posthoc-dry-") as temporary:
        root = Path(temporary)
        _, _, metadata, checkpoint = _new_run(frozen, root)
        schedule = _specs(frozen["manifest"])
        blind = [
            "RIGHT_HIGHER", "LEFT_HIGHER", "RIGHT_HIGHER",
            "LEFT_HIGHER", "RIGHT_HIGHER", "LEFT_HIGHER",
            "EQUIVALENT_OR_BOTH_ACCEPTABLE",
            "EQUIVALENT_OR_BOTH_ACCEPTABLE",
            "EQUIVALENT_OR_BOTH_ACCEPTABLE",
        ]
        for index, (spec, verdict) in enumerate(zip(schedule, blind), 1):
            _append_prepared(checkpoint, spec, index, frozen)
            expected = frozen["manifest"]["expected_request_sha256"][spec.case_id][spec.repeat - 1]
            checkpoint.append(
                "API_RESPONSE", call_index=index, call_spec=asdict(spec),
                request_sha256=expected, response_raw={"dry": True}, token_usage={},
            )
            checkpoint.append(
                "CALL_COMPLETED", call_index=index, call_spec=asdict(spec),
                normalized_result={
                    "blind_verdict": verdict,
                    "comparison_verdict": _normalize(verdict, bool(spec.selected_on_left)),
                },
            )
        completed = validate_checkpoint(metadata, checkpoint.read(), frozen, for_resume=False)
        if len(completed) != 9:
            raise base.RunnerError("protocol_failure", "Dry schedule incomplete")
        _append_aggregates(checkpoint, frozen)
        completed_events = checkpoint.read()
        validate_checkpoint(metadata, completed_events, frozen, for_resume=False)
        dry_sample_aggregates = [
            event["aggregate"] for event in completed_events
            if event["event_type"] == "SAMPLE_AGGREGATE"
        ]

        _, _, prefix_metadata, prefix = _new_run(frozen, root / "prefix")
        spec = schedule[0]
        _append_prepared(prefix, spec, 1, frozen)
        prefix.append(
            "API_RESPONSE", call_index=1, call_spec=asdict(spec),
            request_sha256=frozen["manifest"]["expected_request_sha256"][spec.case_id][0],
            response_raw={"dry": True}, token_usage={},
        )
        prefix.append(
            "CALL_COMPLETED", call_index=1, call_spec=asdict(spec),
            normalized_result={
                "blind_verdict": "LEFT_HIGHER",
                "comparison_verdict": "ASTRA_HIGHER",
            },
        )
        validate_checkpoint(prefix_metadata, prefix.read(), frozen, for_resume=True)

        _, _, dangling_metadata, dangling = _new_run(frozen, root / "dangling")
        _append_prepared(dangling, schedule[0], 1, frozen)
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
        wrong = schedule[1]
        _append_prepared(wrong_order, wrong, 1, frozen)
        wrong_order.append(
            "API_RESPONSE", call_index=1, call_spec=asdict(wrong),
            request_sha256=frozen["manifest"]["expected_request_sha256"][wrong.case_id][wrong.repeat - 1],
            response_raw={"dry": True}, token_usage={},
        )
        wrong_order.append(
            "CALL_COMPLETED", call_index=1, call_spec=asdict(wrong),
            normalized_result={
                "blind_verdict": "LEFT_HIGHER",
                "comparison_verdict": _normalize("LEFT_HIGHER", bool(wrong.selected_on_left)),
            },
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
            "result": "PASS", "api_calls": 0, "head": HEAD,
            "production_calls_simulated": 0,
            "evaluator_calls_simulated": 9,
            "maximum_call_budget": 9,
            "sample_aggregates_simulated": dry_sample_aggregates,
            "aggregation_contract": "PASS",
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
