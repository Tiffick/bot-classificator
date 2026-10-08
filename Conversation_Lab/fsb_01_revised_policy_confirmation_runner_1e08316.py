"""Resume-safe runner for the frozen revised-policy confirmation experiment.

``--dry-run`` is offline. Live work requires the explicit ``--live-new-run``
mode or a safe ``--resume RUN_ID`` of that already authorized run.
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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


HEAD = "1e08316"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Conversation_Lab import fsb_01_revised_policy_runner_1e08316 as source  # noqa: E402


base = source.base
PACKAGE = ROOT / "Conversation_Lab/frozen/fsb_01_revised_policy_confirmation_1e08316"
RUNS_ROOT = ROOT / "Conversation_Lab/traces/fsb_01_revised_policy_confirmation_1e08316_runs"
MANIFEST_NAME = "fsb_01_confirmation_manifest_1e08316.json"
VALIDATOR_NAME = "fsb_01_confirmation_offline_validator_1e08316.py"
SOURCE_RUNNER = ROOT / "Conversation_Lab/fsb_01_revised_policy_runner_1e08316.py"
SOURCE_RUNNER_SHA256 = "9AAFDDF1F58972071B79B0A5258EF06D92CC66267AAEABF129907FF529FA1546"
FROZEN_SHA256 = {
    "fsb_01_confirmation_freeze_report_1e08316.md": "72C8FE142A653FF940B4E3BCDDD6B6D04CC74ED51B3B7F48B7F63978B03CE83F",
    "fsb_01_confirmation_manifest_1e08316.json": "2BE5CC3C7293ECB2253067295F854B69135529FBDBF1F455A01B65BF615F94F2",
    "fsb_01_confirmation_measurement_plan_1e08316.md": "C26897BDA2525F91FECB94E53D0E0D9239CFFA0B1BAADE8ACFF7E98B3FD87425",
    "fsb_01_confirmation_offline_validator_1e08316.py": "4643878F78FA1D9677232C978F4DE5191BA6EB8EF173101951C38425EC5F97EF",
}
TERMINAL_STOPS = {
    "hash_failure", "head_failure", "offline_validation_failure",
    "protocol_failure", "structural_failure", "structured_output_failure",
    "validation_failure", "api_failure", "billing_failure", "transport_failure",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_bytes(spec: str) -> bytes:
    return subprocess.run(
        ["git", "show", spec], cwd=ROOT, check=True, capture_output=True
    ).stdout


def _base_case(manifest: dict[str, Any], sample_id: str) -> str:
    try:
        return manifest["sample_base_case"][sample_id]
    except KeyError as error:
        raise base.RunnerError("protocol_failure", f"Unknown sample: {sample_id}") from error


def _specs(manifest: dict[str, Any]) -> list[base.CallSpec]:
    specs = [
        base.CallSpec(
            key=f"production.{sample_id}", phase="production",
            case_id=sample_id, role="production_decision",
            model=manifest["production_model"],
        )
        for sample_id in manifest["production_order"]
    ]
    historical = set(manifest["historical_evaluator_cases"])
    for sample_id in manifest["evaluator_order"]:
        case_id = _base_case(manifest, sample_id)
        role = "historical_evaluator" if case_id in historical else "product_evaluator"
        specs.extend(
            base.CallSpec(
                key=f"posthoc.{sample_id}.{judgment}", phase="posthoc",
                case_id=sample_id, role=role, model=manifest["evaluator_model"],
                repeat=judgment, selected_on_left=selected_on_left,
            )
            for judgment, selected_on_left in enumerate(
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
    names = {path.name for path in PACKAGE.iterdir() if path.is_file()}
    if names != set(FROZEN_SHA256):
        raise base.RunnerError("hash_failure", f"Frozen file set mismatch: {sorted(names)}")
    hashes: dict[str, str] = {}
    for name, expected in FROZEN_SHA256.items():
        actual = base._sha_file(PACKAGE / name)
        if actual != expected:
            raise base.RunnerError("hash_failure", f"Frozen SHA mismatch: {name}")
        hashes[name] = actual
    if base._sha_file(SOURCE_RUNNER) != SOURCE_RUNNER_SHA256:
        raise base.RunnerError("hash_failure", "Source revised-policy runner changed")

    manifest = _read_json(PACKAGE / MANIFEST_NAME)
    validation = subprocess.run(
        [sys.executable, str(PACKAGE / VALIDATOR_NAME)], cwd=ROOT,
        capture_output=True, text=True,
    )
    if validation.returncode != 0:
        raise base.RunnerError(
            "offline_validation_failure",
            f"Offline validator exit={validation.returncode}: {validation.stderr}",
        )
    try:
        validation_payload = json.loads(validation.stdout)
    except json.JSONDecodeError as error:
        raise base.RunnerError("offline_validation_failure", "Validator output is not JSON") from error
    if validation_payload.get("result") != "READY" or validation_payload.get("api_calls") != 0:
        raise base.RunnerError("offline_validation_failure", repr(validation_payload))

    source_frozen = source.preflight()
    samples: dict[str, dict[str, Any]] = {}
    for sample_id in manifest["production_order"]:
        case_id = _base_case(manifest, sample_id)
        case = json.loads(json.dumps(source_frozen["cases"][case_id]))
        case["case_id"] = sample_id
        samples[sample_id] = case
    schedule = _specs(manifest)
    if len(schedule) != 40:
        raise base.RunnerError("protocol_failure", "Schedule is not exactly 40 calls")
    if any(spec.role != "production_decision" for spec in schedule[:10]):
        raise base.RunnerError("protocol_failure", "All ten production calls must be first")
    if any(spec.role == "production_decision" for spec in schedule[10:]):
        raise base.RunnerError("protocol_failure", "Evaluator phase contains production")
    return {
        "head": actual_head, "manifest": manifest,
        "manifest_sha256": hashes[MANIFEST_NAME], "freeze_sha256": hashes,
        "offline_validation": validation_payload, "source": source_frozen,
        "samples": samples, "runner_sha256": base._sha_file(Path(__file__)),
        "source_runner_sha256": SOURCE_RUNNER_SHA256,
    }


def _completed(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event["event_type"] == "CALL_COMPLETED"]


def _aggregate_verdict(value: Any) -> str:
    return value["verdict"] if isinstance(value, dict) else value


def _sample_values(events: list[dict[str, Any]], sample_id: str) -> list[str]:
    return [
        event["normalized_result"]["observable_verdict"]
        for event in _completed(events)
        if event["call_spec"]["phase"] == "posthoc"
        and event["call_spec"]["case_id"] == sample_id
    ]


def validate_checkpoint(
    metadata: dict[str, Any], events: list[dict[str, Any]], frozen: dict[str, Any],
    *, for_resume: bool,
) -> list[dict[str, Any]]:
    for field in (
        "head", "manifest_sha256", "freeze_sha256", "runner_sha256",
        "source_runner_sha256",
    ):
        if metadata.get(field) != frozen[field]:
            raise base.RunnerError("protocol_failure", f"Run {field} mismatch")
    if metadata.get("maximum_api_calls") != 40 or metadata.get("retry") != 0:
        raise base.RunnerError("protocol_failure", "Run budget/retry mismatch")
    if not events or events[0]["event_type"] != "RUN_STARTED":
        raise base.RunnerError("protocol_failure", "Missing RUN_STARTED")
    if sum(event["event_type"] == "RUN_STARTED" for event in events) != 1:
        raise base.RunnerError("protocol_failure", "RUN_STARTED must be unique")
    finals = [i for i, event in enumerate(events) if event["event_type"] == "RUN_COMPLETED"]
    if finals and (len(finals) != 1 or finals[0] != len(events) - 1):
        raise base.RunnerError("protocol_failure", "RUN_COMPLETED must be unique and final")
    if finals and for_resume:
        raise base.RunnerError("protocol_failure", "Completed run cannot resume")
    if for_resume and any(
        event["event_type"] == "STOP" and event.get("category") in TERMINAL_STOPS
        for event in events
    ):
        raise base.RunnerError("protocol_failure", "Terminal STOP cannot resume")
    stops = [i for i, event in enumerate(events) if event["event_type"] == "STOP"]
    if stops and (len(stops) != 1 or stops[0] != len(events) - 1):
        raise base.RunnerError("protocol_failure", "STOP must be unique and final")

    prepared = [event for event in events if event["event_type"] == "CALL_PREPARED"]
    responses = [event for event in events if event["event_type"] == "API_RESPONSE"]
    done = _completed(events)
    if not (len(prepared) == len(responses) == len(done)):
        raise base.RunnerError("protocol_failure", "Dangling/partial API call is not resumable")
    for index, (before, response, after) in enumerate(zip(prepared, responses, done), 1):
        if {before["call_index"], response["call_index"], after["call_index"]} != {index}:
            raise base.RunnerError("protocol_failure", "Non-contiguous call index")
        if not (before["call_spec"] == response["call_spec"] == after["call_spec"]):
            raise base.RunnerError("protocol_failure", "Call spec changed inside slot")
        positions = [events.index(before), events.index(response), events.index(after)]
        if positions != [positions[0], positions[0] + 1, positions[0] + 2]:
            raise base.RunnerError("protocol_failure", "Call events are not adjacent")
    schedule = _specs(frozen["manifest"])
    if [base.CallSpec(**event["call_spec"]) for event in done] != schedule[:len(done)]:
        raise base.RunnerError("protocol_failure", "Calls are not a frozen-order prefix")
    if len(done) > 40:
        raise base.RunnerError("protocol_failure", "Maximum API budget exceeded")

    sample_aggregates = [event for event in events if event["event_type"] == "SAMPLE_AGGREGATE"]
    keys = [(event["sample_id"], event.get("aggregate")) for event in sample_aggregates]
    if len({sample_id for sample_id, _ in keys}) != len(keys):
        raise base.RunnerError("protocol_failure", "Duplicate sample aggregate")
    for event in sample_aggregates:
        values = _sample_values(events, event["sample_id"])
        if len(values) != 3 or event.get("raw_normalized_verdicts") != values:
            raise base.RunnerError("protocol_failure", "Sample aggregate inputs mismatch")
        if event.get("aggregate") != base._aggregate(values):
            raise base.RunnerError("protocol_failure", "Sample aggregate mismatch")
    confirmations = [event for event in events if event["event_type"] == "CASE_CONFIRMATION"]
    if len({event["case_id"] for event in confirmations}) != len(confirmations):
        raise base.RunnerError("protocol_failure", "Duplicate case confirmation")
    for event in confirmations:
        sample_events = [
            aggregate for aggregate in sample_aggregates
            if aggregate["case_id"] == event["case_id"]
        ]
        verdicts = [_aggregate_verdict(aggregate["aggregate"]) for aggregate in sample_events]
        if len(verdicts) != 2 or event.get("new_production_sample_verdicts") != verdicts:
            raise base.RunnerError("protocol_failure", "Case confirmation inputs mismatch")
        if event.get("classification") != _classify_case(verdicts):
            raise base.RunnerError("protocol_failure", "Case confirmation classification mismatch")
    experiment_verdicts = [
        event for event in events if event["event_type"] == "EXPERIMENT_VERDICT"
    ]
    if len(experiment_verdicts) > 1:
        raise base.RunnerError("protocol_failure", "Duplicate experiment verdict")
    if experiment_verdicts:
        expected = _overall_verdict(events, frozen["manifest"]["cases"])
        if experiment_verdicts[0].get("verdict") != expected:
            raise base.RunnerError("protocol_failure", "Experiment verdict mismatch")
    if finals:
        if not experiment_verdicts:
            raise base.RunnerError("protocol_failure", "Completed run lacks experiment verdict")
        final = events[-1]
        if (
            len(done) != 40 or final.get("api_calls") != 40
            or final.get("production_calls") != 10
            or final.get("evaluator_calls") != 30
            or final.get("verdict") != experiment_verdicts[0]["verdict"]
        ):
            raise base.RunnerError("protocol_failure", "RUN_COMPLETED accounting mismatch")
    return done


def _append_prepared(
    checkpoint: base.Checkpoint, spec: base.CallSpec, index: int,
    frozen: dict[str, Any],
) -> None:
    checkpoint.append(
        "CALL_PREPARED", call_index=index, call_spec=asdict(spec),
        request_role=spec.role,
        relevant_hashes={
            "manifest_sha256": frozen["manifest_sha256"],
            "source_manifest_sha256": frozen["source"]["manifest_sha256"],
            "source_checkpoint_sha256": frozen["manifest"]["source_run"]["checkpoint_sha256"],
        },
    )


def _run_production(
    client: Any, checkpoint: base.Checkpoint, spec: base.CallSpec, index: int,
    case: dict[str, Any], frozen: dict[str, Any],
) -> None:
    from ai.cognitive_core import CognitiveCore
    from Conversation_Lab.selected_action_evaluator import selected_action_from_cognitive_turn

    prompt_spec = frozen["source"]["manifest"]["production_prompt"]["git_path"]
    prompt_sha = frozen["source"]["manifest"]["production_prompt"]["sha256"]
    prompt_bytes = _git_bytes(prompt_spec)
    if hashlib.sha256(prompt_bytes).hexdigest().upper() != prompt_sha:
        raise base.RunnerError("hash_failure", "Production prompt changed")
    FrozenCore = type(
        "FrozenConfirmationCore", (CognitiveCore,),
        {"_system_prompt": staticmethod(lambda: prompt_bytes.decode("utf-8"))},
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
            "expanded_observable_action": source._expanded_projection(result),
        },
    )


def _production_event(
    events: list[dict[str, Any]], sample_id: str,
) -> dict[str, Any]:
    matches = [
        event for event in _completed(events)
        if event["call_spec"]["key"] == f"production.{sample_id}"
    ]
    if len(matches) != 1:
        raise base.RunnerError("protocol_failure", f"Missing production sample: {sample_id}")
    return matches[0]


def _run_evaluator(
    client: Any, checkpoint: base.Checkpoint, spec: base.CallSpec, index: int,
    frozen: dict[str, Any], events: list[dict[str, Any]],
) -> None:
    if spec.selected_on_left is None:
        raise base.RunnerError("protocol_failure", "Evaluator orientation missing")
    case_id = _base_case(frozen["manifest"], spec.case_id)
    case = frozen["source"]["cases"][case_id]
    production = _production_event(events, spec.case_id)["normalized_result"]
    if spec.role == "historical_evaluator":
        selected = production["observable_action"]
        challenger = frozen["source"]["challengers"][case_id]
        prompt_path = source.HISTORICAL / "fsb_01_evaluator_prompt_672455a.txt"
        schema_path = source.HISTORICAL / "fsb_01_evaluator_schema_672455a.json"
        schema_name = "confirmation_historical_verdict"
    else:
        selected = production["expanded_observable_action"]
        benchmark = (
            frozen["source"]["m_pair"]["action_b"]
            if case_id == "FSB-M"
            else frozen["source"]["d_pair"]["direct_discrimination"]
        )
        challenger = source._benchmark_projection(benchmark)
        prompt_path = source.PRODUCT / "fsb_m_product_policy_evaluator_prompt_1550ec2.txt"
        schema_path = source.PRODUCT / "fsb_m_product_policy_evaluator_schema_1550ec2.json"
        schema_name = "confirmation_product_verdict"
    left, right = (
        (selected, challenger) if spec.selected_on_left else (challenger, selected)
    )
    request = source._evaluator_request(
        model=spec.model, prompt_path=prompt_path, schema_path=schema_path,
        context=case["frozen_input"], left=left, right=right,
        schema_name=schema_name,
    )
    response = client.chat.completions.create(**request)
    base._append_response(checkpoint, spec, index, response, request)
    try:
        payload = json.loads(response.choices[0].message.content)
        if set(payload) != {"verdict"}:
            raise ValueError("Unexpected evaluator fields")
        verdict = payload["verdict"]
        allowed = set(_read_json(schema_path)["properties"]["verdict"]["enum"])
        if verdict not in allowed:
            raise ValueError("Unknown evaluator verdict")
    except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise base.RunnerError("structural_failure", "Invalid evaluator response") from error
    checkpoint.append(
        "CALL_COMPLETED", call_index=index, call_spec=asdict(spec),
        normalized_result={
            "blind_verdict": verdict,
            "observable_verdict": base._normalize_blind(verdict, bool(spec.selected_on_left)),
        },
    )


def _classify_case(verdicts: list[str]) -> str:
    if len(verdicts) != 2 or "ABSTAIN" in verdicts:
        return "INCONCLUSIVE"
    misses = verdicts.count("CHALLENGER_HIGHER")
    if misses == 2:
        return "REPRODUCED"
    if misses == 0:
        return "NOT_REPRODUCED"
    return "MIXED"


def _append_derived_events(
    checkpoint: base.Checkpoint, events: list[dict[str, Any]],
    frozen: dict[str, Any],
) -> None:
    existing_samples = {
        event["sample_id"] for event in events if event["event_type"] == "SAMPLE_AGGREGATE"
    }
    for sample_id in frozen["manifest"]["evaluator_order"]:
        values = _sample_values(events, sample_id)
        if len(values) == 3 and sample_id not in existing_samples:
            checkpoint.append(
                "SAMPLE_AGGREGATE", sample_id=sample_id,
                case_id=_base_case(frozen["manifest"], sample_id),
                raw_normalized_verdicts=values,
                aggregate=base._aggregate(values),
            )
            events = checkpoint.read()
            existing_samples.add(sample_id)
    existing_cases = {
        event["case_id"] for event in events if event["event_type"] == "CASE_CONFIRMATION"
    }
    for case_id in frozen["manifest"]["cases"]:
        sample_events = [
            event for event in events
            if event["event_type"] == "SAMPLE_AGGREGATE" and event["case_id"] == case_id
        ]
        if len(sample_events) == 2 and case_id not in existing_cases:
            verdicts = [_aggregate_verdict(event["aggregate"]) for event in sample_events]
            checkpoint.append(
                "CASE_CONFIRMATION", case_id=case_id,
                new_production_sample_verdicts=verdicts,
                classification=_classify_case(verdicts),
                original_frozen_verdict="CHALLENGER_HIGHER",
            )
            events = checkpoint.read()
            existing_cases.add(case_id)


def _overall_verdict(events: list[dict[str, Any]], cases: list[str]) -> str:
    classifications = {
        event["case_id"]: event["classification"]
        for event in events if event["event_type"] == "CASE_CONFIRMATION"
    }
    if set(classifications) != set(cases) or "INCONCLUSIVE" in classifications.values():
        return "INCONCLUSIVE"
    values = list(classifications.values())
    if all(value == "REPRODUCED" for value in values):
        return "FAILURE_PATTERN_REPRODUCED"
    if all(value == "NOT_REPRODUCED" for value in values):
        return "FAILURE_PATTERN_NOT_REPRODUCED"
    return "FAILURE_PATTERN_PARTIALLY_REPRODUCED"


def _new_run(
    frozen: dict[str, Any], root: Optional[Path] = None,
) -> tuple[str, Path, base.Checkpoint]:
    output = RUNS_ROOT if root is None else root
    run_id = f"fsbrevconf-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = output / run_id
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "head": HEAD, "manifest_sha256": frozen["manifest_sha256"],
        "freeze_sha256": frozen["freeze_sha256"],
        "runner_sha256": frozen["runner_sha256"],
        "source_runner_sha256": frozen["source_runner_sha256"],
        "retry": 0, "smoke_calls": 0, "adaptive_sampling": False,
        "maximum_api_calls": 40,
    }
    base._write_metadata(run_dir / "run.json", metadata)
    checkpoint = base.Checkpoint(run_dir / "checkpoint.jsonl")
    checkpoint.append(
        "RUN_STARTED", run_id=run_id, head=HEAD,
        manifest_sha256=frozen["manifest_sha256"],
        authorization_mode="explicit --live-new-run",
    )
    return run_id, run_dir, checkpoint


def _load_run(run_id: str, frozen: dict[str, Any]):
    if not run_id or Path(run_id).name != run_id:
        raise base.RunnerError("protocol_failure", "Invalid run id")
    run_dir = RUNS_ROOT / run_id
    metadata_path = run_dir / "run.json"
    checkpoint = base.Checkpoint(run_dir / "checkpoint.jsonl")
    if not metadata_path.is_file() or not checkpoint.path.is_file():
        raise base.RunnerError("protocol_failure", f"Unknown/incomplete run: {run_id}")
    metadata = _read_json(metadata_path)
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
        done = validate_checkpoint(metadata, events, frozen, for_resume=False)
        _append_derived_events(checkpoint, events, frozen)
        events = checkpoint.read()
        done = validate_checkpoint(metadata, events, frozen, for_resume=False)
        if len(done) == 40:
            verdict = _overall_verdict(events, frozen["manifest"]["cases"])
            checkpoint.append("EXPERIMENT_VERDICT", verdict=verdict)
            checkpoint.append(
                "RUN_COMPLETED", api_calls=40, production_calls=10,
                evaluator_calls=30, verdict=verdict,
            )
            print(f"RUN_COMPLETED={metadata['run_id']}", flush=True)
            print(f"CHECKPOINT={checkpoint.path}", flush=True)
            return
        spec = schedule[len(done)]
        index = len(done) + 1
        _append_prepared(checkpoint, spec, index, frozen)
        try:
            if spec.role == "production_decision":
                _run_production(
                    client, checkpoint, spec, index,
                    frozen["samples"][spec.case_id], frozen,
                )
            else:
                _run_evaluator(client, checkpoint, spec, index, frozen, events)
        except BaseException as error:
            checkpoint.append(
                "STOP", category=getattr(error, "category", "api_failure"),
                call_index=index, call_spec=asdict(spec),
                error_type=type(error).__name__, error_message=str(error),
            )
            raise


def _synthetic_complete(
    checkpoint: base.Checkpoint, metadata: dict[str, Any],
    frozen: dict[str, Any], spec: base.CallSpec,
) -> None:
    done = validate_checkpoint(metadata, checkpoint.read(), frozen, for_resume=False)
    index = len(done) + 1
    _append_prepared(checkpoint, spec, index, frozen)
    checkpoint.append(
        "API_RESPONSE", call_index=index, call_spec=asdict(spec),
        request_sha256=base._sha_value({"dry": spec.key}),
        response_raw={"synthetic": True}, token_usage={"total_tokens": 0},
    )
    if spec.role == "production_decision":
        normalized = {
            "cognitive_turn_result": {"dry_run": True},
            "observable_action": {"decision_intent": "dry", "semantic_focus": spec.key, "basis_evidence_refs": []},
            "expanded_observable_action": {"decision_intent": "dry", "action_content_kind": "dry", "semantic_content": spec.key, "response_target_interaction": "dry", "reply_text": spec.key},
        }
    else:
        verdict = "RIGHT_HIGHER" if spec.selected_on_left else "LEFT_HIGHER"
        normalized = {"blind_verdict": verdict, "observable_verdict": "CHALLENGER_HIGHER"}
    checkpoint.append(
        "CALL_COMPLETED", call_index=index, call_spec=asdict(spec),
        normalized_result=normalized,
    )


def dry_run(frozen: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="fsbrevconf-dry-") as temporary:
        root = Path(temporary)
        _, run_dir, checkpoint = _new_run(frozen, root)
        metadata = _read_json(run_dir / "run.json")
        schedule = _specs(frozen["manifest"])
        for spec in schedule:
            _synthetic_complete(checkpoint, metadata, frozen, spec)
            _append_derived_events(checkpoint, checkpoint.read(), frozen)
        events = checkpoint.read()
        done = validate_checkpoint(metadata, events, frozen, for_resume=False)
        if len(done) != 40:
            raise base.RunnerError("protocol_failure", "Dry-run schedule did not complete")
        if _overall_verdict(events, frozen["manifest"]["cases"]) != "FAILURE_PATTERN_REPRODUCED":
            raise base.RunnerError("protocol_failure", "Dry-run classification failed")
        checkpoint.append("EXPERIMENT_VERDICT", verdict="FAILURE_PATTERN_REPRODUCED")
        checkpoint.append(
            "RUN_COMPLETED", api_calls=40, production_calls=10,
            evaluator_calls=30, verdict="FAILURE_PATTERN_REPRODUCED",
        )
        validate_checkpoint(metadata, checkpoint.read(), frozen, for_resume=False)

        _, prefix_dir, prefix = _new_run(frozen, root)
        prefix_meta = _read_json(prefix_dir / "run.json")
        for spec in schedule[:4]:
            _synthetic_complete(prefix, prefix_meta, frozen, spec)
        validate_checkpoint(prefix_meta, prefix.read(), frozen, for_resume=True)

        _, dangling_dir, dangling = _new_run(frozen, root)
        dangling_meta = _read_json(dangling_dir / "run.json")
        _append_prepared(dangling, schedule[0], 1, frozen)
        try:
            validate_checkpoint(dangling_meta, dangling.read(), frozen, for_resume=True)
        except base.RunnerError:
            dangling_rejected = True
        else:
            dangling_rejected = False
        if not dangling_rejected:
            raise base.RunnerError("protocol_failure", "Dangling resume was accepted")

        _, order_dir, wrong_order = _new_run(frozen, root)
        order_meta = _read_json(order_dir / "run.json")
        _synthetic_complete(wrong_order, order_meta, frozen, schedule[1])
        try:
            validate_checkpoint(order_meta, wrong_order.read(), frozen, for_resume=False)
        except base.RunnerError:
            order_rejected = True
        else:
            order_rejected = False
        if not order_rejected:
            raise base.RunnerError("protocol_failure", "Out-of-order call was accepted")

        _, stop_dir, stopped = _new_run(frozen, root)
        stop_meta = _read_json(stop_dir / "run.json")
        stopped.append("STOP", category="protocol_failure", error_message="dry")
        try:
            validate_checkpoint(stop_meta, stopped.read(), frozen, for_resume=True)
        except base.RunnerError:
            stop_rejected = True
        else:
            stop_rejected = False
        if not stop_rejected:
            raise base.RunnerError("protocol_failure", "Terminal STOP resume was accepted")

        return {
            "result": "PASS", "api_calls": 0, "head": HEAD,
            "cases": frozen["manifest"]["cases"],
            "new_production_selections": 10, "production_repetitions_per_case": 2,
            "evaluator_calls": 30, "maximum_calls": 40,
            "all_production_precedes_evaluation": "PASS",
            "checkpoint_hash_chain": "PASS", "clean_prefix_resume": "PASS",
            "dangling_resume_rejected": "PASS", "out_of_order_rejected": "PASS",
            "terminal_stop_resume_rejected": "PASS",
            "pseudoreplication_guard": "PASS",
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--live-new-run", action="store_true")
    modes.add_argument("--resume", metavar="RUN_ID")
    modes.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        frozen = preflight()
        if args.dry_run:
            print(json.dumps(dry_run(frozen), ensure_ascii=False, sort_keys=True))
            return 0
        if args.live_new_run:
            _, _, checkpoint = _new_run(frozen)
            metadata = _read_json(checkpoint.path.parent / "run.json")
        else:
            _, metadata, checkpoint = _load_run(args.resume, frozen)
        execute_live(frozen, metadata, checkpoint)
        return 0
    except base.RunnerError as error:
        print(f"STOP_CATEGORY={error.category}", file=sys.stderr, flush=True)
        print(f"STOP_MESSAGE={error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
