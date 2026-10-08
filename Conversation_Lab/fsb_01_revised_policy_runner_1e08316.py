"""Resume-safe runner for the FSB-01 revised value-policy regression.

No mode performs live work implicitly. ``--dry-run`` never imports OpenAI and
uses synthetic checkpoint events only. ``--qualification-only`` (and the
backward-compatible ``--new-run`` alias) is capped at the six qualification
calls. Production requires a later explicit ``--resume-production RUN_ID``.
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

from Conversation_Lab import fsb_01_live_runner_672455a as base  # noqa: E402


PACKAGE = ROOT / "Conversation_Lab/frozen/fsb_01_revised_policy_1e08316"
HISTORICAL = ROOT / "Conversation_Lab/frozen/fsb_01_672455a"
PRODUCT = ROOT / "Conversation_Lab/frozen/fsb_m_product_policy_ab_1550ec2"
RUNS_ROOT = ROOT / "Conversation_Lab/traces/fsb_01_revised_policy_1e08316_runs"
MANIFEST_NAME = "fsb_01_revised_manifest_1e08316.json"
VALIDATOR_NAME = "fsb_01_revised_offline_validator_1e08316.py"
BASE_RUNNER = ROOT / "Conversation_Lab/fsb_01_live_runner_672455a.py"
BASE_RUNNER_SHA256 = "00038CB5913D51C54A454DD3B3F0D5CF2444CA9B39E6C9BDC14C144637E562BF"
FROZEN_SHA256 = {
    "fsb_01_revised_freeze_report_1e08316.md": "F771813911F921845FBC0D9734684C23299586C965BE8095A9F0B44336EA4E89",
    "fsb_01_revised_manifest_1e08316.json": "11215F5032A466E8B4000303348BA80E867ABAF7EE81D5C5F7220961DACD3514",
    "fsb_01_revised_measurement_plan_1e08316.md": "7E0738A822BF9203F29290F1DBCCA8029C26A12A3E6597940BBFE5F36C9AFD49",
    "fsb_01_revised_offline_validator_1e08316.py": "62E47EE70CFDF4FE43A20235BECF70487C029081A901D0A8562B0D95393779EE",
    "fsb_d_case_1e08316.json": "27B60F09C230C6326EF5641C45B851A1346E6D9878E191DB7CE6A4BCAEC8117F",
    "fsb_d_policy_pair_1e08316.json": "58819C30A951CE6A2BF68EC24611B270BA8D3548BB9344EAF78F679FC73DA973",
}
TERMINAL_STOPS = {
    "hash_failure", "head_failure", "offline_validation_failure",
    "protocol_failure", "qualification_failure", "structural_failure",
    "structured_output_failure", "validation_failure", "api_failure",
    "billing_failure", "transport_failure",
}
QUALIFICATION_CALL_CAP = 6


def _git_bytes(spec: str) -> bytes:
    return subprocess.run(
        ["git", "show", spec], cwd=ROOT, check=True, capture_output=True
    ).stdout


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _specs(manifest: dict[str, Any]) -> list[base.CallSpec]:
    specs = [
        base.CallSpec(
            key=f"qualification.FSB-D.{repeat}", phase="qualification",
            case_id="FSB-D", role="qualification_evaluator",
            model=manifest["evaluator_model"], repeat=repeat,
            selected_on_left=direct_on_left,
        )
        for repeat, direct_on_left in enumerate(
            manifest["qualification"]["direct_on_left"], 1
        )
    ]
    specs.extend(
        base.CallSpec(
            key=f"production.{case_id}", phase="production",
            case_id=case_id, role="production_decision",
            model=manifest["production_model"],
        )
        for case_id in manifest["production_order"]
    )
    historical = set(manifest["historical_evaluator_cases"])
    for case_id in manifest["production_order"]:
        role = "historical_evaluator" if case_id in historical else "product_evaluator"
        specs.extend(
            base.CallSpec(
                key=f"posthoc.{case_id}.{repeat}", phase="posthoc",
                case_id=case_id, role=role, model=manifest["evaluator_model"],
                repeat=repeat, selected_on_left=selected_on_left,
            )
            for repeat, selected_on_left in enumerate(
                manifest["posthoc_orientations_selected_on_left"][case_id], 1
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
            raise base.RunnerError("hash_failure", f"SHA mismatch: {name}")
        hashes[name] = actual
    if base._sha_file(BASE_RUNNER) != BASE_RUNNER_SHA256:
        raise base.RunnerError("hash_failure", "Checkpoint helper changed")

    manifest = _read_json(PACKAGE / MANIFEST_NAME)
    for section, root in (("historical_source", HISTORICAL), ("product_policy_source", PRODUCT)):
        for name, expected in manifest[section]["sha256"].items():
            if base._sha_file(root / name) != expected:
                raise base.RunnerError("hash_failure", f"Dependency changed: {name}")
    prompt = _git_bytes(manifest["production_prompt"]["git_path"])
    if hashlib.sha256(prompt).hexdigest().upper() != manifest["production_prompt"]["sha256"]:
        raise base.RunnerError("hash_failure", "Production prompt changed")
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
    schedule = _specs(manifest)
    if len(schedule) != 34:
        raise base.RunnerError("protocol_failure", "Schedule is not exactly 34 calls")
    if any(spec.role != "qualification_evaluator" for spec in schedule[:6]):
        raise base.RunnerError("protocol_failure", "Qualification must be first")
    if any(spec.role != "production_decision" for spec in schedule[6:13]):
        raise base.RunnerError("protocol_failure", "Seven production calls must follow qualification")

    historical_cases = _read_json(HISTORICAL / "fsb_01_cases_672455a.json")
    cases = {case["case_id"]: case for case in historical_cases["cases"]}
    cases["FSB-D"] = _read_json(PACKAGE / "fsb_d_case_1e08316.json")
    challengers = {
        item["case_id"]: item["challenger"]
        for item in _read_json(HISTORICAL / "fsb_01_challengers_672455a.json")["challengers"]
    }
    d_pair = _read_json(PACKAGE / "fsb_d_policy_pair_1e08316.json")
    m_pair = _read_json(PRODUCT / "fsb_m_product_policy_pair_1550ec2.json")
    return {
        "head": actual_head, "manifest": manifest,
        "manifest_sha256": hashes[MANIFEST_NAME], "freeze_sha256": hashes,
        "offline_validation": validation_payload, "cases": cases,
        "challengers": challengers, "d_pair": d_pair, "m_pair": m_pair,
        "runner_sha256": base._sha_file(Path(__file__)),
        "base_runner_sha256": BASE_RUNNER_SHA256,
    }


def _completed(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event["event_type"] == "CALL_COMPLETED"]


def _qualification_values(events: list[dict[str, Any]]) -> list[str]:
    return [
        event["normalized_result"]["qualification_verdict"]
        for event in _completed(events)
        if event["call_spec"]["role"] == "qualification_evaluator"
    ]


def _posthoc_values(events: list[dict[str, Any]], case_id: str) -> list[str]:
    return [
        event["normalized_result"]["observable_verdict"]
        for event in _completed(events)
        if event["call_spec"]["phase"] == "posthoc"
        and event["call_spec"]["case_id"] == case_id
    ]


def _qualification_passed(events: list[dict[str, Any]]) -> bool:
    gates = [event for event in events if event["event_type"] == "QUALIFICATION_AGGREGATE"]
    return len(gates) == 1 and gates[0].get("aggregate") == "DIRECT_HIGHER"


def _production_authorized(events: list[dict[str, Any]]) -> bool:
    return sum(event["event_type"] == "PRODUCTION_AUTHORIZED" for event in events) == 1


def validate_checkpoint(
    metadata: dict[str, Any], events: list[dict[str, Any]], frozen: dict[str, Any],
    *, for_resume: bool,
) -> list[dict[str, Any]]:
    for field in (
        "head", "manifest_sha256", "freeze_sha256", "runner_sha256",
        "base_runner_sha256",
    ):
        if metadata.get(field) != frozen[field]:
            raise base.RunnerError("protocol_failure", f"Run {field} mismatch")
    if (
        metadata.get("maximum_api_calls") != 34
        or metadata.get("qualification_call_cap") != QUALIFICATION_CALL_CAP
        or metadata.get("retry") != 0
    ):
        raise base.RunnerError("protocol_failure", "Run budget/retry metadata mismatch")
    if not events or events[0]["event_type"] != "RUN_STARTED":
        raise base.RunnerError("protocol_failure", "Missing RUN_STARTED")
    if sum(event["event_type"] == "RUN_STARTED" for event in events) != 1:
        raise base.RunnerError("protocol_failure", "RUN_STARTED must be unique")
    completed_markers = [i for i, event in enumerate(events) if event["event_type"] == "RUN_COMPLETED"]
    if completed_markers and (len(completed_markers) != 1 or completed_markers[0] != len(events) - 1):
        raise base.RunnerError("protocol_failure", "RUN_COMPLETED must be unique and final")
    if completed_markers and for_resume:
        raise base.RunnerError("protocol_failure", "Completed run cannot resume")
    if for_resume and any(
        event["event_type"] == "STOP" and event.get("category") in TERMINAL_STOPS
        for event in events
    ):
        raise base.RunnerError("protocol_failure", "Terminal STOP cannot resume")

    pauses = [event for event in events if event["event_type"] == "QUALIFICATION_PAUSED"]
    authorizations = [event for event in events if event["event_type"] == "PRODUCTION_AUTHORIZED"]
    if len(pauses) > 1 or len(authorizations) > 1:
        raise base.RunnerError("protocol_failure", "Duplicate qualification pause/authorization")
    if authorizations and not pauses:
        raise base.RunnerError("protocol_failure", "Production authorization has no review pause")
    if for_resume and pauses and not authorizations:
        raise base.RunnerError(
            "protocol_failure",
            "Qualification review is paused; use --resume-production explicitly",
        )

    prepared = [event for event in events if event["event_type"] == "CALL_PREPARED"]
    responses = [event for event in events if event["event_type"] == "API_RESPONSE"]
    done = _completed(events)
    if not (len(prepared) == len(responses) == len(done)):
        raise base.RunnerError("protocol_failure", "Dangling/partial call could duplicate an API request")
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
    if len(done) > 34:
        raise base.RunnerError("protocol_failure", "Maximum API budget exceeded")

    gates = [event for event in events if event["event_type"] == "QUALIFICATION_AGGREGATE"]
    if len(gates) > 1:
        raise base.RunnerError("protocol_failure", "Duplicate qualification aggregate")
    if len(done) > QUALIFICATION_CALL_CAP and (
        not _qualification_passed(events) or not _production_authorized(events)
    ):
        raise base.RunnerError(
            "protocol_failure",
            "Production occurred without passed qualification and explicit authorization",
        )
    if gates:
        values = _qualification_values(events)
        if len(values) != 6 or gates[0].get("raw_normalized_verdicts") != values:
            raise base.RunnerError("protocol_failure", "Qualification inputs mismatch")
    if pauses:
        if len(done) < QUALIFICATION_CALL_CAP or not _qualification_passed(events):
            raise base.RunnerError("protocol_failure", "Invalid qualification review pause")
        if pauses[0].get("api_calls") != QUALIFICATION_CALL_CAP:
            raise base.RunnerError("protocol_failure", "Qualification pause call cap mismatch")
    if authorizations:
        if authorizations[0].get("qualification_aggregate") != "DIRECT_HIGHER":
            raise base.RunnerError("protocol_failure", "Invalid production authorization")

    aggregates = [event for event in events if event["event_type"] == "AGGREGATE"]
    if len({event["case_id"] for event in aggregates}) != len(aggregates):
        raise base.RunnerError("protocol_failure", "Duplicate post-hoc aggregate")
    for event in aggregates:
        values = _posthoc_values(events, event["case_id"])
        if len(values) != 3 or event.get("raw_normalized_verdicts") != values:
            raise base.RunnerError("protocol_failure", "Post-hoc aggregate inputs mismatch")
        if event.get("aggregate") != base._aggregate(values):
            raise base.RunnerError("protocol_failure", "Post-hoc aggregate mismatch")
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
            "production_prompt_sha256": frozen["manifest"]["production_prompt"]["sha256"] if spec.role == "production_decision" else None,
            "case_sha256": frozen["manifest"]["fsb_d"]["case_sha256"] if spec.case_id == "FSB-D" else None,
        },
    )


def _expanded_projection(result: Any) -> dict[str, str]:
    intent = getattr(result.decision_intent, "value", result.decision_intent)
    contents = tuple(result.system_action.contents)
    targets = tuple(result.system_action.response_targets)
    segments = tuple(result.reply_segments)
    return {
        "decision_intent": str(intent),
        "action_content_kind": " + ".join(str(getattr(item.kind, "value", item.kind)) for item in contents),
        "semantic_content": "\n".join(item.semantic_content for item in contents),
        "response_target_interaction": " + ".join(str(getattr(item.interaction, "value", item.interaction)) for item in targets) if targets else "none",
        "reply_text": "\n".join(item.text for item in segments),
    }


def _benchmark_projection(action: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in action.items() if key != "strategy"}


def _production_event(events: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    matches = [
        event for event in _completed(events)
        if event["call_spec"]["key"] == f"production.{case_id}"
    ]
    if len(matches) != 1:
        raise base.RunnerError("protocol_failure", f"Missing production result: {case_id}")
    return matches[0]


def _run_production(
    client: Any, checkpoint: base.Checkpoint, spec: base.CallSpec, index: int,
    case: dict[str, Any], frozen: dict[str, Any],
) -> None:
    from ai.cognitive_core import CognitiveCore
    from Conversation_Lab.selected_action_evaluator import selected_action_from_cognitive_turn

    prompt_bytes = _git_bytes(frozen["manifest"]["production_prompt"]["git_path"])
    if hashlib.sha256(prompt_bytes).hexdigest().upper() != frozen["manifest"]["production_prompt"]["sha256"]:
        raise base.RunnerError("hash_failure", "Prompt changed after preflight")
    FrozenCore = type(
        "FrozenPromptCognitiveCore", (CognitiveCore,),
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
            "expanded_observable_action": _expanded_projection(result),
        },
    )


def _evaluator_request(
    *, model: str, prompt_path: Path, schema_path: Path,
    context: dict[str, Any], left: dict[str, Any], right: dict[str, Any],
    schema_name: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": json.dumps({"frozen_context": context, "left_action": left, "right_action": right}, ensure_ascii=False, sort_keys=True)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": _read_json(schema_path)},
        },
    }


def _run_evaluator(
    client: Any, checkpoint: base.Checkpoint, spec: base.CallSpec, index: int,
    frozen: dict[str, Any], events: list[dict[str, Any]],
) -> None:
    if spec.selected_on_left is None:
        raise base.RunnerError("protocol_failure", "Evaluator orientation missing")
    case = frozen["cases"][spec.case_id]
    product_prompt = PRODUCT / "fsb_m_product_policy_evaluator_prompt_1550ec2.txt"
    product_schema = PRODUCT / "fsb_m_product_policy_evaluator_schema_1550ec2.json"
    historical_prompt = HISTORICAL / "fsb_01_evaluator_prompt_672455a.txt"
    historical_schema = HISTORICAL / "fsb_01_evaluator_schema_672455a.json"

    if spec.role == "qualification_evaluator":
        direct = _benchmark_projection(frozen["d_pair"]["direct_discrimination"])
        working = _benchmark_projection(frozen["d_pair"]["testable_working_version"])
        left, right = (direct, working) if spec.selected_on_left else (working, direct)
        prompt_path, schema_path = product_prompt, product_schema
        schema_name = "fsb_d_qualification_verdict"
    else:
        production = _production_event(events, spec.case_id)["normalized_result"]
        if spec.role == "historical_evaluator":
            selected = production["observable_action"]
            challenger = frozen["challengers"][spec.case_id]
            prompt_path, schema_path = historical_prompt, historical_schema
        else:
            selected = production["expanded_observable_action"]
            benchmark = (
                frozen["m_pair"]["action_b"] if spec.case_id == "FSB-M"
                else frozen["d_pair"]["direct_discrimination"]
            )
            challenger = _benchmark_projection(benchmark)
            prompt_path, schema_path = product_prompt, product_schema
        left, right = (selected, challenger) if spec.selected_on_left else (challenger, selected)
        schema_name = "revised_policy_posthoc_verdict"

    request = _evaluator_request(
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
    if spec.role == "qualification_evaluator":
        direct_higher = verdict == "LEFT_HIGHER" if spec.selected_on_left else verdict == "RIGHT_HIGHER"
        normalized = {
            "blind_verdict": verdict,
            "qualification_verdict": (
                "DIRECT_HIGHER" if direct_higher else
                "EQUIVALENT_OR_BOTH_ACCEPTABLE" if verdict == "EQUIVALENT_OR_BOTH_ACCEPTABLE" else
                "ABSTAIN" if verdict == "INSUFFICIENT_BASIS" else
                "WORKING_VERSION_HIGHER"
            ),
        }
    else:
        normalized = {
            "blind_verdict": verdict,
            "observable_verdict": base._normalize_blind(verdict, bool(spec.selected_on_left)),
        }
    checkpoint.append(
        "CALL_COMPLETED", call_index=index, call_spec=asdict(spec),
        normalized_result=normalized,
    )


def _append_aggregates(
    checkpoint: base.Checkpoint, events: list[dict[str, Any]],
    frozen: dict[str, Any],
) -> None:
    if len(_qualification_values(events)) == 6 and not any(
        event["event_type"] == "QUALIFICATION_AGGREGATE" for event in events
    ):
        values = _qualification_values(events)
        qualification_events = [
            event for event in _completed(events)
            if event["call_spec"]["role"] == "qualification_evaluator"
        ]
        left_support = any(
            event["normalized_result"]["qualification_verdict"] == "DIRECT_HIGHER"
            and event["call_spec"]["selected_on_left"]
            for event in qualification_events
        )
        right_support = any(
            event["normalized_result"]["qualification_verdict"] == "DIRECT_HIGHER"
            and not event["call_spec"]["selected_on_left"]
            for event in qualification_events
        )
        passed = values.count("DIRECT_HIGHER") >= 4 and left_support and right_support
        checkpoint.append(
            "QUALIFICATION_AGGREGATE", case_id="FSB-D",
            raw_normalized_verdicts=values,
            aggregate="DIRECT_HIGHER" if passed else "INDETERMINATE_OR_NOT_DIRECT_HIGHER",
            direct_support_both_orientations=left_support and right_support,
        )
        if not passed:
            checkpoint.append(
                "STOP", category="qualification_failure",
                error_message="Frozen FSB-D direct control did not qualify",
                api_calls=6, production_calls=0,
            )
        else:
            checkpoint.append(
                "QUALIFICATION_PAUSED", case_id="FSB-D",
                aggregate="DIRECT_HIGHER", api_calls=QUALIFICATION_CALL_CAP,
                production_calls=0,
                continuation="explicit --resume-production authorization required",
            )
        events = checkpoint.read()
    existing = {
        event["case_id"] for event in events if event["event_type"] == "AGGREGATE"
    }
    for case_id in frozen["manifest"]["production_order"]:
        values = _posthoc_values(events, case_id)
        if len(values) == 3 and case_id not in existing:
            checkpoint.append(
                "AGGREGATE", case_id=case_id,
                raw_normalized_verdicts=values,
                aggregate=base._aggregate(values),
            )


def _new_run(
    frozen: dict[str, Any], root: Optional[Path] = None,
) -> tuple[str, Path, base.Checkpoint]:
    output = RUNS_ROOT if root is None else root
    run_id = f"fsbrev-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = output / run_id
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "head": HEAD, "manifest_sha256": frozen["manifest_sha256"],
        "freeze_sha256": frozen["freeze_sha256"],
        "runner_sha256": frozen["runner_sha256"],
        "base_runner_sha256": frozen["base_runner_sha256"],
        "retry": 0, "smoke_calls": 0, "confirmations": 0,
        "adaptive_sampling": False, "maximum_api_calls": 34,
        "qualification_call_cap": QUALIFICATION_CALL_CAP,
    }
    base._write_metadata(run_dir / "run.json", metadata)
    checkpoint = base.Checkpoint(run_dir / "checkpoint.jsonl")
    checkpoint.append(
        "RUN_STARTED", run_id=run_id, head=HEAD,
        manifest_sha256=frozen["manifest_sha256"],
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


def _authorize_production(
    run_id: str, frozen: dict[str, Any],
) -> tuple[Path, dict[str, Any], base.Checkpoint]:
    if not run_id or Path(run_id).name != run_id:
        raise base.RunnerError("protocol_failure", "Invalid run id")
    run_dir = RUNS_ROOT / run_id
    metadata_path = run_dir / "run.json"
    checkpoint = base.Checkpoint(run_dir / "checkpoint.jsonl")
    if not metadata_path.is_file() or not checkpoint.path.is_file():
        raise base.RunnerError("protocol_failure", f"Unknown/incomplete run: {run_id}")
    metadata = _read_json(metadata_path)
    events = checkpoint.read()
    done = validate_checkpoint(metadata, events, frozen, for_resume=False)
    if len(done) != QUALIFICATION_CALL_CAP or not _qualification_passed(events):
        raise base.RunnerError("protocol_failure", "Qualification is not passed and review-paused")
    if not any(event["event_type"] == "QUALIFICATION_PAUSED" for event in events):
        raise base.RunnerError("protocol_failure", "Qualification review pause is missing")
    if _production_authorized(events):
        raise base.RunnerError("protocol_failure", "Production is already authorized")
    checkpoint.append(
        "PRODUCTION_AUTHORIZED", qualification_aggregate="DIRECT_HIGHER",
        qualification_api_calls=QUALIFICATION_CALL_CAP,
        authorization_mode="explicit --resume-production",
    )
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
        _append_aggregates(checkpoint, events, frozen)
        events = checkpoint.read()
        if any(event["event_type"] == "STOP" for event in events):
            raise base.RunnerError("qualification_failure", "Run stopped by frozen gate")
        if (
            len(done) == QUALIFICATION_CALL_CAP
            and _qualification_passed(events)
            and not _production_authorized(events)
        ):
            print(f"QUALIFICATION_PAUSED={metadata['run_id']}", flush=True)
            print(f"CHECKPOINT={checkpoint.path}", flush=True)
            return
        done = validate_checkpoint(metadata, events, frozen, for_resume=False)
        if len(done) == 34:
            checkpoint.append(
                "RUN_COMPLETED", api_calls=34, production_calls=7,
                evaluator_calls=27,
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
                    frozen["cases"][spec.case_id], frozen,
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
            "observable_action": {"decision_intent": "dry_run", "semantic_focus": spec.key, "basis_evidence_refs": []},
            "expanded_observable_action": {"decision_intent": "dry_run", "action_content_kind": "dry_run", "semantic_content": spec.key, "response_target_interaction": "dry_run", "reply_text": spec.key},
        }
    elif spec.role == "qualification_evaluator":
        normalized = {"blind_verdict": "LEFT_HIGHER" if spec.selected_on_left else "RIGHT_HIGHER", "qualification_verdict": "DIRECT_HIGHER"}
    else:
        normalized = {"blind_verdict": "EQUIVALENT_OR_BOTH_ACCEPTABLE", "observable_verdict": "EQUIVALENT_OR_BOTH_ACCEPTABLE"}
    checkpoint.append(
        "CALL_COMPLETED", call_index=index, call_spec=asdict(spec),
        normalized_result=normalized,
    )


def dry_run(frozen: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="fsbrev-dry-") as temporary:
        root = Path(temporary)
        _, run_dir, checkpoint = _new_run(frozen, root)
        metadata = _read_json(run_dir / "run.json")
        schedule = _specs(frozen["manifest"])
        for spec in schedule[:QUALIFICATION_CALL_CAP]:
            _synthetic_complete(checkpoint, metadata, frozen, spec)
            _append_aggregates(checkpoint, checkpoint.read(), frozen)
        events = checkpoint.read()
        done = validate_checkpoint(metadata, events, frozen, for_resume=False)
        if len(done) != QUALIFICATION_CALL_CAP or not _qualification_passed(events):
            raise base.RunnerError("protocol_failure", "Dry-run qualification schedule/gate failed")
        if not any(event["event_type"] == "QUALIFICATION_PAUSED" for event in events):
            raise base.RunnerError("protocol_failure", "Dry-run qualification did not pause")
        try:
            validate_checkpoint(metadata, events, frozen, for_resume=True)
        except base.RunnerError:
            ordinary_resume_at_pause_rejected = True
        else:
            ordinary_resume_at_pause_rejected = False
        if not ordinary_resume_at_pause_rejected:
            raise base.RunnerError("protocol_failure", "Ordinary resume crossed review pause")
        _, unauthorized_dir, unauthorized = _new_run(frozen, root)
        unauthorized_meta = _read_json(unauthorized_dir / "run.json")
        for spec in schedule[:QUALIFICATION_CALL_CAP]:
            _synthetic_complete(unauthorized, unauthorized_meta, frozen, spec)
            _append_aggregates(unauthorized, unauthorized.read(), frozen)
        _synthetic_complete(
            unauthorized, unauthorized_meta, frozen,
            schedule[QUALIFICATION_CALL_CAP],
        )
        try:
            validate_checkpoint(
                unauthorized_meta, unauthorized.read(), frozen, for_resume=False
            )
        except base.RunnerError:
            unauthorized_production_rejected = True
        else:
            unauthorized_production_rejected = False
        if not unauthorized_production_rejected:
            raise base.RunnerError("protocol_failure", "Qualification-only mode exceeded six calls")
        checkpoint.append(
            "PRODUCTION_AUTHORIZED", qualification_aggregate="DIRECT_HIGHER",
            qualification_api_calls=QUALIFICATION_CALL_CAP,
            authorization_mode="dry-run explicit authorization",
        )
        for spec in schedule[QUALIFICATION_CALL_CAP:]:
            _synthetic_complete(checkpoint, metadata, frozen, spec)
            _append_aggregates(checkpoint, checkpoint.read(), frozen)
        events = checkpoint.read()
        done = validate_checkpoint(metadata, events, frozen, for_resume=False)
        if len(done) != 34:
            raise base.RunnerError("protocol_failure", "Dry-run full authorized schedule failed")

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

        _, stop_dir, stopped = _new_run(frozen, root)
        stop_meta = _read_json(stop_dir / "run.json")
        stopped.append("STOP", category="qualification_failure", error_message="dry")
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
            "schedule_calls": 34, "qualification_calls": 6,
            "production_calls": 7, "posthoc_evaluator_calls": 21,
            "qualification_gate": "PASS_SYNTHETIC",
            "qualification_only_maximum_calls": QUALIFICATION_CALL_CAP,
            "qualification_review_pause": "PASS",
            "ordinary_resume_at_pause_rejected": "PASS",
            "unauthorized_seventh_call_rejected": "PASS",
            "checkpoint_hash_chain": "PASS",
            "clean_prefix_resume": "PASS",
            "dangling_resume_rejected": "PASS",
            "terminal_stop_resume_rejected": "PASS",
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--new-run", action="store_true", help="Alias for --qualification-only")
    modes.add_argument("--qualification-only", action="store_true")
    modes.add_argument("--resume", metavar="RUN_ID")
    modes.add_argument("--resume-production", metavar="RUN_ID")
    modes.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        frozen = preflight()
        if args.dry_run:
            print(json.dumps(dry_run(frozen), ensure_ascii=False, sort_keys=True))
            return 0
        if args.new_run or args.qualification_only:
            _, _, checkpoint = _new_run(frozen)
            metadata = _read_json(checkpoint.path.parent / "run.json")
        elif args.resume_production:
            _, metadata, checkpoint = _authorize_production(args.resume_production, frozen)
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
