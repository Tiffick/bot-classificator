"""Resume-safe runner for the frozen FSB-M product-policy A/B evaluation."""

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


HEAD = "1550ec2"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Conversation_Lab import fsb_mrs_live_runner_1fe07e9 as base  # noqa: E402


FREEZE = ROOT / "Conversation_Lab/frozen/fsb_m_product_policy_ab_1550ec2"
SOURCE = ROOT / "Conversation_Lab/frozen/fsb_01_672455a"
RUNS_ROOT = ROOT / "Conversation_Lab/traces/fsb_m_product_policy_ab_1550ec2_runs"
MANIFEST_NAME = "fsb_m_product_policy_manifest_1550ec2.json"
PAIR_NAME = "fsb_m_product_policy_pair_1550ec2.json"
PROMPT_NAME = "fsb_m_product_policy_evaluator_prompt_1550ec2.txt"
SCHEMA_NAME = "fsb_m_product_policy_evaluator_schema_1550ec2.json"
VALIDATOR_NAME = "fsb_m_product_policy_offline_validator_1550ec2.py"
SELF = Path(__file__).resolve()
FREEZE_SHA256 = {
    "fsb_m_product_policy_evaluator_prompt_1550ec2.txt": "828E44FA534E2203284712B093E9A86A31DD12AEA8AA617A7E536D22F1F34453",
    "fsb_m_product_policy_evaluator_schema_1550ec2.json": "0CF50E97EBF46F90F8ABECD8FC2A104BD8EC86CF9906D24A718F37CF5C96A216",
    "fsb_m_product_policy_freeze_report_1550ec2.md": "CC859AE2B2C3D6F6C54FA09DE79194E751E6D4DA9749EC4C07DA48CA48048845",
    "fsb_m_product_policy_guards_1550ec2.json": "3BD4F1D63BD508DA31111285EAEB3033733749D7C9F2516CC2FF3B0DBBE561A4",
    "fsb_m_product_policy_manifest_1550ec2.json": "E65EEEDD79B6B0BE091B71F981A646544A5851936646511367C255F2BC65637E",
    "fsb_m_product_policy_offline_validator_1550ec2.py": "B48A3DECC3A3C7C28A94DB6B05ECFA140576CABBDBABDC6D1053DC9CBDA31F9E",
    "fsb_m_product_policy_pair_1550ec2.json": "263F877DCD5346E399301E9C1B72EBC3BB3667C62791F10A0986C10C0E14C924",
}


def _specs(manifest: dict[str, Any]) -> list[base.CallSpec]:
    return [
        base.CallSpec(
            key=f"evaluator.FSB-M.{repeat}", phase="product_policy_ab",
            case_id="FSB-M", role="blind_evaluator",
            model=manifest["evaluator_model"], repeat=repeat,
            selected_on_left=a_on_left,
        )
        for repeat, a_on_left in enumerate(manifest["orientation_a_on_left"], 1)
    ]


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
    hashes: dict[str, str] = {}
    for name, expected in FREEZE_SHA256.items():
        actual = base._sha_file(FREEZE / name)
        if actual != expected:
            raise base.RunnerError("hash_failure", f"Freeze SHA mismatch: {name}")
        hashes[name] = actual

    manifest = base._read_json(FREEZE / MANIFEST_NAME)
    for name, expected in manifest["source_artifact_sha256"].items():
        if base._sha_file(SOURCE / name) != expected:
            raise base.RunnerError("hash_failure", f"Source artifact changed: {name}")
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
    if manifest["api_budget"] != {"production": 0, "evaluator": 6, "maximum_total": 6}:
        raise base.RunnerError("protocol_failure", "Frozen API budget mismatch")
    if manifest["evaluator_model"] != "gpt-6-astra" or len(_specs(manifest)) != 6:
        raise base.RunnerError("protocol_failure", "Frozen evaluator schedule mismatch")

    cases = base._read_json(SOURCE / "fsb_01_cases_672455a.json")["cases"]
    case = next(item for item in cases if item["case_id"] == "FSB-M")
    return {
        "head": actual_head,
        "manifest": manifest,
        "manifest_sha256": hashes[MANIFEST_NAME],
        "freeze_sha256": {"product_policy": hashes, "source": manifest["source_artifact_sha256"]},
        "pair": base._read_json(FREEZE / PAIR_NAME),
        "case": case,
        "offline_validation": validation,
        "runner_dependency_sha256": base._sha_file(SELF),
    }


def _normalize(verdict: str, a_on_left: bool) -> str:
    if verdict == "EQUIVALENT_OR_BOTH_ACCEPTABLE":
        return verdict
    if verdict == "INSUFFICIENT_BASIS":
        return "ABSTAIN"
    a_higher = verdict == "LEFT_HIGHER" if a_on_left else verdict == "RIGHT_HIGHER"
    return "A_HIGHER" if a_higher else "B_HIGHER"


def _aggregate(values: list[str]) -> str:
    for verdict in ("A_HIGHER", "B_HIGHER", "EQUIVALENT_OR_BOTH_ACCEPTABLE"):
        if values.count(verdict) >= 4:
            return verdict
    return "INDETERMINATE"


def _completed(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event["event_type"] == "CALL_COMPLETED"]


def validate_checkpoint(
    metadata: dict[str, Any], events: list[dict[str, Any]], frozen: dict[str, Any],
    *, for_resume: bool,
) -> list[dict[str, Any]]:
    for field in ("head", "manifest_sha256", "freeze_sha256", "runner_dependency_sha256"):
        if metadata.get(field) != frozen[field]:
            raise base.RunnerError("protocol_failure", f"Run {field} mismatch")
    if metadata.get("maximum_api_calls") != 6 or metadata.get("production_calls") != 0:
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
    if for_resume and any(event["event_type"] == "STOP" for event in events):
        raise base.RunnerError("protocol_failure", "Stopped run cannot be resumed")

    prepared = [event for event in events if event["event_type"] == "CALL_PREPARED"]
    responses = [event for event in events if event["event_type"] == "API_RESPONSE"]
    done = _completed(events)
    if len(prepared) != len(responses) or len(responses) != len(done):
        raise base.RunnerError("protocol_failure", "Dangling/partial call could cause duplicate request")
    for index, (before, response, after) in enumerate(zip(prepared, responses, done), 1):
        if {before["call_index"], response["call_index"], after["call_index"]} != {index}:
            raise base.RunnerError("protocol_failure", "Non-contiguous call indices")
        if not (before["call_spec"] == response["call_spec"] == after["call_spec"]):
            raise base.RunnerError("protocol_failure", "Call spec changed within slot")
        positions = [events.index(before), events.index(response), events.index(after)]
        if positions != [positions[0], positions[0] + 1, positions[0] + 2]:
            raise base.RunnerError("protocol_failure", "Call events are not adjacent")
    schedule = _specs(frozen["manifest"])
    if [base.CallSpec(**event["call_spec"]) for event in done] != schedule[:len(done)]:
        raise base.RunnerError("protocol_failure", "Calls are not a frozen-order prefix")
    if len(done) > 6:
        raise base.RunnerError("protocol_failure", "Maximum API budget exceeded")

    aggregates = [event for event in events if event["event_type"] == "AGGREGATE"]
    if len(aggregates) > 1:
        raise base.RunnerError("protocol_failure", "Duplicate aggregate")
    if aggregates:
        values = [event["normalized_result"]["policy_verdict"] for event in done]
        aggregate = aggregates[0]
        if len(values) != 6 or aggregate.get("raw_normalized_verdicts") != values:
            raise base.RunnerError("protocol_failure", "Aggregate inputs mismatch")
        if aggregate.get("aggregate") != _aggregate(values):
            raise base.RunnerError("protocol_failure", "Aggregate result mismatch")
    return done


def _append_prepared(checkpoint: base.Checkpoint, spec: base.CallSpec, index: int, frozen: dict[str, Any]) -> None:
    checkpoint.append(
        "CALL_PREPARED", call_index=index, call_spec=asdict(spec),
        request_role="blind_product_policy_evaluator",
        relevant_hashes={
            "manifest_sha256": frozen["manifest_sha256"],
            "source_case_sha256": frozen["manifest"]["source_case"]["case_sha256"],
            "action_a_sha256": frozen["manifest"]["action_sha256"]["A"],
            "action_b_sha256": frozen["manifest"]["action_sha256"]["B"],
            "rubric_sha256": frozen["manifest"]["frozen_artifact_sha256"][PROMPT_NAME],
        },
    )


def _request(frozen: dict[str, Any], spec: base.CallSpec) -> dict[str, Any]:
    if spec.selected_on_left is None:
        raise base.RunnerError("protocol_failure", "Evaluator orientation missing")
    pair = frozen["pair"]
    left = pair["action_a"] if spec.selected_on_left else pair["action_b"]
    right = pair["action_b"] if spec.selected_on_left else pair["action_a"]
    payload = {
        "frozen_context": frozen["case"]["frozen_input"],
        "left_action": left,
        "right_action": right,
    }
    return {
        "model": frozen["manifest"]["evaluator_model"],
        "messages": [
            {"role": "system", "content": (FREEZE / PROMPT_NAME).read_text(encoding="utf-8")},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "fsb_m_product_policy_verdict", "strict": True,
                "schema": base._read_json(FREEZE / SCHEMA_NAME),
            },
        },
    }


def _run_evaluator(
    client: Any, checkpoint: base.Checkpoint, spec: base.CallSpec,
    index: int, frozen: dict[str, Any],
) -> None:
    request = _request(frozen, spec)
    response = client.chat.completions.create(**request)
    base._append_response(checkpoint, spec, index, response, request)
    try:
        payload = json.loads(response.choices[0].message.content)
        if set(payload) != {"verdict"}:
            raise ValueError("Unexpected evaluator fields")
        verdict = payload["verdict"]
        allowed = set(base._read_json(FREEZE / SCHEMA_NAME)["properties"]["verdict"]["enum"])
        if verdict not in allowed:
            raise ValueError("Unknown evaluator verdict")
    except (AttributeError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise base.RunnerError("structural_failure", "Invalid evaluator response") from error
    checkpoint.append(
        "CALL_COMPLETED", call_index=index, call_spec=asdict(spec),
        normalized_result={
            "blind_verdict": verdict,
            "policy_verdict": _normalize(verdict, bool(spec.selected_on_left)),
        },
    )


def _append_aggregate(checkpoint: base.Checkpoint, events: list[dict[str, Any]]) -> None:
    if any(event["event_type"] == "AGGREGATE" for event in events):
        return
    values = [event["normalized_result"]["policy_verdict"] for event in _completed(events)]
    if len(values) == 6:
        checkpoint.append(
            "AGGREGATE", case_id="FSB-M", raw_normalized_verdicts=values,
            aggregate=_aggregate(values),
        )


def _new_run(frozen: dict[str, Any], root: Optional[Path] = None):
    output_root = RUNS_ROOT if root is None else root
    run_id = f"fsbmpp-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = output_root / run_id
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "head": HEAD,
        "manifest_sha256": frozen["manifest_sha256"],
        "freeze_sha256": frozen["freeze_sha256"],
        "runner_dependency_sha256": frozen["runner_dependency_sha256"],
        "retry": 0, "smoke_calls": 0, "production_calls": 0,
        "maximum_api_calls": 6,
    }
    base._write_metadata(run_dir / "run.json", metadata)
    checkpoint = base.Checkpoint(run_dir / "checkpoint.jsonl")
    checkpoint.append("RUN_STARTED", run_id=run_id, head=HEAD, manifest_sha256=frozen["manifest_sha256"])
    return run_id, run_dir, checkpoint


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


def execute_live(frozen: dict[str, Any], metadata: dict[str, Any], checkpoint: base.Checkpoint) -> None:
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
        _append_aggregate(checkpoint, events)
        events = checkpoint.read()
        done = validate_checkpoint(metadata, events, frozen, for_resume=False)
        if len(done) == 6:
            checkpoint.append("RUN_COMPLETED", api_calls=6, production_calls=0, evaluator_calls=6)
            print(f"RUN_COMPLETED={metadata['run_id']}", flush=True)
            print(f"CHECKPOINT={checkpoint.path}", flush=True)
            return
        spec = schedule[len(done)]
        index = len(done) + 1
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


def _synthetic_complete(
    checkpoint: base.Checkpoint, metadata: dict[str, Any], frozen: dict[str, Any], spec: base.CallSpec,
) -> None:
    done = validate_checkpoint(metadata, checkpoint.read(), frozen, for_resume=False)
    index = len(done) + 1
    _append_prepared(checkpoint, spec, index, frozen)
    checkpoint.append(
        "API_RESPONSE", call_index=index, call_spec=asdict(spec),
        request_sha256=base._sha_value({"dry": spec.key}),
        response_raw={"synthetic": True}, token_usage={"total_tokens": 0},
    )
    checkpoint.append(
        "CALL_COMPLETED", call_index=index, call_spec=asdict(spec),
        normalized_result={
            "blind_verdict": "EQUIVALENT_OR_BOTH_ACCEPTABLE",
            "policy_verdict": "EQUIVALENT_OR_BOTH_ACCEPTABLE",
        },
    )


def dry_run(frozen: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="fsbmpp-dry-") as temporary:
        root = Path(temporary)
        _, run_dir, checkpoint = _new_run(frozen, root)
        metadata = base._read_json(run_dir / "run.json")
        schedule = _specs(frozen["manifest"])
        for spec in schedule:
            _synthetic_complete(checkpoint, metadata, frozen, spec)
        _append_aggregate(checkpoint, checkpoint.read())
        if len(validate_checkpoint(metadata, checkpoint.read(), frozen, for_resume=False)) != 6:
            raise base.RunnerError("protocol_failure", "Dry-run count mismatch")

        _, prefix_dir, prefix = _new_run(frozen, root)
        prefix_meta = base._read_json(prefix_dir / "run.json")
        for spec in schedule[:2]:
            _synthetic_complete(prefix, prefix_meta, frozen, spec)
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

        _, stop_dir, stopped = _new_run(frozen, root)
        stop_meta = base._read_json(stop_dir / "run.json")
        stopped.append("STOP", category="structural_failure", error_message="dry")
        try:
            validate_checkpoint(stop_meta, stopped.read(), frozen, for_resume=True)
        except base.RunnerError:
            stop_rejected = True
        else:
            stop_rejected = False
        if not stop_rejected:
            raise base.RunnerError("protocol_failure", "Stopped run was resumable")

        _, wrong_dir, wrong = _new_run(frozen, root)
        wrong_meta = base._read_json(wrong_dir / "run.json")
        _synthetic_complete(wrong, wrong_meta, frozen, schedule[1])
        try:
            validate_checkpoint(wrong_meta, wrong.read(), frozen, for_resume=True)
        except base.RunnerError:
            order_rejected = True
        else:
            order_rejected = False
        if not order_rejected:
            raise base.RunnerError("protocol_failure", "Out-of-order call was accepted")

        _, hash_dir, hashed = _new_run(frozen, root)
        text = hashed.path.read_text(encoding="utf-8")
        hashed.path.write_text(text.replace("RUN_STARTED", "RUN_TAMPERED", 1), encoding="utf-8")
        try:
            hashed.read()
        except base.RunnerError:
            tamper_rejected = True
        else:
            tamper_rejected = False
        if not tamper_rejected:
            raise base.RunnerError("protocol_failure", "Hash-chain tampering was accepted")

        return {
            "result": "PASS", "api_calls": 0, "head": HEAD,
            "production_calls_simulated": 0, "evaluator_calls_simulated": 6,
            "orientation_balance": "A-left 3 / B-left 3",
            "maximum_call_budget": 6, "checkpoint_hash_chain": "PASS",
            "clean_prefix_resume": "PASS", "dangling_call_resume_rejected": "PASS",
            "terminal_stop_resume_rejected": "PASS", "out_of_order_call_rejected": "PASS",
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
            metadata = base._read_json(run_dir / "run.json")
            print(f"NEW_RUN_ID={run_id}", flush=True)
        else:
            _, metadata, checkpoint = _load_run(args.resume, frozen)
            print(f"RESUMING_RUN_ID={args.resume}", flush=True)
        execute_live(frozen, metadata, checkpoint)
        return 0
    except base.RunnerError as error:
        print(f"STOP_CATEGORY={error.category}", file=sys.stderr, flush=True)
        print(f"STOP_MESSAGE={error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
