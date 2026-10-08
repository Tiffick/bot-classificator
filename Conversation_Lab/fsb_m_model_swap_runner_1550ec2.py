"""Resume-safe runner for the frozen FSB-M production-model swap."""

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


HEAD = "1550ec2"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Conversation_Lab import fsb_mrs_live_runner_1fe07e9 as base  # noqa: E402


FREEZE = ROOT / "Conversation_Lab/frozen/fsb_m_model_swap_1550ec2"
SOURCE = ROOT / "Conversation_Lab/frozen/fsb_01_mrs_policy_patch_1fe07e9"
RUNS_ROOT = ROOT / "Conversation_Lab/traces/fsb_m_model_swap_1550ec2_runs"
MANIFEST_NAME = "fsb_m_model_swap_manifest_1550ec2.json"
VALIDATOR_NAME = "fsb_m_model_swap_offline_validator_1550ec2.py"
SELF = Path(__file__).resolve()
PARENT_RUNNER = ROOT / "Conversation_Lab/fsb_m_prompt_ab_runner_1550ec2.py"
PARENT_RUNNER_SHA256 = "280FBD22705E7918686F08B26C68A6A766A2F474EE1512D9B4CB4F46FDB8BBB8"
FREEZE_SHA256 = {
    "fsb_m_model_swap_freeze_report_1550ec2.md": "AE8012664A9519886D43D5DFC50D83277B433EC56D4CF0CFC878F29922050789",
    "fsb_m_model_swap_manifest_1550ec2.json": "DA36B20078EA5AEFE091CEB584A472AC15EAA97BCBAFCBEB9DCA5938E55CD397",
    "fsb_m_model_swap_offline_validator_1550ec2.py": "92E812FAFC4A9C7808D3F064CA91AB186788127B69E31AC811C5AC3FEA0BEB64",
}


def _git_bytes(spec: str) -> bytes:
    return subprocess.run(
        ["git", "show", spec], cwd=ROOT, check=True, capture_output=True
    ).stdout


def _specs(manifest: dict[str, Any]) -> list[base.CallSpec]:
    return [
        base.CallSpec(
            key=f"production.{sample_id}",
            phase="model_swap",
            case_id=sample_id,
            role="production_decision",
            model=manifest["mini_arm"]["model"],
        )
        for sample_id in manifest["mini_arm"]["sample_order"]
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
    actual_hashes = {}
    for name, expected in FREEZE_SHA256.items():
        actual = base._sha_file(FREEZE / name)
        if actual != expected:
            raise base.RunnerError("hash_failure", f"Freeze SHA mismatch: {name}")
        actual_hashes[name] = actual
    if base._sha_file(PARENT_RUNNER) != PARENT_RUNNER_SHA256:
        raise base.RunnerError("hash_failure", "Checkpoint helper dependency changed")

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

    manifest = base._read_json(FREEZE / MANIFEST_NAME)
    if manifest["head"] != HEAD or manifest["api_budget"] != {
        "production": 3, "evaluator": 0, "maximum_total": 3,
    }:
        raise base.RunnerError("protocol_failure", "Manifest HEAD/budget mismatch")
    params = manifest["production_parameters"]
    if any((params["retry"], params["smoke_calls"], params["confirmations"], params["evaluator_calls"])):
        raise base.RunnerError("protocol_failure", "Extra calls are enabled")
    if params["adaptive_sampling"] or not params["selected_only"]:
        raise base.RunnerError("protocol_failure", "Adaptive/selected-only policy mismatch")
    if manifest["mini_arm"]["model"] != "gpt-5-mini" or len(_specs(manifest)) != 3:
        raise base.RunnerError("protocol_failure", "Mini arm schedule mismatch")

    source_doc = base._read_json(SOURCE / "fsb_mrs_cases_1fe07e9.json")
    source_case = next(item for item in source_doc["cases"] if item["case_id"] == "FSB-M")
    cases = []
    for sample_id in manifest["mini_arm"]["sample_order"]:
        case = json.loads(json.dumps(source_case))
        case["case_id"] = sample_id
        cases.append(case)
    return {
        "head": actual_head,
        "manifest": manifest,
        "manifest_sha256": actual_hashes[MANIFEST_NAME],
        "freeze_sha256": actual_hashes,
        "offline_validation": validation,
        "cases": {"cases": cases},
        "runner_sha256": base._sha_file(SELF),
        "parent_runner_sha256": PARENT_RUNNER_SHA256,
    }


def validate_checkpoint(
    metadata: dict[str, Any], events: list[dict[str, Any]], frozen: dict[str, Any],
    *, for_resume: bool,
) -> list[dict[str, Any]]:
    if metadata.get("head") != HEAD or metadata.get("head") != frozen["head"]:
        raise base.RunnerError("protocol_failure", "Run HEAD mismatch")
    for field in (
        "manifest_sha256", "freeze_sha256", "runner_sha256", "parent_runner_sha256"
    ):
        if metadata.get(field) != frozen[field]:
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
    stops = [event for event in events if event["event_type"] == "STOP"]
    if stops and for_resume:
        raise base.RunnerError("protocol_failure", "Stopped run cannot be resumed")

    prepared = [event for event in events if event["event_type"] == "CALL_PREPARED"]
    responses = [event for event in events if event["event_type"] == "API_RESPONSE"]
    completed = [event for event in events if event["event_type"] == "CALL_COMPLETED"]
    if len(prepared) != len(responses) or len(responses) != len(completed):
        raise base.RunnerError("protocol_failure", "Dangling call could duplicate an API request")
    schedule = _specs(frozen["manifest"])
    for index, (before, response, after) in enumerate(zip(prepared, responses, completed), 1):
        if before["call_index"] != index or response["call_index"] != index or after["call_index"] != index:
            raise base.RunnerError("protocol_failure", "Non-contiguous call index")
        if not (before["call_spec"] == response["call_spec"] == after["call_spec"]):
            raise base.RunnerError("protocol_failure", "Call spec changed within slot")
        positions = [events.index(before), events.index(response), events.index(after)]
        if positions != [positions[0], positions[0] + 1, positions[0] + 2]:
            raise base.RunnerError("protocol_failure", "Call events are not adjacent")
        if response.get("request_sha256") != frozen["manifest"]["identity"]["mini_request_sha256"]:
            raise base.RunnerError("protocol_failure", "Production request presentation drift")
    if [base.CallSpec(**event["call_spec"]) for event in completed] != schedule[:len(completed)]:
        raise base.RunnerError("protocol_failure", "Calls are not a frozen-order prefix")
    if len(completed) > 3:
        raise base.RunnerError("protocol_failure", "Maximum API call budget exceeded")
    if any(event["call_spec"]["role"] != "production_decision" for event in completed):
        raise base.RunnerError("protocol_failure", "Non-production call detected")
    if any(event["event_type"] == "AGGREGATE" for event in events):
        raise base.RunnerError("protocol_failure", "Evaluator aggregate detected")
    return completed


def _append_prepared(
    checkpoint: base.Checkpoint, spec: base.CallSpec, index: int, frozen: dict[str, Any]
) -> None:
    checkpoint.append(
        "CALL_PREPARED", call_index=index, call_spec=asdict(spec),
        request_role="production_decision",
        relevant_hashes={
            "manifest_sha256": frozen["manifest_sha256"],
            "case_sha256": frozen["manifest"]["identity"]["case_sha256"],
            "state_sha256": frozen["manifest"]["identity"]["state_sha256"],
            "prompt_sha256": frozen["manifest"]["prompt"]["sha256"],
            "model_excluded_request_sha256": frozen["manifest"]["identity"]["model_excluded_request_sha256"],
        },
    )


def _structural_class(result: Any) -> str:
    intent = result.decision_intent.value
    kinds = {content.kind.value for content in result.system_action.contents}
    interactions = {target.interaction.value for target in result.system_action.response_targets}
    if intent == "reflection" and "system_reflection" in kinds and "evaluation" in interactions:
        return "REFLECTION_EVALUATION"
    if intent == "mechanism_discovery" and "system_question" in kinds and "open_response" in interactions:
        return "DIRECT_EXPERIENTIAL_DISCRIMINATION_CANDIDATE"
    return "OTHER"


def _run_production(
    client: Any, checkpoint: base.Checkpoint, spec: base.CallSpec, index: int,
    case: dict[str, Any], frozen: dict[str, Any],
) -> None:
    from ai.cognitive_core import CognitiveCore

    prompt_bytes = _git_bytes(frozen["manifest"]["prompt"]["git_path"])
    if hashlib.sha256(prompt_bytes).hexdigest().upper() != frozen["manifest"]["prompt"]["sha256"]:
        raise base.RunnerError("hash_failure", "Prompt changed after preflight")
    prompt_text = prompt_bytes.decode("utf-8")
    FrozenCore = type(
        "FrozenPromptCognitiveCore", (CognitiveCore,),
        {"_system_prompt": staticmethod(lambda: prompt_text)},
    )
    current, human_model, previous_acs, history = base._load_production_input(case)

    def captured(response: Any, request: Any) -> None:
        if base._sha_value(request) != frozen["manifest"]["identity"]["mini_request_sha256"]:
            raise base.RunnerError("protocol_failure", "Actual request presentation drift")
        without_model = dict(request)
        without_model.pop("model", None)
        if base._sha_value(without_model) != frozen["manifest"]["identity"]["model_excluded_request_sha256"]:
            raise base.RunnerError("protocol_failure", "Model-excluded request drift")
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
            "structural_action_class": _structural_class(result),
            "semantic_discrimination_requires_offline_review": True,
            "production_prompt_sha256": frozen["manifest"]["prompt"]["sha256"],
        },
    )


def _new_run(frozen: dict[str, Any], root: Optional[Path] = None):
    output_root = RUNS_ROOT if root is None else root
    run_id = f"fsbms-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    run_dir = output_root / run_id
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "head": HEAD,
        "manifest_sha256": frozen["manifest_sha256"],
        "freeze_sha256": frozen["freeze_sha256"],
        "runner_sha256": frozen["runner_sha256"],
        "parent_runner_sha256": frozen["parent_runner_sha256"],
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


def execute_live(frozen: dict[str, Any], metadata: dict[str, Any], checkpoint: base.Checkpoint) -> None:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(dotenv_path=ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise base.RunnerError("api_failure", "OPENAI_API_KEY is missing")
    client = OpenAI(max_retries=0, timeout=120.0)
    cases = {item["case_id"]: item for item in frozen["cases"]["cases"]}
    schedule = _specs(frozen["manifest"])
    while True:
        events = checkpoint.read()
        completed = validate_checkpoint(metadata, events, frozen, for_resume=False)
        if len(completed) == 3:
            checkpoint.append("RUN_COMPLETED", api_calls=3, production_calls=3, evaluator_calls=0)
            print(f"RUN_COMPLETED={metadata['run_id']}", flush=True)
            print(f"CHECKPOINT={checkpoint.path}", flush=True)
            return
        spec = schedule[len(completed)]
        index = len(completed) + 1
        _append_prepared(checkpoint, spec, index, frozen)
        try:
            _run_production(client, checkpoint, spec, index, cases[spec.case_id], frozen)
        except BaseException as error:
            checkpoint.append(
                "STOP", category=getattr(error, "category", "api_failure"),
                call_index=index, call_spec=asdict(spec),
                error_type=type(error).__name__, error_message=str(error),
            )
            raise


def dry_run(frozen: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="fsbm-model-swap-dry-") as temporary:
        root = Path(temporary)
        _, _, metadata, checkpoint = _new_run(frozen, root)
        specs = _specs(frozen["manifest"])
        for index, spec in enumerate(specs, 1):
            _append_prepared(checkpoint, spec, index, frozen)
            checkpoint.append(
                "API_RESPONSE", call_index=index, call_spec=asdict(spec),
                request_sha256=frozen["manifest"]["identity"]["mini_request_sha256"],
                response_raw={"dry": True}, token_usage={},
            )
            checkpoint.append(
                "CALL_COMPLETED", call_index=index, call_spec=asdict(spec),
                normalized_result={"dry": True},
            )
        completed = validate_checkpoint(metadata, checkpoint.read(), frozen, for_resume=False)
        if len(completed) != 3:
            raise base.RunnerError("protocol_failure", "Dry complete prefix mismatch")

        _, _, prefix_metadata, prefix = _new_run(frozen, root / "prefix")
        spec = specs[0]
        _append_prepared(prefix, spec, 1, frozen)
        prefix.append(
            "API_RESPONSE", call_index=1, call_spec=asdict(spec),
            request_sha256=frozen["manifest"]["identity"]["mini_request_sha256"],
            response_raw={"dry": True}, token_usage={},
        )
        prefix.append(
            "CALL_COMPLETED", call_index=1, call_spec=asdict(spec),
            normalized_result={"dry": True},
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

        _, _, order_metadata, out_of_order = _new_run(frozen, root / "out-of-order")
        wrong = specs[1]
        _append_prepared(out_of_order, wrong, 1, frozen)
        out_of_order.append(
            "API_RESPONSE", call_index=1, call_spec=asdict(wrong),
            request_sha256=frozen["manifest"]["identity"]["mini_request_sha256"],
            response_raw={"dry": True}, token_usage={},
        )
        out_of_order.append(
            "CALL_COMPLETED", call_index=1, call_spec=asdict(wrong),
            normalized_result={"dry": True},
        )
        order_rejected = False
        try:
            validate_checkpoint(order_metadata, out_of_order.read(), frozen, for_resume=True)
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
            "astra_reference": "0/3 direct; 3/3 reflection/evaluation",
            "mini_calls_simulated": 3,
            "evaluator_calls_simulated": 0,
            "maximum_call_budget": 3,
            "checkpoint_hash_chain": "PASS",
            "clean_prefix_resume": "PASS",
            "dangling_resume_rejected": "PASS",
            "terminal_stop_resume_rejected": "PASS",
            "out_of_order_resume_rejected": "PASS",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
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
