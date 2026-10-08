"""Lab-only matched replay of the failed Product Boundary turn.

The runner performs no semantic grading.  It captures three independent
production CognitiveCore selections for later human/read-only review.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_RUN_ID = "dialog-multiturn-20260924T083709Z-1a291f0b"
SOURCE_USER_ID = 386618721
SOURCE_CHECKPOINT_SHA256 = "D9B2CCEB8D3C71298DDF132D7FE31A2DE4865412B40F0FEFBB586E5BA41760FD"
SOURCE_SUMMARY_SHA256 = "F6DD7E935C65CF18D2F95A58A733890A8304035FB8F08134D340F1DC889AFE7A"
SOURCE_PRE_TURN_6_STATE_SHA256 = "3135FC72FC8DDF24ADA51C61D6311430D3CA5972DF9E72EDA35427BFD03760D8"
SOURCE_USER_MESSAGE_ID = "msg_5339a879fdf8412b8eee8eca09fc0a54"
USER_TEXT = (
    "Да, почти каждый раз примерно так. Я уже понял, что после первого срыва "
    "всё бросаю. Что мне конкретно делать, чтобы через пару месяцев опять не "
    "вернуться к прежнему режиму?"
)

EXPECTED_GIT_HEAD = "94390e6c15502e657b459a8fa123aac2f71adb10"
MODEL = "gpt-5-mini"
SAMPLE_COUNT = 3

POLICY_FILES = (
    "Research/Human_Experience/Weight/10_TRANSITION_TO_HUMAN_CONSULTANT.txt",
    "Research/Human_Experience/Weight/11_BOUNDARIES.txt",
    "Research/Human_Experience/Weight/12_Cognitive_Cycle.txt",
    "ai/cognitive_core.py",
    "ai/cognitive_semantic_wire.py",
    "ai/prompts/cognitive_core_system_prompt.txt",
    "tests/test_cognitive_core.py",
    "tests/test_cognitive_semantic_wire.py",
)
EXPECTED_POLICY_FILE_SHA256 = {
    "Research/Human_Experience/Weight/10_TRANSITION_TO_HUMAN_CONSULTANT.txt":
        "B0D3CD449E56E9881ECC338BF25035368BE23A09E9F1BD3B6309D4ECB1DC88B6",
    "Research/Human_Experience/Weight/11_BOUNDARIES.txt":
        "576DE65321D1C0FF194759ED7EEBC2770308996AFE78ECB40740888EB1E1B753",
    "Research/Human_Experience/Weight/12_Cognitive_Cycle.txt":
        "CC6E72A0982F7789810C8A2A82309C7D830E3B1A39E1B4408ACC67B84BE1297D",
    "ai/cognitive_core.py":
        "8B3A67DEA18E7899C942FCBCAE46F21A1FEE640ABAE9815A3A178C03F6E29B54",
    "ai/cognitive_semantic_wire.py":
        "DE563C757856394D98BE4244B971329B24FE39EA81A079BAE6DE68046BAF48C0",
    "ai/prompts/cognitive_core_system_prompt.txt":
        "AF0B033AF8538C3D60DDA29F696739672B2A99DEDA3217FCF7681424C10C8AEA",
    "tests/test_cognitive_core.py":
        "AEF5890192024ED5F7BEF84A27EC48FD84D3889B39446B5EE7CE45EE476A4640",
    "tests/test_cognitive_semantic_wire.py":
        "FE18E00FAE0F5ADB55A9DA90EDA678586564330D686C3D783688321E36C31DA1",
}
EXPECTED_POLICY_DIFF_SHA256 = "7A5A12C76153C6180B35853E0DCF51E86571CC765EC5E7F501657E54DA9692A8"


class ProtocolStop(RuntimeError):
    """Fail-closed stop: no retry or continuation is permitted."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return repr(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Checkpoint:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / "checkpoint.jsonl"
        self.summary_path = run_dir / "run.json"
        self.previous_hash = "0" * 64
        self.index = 0
        run_dir.mkdir(parents=True, exist_ok=False)

    def append(self, event: str, payload: dict[str, Any]) -> None:
        body = {
            "index": self.index,
            "timestamp_utc": _utc_now(),
            "event": event,
            "previous_hash": self.previous_hash,
            "payload": _jsonable(payload),
        }
        record = {**body, "record_hash": _sha256_json(body)}
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.previous_hash = record["record_hash"]
        self.index += 1

    def summary(self, payload: dict[str, Any]) -> None:
        _atomic_json(
            self.summary_path,
            {
                **_jsonable(payload),
                "checkpoint_records": self.index,
                "checkpoint_tip_hash": self.previous_hash,
                "updated_at_utc": _utc_now(),
            },
        )


@dataclass(frozen=True)
class SourceBundle:
    state: dict[str, Any]
    user_message: dict[str, Any]
    checkpoint_path: Path
    summary_path: Path
    checkpoint_tip_hash: str


@dataclass
class LiveRecorder:
    checkpoint: Checkpoint
    current_sample: int = 0
    total_api_calls: int = 0
    sample_api_calls: int = 0
    raw_content: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    request_sha256: str | None = None

    def begin_sample(self, sample: int) -> None:
        self.current_sample = sample
        self.sample_api_calls = 0
        self.raw_content = None
        self.usage = {}
        self.request_sha256 = None


class RecordingCompletions:
    def __init__(self, delegate: Any, recorder: LiveRecorder) -> None:
        self.delegate = delegate
        self.recorder = recorder

    def create(self, **kwargs: Any) -> Any:
        recorder = self.recorder
        if recorder.current_sample not in range(1, SAMPLE_COUNT + 1):
            raise ProtocolStop("API request occurred outside a prepared sample.")
        if recorder.sample_api_calls != 0:
            raise ProtocolStop("More than one API request was attempted in one sample.")
        if recorder.total_api_calls >= SAMPLE_COUNT:
            raise ProtocolStop("The three-call API budget is exhausted.")
        if kwargs.get("model") != MODEL:
            raise ProtocolStop(f"Unexpected model: {kwargs.get('model')!r}")

        recorder.sample_api_calls += 1
        recorder.total_api_calls += 1
        recorder.request_sha256 = _sha256_json(kwargs)
        recorder.checkpoint.append(
            "API_REQUEST",
            {
                "sample": recorder.current_sample,
                "request_index": recorder.total_api_calls,
                "model": kwargs.get("model"),
                "request_sha256": recorder.request_sha256,
            },
        )
        try:
            response = self.delegate.create(**kwargs)
        except Exception as error:
            recorder.checkpoint.append(
                "API_ERROR",
                {
                    "sample": recorder.current_sample,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
            raise

        recorder.raw_content = response.choices[0].message.content
        recorder.usage = _usage(response, kwargs.get("model"))
        recorder.checkpoint.append(
            "API_RESPONSE",
            {
                "sample": recorder.current_sample,
                "raw_structured_content": recorder.raw_content,
                "usage": recorder.usage,
            },
        )
        return response


class RecordingClient:
    def __init__(self, delegate: Any, recorder: LiveRecorder) -> None:
        class Chat:
            def __init__(self, completions: Any) -> None:
                self.completions = completions

        self.chat = Chat(RecordingCompletions(delegate.chat.completions, recorder))


def _usage(response: Any, requested_model: str | None) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "requested_model": requested_model,
        "actual_model_snapshot": getattr(response, "model", None),
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "cached_input_tokens": getattr(details, "cached_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True
    ).stdout


def _policy_identity() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD").decode("ascii").strip()
    file_hashes = {
        relative: _sha256_file(REPO_ROOT / relative) for relative in POLICY_FILES
    }
    diff = _git("diff", "--binary", "--no-ext-diff", "HEAD", "--", *POLICY_FILES)
    return {
        "git_head": head,
        "policy_file_sha256": file_hashes,
        "policy_diff_sha256": hashlib.sha256(diff).hexdigest().upper(),
        "policy_diff_bytes": len(diff),
        "git_status_porcelain": _git("status", "--porcelain=v1").decode(
            "utf-8", errors="replace"
        ).splitlines(),
    }


def _assert_policy_identity() -> dict[str, Any]:
    identity = _policy_identity()
    if identity["git_head"] != EXPECTED_GIT_HEAD:
        raise ProtocolStop("Git HEAD no longer matches the prepared replay.")
    if identity["policy_file_sha256"] != EXPECTED_POLICY_FILE_SHA256:
        raise ProtocolStop("A Product Boundary policy file hash changed after preparation.")
    if identity["policy_diff_sha256"] != EXPECTED_POLICY_DIFF_SHA256:
        raise ProtocolStop("The prepared Product Boundary patch diff changed.")
    return identity


def _validate_checkpoint_chain(path: Path, expected_tip: str) -> int:
    previous = "0" * 64
    rows = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(rows):
        record = json.loads(line)
        if record["index"] != index or record["previous_hash"] != previous:
            raise ProtocolStop(f"Source checkpoint chain breaks at record {index}.")
        body = {key: value for key, value in record.items() if key != "record_hash"}
        if _sha256_json(body) != record["record_hash"]:
            raise ProtocolStop(f"Source checkpoint hash mismatch at record {index}.")
        previous = record["record_hash"]
    if previous != expected_tip:
        raise ProtocolStop("Source checkpoint tip differs from summary.json.")
    return len(rows)


def _load_source_bundle() -> SourceBundle:
    source_dir = (
        Path(tempfile.gettempdir())
        / "dialogengine_multiturn_runtime_validation"
        / SOURCE_RUN_ID
    )
    checkpoint_path = source_dir / "checkpoint.jsonl"
    summary_path = source_dir / "summary.json"
    if not checkpoint_path.is_file() or not summary_path.is_file():
        raise ProtocolStop("The immutable source run artifacts are unavailable.")
    if _sha256_file(checkpoint_path) != SOURCE_CHECKPOINT_SHA256:
        raise ProtocolStop("Source checkpoint file SHA-256 mismatch.")
    if _sha256_file(summary_path) != SOURCE_SUMMARY_SHA256:
        raise ProtocolStop("Source summary file SHA-256 mismatch.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("run_id") != SOURCE_RUN_ID or summary.get("user_id") != SOURCE_USER_ID:
        raise ProtocolStop("Source run identity mismatch.")
    if summary.get("status") != "COMPLETED" or summary.get("completed_turns") != 6:
        raise ProtocolStop("Source run is not the expected completed six-turn run.")
    records = _validate_checkpoint_chain(
        checkpoint_path, summary["checkpoint_tip_hash"]
    )
    if records != summary["checkpoint_records"]:
        raise ProtocolStop("Source checkpoint record count mismatch.")

    turn5 = next(turn for turn in summary["turns"] if turn["turn"] == 5)
    turn6 = next(turn for turn in summary["turns"] if turn["turn"] == 6)
    if turn5["after_state"] != turn6["before_state"]:
        raise ProtocolStop("Turn-5 after-state differs from turn-6 before-state.")
    state = copy.deepcopy(turn6["before_state"])
    if _sha256_json(state) != SOURCE_PRE_TURN_6_STATE_SHA256:
        raise ProtocolStop("Pre-turn-6 state SHA-256 mismatch.")
    if turn6["before_state_sha256"] != SOURCE_PRE_TURN_6_STATE_SHA256:
        raise ProtocolStop("Recorded pre-turn-6 binding mismatch.")

    user_message = copy.deepcopy(turn6["after_state"]["dialogue_history"]["messages"][-2])
    if (
        user_message["id"] != SOURCE_USER_MESSAGE_ID
        or user_message["sequence"] != 11
        or user_message["role"] != "user"
        or user_message["text"] != USER_TEXT
        or turn6["user_text"] != USER_TEXT
    ):
        raise ProtocolStop("Exact source USER turn could not be reconstructed.")

    _validate_models(state, user_message)
    return SourceBundle(
        state=state,
        user_message=user_message,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        checkpoint_tip_hash=summary["checkpoint_tip_hash"],
    )


def _validate_models(state: dict[str, Any], user_message: dict[str, Any]) -> None:
    from ai.discovery_data_model import (
        ActiveConversationState,
        DialogueHistory,
        DialogueMessage,
        DialogueRole,
        HumanModel,
        validate_discovery_memory,
    )

    human_model = HumanModel.model_validate(state["human_model"])
    acs = ActiveConversationState.model_validate(state["active_conversation_state"])
    history = DialogueHistory.model_validate(state["dialogue_history"])
    message = DialogueMessage.model_validate(user_message)
    validate_discovery_memory(human_model, acs, history)
    if message.role != DialogueRole.USER:
        raise ProtocolStop("Reconstructed current message is not USER.")
    expected_sequence = history.messages[-1].sequence + 1
    if message.sequence != expected_sequence:
        raise ProtocolStop("Reconstructed USER sequence is not next in history.")


def _fresh_inputs(bundle: SourceBundle) -> tuple[Any, Any, Any, Any]:
    from ai.discovery_data_model import (
        ActiveConversationState,
        DialogueHistory,
        DialogueMessage,
        HumanModel,
    )

    state = copy.deepcopy(bundle.state)
    message = copy.deepcopy(bundle.user_message)
    return (
        DialogueMessage.model_validate(message),
        HumanModel.model_validate(state["human_model"]),
        ActiveConversationState.model_validate(state["active_conversation_state"]),
        DialogueHistory.model_validate(state["dialogue_history"]),
    )


def _input_dump(message: Any, human_model: Any, acs: Any, history: Any) -> dict[str, Any]:
    return {
        "current_user_message": message.model_dump(mode="json"),
        "human_model": human_model.model_dump(mode="json"),
        "active_conversation_state": acs.model_dump(mode="json"),
        "dialogue_history": history.model_dump(mode="json"),
    }


def _preflight() -> tuple[SourceBundle, dict[str, Any]]:
    from ai.cognitive_semantic_wire import SEMANTIC_TURN_JSON_SCHEMA
    from ai.cognitive_turn import CognitiveTurnResult

    policy = _assert_policy_identity()
    bundle = _load_source_bundle()
    clones = [_fresh_inputs(bundle) for _ in range(SAMPLE_COUNT)]
    clone_hashes = [_sha256_json(_input_dump(*clone)) for clone in clones]
    if len(set(clone_hashes)) != 1:
        raise ProtocolStop("Prepared sample inputs are not semantically identical.")
    object_ids = [tuple(id(part) for part in clone) for clone in clones]
    if len({item for row in object_ids for item in row}) != SAMPLE_COUNT * 4:
        raise ProtocolStop("Prepared samples do not use independent model objects.")
    return bundle, {
        "source_run_id": SOURCE_RUN_ID,
        "source_user_id": SOURCE_USER_ID,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "source_summary_sha256": SOURCE_SUMMARY_SHA256,
        "source_checkpoint_tip_hash": bundle.checkpoint_tip_hash,
        "pre_turn_6_state_sha256": SOURCE_PRE_TURN_6_STATE_SHA256,
        "user_message_sha256": _sha256_json(bundle.user_message),
        "sample_input_sha256": clone_hashes[0],
        "sample_count": SAMPLE_COUNT,
        "model": MODEL,
        "max_retries": 0,
        "evaluator_calls": 0,
        "policy_identity": policy,
        "semantic_turn_schema_sha256": _sha256_json(SEMANTIC_TURN_JSON_SCHEMA),
        "cognitive_turn_result_schema_sha256": _sha256_json(
            CognitiveTurnResult.model_json_schema()
        ),
    }


def _sample_result(sample: int, recorder: LiveRecorder, semantic_turn: Any, result: Any) -> dict[str, Any]:
    return {
        "sample": sample,
        "semantic_turn": semantic_turn.model_dump(mode="json"),
        "cognitive_turn_result": result.model_dump(mode="json"),
        "reply": "".join(segment.text for segment in result.reply_segments),
        "decision_intent": result.decision_intent.value,
        "response_targets": [
            target.model_dump(mode="json") for target in result.system_action.response_targets
        ],
        "reconciliation": [row.model_dump(mode="json") for row in result.reconciliation],
        "token_usage": recorder.usage,
        "request_sha256": recorder.request_sha256,
        "api_calls": recorder.sample_api_calls,
    }


def _run_live() -> int:
    from dotenv import load_dotenv
    from openai import OpenAI
    from ai.cognitive_core import CognitiveCore
    from ai.cognitive_semantic_wire import SemanticTurn

    bundle, preflight = _preflight()
    run_id = f"pb-turn6-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    run_dir = Path(tempfile.gettempdir()) / "product_boundary_turn6_replay" / run_id
    checkpoint = Checkpoint(run_dir)
    checkpoint.append("RUN_STARTED", {"run_id": run_id, "preflight": preflight})
    summary: dict[str, Any] = {
        "run_id": run_id,
        "status": "RUNNING",
        "preflight": preflight,
        "samples": [],
        "api_calls": 0,
        "production_calls": 0,
        "evaluator_calls": 0,
    }
    checkpoint.summary(summary)

    load_dotenv()
    delegate = OpenAI(max_retries=0)
    recorder = LiveRecorder(checkpoint=checkpoint)
    client = RecordingClient(delegate, recorder)
    source_hash_before = _sha256_json(bundle.state)

    try:
        for sample in range(1, SAMPLE_COUNT + 1):
            recorder.begin_sample(sample)
            message, human_model, acs, history = _fresh_inputs(bundle)
            before = _input_dump(message, human_model, acs, history)
            checkpoint.append(
                "SAMPLE_PREPARED",
                {
                    "sample": sample,
                    "sample_input_sha256": _sha256_json(before),
                    "state_sha256": _sha256_json(bundle.state),
                },
            )
            try:
                core = CognitiveCore(client=client, model=MODEL)
                summary["production_calls"] += 1
                result = core.propose(message, human_model, acs, history)
                if recorder.sample_api_calls != 1:
                    raise ProtocolStop("Sample did not make exactly one API request.")
                if not isinstance(recorder.raw_content, str):
                    raise ProtocolStop("No raw structured response was captured.")
                semantic_turn = SemanticTurn.model_validate_json(recorder.raw_content)
                after = _input_dump(message, human_model, acs, history)
                if _sha256_json(after) != _sha256_json(before):
                    raise ProtocolStop("CognitiveCore.propose mutated a replay input.")
                if _sha256_json(bundle.state) != source_hash_before:
                    raise ProtocolStop("The immutable source state was mutated.")
                record = _sample_result(sample, recorder, semantic_turn, result)
                checkpoint.append("SAMPLE_COMPLETED", record)
                summary["samples"].append(record)
                summary["api_calls"] = recorder.total_api_calls
                checkpoint.summary(summary)
            except Exception as error:
                failure = {
                    "sample": sample,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                    "sample_api_calls": recorder.sample_api_calls,
                    "total_api_calls": recorder.total_api_calls,
                    "source_state_unchanged": _sha256_json(bundle.state) == source_hash_before,
                }
                checkpoint.append("SAMPLE_FAILED", failure)
                summary.update(status="HARD_STOP", failure=failure, api_calls=recorder.total_api_calls)
                checkpoint.summary(summary)
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                return 1

        if recorder.total_api_calls != SAMPLE_COUNT:
            raise ProtocolStop("Completed run did not use exactly three API calls.")
        checkpoint.append(
            "RUN_COMPLETED",
            {"samples": SAMPLE_COUNT, "api_calls": recorder.total_api_calls},
        )
        summary.update(status="COMPLETED", api_calls=recorder.total_api_calls)
        checkpoint.summary(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        failure = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "total_api_calls": recorder.total_api_calls,
        }
        checkpoint.append("RUN_FAILED", failure)
        summary.update(status="HARD_STOP", failure=failure, api_calls=recorder.total_api_calls)
        checkpoint.summary(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1


def _dry_run() -> int:
    bundle, preflight = _preflight()
    run_id = f"dry-{uuid4().hex}"
    run_dir = (
        Path(tempfile.gettempdir())
        / "product_boundary_turn6_replay_dry_run"
        / run_id
    )
    checkpoint = Checkpoint(run_dir)
    checkpoint.append(
        "DRY_RUN_COMPLETED",
        {
            "run_id": run_id,
            "preflight": preflight,
            "source_state_unchanged": _sha256_json(bundle.state)
            == SOURCE_PRE_TURN_6_STATE_SHA256,
            "planned_production_calls": SAMPLE_COUNT,
            "planned_api_calls": SAMPLE_COUNT,
            "api_calls": 0,
        },
    )
    result = {
        "run_id": run_id,
        "status": "DRY_RUN_PASS",
        "artifact_directory": str(run_dir),
        "preflight": preflight,
        "planned_api_calls": SAMPLE_COUNT,
        "api_calls": 0,
    }
    checkpoint.summary(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Three-sample Product Boundary replay; no semantic auto-grading."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate only; API calls = 0.")
    mode.add_argument("--new-run", action="store_true", help="Execute exactly three live samples.")
    args = parser.parse_args()
    return _dry_run() if args.dry_run else _run_live()


if __name__ == "__main__":
    raise SystemExit(main())
