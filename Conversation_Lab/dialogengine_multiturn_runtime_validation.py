"""Interactive live validation of the production DialogEngine across turns.

The runner never generates a USER turn.  It keeps one production process alive,
durably records every boundary outside the repository, and waits for a human at
each gate.  ``--dry-run`` performs no OpenAI request.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


MAX_TURNS = 6
EXPECTED_MODEL = "gpt-5-mini"
LEGACY_ENGINE_NAMES = (
    "ShadowPerceptionEngine",
    "ShadowIntegrationEngine",
    "LLMSemanticEngine",
    "SemanticEngine",
    "HumanModelEngine",
    "ReasoningEngine",
    "DecisionEngine",
    "ImpactEngine",
    "EmotionalEngine",
    "ResponseEngine",
)
INITIAL_SCENARIO = (
    "Последние пару лет набрал вес, и больше всего меня раздражает живот."
)


class ProtocolStop(RuntimeError):
    """The live protocol cannot safely continue."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest().upper()


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


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Checkpoint:
    """Append-only, hash-chained and fsync'ed run evidence."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / "checkpoint.jsonl"
        self.summary_path = run_dir / "summary.json"
        self.previous_hash = "0" * 64
        self.index = 0
        run_dir.mkdir(parents=True, exist_ok=False)

    def append(self, event: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = {
            "index": self.index,
            "timestamp_utc": _utc_now(),
            "event": event,
            "previous_hash": self.previous_hash,
            "payload": _jsonable(payload),
        }
        record = {**body, "record_hash": _sha256(body)}
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.previous_hash = record["record_hash"]
        self.index += 1
        return record

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


def _memory_dump(memory: Any) -> Optional[dict[str, Any]]:
    if memory is None:
        return None
    return {
        "human_model": memory.human_model.model_dump(mode="json"),
        "active_conversation_state": (
            None
            if memory.active_conversation_state is None
            else memory.active_conversation_state.model_dump(mode="json")
        ),
        "dialogue_history": memory.dialogue_history.model_dump(mode="json"),
    }


def _target_view(memory: Any) -> list[dict[str, Any]]:
    if memory is None or memory.active_conversation_state is None:
        return []
    acs = memory.active_conversation_state
    return [target.model_dump(mode="json") for target in acs.response_targets.values()]


def _usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "requested_model": None,
        "actual_model_snapshot": getattr(response, "model", None),
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "cached_input_tokens": getattr(prompt_details, "cached_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


@dataclass
class TurnTrace:
    turn: int
    user_text: str
    before_dump: Optional[dict[str, Any]]
    before_hash: str
    api_calls: int = 0
    propose_calls: int = 0
    adapter_calls: int = 0
    apply_calls: int = 0
    persistence_calls: int = 0
    persistence_after_apply: Optional[bool] = None
    requested_model: Optional[str] = None
    response: Any = None
    raw_structured_content: Optional[str] = None
    turn_result: Any = None
    applied_turn: Any = None
    boundary: str = "prepared"


@dataclass
class RunState:
    checkpoint: Checkpoint
    run_id: str
    commit: str
    user_id: int
    api_calls: int = 0
    propose_calls: int = 0
    completed_turns: int = 0
    stopped: bool = False
    current: Optional[TurnTrace] = None
    turns: list[dict[str, Any]] = field(default_factory=list)

    def write_summary(self, status: str, **extra: Any) -> None:
        self.checkpoint.summary(
            {
                "status": status,
                "run_id": self.run_id,
                "git_commit": self.commit,
                "user_id": self.user_id,
                "model": EXPECTED_MODEL,
                "max_turns": MAX_TURNS,
                "completed_turns": self.completed_turns,
                "api_calls": self.api_calls,
                "cognitive_core_propose_calls": self.propose_calls,
                "turns": self.turns,
                **extra,
            }
        )


class Instrumentation:
    """Transparent observation of the production boundary; no semantic fallback."""

    def __init__(self, state: RunState) -> None:
        self.state = state
        self.originals: dict[str, Any] = {}

    def install(self) -> None:
        from openai import OpenAI as RealOpenAI
        import ai.cognitive_semantic_wire as semantic_wire
        import ai.dialog_engine as dialog_engine
        from ai.cognitive_core import CognitiveCore
        from memory.user_memory import set_shadow_discovery_memory

        present = [name for name in LEGACY_ENGINE_NAMES if hasattr(dialog_engine, name)]
        if present:
            raise ProtocolStop(f"Legacy Engines are connected to DialogEngine: {present}")

        self.originals = {
            "dialog_engine.OpenAI": dialog_engine.OpenAI,
            "CognitiveCore.propose": CognitiveCore.propose,
            "semantic_wire.adapt_semantic_turn": semantic_wire.adapt_semantic_turn,
            "dialog_engine.apply_cognitive_turn": dialog_engine.apply_cognitive_turn,
            "dialog_engine.set_shadow_discovery_memory": (
                dialog_engine.set_shadow_discovery_memory
            ),
        }
        state = self.state

        class RecordingCompletions:
            def __init__(self, delegate: Any) -> None:
                self.delegate = delegate

            def create(self, **kwargs: Any) -> Any:
                trace = _current_trace(state)
                if trace.api_calls != 0:
                    raise ProtocolStop("A second API request was attempted in one turn.")
                if state.api_calls >= MAX_TURNS:
                    raise ProtocolStop("Run API budget is exhausted.")
                trace.api_calls += 1
                state.api_calls += 1
                trace.requested_model = kwargs.get("model")
                trace.boundary = "api_request"
                state.checkpoint.append(
                    "API_REQUEST",
                    {
                        "turn": trace.turn,
                        "request_index": state.api_calls,
                        "requested_model": trace.requested_model,
                    },
                )
                try:
                    response = self.delegate.create(**kwargs)
                except Exception as error:
                    state.checkpoint.append(
                        "API_ERROR",
                        {
                            "turn": trace.turn,
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        },
                    )
                    raise
                trace.response = response
                try:
                    trace.raw_structured_content = response.choices[0].message.content
                except (AttributeError, IndexError, TypeError):
                    trace.raw_structured_content = None
                trace.boundary = "api_response"
                usage = _usage(response)
                usage["requested_model"] = trace.requested_model
                state.checkpoint.append(
                    "API_RESPONSE",
                    {
                        "turn": trace.turn,
                        "usage": usage,
                        "raw_structured_content": trace.raw_structured_content,
                    },
                )
                return response

        class RecordingChat:
            def __init__(self, delegate: Any) -> None:
                self.completions = RecordingCompletions(delegate.completions)

        class RecordingClient:
            def __init__(self, delegate: Any) -> None:
                self.chat = RecordingChat(delegate.chat)

        def recording_openai(**kwargs: Any) -> Any:
            if kwargs.get("max_retries") != 0:
                raise ProtocolStop("Production client must set max_retries=0.")
            return RecordingClient(RealOpenAI(**kwargs))

        original_propose = CognitiveCore.propose

        def tracked_propose(core: Any, *args: Any, **kwargs: Any) -> Any:
            trace = _current_trace(state)
            if trace.propose_calls != 0:
                raise ProtocolStop("A second CognitiveCore.propose() was attempted.")
            trace.propose_calls += 1
            state.propose_calls += 1
            trace.boundary = "cognitive_core_propose"
            result = original_propose(core, *args, **kwargs)
            trace.turn_result = result
            trace.boundary = "cognitive_turn_result"
            return result

        original_adapt = semantic_wire.adapt_semantic_turn

        def tracked_adapt(*args: Any, **kwargs: Any) -> Any:
            trace = _current_trace(state)
            trace.adapter_calls += 1
            if trace.adapter_calls != 1:
                raise ProtocolStop("Semantic adapter ran more than once.")
            trace.boundary = "semantic_adapter"
            result = original_adapt(*args, **kwargs)
            trace.boundary = "semantic_adapter_complete"
            return result

        original_apply = dialog_engine.apply_cognitive_turn

        def tracked_apply(*args: Any, **kwargs: Any) -> Any:
            trace = _current_trace(state)
            trace.apply_calls += 1
            if trace.apply_calls != 1:
                raise ProtocolStop("StateApplier ran more than once.")
            trace.boundary = "state_applier"
            applied = original_apply(*args, **kwargs)
            trace.applied_turn = applied
            trace.boundary = "state_applier_complete"
            state.checkpoint.append(
                "STATE_APPLIED",
                {
                    "turn": trace.turn,
                    "mappings": {
                        "items": applied.item_id_mapping,
                        "relations": applied.relation_id_mapping,
                        "active_content": applied.active_content_id_mapping,
                        "response_targets": applied.response_target_id_mapping,
                    },
                },
            )
            return applied

        def tracked_persist(user_id: int, memory: Any) -> None:
            trace = _current_trace(state)
            trace.persistence_calls += 1
            trace.persistence_after_apply = trace.applied_turn is not None
            if trace.persistence_calls != 1:
                raise ProtocolStop("Discovery memory was persisted more than once.")
            if not trace.persistence_after_apply:
                raise ProtocolStop("Persistence was attempted before StateApplier success.")
            trace.boundary = "persistence"
            set_shadow_discovery_memory(user_id, memory)
            trace.boundary = "persistence_complete"

        dialog_engine.OpenAI = recording_openai
        CognitiveCore.propose = tracked_propose
        semantic_wire.adapt_semantic_turn = tracked_adapt
        dialog_engine.apply_cognitive_turn = tracked_apply
        dialog_engine.set_shadow_discovery_memory = tracked_persist

    def restore(self) -> None:
        if not self.originals:
            return
        import ai.cognitive_semantic_wire as semantic_wire
        import ai.dialog_engine as dialog_engine
        from ai.cognitive_core import CognitiveCore

        dialog_engine.OpenAI = self.originals["dialog_engine.OpenAI"]
        CognitiveCore.propose = self.originals["CognitiveCore.propose"]
        semantic_wire.adapt_semantic_turn = self.originals[
            "semantic_wire.adapt_semantic_turn"
        ]
        dialog_engine.apply_cognitive_turn = self.originals[
            "dialog_engine.apply_cognitive_turn"
        ]
        dialog_engine.set_shadow_discovery_memory = self.originals[
            "dialog_engine.set_shadow_discovery_memory"
        ]


def _current_trace(state: RunState) -> TurnTrace:
    if state.current is None:
        raise ProtocolStop("Instrumentation observed a call outside a prepared turn.")
    return state.current


def _assert_source_boundary() -> dict[str, Any]:
    import ai.dialog_engine as dialog_engine

    source = inspect.getsource(dialog_engine)
    present = [name for name in LEGACY_ENGINE_NAMES if hasattr(dialog_engine, name)]
    if present:
        raise ProtocolStop(f"Legacy Engine symbols are active: {present}")
    if "OpenAI(max_retries=0)" not in source:
        raise ProtocolStop("DialogEngine does not visibly enforce max_retries=0.")
    if source.count("core.propose") != 1:
        raise ProtocolStop("DialogEngine source does not contain exactly one propose call site.")
    return {
        "legacy_symbols_present": present,
        "max_retries_zero": True,
        "propose_call_sites": 1,
    }


def _turn_record(
    trace: TurnTrace,
    result: dict[str, Any],
    after_memory: Any,
) -> dict[str, Any]:
    after_dump = _memory_dump(after_memory)
    usage = _usage(trace.response)
    usage["requested_model"] = trace.requested_model
    acs = after_memory.active_conversation_state
    history = after_memory.dialogue_history.messages
    mappings = trace.applied_turn
    return {
        "turn": trace.turn,
        "user_text": trace.user_text,
        "returned_reply": result["reply"],
        "return_shape": sorted(result),
        "compatibility_update": result["update"],
        "decision_intent": trace.turn_result.decision_intent.value,
        "cognitive_turn_result": trace.turn_result.model_dump(mode="json"),
        "reconciliation": [
            resolution.model_dump(mode="json")
            for resolution in trace.turn_result.reconciliation
        ],
        "before_state": trace.before_dump,
        "before_state_sha256": trace.before_hash,
        "after_state": after_dump,
        "after_state_sha256": _sha256(after_dump),
        "applied_turn_mappings": {
            "items": mappings.item_id_mapping,
            "relations": mappings.relation_id_mapping,
            "active_content": mappings.active_content_id_mapping,
            "response_targets": mappings.response_target_id_mapping,
        },
        "persisted_system_text": history[-1].text,
        "system_reply_matches_returned_reply": history[-1].text == result["reply"],
        "human_model_item_count": len(after_memory.human_model.items),
        "human_model_relation_count": len(after_memory.human_model.relations),
        "acs_is_none": acs is None,
        "acs_content_count": None if acs is None else len(acs.active_content),
        "acs_target_count": None if acs is None else len(acs.response_targets),
        "response_targets": _target_view(after_memory),
        "dialogue_history_message_count": len(history),
        "call_counts": {
            "api": trace.api_calls,
            "propose": trace.propose_calls,
            "adapter": trace.adapter_calls,
            "state_applier": trace.apply_calls,
            "persistence": trace.persistence_calls,
        },
        "persistence_after_apply": trace.persistence_after_apply,
        "token_usage": usage,
    }


def _validate_success(
    state: RunState,
    trace: TurnTrace,
    result: dict[str, Any],
    before_memory: Any,
    after_memory: Any,
) -> None:
    from ai.discovery_data_model import DialogueRole, validate_discovery_memory

    if set(result) != {"reply", "update"} or not result["reply"]:
        raise ProtocolStop("DialogEngine return contract is invalid.")
    if (trace.api_calls, trace.propose_calls, trace.adapter_calls) != (1, 1, 1):
        raise ProtocolStop("One-call Cognitive boundary was violated.")
    if (trace.apply_calls, trace.persistence_calls) != (1, 1):
        raise ProtocolStop("State application/persistence count is invalid.")
    if not trace.persistence_after_apply:
        raise ProtocolStop("Persistence did not follow StateApplier.")
    if trace.requested_model != EXPECTED_MODEL:
        raise ProtocolStop(f"Unexpected requested model: {trace.requested_model!r}")
    validate_discovery_memory(
        after_memory.human_model,
        after_memory.active_conversation_state,
        after_memory.dialogue_history,
    )
    before_count = 0 if before_memory is None else len(before_memory.dialogue_history.messages)
    history = after_memory.dialogue_history.messages
    if len(history) != before_count + 2:
        raise ProtocolStop("DialogueHistory did not grow by exactly one message pair.")
    if [history[-2].role, history[-1].role] != [DialogueRole.USER, DialogueRole.SYSTEM]:
        raise ProtocolStop("Persisted turn is not a USER/SYSTEM pair.")
    if history[-2].text != trace.user_text:
        raise ProtocolStop("Persisted USER text differs from submitted text.")
    if history[-1].text != result["reply"]:
        raise ProtocolStop("Persisted SYSTEM text differs from returned reply.")
    if state.api_calls != state.propose_calls:
        raise ProtocolStop("Cumulative API/propose counts diverged.")


def _failure_payload(
    state: RunState,
    trace: TurnTrace,
    error: Exception,
    after_memory: Any,
) -> dict[str, Any]:
    after_dump = _memory_dump(after_memory)
    chain = []
    current: Optional[BaseException] = error
    while current is not None:
        chain.append({"type": type(current).__name__, "message": str(current)})
        current = current.__cause__
    return {
        "turn": trace.turn,
        "user_text": trace.user_text,
        "failure_boundary": trace.boundary,
        "error_chain": chain,
        "traceback": traceback.format_exc(),
        "call_counts": {
            "api": trace.api_calls,
            "propose": trace.propose_calls,
            "adapter": trace.adapter_calls,
            "state_applier": trace.apply_calls,
            "persistence": trace.persistence_calls,
        },
        "before_state_sha256": trace.before_hash,
        "after_state_sha256": _sha256(after_dump),
        "partial_persistence_detected": _sha256(after_dump) != trace.before_hash,
        "cumulative_api_calls": state.api_calls,
        "cumulative_propose_calls": state.propose_calls,
    }


def _print_gate(turn_record: Optional[dict[str, Any]], next_turn: int) -> None:
    print("\n" + "=" * 72)
    print(f"HUMAN GATE — proposed USER turn {next_turn}/{MAX_TURNS}")
    if turn_record is None:
        print("Suggested first scenario:")
        print(INITIAL_SCENARIO)
    else:
        print("Actual assistant reply:")
        print(turn_record["returned_reply"])
        print("DecisionIntent:", turn_record["decision_intent"])
        print("Persisted ResponseTargets:")
        print(json.dumps(turn_record["response_targets"], ensure_ascii=False, indent=2))
        print("Inspect the reply and targets before choosing the next USER turn.")
    print("Enter exact USER text, or :stop. No message is sent automatically.")
    print("=" * 72)


async def _execute_turn(state: RunState, user_text: str) -> dict[str, Any]:
    import ai.dialog_engine as dialog_engine
    from memory.user_memory import (
        get_user_memory,
        get_user_profile,
        set_last_question,
        update_user_profile,
    )

    turn = state.completed_turns + 1
    before_memory = get_user_memory(state.user_id)["shadow_discovery_memory"]
    before_dump = _memory_dump(before_memory)
    trace = TurnTrace(
        turn=turn,
        user_text=user_text,
        before_dump=before_dump,
        before_hash=_sha256(before_dump),
    )
    state.current = trace
    state.checkpoint.append(
        "CALL_PREPARED",
        {
            "turn": turn,
            "user_text": user_text,
            "before_state": before_dump,
            "before_state_sha256": trace.before_hash,
            "previous_response_targets": _target_view(before_memory),
        },
    )
    try:
        profile = get_user_profile(state.user_id)
        result = await dialog_engine.run_dialog_engine(user_text, profile, state.user_id)
        after_memory = get_user_memory(state.user_id)["shadow_discovery_memory"]
        _validate_success(state, trace, result, before_memory, after_memory)
        update_user_profile(state.user_id, result["update"])
        set_last_question(state.user_id, result["reply"])
        record = _turn_record(trace, result, after_memory)
        state.checkpoint.append("TURN_COMPLETED", record)
        state.turns.append(record)
        state.completed_turns += 1
        state.write_summary("WAITING_FOR_HUMAN")
        return record
    except Exception as error:
        after_memory = get_user_memory(state.user_id)["shadow_discovery_memory"]
        failure = _failure_payload(state, trace, error, after_memory)
        state.checkpoint.append("TURN_FAILED", failure)
        state.stopped = True
        state.write_summary("HARD_STOP", failure=failure)
        raise
    finally:
        state.current = None


def _new_user_id(existing: dict[int, Any]) -> int:
    while True:
        candidate = 100_000_000 + (uuid4().int % 800_000_000)
        if candidate not in existing:
            return candidate


def _run_live(repo_root: Path) -> int:
    from memory.user_memory import reset_user_profile, user_profiles

    commit = _git_head(repo_root)
    run_id = f"dialog-multiturn-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    root = Path(tempfile.gettempdir()) / "dialogengine_multiturn_runtime_validation"
    checkpoint = Checkpoint(root / run_id)
    user_id = _new_user_id(user_profiles)
    state = RunState(
        checkpoint=checkpoint,
        run_id=run_id,
        commit=commit,
        user_id=user_id,
    )
    preflight = _assert_source_boundary()
    checkpoint.append(
        "RUN_STARTED",
        {
            "run_id": run_id,
            "git_commit": commit,
            "started_at_utc": _utc_now(),
            "model": EXPECTED_MODEL,
            "max_turns": MAX_TURNS,
            "user_id": user_id,
            "artifact_directory": str(checkpoint.run_dir),
            "preflight": preflight,
        },
    )
    state.write_summary("READY_FOR_HUMAN")
    reset_user_profile(user_id)
    instrumentation = Instrumentation(state)
    try:
        instrumentation.install()
        last_record = None
        while state.completed_turns < MAX_TURNS:
            next_turn = state.completed_turns + 1

            checkpoint.append(
                "HUMAN_GATE",
                {
                    "next_turn": next_turn,
                    "previous_turn": last_record,
                    "previous_target_required": False,
                },
            )
            state.write_summary("WAITING_FOR_HUMAN")
            _print_gate(last_record, next_turn)
            try:
                user_text = input("USER> ")
            except EOFError:
                user_text = ":stop"
            if user_text.strip() == ":stop":
                checkpoint.append(
                    "RUN_COMPLETED",
                    {
                        "reason": "human_stop",
                        "completed_turns": state.completed_turns,
                    },
                )
                state.write_summary("COMPLETED")
                return 0
            if not user_text.strip():
                print("Empty input was not sent. Enter USER text or :stop.")
                continue
            last_record = asyncio.run(_execute_turn(state, user_text))
        checkpoint.append(
            "RUN_COMPLETED",
            {"reason": "maximum_turns_reached", "completed_turns": state.completed_turns},
        )
        state.write_summary("COMPLETED")
        return 0
    except Exception as error:
        print(f"HARD STOP: {type(error).__name__}: {error}", file=sys.stderr)
        print(f"Artifacts: {checkpoint.run_dir}", file=sys.stderr)
        return 1
    finally:
        instrumentation.restore()


def _dry_run(repo_root: Path) -> int:
    commit = _git_head(repo_root)
    boundary = _assert_source_boundary()
    root = Path(tempfile.gettempdir()) / "dialogengine_multiturn_runtime_validation_dry_run"
    run_dir = root / f"dry-{uuid4().hex}"
    checkpoint = Checkpoint(run_dir)
    checkpoint.append(
        "DRY_RUN",
        {
            "git_commit": commit,
            "model": EXPECTED_MODEL,
            "max_turns": MAX_TURNS,
            "api_calls": 0,
            "boundary": boundary,
            "scenario": INITIAL_SCENARIO,
        },
    )
    checkpoint.summary(
        {
            "status": "DRY_RUN_PASS",
            "git_commit": commit,
            "api_calls": 0,
            "artifact_directory": str(run_dir),
        }
    )
    print(
        json.dumps(
            {
                "status": "DRY_RUN_PASS",
                "git_commit": commit,
                "api_calls": 0,
                "artifact_directory": str(run_dir),
                "checkpoint": str(checkpoint.path),
                "summary": str(checkpoint.summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--new-run", action="store_true", help="Start an interactive live run.")
    mode.add_argument("--dry-run", action="store_true", help="Validate mechanics with zero API calls.")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if args.dry_run:
        return _dry_run(repo_root)
    return _run_live(repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
