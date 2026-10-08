"""One-call replay of the saved Turn-3 reconciliation hard stop."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


SOURCE_RUN_ID = "dialog-multiturn-20261003T034000Z-75734633"
SOURCE_TURN = 3
EXACT_USER = "Примерно одинаково."
EXPECTED_MODEL = "gpt-5-mini"
INTENDED_FILES = (
    "Research/Human_Experience/Weight/13_Active_Conversation_State.txt",
    "ai/cognitive_semantic_wire.py",
    "ai/prompts/cognitive_core_system_prompt.txt",
    "tests/test_cognitive_semantic_wire.py",
    "tests/test_cognitive_core.py",
)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value):
    if isinstance(value, bytes):
        encoded = value
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        encoded = _canonical(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _file_sha(path: Path) -> str:
    return _sha256(path.read_bytes())


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def _source_events(repo: Path):
    path = (
        repo
        / "Conversation_Lab"
        / "traces"
        / "dialogengine_multiturn_runtime_validation"
        / SOURCE_RUN_ID
        / "checkpoint.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    prepared = next(
        row for row in rows
        if row["event"] == "CALL_PREPARED" and row["payload"]["turn"] == SOURCE_TURN
    )
    return path, prepared


def _memory_dump(memory):
    return {
        "human_model": memory.human_model.model_dump(mode="json"),
        "active_conversation_state": memory.active_conversation_state.model_dump(mode="json"),
        "dialogue_history": memory.dialogue_history.model_dump(mode="json"),
    }


def _roundtrip_memory(memory):
    from ai.discovery_data_model import ActiveConversationState, DialogueHistory, HumanModel
    from memory.user_memory import ShadowDiscoveryMemory

    dumped = _memory_dump(memory)
    return ShadowDiscoveryMemory(
        human_model=HumanModel.model_validate(dumped["human_model"]),
        active_conversation_state=ActiveConversationState.model_validate(
            dumped["active_conversation_state"]
        ),
        dialogue_history=DialogueHistory.model_validate(dumped["dialogue_history"]),
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from Conversation_Lab.dialogengine_multiturn_runtime_validation import (
        Checkpoint,
        Instrumentation,
        RunState,
        TurnTrace,
        _assert_source_boundary,
        _usage,
    )
    from ai.cognitive_core import _previous_target_handle_map
    from ai.discovery_data_model import (
        ActiveConversationState,
        DialogueHistory,
        HumanModel,
        validate_discovery_memory,
    )
    from memory.user_memory import (
        ShadowDiscoveryMemory,
        get_shadow_discovery_memory,
        get_user_profile,
        reset_user_profile,
        set_shadow_discovery_memory,
        user_profiles,
    )
    import ai.dialog_engine as dialog_engine

    source_path, prepared = _source_events(repo)
    payload = prepared["payload"]
    if payload["user_text"] != EXACT_USER:
        raise RuntimeError("Saved USER text does not match the authorized replay text.")
    before = payload["before_state"]
    if _sha256(before) != payload["before_state_sha256"]:
        raise RuntimeError("Saved pre-turn state hash does not match its checkpoint binding.")

    frozen_memory = ShadowDiscoveryMemory(
        human_model=HumanModel.model_validate(before["human_model"]),
        active_conversation_state=ActiveConversationState.model_validate(
            before["active_conversation_state"]
        ),
        dialogue_history=DialogueHistory.model_validate(before["dialogue_history"]),
    )
    validate_discovery_memory(
        frozen_memory.human_model,
        frozen_memory.active_conversation_state,
        frozen_memory.dialogue_history,
    )
    handle_map = _previous_target_handle_map(frozen_memory.active_conversation_state)
    if handle_map != {
        "previous_target_1": next(iter(frozen_memory.active_conversation_state.response_targets))
    }:
        raise RuntimeError("Saved target does not have the expected unique semantic handle.")

    run_id = (
        "ownership-turn3-replay-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid4().hex[:8]
    )
    run_dir = repo / "Conversation_Lab" / "traces" / "ownership_turn3_replay" / run_id
    checkpoint = Checkpoint(run_dir)
    commit = _git(repo, "rev-parse", "HEAD").strip()
    patch = _git(repo, "diff", "--binary", "--", *INTENDED_FILES)
    identities = {
        "git_head": commit,
        "intended_patch_sha256": _sha256(patch),
        "files": {name: _file_sha(repo / name) for name in INTENDED_FILES},
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": _file_sha(source_path),
        "source_before_state_sha256": payload["before_state_sha256"],
    }
    preflight = _assert_source_boundary()
    checkpoint.append(
        "RUN_STARTED",
        {
            "run_id": run_id,
            "source_run_id": SOURCE_RUN_ID,
            "source_turn": SOURCE_TURN,
            "exact_user": EXACT_USER,
            "model": EXPECTED_MODEL,
            "maximum_api_calls": 1,
            "identities": identities,
            "preflight": preflight,
        },
    )

    user_id = 900_000_000 + (uuid4().int % 90_000_000)
    while user_id in user_profiles:
        user_id = 900_000_000 + (uuid4().int % 90_000_000)
    reset_user_profile(user_id)
    set_shadow_discovery_memory(user_id, frozen_memory)
    loaded_before = get_shadow_discovery_memory(user_id)
    before_dump = _memory_dump(loaded_before)
    before_hash = _sha256(before_dump)
    if before_hash != payload["before_state_sha256"]:
        raise RuntimeError("Seeded isolated memory differs from the saved pre-turn state.")

    state = RunState(
        checkpoint=checkpoint,
        run_id=run_id,
        commit=commit,
        user_id=user_id,
    )
    trace = TurnTrace(
        turn=SOURCE_TURN,
        user_text=EXACT_USER,
        before_dump=before_dump,
        before_hash=before_hash,
    )
    state.current = trace
    checkpoint.append(
        "CALL_PREPARED",
        {
            "turn": SOURCE_TURN,
            "user_id": user_id,
            "user_text": EXACT_USER,
            "before_state": before_dump,
            "before_state_sha256": before_hash,
            "previous_target_handle_map": handle_map,
        },
    )

    instrumentation = Instrumentation(state)
    try:
        instrumentation.install()
        result = asyncio.run(
            dialog_engine.run_dialog_engine(EXACT_USER, get_user_profile(user_id), user_id)
        )
        loaded_after = get_shadow_discovery_memory(user_id)
        validate_discovery_memory(
            loaded_after.human_model,
            loaded_after.active_conversation_state,
            loaded_after.dialogue_history,
        )
        reloaded = _roundtrip_memory(loaded_after)
        validate_discovery_memory(
            reloaded.human_model,
            reloaded.active_conversation_state,
            reloaded.dialogue_history,
        )
        if (trace.api_calls, trace.propose_calls, trace.adapter_calls) != (1, 1, 1):
            raise RuntimeError("The one-call Cognitive boundary was violated.")
        if (trace.apply_calls, trace.persistence_calls) != (1, 1):
            raise RuntimeError("State application/persistence count was not exactly one.")
        if trace.requested_model != EXPECTED_MODEL:
            raise RuntimeError("Unexpected requested model.")
        if state.api_calls != 1 or state.propose_calls != 1:
            raise RuntimeError("Hard API/propose budget was violated.")
        if not trace.persistence_after_apply:
            raise RuntimeError("Persistence did not occur after StateApplier.")
        if _memory_dump(frozen_memory) != before_dump:
            raise RuntimeError("The frozen input state was mutated.")
        if set(result) != {"reply", "update"} or not result["reply"]:
            raise RuntimeError("DialogEngine returned an invalid reply shape.")
        history = loaded_after.dialogue_history.messages
        if history[-2].text != EXACT_USER or history[-1].text != result["reply"]:
            raise RuntimeError("Persisted USER/SYSTEM pair differs from executed turn.")

        raw = json.loads(trace.raw_structured_content)
        raw_reconciliation = raw.get("reconciliation", [])
        ordinary = [
            row for row in raw_reconciliation
            if row.get("resolution_kind") == "ordinary_addressed"
        ]
        target_id = handle_map["previous_target_1"]
        internal = [
            row for row in trace.turn_result.reconciliation
            if target_id in row.previous_response_target_ids
        ]
        ordinary_ok = (
            len(ordinary) == 1
            and ordinary[0].get("previous_target_handle") == "previous_target_1"
        )
        internal_ok = (
            len(internal) == 1 and internal[0].outcome.value == "answered"
        )
        status = "PASS" if ordinary_ok and internal_ok else "INCONCLUSIVE"
        evidence = {
            "status": status,
            "run_id": run_id,
            "user_id": user_id,
            "exact_user": EXACT_USER,
            "exact_system_reply": result["reply"],
            "raw_semantic_turn": raw,
            "raw_reconciliation": raw_reconciliation,
            "semantic_target_handle": (
                ordinary[0].get("previous_target_handle") if len(ordinary) == 1 else None
            ),
            "persistent_target_id": target_id,
            "cognitive_turn_result": trace.turn_result.model_dump(mode="json"),
            "decision_intent": trace.turn_result.decision_intent.value,
            "internal_reconciliation": [
                row.model_dump(mode="json") for row in trace.turn_result.reconciliation
            ],
            "state_operations": trace.turn_result.state_patch.model_dump(mode="json"),
            "system_action": trace.turn_result.system_action.model_dump(mode="json"),
            "adapter_complete": trace.adapter_calls == 1,
            "state_applier_complete": trace.apply_calls == 1,
            "persistence_complete": trace.persistence_calls == 1,
            "persistence_after_apply": trace.persistence_after_apply,
            "persistence_reload_valid": True,
            "returned_reply_equals_persisted_system": history[-1].text == result["reply"],
            "before_state_sha256": before_hash,
            "after_state_sha256": _sha256(_memory_dump(loaded_after)),
            "frozen_input_unchanged": _memory_dump(frozen_memory) == before_dump,
            "call_counts": {
                "api": state.api_calls,
                "propose": state.propose_calls,
                "adapter": trace.adapter_calls,
                "state_applier": trace.apply_calls,
                "persistence": trace.persistence_calls,
                "retry": 0,
                "fallback": 0,
            },
            "token_usage": {**_usage(trace.response), "requested_model": trace.requested_model},
            "identities": identities,
        }
        checkpoint.append("TURN_COMPLETED", evidence)
        checkpoint.append("RUN_COMPLETED", {"status": status, "api_calls": state.api_calls})
        checkpoint.summary(evidence)
        print(json.dumps({**evidence, "artifact_directory": str(run_dir)}, ensure_ascii=False, indent=2))
        return 0 if status == "PASS" else 2
    except Exception as error:
        loaded_after = get_shadow_discovery_memory(user_id)
        failure = {
            "status": "FAIL",
            "run_id": run_id,
            "user_id": user_id,
            "exact_user": EXACT_USER,
            "failure_boundary": trace.boundary,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "before_state_sha256": before_hash,
            "after_state_sha256": _sha256(_memory_dump(loaded_after)),
            "partial_persistence": _sha256(_memory_dump(loaded_after)) != before_hash,
            "raw_semantic_turn": trace.raw_structured_content,
            "call_counts": {
                "api": state.api_calls,
                "propose": state.propose_calls,
                "adapter": trace.adapter_calls,
                "state_applier": trace.apply_calls,
                "persistence": trace.persistence_calls,
                "retry": 0,
                "fallback": 0,
            },
            "token_usage": (
                None
                if trace.response is None
                else {**_usage(trace.response), "requested_model": trace.requested_model}
            ),
            "identities": identities,
        }
        checkpoint.append("TURN_FAILED", failure)
        checkpoint.append("RUN_COMPLETED", {"status": "FAIL", "api_calls": state.api_calls})
        checkpoint.summary(failure)
        print(json.dumps({**failure, "artifact_directory": str(run_dir)}, ensure_ascii=False, indent=2))
        return 1
    finally:
        instrumentation.restore()
        state.current = None


if __name__ == "__main__":
    raise SystemExit(main())
