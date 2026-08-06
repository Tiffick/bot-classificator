"""Lab-only runner for a continuous Conversation Lab session.

It invokes the active DialogEngine repeatedly in one Python process and records
the Context values observed at every boundary.  It never changes application
modules, production memory code, or Engine behaviour.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# A direct ``python Conversation_Lab/session_runner.py`` run starts from the
# lab directory.  Make the project root importable without changing the app.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ai.dialog_engine as dialog_engine
from memory.user_memory import (
    get_human_model,
    get_user_profile,
    reset_user_profile,
    update_user_profile,
)
from ai.engines.llm_semantic_engine import (
    LLMSemanticEngine,
    compare_semantic_contexts,
)


def _serialise(value: Any) -> Any:
    """Convert Context objects to JSON-safe data without changing them."""
    if is_dataclass(value):
        return _serialise(asdict(value))
    if isinstance(value, dict):
        return {str(key): _serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialise(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


class _SessionTracer:
    def __init__(self, process_id: int, user_id: int) -> None:
        self.process_id = process_id
        self.user_id = user_id
        self.current_turn: dict[str, Any] | None = None

    def record(self, name: str, value: Any) -> None:
        if self.current_turn is None:
            raise RuntimeError("A Context was recorded outside a Conversation Lab turn.")
        self.current_turn[name] = _serialise(deepcopy(value))


def _consultation_state(memory: dict[str, Any]) -> dict[str, Any]:
    """Expose the flattened state used by the current DialogEngine contract."""
    return {
        "last_question": memory.get("last_question"),
        "discovery_complete": memory.get("discovery_complete", False),
    }


@contextmanager
def _instrument_dialog_engine(tracer: _SessionTracer, *, live_response: bool):
    """Temporarily observe only this process's DialogEngine calls.

    The substitutions are restored in ``finally``.  They exist solely in the
    lab runner's Python process and therefore cannot affect the running bot.
    """
    original = {
        name: getattr(dialog_engine, name)
        for name in (
            "SemanticEngine",
            "HumanModelEngine",
            "ReasoningEngine",
            "DecisionEngine",
            "ImpactEngine",
            "EmotionalEngine",
            "ResponseEngine",
        )
    }

    class TracingSemanticEngine(original["SemanticEngine"]):
        def analyze(self, user_text: str):
            result = super().analyze(user_text)
            tracer.record("semantic_context", result)
            return result

    class TracingHumanModelEngine(original["HumanModelEngine"]):
        def build(self, semantic_context, previous_human_model, memory):
            tracer.record("previous_human_model_input", previous_human_model)
            tracer.record("memory_input", memory)
            tracer.record("consultation_state_input", _consultation_state(memory))
            result = super().build(semantic_context, previous_human_model, memory)
            tracer.record("human_model", result)
            return result

    class TracingReasoningEngine(original["ReasoningEngine"]):
        def reason(self, *args, **kwargs):
            result = super().reason(*args, **kwargs)
            tracer.record("reasoning_context", result)
            return result

    class TracingDecisionEngine(original["DecisionEngine"]):
        def decide(self, *args, **kwargs):
            result = super().decide(*args, **kwargs)
            tracer.record("decision_context", result)
            return result

    class TracingImpactEngine(original["ImpactEngine"]):
        def evaluate(self, *args, **kwargs):
            result = super().evaluate(*args, **kwargs)
            tracer.record("impact_context", result)
            return result

    class TracingEmotionalEngine(original["EmotionalEngine"]):
        def choose(self, *args, **kwargs):
            result = super().choose(*args, **kwargs)
            tracer.record("emotional_context", result)
            return result

    if live_response:

        class TracingResponseEngine(original["ResponseEngine"]):
            def generate(self, *args, **kwargs):
                result = super().generate(*args, **kwargs)
                tracer.record("response", result)
                return result

    else:

        class TracingResponseEngine:
            """A Lab diagnostic stub; official experiments must use live mode."""

            def __init__(self, *args, **kwargs) -> None:
                pass

            def generate(self, *args, **kwargs) -> str:
                result = "[Conversation Lab technical response stub]"
                tracer.record("response", result)
                return result

    replacements = {
        "SemanticEngine": TracingSemanticEngine,
        "HumanModelEngine": TracingHumanModelEngine,
        "ReasoningEngine": TracingReasoningEngine,
        "DecisionEngine": TracingDecisionEngine,
        "ImpactEngine": TracingImpactEngine,
        "EmotionalEngine": TracingEmotionalEngine,
        "ResponseEngine": TracingResponseEngine,
    }
    try:
        for name, replacement in replacements.items():
            setattr(dialog_engine, name, replacement)
        yield
    finally:
        for name, engine_class in original.items():
            setattr(dialog_engine, name, engine_class)


def _continuity_check(turns: list[dict[str, Any]]) -> dict[str, bool]:
    if len(turns) < 2:
        return {
            "same_process": False,
            "same_user_id": False,
            "history_preserved": False,
            "previous_human_model_preserved": False,
            "known_facts_preserved": False,
            "consultation_state_preserved": False,
        }

    first, second = turns[0], turns[1]
    first_facts = first.get("human_model", {}).get("facts", {})
    second_facts = second.get("human_model", {}).get("facts", {})
    known_facts_preserved = all(
        second_facts.get(key) == value
        for key, value in first_facts.items()
        if value is not None
    )
    return {
        "same_process": first["process_id"] == second["process_id"],
        "same_user_id": first["user_id"] == second["user_id"],
        "history_preserved": len(second.get("memory_input", {}).get("history", [])) >= 2,
        "previous_human_model_preserved": (
            second.get("previous_human_model_input") == first.get("human_model")
        ),
        "known_facts_preserved": known_facts_preserved,
        "consultation_state_preserved": (
            second.get("consultation_state_input")
            == first.get("consultation_state_after_turn")
        ),
    }


async def run_session(
    messages: Iterable[str],
    *,
    user_id: int,
    live_response: bool = True,
    reset_memory: bool = True,
    shadow_semantic: bool = False,
    llm_semantic_engine=None,
) -> dict[str, Any]:
    """Run all supplied messages through one active DialogEngine process."""
    messages = list(messages)
    if not messages:
        raise ValueError("Conversation Lab requires at least one user message.")
    if reset_memory:
        reset_user_profile(user_id)

    process_id = os.getpid()
    tracer = _SessionTracer(process_id, user_id)
    turns: list[dict[str, Any]] = []
    shadow_engine = (
        llm_semantic_engine or LLMSemanticEngine() if shadow_semantic else None
    )

    with _instrument_dialog_engine(tracer, live_response=live_response):
        for number, user_text in enumerate(messages, start=1):
            turn: dict[str, Any] = {
                "turn": number,
                "process_id": process_id,
                "user_id": user_id,
                "user_text": user_text,
            }
            tracer.current_turn = turn
            if shadow_engine is not None:
                llm_context = shadow_engine.analyze(user_text)
                turn["llm_semantic_context"] = _serialise(llm_context)
            profile = get_user_profile(user_id)
            result = await dialog_engine.run_dialog_engine(user_text, profile, user_id)
            if shadow_engine is not None:
                turn["semantic_comparison"] = compare_semantic_contexts(
                    turn["semantic_context"],
                    llm_context,
                )
            update_user_profile(user_id, result["update"])
            turn["update"] = _serialise(result["update"])
            turn["memory_after_turn"] = _serialise(get_user_profile(user_id))
            turn["consultation_state_after_turn"] = _consultation_state(
                turn["memory_after_turn"]
            )
            turn["human_model_after_turn"] = _serialise(get_human_model(user_id))
            turns.append(turn)
            tracer.current_turn = None

    return {
        "runner": "Conversation_Lab.session_runner",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "process_id": process_id,
        "user_id": user_id,
        "live_response": live_response,
        "shadow_semantic": shadow_semantic,
        "turns": turns,
        "continuity_check": _continuity_check(turns),
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one continuous Conversation Lab session.")
    parser.add_argument("messages", nargs="+", help="User messages, in chronological order.")
    parser.add_argument("--user-id", type=int, required=True, help="Dedicated test user ID.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use only a Lab response stub. Never use this mode for an official experiment.",
    )
    parser.add_argument(
        "--shadow-semantic",
        action="store_true",
        help="Run LLM semantic extraction for Lab comparison only.",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        help="Optional destination for the JSON trace.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    trace = asyncio.run(
        run_session(
            args.messages,
            user_id=args.user_id,
            live_response=not args.dry_run,
            shadow_semantic=args.shadow_semantic,
        )
    )
    trace_file = args.trace_file or (
        Path(__file__).parent
        / "traces"
        / f"session_{trace['process_id']}_{uuid.uuid4().hex[:8]}.json"
    )
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"trace_file": str(trace_file), "continuity_check": trace["continuity_check"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
