"""Laboratory-only controlled A/B runner for Semantic Engine experiments."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ai.dialog_engine as dialog_engine
import ai.engines.response_engine as response_engine_module
from Conversation_Lab.session_runner import (
    _SessionTracer,
    _consultation_state,
    _instrument_dialog_engine,
    _serialise,
)
from ai.engines.llm_semantic_engine import LLMSemanticEngine
from ai.engines.semantic_engine import SemanticEngine
from memory.user_memory import get_human_model, get_user_profile, reset_user_profile, update_user_profile
from openai import OpenAI


class _ApiAudit:
    def __init__(self) -> None:
        self.turn: dict[str, Any] | None = None

    def record(self, role: str, started: float, response: Any) -> None:
        if self.turn is None:
            return
        usage = getattr(response, "usage", None)
        self.turn.setdefault("api_calls", []).append(
            {
                "role": role,
                "runtime_ms": round((time.perf_counter() - started) * 1000, 2),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "raw_content": getattr(response.choices[0].message, "content", None),
            }
        )


class _AuditedCompletions:
    def __init__(self, delegate, audit: _ApiAudit, role: str) -> None:
        self._delegate, self._audit, self._role = delegate, audit, role

    def create(self, **kwargs):
        started = time.perf_counter()
        response = self._delegate.create(**kwargs)
        self._audit.record(self._role, started, response)
        return response


class _AuditedClient:
    def __init__(self, delegate, audit: _ApiAudit, role: str) -> None:
        self.chat = type("Chat", (), {"completions": _AuditedCompletions(delegate.chat.completions, audit, role)})()


class _ScenarioUser:
    def __init__(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario
        self.disclosed: set[str] = set()
        self.vague_count = 0
        self.repeat_count = 0
        self.closed = False

    def first_message(self) -> str:
        return "Домой прихожу как овощ."

    def next_message(self, reply: str, discovery_complete: bool) -> tuple[str | None, str | None]:
        text = reply.lower()
        if self.closed:
            return None, "user_paused"
        if discovery_complete and "?" not in reply:
            self.closed = True
            return "Спасибо, я подумаю.", "user_completed_current_step"
        if "motivation" in text or "тему " in text:
            return self._vague_reply()

        conditions = (
            ("duration", ("как давно", "когда", "сколько", "лет", "месяц")),
            ("impact", ("мешает", "влияет", "энерги", "самочув", "жизни", "повседнев")),
            ("goal", ("хотелось", "изменить", "вернуть", "цель", "хочешь")),
            ("barrier", ("сложно", "трудно", "мешает", "препят", "получается")),
            ("attempt", ("пробовал", "пытал", "раньше", "попыт")),
            ("preference", ("диет", "калор", "предпоч", "огранич")),
        )
        for key, markers in conditions:
            if any(marker in text for marker in markers):
                return self._disclose(key)
        return self._vague_reply()

    def _disclose(self, key: str) -> tuple[str, str | None]:
        messages = {
            "impact": "К вечеру сил почти нет: хочется играть с сыном, а я просто лежу.",
            "duration": "С тех пор как родился ребёнок — уже больше года.",
            "goal": "Не быть выжатым и немного сбросить вес.",
            "barrier": "После работы нет сил готовить, поэтому часто заказываю еду.",
            "attempt": "Пробовал считать калории, но быстро бросал.",
            "preference": "Жёсткие диеты не хочу, только от них хуже.",
        }
        if key not in self.disclosed:
            self.disclosed.add(key)
            return messages[key], None
        self.repeat_count += 1
        if self.repeat_count >= self.scenario["limits"]["max_repetitions"]:
            self.closed = True
            return "Давайте потом.", "user_paused"
        return "Я это уже сказал: к вечеру сил почти нет.", None

    def _vague_reply(self) -> tuple[str, str | None]:
        self.vague_count += 1
        if self.vague_count >= 2:
            self.closed = True
            return "Давайте потом.", "user_paused"
        return "Не знаю, уже устал это объяснять.", None


@contextmanager
def _temporary_semantic_source(variant: str, audit: _ApiAudit):
    original_semantic = dialog_engine.SemanticEngine
    original_openai = response_engine_module.OpenAI

    class TimedDeterministicSemanticEngine(SemanticEngine):
        def analyze(self, user_text: str):
            started = time.perf_counter()
            result = super().analyze(user_text)
            if audit.turn is not None:
                audit.turn["semantic_execution"] = {
                    "source": "deterministic",
                    "runtime_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": None,
                    "fallback": False,
                    "api_calls": 0,
                    "tokens": None,
                }
            return result

    class TimedLLMSemanticEngine:
        def __init__(self) -> None:
            self._engine = LLMSemanticEngine(
                client=_AuditedClient(OpenAI(), audit, "semantic"),
            )

        def analyze(self, user_text: str):
            started = time.perf_counter()
            result = self._engine.analyze(user_text)
            metadata = {
                "source": "llm:gpt-5-mini",
                "runtime_ms": round((time.perf_counter() - started) * 1000, 2),
                "error": None,
                "fallback": False,
                "api_calls": 1,
                "tokens": None,
                "raw_structured_output": None,
                "rejected_fields": [],
                "validation_reason": None,
                "unconfirmed_interpretations": [],
            }
            if audit.turn is not None:
                calls = [call for call in audit.turn.get("api_calls", []) if call["role"] == "semantic"]
                if calls:
                    last = calls[-1]
                    metadata["raw_structured_output"] = last["raw_content"]
                    metadata["tokens"] = last["total_tokens"]
                    try:
                        payload = json.loads(last["raw_content"] or "")
                        self._engine._context_from_payload(payload, user_text)
                    except (ValueError, TypeError, json.JSONDecodeError) as error:
                        metadata["fallback"] = True
                        metadata["validation_reason"] = str(error)
                    metadata["unconfirmed_interpretations"] = _unconfirmed_interpretations(
                        user_text, asdict(result)
                    )
            if result == type(result)() and user_text.strip():
                metadata["fallback"] = metadata["fallback"] or False
            if audit.turn is not None:
                audit.turn["semantic_execution"] = metadata
            return result

    def response_client_factory(*args, **kwargs):
        return _AuditedClient(OpenAI(*args, **kwargs), audit, "response")

    dialog_engine.SemanticEngine = TimedDeterministicSemanticEngine if variant == "A" else TimedLLMSemanticEngine
    response_engine_module.OpenAI = response_client_factory
    try:
        yield
    finally:
        dialog_engine.SemanticEngine = original_semantic
        response_engine_module.OpenAI = original_openai


def _unconfirmed_interpretations(user_text: str, context: dict[str, Any]) -> list[str]:
    text = user_text.lower()
    findings = []
    if context["goal"] == "restore_energy" and not any(word in text for word in ("хочу", "не быть", "вернуть")):
        findings.append("goal:restore_energy may be a value rather than an explicit goal")
    if "fatigue_after_work" in context["constraints"] and "работ" not in text:
        findings.append("constraint:fatigue_after_work infers work from arriving home")
    if "closedness" in context["emotional_signals"] and "закры" not in text:
        findings.append("emotional_signal:closedness is an interpretation of a boundary")
    if "weight" in context["topics"] and not any(word in text for word in ("вес", "похуд", "калори")):
        findings.append("topic:weight is not explicit")
    return findings


async def run_variant(variant: str, scenario: dict[str, Any], user_id: int) -> dict[str, Any]:
    reset_user_profile(user_id)
    tracer = _SessionTracer(os.getpid(), user_id)
    audit = _ApiAudit()
    simulator = _ScenarioUser(scenario)
    turns: list[dict[str, Any]] = []
    user_text = simulator.first_message()
    stop_reason = "max_user_turns"
    pending_stop: str | None = None

    with _temporary_semantic_source(variant, audit), _instrument_dialog_engine(tracer, live_response=True):
        for number in range(1, scenario["limits"]["max_user_turns"] + 1):
            turn: dict[str, Any] = {
                "turn": number,
                "variant": variant,
                "process_id": os.getpid(),
                "user_id": user_id,
                "user_text": user_text,
            }
            tracer.current_turn = turn
            audit.turn = turn
            profile = get_user_profile(user_id)
            result = await dialog_engine.run_dialog_engine(user_text, profile, user_id)
            update_user_profile(user_id, result["update"])
            turn["update"] = _serialise(result["update"])
            turn["memory_after_turn"] = _serialise(get_user_profile(user_id))
            turn["consultation_state_after_turn"] = _consultation_state(turn["memory_after_turn"])
            turn["human_model_after_turn"] = _serialise(get_human_model(user_id))
            turns.append(turn)
            tracer.current_turn = None
            audit.turn = None

            if pending_stop is not None:
                break

            next_message, natural_stop = simulator.next_message(
                result["reply"], result["update"].get("discovery_complete", False)
            )
            if natural_stop is not None:
                stop_reason = natural_stop
                pending_stop = natural_stop
            if next_message is None:
                break
            user_text = next_message

    return {
        "scenario_id": scenario["scenario_id"],
        "variant": variant,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "process_id": turns[0]["process_id"],
        "user_id": user_id,
        "turns": turns,
        "stop_reason": stop_reason,
        "disclosed_conditions": sorted(simulator.disclosed),
        "user_simulator_state": {"vague_count": simulator.vague_count, "repeat_count": simulator.repeat_count},
    }


def main() -> None:
    scenario_path = Path(__file__).parent / "scenarios" / "2026-08-06_CL-005.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    traces = Path(__file__).parent / "traces"
    for variant, user_id in (("A", 950051), ("B", 950052)):
        result = asyncio.run(run_variant(variant, scenario, user_id))
        path = traces / f"2026-08-06_CL-005-{variant}_trace.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"variant": variant, "trace": str(path), "stop_reason": result["stop_reason"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
