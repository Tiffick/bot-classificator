import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

from Conversation_Lab.session_runner import run_session
from ai.engines.llm_semantic_engine import (
    LLMSemanticEngine,
    compare_semantic_contexts,
)
from ai.engines.semantic_engine import SemanticContext


class FakeCompletions:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.payload)))]
        )


class FakeClient:
    def __init__(self, payload):
        self.chat = SimpleNamespace(completions=FakeCompletions(payload))


def _payload(**overrides):
    payload = {
        "topics": [],
        "intent": "information_statement",
        "goal": None,
        "facts": {
            "age": None,
            "current_weight": None,
            "height": None,
            "duration": None,
        },
        "preferences": [],
        "constraints": [],
        "emotional_signals": [],
        "events": [],
        "questions": [],
        "confidence": 0.8,
    }
    payload.update(overrides)
    return payload


def test_llm_semantic_engine_uses_strict_schema_and_existing_context():
    client = FakeClient(
        _payload(
            topics=["weight"],
            facts={"age": None, "current_weight": None, "height": None, "duration": "уже год"},
            emotional_signals=["situation_fatigue"],
        )
    )

    context = LLMSemanticEngine(client=client).analyze("Вес уже год растёт, и мне от этого тяжело.")

    assert context.facts == {"duration": "уже год"}
    assert context.topics == ["weight"]
    assert context.emotional_signals == ["situation_fatigue"]
    call = client.chat.completions.calls[0]
    assert call["model"] == "gpt-5-mini"
    assert "temperature" not in call
    assert call["timeout"] == 15.0
    schema = call["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False


def test_llm_semantic_engine_falls_back_on_unknown_fields_or_diagnoses():
    payload = _payload(emotional_signals=["depression"])
    engine = LLMSemanticEngine(client=FakeClient(payload))

    context = engine.analyze("Мне тяжело.")

    assert context == SemanticContext()
    assert engine.last_diagnostics["success"] is False


def test_llm_semantic_engine_preserves_negation_and_does_not_invent_facts():
    payload = _payload(
        intent="pause_or_end",
        facts={"age": 30, "current_weight": None, "height": None, "duration": None},
    )

    context = LLMSemanticEngine(client=FakeClient(payload)).analyze(
        "Спасибо, но я пока не готова обсуждать дальше."
    )

    assert context == SemanticContext()


def test_llm_semantic_engine_accepts_explicit_pause_without_motivation():
    payload = _payload(intent="pause_or_end", confidence=0.95)

    context = LLMSemanticEngine(client=FakeClient(payload)).analyze(
        "Спасибо, но я пока не готова обсуждать дальше."
    )

    assert context.intent == "pause_or_end"
    assert context.emotional_signals == []
    assert context.facts == {}


def test_llm_semantic_engine_keeps_unconfirmed_work_in_diagnostics_only():
    payload = _payload(
        topics=["energy"],
        constraints=["fatigue_after_work"],
        events=["daily_effect:reduced_energy"],
    )
    engine = LLMSemanticEngine(client=FakeClient(payload))

    context = engine.analyze("Домой прихожу как овощ.")

    assert context.topics == ["energy"]
    assert context.events == ["daily_effect:reduced_energy"]
    assert context.constraints == []
    assert engine.last_diagnostics["success"] is True
    assert engine.last_diagnostics["rejected_fields"] == [
        "constraints.fatigue_after_work"
    ]


def test_llm_semantic_engine_keeps_explicit_duration_linked_to_childbirth():
    payload = _payload(
        facts={
            "age": None,
            "current_weight": None,
            "height": None,
            "duration": "с тех пор как родился ребёнок — уже больше года",
        }
    )

    context = LLMSemanticEngine(client=FakeClient(payload)).analyze(
        "С тех пор как родился ребёнок — уже больше года."
    )

    assert context.facts["duration"] == "с тех пор как родился ребёнок — уже больше года"


def test_llm_semantic_engine_extracts_explicit_energy_and_weight_goal():
    payload = _payload(
        topics=["energy", "weight"],
        intent="goal_statement",
        goal="restore_energy",
    )

    context = LLMSemanticEngine(client=FakeClient(payload)).analyze(
        "Не быть выжатым и немного сбросить вес."
    )

    assert context.topics == ["energy", "weight"]
    assert context.goal == "restore_energy"
    assert context.intent == "goal_statement"


def test_llm_semantic_engine_records_timeout_as_a_fallback_reason():
    class TimeoutCompletions:
        def create(self, **kwargs):
            raise TimeoutError("semantic timeout")

    client = SimpleNamespace(chat=SimpleNamespace(completions=TimeoutCompletions()))
    engine = LLMSemanticEngine(client=client)

    context = engine.analyze("Домой прихожу как овощ.")

    assert context == SemanticContext()
    assert engine.last_diagnostics["success"] is False
    assert "TimeoutError" in engine.last_diagnostics["fallback_reason"]


def test_compares_two_semantic_contexts_without_mutation():
    deterministic = SemanticContext(topics=["energy"], intent="information_statement")
    llm = SemanticContext(
        topics=["energy"],
        constraints=["fatigue_after_work"],
        events=["daily_effect:difficulty_cooking"],
        intent="information_statement",
    )
    before = deepcopy((deterministic, llm))

    comparison = compare_semantic_contexts(deterministic, llm)

    assert comparison["only_llm"]["constraints"] == ["fatigue_after_work"]
    assert comparison["only_llm"]["events"] == ["daily_effect:difficulty_cooking"]
    assert (deterministic, llm) == before


def test_shadow_semantic_result_does_not_change_active_dialog_route():
    class ShadowEngine:
        def analyze(self, user_text):
            return SemanticContext(topics=["health"], intent="pause_or_end")

    baseline = asyncio.run(
        run_session(["я толстый"], user_id=991001, live_response=False)
    )
    shadowed = asyncio.run(
        run_session(
            ["я толстый"],
            user_id=991002,
            live_response=False,
            shadow_semantic=True,
            llm_semantic_engine=ShadowEngine(),
        )
    )

    assert baseline["turns"][0]["semantic_context"] == shadowed["turns"][0]["semantic_context"]
    assert baseline["turns"][0]["response"] == shadowed["turns"][0]["response"]
    assert shadowed["turns"][0]["llm_semantic_context"]["intent"] == "pause_or_end"
    assert shadowed["turns"][0]["semantic_comparison"]["only_llm"]["intent"] == "pause_or_end"
