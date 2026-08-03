import asyncio

from ai.engines.decision_engine import DecisionEngine
from ai.engines.human_model import HumanModel
from ai.engines.human_model_engine import HumanModelEngine
from ai.engines.reasoning_engine import ReasoningContext
from ai.engines.semantic_engine import SemanticEngine
from memory.user_memory import (
    get_human_model,
    get_user_profile,
    reset_user_profile,
    update_user_profile,
)


def _build_discovery_model(messages):
    engine = HumanModelEngine()
    model = None
    for message in messages:
        model = engine.build(
            SemanticEngine().analyze(message),
            model,
            {"facts": {}},
        )
    return model


def test_discovery_completes_when_human_model_has_minimum_understanding():
    model = _build_discovery_model(
        [
            "я толстый",
            "хочу похудеть",
            "это уже 2 года",
            "не получается держать режим",
        ]
    )

    assert model.understanding_score["problem"] == 1.0
    assert model.understanding_score["motivation"] == 1.0
    assert model.understanding_score["pain"] == 1.0
    assert model.understanding_score["limitations"] == 1.0
    assert model.consultation_stage == "discovery_complete"
    assert HumanModelEngine().is_discovery_complete(model) is True


def test_discovery_remains_open_when_important_information_is_missing():
    model = _build_discovery_model(
        ["я толстый", "хочу похудеть", "это уже 2 года"]
    )

    assert model.understanding_score["limitations"] == 0.0
    assert model.consultation_stage == "discovery"
    assert HumanModelEngine().is_discovery_complete(model) is False


def test_discovery_does_not_complete_from_facts_without_human_understanding():
    model = HumanModel(
        facts={"age": 30, "duration": "уже 2 года"},
        understanding_score={
            "problem": 0.0,
            "motivation": 0.0,
            "pain": 1.0,
            "fear": 0.0,
            "self_perception": 0.0,
            "limitations": 0.0,
            "trust": 0.0,
        },
    )

    assert HumanModelEngine().is_discovery_complete(model) is False


def test_decision_starts_consultation_after_discovery_is_complete():
    model = _build_discovery_model(
        [
            "я толстый",
            "хочу похудеть",
            "это уже 2 года",
            "не получается держать режим",
        ]
    )

    decision = DecisionEngine().decide(
        SemanticEngine().analyze("да"),
        model,
        ReasoningContext(),
        {"facts": {}},
    )

    assert decision.next_goal == "begin_consultation"
    assert decision.needs_additional_information is False
    assert decision.ready_for_next_stage is True
    assert decision.response_type == "statement"


def test_full_discovery_persists_completion_and_stops_questions(fake_openai):
    from ai.dialog_engine import run_dialog_engine

    user_id = 104
    reset_user_profile(user_id)
    result = None
    for message in (
        "я толстый",
        "хочу похудеть",
        "это уже 2 года",
        "не получается держать режим",
    ):
        result = asyncio.run(
            run_dialog_engine(message, get_user_profile(user_id), user_id)
        )
        update_user_profile(user_id, result["update"])

    assert result["update"]["discovery_complete"] is True
    assert "?" not in result["reply"]
    assert get_human_model(user_id).consultation_stage == "discovery_complete"
