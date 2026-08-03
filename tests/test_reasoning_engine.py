from copy import deepcopy

from ai.engines.human_model import HumanModel
from ai.engines.reasoning_engine import ReasoningEngine
from ai.engines.semantic_engine import SemanticEngine


def test_builds_explainable_hypothesis_without_changing_human_model():
    semantic_context = SemanticEngine().analyze(
        "Хочу похудеть, но у меня не получается."
    )
    human_model = HumanModel(
        goals=["lose_weight"],
        barriers=["не получается"],
        emotional_features=["frustration"],
    )
    original_model = deepcopy(human_model)

    context = ReasoningEngine().reason(
        semantic_context,
        human_model,
        {"facts": {}},
    )

    hypothesis = context.hypotheses["change_difficulty_requires_clarification"]

    assert hypothesis["status"] == "working"
    assert context.confidence["change_difficulty_requires_clarification"] == 0.6
    assert context.foundations["change_difficulty_requires_clarification"] == [
        "barrier_present",
        "frustration_signal_present",
    ]
    assert human_model == original_model


def test_preserves_uncertainty_when_information_is_missing():
    semantic_context = SemanticEngine().analyze("Не знаю.")

    context = ReasoningEngine().reason(
        semantic_context,
        HumanModel(),
        {"facts": {}},
    )

    assert context.hypotheses == {}
    assert context.missing_information == [
        "goals",
        "motivation",
        "barriers",
        "trust",
    ]
    assert context.priorities == context.missing_information
    assert "current_message_contains_no_confirmed_facts" in context.uncertainties


def test_records_explicit_contradiction_without_resolving_it():
    semantic_context = SemanticEngine().analyze("Хочу похудеть.")
    human_model = HumanModel(
        goals=["lose_weight"],
        barriers=["не хочу худеть"],
    )

    context = ReasoningEngine().reason(
        semantic_context,
        human_model,
        {"facts": {}},
    )

    assert context.contradictions == [
        {
            "name": "stated_goal_conflicts_with_stated_barrier",
            "goal": "lose_weight",
        }
    ]
