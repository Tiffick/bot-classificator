from copy import deepcopy

from ai.engines.decision_engine import DecisionContext
from ai.engines.human_model import HumanModel
from ai.engines.impact_engine import ImpactEngine
from ai.engines.reasoning_engine import ReasoningContext
from ai.engines.semantic_engine import SemanticEngine


def test_translates_decision_into_one_non_manipulative_impact_goal():
    decision = DecisionContext(
        next_goal="clarify_motivation",
        reason="missing_information:motivation",
        confidence=0.6,
    )

    impact = ImpactEngine().evaluate(
        SemanticEngine().analyze("Не знаю."),
        HumanModel(),
        ReasoningContext(priorities=["motivation"]),
        decision,
        {"facts": {}},
    )

    assert impact.main_goal == "increase_self_understanding"
    assert impact.expected_understanding_change == "understanding_of_motivation_increases"
    assert impact.success_criterion == "user_provides_or_recognizes_motivation"
    assert impact.confidence == 0.6


def test_uses_contradiction_decision_to_reduce_tension_without_pressure():
    decision = DecisionContext(
        next_goal="clarify_contradiction",
        confidence=0.8,
    )

    impact = ImpactEngine().evaluate(
        SemanticEngine().analyze("Хочу похудеть."),
        HumanModel(goals=["lose_weight"]),
        ReasoningContext(),
        decision,
        {"facts": {}},
    )

    assert impact.main_goal == "reduce_internal_conflict"
    assert impact.expected_emotional_change == "reduced_tension"
    assert impact.success_criterion == "contradiction_is_acknowledged_without_pressure"


def test_does_not_change_any_input_context():
    semantic_context = SemanticEngine().analyze("Мне не получается похудеть.")
    human_model = HumanModel(goals=["lose_weight"], barriers=["routine"])
    reasoning_context = ReasoningContext(
        hypotheses={"change_difficulty_requires_clarification": {"status": "working"}}
    )
    decision_context = DecisionContext(
        next_goal="verify_hypothesis",
        confidence=0.6,
    )
    original_inputs = deepcopy(
        (semantic_context, human_model, reasoning_context, decision_context)
    )

    impact = ImpactEngine().evaluate(
        semantic_context,
        human_model,
        reasoning_context,
        decision_context,
        {"facts": {}},
    )

    assert impact.main_goal == "support_shared_investigation"
    assert (semantic_context, human_model, reasoning_context, decision_context) == original_inputs
