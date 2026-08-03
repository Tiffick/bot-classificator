from copy import deepcopy

from ai.engines.decision_engine import DecisionEngine
from ai.engines.human_model import HumanModel
from ai.engines.reasoning_engine import ReasoningContext
from ai.engines.semantic_engine import SemanticEngine


def test_prioritizes_clarifying_contradiction():
    human_model = HumanModel(goals=["lose_weight"])
    reasoning_context = ReasoningContext(
        contradictions=[{"name": "stated_goal_conflicts_with_stated_barrier"}],
        priorities=["motivation"],
    )

    decision = DecisionEngine().decide(
        SemanticEngine().analyze("Хочу похудеть."),
        human_model,
        reasoning_context,
        {"facts": {}},
    )

    assert decision.next_goal == "clarify_contradiction"
    assert decision.priority == "high"
    assert decision.needs_additional_information is True
    assert decision.reason == "stated_goal_conflicts_with_stated_barrier"


def test_selects_one_highest_priority_missing_area():
    human_model = HumanModel()
    original_model = deepcopy(human_model)
    reasoning_context = ReasoningContext(
        priorities=["motivation", "trust"],
        missing_information=["motivation", "trust"],
    )

    decision = DecisionEngine().decide(
        SemanticEngine().analyze("Не знаю."),
        human_model,
        reasoning_context,
        {"facts": {}},
    )

    assert decision.next_goal == "clarify_motivation"
    assert decision.expected_outcome == "understanding_of_motivation_increases"
    assert decision.response_type == "question"
    assert human_model == original_model


def test_selects_concrete_weight_impact_step_for_weight_concern():
    decision = DecisionEngine().decide(
        SemanticEngine().analyze("я толстый"),
        HumanModel(),
        ReasoningContext(priorities=["goals", "motivation"]),
        {"facts": {}},
    )

    assert decision.next_goal == "clarify_weight_impact"
    assert decision.reason == "topic:weight; missing_information:weight_impact"
    assert decision.needs_additional_information is True
    assert decision.expected_outcome == "understanding_of_weight_impact_increases"
    assert decision.response_type == "question"


def test_selects_duration_after_energy_or_health_signal():
    decision = DecisionEngine().decide(
        SemanticEngine().analyze("энергии совсем нет"),
        HumanModel(),
        ReasoningContext(priorities=["goals", "motivation"]),
        {"facts": {}},
    )

    assert decision.next_goal == "clarify_problem_duration"
    assert decision.reason == "topic:energy_or_health; missing_information:problem_duration"
    assert decision.expected_outcome == "understanding_of_problem_duration_increases"


def test_selects_desired_change_after_duration_is_known():
    decision = DecisionEngine().decide(
        SemanticEngine().analyze("это уже 2 года"),
        HumanModel(),
        ReasoningContext(priorities=["goals", "motivation"]),
        {"facts": {}},
    )

    assert decision.next_goal == "clarify_desired_change"
    assert decision.reason == "fact:duration; missing_information:desired_change"


def test_verifies_working_hypothesis_when_information_is_sufficient():
    reasoning_context = ReasoningContext(
        hypotheses={"change_difficulty_requires_clarification": {"status": "working"}},
        confidence={"change_difficulty_requires_clarification": 0.6},
    )

    decision = DecisionEngine().decide(
        SemanticEngine().analyze("Мне не получается похудеть."),
        HumanModel(goals=["lose_weight"], motivation=["health"], barriers=["routine"], trust=0.5),
        reasoning_context,
        {"facts": {}},
    )

    assert decision.next_goal == "verify_hypothesis"
    assert decision.reason == "working_hypothesis:change_difficulty_requires_clarification"
    assert decision.confidence == 0.6
