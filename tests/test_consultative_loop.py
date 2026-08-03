from ai.engines.decision_engine import DecisionEngine
from ai.engines.human_model import HumanModel
from ai.engines.reasoning_engine import ReasoningEngine
from ai.engines.semantic_engine import SemanticEngine


def _next_goal(message: str) -> str:
    semantic_context = SemanticEngine().analyze(message)
    human_model = HumanModel()
    reasoning_context = ReasoningEngine().reason(
        semantic_context,
        human_model,
        {"facts": {}},
    )
    return DecisionEngine().decide(
        semantic_context,
        human_model,
        reasoning_context,
        {"facts": {}},
    ).next_goal


def test_discovery_questions_progress_without_repeating_the_same_axis():
    next_goals = [
        _next_goal("я толстый"),
        _next_goal("больше всего самочувствие"),
        _next_goal("это уже 2 года"),
    ]

    assert next_goals == [
        "clarify_weight_impact",
        "clarify_problem_duration",
        "clarify_desired_change",
    ]
