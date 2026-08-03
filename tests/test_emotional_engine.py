from copy import deepcopy

from ai.engines.decision_engine import DecisionContext
from ai.engines.emotional_engine import EmotionalEngine
from ai.engines.human_model import HumanModel
from ai.engines.impact_engine import ImpactContext
from ai.engines.reasoning_engine import ReasoningContext
from ai.engines.semantic_engine import SemanticEngine


def test_chooses_gentle_supportive_form_for_tension_reduction():
    emotional_context = EmotionalEngine().choose(
        SemanticEngine().analyze("Хочу похудеть."),
        HumanModel(goals=["lose_weight"]),
        ReasoningContext(),
        DecisionContext(next_goal="clarify_contradiction"),
        ImpactContext(main_goal="reduce_internal_conflict"),
        {"facts": {}},
    )

    assert emotional_context.tone == "calm_supportive"
    assert emotional_context.empathy_level == "high"
    assert emotional_context.directness == "gentle"
    assert emotional_context.user_specific_notes == ["avoid_pressure"]


def test_chooses_reflective_form_for_self_understanding():
    emotional_context = EmotionalEngine().choose(
        SemanticEngine().analyze("Не знаю."),
        HumanModel(),
        ReasoningContext(priorities=["motivation"]),
        DecisionContext(next_goal="clarify_motivation"),
        ImpactContext(main_goal="increase_self_understanding"),
        {"facts": {}},
    )

    assert emotional_context.tone == "reflective"
    assert emotional_context.explanation_depth == "moderate"
    assert emotional_context.initiative_level == "inviting"


def test_does_not_change_any_previous_context():
    semantic_context = SemanticEngine().analyze("Мне не получается похудеть.")
    human_model = HumanModel(goals=["lose_weight"], barriers=["routine"])
    reasoning_context = ReasoningContext(
        hypotheses={"change_difficulty_requires_clarification": {"status": "working"}}
    )
    decision_context = DecisionContext(next_goal="verify_hypothesis")
    impact_context = ImpactContext(main_goal="support_shared_investigation")
    original_inputs = deepcopy(
        (
            semantic_context,
            human_model,
            reasoning_context,
            decision_context,
            impact_context,
        )
    )

    emotional_context = EmotionalEngine().choose(
        semantic_context,
        human_model,
        reasoning_context,
        decision_context,
        impact_context,
        {"facts": {}},
    )

    assert emotional_context.tone == "curious_supportive"
    assert emotional_context.directness == "tentative"
    assert (
        semantic_context,
        human_model,
        reasoning_context,
        decision_context,
        impact_context,
    ) == original_inputs
