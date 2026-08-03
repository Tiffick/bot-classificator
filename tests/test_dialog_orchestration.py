import asyncio
from copy import deepcopy

from ai.engines.decision_engine import DecisionContext
from ai.engines.emotional_engine import EmotionalContext
from ai.engines.human_model import HumanModel
from ai.engines.impact_engine import ImpactContext
from ai.engines.impact_engine import ImpactEngine
from ai.engines.emotional_engine import EmotionalEngine
from ai.engines.reasoning_engine import ReasoningContext
from ai.engines.semantic_engine import SemanticContext


def test_dialog_engine_calls_all_engines_in_architectural_order(monkeypatch):
    import ai.dialog_engine as dialog_engine

    calls = []

    class Semantic:
        def analyze(self, user_text):
            calls.append("semantic")
            return SemanticContext(facts={"age": 30}, confidence=1.0)

    class Human:
        def build(self, semantic_context, previous_human_model, memory):
            calls.append("human_model")
            return HumanModel(facts=semantic_context.facts)

        def apply_update(self, profile, update):
            return update.copy()

        def is_discovery_complete(self, profile):
            return False

    class Reasoning:
        def reason(self, semantic_context, human_model, memory):
            calls.append("reasoning")
            return ReasoningContext()

    class Decision:
        def decide(self, semantic_context, human_model, reasoning_context, memory):
            calls.append("decision")
            return DecisionContext()

    class Impact:
        def evaluate(self, semantic_context, human_model, reasoning_context, decision_context, memory):
            calls.append("impact")
            return ImpactContext()

    class Emotional:
        def choose(self, semantic_context, human_model, reasoning_context, decision_context, impact_context, memory):
            calls.append("emotional")
            return EmotionalContext()

    class Response:
        def generate(self, semantic_context, human_model, reasoning_context, decision_context, impact_context, emotional_context, memory):
            calls.append("response")
            return "Оркестрованный ответ"

    monkeypatch.setattr(dialog_engine, "SemanticEngine", Semantic)
    monkeypatch.setattr(dialog_engine, "HumanModelEngine", Human)
    monkeypatch.setattr(dialog_engine, "ReasoningEngine", Reasoning)
    monkeypatch.setattr(dialog_engine, "DecisionEngine", Decision)
    monkeypatch.setattr(dialog_engine, "ImpactEngine", Impact)
    monkeypatch.setattr(dialog_engine, "EmotionalEngine", Emotional)
    monkeypatch.setattr(dialog_engine, "ResponseEngine", Response)

    result = asyncio.run(dialog_engine.run_dialog_engine("Мне 30 лет", {}))

    assert result["reply"] == "Оркестрованный ответ"
    assert calls == [
        "semantic",
        "human_model",
        "reasoning",
        "decision",
        "impact",
        "emotional",
        "response",
    ]


def test_decision_context_is_not_changed_by_impact_or_emotional_engines():
    semantic_context = SemanticContext(topics=["weight"], confidence=0.5)
    human_model = HumanModel()
    reasoning_context = ReasoningContext(priorities=["goals"])
    decision_context = DecisionContext(
        next_goal="clarify_weight_impact",
        reason="topic:weight; missing_information:weight_impact",
        needs_additional_information=True,
        expected_outcome="understanding_of_weight_impact_increases",
        response_type="question",
    )
    original_decision = deepcopy(decision_context)

    impact_context = ImpactEngine().evaluate(
        semantic_context,
        human_model,
        reasoning_context,
        decision_context,
        {"facts": {}},
    )
    EmotionalEngine().choose(
        semantic_context,
        human_model,
        reasoning_context,
        decision_context,
        impact_context,
        {"facts": {}},
    )

    assert decision_context == original_decision
